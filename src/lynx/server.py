"""
MCP server for the lynx project (multi-source).

The bulk of the work happens inside `run_server(config_path)`:
  - load config (v2 schema, validates `sources` block)
  - construct `SourceManager` in a background thread (heavy: loads embedding
    model and builds backends)
  - register MCP tools dynamically based on configured sources
  - start per-source watchers once backends are ready
  - call mcp.run() (blocking)

The stdout-redirect dance at module top must run BEFORE any heavy import,
so it is intentionally a side effect of importing this module — see the
comment by `_REAL_STDOUT_FD` for why.
"""

import os
import sys
import warnings
import threading

# Silence EVERYTHING before any library writes to stdout/stderr.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# CRITICAL for MCP: some libraries write to stdout during import/init
# (llama_index prints "LLM is explicitly disabled. Using MockLLM.";
# sentence_transformers logs the model load; etc.). In MCP stdio mode,
# stdout is the JSON-RPC channel and any spurious byte breaks the protocol
# and freezes tool calls. We save the real fd 1 and redirect it to fd 2 for
# the entire import/loading phase, then restore it just before mcp.run().
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)

from mcp.server.fastmcp import FastMCP

from .config import load_config


def _restore_real_stdout():
    """Restore fd 1 to the real stdout, draining the Python buffer first."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.dup2(_REAL_STDOUT_FD, 1)


# ----------------------------------------------------------------------
# Output formatting helpers (shared by per-source and global tools)
# ----------------------------------------------------------------------


def _format_one_result(i, res):
    """Render one result block: header with symbol/line context + content body.

    The header surfaces the AST chunker metadata so the AI can cite the
    result precisely (e.g. "see MyClass.handleClick in foo.cs L42-58")
    instead of just naming the file.
    """
    score = res.get("score", 0.0)
    fname = res.get("file", "unknown")
    sym = res.get("symbol_name") or ""
    start = res.get("start_line") or 0
    end = res.get("end_line") or 0

    # Build a concise location string: "foo.cs:42-58  MyClass.handleClick"
    loc_parts = [fname]
    if start and end and start != end:
        loc_parts.append(f"L{start}-{end}")
    elif start:
        loc_parts.append(f"L{start}")
    loc = " ".join(loc_parts)
    sym_suffix = f"  {sym}" if sym and sym not in ("<header>", "<chunk1>") and not sym.startswith("<chunk") else ""

    header = f"--- {i}. {loc}{sym_suffix} (Score: {score:.4f}) ---\n"
    body = ""
    if "source" in res:
        body += f"    source: {res['source']}\n"
    body += f"{res.get('content', '')}\n\n"
    return header + body


def _format_search_results(query, results, source_label, filter_suffix):
    if not results:
        return f"No results found for '{query}' in {source_label}{filter_suffix}."
    out = f"Found {len(results)} results for '{query}' in {source_label}{filter_suffix}:\n\n"
    for i, res in enumerate(results, 1):
        out += _format_one_result(i, res)
    return out


def _format_deep_response(response, queries, source_label, meta_suffix):
    results = response["results"]
    tried = response["variants_tried"]
    total = len(queries)
    winning = response["winning_variant_index"]
    all_weak = response["all_weak"]

    if all_weak:
        if not results:
            return (
                f"No results found across {tried} query variant(s) in "
                f"{source_label}{meta_suffix}. Consider relaxing filters or "
                f"rephrasing the question."
            )
        header = (
            f"WARNING: all {tried} query variants returned WEAK results "
            f"in {source_label}{meta_suffix} (no variant crossed the threshold). "
            f"Showing the strongest weak set below.\n\n"
        )
    else:
        header = (
            f"Found {len(results)} results in {source_label} "
            f"(variant {winning}/{total} won{meta_suffix}):\n\n"
        )

    out = header
    for i, res in enumerate(results, 1):
        out += _format_one_result(i, res)

    if "per_variant" in response:
        out += "\n--- Per-variant summary ---\n"
        for i, v in enumerate(response["per_variant"], 1):
            status_label = "PASSED" if v["passed_threshold"] else "weak"
            n = len(v["results"])
            ts = v.get("top_score")
            ts_str = f"{ts:.4f}" if isinstance(ts, (int, float)) else "n/a"
            out += (
                f"  variant {i}/{total} ({status_label}): "
                f"{n} result(s), top_score={ts_str}, query={v['query']!r}\n"
            )
    return out


def _build_filter_suffix(file_glob, extensions, path_contains):
    parts = []
    if file_glob:
        parts.append(f"file_glob={file_glob!r}")
    if extensions:
        parts.append(f"extensions={list(extensions)!r}")
    if path_contains:
        parts.append(f"path_contains={path_contains!r}")
    return f" (filters: {', '.join(parts)})" if parts else ""


# ----------------------------------------------------------------------
# Tool registration: per-source pair + globals
# ----------------------------------------------------------------------


def _register_source_tools(mcp, manager, source_name: str, source_type: str, source_config: dict):
    """Register `search_<name>` and `deep_search_<name>` for one source.

    The tool docstrings include the source name and type so the AI client
    has enough context to pick the right tool from its tool list.
    """
    # The fields we expose depend on the source type. M1 only has `codebase`,
    # so we expose the codebase-specific filter parameters. When we add
    # webdoc/pdf in M2/M3 we extend the dispatch here.
    path_hint = source_config.get("path", "")

    search_tool_name = f"search_{source_name}"
    deep_tool_name = f"deep_search_{source_name}"

    # Per-source tool descriptions. We pass them via `description=` rather
    # than as Python docstrings because an f-string as the first statement
    # of a function is just an evaluated expression — it never becomes
    # __doc__, so FastMCP would see an empty description and the AI client
    # would have no idea when to use which tool.
    _desc_search = (
        f"Semantic search over the {source_name!r} source (type: {source_type}). "
        f"Indexed location: {path_hint}. "
        f"This is your PRIMARY search tool — use it FIRST for any question about this source. "
        f"Only escalate to `{deep_tool_name}` if results are weak or empty. "
        f"Best practices: use natural-language descriptions of what the code does, not exact "
        f"identifiers (use grep for those). Good: 'method that handles player damage calculation'. "
        f"Bad: 'CalculateDamage'. Args: query (natural language); top_k (default from config); "
        f"file_glob, extensions, path_contains (optional filters)."
    )

    _desc_deep = (
        f"Multi-query fallback search over the {source_name!r} source (type: {source_type}). "
        f"ESCALATION TOOL — use only when `{search_tool_name}` returned weak or empty results, "
        f"or when the user explicitly asks for a more thorough search. Slower than `{search_tool_name}` "
        f"because it runs multiple retrievals. "
        f"How it works: tries each query variant in order; stops at the first whose results pass the "
        f"weakness threshold. If all fail, returns the strongest weak set with a warning. "
        f"Best practices: provide 2-4 GENUINELY DIFFERENT phrasings (different angles, not paraphrases). "
        f"Good: ['player health system', 'damage and healing logic', 'HP component lifecycle']. "
        f"Bad: ['player health', 'health of the player'] (too similar). "
        f"Args: queries (ordered list of variants); top_k; mode ('dense'|'sparse'|'hybrid' override); "
        f"file_glob, extensions, path_contains (same filters as `{search_tool_name}`); "
        f"min_score, min_results (threshold overrides); return_all_variants (include per-variant diagnostics)."
    )

    @mcp.tool(name=search_tool_name, description=_desc_search)
    def _search(
        query: str,
        top_k: int | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
    ) -> str:
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            results = manager.search(
                source_name,
                query,
                top_k=effective_top_k,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
            )
            filter_suffix = _build_filter_suffix(file_glob, extensions, path_contains)
            return _format_search_results(query, results, f"source {source_name!r}", filter_suffix)
        except Exception as e:
            return f"Error during search in {source_name!r}: {str(e)}"

    @mcp.tool(name=deep_tool_name, description=_desc_deep)
    def _deep_search(
        queries: list[str],
        top_k: int | None = None,
        mode: str | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
        min_score: float | None = None,
        min_results: int | None = None,
        return_all_variants: bool = False,
    ) -> str:
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            response = manager.deep_search(
                source_name,
                queries=queries,
                top_k=effective_top_k,
                mode=mode,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
                min_score=min_score,
                min_results=min_results,
                return_all_variants=return_all_variants,
            )
            meta_parts = []
            if mode:
                meta_parts.append(f"mode={mode!r}")
            if file_glob:
                meta_parts.append(f"file_glob={file_glob!r}")
            if extensions:
                meta_parts.append(f"extensions={list(extensions)!r}")
            if path_contains:
                meta_parts.append(f"path_contains={path_contains!r}")
            meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
            return _format_deep_response(response, queries, f"source {source_name!r}", meta_suffix)
        except Exception as e:
            return f"Error during deep search in {source_name!r}: {str(e)}"


def _register_global_tools(mcp, manager):
    """Register cross-source / management tools that don't depend on a
    specific source name."""

    @mcp.tool()
    def list_sources() -> str:
        """List all configured sources with their type, location, chunk count, and drift status.

        Call this first when you don't know which sources are available.
        Each source has its own `search_<name>` and `deep_search_<name>` tools.
        Use the source name from this list to pick the right search tool.
        """
        lines = [f"Sources ({len(manager.backends)}):"]
        for status in manager.list_sources():
            line = (
                f"  - {status['name']} (type: {status['type']}, "
                f"chunks: {status.get('chunk_count', 'n/a')})"
            )
            if status.get("path"):
                line += f"\n      path: {status['path']}"
            if status.get("drift_severity"):
                line += f"\n      drift: {status['drift_severity'].upper()}"
            lines.append(line)
        return "\n".join(lines)

    @mcp.tool()
    def search_all_sources(
        query: str,
        top_k: int | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
    ) -> str:
        """Search every configured source in parallel and fuse rankings via RRF.

        Useful when you don't know which source has the answer (e.g. "is X a
        feature of my code or of a library I'm using?"). Each result is tagged
        with its source. Slower than searching one source — runs N retrievals.

        When to use vs source-specific search:
        - Use this when the answer could be in ANY source and you don't want to guess.
        - Use source-specific `search_<name>` when you know which codebase to target.
        - Same query best practices apply: use natural-language, describe behavior.
        """
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            results = manager.search_all(
                query,
                top_k=effective_top_k,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
            )
            filter_suffix = _build_filter_suffix(file_glob, extensions, path_contains)
            return _format_search_results(query, results, "all sources", filter_suffix)
        except Exception as e:
            return f"Error during cross-source search: {str(e)}"

    @mcp.tool()
    def deep_search_all_sources(
        queries: list[str],
        top_k: int | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
        min_score: float | None = None,
        min_results: int | None = None,
    ) -> str:
        """Multi-query fallback search across ALL sources.

        Combines the deep_search variant-ladder with cross-source RRF fusion.
        Each variant is run on every source; results fused; first variant
        that passes the threshold wins. Use very sparingly — runs N*M
        retrievals in the worst case (N variants x M sources).

        LAST RESORT: Only use this when both `search_all_sources` and source-specific
        deep_search tools have failed. This is the most expensive operation.
        Same query variant best practices as deep_search_<source> apply.
        """
        try:
            effective_top_k = top_k if top_k is not None else manager.config.search.default_top_k
            response = manager.deep_search_all(
                queries=queries,
                top_k=effective_top_k,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
                min_score=min_score,
                min_results=min_results,
            )
            filter_suffix = _build_filter_suffix(file_glob, extensions, path_contains)
            return _format_deep_response(response, queries, "all sources", filter_suffix)
        except Exception as e:
            return f"Error during cross-source deep search: {str(e)}"

    @mcp.tool()
    def update_source_index(source: str, force: bool = False) -> str:
        """Force a full rebuild of a specific source's index.

        Day-to-day the watcher keeps the index in sync automatically; you rarely need this.
        Use it after a complex merge, a bulk rename, or when drift detection flags a
        critical change (e.g. embedding model swap).

        Do NOT call this routinely — it is expensive and blocks until complete.
        Call `get_rag_status` first to check if a rebuild is actually needed.

        Args:
            source: Name of the source to rebuild. Use `list_sources` to discover names.
            force: If True, rebuild even when no new git commits are detected.
        """
        try:
            manager.update(source, force=force)
            return f"Source {source!r} rebuilt successfully."
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rebuilding source {source!r}: {str(e)}"

    @mcp.tool()
    def get_rag_status(source: str | None = None) -> str:
        """Report state of the RAG index for one source or all sources.

        Use this to check if the index is up to date before deciding whether to
        call `update_source_index`. Also useful for debugging when search results
        seem stale or incomplete.

        Args:
            source: Specific source name to inspect. If None, returns the
                status of every configured source.
        """
        try:
            statuses = (
                [manager.get(source).status()]
                if source is not None
                else [b.status() for b in manager.backends.values()]
            )
            lines = []
            for s in statuses:
                name = s["name"]
                drift_text = manager.get(name).drift_status_text()
                needs = (
                    manager.get(name).needs_update()
                    if hasattr(manager.get(name), "needs_update")
                    else False
                )
                lines.append(f"=== Source: {name} (type: {s['type']}) ===")
                lines.append(f"Status:       {'Needs update' if needs else 'Up to date'}")
                if s.get("path"):
                    lines.append(f"Path:         {s['path']}")
                lines.append(f"Chunks:       {s.get('chunk_count', 'n/a')}")
                if s.get("last_commit"):
                    lines.append(f"Last commit:  {s['last_commit']}")
                lines.append(f"Last update:  {s.get('last_update', 'Never')}")
                lines.append("")
                lines.append(drift_text)
                lines.append("")
            return "\n".join(lines).rstrip()
        except KeyError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading status: {str(e)}"


# ----------------------------------------------------------------------
# Graph tools (registered per source when graph.enabled=true)
# ----------------------------------------------------------------------


def _format_node_brief(n: dict) -> str:
    """One-line summary of a graph node for tool text output."""
    label = n.get("label", "?")
    fp = n.get("file") or ""
    sl, el = n.get("start_line") or 0, n.get("end_line") or 0
    loc = f"{fp}:L{sl}" if sl == el else f"{fp}:L{sl}-{el}"
    return f"{label}  [{n.get('kind', '?')}] @ {loc}"


def _format_edge_lines(edges: list, header: str) -> str:
    if not edges:
        return f"{header}\n  (no results)"
    out = [header]
    for e in edges:
        src, tgt = e.get("source", {}), e.get("target", {})
        conf = e.get("confidence")
        rel = e.get("relation", "?")
        # Inheritance edges carry an extra base_kind (extends / implements /
        # extends_or_implements); surface it alongside the relation label so
        # the AI client knows whether a base was a class extension or interface.
        kind_part = ""
        if rel == "inherits" and e.get("base_kind"):
            kind_part = f"({e['base_kind']})"
        conf_part = f" [{conf}]" if conf else ""
        out.append(f"  • {_format_node_brief(src)}")
        out.append(f"      --{rel}{kind_part}{conf_part}--> {_format_node_brief(tgt)}")
        if e.get("from_file"):
            out.append(f"        at {e['from_file']}:L{e.get('from_line') or '?'}")
    return "\n".join(out)


def _register_graph_tools(mcp, manager, source_name: str):
    """Register 9 graph-layer MCP tools, namespaced as `<verb>_<source_name>`.

    The naming mirrors `search_<source>` / `deep_search_<source>` so the AI
    client sees a consistent per-source toolset. Tools are only registered
    when `backend.graph` is not None; the manager itself raises ValueError
    if a call slips through to a source whose graph is disabled.

    Tools registered: get_callers, get_callees, get_subclasses, get_superclasses,
    get_imports, get_neighbors, shortest_path, architectural_overview,
    surprising_connections, graph_status.
    """

    # NOTE on descriptions: we pass them via @mcp.tool(description=...)
    # rather than as Python docstrings. An f-string as the first statement
    # of a function is just an evaluated expression — it never becomes
    # __doc__, so FastMCP would see an empty description and the AI client
    # would have no idea when to call each tool.

    _desc_get_callers = (
        f"List functions that CALL `symbol` in source {source_name!r}. "
        f"Use when the user asks 'who calls X?', 'what uses X?', 'what depends on X?'. "
        f"The result includes file path + line so you can cite the caller. "
        f"Symbol matching is fuzzy (case-insensitive substring) — pass an identifier, "
        f"not a description. Args: symbol (function/method name); limit (max edges, default 50)."
    )

    @mcp.tool(name=f"get_callers_{source_name}", description=_desc_get_callers)
    def _get_callers(symbol: str, limit: int = 50) -> str:
        try:
            edges = manager.get_callers(source_name, symbol, limit=limit)
            return _format_edge_lines(edges, f"Callers of {symbol!r} in {source_name!r}:")
        except Exception as e:
            return f"Error: {e}"

    _desc_get_callees = (
        f"List functions CALLED BY `symbol` in source {source_name!r}. "
        f"Use when the user asks 'what does X call?', 'what does X depend on?', "
        f"'trace what X does'. Args: symbol (function/method name); limit (max edges, default 50)."
    )

    @mcp.tool(name=f"get_callees_{source_name}", description=_desc_get_callees)
    def _get_callees(symbol: str, limit: int = 50) -> str:
        try:
            edges = manager.get_callees(source_name, symbol, limit=limit)
            return _format_edge_lines(edges, f"Callees of {symbol!r} in {source_name!r}:")
        except Exception as e:
            return f"Error: {e}"

    _desc_get_subclasses = (
        f"List concrete types that INHERIT FROM `symbol` in source {source_name!r}. "
        f"Use when the user asks 'what are the concrete subclasses of X?', "
        f"'who implements interface X?', 'who derives from base class X?'. "
        f"Works for `extends`, `implements`, and language-specific bases "
        f"(C# `:`, Python multiple-bases, Rust `impl Trait for Type`, ...). "
        f"Each edge carries `base_kind` (extends/implements/extends_or_implements) "
        f"when the language exposes the distinction. "
        f"Args: symbol (class/interface name); limit (max edges, default 50)."
    )

    @mcp.tool(name=f"get_subclasses_{source_name}", description=_desc_get_subclasses)
    def _get_subclasses(symbol: str, limit: int = 50) -> str:
        try:
            edges = manager.get_subclasses(source_name, symbol, limit=limit)
            return _format_edge_lines(edges, f"Subclasses of {symbol!r} in {source_name!r}:")
        except Exception as e:
            return f"Error: {e}"

    _desc_get_superclasses = (
        f"List the types that `symbol` INHERITS FROM in source {source_name!r}. "
        f"Use when the user asks 'what does X extend?', 'which interfaces does X implement?', "
        f"'what's the base class of X?'. Returns the out-edge mirror of get_subclasses. "
        f"Args: symbol (class/interface name); limit (max edges, default 50)."
    )

    @mcp.tool(name=f"get_superclasses_{source_name}", description=_desc_get_superclasses)
    def _get_superclasses(symbol: str, limit: int = 50) -> str:
        try:
            edges = manager.get_superclasses(source_name, symbol, limit=limit)
            return _format_edge_lines(edges, f"Superclasses of {symbol!r} in {source_name!r}:")
        except Exception as e:
            return f"Error: {e}"

    _desc_get_imports = (
        f"List import edges originating from a file (or from the file that contains "
        f"the given symbol) in source {source_name!r}. Pass either a file path substring "
        f"(e.g. 'main.py') or the name of a symbol defined in the file. Useful for "
        f"'what does this file depend on?', 'what third-party packages are used here?'. "
        f"Args: file_or_symbol; limit (max edges, default 100)."
    )

    @mcp.tool(name=f"get_imports_{source_name}", description=_desc_get_imports)
    def _get_imports(file_or_symbol: str, limit: int = 100) -> str:
        try:
            edges = manager.get_imports(source_name, file_or_symbol, limit=limit)
            return _format_edge_lines(edges, f"Imports from {file_or_symbol!r} in {source_name!r}:")
        except Exception as e:
            return f"Error: {e}"

    _desc_get_neighbors = (
        f"All graph neighbors of `symbol` within `depth` hops in source {source_name!r}, "
        f"optionally filtered by edge relation. Use for 'show me everything around X', "
        f"'expand context on X'. When you only need callers OR callees, prefer the dedicated tools. "
        f"Args: symbol; relation_filter (optional 'calls' | 'imports' | 'imports_from' | 'contains'); "
        f"depth (1-6, default 1); limit (max edges, default 100)."
    )

    @mcp.tool(name=f"get_neighbors_{source_name}", description=_desc_get_neighbors)
    def _get_neighbors(
        symbol: str,
        relation_filter: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> str:
        try:
            edges = manager.get_neighbors(
                source_name, symbol,
                relation_filter=relation_filter, depth=depth, limit=limit,
            )
            label = f"Neighbors of {symbol!r}"
            if relation_filter:
                label += f" (relation={relation_filter!r})"
            label += f" depth={depth} in {source_name!r}:"
            return _format_edge_lines(edges, label)
        except Exception as e:
            return f"Error: {e}"

    _desc_shortest_path = (
        f"Shortest directed path between two symbols in source {source_name!r}. "
        f"Use for 'how does A reach B?', 'what's the call chain from A to B?'. "
        f"Both endpoints are resolved fuzzily; if multiple matches exist on either side, "
        f"the shortest path among combinations is returned. "
        f"Args: source (starting symbol); target (destination symbol); max_hops (1-20, default 8)."
    )

    @mcp.tool(name=f"shortest_path_{source_name}", description=_desc_shortest_path)
    def _shortest_path(source: str, target: str, max_hops: int = 8) -> str:
        try:
            path = manager.shortest_path(source_name, source, target, max_hops=max_hops)
            if path is None:
                return f"No directed path from {source!r} to {target!r} (within {max_hops} hops)."
            lines = [f"Path from {source!r} → {target!r} ({path['hops']} hops):"]
            for n in path["nodes"]:
                lines.append(f"  • {_format_node_brief(n)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    _desc_overview = (
        f"High-level architectural snapshot of source {source_name!r}: top god-nodes "
        f"(most-connected symbols — the architectural hubs), detected communities "
        f"(tightly-coupled symbol groups), and basic graph stats. Use for "
        f"'give me an overview of this codebase', 'what are the main abstractions?', "
        f"'how is this organized?'. Call once at the start of an unfamiliar session. "
        f"Args: top_n_gods (default 10); min_community_size (default 3)."
    )

    @mcp.tool(name=f"architectural_overview_{source_name}", description=_desc_overview)
    def _architectural_overview(top_n_gods: int = 10, min_community_size: int = 3) -> str:
        try:
            ov = manager.architectural_overview(
                source_name, top_n_gods=top_n_gods, min_community_size=min_community_size,
            )
            lines = [f"=== Architectural overview of {source_name!r} ==="]
            st = ov.get("status", {})
            lines.append(f"Graph: {st.get('nodes', '?')} nodes, {st.get('edges', '?')} edges, "
                         f"{st.get('files_indexed', '?')} files, last_update={st.get('last_update')}")
            lines.append("\n--- God nodes (most-connected) ---")
            for g in ov["god_nodes"]:
                lines.append(f"  • {g['label']:40}  degree={g['degree']}  "
                             f"(in={g['in_degree']}, out={g['out_degree']})  "
                             f"{g.get('file', '')}")
            lines.append(f"\n--- Communities ({len(ov['communities'])}) ---")
            for c in ov["communities"][:10]:
                sample = ", ".join(c["members_sample"][:5])
                more = "" if c["size"] <= 5 else f", +{c['size'] - 5} more"
                lines.append(f"  [{c['id']}] {c['name']!r}  size={c['size']}  "
                             f"langs={c['by_language']}  members: {sample}{more}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    _desc_surprising = (
        f"Edges that bridge distant parts of source {source_name!r}, computed via "
        f"edge betweenness centrality. High-betweenness edges are structural bottlenecks: "
        f"removing them would split the codebase into disconnected components. Useful for "
        f"spotting non-obvious dependencies and god-class antipatterns. "
        f"Args: top_n (number of bridge edges to return, default 5)."
    )

    @mcp.tool(name=f"surprising_connections_{source_name}", description=_desc_surprising)
    def _surprising(top_n: int = 5) -> str:
        try:
            surprises = manager.surprising_connections(source_name, top_n=top_n)
            if not surprises:
                return f"No surprising connections detected in {source_name!r}."
            lines = [f"Top {len(surprises)} bridge edges in {source_name!r} (by betweenness):"]
            for s in surprises:
                lines.append(
                    f"  • {s['source_label']!r} --{s['relation']}--> {s['target_label']!r}  "
                    f"betweenness={s['betweenness']}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    _desc_graph_status = (
        f"Counts and metadata for the graph layer of source {source_name!r}: total "
        f"nodes/edges, breakdowns by language and kind, count of unresolved cross-file "
        f"calls, last update timestamp. Useful for diagnosing why a query returned empty results."
    )

    @mcp.tool(name=f"graph_status_{source_name}", description=_desc_graph_status)
    def _graph_status() -> str:
        try:
            st = manager.graph_status(source_name)
            lines = [f"=== Graph status for {source_name!r} ==="]
            lines.append(f"Schema version:    {st['schema_version']}")
            lines.append(f"Nodes:             {st['nodes']}")
            lines.append(f"Edges:             {st['edges']}")
            lines.append(f"Files indexed:     {st['files_indexed']}")
            lines.append(f"Raw calls pending: {st['raw_calls_pending']}")
            lines.append(f"Raw inherits pending: {st.get('raw_inherits_pending', 0)}")
            lines.append(f"Last update:       {st['last_update']}")
            lines.append(f"Last full rebuild: {st['last_full_rebuild']}")
            lines.append(f"By language: {st['by_language']}")
            lines.append(f"By kind:     {st['by_kind']}")
            lines.append(f"By relation: {st['by_relation']}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


# ----------------------------------------------------------------------
# Main server entry point
# ----------------------------------------------------------------------


def run_server(config_path=None):
    """Boot the MCP server. Blocks on mcp.run() until the client disconnects."""
    config = load_config(config_path=config_path)
    mcp = FastMCP("lynx")

    state = {
        "manager": None,
        "ready": threading.Event(),
        "error": None,
    }

    def _load_background():
        try:
            from .source_manager import SourceManager
            mgr = SourceManager(config)
            state["manager"] = mgr
        except Exception as e:
            state["error"] = str(e)
            state["ready"].set()
            return

        try:
            mgr.start_watchers()
        except Exception as e:
            print(f"[server] failed to start watchers: {e}", file=sys.stderr)
        state["ready"].set()

    threading.Thread(target=_load_background, daemon=True).start()

    # Wait for the manager to load before opening the JSON-RPC transport.
    # Any spurious print() during model load (llama_index's MockLLM warning,
    # sentence_transformers logs) lands on stderr because fd 1 is redirected.
    state["ready"].wait(timeout=config.loading_timeout_seconds)

    if state["error"] is not None:
        print(f"[server] FATAL: source manager failed to load: {state['error']}", file=sys.stderr)
        sys.exit(1)
    if state["manager"] is None:
        print(
            "[server] FATAL: source manager did not load within "
            f"{config.loading_timeout_seconds}s (loading_timeout_seconds in config).",
            file=sys.stderr,
        )
        sys.exit(1)

    manager = state["manager"]

    # Dynamically register tools — two per source plus the global ones.
    for name, backend in manager.backends.items():
        _register_source_tools(
            mcp, manager, name, backend.type_name, backend.source_config
        )
    _register_global_tools(mcp, manager)

    # Per-source graph tools — only when the source has graph.enabled=true.
    # We probe `backend.graph` rather than `backend.type_name == "codebase"`
    # so future source types (e.g. a pdf backend with a graph) can opt in
    # too without modifying this loop.
    for name, backend in manager.backends.items():
        if getattr(backend, "graph", None) is not None:
            _register_graph_tools(mcp, manager, name)

    # Loading phase done: restore fd 1 and start the transport.
    _restore_real_stdout()
    mcp.run()


if __name__ == "__main__":
    # Allow `python -m lynx.server` for ad-hoc invocation.
    # Normal entry point goes through cli:main (`lynx serve`).
    run_server()
