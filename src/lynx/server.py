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

    @mcp.tool(name=search_tool_name)
    def _search(
        query: str,
        top_k: int | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
    ) -> str:
        f"""Semantic search over the {source_name!r} source (type: {source_type}).

        Indexed location: {path_hint}

        Use natural-language descriptions of what the code does, not exact
        identifiers — that's where semantic search beats grep.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return. Defaults to search.default_top_k.
            file_glob: Optional Unix-shell glob (e.g. "**/Bullet*/*.cs").
            extensions: Optional list of file extensions (e.g. [".py", ".pyi"]).
            path_contains: Optional substring required in the file path.
        """
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

    @mcp.tool(name=deep_tool_name)
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
        f"""Deeper, fallback search over the {source_name!r} source
        (type: {source_type}).

        Use ONLY when `{search_tool_name}` returned weak or empty results, or
        when the user explicitly asks for a more thorough search.

        Tries each query variant in order. Stops at the first variant whose
        results pass the weakness threshold. If all variants fail, returns
        the strongest weak set with a warning.

        Args:
            queries: Ordered list of query variants. Use *genuinely different*
                phrasings, not paraphrases.
            top_k: Defaults to search.default_top_k.
            mode: Per-call mode override: "dense" | "sparse" | "hybrid".
            file_glob, extensions, path_contains: Same as the matching
                `{search_tool_name}` params.
            min_score: Per-call weakness threshold override.
            min_results: Per-call min-results override.
            return_all_variants: Include per-variant summary in the response.
        """
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
        """List all configured sources with their type, location, chunk
        count, and drift status."""
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

        Day-to-day the watcher keeps the index in sync; use this after a
        complex merge, a bulk rename, or when drift detection flags a
        critical change (e.g. embedding model swap).

        Args:
            source: Name of the source to rebuild. Use `list_sources` to
                discover available names.
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

    # Loading phase done: restore fd 1 and start the transport.
    _restore_real_stdout()
    mcp.run()


if __name__ == "__main__":
    # Allow `python -m lynx.server` for ad-hoc invocation.
    # Normal entry point goes through cli:main (`lynx serve`).
    run_server()
