"""Output formatting helpers for the MCP tools.

Pure presentation: each function turns a result/edge/summary dict into the plain
text the MCP tool returns. Extracted from `server.py` (which had grown to ~1.5k
lines) so the tool-registration logic and the rendering live apart. `server.py`
re-imports every name here, so `lynx.server._format_*` keeps resolving for
callers/tests that referenced them at their old home.
"""
from __future__ import annotations

from .outline import doc_of, signature_for


# ----------------------------------------------------------------------
# Search / deep-search rendering
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


def _format_one_outline(i, res):
    """Compact, body-free render of one result for cheap triage: the same
    header/citation as the full view, but `signature` + a doc line instead of
    the body. The agent reads the real body on demand (find_definition or the
    cited file:line). Derivation shared with `/api/v1/search?view=outline`."""
    score = res.get("score", 0.0)
    fname = res.get("file", "unknown")
    sym = res.get("symbol_name") or ""
    start = res.get("start_line") or 0
    end = res.get("end_line") or 0
    loc_parts = [fname]
    if start and end and start != end:
        loc_parts.append(f"L{start}-{end}")
    elif start:
        loc_parts.append(f"L{start}")
    loc = " ".join(loc_parts)
    sym_suffix = f"  {sym}" if sym and not sym.startswith("<") else ""
    out = f"--- {i}. {loc}{sym_suffix} (Score: {score:.4f}) ---\n"
    if "source" in res:
        out += f"    source: {res['source']}\n"
    content = res.get("content", "")
    out += f"    {signature_for(content, res.get('symbol_kind', ''), res.get('language', ''))}\n"
    doc = doc_of(content, res.get("language", ""))
    if doc:
        out += f"    ↳ {doc}\n"
    return out + "\n"


def _format_outline_results(query, results, source_label, filter_suffix):
    if not results:
        return f"No results found for '{query}' in {source_label}{filter_suffix}."
    out = (
        f"Found {len(results)} results for '{query}' in {source_label}"
        f"{filter_suffix} — OUTLINE (signatures only; read a body you need with "
        f"find_definition, or by its file:line):\n\n"
    )
    for i, res in enumerate(results, 1):
        out += _format_one_outline(i, res)
    return out


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
# Graph edge / node rendering
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


# ----------------------------------------------------------------------
# Combined-tool rendering (find_* / describe / impact / module / overview)
# ----------------------------------------------------------------------


def _format_definition_results(symbol: str, results: list) -> str:
    if not results:
        return f"No definition found for {symbol!r}."
    lines = [f"Definitions of {symbol!r}:"]
    for r in results:
        loc = f"{r.get('file', '?')}:L{r.get('start_line') or '?'}"
        if r.get("end_line") and r["end_line"] != r.get("start_line"):
            loc += f"-{r['end_line']}"
        lines.append(f"  • {r.get('symbol', symbol)}  [{r.get('kind', '?')}]  @ {loc}  ({r.get('source')})")
    return "\n".join(lines)


def _format_usage_results(symbol: str, results: list) -> str:
    if not results:
        return f"No usages found for {symbol!r}."
    lines = [f"Usages of {symbol!r}:"]
    for r in results:
        loc = f"{r.get('file', '?')}:L{r.get('start_line') or '?'}"
        if r.get("end_line") and r["end_line"] != r.get("start_line"):
            loc += f"-{r['end_line']}"
        src_tag = r.get("source", "")
        rel = r.get("edge_relation", "")
        rel_part = f" [{rel}]" if rel else ""
        lines.append(f"  • {r.get('symbol') or '?'}  @ {loc}  ({src_tag}{rel_part})")
    return "\n".join(lines)


def _format_test_results(symbol: str, results: list) -> str:
    if not results:
        return f"No tests found mentioning {symbol!r}."
    lines = [f"Tests mentioning {symbol!r} ({len(results)}):"]
    for r in results:
        loc = f"{r.get('file', '?')}:L{r.get('start_line') or '?'}-{r.get('end_line') or '?'}"
        lines.append(f"  • {r.get('symbol') or '?'}  @ {loc}  score={r.get('score'):.3f}" if isinstance(r.get('score'), (int, float)) else f"  • {r.get('symbol') or '?'}  @ {loc}")
    return "\n".join(lines)


def _describe_loc(node: dict) -> str:
    """`file:Lstart-end` for a node/result dict, tolerating missing lines."""
    loc = f"{node.get('file', '?')}:L{node.get('start_line') or '?'}"
    if node.get("end_line") and node["end_line"] != node.get("start_line"):
        loc += f"-{node['end_line']}"
    return loc


def _format_describe_symbol(symbol: str, d: dict) -> str:
    definition = d.get("definition") or []
    called_by = d.get("called_by") or []
    calls = d.get("calls") or []
    tests = d.get("tests") or []

    if not (definition or called_by or calls or tests):
        return f"No information found for {symbol!r} in this codebase."

    lines = [f"Symbol context for {symbol!r}:", ""]

    lines.append("DEFINITION:")
    if definition:
        for r in definition:
            lines.append(
                f"  • {r.get('symbol', symbol)}  [{r.get('kind', '?')}]  "
                f"@ {_describe_loc(r)}  ({r.get('source')})"
            )
    else:
        lines.append("  (not found)")

    # called_by / calls are graph edge dicts (source/target node dicts).
    lines.append("")
    lines.append("CALLED BY:" if d.get("graph_enabled") else "CALLED BY: (enable the graph layer for call data)")
    for e in called_by:
        caller = e.get("source") or {}
        lines.append(f"  • {caller.get('label', '?')}  @ {_describe_loc(caller)}  ({e.get('confidence', '?')})")
    if d.get("graph_enabled") and not called_by:
        lines.append("  (none)")

    lines.append("")
    lines.append("CALLS:" if d.get("graph_enabled") else "CALLS: (enable the graph layer for call data)")
    for e in calls:
        callee = e.get("target") or {}
        lines.append(f"  • {callee.get('label', '?')}  @ {_describe_loc(callee)}  ({e.get('confidence', '?')})")
    if d.get("graph_enabled") and not calls:
        lines.append("  (none)")

    lines.append("")
    lines.append("TESTS:")
    if tests:
        for r in tests:
            lines.append(f"  • {r.get('symbol') or '?'}  @ {_describe_loc(r)}")
    else:
        lines.append("  (none found)")

    return "\n".join(lines)


def _format_impact(symbol: str, d: dict) -> str:
    callers = d.get("callers") or []
    tests = d.get("tests") or []
    if not callers and not tests:
        if not d.get("graph_enabled"):
            return (f"Impact of {symbol!r}: enable the graph layer for transitive "
                    f"callers. No tests found either.")
        return f"Impact of {symbol!r}: no callers and no tests found."

    lines = [f"Impact / blast radius of {symbol!r}:", ""]
    lines.append(
        "REACHED BY (transitive callers, [dN] = hops away):" if d.get("graph_enabled")
        else "REACHED BY: (enable the graph layer for call data)"
    )
    for c in callers:
        node = c.get("node") or {}
        lines.append(
            f"  [d{c.get('depth')}] {node.get('label', '?')}  "
            f"@ {_describe_loc(node)}  ({c.get('confidence', '?')})"
        )
    if d.get("graph_enabled") and not callers:
        lines.append("  (none — nothing calls it)")

    lines.append("")
    lines.append("TESTS to re-run:")
    if tests:
        for r in tests:
            lines.append(f"  • {r.get('symbol') or '?'}  @ {_describe_loc(r)}")
    else:
        lines.append("  (none found)")
    return "\n".join(lines)


def _format_module_summary(file: str, d: dict) -> str:
    if not d.get("graph_enabled"):
        return (f"Module summary for {file!r} needs the graph layer — enable "
                f"`graph` for this source.")
    symbols = d.get("symbols") or []
    imports = d.get("imports") or []
    deps = d.get("dependent_files") or []
    if not (symbols or imports or deps):
        return f"No graph data for {file!r} (no matching file or symbols)."

    lines = [f"Module summary for {file!r}:", ""]
    lines.append(f"DEFINES ({len(symbols)}):")
    for s in symbols:
        lines.append(f"  • {s.get('label', '?')}  [{s.get('kind', '?')}]  @ {_describe_loc(s)}")
    if not symbols:
        lines.append("  (none)")

    lines.append("")
    lines.append("IMPORTS:")
    for e in imports:
        tgt = e.get("target") or {}
        lines.append(f"  • {e.get('module') or tgt.get('label') or '?'}")
    if not imports:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"DEPENDED ON BY ({len(deps)} file(s), via call graph):")
    for f in deps:
        lines.append(f"  • {f}")
    if not deps:
        lines.append("  (none found)")
    return "\n".join(lines)


def _format_repo_overview(d: dict) -> str:
    if d.get("error"):
        return f"repo_overview: {d['error']}"
    lines = [
        f"Repository overview — {d.get('root', '?')}  ({d.get('file_count', '?')} files):",
        "",
    ]
    langs = d.get("languages") or []
    lines.append("LANGUAGES: " + (
        ", ".join(f"{l['language']} ({l['files']})" for l in langs) or "(none detected)"
    ))
    lines.append("FRAMEWORKS: " + (", ".join(d.get("frameworks") or []) or "(none detected)"))
    lines.append("MANIFESTS: " + (", ".join(d.get("manifests") or []) or "(none)"))

    lines.append("")
    lines.append("ENTRY POINTS:")
    eps = d.get("entry_points") or []
    for e in eps:
        lines.append(f"  • {e['file']}  — {e['hint']}")
    if not eps:
        lines.append("  (none detected)")

    lines.append("")
    lines.append("COMMANDS:")
    cmds = d.get("commands") or {}
    any_cmd = False
    for label in ("build", "test", "run"):
        for c in cmds.get(label, []):
            lines.append(f"  {label}: {c}")
            any_cmd = True
    if not any_cmd:
        lines.append("  (none detected)")
    return "\n".join(lines)


def _format_similar_results(results: list) -> str:
    if not results:
        return "No similar code found."
    lines = [f"Found {len(results)} similar chunks:"]
    for r in results:
        loc = f"{r.get('file', '?')}:L{r.get('start_line') or '?'}-{r.get('end_line') or '?'}"
        score = r.get('score')
        score_str = f"  score={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"  • {r.get('symbol') or '?'}  @ {loc}{score_str}")
        content = r.get('content', '')
        if content:
            snippet = content[:200].replace("\n", " ")
            if len(content) > 200:
                snippet += "..."
            lines.append(f"      {snippet}")
    return "\n".join(lines)
