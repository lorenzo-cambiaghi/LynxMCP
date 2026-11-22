"""Output formatting helpers for the MCP tools.

Pure presentation: each function turns a result/edge/summary dict into the plain
text the MCP tool returns. Extracted from `server.py` (which had grown to ~1.5k
lines) so the tool-registration logic and the rendering live apart. `server.py`
re-imports every name here, so `lynx.server._format_*` keeps resolving for
callers/tests that referenced them at their old home.
"""
from __future__ import annotations

from .fs_scan import to_rel_path
from .outline import doc_of, signature_for


# ----------------------------------------------------------------------
# Search / deep-search rendering
# ----------------------------------------------------------------------


# How much code one answer carries. Chunks are small as a rule (median 120
# to 280 characters on four real indexes, p99 about 4,000), but a search
# that met a few generated methods of 8,000 to 12,000 characters each used
# to hand the model 40,000 or more. A body is cut past RESULT_BODY_CHARS (1%
# of chunks are that long), and once a response holds RESPONSE_BODY_CHARS
# of bodies, the remaining hits come as outlines: still ranked and cited,
# one read away.
RESULT_BODY_CHARS = 4000
RESPONSE_BODY_CHARS = 16000
# What the MCP `search` answer carries: the best hit with its code (or the
# first few, when they are short), the rest as signatures. An agent that is
# about to edit reads the lines anyway, so bodies past the first were paid
# twice. Measured on the Sketchbook bench (2026-11-21): two default searches
# returned 4,000-6,400 characters each and Claude still opened the files;
# with signatures the same hits took about 1,000. The CLI keeps the full view.
SEARCH_BODY_CHARS = 2000


def _display_path(item: dict, root=None) -> str:
    """Where a result lives, as the model should read it: the path inside its
    source (`django/urls/resolvers.py`), not a bare file name that a large
    repository repeats in a hundred folders, and not the absolute path that
    repeats the same prefix on every line. `rel_path` when the result carries
    one; otherwise the absolute path made relative to `root`."""
    rel = item.get("rel_path")
    if rel:
        return rel
    path = item.get("file_path") or item.get("file") or ""
    if not path:
        return "?"
    return to_rel_path(path, root)


def _line_span(start, end) -> str:
    """`L12-30`, `L12`, or "" when the lines are unknown."""
    if start and end and start != end:
        return f"L{start}-{end}"
    return f"L{start}" if start else ""


def _loc(item: dict, root=None) -> str:
    """`path:L12-30` for a result, node or edge endpoint."""
    span = _line_span(item.get("start_line") or 0, item.get("end_line") or 0)
    path = _display_path(item, root)
    return f"{path}:{span}" if span else path


def _capped(content: str, limit: int) -> tuple:
    """(text, lines left out). Cut at a line break so no line is half shown."""
    if len(content) <= limit:
        return content, 0
    cut = content.rfind("\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    shown = content[:cut]
    return shown, content.count("\n", cut) + (0 if content.endswith("\n") else 1)


def _result_header(i, res) -> str:
    score = res.get("score", 0.0)
    sym = res.get("symbol_name") or ""
    sym_suffix = f"  {sym}" if sym and not sym.startswith("<") else ""
    span = _line_span(res.get("start_line") or 0, res.get("end_line") or 0)
    loc = f"{_display_path(res)} {span}".rstrip()
    head = f"--- {i}. {loc}{sym_suffix} (Score: {score:.4f}) ---\n"
    if "source" in res:
        head += f"    source: {res['source']}\n"
    return head


def _format_one_result(i, res, limit: int = RESULT_BODY_CHARS):
    """Render one result block: header with symbol/line context + content body.

    The header surfaces the AST chunker metadata so the AI can cite the
    result precisely (e.g. "see MyClass.handleClick in Combat/Foo.cs L42-58")
    instead of just naming the file. A body longer than `limit` is cut, and
    the cut says which lines are left to read.
    """
    content = res.get("content", "") or ""
    shown, left = _capped(content, limit)
    body = shown.rstrip("\n")
    if left:
        end = res.get("end_line") or 0
        first = end - left + 1 if end else 0
        where = f"{_display_path(res)}:{_line_span(first, end)}" if first > 0 else _display_path(res)
        body += f"\n    [... {left} more lines, cut at {limit} chars: read {where}]"
    return _result_header(i, res) + body + "\n\n"


def _format_one_outline(i, res):
    """Compact, body-free render of one result for cheap triage: the same
    header/citation as the full view, but `signature` + a doc line instead of
    the body. The agent reads the real body on demand (find_definition or the
    cited file:line). Derivation shared with `/api/v1/search?view=outline`."""
    out = _result_header(i, res)
    content = res.get("content", "")
    out += f"    {signature_for(content, res.get('symbol_kind', ''), res.get('language', ''))}\n"
    doc = doc_of(content, res.get("language", ""))
    if doc:
        out += f"    ↳ {doc}\n"
    return out + "\n"


# Less room than this left in the budget: the next hit comes as an outline
# rather than as a few lines of its body.
_MIN_BODY_CHARS = 300


def _render_results(results, budget: int = RESPONSE_BODY_CHARS) -> str:
    """Bodies until the response holds `budget` characters of them, then
    outlines for the rest, with one line saying so. Each body is cut to the
    room left (and to RESULT_BODY_CHARS), so the budget is a ceiling."""
    out, used, switched = [], 0, False
    for i, res in enumerate(results, 1):
        room = min(RESULT_BODY_CHARS, budget - used)
        if room < _MIN_BODY_CHARS or switched:
            if not switched:
                out.append(f"(Output budget of {budget} chars of code reached: "
                           f"results {i}-{len(results)} as outlines; read a body by its "
                           f"file:line.)\n\n")
                switched = True
            out.append(_format_one_outline(i, res))
            continue
        out.append(_format_one_result(i, res, room))
        # What the answer holds, not the chunk: an 8,000-char chunk shows 4,000.
        used += len(_capped(res.get("content") or "", room)[0])
    return "".join(out)


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


def _format_search_results(query, results, source_label, filter_suffix,
                           budget: int = RESPONSE_BODY_CHARS):
    if not results:
        return f"No results found for '{query}' in {source_label}{filter_suffix}."
    out = f"Found {len(results)} results for '{query}' in {source_label}{filter_suffix}:\n\n"
    return out + _render_results(results, budget)


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

    out = header + _render_results(results)

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


def _format_search_diff(source: str, out: dict, root=None) -> str:
    """Render a `search_diff` response: base branch, the modified files, and
    the hits inside them. Lived inline in the MCP tool until `lynx
    search-diff` needed the same text."""
    modified = out.get("modified_files", [])
    lines = [
        f"search_diff in {source!r} vs base {out.get('base')!r}:",
        f"  Modified files ({len(modified)}): "
        + ", ".join(modified[:20])
        + ("..." if len(modified) > 20 else ""),
    ]
    if out.get("note"):
        lines.append(f"  Note: {out['note']}")
        return "\n".join(lines)
    hits = out.get("hits", [])
    if not hits:
        lines.append("  No matching chunks in the modified files.")
        return "\n".join(lines)
    lines.append(f"  Hits ({len(hits)}):")
    for h in hits:
        score = h.get("score")
        score_str = f"  score={score:.4f}" if isinstance(score, (int, float)) else ""
        sym = h.get("symbol_name") or ""
        sym_part = f"  {sym}" if sym and not sym.startswith("<") else ""
        lines.append(f"    • {_loc(h, root)}{sym_part}{score_str}")
    return "\n".join(lines)


def _format_node_brief(n: dict, root=None) -> str:
    """One-line summary of a graph node for tool text output."""
    return f"{n.get('label', '?')}  [{n.get('kind', '?')}] @ {_loc(n, root)}"


def _member_line(n: dict, extra: str = "") -> str:
    """`Label [kind] L12-30` plus an optional tail, for a node listed under
    its file's heading (the file is not repeated)."""
    span = _line_span(n.get("start_line") or 0, n.get("end_line") or 0)
    head = f"{n.get('label', '?')} [{n.get('kind', '?')}]"
    return " ".join(p for p in (head, span, extra) if p)


def _by_file(items, path_of) -> list:
    """[(path, [items])] in first-seen order: a list grouped under one heading
    per file instead of the same path on every line."""
    groups: dict = {}
    for it in items:
        groups.setdefault(path_of(it), []).append(it)
    return list(groups.items())


# How sure the graph is of an edge: `extracted` (both ends in one file) and
# `resolved` (one cross-file candidate) are the norm; only the rest is worth
# a word on every line.
_SURE_EDGES = frozenset({"resolved", "extracted"})


def _conf_tag(conf) -> str:
    """Only the exception is spelled out: an edge is certain unless it says so."""
    return f" ({conf})" if conf and conf not in _SURE_EDGES else ""


# Which end of an edge a one-directional operation is about: the callers of
# X are the sources of its `calls` in-edges, its callees the targets, and so on.
EDGE_FAR_END = {
    "callers": "source",
    "callees": "target",
    "subclasses": "source",
    "superclasses": "target",
    "imports": "target",
}


def _format_edge_lines(edges: list, header: str, root=None, far: str | None = None) -> str:
    """Render graph edges.

    With `far` ("source" or "target") the edges all hang off the queried
    symbol, so that end is said once and each edge is one line under the
    file of its other end. Without it (neighbors), one line per edge with
    both ends.
    """
    if not edges:
        return f"{header}\n  (no results)"
    out = [f"{header.rstrip(':')} ({len(edges)}):"]
    if far in ("source", "target"):
        near = "target" if far == "source" else "source"
        seeds = {(e.get(near) or {}).get("label") for e in edges}
        if len(seeds) == 1:
            seed = edges[0].get(near) or {}
            out[0] += f"  [{seed.get('label', '?')} @ {_loc(seed, root)}]"

        def other_path(e):
            node = e.get(far) or {}
            return _display_path(node, root) if (node.get("file") or node.get("rel_path")) else "(external)"

        for path, group in _by_file(edges, other_path):
            out.append(f"  {path}")
            for e in group:
                node = e.get(far) or {}
                bits = []
                if e.get("relation") == "inherits" and e.get("base_kind"):
                    bits.append(e["base_kind"])
                if e.get("relation") in ("imports", "imports_from") and e.get("module"):
                    bits.append(f"module {e['module']}")
                line = e.get("from_line")
                if line and e.get("relation") == "calls":
                    same = e.get("from_file") and e.get("from_file") == node.get("file")
                    site = e.get("from_rel_path") or to_rel_path(e.get("from_file"), root)
                    bits.append(f"call at L{line}" if same or not e.get("from_file")
                                else f"call at {site}:L{line}")
                if len(seeds) > 1:
                    bits.append(f"-> {(e.get(near) or {}).get('label', '?')}")
                tail = ", ".join(bits) + _conf_tag(e.get("confidence"))
                out.append(f"    • {_member_line(node, tail.strip())}")
        return "\n".join(out)
    for e in edges:
        src, tgt = e.get("source", {}), e.get("target", {})
        rel = e.get("relation", "?")
        kind_part = f"({e['base_kind']})" if rel == "inherits" and e.get("base_kind") else ""
        at = f" at L{e['from_line']}" if e.get("from_line") else ""
        out.append(f"  • {_format_node_brief(src, root)} --{rel}{kind_part}{at}-->"
                   f" {_format_node_brief(tgt, root)}{_conf_tag(e.get('confidence'))}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Combined-tool rendering (find_* / describe / impact / module / overview)
# ----------------------------------------------------------------------


def _code_block(text: str, indent: str = "    ") -> str:
    return "\n".join(indent + ln for ln in (text or "").rstrip("\n").splitlines())


def _definition_lines(r: dict, root=None, *, with_body: bool) -> list:
    """A definition: where it is, then its signature and doc, or its body."""
    lines = [f"  • {r.get('symbol') or '?'} [{r.get('kind', '?')}] @ {_loc(r, root)}"
             f"  ({r.get('source')})"]
    if with_body and r.get("body"):
        lines.append(_code_block(r["body"]))
        if r.get("body_cut"):
            lines.append(f"    [... {r['body_cut']} more lines: read {_loc(r, root)}]")
    else:
        if r.get("signature"):
            lines.append(f"      {r['signature']}")
        if r.get("doc"):
            lines.append(f"      ↳ {r['doc']}")
    return lines


def _format_definition_results(symbol: str, results: list, root=None) -> str:
    if not results:
        return f"No definition found for {symbol!r}."
    lines = [f"Definitions of {symbol!r} ({len(results)}):"]
    for i, r in enumerate(results):
        lines.extend(_definition_lines(r, root, with_body=(i == 0)))
    return "\n".join(lines)


def _usage_tail(r: dict) -> str:
    if r.get("edge_relation") == "calls":
        return "calls it"
    if r.get("edge_relation") == "inherits":
        return f"inherits ({r['base_kind']})" if r.get("base_kind") else "inherits"
    hits = r.get("match_lines") or []
    if hits:
        more = f" +{len(hits) - 3}" if len(hits) > 3 else ""
        return "mentions it at " + ", ".join(f"L{n}" for n in hits[:3]) + more
    return "mentions it"


def _format_usage_results(symbol: str, results: list, root=None) -> str:
    if not results:
        return f"No usages found for {symbol!r}."
    lines = [f"Usages of {symbol!r} ({len(results)}):"]
    for path, group in _by_file(results, lambda r: _display_path(r, root)):
        lines.append(f"  {path}")
        for r in group:
            node = {"label": r.get("symbol") or "?", "kind": r.get("kind") or "?",
                    "start_line": r.get("start_line"), "end_line": r.get("end_line")}
            lines.append(f"    • {_member_line(node, _usage_tail(r))}")
    return "\n".join(lines)


def _test_tail(r: dict) -> str:
    if r.get("depth"):
        return f"[d{r['depth']}]" + ("" if r["depth"] == 1 else " via the call graph")
    return "mentions it"


def _format_test_results(symbol: str, results: list, root=None) -> str:
    if not results:
        return f"No tests found for {symbol!r}."
    lines = [f"Tests for {symbol!r} ({len(results)}; [dN] = N calls away):"]
    for path, group in _by_file(results, lambda r: _display_path(r, root)):
        lines.append(f"  {path}")
        for r in group:
            node = {"label": r.get("symbol") or "?", "kind": r.get("kind") or "?",
                    "start_line": r.get("start_line"), "end_line": r.get("end_line")}
            lines.append(f"    • {_member_line(node, _test_tail(r))}")
    return "\n".join(lines)


def _describe_loc(node: dict, root=None) -> str:
    """`file:Lstart-end` for a node/result dict, tolerating missing lines."""
    return _loc(node, root)


def _edge_end_lines(edges: list, end: str, root=None) -> list:
    return [f"  • {(e.get(end) or {}).get('label', '?')}  @ {_loc(e.get(end) or {}, root)}"
            f"{_conf_tag(e.get('confidence'))}" for e in edges]


def _format_describe_symbol(symbol: str, d: dict, root=None) -> str:
    definition = d.get("definition") or []
    called_by = d.get("called_by") or []
    calls = d.get("calls") or []
    tests = d.get("tests") or []
    supers = d.get("superclasses") or []
    subs = d.get("subclasses") or []
    members = d.get("members") or []

    if not (definition or called_by or calls or tests):
        return f"No information found for {symbol!r} in this codebase."

    lines = [f"Symbol context for {symbol!r}:", ""]

    lines.append("DEFINITION:")
    if definition:
        for i, r in enumerate(definition):
            lines.extend(_definition_lines(r, root, with_body=(i == 0)))
    else:
        lines.append("  (not found)")
    more = d.get("other_definitions") or 0
    if more:
        where = ", ".join(d.get("other_definition_files") or [])
        tail = " and others" if more > len(d.get("other_definition_files") or []) else ""
        lines.append(f"  ... and {more} other definition(s) of this name: {where}{tail}. "
                     f"Pass `file` (a path fragment) to describe one of them.")

    if d.get("is_type"):
        lines.append("")
        lines.append(f"INHERITS FROM ({len(supers)}):" if supers else "INHERITS FROM: (none)")
        for e in supers:
            tgt = e.get("target") or {}
            kind = f" ({e['base_kind']})" if e.get("base_kind") else ""
            where = f"  @ {_loc(tgt, root)}" if tgt.get("file") else ""
            lines.append(f"  • {tgt.get('label', '?')}{kind}{where}")
        lines.append("")
        lines.append(f"SUBTYPES ({len(subs)}):" if subs else "SUBTYPES: (none)")
        for e in subs:
            lines.append(f"  • {(e.get('source') or {}).get('label', '?')}  "
                         f"@ {_loc(e.get('source') or {}, root)}")
        lines.append("")
        lines.append(f"MEMBERS ({len(members)}):" if members else "MEMBERS: (none)")
        for m in members:
            span = _line_span(m.get("start_line") or 0, m.get("end_line") or 0)
            sig = m.get("signature") or m.get("label", "?")
            lines.append(f"  • {sig}  {span}".rstrip())

    # called_by / calls are graph edge dicts (source/target node dicts).
    lines.append("")
    lines.append("CALLED BY:" if d.get("graph_enabled") else "CALLED BY: (enable the graph layer for call data)")
    lines.extend(_edge_end_lines(called_by, "source", root))
    if d.get("graph_enabled") and not called_by:
        lines.append("  (none)")

    lines.append("")
    lines.append("CALLS:" if d.get("graph_enabled") else "CALLS: (enable the graph layer for call data)")
    lines.extend(_edge_end_lines(calls, "target", root))
    if d.get("graph_enabled") and not calls:
        lines.append("  (none)")

    lines.append("")
    lines.append("TESTS:")
    if tests:
        for r in tests:
            lines.append(f"  • {r.get('symbol') or '?'}  @ {_loc(r, root)}"
                         + (f"  [d{r['depth']}]" if r.get("depth") else ""))
    else:
        lines.append("  (none found)")

    return "\n".join(lines)


def _format_impact(symbol: str, d: dict, root=None) -> str:
    callers = d.get("callers") or []
    tests = d.get("tests") or []
    if not callers and not tests:
        if not d.get("graph_enabled"):
            return (f"Impact of {symbol!r}: enable the graph layer for transitive "
                    f"callers. No tests found either.")
        return f"Impact of {symbol!r}: no callers and no tests found."

    lines = [f"Impact / blast radius of {symbol!r}:", ""]
    lines.append(
        f"REACHED BY ({len(callers)} transitive callers, [dN] = hops away):" if d.get("graph_enabled")
        else "REACHED BY: (enable the graph layer for call data)"
    )
    for c in callers:
        node = c.get("node") or {}
        lines.append(
            f"  [d{c.get('depth')}] {node.get('label', '?')}  "
            f"@ {_loc(node, root)}{_conf_tag(c.get('confidence'))}"
        )
    if d.get("graph_enabled") and not callers:
        lines.append("  (none — nothing calls it)")

    lines.append("")
    lines.append("TESTS to re-run:")
    if tests:
        for r in tests:
            lines.append(f"  • {r.get('symbol') or '?'}  @ {_loc(r, root)}"
                         + (f"  [d{r['depth']}]" if r.get("depth") else ""))
    else:
        lines.append("  (none found)")
    return "\n".join(lines)


# A visibility keyword repeated on every member says nothing the line needs.
_NOISE_PREFIXES = ("public ",)
# Past this many dependents a summary says how many more; `impact` lists them all.
_MAX_DEPENDENTS = 15


def _summary_line(s: dict, containers: set) -> str:
    """One symbol of a module summary on one line: its span, then the
    signature (the name is in it), then the doc line if any. A member listed
    under its own class drops the `Class.` prefix and is indented; a symbol
    without a signature falls back to `name [kind]`."""
    label = s.get("label", "?")
    parent, _, short = label.rpartition(".")
    nested = bool(parent) and parent in containers
    name = short if nested else label
    sig = s.get("signature") or ""
    for prefix in _NOISE_PREFIXES:
        if sig.startswith(prefix):
            sig = sig[len(prefix):]
    text = sig if sig and short in sig else (f"{name}: {sig}" if sig else f"{name} [{s.get('kind', '?')}]")
    doc = f"  ↳ {s['doc']}" if s.get("doc") else ""
    span = _line_span(s.get("start_line") or 0, s.get("end_line") or 0)
    return f"{'    ' if nested else '  '}{span:<10} {text}{doc}".rstrip()


def _format_module_summary(file: str, d: dict, root=None) -> str:
    """What a file defines, one line per symbol, then its imports and the
    files that depend on it. Kept to one line per symbol on purpose: the
    earlier layout (name and kind, then the signature that repeats the name,
    then `public` on every member) spent about 200 characters per symbol,
    9,700 for one 988-line TypeScript class, the size of reading half of it."""
    symbols = d.get("symbols") or []
    imports = d.get("imports") or []
    deps = d.get("dependent_files") or []
    if not (symbols or imports or deps):
        return f"No file matches {file!r} (or it defines nothing)."

    files = [f for f in (d.get("files") or []) if f != file]
    where = f" ({', '.join(files)})" if files else ""
    lines = [f"Module summary for {file!r}{where}:", ""]
    lines.append(f"DEFINES ({len(symbols)}):")
    containers = {s.get("label") for s in symbols}
    for s in symbols:
        lines.append(_summary_line(s, containers))
    if not symbols:
        lines.append("  (none)")

    lines.append("")
    if imports:
        names = [e.get("module") or (e.get("target") or {}).get("label") or "?" for e in imports]
        lines.append("IMPORTS: " + ", ".join(names))
    else:
        lines.append("IMPORTS: (none)" if d.get("graph_enabled") else
                     "IMPORTS: (see the file header; the graph layer lists resolved imports)")

    if d.get("graph_enabled"):
        shown = [to_rel_path(f, root) for f in deps[:_MAX_DEPENDENTS]]
        more = len(deps) - len(shown)
        listed = ", ".join(shown) + (f" and {more} more" if more > 0 else "")
        lines.append(f"DEPENDED ON BY ({len(deps)} file(s), imports and calls): "
                     f"{listed or '(none found)'}")
    else:
        lines.append("DEPENDED ON BY: (enable the graph layer for call data)")
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


def _format_similar_results(results: list, root=None) -> str:
    if not results:
        return "No similar code found."
    lines = [f"Found {len(results)} similar chunks:"]
    for r in results:
        loc = _loc(r, root)
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
