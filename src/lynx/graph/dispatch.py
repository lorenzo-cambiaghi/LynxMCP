"""Named graph operations — shared by the MCP `graph_query` tool and the
`lynx graph query` CLI.

This used to live inline in `server.py`'s tool closure, which meant the
CLI had no way to reach it without importing FastMCP and registering a
tool. Pulled out here it is a plain function over a `SourceManager`, so
both front-ends produce byte-identical output: a formatting fix lands in
the agent's answer and in the terminal at the same time.

Two entry points over the same dispatch:

  - `query_graph()` returns a `GraphQueryResult` — the rendered text, the
    structured payload behind it, and an `ok` flag. The CLI uses it for
    `--json` and for its exit code.
  - `run_graph_query()` returns just the text. The MCP tool hands its
    return value straight to the model, which has no use for anything
    else.

`source` must already be resolved to a graph-enabled source name — the
two callers disagree on what to do when it's ambiguous (the tool returns
an error string, the CLI exits non-zero), so that decision stays with
them.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

from .._format import _format_edge_lines, _format_node_brief


# Operations that are meaningless without a symbol to act on.
SYMBOL_OPS = frozenset({
    "callers", "callees", "subclasses", "superclasses",
    "imports", "neighbors", "shortest_path",
})

# Every operation `query_graph` accepts, in the order they're listed to
# the user. 'architectural_overview' is an accepted alias of 'overview'
# (the pre-consolidation tool name) and is deliberately absent here.
OPERATIONS = (
    "callers", "callees", "subclasses", "superclasses", "imports",
    "neighbors", "shortest_path", "overview", "surprising_connections",
    "status",
)


class GraphQueryResult(NamedTuple):
    """What a graph operation produced.

    `ok` is False only for a *usage* problem (unknown operation, missing
    argument) — an operation that ran and found nothing is a legitimate
    answer, so it stays True. The CLI maps `ok` to its exit code, which is
    why this is a field and not a prefix on `text`: deciding "did this
    fail" by looking for "Error:" at the start of a string written for a
    language model is a contract nobody can safely refactor.

    `matched` is True/False when the query mentioned a symbol and we could
    check whether the graph knows it, None when it doesn't apply or the
    graph wasn't reachable to ask.
    """
    ok: bool
    operation: str
    text: str
    data: Any
    matched: Optional[bool] = None


def _seed_exists(manager, source: str, symbol: str) -> Optional[bool]:
    """Does the graph know anything called `symbol`?

    Only consulted when an operation came back empty, to tell "this symbol
    has no callers" apart from "this symbol isn't in the graph" — by far
    the most common reason for a confusing empty answer.

    Deliberately broader than any single operation's own seed resolution
    (symbol nodes AND file nodes): the operations disagree on what they
    accept — `imports` takes a file path, the rest take symbols — and
    re-implementing each rule here would drift. Erring towards True means
    the worst case is staying quiet, never wrongly telling someone their
    symbol doesn't exist.

    Returns None when the graph can't be reached, in which case the caller
    says nothing rather than guessing.
    """
    try:
        from .query import find_symbols
        graph = manager.get(source).graph.graph
        if find_symbols(graph, symbol):
            return True
        needle = (symbol or "").lower()
        if not needle:
            return False
        for _nid, data in graph.nodes(data=True):
            if data.get("kind") != "file":
                continue
            if (needle in (data.get("file") or "").lower()
                    or needle in (data.get("label") or "").lower()):
                return True
        return False
    except Exception:
        # A stubbed manager in tests, a backend without the attribute, a
        # graph mid-rebuild: none of those justify failing a query that
        # already produced its answer.
        return None


def _envelope(payload: dict, op: str, source: str) -> dict:
    """Add the `operation`/`source` keys every other payload carries.

    `status` and `overview` return a whole dict of their own, so the keys
    are merged in rather than nested: a script can read `.operation`
    uniformly across operations AND still reach `.nodes` without an extra
    level of indirection. None of the manager's own keys collide.
    """
    return {"operation": op, "source": source, **payload}


def _no_seed_hint(symbol: str) -> str:
    return (
        f"\n  (nothing in this graph is called {symbol!r} — check the spelling, "
        f"or run the 'status' operation to confirm the graph covers this source)"
    )


def _edges_result(manager, source, op, symbol, edges, header) -> GraphQueryResult:
    """Render an edge list, and explain an empty one."""
    text = _format_edge_lines(edges, header)
    if edges:
        # Results exist, so the seed obviously resolved — say so rather than
        # leaving a null a JSON consumer has to special-case.
        matched = True
    else:
        matched = _seed_exists(manager, source, symbol)
        if matched is False:
            text += _no_seed_hint(symbol)
    return GraphQueryResult(
        True, op, text,
        {"operation": op, "source": source, "symbol": symbol,
         "matched": matched, "count": len(edges), "edges": edges},
        matched,
    )


def query_graph(
    manager,
    source: str,
    operation: str,
    *,
    symbol: str | None = None,
    target: str | None = None,
    relation_filter: str | None = None,
    depth: int = 1,
    limit: int = 50,
    max_hops: int = 8,
    top_n: int = 10,
    min_community_size: int = 3,
) -> GraphQueryResult:
    """Run `operation` against `source`'s graph layer.

    Argument problems come back as `ok=False` results rather than
    exceptions, because the MCP tool hands `.text` straight to the model.
    """
    op = (operation or "").strip().lower()

    def _error(msg: str) -> GraphQueryResult:
        # Errors carry operation/source like every success payload does, so
        # a JSON consumer reads the same keys whichever way the call went.
        return GraphQueryResult(
            False, op, f"Error: {msg}",
            {"operation": op, "source": source, "error": msg},
        )

    if op in SYMBOL_OPS and not symbol:
        return _error(f"operation {op!r} requires `symbol`.")

    if op in ("callers", "callees", "subclasses", "superclasses", "imports"):
        # Method NAMES, resolved after indexing: a table of bound methods
        # would touch all five attributes to serve one operation, so a
        # manager exposing a subset (a test fake, a future backend) would
        # die on an AttributeError for a method the call never needed.
        method, label = {
            "callers": ("get_callers", "Callers of {s!r} in {src!r}:"),
            "callees": ("get_callees", "Callees of {s!r} in {src!r}:"),
            "subclasses": ("get_subclasses", "Subclasses of {s!r} in {src!r}:"),
            "superclasses": ("get_superclasses", "Superclasses of {s!r} in {src!r}:"),
            "imports": ("get_imports", "Imports from {s!r} in {src!r}:"),
        }[op]
        edges = getattr(manager, method)(source, symbol, limit=limit)
        return _edges_result(manager, source, op, symbol, edges,
                             label.format(s=symbol, src=source))

    if op == "neighbors":
        edges = manager.get_neighbors(
            source, symbol,
            relation_filter=relation_filter, depth=depth, limit=limit,
        )
        label = f"Neighbors of {symbol!r}"
        if relation_filter:
            label += f" (relation={relation_filter!r})"
        label += f" depth={depth} in {source!r}:"
        return _edges_result(manager, source, op, symbol, edges, label)

    if op == "shortest_path":
        if not target:
            return _error("operation 'shortest_path' requires `target`.")
        path = manager.shortest_path(source, symbol, target, max_hops=max_hops)
        payload = {"operation": op, "source": source, "symbol": symbol,
                   "target": target, "path": path}
        if path is None:
            text = (f"No directed path from {symbol!r} to {target!r} "
                    f"(within {max_hops} hops).")
            matched = _seed_exists(manager, source, symbol)
            if matched is False:
                text += _no_seed_hint(symbol)
            elif _seed_exists(manager, source, target) is False:
                text += _no_seed_hint(target)
                matched = False
            payload["matched"] = matched
            return GraphQueryResult(True, op, text, payload, matched)
        lines = [f"Path from {symbol!r} → {target!r} ({path['hops']} hops):"]
        for n in path["nodes"]:
            lines.append(f"  • {_format_node_brief(n)}")
        payload["matched"] = True
        return GraphQueryResult(True, op, "\n".join(lines), payload, True)

    if op in ("overview", "architectural_overview"):
        ov = manager.architectural_overview(
            source, top_n_gods=top_n, min_community_size=min_community_size,
        )
        lines = [f"=== Architectural overview of {source!r} ==="]
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
        return GraphQueryResult(True, op, "\n".join(lines),
                                _envelope(ov, op, source))

    if op == "surprising_connections":
        surprises = manager.surprising_connections(source, top_n=top_n)
        payload = {"operation": op, "source": source,
                   "count": len(surprises), "connections": surprises}
        if not surprises:
            return GraphQueryResult(
                True, op, f"No surprising connections detected in {source!r}.",
                payload,
            )
        lines = [f"Top {len(surprises)} bridge edges in {source!r} (by betweenness):"]
        for s in surprises:
            lines.append(
                f"  • {s['source_label']!r} --{s['relation']}--> {s['target_label']!r}  "
                f"betweenness={s['betweenness']}"
            )
        return GraphQueryResult(True, op, "\n".join(lines), payload)

    if op == "status":
        st = manager.graph_status(source)
        lines = [f"=== Graph status for {source!r} ==="]
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
        return GraphQueryResult(True, op, "\n".join(lines),
                                _envelope(st, op, source))

    return _error(
        f"unknown operation {operation!r}. Valid operations: "
        f"{', '.join(OPERATIONS)}."
    )


def run_graph_query(manager, source: str, operation: str, **kwargs) -> str:
    """Text-only façade over `query_graph`, for the MCP tool."""
    return query_graph(manager, source, operation, **kwargs).text
