"""Query API: structural navigation of the graph.

Each function takes a `nx.DiGraph` plus a symbol string and returns a
list/path of result dicts ready to be JSON-serialized for an MCP tool.

Symbol matching is fuzzy: a query `"helper"` matches any node whose
`label` contains "helper" (case-insensitive). Exact matches are tried
first (case-insensitive equality), then substring matches. Returns ALL
matching seeds — the consumer can narrow the scope after seeing what's
there.

Output dicts are intentionally rich (file/lines/confidence) so the AI
client can cite the source precisely without a follow-up tool call.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx


_MAX_NEIGHBORS_PER_SEED = 100


def _node_to_dict(G: nx.DiGraph, nid: str) -> dict:
    d = G.nodes[nid]
    return {
        "id": nid,
        "label": d.get("label", nid),
        "kind": d.get("kind"),
        "file": d.get("file"),
        "start_line": d.get("start_line"),
        "end_line": d.get("end_line"),
        "lang_key": d.get("lang_key"),
    }


def _edge_to_dict(G: nx.DiGraph, u: str, v: str, data: dict) -> dict:
    src = _node_to_dict(G, u)
    tgt = _node_to_dict(G, v)
    out = {
        "source": src,
        "target": tgt,
        "relation": data.get("relation"),
        "confidence": data.get("confidence"),
        "module": data.get("module"),
        "from_file": data.get("from_file"),
        "from_line": data.get("from_line"),
    }
    # `inherits` edges carry an extra hint when the language exposed it.
    if "base_kind" in data:
        out["base_kind"] = data.get("base_kind")
    return out


def find_symbols(G: nx.DiGraph, symbol: str) -> list:
    """Return all node IDs whose label matches `symbol`.

    Match strategy: exact (case-insensitive) first; if nothing matches,
    fall back to case-insensitive substring. Skips file/external nodes
    so a query for "helper" doesn't match a file named "helper.py".
    """
    if not symbol:
        return []
    needle = symbol.lower()
    exact = []
    fuzzy = []
    for nid, data in G.nodes(data=True):
        if data.get("kind") in ("file", "external"):
            continue
        label = (data.get("label") or "").lower()
        if not label:
            continue
        if label == needle:
            exact.append(nid)
            continue
        # Match either the leaf segment or the full qualified name.
        leaf = label.split(".")[-1]
        if leaf == needle:
            exact.append(nid)
        elif needle in label:
            fuzzy.append(nid)
    return exact if exact else fuzzy


# ---------------------------------------------------------------------------
# Callers / callees
# ---------------------------------------------------------------------------


def get_callers(G: nx.DiGraph, symbol: str, limit: int = 50) -> list:
    """Functions that call `symbol`. Returns list of edge dicts.

    A "calls" edge with target = a node matching `symbol` is a caller
    edge — we walk the in-edges of every seed and filter by relation.
    """
    seeds = find_symbols(G, symbol)
    if not seeds:
        return []
    out = []
    for nid in seeds:
        for src, _tgt, data in G.in_edges(nid, data=True):
            if data.get("relation") != "calls":
                continue
            out.append(_edge_to_dict(G, src, nid, data))
            if len(out) >= limit:
                return out
    return out


def get_callees(G: nx.DiGraph, symbol: str, limit: int = 50) -> list:
    """Functions called BY `symbol`. Mirror of get_callers using out-edges."""
    seeds = find_symbols(G, symbol)
    if not seeds:
        return []
    out = []
    for nid in seeds:
        for _src, tgt, data in G.out_edges(nid, data=True):
            if data.get("relation") != "calls":
                continue
            out.append(_edge_to_dict(G, nid, tgt, data))
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Inheritance (subclasses / superclasses)
# ---------------------------------------------------------------------------
#
# Same edge_to_dict shape as callers/callees. The extra `base_kind` attribute
# on the inheritance edge tells whether the language semantically saw an
# `extends` vs `implements` (Java, TS) or just an undifferentiated base list
# (C#, C++).


def get_subclasses(G: nx.DiGraph, symbol: str, limit: int = 50) -> list:
    """Classes / interfaces / structs that inherit FROM `symbol`.

    Returns the in-edges of every node matching `symbol` whose relation is
    "inherits". Answers "what concrete types extend or implement X?".
    Useful when X is an abstract base class or an interface — the typical
    discovery question structural search cannot reliably answer.
    """
    seeds = find_symbols(G, symbol)
    if not seeds:
        return []
    out = []
    for nid in seeds:
        for src, _tgt, data in G.in_edges(nid, data=True):
            if data.get("relation") != "inherits":
                continue
            out.append(_edge_to_dict(G, src, nid, data))
            if len(out) >= limit:
                return out
    return out


def get_superclasses(G: nx.DiGraph, symbol: str, limit: int = 50) -> list:
    """Types that `symbol` inherits from. Out-edge mirror of get_subclasses.

    For languages that distinguish them (Java, TypeScript), each edge's
    `base_kind` attribute is `"extends"` for the concrete superclass and
    `"implements"` for interfaces.
    """
    seeds = find_symbols(G, symbol)
    if not seeds:
        return []
    out = []
    for nid in seeds:
        for _src, tgt, data in G.out_edges(nid, data=True):
            if data.get("relation") != "inherits":
                continue
            out.append(_edge_to_dict(G, nid, tgt, data))
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def get_imports(G: nx.DiGraph, file_or_symbol: str, limit: int = 100) -> list:
    """Import edges originating from a file (or from the file that
    contains the given symbol).

    Resolves the seed with `find_symbols` first; if no match, treats the
    string as a file path substring and looks for a `kind="file"` node.
    Returns edges with relation in {"imports", "imports_from"}.
    """
    seeds: list = []
    # Try symbol-based: walk up from the symbol's node to its file.
    sym_seeds = find_symbols(G, file_or_symbol)
    for nid in sym_seeds:
        file_path = G.nodes[nid].get("file")
        if not file_path:
            continue
        # Find the file node by matching its `file` attribute.
        for fid, fdata in G.nodes(data=True):
            if fdata.get("kind") == "file" and fdata.get("file") == file_path:
                if fid not in seeds:
                    seeds.append(fid)
                break

    if not seeds:
        # File-based: substring match on `file` attribute.
        needle = file_or_symbol.lower()
        for nid, data in G.nodes(data=True):
            if data.get("kind") != "file":
                continue
            file_path = (data.get("file") or "").lower()
            label = (data.get("label") or "").lower()
            if needle in file_path or needle in label:
                seeds.append(nid)

    out = []
    for nid in seeds:
        for _src, tgt, data in G.out_edges(nid, data=True):
            if data.get("relation") not in ("imports", "imports_from"):
                continue
            out.append(_edge_to_dict(G, nid, tgt, data))
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Neighbors / shortest path
# ---------------------------------------------------------------------------


def get_neighbors(
    G: nx.DiGraph,
    symbol: str,
    relation_filter: Optional[str] = None,
    depth: int = 1,
    limit: int = 100,
) -> list:
    """All graph neighbors of `symbol` within `depth` hops, optionally
    filtered by edge relation (e.g. "calls", "contains", "imports").
    BFS on the undirected view so both predecessors and successors are
    walked uniformly.

    Returns list of edge dicts at every level of the BFS, deduplicated
    by (source_id, target_id).
    """
    seeds = find_symbols(G, symbol)
    if not seeds:
        return []
    depth = max(1, min(depth, 6))  # cap pathological depth
    UG = G.to_undirected(as_view=True)
    visited = set(seeds)
    frontier = list(seeds)
    out = []
    seen_pairs = set()
    for _ in range(depth):
        next_frontier = []
        for nid in frontier:
            for neigh in UG.neighbors(nid):
                # Use directed edge data if it exists in the original DiGraph.
                if G.has_edge(nid, neigh):
                    data = G.get_edge_data(nid, neigh) or {}
                    src, tgt = nid, neigh
                else:
                    data = G.get_edge_data(neigh, nid) or {}
                    src, tgt = neigh, nid
                if relation_filter and data.get("relation") != relation_filter:
                    pass  # don't emit but still traverse
                else:
                    pair = (src, tgt)
                    if pair not in seen_pairs:
                        out.append(_edge_to_dict(G, src, tgt, data))
                        seen_pairs.add(pair)
                        if len(out) >= limit:
                            return out
                if neigh not in visited:
                    visited.add(neigh)
                    next_frontier.append(neigh)
        frontier = next_frontier
        if not frontier:
            break
    return out


def shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    max_hops: int = 8,
) -> "Optional[dict]":
    """Shortest directed path from a node matching `source` to one
    matching `target`. Returns None if no path within `max_hops`.

    Both source and target are resolved with `find_symbols` (multi-match
    is allowed: we try every pairwise combination and return the
    shortest path found).
    """
    if not source or not target:
        return None
    srcs = find_symbols(G, source)
    tgts = find_symbols(G, target)
    if not srcs or not tgts:
        return None
    best: Optional[list] = None
    for s in srcs:
        for t in tgts:
            if s == t:
                continue
            try:
                path = nx.shortest_path(G, source=s, target=t)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) - 1 > max_hops:
                continue
            if best is None or len(path) < len(best):
                best = path
    if best is None:
        return None
    edges = []
    for i in range(len(best) - 1):
        u, v = best[i], best[i + 1]
        data = G.get_edge_data(u, v) or {}
        edges.append(_edge_to_dict(G, u, v, data))
    return {
        "hops": len(best) - 1,
        "nodes": [_node_to_dict(G, n) for n in best],
        "edges": edges,
    }
