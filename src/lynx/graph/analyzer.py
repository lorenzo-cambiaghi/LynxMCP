"""Whole-graph analytics: god nodes, communities, surprising connections.

All three functions are pure (G in → list out) and use only NetworkX
built-ins — no Leiden, no graspologic, no scipy. The trade-off vs
graphify's analyzer:
  - god_nodes: same intent (degree-based hubs) but skips the LLM-derived
    `_is_concept_node` check (Lynx never injects concept nodes).
  - communities: greedy_modularity_communities instead of Leiden. Slightly
    lower quality on large graphs but built-in and deterministic.
  - surprising_connections: edge_betweenness_centrality with an automatic
    cap + sampling for graphs over ~5k nodes. Same goal as graphify
    (highlight bridge edges between distant communities) but without
    scipy as a hard dep.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


# Hard cap on nodes for edge_betweenness_centrality (cubic-ish complexity).
# Beyond this size we use the sampling variant `k=`.
_SURPRISE_FULL_BETWEENNESS_CAP = 1500
_SURPRISE_SAMPLE_K = 200


# ---------------------------------------------------------------------------
# God nodes
# ---------------------------------------------------------------------------


def god_nodes(G: nx.DiGraph, top_n: int = 10) -> list:
    """Top-N most-connected real entities — the architectural hubs.

    Excludes `kind="file"` and `kind="external"` nodes. File hubs
    accumulate import/contains edges mechanically; external stubs are
    placeholders for stdlib / third-party modules — neither is a real
    architectural abstraction.

    Returns list of dicts ordered by total degree (in + out) descending.
    """
    candidates = []
    for nid, data in G.nodes(data=True):
        kind = data.get("kind")
        if kind in ("file", "external"):
            continue
        deg = G.in_degree(nid) + G.out_degree(nid)
        candidates.append((deg, nid, data))
    candidates.sort(key=lambda t: (-t[0], t[1]))  # tie-break alpha for stability
    out = []
    for deg, nid, data in candidates[:top_n]:
        out.append({
            "id": nid,
            "label": data.get("label", nid),
            "kind": data.get("kind"),
            "subkind": data.get("subkind"),
            "file": data.get("file"),
            "degree": deg,
            "in_degree": G.in_degree(nid),
            "out_degree": G.out_degree(nid),
        })
    return out


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------


def communities(G: nx.DiGraph, min_size: int = 3) -> list:
    """Detect communities via greedy modularity on the undirected view.

    `min_size` filters out tiny clusters (single-method classes, isolated
    file pairs) that add noise without insight.

    Returns list of {id, size, members_sample, lang_breakdown} ordered by
    size desc. `members_sample` is a small subset of labels for the
    human-readable summary; full membership stays in the graph attributes.
    """
    if G.number_of_nodes() == 0:
        return []
    H = G.to_undirected()
    # Drop pure file/external nodes from community detection so the
    # partition reflects code structure rather than directory shape.
    keep = [n for n, d in H.nodes(data=True)
            if d.get("kind") not in ("file", "external")]
    sub = H.subgraph(keep)
    if sub.number_of_nodes() == 0:
        return []
    try:
        raw = greedy_modularity_communities(sub)
    except Exception:
        # Disconnected graphs or single-node components occasionally trip
        # the algorithm; degrade to one community per connected component.
        raw = list(nx.connected_components(sub))

    out = []
    for i, members in enumerate(raw):
        if len(members) < min_size:
            continue
        labels = []
        lang_count: dict = {}
        for nid in members:
            data = sub.nodes[nid]
            labels.append(data.get("label", nid))
            lang = data.get("lang_key") or "?"
            lang_count[lang] = lang_count.get(lang, 0) + 1
        # Heuristic name: most-connected node in the community.
        ranked = sorted(members, key=lambda n: H.degree(n), reverse=True)
        out.append({
            "id": i,
            "size": len(members),
            "name": sub.nodes[ranked[0]].get("label", ranked[0]),
            "members_sample": sorted(labels)[:10],
            "by_language": lang_count,
        })
    out.sort(key=lambda c: -c["size"])
    # Renumber by sorted order so `id=0` is always the largest community.
    for i, c in enumerate(out):
        c["id"] = i
    return out


# ---------------------------------------------------------------------------
# Surprising connections
# ---------------------------------------------------------------------------


def surprising_connections(G: nx.DiGraph, top_n: int = 5) -> list:
    """Edges that bridge distant communities — the non-obvious couplings.

    Uses edge_betweenness_centrality on the undirected view. On graphs
    larger than `_SURPRISE_FULL_BETWEENNESS_CAP` nodes, we use the
    sampling variant (`k=200`) so latency stays sub-second even on big
    codebases at the cost of some ranking precision.

    Returns list of {source, target, source_label, target_label,
    betweenness, relation, ...} ordered by betweenness desc.
    """
    if G.number_of_edges() == 0:
        return []
    H = G.to_undirected()
    # Drop file/external nodes — bridges via file hubs are uninteresting.
    keep = [n for n, d in H.nodes(data=True)
            if d.get("kind") not in ("file", "external")]
    sub = H.subgraph(keep)
    if sub.number_of_edges() == 0:
        return []

    if sub.number_of_nodes() > _SURPRISE_FULL_BETWEENNESS_CAP:
        try:
            scores = nx.edge_betweenness_centrality(
                sub, k=min(_SURPRISE_SAMPLE_K, sub.number_of_nodes()),
                normalized=True, seed=42,
            )
        except Exception:
            return []
    else:
        try:
            scores = nx.edge_betweenness_centrality(sub, normalized=True)
        except Exception:
            return []

    # Pair each edge with its (community-bridging?) bonus. We don't have
    # per-node community labels here, so just rank by betweenness — the
    # consumer can filter further.
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out = []
    for (u, v), score in ranked[:top_n]:
        # Pull the edge attributes from the (directed) graph; prefer the
        # directed orientation that exists.
        if G.has_edge(u, v):
            data = G.get_edge_data(u, v) or {}
            src, tgt = u, v
        elif G.has_edge(v, u):
            data = G.get_edge_data(v, u) or {}
            src, tgt = v, u
        else:
            data = {}
            src, tgt = u, v
        out.append({
            "source": src,
            "target": tgt,
            "source_label": G.nodes[src].get("label", src),
            "target_label": G.nodes[tgt].get("label", tgt),
            "relation": data.get("relation", "?"),
            "betweenness": round(float(score), 6),
        })
    return out
