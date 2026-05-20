"""Unit tests for the graph analyzer (god_nodes / communities /
surprising_connections). Builds a hand-crafted nx.DiGraph so the expected
output is deterministic without involving the extractor pipeline.

Scenarios:
  1. god_nodes ranks by total degree, excludes kind=file/external
  2. communities returns groups >= min_size, ordered by size
  3. surprising_connections returns bridge edges with betweenness
  4. empty graph returns []
"""
from __future__ import annotations

import sys
import networkx as nx


def _build_graph():
    """Two tightly-connected clusters joined by a single bridge edge.

    Cluster A: a1..a4 — fully connected (each calls each)
    Cluster B: b1..b4 — fully connected
    Bridge:    a1 → b1 (the surprising connection)
    Plus: a "file" node and an "external" node (both must be filtered
          out by god_nodes/communities/surprising).
    """
    G = nx.DiGraph()
    for n in ("a1", "a2", "a3", "a4"):
        G.add_node(n, kind="function", label=n, lang_key="python", file="A.py")
    for n in ("b1", "b2", "b3", "b4"):
        G.add_node(n, kind="function", label=n, lang_key="python", file="B.py")
    G.add_node("fileA", kind="file", label="A.py", lang_key="python", file="A.py")
    G.add_node("extstd", kind="external", label="stdlib", lang_key="")

    # Cluster A: every-pair call edges
    for u in ("a1", "a2", "a3", "a4"):
        for v in ("a1", "a2", "a3", "a4"):
            if u != v:
                G.add_edge(u, v, relation="calls", confidence="extracted")
    # Cluster B: same
    for u in ("b1", "b2", "b3", "b4"):
        for v in ("b1", "b2", "b3", "b4"):
            if u != v:
                G.add_edge(u, v, relation="calls", confidence="extracted")
    # Bridge: a1 → b1
    G.add_edge("a1", "b1", relation="calls", confidence="resolved")
    # File hub: fileA contains all A nodes (this would boost fileA's
    # degree massively if not filtered out)
    for n in ("a1", "a2", "a3", "a4"):
        G.add_edge("fileA", n, relation="contains")
    # External import
    G.add_edge("fileA", "extstd", relation="imports", module="os")
    return G


def main() -> int:
    from lynx.graph import god_nodes, communities, surprising_connections

    # ============================================================
    # 1. god_nodes
    # ============================================================
    G = _build_graph()
    gods = god_nodes(G, top_n=5)
    if not gods:
        print("[test] FAIL [1/4]: god_nodes returned []")
        return 1
    ids = [g["id"] for g in gods]
    # No file/external must appear
    if "fileA" in ids or "extstd" in ids:
        print(f"[test] FAIL [1/4]: file/external leaked into god_nodes: {ids}")
        return 1
    # a1 has the highest degree (cluster A + bridge edge)
    if gods[0]["id"] != "a1":
        print(f"[test] FAIL [1/4]: expected 'a1' as top god node, got {gods[0]}")
        return 1
    # Degrees must be present and consistent
    if gods[0]["degree"] != gods[0]["in_degree"] + gods[0]["out_degree"]:
        print(f"[test] FAIL [1/4]: degree != in+out for {gods[0]}")
        return 1
    print(f"[test] OK [1/4] god_nodes: top is {gods[0]['id']} with degree {gods[0]['degree']}")

    # ============================================================
    # 2. communities
    # ============================================================
    comms = communities(G, min_size=3)
    if len(comms) < 2:
        print(f"[test] FAIL [2/4]: expected >=2 communities of size >=3, got {comms}")
        return 2
    # Each community must have only function nodes (no file/external)
    all_members_lower = set()
    for c in comms:
        for label in c["members_sample"]:
            all_members_lower.add(label.lower())
    if "stdlib" in all_members_lower or "a.py" in all_members_lower:
        print(f"[test] FAIL [2/4]: file/external present in communities: {comms}")
        return 2
    # ID ordering: id=0 must be the largest
    if comms[0]["size"] < comms[-1]["size"]:
        print(f"[test] FAIL [2/4]: communities not sorted by size desc: {comms}")
        return 2
    print(f"[test] OK [2/4] communities: {len(comms)} found, largest size={comms[0]['size']}")

    # ============================================================
    # 3. surprising_connections
    # ============================================================
    surprises = surprising_connections(G, top_n=3)
    if not surprises:
        print("[test] FAIL [3/4]: surprising_connections returned [] on a bridged graph")
        return 3
    # The a1↔b1 bridge MUST be in the top results — it's the only edge
    # between the two clusters, so its betweenness should be very high.
    found_bridge = any(
        (s["source"] == "a1" and s["target"] == "b1") or
        (s["source"] == "b1" and s["target"] == "a1")
        for s in surprises
    )
    if not found_bridge:
        print(f"[test] FAIL [3/4]: a1↔b1 bridge missing from top surprises: {surprises}")
        return 3
    if surprises[0]["betweenness"] <= 0:
        print(f"[test] FAIL [3/4]: top betweenness must be > 0: {surprises[0]}")
        return 3
    print(f"[test] OK [3/4] surprising_connections: bridge {surprises[0]['source']}↔"
          f"{surprises[0]['target']} betweenness={surprises[0]['betweenness']}")

    # ============================================================
    # 4. Empty graph
    # ============================================================
    empty = nx.DiGraph()
    from lynx.graph import god_nodes as gn, communities as comm, surprising_connections as sc
    if gn(empty) != [] or comm(empty) != [] or sc(empty) != []:
        print("[test] FAIL [4/4]: analyzers must return [] on empty graph")
        return 4
    print(f"[test] OK [4/4] empty graph: all analyzers return []")

    print("\n[test] === SUCCESS: graph analyzer works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
