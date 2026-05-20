"""Unit tests for the graph query API (get_callers / get_callees /
get_imports / get_neighbors / shortest_path)."""
from __future__ import annotations

import sys
import networkx as nx


def _build_graph():
    """Tiny call graph:

      file_main.py (file)         file_util.py (file)
            ├─ contains: go              ├─ contains: helper
            ├─ imports_from: util        └─ contains: other
            │
        go ──calls──> Service.run ──calls──> helper
                                  └─calls──> other
    """
    G = nx.DiGraph()
    # Files
    G.add_node("fmain", kind="file", label="main", file="/x/main.py", lang_key="python")
    G.add_node("futil", kind="file", label="util", file="/x/util.py", lang_key="python")
    # Functions
    G.add_node("go", kind="function", label="go", file="/x/main.py", start_line=1, end_line=3, lang_key="python")
    G.add_node("srv", kind="function", label="Service.run", file="/x/main.py", start_line=5, end_line=8, lang_key="python")
    G.add_node("helper", kind="function", label="helper", file="/x/util.py", start_line=1, end_line=2, lang_key="python")
    G.add_node("other", kind="function", label="other", file="/x/util.py", start_line=4, end_line=5, lang_key="python")
    # Class
    G.add_node("svccls", kind="class", label="Service", file="/x/main.py", start_line=4, end_line=10, lang_key="python")
    # External
    G.add_node("ext_util", kind="external", label="util", file="", lang_key="")

    # Contains
    G.add_edge("fmain", "go", relation="contains")
    G.add_edge("fmain", "srv", relation="contains")
    G.add_edge("svccls", "srv", relation="contains")
    G.add_edge("futil", "helper", relation="contains")
    G.add_edge("futil", "other", relation="contains")
    # Imports
    G.add_edge("fmain", "ext_util", relation="imports_from", module="util")
    # Calls
    G.add_edge("go", "srv", relation="calls", confidence="resolved")
    G.add_edge("srv", "helper", relation="calls", confidence="resolved")
    G.add_edge("srv", "other", relation="calls", confidence="resolved")
    return G


def main() -> int:
    from lynx.graph import (
        get_callers, get_callees, get_imports, get_neighbors,
        shortest_path, find_symbols,
    )

    G = _build_graph()

    # ============================================================
    # 1. find_symbols: exact + fuzzy
    # ============================================================
    if find_symbols(G, "helper") != ["helper"]:
        print(f"[test] FAIL [1/6]: find_symbols('helper') = {find_symbols(G, 'helper')!r}")
        return 1
    # Substring match
    matches = find_symbols(G, "service")
    if "srv" not in matches and "svccls" not in matches:
        print(f"[test] FAIL [1/6]: 'service' should fuzz-match 'Service.run' or 'Service': {matches}")
        return 1
    # Files / external are skipped
    if find_symbols(G, "main") and "fmain" in find_symbols(G, "main"):
        print(f"[test] FAIL [1/6]: file node leaked into find_symbols match")
        return 1
    print(f"[test] OK [1/6] find_symbols: exact + fuzzy + file-skip")

    # ============================================================
    # 2. get_callers
    # ============================================================
    callers = get_callers(G, "helper")
    if len(callers) != 1:
        print(f"[test] FAIL [2/6]: helper should have 1 caller, got {len(callers)}")
        return 2
    if callers[0]["source"]["id"] != "srv" or callers[0]["target"]["id"] != "helper":
        print(f"[test] FAIL [2/6]: caller edge wrong: {callers[0]}")
        return 2
    if callers[0]["relation"] != "calls":
        print(f"[test] FAIL [2/6]: caller relation wrong: {callers[0]}")
        return 2
    # Function with no callers
    if get_callers(G, "go") != []:
        print(f"[test] FAIL [2/6]: 'go' should have no callers")
        return 2
    print(f"[test] OK [2/6] get_callers: helper has 1 caller (Service.run)")

    # ============================================================
    # 3. get_callees
    # ============================================================
    callees = get_callees(G, "Service.run")
    callee_labels = sorted(c["target"]["label"] for c in callees)
    if callee_labels != ["helper", "other"]:
        print(f"[test] FAIL [3/6]: Service.run callees should be [helper, other], got {callee_labels}")
        return 3
    print(f"[test] OK [3/6] get_callees: Service.run → [helper, other]")

    # ============================================================
    # 4. get_imports — by file substring match
    # ============================================================
    imps = get_imports(G, "main.py")
    if len(imps) != 1:
        print(f"[test] FAIL [4/6]: main.py should have 1 import, got {len(imps)}")
        return 4
    if imps[0]["target"]["label"] != "util":
        print(f"[test] FAIL [4/6]: import target wrong: {imps[0]}")
        return 4
    # Also resolvable via the symbol it contains
    imps2 = get_imports(G, "go")
    if len(imps2) != 1:
        print(f"[test] FAIL [4/6]: get_imports('go') (resolve via containing file) failed: {imps2}")
        return 4
    print(f"[test] OK [4/6] get_imports: file-substring AND symbol-via-file both work")

    # ============================================================
    # 5. get_neighbors
    # ============================================================
    # Service.run neighbors at depth=1: go (caller), helper, other, svccls (container)
    nbrs = get_neighbors(G, "Service.run", depth=1)
    nbr_pairs = {(e["source"]["id"], e["target"]["id"]) for e in nbrs}
    expected_pairs = {
        ("srv", "helper"), ("srv", "other"),  # callees
        ("go", "srv"),                         # incoming call
        ("svccls", "srv"),                     # contained-in
        ("fmain", "srv"),                      # contained-in (also via file)
    }
    if not expected_pairs.issubset(nbr_pairs):
        print(f"[test] FAIL [5/6]: missing expected neighbor pairs. expected ⊆: "
              f"{expected_pairs - nbr_pairs}, got: {nbr_pairs}")
        return 5
    # With relation filter, should only return "calls" edges
    nbrs_calls = get_neighbors(G, "Service.run", relation_filter="calls", depth=1)
    if not all(e["relation"] == "calls" for e in nbrs_calls):
        print(f"[test] FAIL [5/6]: relation_filter ignored: {nbrs_calls}")
        return 5
    print(f"[test] OK [5/6] get_neighbors: depth=1 returns all neighbors, relation_filter works")

    # ============================================================
    # 6. shortest_path
    # ============================================================
    path = shortest_path(G, "go", "helper")
    if path is None:
        print(f"[test] FAIL [6/6]: shortest_path go->helper returned None")
        return 6
    if path["hops"] != 2:
        print(f"[test] FAIL [6/6]: expected 2 hops (go->srv->helper), got {path}")
        return 6
    node_ids = [n["id"] for n in path["nodes"]]
    if node_ids != ["go", "srv", "helper"]:
        print(f"[test] FAIL [6/6]: path nodes wrong: {node_ids}")
        return 6
    # No path: other → go (reverse direction, no edge)
    if shortest_path(G, "other", "go") is not None:
        print(f"[test] FAIL [6/6]: shortest_path other->go should be None (no directed path)")
        return 6
    # Unknown symbol
    if shortest_path(G, "nonexistent", "helper") is not None:
        print(f"[test] FAIL [6/6]: shortest_path with unknown source must be None")
        return 6
    print(f"[test] OK [6/6] shortest_path: go->helper in 2 hops, unreachable/unknown handled")

    print("\n[test] === SUCCESS: graph query API works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
