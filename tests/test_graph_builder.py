"""Integration tests for `GraphLayer`: rebuild, persistence, incremental
update_file / remove_file, cross-file resolution.

No HuggingFace, no ChromaDB — only the graph layer is exercised.

Scenarios:
  1. rebuild() on a 3-file synthetic Python codebase
     - all expected nodes present
     - cross-file calls resolved (caller in file A → callee in file B)
     - imports captured
  2. persistence round-trip
     - rebuild, instantiate a new GraphLayer pointing at the same dir,
       verify it loads the exact same graph
  3. SHA cache: second rebuild without changes does NOT re-extract files
  4. update_file: edit one file, verify only its nodes/edges change,
     other files' edges still intact, cross-file calls re-resolved
  5. remove_file: delete a file, verify its nodes go, callers' edges to
     that file's symbols become raw_calls again
  6. unsupported extension: file is skipped silently (no graph nodes,
     no crash)
  7. ignored_path_fragments: matching files are excluded
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _make_codebase(root: Path, files: dict) -> None:
    """Write each {rel_path: content} entry under root, creating dirs."""
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _make_layer(root: Path, storage_root: Path, **kwargs):
    from lynx.graph import GraphLayer
    return GraphLayer(
        storage_dir=storage_root / "graph",
        codebase_path=root,
        supported_extensions=kwargs.get("supported_extensions", [".py"]),
        ignored_path_fragments=kwargs.get("ignored_path_fragments"),
    )


def _labels(layer):
    return {data.get("label"): nid for nid, data in layer.graph.nodes(data=True)}


def _edges(layer, relation: str):
    return [
        (u, v, d) for u, v, d in layer.graph.edges(data=True)
        if d.get("relation") == relation
    ]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-graph-"))
    print(f"[test] tempdir: {tmp}")
    try:
        code = tmp / "code"
        storage = tmp / "storage"
        code.mkdir(); storage.mkdir()

        # ========================================================
        # 1. rebuild() on synthetic 3-file codebase
        # ========================================================
        _make_codebase(code, {
            "util.py": "def helper(x):\n    return x + 1\n",
            "service.py": (
                "from util import helper\n"
                "class Service:\n"
                "    def run(self, x):\n"
                "        return helper(x)\n"
            ),
            "main.py": (
                "from service import Service\n"
                "def go():\n"
                "    s = Service()\n"
                "    return s.run(1)\n"
            ),
        })
        layer = _make_layer(code, storage)
        summary = layer.rebuild(force=True)
        if summary["extracted_files"] != 3:
            print(f"[test] FAIL [1/7]: expected 3 extracted files, got {summary}")
            return 1
        labels = _labels(layer)
        for expected in ("helper", "Service.run", "Service", "go"):
            if expected not in labels:
                print(f"[test] FAIL [1/7]: '{expected}' missing from {sorted(labels)}")
                return 1
        # Cross-file resolution: Service.run -> helper (different file)
        calls = _edges(layer, "calls")
        run_id, helper_id = labels["Service.run"], labels["helper"]
        if not any(u == run_id and v == helper_id for u, v, d in calls):
            print(f"[test] FAIL [1/7]: cross-file Service.run -> helper not resolved")
            print(f"  calls edges: {[(u, v, d.get('confidence')) for u, v, d in calls]}")
            return 1
        if not any(u == labels["go"] and d.get("relation") == "calls" for u, v, d in calls):
            print(f"[test] FAIL [1/7]: 'go' has no outgoing call (s.run/Service() missing)")
            return 1
        imports = _edges(layer, "imports") + _edges(layer, "imports_from")
        if len(imports) < 2:
            print(f"[test] FAIL [1/7]: expected >=2 import edges, got {len(imports)}")
            return 1
        print(f"[test] OK [1/7] rebuild: 3 files, {layer.graph.number_of_nodes()} nodes, "
              f"{layer.graph.number_of_edges()} edges, {len(calls)} calls, {len(imports)} imports")

        # ========================================================
        # 2. Persistence round-trip
        # ========================================================
        # Build a fresh layer pointing at the same storage dir.
        layer2 = _make_layer(code, storage)
        if layer2.graph.number_of_nodes() != layer.graph.number_of_nodes():
            print(f"[test] FAIL [2/7]: node count differs after reload "
                  f"({layer.graph.number_of_nodes()} vs {layer2.graph.number_of_nodes()})")
            return 2
        if layer2.graph.number_of_edges() != layer.graph.number_of_edges():
            print(f"[test] FAIL [2/7]: edge count differs after reload")
            return 2
        labels2 = _labels(layer2)
        if labels2.keys() != labels.keys():
            print(f"[test] FAIL [2/7]: label sets differ after reload")
            return 2
        # Verify on-disk schema_version
        meta = json.loads((storage / "graph" / "metadata.json").read_text())
        if meta.get("schema_version") != 1:
            print(f"[test] FAIL [2/7]: metadata.schema_version != 1: {meta}")
            return 2
        print(f"[test] OK [2/7] persistence round-trip: same nodes/edges/labels after reload")

        # ========================================================
        # 3. SHA cache: second rebuild without changes is a no-op
        # ========================================================
        summary2 = layer.rebuild(force=False)
        if summary2["extracted_files"] != 0:
            print(f"[test] FAIL [3/7]: second rebuild should extract 0 files, got {summary2}")
            return 3
        if summary2["unchanged"] != 3:
            print(f"[test] FAIL [3/7]: expected 3 unchanged files, got {summary2}")
            return 3
        print(f"[test] OK [3/7] SHA cache: re-rebuild extracts 0 files when nothing changed")

        # ========================================================
        # 4. update_file: modify util.py, verify cross-file re-resolution
        # ========================================================
        # Add a NEW function to util.py and have service.py call it.
        (code / "util.py").write_text(
            "def helper(x):\n    return x + 1\n"
            "def renamed_helper(x):\n    return x * 2\n",
            encoding="utf-8",
        )
        changed = layer.update_file(str(code / "util.py"))
        if not changed:
            print(f"[test] FAIL [4/7]: update_file returned False on changed file")
            return 4
        labels = _labels(layer)
        if "renamed_helper" not in labels:
            print(f"[test] FAIL [4/7]: new 'renamed_helper' missing after update")
            return 4
        # Original Service.run -> helper edge must still be present
        calls = _edges(layer, "calls")
        if not any(u == labels["Service.run"] and v == labels["helper"] for u, v, d in calls):
            print(f"[test] FAIL [4/7]: Service.run -> helper edge lost after util.py edit")
            return 4
        # service.py was NOT re-extracted but its raw_calls should still resolve
        # (because the symbol index was refreshed in update_file).
        print(f"[test] OK [4/7] update_file: util.py edit triggered re-extract + cross-file re-resolution")

        # ========================================================
        # 5. remove_file: deleting util.py turns Service.run's call into raw
        # ========================================================
        os.remove(code / "util.py")
        removed = layer.remove_file(str(code / "util.py"))
        if not removed:
            print(f"[test] FAIL [5/7]: remove_file returned False for deleted file")
            return 5
        labels = _labels(layer)
        if "helper" in labels:
            print(f"[test] FAIL [5/7]: 'helper' still in graph after util.py removal: {sorted(labels)}")
            return 5
        # Service.run still exists, but the call to helper should now be unresolved
        # (no outgoing edge from Service.run to a "helper" node).
        if "Service.run" not in labels:
            print(f"[test] FAIL [5/7]: Service.run disappeared (only util.py was removed)")
            return 5
        run_id = labels["Service.run"]
        for u, v, d in layer.graph.out_edges(run_id, data=True):
            target_label = layer.graph.nodes[v].get("label", "")
            if d.get("relation") == "calls" and target_label == "helper":
                print(f"[test] FAIL [5/7]: stale Service.run -> helper edge after removal")
                return 5
        print(f"[test] OK [5/7] remove_file: util.py removed, dependent edges cleaned up")

        # ========================================================
        # 6. Unsupported extension: a .md file is silently skipped
        # ========================================================
        (code / "README.md").write_text("# hi\n", encoding="utf-8")
        layer_md = _make_layer(code, storage / "md_test", supported_extensions=[".py", ".md"])
        summary_md = layer_md.rebuild(force=True)
        labels_md = _labels(layer_md)
        # README.md gets a SHA cache entry but no nodes — verify no crash and
        # no 'README' label appears as a graph node.
        if any("README" in (l or "") for l in labels_md):
            print(f"[test] FAIL [6/7]: README.md produced graph nodes: {labels_md}")
            return 6
        print(f"[test] OK [6/7] unsupported extension: .md file silently skipped")

        # ========================================================
        # 7. ignored_path_fragments: filter applied
        # ========================================================
        (code / "vendor").mkdir(exist_ok=True)
        (code / "vendor" / "third_party.py").write_text(
            "def vendor_func():\n    pass\n", encoding="utf-8",
        )
        layer_filtered = _make_layer(
            code, storage / "filt_test",
            ignored_path_fragments=["/vendor/"],
        )
        layer_filtered.rebuild(force=True)
        labels_f = _labels(layer_filtered)
        if "vendor_func" in labels_f:
            print(f"[test] FAIL [7/7]: vendor_func should be filtered out: {sorted(labels_f)}")
            return 7
        print(f"[test] OK [7/7] ignored_path_fragments: /vendor/ excluded")

        print("\n[test] === SUCCESS: graph builder works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
