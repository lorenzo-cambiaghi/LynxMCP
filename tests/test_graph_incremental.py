"""Incremental graph resolution must equal a full rebuild.

`GraphLayer.update_file` / `remove_file` re-resolve cross-file calls and
inheritance only for the symbol names touched by the change (instead of
re-resolving the whole graph on every save). These tests pin the correctness
invariant: after ANY sequence of incremental edits, the live graph is
byte-for-byte equivalent (same nodes, same derived edges with same confidence)
to building the graph from scratch over the same files.

Pure graph layer — no HuggingFace, no ChromaDB. Fast.
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _layer(code: Path, storage: Path) -> GraphLayer:
    return GraphLayer(
        storage_dir=storage / "graph",
        codebase_path=code,
        supported_extensions=[".py"],
    )


def _edge_sigs(layer: GraphLayer) -> set:
    """Derived-edge signature, ignoring provenance fields (from_file/line) that
    can legitimately differ by resolution order without changing structure."""
    return {
        (u, v, d.get("relation"), d.get("confidence"), d.get("base_kind"))
        for u, v, d in layer.graph.edges(data=True)
    }


def _node_sigs(layer: GraphLayer) -> set:
    return {
        (nid, d.get("kind"), d.get("label"))
        for nid, d in layer.graph.nodes(data=True)
    }


def _assert_matches_full_rebuild(incremental: GraphLayer, code: Path, tmp: Path, tag: str):
    """Build a fresh layer over the SAME files and compare to the incremental one."""
    gt_storage = tmp / f"gt_{tag}"
    gt = _layer(code, gt_storage)
    gt.rebuild(force=True)
    inc_nodes, gt_nodes = _node_sigs(incremental), _node_sigs(gt)
    assert inc_nodes == gt_nodes, (
        f"[{tag}] node mismatch\n only-incremental: {inc_nodes - gt_nodes}\n "
        f"only-full: {gt_nodes - inc_nodes}"
    )
    inc_edges, gt_edges = _edge_sigs(incremental), _edge_sigs(gt)
    assert inc_edges == gt_edges, (
        f"[{tag}] edge mismatch\n only-incremental: {inc_edges - gt_edges}\n "
        f"only-full: {gt_edges - inc_edges}"
    )


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_incremental_equals_full_rebuild_sequence(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    storage = tmp_path / "storage"

    # --- initial state -----------------------------------------------------
    _write(code / "util.py", "def helper(x):\n    return x + 1\n")
    _write(code / "service.py",
           "from util import helper\n"
           "class Service:\n"
           "    def run(self, x):\n"
           "        return helper(x)\n")
    layer = _layer(code, storage)          # bootstraps a full build on init
    layer.rebuild(force=True)
    _assert_matches_full_rebuild(layer, code, tmp_path, "initial")

    # --- 1. add a brand-new file that calls an existing symbol -------------
    _write(code / "main.py",
           "from service import Service\n"
           "def go():\n"
           "    s = Service()\n"
           "    return s.run(1)\n")
    layer.update_file(str(code / "main.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "add_main")

    # --- 2. resolved -> 0: a call's target disappears ---------------------
    # main.py calls helper indirectly; here we make service call a not-yet
    # existing symbol, confirm it's unresolved, then create it.
    _write(code / "service.py",
           "from util import helper\n"
           "class Service:\n"
           "    def run(self, x):\n"
           "        return helper(x) + extra(x)\n")   # extra() undefined → raw
    layer.update_file(str(code / "service.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "unresolved_extra")

    # --- 3. 0 -> resolved: define the missing symbol ----------------------
    _write(code / "extra.py", "def extra(x):\n    return x * 10\n")
    layer.update_file(str(code / "extra.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "define_extra")

    # --- 4. resolved -> 0 again: remove the definer -----------------------
    (code / "extra.py").unlink()
    layer.remove_file(str(code / "extra.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "remove_extra")

    # --- 5. remove a heavily-referenced file ------------------------------
    (code / "util.py").unlink()
    layer.remove_file(str(code / "util.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "remove_util")


def test_incremental_ambiguous_to_resolved(tmp_path: Path):
    """Two files define `foo`; a third calls it (ambiguous, 2 edges). Removing
    one definer must collapse it to a single `resolved` edge — and the
    incremental result must equal a full rebuild."""
    code = tmp_path / "code"
    code.mkdir()
    storage = tmp_path / "storage"

    _write(code / "a.py", "def foo(x):\n    return x\n")
    _write(code / "b.py", "def foo(x):\n    return -x\n")
    _write(code / "caller.py",
           "def use(x):\n"
           "    return foo(x)\n")
    layer = _layer(code, storage)
    layer.rebuild(force=True)
    _assert_matches_full_rebuild(layer, code, tmp_path, "ambiguous_initial")

    # Sanity: the call to foo should be ambiguous (2 candidate edges).
    foo_call_edges = [
        d for _u, _v, d in layer.graph.edges(data=True)
        if d.get("relation") == "calls" and d.get("confidence") in ("ambiguous", "resolved")
    ]
    assert any(d.get("confidence") == "ambiguous" for d in foo_call_edges), \
        f"expected an ambiguous call edge, got {[d.get('confidence') for d in foo_call_edges]}"

    # Remove one definer → should become a single resolved edge.
    (code / "b.py").unlink()
    layer.remove_file(str(code / "b.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "ambiguous_to_resolved")

    confidences = [
        d.get("confidence") for _u, _v, d in layer.graph.edges(data=True)
        if d.get("relation") == "calls"
    ]
    assert "ambiguous" not in confidences, \
        f"call edge should be resolved after removing the duplicate, got {confidences}"


def test_incremental_inheritance_reresolution(tmp_path: Path):
    """Cross-file inheritance edges re-resolve incrementally like calls do."""
    code = tmp_path / "code"
    code.mkdir()
    storage = tmp_path / "storage"

    _write(code / "base.py", "class Base:\n    def run(self):\n        pass\n")
    _write(code / "concrete.py",
           "from base import Base\n"
           "class Derived(Base):\n    pass\n")
    layer = _layer(code, storage)
    layer.rebuild(force=True)
    _assert_matches_full_rebuild(layer, code, tmp_path, "inh_initial")

    # Add a second subclass via an edit to concrete.py.
    _write(code / "concrete.py",
           "from base import Base\n"
           "class Derived(Base):\n    pass\n"
           "class Derived2(Base):\n    pass\n")
    layer.update_file(str(code / "concrete.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "inh_add_subclass")

    # Remove the base class → inherits edges must disappear.
    (code / "base.py").unlink()
    layer.remove_file(str(code / "base.py"))
    _assert_matches_full_rebuild(layer, code, tmp_path, "inh_remove_base")
