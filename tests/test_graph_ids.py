"""Graph node IDs are unique per file.

They used to be built from the parent directory's name and the file's name,
which two files in different places can share: every Unity package has a
Runtime/AssemblyInfo.cs, and a monorepo can hold two tests/test_utils.py. A
shared ID merged the two files, or two same-named functions in them, into one
node, with one set of callers and one blast radius for two different things.

Pure graph layer, no model, no ChromaDB.
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _write(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _layer(code: Path, storage: Path) -> GraphLayer:
    return GraphLayer(storage_dir=storage, codebase_path=code,
                      supported_extensions=[".py"])


def _twins(code: Path):
    _write(code / "pkg_a" / "tests" / "test_utils.py",
           "def test_parse():\n    return 1\n")
    _write(code / "pkg_b" / "tests" / "test_utils.py",
           "def test_parse():\n    return 2\n")


def test_same_named_files_in_same_named_directories_stay_apart(tmp_path: Path):
    code = tmp_path / "code"
    _twins(code)

    layer = _layer(code, tmp_path / "graph")

    nodes = list(layer._nodes_by_id.values())
    files = {n["file"] for n in nodes if n["kind"] == "file"}
    parse = {n["file"] for n in nodes
             if n["kind"] == "function" and n["label"] == "test_parse"}
    assert len(files) == 2
    assert len(parse) == 2
    # The labels, which is what users read, stay as they were.
    assert {n["label"] for n in nodes if n["kind"] == "file"} == {"tests.test_utils"}


def test_editing_one_twin_gives_the_graph_of_a_full_rebuild(tmp_path: Path):
    # With a shared ID, which twin a node described depended on the order the
    # files were processed in, so a delta rebuild and a full one disagreed.
    code = tmp_path / "code"
    _twins(code)
    layer = _layer(code, tmp_path / "graph")

    _write(code / "pkg_a" / "tests" / "test_utils.py",
           "def test_parse():\n    return 1\n\n\ndef test_more():\n    return 3\n")
    layer.rebuild(force=False)
    full = _layer(code, tmp_path / "graph_full")

    assert layer._nodes_by_id == full._nodes_by_id
    assert (sorted(layer.graph.edges(data="relation"))
            == sorted(full.graph.edges(data="relation")))
