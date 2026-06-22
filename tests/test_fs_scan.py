"""Unit tests for the shared codebase file walk (`lynx.fs_scan`).

Pure filesystem logic — no embedding model, no index — so it runs fast and
without network. Locks in the fix for the bug where `ignored_path_fragments`
was honored by the watcher and the graph layer but NOT by the vector-index
build, so vendored dirs (node_modules, dist) leaked into search.
"""

from __future__ import annotations

import os
from pathlib import Path

from lynx import fs_scan


def _make_tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "pkg" / "util.py").write_text("y = 2\n", encoding="utf-8")
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "vendor.py").write_text("z = 3\n", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("w = 4\n", encoding="utf-8")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.py").write_text("s = 5\n", encoding="utf-8")
    (root / "README.md").write_text("# doc\n", encoding="utf-8")
    (root / ".dotfile.py").write_text("d = 6\n", encoding="utf-8")


def test_extension_and_hidden_filtering(tmp_path: Path):
    _make_tree(tmp_path)
    files = fs_scan.list_candidate_files(tmp_path, [".py"])
    names = {os.path.basename(f) for f in files}
    # .md excluded by extension; dot dir/file excluded as hidden.
    assert "README.md" not in names
    assert "secret.py" not in names      # under .hidden/
    assert ".dotfile.py" not in names
    assert {"app.py", "util.py"} <= names


def test_ignored_fragments_excluded(tmp_path: Path):
    _make_tree(tmp_path)
    files = fs_scan.list_candidate_files(
        tmp_path, [".py"], ["node_modules", "build"]
    )
    names = {os.path.basename(f) for f in files}
    assert "vendor.py" not in names      # under node_modules/
    assert "out.py" not in names         # under build/
    assert names == {"app.py", "util.py"}


def test_ignored_fragments_separator_agnostic(tmp_path: Path):
    """A multi-segment fragment matches regardless of OS separator."""
    _make_tree(tmp_path)
    (tmp_path / "src" / "generated").mkdir(parents=True)
    (tmp_path / "src" / "generated" / "g.py").write_text("g = 7\n", encoding="utf-8")
    # Fragment given with a forward slash even on Windows.
    files = fs_scan.list_candidate_files(tmp_path, [".py"], ["src/generated"])
    names = {os.path.basename(f) for f in files}
    assert "g.py" not in names


def test_results_absolute_and_sorted(tmp_path: Path):
    _make_tree(tmp_path)
    files = fs_scan.list_candidate_files(tmp_path, [".py"])
    assert all(os.path.isabs(f) for f in files)
    assert files == sorted(files)


def test_is_ignored_matches_watcher_use(tmp_path: Path):
    p = str(tmp_path / "node_modules" / "dep" / "vendor.py")
    assert fs_scan.is_ignored(p, ["node_modules"])
    assert not fs_scan.is_ignored(p, ["dist"])
    assert not fs_scan.is_ignored(p, [])
