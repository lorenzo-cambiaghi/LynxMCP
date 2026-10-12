"""The watcher indexes what a build would index, and keeps up with safe saves.

Real watchdog on a temporary tree; the index is a recorder. Editors and Unity
often save by writing a temporary file and renaming it over the real one, and
the watcher must end on an update of the real name, never on a removal.
"""
from __future__ import annotations

import os
import time
import types
from pathlib import Path

import pytest

from lynx.sources.codebase import CodebaseBackend


def _watch(tmp_path: Path):
    code = tmp_path / "code"
    code.mkdir()
    calls = []
    backend = CodebaseBackend.__new__(CodebaseBackend)
    backend.name = "src"
    backend.is_owner = True
    backend._observer = None
    backend._handler = None
    backend.storage_dir = tmp_path / "store"
    backend.source_config = {
        "path": code,
        "supported_extensions": frozenset({".py"}),
        "ignored_path_fragments": ("/build/",),
        "ignored_globs": ("20??-??-??-*.py",),
        "watcher": {"enabled": True, "debounce_seconds": 0.2},
    }
    backend.rag = types.SimpleNamespace(
        update_file=lambda p: calls.append(("update", os.path.normcase(p))),
        remove_file=lambda p: calls.append(("remove", os.path.normcase(p))))
    backend.graph = None
    backend.start_watcher()
    return backend, code, calls


def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _last_for(calls, path):
    key = os.path.normcase(str(path))
    mine = [action for action, p in calls if p == key]
    return mine[-1] if mine else None


@pytest.fixture
def watched(tmp_path):
    backend, code, calls = _watch(tmp_path)
    yield code, calls
    backend.stop_watcher()


def test_a_file_written_as_temporary_and_renamed_is_indexed(watched):
    code, calls = watched
    (code / "tool.py.tmp").write_text("def tool(): pass\n", encoding="utf-8")
    os.replace(code / "tool.py.tmp", code / "tool.py")

    assert _until(lambda: _last_for(calls, code / "tool.py") == "update")


def test_a_safe_save_over_an_existing_file_ends_as_an_update(watched):
    code, calls = watched
    target = code / "a.py"
    target.write_text("def a(): return 1\n", encoding="utf-8")
    assert _until(lambda: _last_for(calls, target) == "update")

    os.remove(target)                                   # delete, then rename in
    (code / "a.py.tmp").write_text("def a(): return 2\n", encoding="utf-8")
    os.replace(code / "a.py.tmp", target)
    time.sleep(1.0)                                     # past the debounce

    assert _last_for(calls, target) == "update"


def test_files_a_build_skips_are_skipped_by_the_watcher_too(watched):
    code, calls = watched
    for rel in (".claude/settings.py", "build/out.py", "2026-06-11-chat.py", "notes.txt"):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text("x = 1\n", encoding="utf-8")
    (code / "real.py").write_text("y = 2\n", encoding="utf-8")   # written last

    assert _until(lambda: _last_for(calls, code / "real.py") == "update")
    time.sleep(0.5)
    touched = {Path(p).name for _action, p in calls}
    assert touched == {"real.py"}, touched
