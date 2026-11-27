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


def test_events_the_system_lost_bring_a_pass_over_every_file(tmp_path, monkeypatch):
    # A train landing hundreds of files at once overflows Windows'
    # ReadDirectoryChangesW buffer (inotify's queue on Linux): watchdog says
    # which watch lost events, not which files. A pass over every file's hash,
    # as at startup, finds them; one pass for a burst of losses.
    from watchdog.events import EventsLostEvent
    import lynx.sources.codebase as codebase

    monkeypatch.setattr(codebase, "LOST_EVENTS_DEBOUNCE_SECONDS", 0.2)
    backend, code, _calls = _watch(tmp_path)
    passes = []
    backend.catch_up = lambda: passes.append(time.monotonic())
    try:
        for _ in range(3):
            backend._handler.dispatch(EventsLostEvent(str(code), is_directory=True))
        assert _until(lambda: passes), "lost events left the index behind"
        time.sleep(0.5)
        assert len(passes) == 1, "a burst of losses is one pass"
    finally:
        backend.stop_watcher()


def test_stopping_the_watcher_cancels_a_pass_still_waiting(tmp_path, monkeypatch):
    from watchdog.events import EventsLostEvent
    import lynx.sources.codebase as codebase

    monkeypatch.setattr(codebase, "LOST_EVENTS_DEBOUNCE_SECONDS", 0.3)
    backend, code, _calls = _watch(tmp_path)
    passes = []
    backend.catch_up = lambda: passes.append(1)
    backend._handler.dispatch(EventsLostEvent(str(code), is_directory=True))
    backend.stop_watcher()
    time.sleep(0.6)
    assert passes == [], "a pass ran on a store this process let go of"


def test_files_changed_during_a_build_go_in_when_it_is_over(tmp_path, monkeypatch):
    # A build deletes a changed file's chunks and puts the new ones in later,
    # with a group: a watcher write in between left the file in twice. While
    # a build is on, the watcher's files wait, and go in after it.
    import lynx.sources.codebase as codebase

    monkeypatch.setattr(codebase, "BUILD_WAIT_SECONDS", 0.2)
    backend, code, calls = _watch(tmp_path)
    try:
        backend.rag.building = True
        (code / "late.py").write_text("x = 1\n", encoding="utf-8")
        handler = backend._handler
        late = os.path.normcase(str(code / "late.py"))
        assert _until(lambda: any(os.path.normcase(p) == late for p in list(handler._deferred))), \
            "not put off while a build was on"
        time.sleep(0.5)                                 # the build still on: it waits
        assert _last_for(calls, code / "late.py") is None, "written while a build was on"
        backend.rag.building = False
        assert _until(lambda: _last_for(calls, code / "late.py") == "update")
    finally:
        backend.stop_watcher()


def test_stopping_the_watcher_drops_the_files_waiting_for_a_build(tmp_path, monkeypatch):
    import lynx.sources.codebase as codebase

    monkeypatch.setattr(codebase, "BUILD_WAIT_SECONDS", 0.2)
    backend, code, calls = _watch(tmp_path)
    backend.rag.building = True
    (code / "late.py").write_text("x = 1\n", encoding="utf-8")
    handler = backend._handler
    assert _until(lambda: handler._deferred), "not put off while a build was on"
    backend.stop_watcher()
    backend.rag.building = False
    time.sleep(0.6)
    assert calls == [], "a file went in on a store this process let go of"
