"""Several Lynx sessions on one index: one owner writes, the rest follow.

ChromaDB lets more than one process read a store at the same time (measured
against a live `lynx serve` on 1.5.9: the second process opens, counts and runs
a KNN query in about a second). What it does not allow is two writers, so
exactly one process claims the store and runs the watcher.

The catch these tests pin down: a follower answers from state loaded when it
opened. `count()` stays live across processes, but the HNSW segment and the
BM25 corpus do not — which is how a second session could report the right
number of chunks and still never find the function you just wrote.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time

import pytest

from lynx import ownership
from lynx.errors import StoreNotOwnedError
from lynx.rag_manager import CodebaseRAG


# ---------------------------------------------------------------------------
# Claiming a store
# ---------------------------------------------------------------------------

def test_first_claim_wins_and_second_one_follows(tmp_path):
    assert ownership.claim(tmp_path) is True
    assert ownership.owned_by_this_process(tmp_path) is True
    # A live claim is not stolen, not even by the process that holds it.
    assert ownership.claim(tmp_path) is False
    assert ownership.owner_of(tmp_path)["pid"] == os.getpid()


def test_a_claim_from_a_dead_process_is_stolen(tmp_path, monkeypatch):
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": 4242, "host": ownership._host(), "since": "2026-01-01T00:00:00",
    }), encoding="utf-8")
    monkeypatch.setattr(ownership, "pid_alive", lambda pid: pid == os.getpid())
    assert ownership.owner_of(tmp_path) is None      # nobody is really there
    assert ownership.claim(tmp_path) is True
    assert ownership.owner_of(tmp_path)["pid"] == os.getpid()


def test_a_claim_from_another_machine_is_left_alone(tmp_path):
    # Shared storage: we cannot check that pid, so the heartbeat is the only
    # evidence, and guessing wrong would put two writers on one store.
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": os.getpid(), "host": "some-other-box", "beat": time.time(),
        "since": "2026-01-01T00:00:00",
    }), encoding="utf-8")
    assert ownership.claim(tmp_path) is False
    assert ownership.owner_of(tmp_path)["host"] == "some-other-box"
    assert "some-other-box" in ownership.describe(tmp_path)


def test_a_claim_nobody_refreshes_goes_up_for_grabs(tmp_path):
    # The case a pid check alone gets wrong: a pid recycled onto an unrelated
    # process, or an owner suspended with the machine. Both look alive.
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": os.getpid(), "host": ownership._host(),
        "beat": time.time() - ownership._STALE_AFTER_SEC - 1, "since": "x",
    }), encoding="utf-8")
    assert ownership.owner_of(tmp_path) is None
    assert ownership.claim(tmp_path) is True


def test_a_claim_from_the_future_is_not_treated_as_stale(tmp_path):
    # Clocks move; a claim stamped ahead of us must not read as infinitely old.
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": os.getpid() + 1, "host": ownership._host(),
        "beat": time.time() + 3600, "since": "x",
    }), encoding="utf-8")
    monkey = ownership.pid_alive
    ownership.pid_alive = lambda pid: True
    try:
        assert ownership.owner_of(tmp_path) is not None
    finally:
        ownership.pid_alive = monkey


def test_the_owner_keeps_its_claim_fresh(tmp_path, monkeypatch):
    # Without a heartbeat every claim would look abandoned after a few minutes
    # and a second session would steal an index that is being written.
    monkeypatch.setattr(ownership, "_HEARTBEAT_SEC", 0.05)
    assert ownership.claim(tmp_path) is True
    first = ownership.read_claim(tmp_path)["beat"]
    deadline = time.time() + 3
    while time.time() < deadline:
        if ownership.read_claim(tmp_path)["beat"] > first:
            break
        time.sleep(0.05)
    assert ownership.read_claim(tmp_path)["beat"] > first
    ownership.release(tmp_path)


def test_an_owner_that_lost_its_claim_is_told_to_stand_down(tmp_path, monkeypatch):
    # A machine asleep long enough loses the index to another session. The
    # sleeper must stop writing rather than become a second writer.
    monkeypatch.setattr(ownership, "_HEARTBEAT_SEC", 0.05)
    told = threading.Event()
    assert ownership.claim(tmp_path, on_lost=told.set) is True
    # Someone else takes it over while we were not looking.
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": os.getpid() + 1, "host": ownership._host(),
        "beat": time.time(), "since": "x",
    }), encoding="utf-8")
    assert told.wait(5) is True


def test_release_only_drops_our_own_claim(tmp_path):
    foreign = {"pid": os.getpid() + 1, "host": ownership._host(),
               "beat": time.time(), "since": "x"}
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps(foreign), encoding="utf-8")
    ownership.release(tmp_path)
    assert (tmp_path / ownership.OWNER_FILE).exists()

    (tmp_path / ownership.OWNER_FILE).unlink()
    assert ownership.claim(tmp_path) is True
    ownership.release(tmp_path)
    assert not (tmp_path / ownership.OWNER_FILE).exists()


def test_a_missing_claim_means_nobody_owns_it(tmp_path):
    assert ownership.owner_of(tmp_path) is None
    assert "no other Lynx process" in ownership.describe(tmp_path)


def test_the_claim_is_how_non_windows_knows_a_store_is_busy(tmp_path, monkeypatch):
    # Windows can ask the OS who holds the file; nothing else can. The claim is
    # what makes the "don't wipe a store someone is using" guards work on macOS
    # and Linux, where they used to be no-ops.
    from lynx import integrity

    assert integrity._claim_based_usage(tmp_path) == "unknown"

    ownership.claim(tmp_path)
    assert integrity._claim_based_usage(tmp_path) == "self"

    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": os.getpid() + 1, "host": ownership._host(),
        "beat": time.time(), "since": "x"}),
        encoding="utf-8")
    monkeypatch.setattr(ownership, "pid_alive", lambda pid: True)
    assert integrity._claim_based_usage(tmp_path) == "other"


def test_pid_liveness():
    import subprocess

    assert ownership.pid_alive(os.getpid()) is True
    # A process we started and reaped: dead for certain, unlike a pid picked
    # out of the air, which a busy machine could legitimately be using.
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    assert ownership.pid_alive(done.pid) is False
    # 0 means "every process in the group" to os.kill; it must never be probed.
    assert ownership.pid_alive(0) is False


# ---------------------------------------------------------------------------
# Noticing the owner's writes
# ---------------------------------------------------------------------------

def _store(tmp_path, rows=(), low=1):
    """A chroma.sqlite3 with just the write log the follower reads."""
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("CREATE TABLE embeddings_queue (seq_id INTEGER PRIMARY KEY, "
               "created_at TIMESTAMP, operation INTEGER, topic TEXT, id TEXT, "
               "vector BLOB, encoding TEXT, metadata TEXT)")
    for i, path in enumerate(rows):
        db.execute("INSERT INTO embeddings_queue (seq_id, operation, topic, id, metadata)"
                   " VALUES (?, 0, 't', ?, ?)",
                   (low + i, f"chunk-{i}", json.dumps({"file_path": path})))
    db.commit()
    db.close()
    return tmp_path


def _rag(tmp_path, *, follower=True):
    """A CodebaseRAG with only the state the follower path touches.

    Built with __new__ on purpose: the real constructor loads an embedding
    model and opens ChromaDB, and none of that is what is under test here.
    """
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag.storage_path = tmp_path
    rag.collection_name = "src"
    rag.follower = follower
    rag._watermark = None
    rag._last_staleness_check = 0.0
    rag._write_lock = threading.Lock()
    rag._bm25_docs = {}
    rag.calls = []
    rag._reopen_store = lambda: rag.calls.append("reopen")
    rag._invalidate_bm25 = lambda: rag.calls.append("bm25_full")
    rag._bm25_reload_file = lambda p: rag.calls.append(f"bm25_file:{p}")
    return rag


def test_watermark_is_the_write_log_head(tmp_path):
    _store(tmp_path, [r"C:\a.py", r"C:\b.py"])
    assert _rag(tmp_path)._store_watermark() == 2


def test_watermark_is_none_without_a_store(tmp_path):
    assert _rag(tmp_path)._store_watermark() is None


def test_changed_files_are_the_rows_above_our_watermark(tmp_path):
    _store(tmp_path, [r"C:\a.py", r"C:\b.py", r"C:\c.py"])
    files, exact = _rag(tmp_path)._files_written_since(1)
    assert exact is True
    assert files == [r"C:\b.py", r"C:\c.py"]


def test_a_purged_log_is_not_a_trustworthy_delta(tmp_path):
    # Chroma purges the log and heal_wal rewrites it. A partial list would
    # leave stale chunks in the BM25 cache, so the caller must reload it all.
    _store(tmp_path, [r"C:\z.py"], low=500)
    files, exact = _rag(tmp_path)._files_written_since(10)
    assert (files, exact) == ([], False)


def test_owner_never_refreshes(tmp_path):
    _store(tmp_path, [r"C:\a.py"])
    rag = _rag(tmp_path, follower=False)
    assert rag.refresh_if_stale() is False
    assert rag.calls == []


def test_follower_reloads_only_the_changed_files(tmp_path):
    _store(tmp_path, [r"C:\a.py", r"C:\b.py"])
    rag = _rag(tmp_path)
    rag._watermark = 1
    rag._bm25_docs = {"chunk-0": ["x"]}      # warm cache: patchable
    assert rag.refresh_if_stale() is True
    assert rag.calls == ["reopen", r"bm25_file:C:\b.py"]
    assert rag._watermark == 2


def test_follower_reloads_everything_when_the_delta_is_unknown(tmp_path):
    _store(tmp_path, [r"C:\z.py"], low=500)
    rag = _rag(tmp_path)
    rag._watermark = 10
    rag._bm25_docs = {"chunk-0": ["x"]}
    assert rag.refresh_if_stale() is True
    assert rag.calls == ["reopen", "bm25_full"]


def test_follower_does_nothing_when_nothing_was_written(tmp_path):
    _store(tmp_path, [r"C:\a.py"])
    rag = _rag(tmp_path)
    rag._watermark = 1
    assert rag.refresh_if_stale() is False
    assert rag.calls == []


def test_the_staleness_check_is_throttled(tmp_path):
    _store(tmp_path, [r"C:\a.py", r"C:\b.py"])
    rag = _rag(tmp_path)
    rag._watermark = 1
    assert rag.refresh_if_stale() is True
    rag._watermark = 1                        # pretend it went stale again
    assert rag.refresh_if_stale() is False    # too soon to look
    rag._last_staleness_check = time.monotonic() - rag._STALENESS_INTERVAL_SEC - 1
    assert rag.refresh_if_stale() is True


def test_a_follower_refuses_to_write(tmp_path):
    rag = _rag(tmp_path)
    with pytest.raises(StoreNotOwnedError) as excinfo:
        rag.update(force=True)
    assert "src" in str(excinfo.value)
    with pytest.raises(StoreNotOwnedError):
        rag.reset_index()
    # The per-file paths a stray watcher timer would reach do not raise, they
    # simply decline: they run on a background thread nobody is watching.
    assert rag.update_file(str(tmp_path / "x.py")) is False
    assert rag.remove_file(str(tmp_path / "x.py")) is False


# ---------------------------------------------------------------------------
# The graph follows too
# ---------------------------------------------------------------------------

def test_graph_follower_reloads_when_the_owner_rewrites_it(tmp_path):
    from lynx.graph import GraphLayer

    code = tmp_path / "code"
    code.mkdir()
    graph_dir = tmp_path / "graph"
    layer = GraphLayer(storage_dir=graph_dir, codebase_path=code,
                       supported_extensions=[".py"], follower=True)
    assert layer.refresh_if_stale() is False          # nothing on disk yet

    loads = []
    layer._load_from_disk = lambda: loads.append(1)
    (graph_dir / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "last_update": "now"}), encoding="utf-8")
    layer._last_staleness_check = 0.0
    assert layer.refresh_if_stale() is True
    assert loads == [1]

    # Unchanged on disk: no reload, however often it is asked.
    layer._last_staleness_check = 0.0
    assert layer.refresh_if_stale() is False
    assert loads == [1]


def test_graph_owner_never_reloads(tmp_path):
    from lynx.graph import GraphLayer

    code = tmp_path / "code"
    code.mkdir()
    graph_dir = tmp_path / "graph"
    layer = GraphLayer(storage_dir=graph_dir, codebase_path=code,
                       supported_extensions=[".py"], follower=False)
    layer._load_from_disk = lambda: pytest.fail("an owner must not reload")
    (graph_dir / "metadata.json").write_text("{}", encoding="utf-8")
    layer._last_staleness_check = 0.0
    assert layer.refresh_if_stale() is False


def test_graph_follower_refuses_to_rebuild(tmp_path):
    from lynx.graph import GraphLayer

    code = tmp_path / "code"
    code.mkdir()
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code,
                       supported_extensions=[".py"], follower=True)
    with pytest.raises(StoreNotOwnedError):
        layer.rebuild(force=True)
    assert layer.update_file(str(code / "x.py")) is False
    assert layer.remove_file(str(code / "x.py")) is False


# ---------------------------------------------------------------------------
# The watcher belongs to the owner
# ---------------------------------------------------------------------------

def _backend(tmp_path, *, is_owner=False):
    from lynx.sources.codebase import CodebaseBackend
    import types

    backend = CodebaseBackend.__new__(CodebaseBackend)
    backend.name = "src"
    backend.is_owner = is_owner
    backend._observer = None
    backend._last_promotion_check = 0.0
    backend.storage_dir = tmp_path
    backend.source_config = {"watcher": {"enabled": True}, "path": str(tmp_path)}
    backend.graph = None
    backend.rag = types.SimpleNamespace(follower=not is_owner,
                                        _last_staleness_check=99.0,
                                        refreshed=0)
    backend.rag.refresh_if_stale = lambda: setattr(
        backend.rag, "refreshed", backend.rag.refreshed + 1)
    backend.started = []
    backend.start_watcher = lambda: backend.started.append(1)
    return backend


def test_a_follower_starts_no_watcher(tmp_path, capsys):
    from lynx.sources.codebase import CodebaseBackend

    backend = _backend(tmp_path)
    del backend.start_watcher                      # exercise the real one
    CodebaseBackend.start_watcher(backend)
    assert backend._observer is None
    assert "another Lynx process owns this index" in capsys.readouterr().err


def test_a_follower_takes_over_when_the_owner_exits(tmp_path, capsys):
    backend = _backend(tmp_path)
    (tmp_path / ownership.OWNER_FILE).write_text(json.dumps({
        "pid": 999999, "host": ownership._host(), "since": "x"}), encoding="utf-8")

    assert backend._claim_if_abandoned() is True
    assert backend.is_owner is True
    assert backend.rag.follower is False
    # It must catch up on the departed owner's writes BEFORE it stops looking.
    assert backend.rag.refreshed == 1
    assert backend.started == [1]
    assert ownership.owned_by_this_process(tmp_path)
    assert "taking over indexing here" in capsys.readouterr().err


def test_a_follower_leaves_a_live_owner_alone(tmp_path):
    backend = _backend(tmp_path)
    ownership.claim(tmp_path)                      # a live claim (this process)
    assert backend._claim_if_abandoned() is False
    assert backend.is_owner is False
    assert backend.started == []


def test_the_takeover_check_is_throttled(tmp_path):
    backend = _backend(tmp_path)
    ownership.claim(tmp_path)
    assert backend._claim_if_abandoned() is False
    calls = []
    backend_owner_of = ownership.owner_of
    try:
        ownership.owner_of = lambda p: calls.append(1) or backend_owner_of(p)
        assert backend._claim_if_abandoned() is False
        assert calls == []                          # too soon to look again
    finally:
        ownership.owner_of = backend_owner_of


def test_an_owner_never_tries_to_take_over(tmp_path):
    backend = _backend(tmp_path, is_owner=True)
    assert backend._claim_if_abandoned() is False
