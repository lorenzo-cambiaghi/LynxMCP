"""WAL-wedge fingerprint and surgical heal.

A process killed mid-write leaves pending rows in chroma's
``embeddings_queue`` with a segment's ``max_seq_id`` behind the queue tail.
chromadb (observed on 1.5.9) deadlocks forever on that state — every open
hangs at zero CPU, which surfaces as integrity-probe timeouts at every
serve/UI start. ``inspect_wal`` must recognize the fingerprint via plain
read-only sqlite (never a Chroma client), and ``heal_wal`` must purge it
and queue the affected files for re-indexing.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from lynx import integrity


def _make_store(storage_dir, *, pending=0, age_s=0, lagging=False,
                locks=0, file_paths=()):
    """Fabricate the minimal chroma.sqlite3 the inspector reads."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(storage_dir / "chroma.sqlite3")
    db.execute(
        "CREATE TABLE embeddings_queue (seq_id INTEGER PRIMARY KEY, "
        "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
        "operation INTEGER NOT NULL, topic TEXT NOT NULL, id TEXT NOT NULL, "
        "vector BLOB, encoding TEXT, metadata TEXT)")
    db.execute("CREATE TABLE max_seq_id (segment_id TEXT PRIMARY KEY, "
               "seq_id INTEGER)")
    db.execute("CREATE TABLE acquire_write (id INTEGER PRIMARY KEY, "
               "lock_status INTEGER NOT NULL)")

    paths = list(file_paths) or [None] * pending
    for i in range(pending):
        meta = (json.dumps({"file_path": paths[i % len(paths)]})
                if paths[i % len(paths)] else None)
        db.execute(
            "INSERT INTO embeddings_queue "
            "(seq_id, created_at, operation, topic, id, metadata) "
            f"VALUES (?, datetime('now', '-{int(age_s)} seconds'), 0, 't', ?, ?)",
            (100 + i, f"chunk-{i}", meta))
    # Two segments: metadata caught up; vector optionally lagging.
    tail = 100 + pending - 1 if pending else 0
    db.execute("INSERT INTO max_seq_id VALUES ('meta-segment', ?)", (tail,))
    db.execute("INSERT INTO max_seq_id VALUES ('vector-segment', ?)",
               (tail - pending if lagging and pending else tail,))
    for i in range(locks):
        db.execute("INSERT INTO acquire_write VALUES (?, 1)", (i + 1,))
    db.commit()
    db.close()
    return storage_dir


def test_missing_store_returns_none(tmp_path):
    assert integrity.inspect_wal(tmp_path) is None


def test_clean_store_is_not_wedged(tmp_path):
    _make_store(tmp_path)
    info = integrity.inspect_wal(tmp_path)
    assert info["pending_ops"] == 0
    assert info["wedged"] is False


def test_old_pending_with_lagging_segment_is_wedged(tmp_path):
    _make_store(tmp_path, pending=3, age_s=1200, lagging=True, locks=5,
                file_paths=[r"C:\proj\a.cs", r"C:\proj\b.cs"])
    info = integrity.inspect_wal(tmp_path)
    assert info["wedged"] is True
    assert info["pending_ops"] == 3
    assert info["lagging_segments"] == 1
    assert info["stale_locks"] == 5
    assert info["oldest_pending_s"] > 600
    assert info["affected_files"] == [r"C:\proj\a.cs", r"C:\proj\b.cs"]


def test_young_pending_is_in_flight_not_wedged(tmp_path):
    # A live indexer legitimately has ops in flight; age is the discriminator.
    _make_store(tmp_path, pending=3, age_s=5, lagging=True)
    assert integrity.inspect_wal(tmp_path)["wedged"] is False


def test_caught_up_segments_are_not_wedged(tmp_path):
    # Old pending rows but every segment consumed them (purge just hasn't
    # run) — not the deadlock state.
    _make_store(tmp_path, pending=3, age_s=1200, lagging=False)
    assert integrity.inspect_wal(tmp_path)["wedged"] is False


def test_heal_purges_and_queues_reindex(tmp_path, monkeypatch):
    affected = r"C:\proj\a.cs"
    _make_store(tmp_path, pending=2, age_s=1200, lagging=True, locks=3,
                file_paths=[affected])
    (tmp_path / "file_hashes.json").write_text(json.dumps({
        affected: "sha-a",
        r"C:\proj\other.cs": "sha-other",
    }), encoding="utf-8")

    monkeypatch.setattr(integrity, "_store_usage", lambda p: "free")
    outcome = integrity.heal_wal(tmp_path)

    assert outcome["purged_ops"] == 2
    assert outcome["purged_locks"] == 3
    assert outcome["reindex_files"] == [affected]
    # The wedge is gone and the untouched hash entry survived.
    info = integrity.inspect_wal(tmp_path)
    assert info["pending_ops"] == 0 and info["wedged"] is False
    hashes = json.loads((tmp_path / "file_hashes.json").read_text(encoding="utf-8"))
    assert list(hashes) == [r"C:\proj\other.cs"]


def test_heal_refuses_store_in_use(tmp_path, monkeypatch):
    _make_store(tmp_path, pending=1, age_s=1200, lagging=True)
    monkeypatch.setattr(integrity, "_store_usage", lambda p: "other")
    with pytest.raises(RuntimeError, match="stop it and retry"):
        integrity.heal_wal(tmp_path)
    # Nothing was purged.
    assert integrity.inspect_wal(tmp_path)["pending_ops"] == 1
