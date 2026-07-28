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


# ---------------------------------------------------------------------------
# The quieter wedge: an emptied queue under high segment watermarks
# ---------------------------------------------------------------------------

def test_empty_queue_with_high_watermark_discards_writes(tmp_path):
    """`embeddings_queue.seq_id` is INTEGER PRIMARY KEY with no AUTOINCREMENT:
    once emptied, ids restart at 1, and segments parked at N drop every new row
    as already-applied. The store then accepts writes and stores none — no
    exception, count() unchanged, index frozen. Observed in the field: 25 days
    of a watcher logging success into a void."""
    _make_store(tmp_path)          # empty queue...
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("UPDATE max_seq_id SET seq_id = 64111")   # ...high watermarks
    db.commit()
    db.close()

    info = integrity.inspect_wal(tmp_path)
    assert info["pending_ops"] == 0
    assert info["watermark"] == 64111
    assert info["discarding_writes"] is True


def test_clean_store_is_not_discarding_writes(tmp_path):
    _make_store(tmp_path)
    assert integrity.inspect_wal(tmp_path)["discarding_writes"] is False


def test_pending_writes_are_not_mistaken_for_discarding(tmp_path):
    """With rows still queued the watermarks are legitimately in use; calling
    that 'discarding' would send a healthy indexing run to the repair shop."""
    _make_store(tmp_path, pending=3, age_s=5)
    assert integrity.inspect_wal(tmp_path)["discarding_writes"] is False


def test_heal_anchors_the_queue_above_the_watermark(tmp_path, monkeypatch):
    """The cure moves the QUEUE up to the segments, never the segments down to
    the queue: a vector segment's persisted HNSW is built up to its watermark,
    so resetting it to 0 asks for a replay of operations the WAL no longer has
    — verified in the field to leave the segment lagging and chromadb
    deadlocked at zero CPU."""
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path)
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("UPDATE max_seq_id SET seq_id = 64111")
    db.commit()
    db.close()

    outcome = integrity.heal_wal(tmp_path)
    assert outcome["anchored_seq"] == 64111

    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    try:
        # The watermarks are untouched; the next write gets 64112.
        assert db.execute("SELECT DISTINCT seq_id FROM max_seq_id"
                          ).fetchall() == [(64111,)]
        db.execute("INSERT INTO embeddings_queue (operation, topic, id) "
                   "VALUES (0, 't', 'real-write')")
        assert db.execute("SELECT seq_id FROM embeddings_queue WHERE "
                          "id = 'real-write'").fetchone()[0] == 64112
    finally:
        db.close()
    assert integrity.inspect_wal(tmp_path)["discarding_writes"] is False


def test_the_anchor_is_not_a_pending_write(tmp_path, monkeypatch):
    """It lives in the queue forever by design; counting it as pending would
    make every cured store look wedged at the next doctor run."""
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path)
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("UPDATE max_seq_id SET seq_id = 500")
    db.commit()
    db.close()
    integrity.heal_wal(tmp_path)

    info = integrity.inspect_wal(tmp_path)
    assert info["pending_ops"] == 0
    assert info["wedged"] is False
    assert info["affected_files"] == []


def test_heal_fast_forwards_a_segment_stuck_on_purged_rows(tmp_path, monkeypatch):
    """A segment consumes the WAL from its own watermark. Once the rows it
    still needs are purged they are gone forever, so it never advances — and
    every later write waits on it (measured: 5 chunks/minute, vector index
    frozen). It must be moved up; the chunks left without a vector are caught
    by inspect_coverage and rebuilt from their files."""
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path)
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("UPDATE max_seq_id SET seq_id = 64116 WHERE segment_id = 'meta-segment'")
    db.execute("UPDATE max_seq_id SET seq_id = 63966 WHERE segment_id = 'vector-segment'")
    db.commit()
    db.close()

    outcome = integrity.heal_wal(tmp_path)
    assert outcome["forwarded_segments"] == 1
    assert outcome["anchored_seq"] == 64116

    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    try:
        assert db.execute("SELECT DISTINCT seq_id FROM max_seq_id"
                          ).fetchall() == [(64116,)]
    finally:
        db.close()


def test_fast_forward_is_a_no_op_while_writes_are_pending(tmp_path):
    """With rows still queued, a lagging segment is just behind — not stranded.
    Skipping them would discard work the segment could still apply."""
    _make_store(tmp_path, pending=2, age_s=5, lagging=True)
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    try:
        assert integrity._fast_forward_lagging_segments(db) == 0
    finally:
        db.close()


def test_anchor_is_a_no_op_while_writes_are_pending(tmp_path):
    """With rows still queued the numbering is in use and consistent; touching
    it would renumber live work."""
    _make_store(tmp_path, pending=2, age_s=5)
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    try:
        assert integrity._anchor_queue_sequence(db) == 0
    finally:
        db.close()


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
