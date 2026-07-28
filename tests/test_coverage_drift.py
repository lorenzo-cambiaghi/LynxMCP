"""Coverage drift: the SHA cache claiming files the index does not hold.

``file_hashes.json`` is what ``_partition_files`` trusts to decide what needs
work: a file listed there with a matching SHA is classified *unchanged* and
skipped. So an entry written for a file whose chunks never landed in the index
makes that file permanently invisible to search — no error, no retry, and
``count()`` stays perfectly healthy because every other file is fine.

That is not hypothetical: on 2026-07-03 the `framework` source recorded 202
such files (an insert that raised was logged and then recorded as indexed
anyway), and nothing noticed for 25 days.

``inspect_coverage`` must recognize the state read-only (never a Chroma
client, which would contend with a running `lynx serve`), and ``heal_coverage``
must drop exactly those entries so the next pass re-indexes them.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from lynx import integrity


def _make_store(storage_dir, *, indexed_files=(), cached=None, legacy=False,
                vectorized=None):
    """Fabricate the minimal store the inspector reads: a chroma.sqlite3 whose
    metadata names `indexed_files`, plus a file_hashes.json holding `cached`
    ({path: entry}). `legacy` writes the pre-envelope flat cache shape.

    `vectorized`, when given, is the subset of chunk ids the fake HNSW segment
    knows — anything else is a chunk with metadata but no vector."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(storage_dir / "chroma.sqlite3")
    db.execute("CREATE TABLE embedding_metadata (id INTEGER, key TEXT, "
               "string_value TEXT, int_value INTEGER, float_value REAL, "
               "bool_value INTEGER)")
    db.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, "
               "segment_id TEXT, embedding_id TEXT, seq_id BLOB, "
               "created_at TIMESTAMP)")
    for i, path in enumerate(indexed_files):
        db.execute("INSERT INTO embedding_metadata VALUES (?,?,?,NULL,NULL,NULL)",
                   (i, "file_path", str(path)))
        db.execute("INSERT INTO embeddings VALUES (?,'seg',?,NULL,NULL)",
                   (i, f"chunk-{i}"))
    db.commit()
    db.close()

    if vectorized is not None:
        import pickle
        seg = storage_dir / "0000-vector-segment"
        seg.mkdir(exist_ok=True)
        with open(seg / "index_metadata.pickle", "wb") as fh:
            pickle.dump({"id_to_label": {cid: i for i, cid in
                                         enumerate(vectorized)}}, fh)

    if cached is not None:
        payload = cached if legacy else {
            "schema_version": 1,
            "config_snapshot": {"embedding_model": "test"},
            "files": cached,
        }
        (storage_dir / "file_hashes.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    return storage_dir


def _entry(sha="abc", chunks=None):
    e = {"sha256": sha, "last_indexed_at": "2026-07-03T21:57:04"}
    if chunks is not None:
        e["chunks"] = chunks
    return e


# ---------------------------------------------------------------------------
# inspect_coverage
# ---------------------------------------------------------------------------

def test_missing_store_returns_none(tmp_path):
    assert integrity.inspect_coverage(tmp_path) is None


def test_cache_matching_the_index_is_not_drifted(tmp_path):
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs", r"C:\proj\b.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\b.cs": _entry()})
    report = integrity.inspect_coverage(tmp_path)
    assert report["drifted"] is False
    assert report["phantom_files"] == []
    assert report["indexed_files"] == 2
    assert report["cached_files"] == 2


def test_file_cached_but_absent_from_the_index_is_phantom(tmp_path):
    """The incident state: the insert failed, the SHA was recorded anyway."""
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\lost.cs": _entry()})
    report = integrity.inspect_coverage(tmp_path)
    assert report["drifted"] is True
    assert report["phantom_files"] == [r"C:\proj\lost.cs"]


def test_zero_chunk_files_are_expected_absences_not_phantoms(tmp_path):
    """A file that legitimately produces no chunks is absent BY DESIGN; flagging
    it would make the check cry wolf on every empty file in the tree."""
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry(),
                        r"C:\proj\empty.cs": _entry(chunks=0)})
    report = integrity.inspect_coverage(tmp_path)
    assert report["drifted"] is False
    assert report["empty_files"] == [r"C:\proj\empty.cs"]


def test_paths_are_compared_normalized(tmp_path):
    """The cache and the index may disagree on slash style / case; a phantom
    verdict must come from a real absence, not from spelling."""
    _make_store(tmp_path,
                indexed_files=["C:/proj/Sub/a.cs"],
                cached={r"C:\proj\sub\a.cs": _entry()})
    assert integrity.inspect_coverage(tmp_path)["drifted"] is False


def test_legacy_flat_cache_is_understood(tmp_path):
    _make_store(tmp_path, indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\lost.cs": _entry()},
                legacy=True)
    report = integrity.inspect_coverage(tmp_path)
    assert report["phantom_files"] == [r"C:\proj\lost.cs"]


def test_no_vector_segment_means_no_claim(tmp_path):
    """Without a readable segment mapping we must not invent a verdict: a
    Chroma build that stores it elsewhere would otherwise look 100% broken."""
    _make_store(tmp_path, indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry()})
    report = integrity.inspect_coverage(tmp_path)
    assert report["unvectorized"] == 0
    assert report["drifted"] is False


def test_chunks_without_a_vector_are_drift(tmp_path):
    """The state that kills every search while count() stays green: metadata
    holds ids the vector index cannot resolve (incident of 2026-07-03 —
    157 chunks across 3 files, an insert interrupted between the two)."""
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs", r"C:\proj\half.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\half.cs": _entry()},
                vectorized=["chunk-0"])          # chunk-1 has no vector
    report = integrity.inspect_coverage(tmp_path)
    assert report["unvectorized"] == 1
    assert report["unvectorized_files"] == [r"C:\proj\half.cs"]
    assert report["drifted"] is True


def test_chunks_still_deliverable_by_the_queue_are_not_drift(tmp_path):
    """Chroma persists the vector segment on a threshold, so the tail of a
    fresh build sits in the queue until the next open. Calling that damage
    would fire after every build — and a check that cries wolf gets ignored."""
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs", r"C:\proj\fresh.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\fresh.cs": _entry()},
                vectorized=["chunk-0"])          # chunk-1 not in the HNSW yet
    db = sqlite3.connect(tmp_path / "chroma.sqlite3")
    db.execute("CREATE TABLE embeddings_queue (seq_id INTEGER PRIMARY KEY, "
               "created_at TIMESTAMP, operation INTEGER, topic TEXT, id TEXT)")
    db.execute("CREATE TABLE max_seq_id (segment_id TEXT PRIMARY KEY, seq_id INTEGER)")
    db.execute("INSERT INTO max_seq_id VALUES ('vector', 10)")
    db.execute("INSERT INTO max_seq_id VALUES ('meta', 11)")
    db.execute("INSERT INTO embeddings_queue VALUES (11, NULL, 0, 't', 'chunk-1')")
    db.commit()
    db.close()

    report = integrity.inspect_coverage(tmp_path)
    assert report["unvectorized"] == 0
    assert report["drifted"] is False


def test_heal_requeues_unvectorized_files_too(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs", r"C:\proj\half.cs"],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\half.cs": _entry()},
                vectorized=["chunk-0"])
    outcome = integrity.heal_coverage(tmp_path)
    assert outcome["unvectorized"] == 1
    assert outcome["reindex_files"] == [r"C:\proj\half.cs"]


# ---------------------------------------------------------------------------
# _drop_from_file_hashes — the envelope bug
# ---------------------------------------------------------------------------

def test_drop_reaches_entries_inside_the_envelope(tmp_path):
    """Regression: iterating the top level of the current schema only ever sees
    schema_version / config_snapshot / files, so every heal reported success
    while dropping nothing — including `heal_wal`, which shares this helper."""
    _make_store(tmp_path, indexed_files=[],
                cached={r"C:\proj\a.cs": _entry(), r"C:\proj\b.cs": _entry()})
    dropped = integrity._drop_from_file_hashes(tmp_path, [r"C:\proj\a.cs"])
    assert dropped == [r"C:\proj\a.cs"]
    left = json.loads((tmp_path / "file_hashes.json").read_text(encoding="utf-8"))
    assert list(left["files"]) == [r"C:\proj\b.cs"]
    assert left["schema_version"] == 1        # envelope preserved
    assert "config_snapshot" in left


def test_drop_still_works_on_the_legacy_flat_cache(tmp_path):
    _make_store(tmp_path, indexed_files=[],
                cached={r"C:\proj\a.cs": _entry()}, legacy=True)
    assert integrity._drop_from_file_hashes(tmp_path, [r"C:\proj\a.cs"])
    assert json.loads(
        (tmp_path / "file_hashes.json").read_text(encoding="utf-8")) == {}


# ---------------------------------------------------------------------------
# heal_coverage
# ---------------------------------------------------------------------------

def test_heal_drops_only_the_phantoms(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path,
                indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry(),
                        r"C:\proj\lost.cs": _entry(),
                        r"C:\proj\empty.cs": _entry(chunks=0)})
    outcome = integrity.heal_coverage(tmp_path)
    assert outcome["phantom"] == 1
    assert outcome["reindex_files"] == [r"C:\proj\lost.cs"]
    left = json.loads((tmp_path / "file_hashes.json").read_text(encoding="utf-8"))
    assert set(left["files"]) == {r"C:\proj\a.cs", r"C:\proj\empty.cs"}


def test_heal_refuses_while_the_store_is_open(tmp_path, monkeypatch):
    """A live `lynx serve` holds the cache in memory and would write it back
    over the repair — worse than not repairing, because it looks done."""
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "other")
    _make_store(tmp_path, indexed_files=[],
                cached={r"C:\proj\lost.cs": _entry()})
    with pytest.raises(RuntimeError, match="open in another running"):
        integrity.heal_coverage(tmp_path)


def test_heal_on_a_clean_store_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "_store_usage", lambda _p: "free")
    _make_store(tmp_path, indexed_files=[r"C:\proj\a.cs"],
                cached={r"C:\proj\a.cs": _entry()})
    outcome = integrity.heal_coverage(tmp_path)
    assert outcome["phantom"] == 0
    assert outcome["reindex_files"] == []
