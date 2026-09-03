"""The WAL wedge fingerprint is necessary, not sufficient.

A clean build whose size is not a multiple of the HNSW sync threshold ends
with queued rows and a lagging vector segment, and stays like that while
idle. Real stores in that state answered the integrity probe in a second
while `doctor` called them "wedged" for 16 days. These tests pin the rule
that the out-of-process probe decides and the fingerprint only explains a
timeout — for the doctor check and for `--heal-wal`.
"""
import json
import sqlite3
import types

import pytest

from lynx import integrity
from lynx.manager import doctor


def _make_store(storage_dir, *, pending=0, age_s=0, lagging=False,
                applied=0, file_paths=()):
    """Fabricate the minimal chroma.sqlite3 the inspector reads.

    `applied` rows sit below the lowest segment watermark (every segment
    consumed them, chroma just hasn't purged the log); `pending` rows sit
    above it when `lagging` is set.
    """
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
    paths = list(file_paths) or [None]
    total = applied + pending
    for i in range(total):
        path = paths[i % len(paths)]
        meta = json.dumps({"file_path": path}) if path else None
        db.execute(
            "INSERT INTO embeddings_queue "
            "(seq_id, created_at, operation, topic, id, metadata) "
            f"VALUES (?, datetime('now', '-{int(age_s)} seconds'), 0, 't', ?, ?)",
            (100 + i, f"chunk-{i}", meta))
    tail = 100 + total - 1 if total else 0
    db.execute("INSERT INTO max_seq_id VALUES ('meta-segment', ?)", (tail,))
    db.execute("INSERT INTO max_seq_id VALUES ('vector-segment', ?)",
               (tail - pending if lagging and pending else tail,))
    db.commit()
    db.close()
    return storage_dir


def _source(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    return {"type": "codebase", "path": str(code), "watcher": {"enabled": False}}


# --- inspect_wal: only unapplied rows name files to re-index --------------

def test_affected_files_are_only_the_unapplied_rows(tmp_path):
    store = tmp_path / "src"
    _make_store(store, applied=2, pending=1, age_s=1200, lagging=True,
                file_paths=[r"C:\proj\old1.cs", r"C:\proj\old2.cs", r"C:\proj\new.cs"])
    info = integrity.inspect_wal(store)
    assert info["pending_ops"] == 3
    assert info["unapplied_ops"] == 1
    assert info["affected_files"] == [r"C:\proj\new.cs"]
    assert info["wedged"] is True


# --- doctor.check_source: the probe decides -------------------------------

def test_fingerprint_with_healthy_probe_is_ok(tmp_path, monkeypatch):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=1200, lagging=True)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "ok", "count": 13})
    result = doctor.check_source("src", _source(tmp_path), tmp_path)
    assert result.status == doctor.STATUS_OK, result
    assert any("awaiting replay" in d for d in result.details)
    assert not any("wedged" in d for d in result.details)


def test_fingerprint_with_timed_out_probe_is_wedged(tmp_path, monkeypatch):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=1200, lagging=True)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "unverified", "detail": "timed out"})
    result = doctor.check_source("src", _source(tmp_path), tmp_path)
    assert result.status == doctor.STATUS_ERROR
    assert "wedged" in result.summary
    assert any("--heal-wal src" in d for d in result.details)


def test_timed_out_probe_without_fingerprint_is_a_warning(tmp_path, monkeypatch):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=5, lagging=True)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "unverified", "detail": "timed out"})
    result = doctor.check_source("src", _source(tmp_path), tmp_path)
    assert result.status == doctor.STATUS_WARN
    assert "could not be verified" in result.summary


def test_corrupt_probe_is_an_error_with_reset_advice(tmp_path, monkeypatch):
    store = tmp_path / "src"
    _make_store(store, applied=10)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "corrupt", "detail": "boom", "crashed": True})
    result = doctor.check_source("src", _source(tmp_path), tmp_path)
    assert result.status == doctor.STATUS_ERROR
    assert "unreadable" in result.summary
    assert any("lynx reset --source src" in d for d in result.details)


def test_store_in_use_skips_the_wedge_verdict(tmp_path, monkeypatch):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=1200, lagging=True)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "in_use", "detail": "held"})
    result = doctor.check_source("src", _source(tmp_path), tmp_path)
    assert result.status == doctor.STATUS_OK
    assert any("in use" in d for d in result.details)


# --- --heal-wal: refuses to cut a store that answers ---------------------

def _heal_args(tmp_path, monkeypatch, store_name="src"):
    cfg = types.SimpleNamespace(sources={store_name: {}}, storage_path=str(tmp_path))
    monkeypatch.setattr("lynx.config.load_config", lambda p: cfg)
    return tmp_path / "config.json"


def test_heal_wal_refuses_a_store_that_answers_the_probe(tmp_path, monkeypatch, capsys):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=1200, lagging=True)
    cfg_path = _heal_args(tmp_path, monkeypatch)
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: {"status": "ok", "count": 13})

    def boom(*a, **k):
        raise AssertionError("heal_wal must not run on a healthy store")
    monkeypatch.setattr(integrity, "heal_wal", boom)

    assert doctor._run_heal_wal("src", cfg_path) == 0
    assert "nothing to heal" in capsys.readouterr().out
    assert integrity.inspect_wal(store)["pending_ops"] == 13  # untouched


def test_heal_wal_runs_when_the_probe_never_answers(tmp_path, monkeypatch, capsys):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=1200, lagging=True)
    cfg_path = _heal_args(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(integrity, "check_index",
                        lambda *a, **k: (calls.append("probe") or
                                         {"status": "unverified", "detail": "timed out"})
                        if not calls else {"status": "ok", "count": 10})
    monkeypatch.setattr(integrity, "_store_usage", lambda p: "free")

    assert doctor._run_heal_wal("src", cfg_path) == 0
    out = capsys.readouterr().out
    assert "confirming" in out and "purged 13" in out
    assert integrity.inspect_wal(store)["pending_ops"] == 0


def test_heal_wal_without_fingerprint_is_a_no_op(tmp_path, monkeypatch, capsys):
    store = tmp_path / "src"
    _make_store(store, applied=10, pending=3, age_s=5, lagging=True)
    cfg_path = _heal_args(tmp_path, monkeypatch)
    monkeypatch.setattr(integrity, "heal_wal",
                        lambda *a, **k: pytest.fail("must not heal"))
    assert doctor._run_heal_wal("src", cfg_path) == 0
    assert "WAL is clean" in capsys.readouterr().out
