"""The doctor checks that read a real store: the lexical index, and the disk
against the index.

Both failures they look for left the index answering, healthy by every other
measure: a lexical index that could not load turned hybrid search into dense
search, and files gone from disk (or never indexed) left searches citing code
that does not exist, or answering "no results" for code that does. A small
ChromaDB store is built for each test, with explicit vectors (no model).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from lynx.manager import doctor


def _store(storage: Path, files: dict):
    """A real ChromaDB store under `storage`, one chunk per {path: text}."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(storage), settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection("src")
    paths = list(files)
    collection.add(
        ids=[f"id{i}" for i in range(len(paths))],
        documents=[files[p] for p in paths],
        embeddings=[[float(i), 1.0] for i in range(len(paths))],
        metadatas=[{"file_path": p, "file_name": os.path.basename(p)} for p in paths],
    )
    return collection


def test_the_lexical_probe_answers_on_a_working_index(tmp_path):
    _store(tmp_path / "store", {"/r/a.py": "def alpha(): return beta()",
                                "/r/b.py": "class Gamma: pass"})

    result = doctor.check_lexical_index(tmp_path / "store", "src")

    assert result["status"] == "ok" and result["chunks"] == 2


def test_the_lexical_probe_fails_when_the_index_cannot_be_read(tmp_path, monkeypatch):
    import lynx.rag_manager as rag_manager

    _store(tmp_path / "store", {"/r/a.py": "def alpha(): return beta()"})

    def too_big(*_a, **_k):
        raise RuntimeError("(code: 1) too many SQL variables")

    monkeypatch.setattr(rag_manager, "read_collection_rows", too_big)
    result = doctor.check_lexical_index(tmp_path / "store", "src")

    assert result["status"] == "error" and "too many SQL variables" in result["detail"]


def test_the_doctor_lists_files_missing_from_the_index_and_files_it_should_drop(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    for name in ("kept.py", "new.py", "2026-01-01-chat.py"):
        (code / name).write_text("x = 1", encoding="utf-8")
    kept, new = str(code / "kept.py"), str(code / "new.py")
    gone = str(code / "gone.py")
    _store(tmp_path / "store" / "src", {kept: "x = 1", gone: "y = 2"})
    src_cfg = {
        "type": "codebase", "path": code, "supported_extensions": frozenset({".py"}),
        "ignored_path_fragments": (), "ignored_globs": ("20??-??-??-*.py",),
        "watcher": {"enabled": True},
    }

    import lynx.integrity as integrity
    integrity_probe = integrity.check_index
    try:
        integrity.check_index = lambda *a, **k: {"status": "ok", "count": 2}
        result = doctor.check_source("src", src_cfg, tmp_path / "store")
    finally:
        integrity.check_index = integrity_probe

    assert result.status == doctor.STATUS_WARN
    text = " ".join(result.details)
    assert "new.py" in text and "gone.py" in text
    assert "2026-01-01-chat.py" not in text      # excluded by a glob: not missing
    assert "lynx build --source src" in text


def test_a_lexical_index_that_cannot_load_turns_the_source_red(tmp_path, monkeypatch):
    import lynx.integrity as integrity

    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text("x = 1", encoding="utf-8")
    _store(tmp_path / "store" / "src", {str(code / "a.py"): "x = 1"})
    monkeypatch.setattr(integrity, "check_index", lambda *a, **k: {"status": "ok", "count": 1})
    monkeypatch.setattr(doctor, "check_lexical_index",
                        lambda *a, **k: {"status": "error", "detail": "loading it failed: boom"})
    src_cfg = {"type": "codebase", "path": code, "supported_extensions": frozenset({".py"}),
               "ignored_path_fragments": (), "ignored_globs": (), "watcher": {"enabled": True}}

    result = doctor.check_source("src", src_cfg, tmp_path / "store")

    assert result.status == doctor.STATUS_ERROR
    assert "lexical index" in result.summary and any("boom" in d for d in result.details)


# ---------------------------------------------------------------------------
# The Lynx serving an index against the one running the check
# ---------------------------------------------------------------------------

def _owned_by(storage: Path, pid: int, lynx=None):
    import json
    import time
    from lynx import ownership

    storage.mkdir(parents=True, exist_ok=True)
    claim = {"pid": pid, "host": ownership._host(), "beat": time.time(), "since": "x"}
    if lynx is not None:
        claim["lynx"] = lynx
    (storage / ownership.OWNER_FILE).write_text(json.dumps(claim), encoding="utf-8")


def test_the_doctor_names_a_server_running_other_code(tmp_path):
    # Same version string, other code: the case two installs produced.
    from lynx.ownership import code_identity

    other = {**code_identity(), "code": "not-this-one", "path": "/elsewhere/lynx"}
    _owned_by(tmp_path / "src", os.getppid(), other)

    result = doctor.check_running_lynx(tmp_path, ["src"])

    assert result.status == doctor.STATUS_WARN
    assert any("/elsewhere/lynx" in d for d in result.details)


def test_a_server_from_before_version_reporting_counts_as_older(tmp_path):
    _owned_by(tmp_path / "src", os.getppid())

    result = doctor.check_running_lynx(tmp_path, ["src"])

    assert result.status == doctor.STATUS_WARN
    assert any("older" in d for d in result.details)


def test_a_server_running_this_code_is_fine(tmp_path):
    from lynx.ownership import code_identity

    _owned_by(tmp_path / "src", os.getppid(), code_identity())

    assert doctor.check_running_lynx(tmp_path, ["src"]).status == doctor.STATUS_OK
