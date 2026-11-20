"""A chunker upgrade re-chunks the index in place instead of wiping it.

The SHA cache records which chunker version cut the stored chunks. When only
that version moved, the vectors live in the same space and stay usable until
each file is replaced, so the cache keeps every file, marked as changed. Any
other change (embedding model, collection, extensions) still means a clean
rebuild: vectors from two models must not mix.
"""
from __future__ import annotations

import json
from pathlib import Path

from lynx.chunking import CHUNKER_VERSION
from lynx.rag_manager import FILE_HASHES_SCHEMA_VERSION, CodebaseRAG


def _rag(tmp_path, snapshot_changes: dict) -> CodebaseRAG:
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag.embedding_model_name = "BAAI/bge-small-en-v1.5"
    rag.collection_name = "demo"
    rag.supported_extensions = frozenset({".py"})
    rag.file_hashes_file = Path(tmp_path) / "file_hashes.json"
    snapshot = {**rag._file_hashes_snapshot(), **snapshot_changes}
    rag.file_hashes_file.write_text(json.dumps({
        "schema_version": FILE_HASHES_SCHEMA_VERSION,
        "config_snapshot": snapshot,
        "files": {"/src/a.py": {"sha256": "abc", "chunks": 3},
                  "/src/b.py": {"sha256": "def", "chunks": 0}},
    }), encoding="utf-8")
    return rag


def test_a_new_chunker_keeps_every_file_as_changed(tmp_path):
    rag = _rag(tmp_path, {"chunker_version": CHUNKER_VERSION - 1})
    files = rag._load_file_hashes()
    assert set(files) == {"/src/a.py", "/src/b.py"}          # not {}: no wipe
    assert all(entry["sha256"] is None for entry in files.values())
    assert files["/src/a.py"]["chunks"] == 3


def test_a_new_embedding_model_still_rebuilds_from_scratch(tmp_path):
    rag = _rag(tmp_path, {"chunker_version": CHUNKER_VERSION - 1,
                          "embedding_model_name": "another/model"})
    assert rag._load_file_hashes() == {}


def test_an_unchanged_snapshot_keeps_the_hashes(tmp_path):
    rag = _rag(tmp_path, {})
    assert rag._load_file_hashes()["/src/a.py"]["sha256"] == "abc"


# A language gains a grammar (GDScript) or loses it (its wheel is missing on
# this platform): the files of that extension were cut the other way, though
# their content is the same. The cache records which extensions had a grammar,
# so those files, and only those, are re-chunked in place.

_FILES = {"/src/a.py": {"sha256": "abc", "chunks": 3},
          "/src/player.gd": {"sha256": "gd1", "chunks": 1},
          "/src/README.md": {"sha256": "md1", "chunks": 2}}


def _rag_with_grammars(tmp_path, monkeypatch, stored, live) -> CodebaseRAG:
    import lynx.rag_manager as rm
    monkeypatch.setattr(rm, "grammar_extensions", lambda: sorted(live))
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag.embedding_model_name = "BAAI/bge-small-en-v1.5"
    rag.collection_name = "demo"
    rag.supported_extensions = frozenset({".py", ".gd", ".md"})
    rag.file_hashes_file = Path(tmp_path) / "file_hashes.json"
    payload = {"schema_version": FILE_HASHES_SCHEMA_VERSION,
               "config_snapshot": rag._file_hashes_snapshot(),
               "files": _FILES}
    if stored is not None:
        payload["grammars"] = sorted(stored)
    rag.file_hashes_file.write_text(json.dumps(payload), encoding="utf-8")
    return rag


def _changed(files: dict) -> set:
    return {path for path, entry in files.items() if entry["sha256"] is None}


def test_a_new_grammar_rechunks_only_its_files(tmp_path, monkeypatch):
    rag = _rag_with_grammars(tmp_path, monkeypatch,
                             stored={".py"}, live={".py", ".gd"})
    files = rag._load_file_hashes()
    assert _changed(files) == {"/src/player.gd"}
    assert files["/src/a.py"]["sha256"] == "abc"


def test_a_grammar_that_went_away_rechunks_its_files(tmp_path, monkeypatch):
    rag = _rag_with_grammars(tmp_path, monkeypatch,
                             stored={".py", ".gd"}, live={".py"})
    assert _changed(rag._load_file_hashes()) == {"/src/player.gd"}


def test_a_cache_without_the_list_was_cut_without_gdscript(tmp_path, monkeypatch):
    with_gd = _rag_with_grammars(tmp_path, monkeypatch,
                                 stored=None, live={".py", ".gd"})
    assert _changed(with_gd._load_file_hashes()) == {"/src/player.gd"}
    without_gd = _rag_with_grammars(tmp_path, monkeypatch,
                                    stored=None, live={".py"})
    assert _changed(without_gd._load_file_hashes()) == set()


def test_a_saved_cache_reads_back_unchanged(tmp_path, monkeypatch):
    rag = _rag_with_grammars(tmp_path, monkeypatch,
                             stored={".py"}, live={".py", ".gd"})
    rag._file_hashes = {p: dict(e) for p, e in _FILES.items()}
    rag._save_file_hashes()
    assert _changed(rag._load_file_hashes()) == set()
