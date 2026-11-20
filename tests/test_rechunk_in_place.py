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
