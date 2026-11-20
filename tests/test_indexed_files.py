"""Which files the index holds: what the store says, not only the SHA cache.

The cache is written when a build ends. During a source's first build, or
after a build was interrupted, the store already holds chunks the cache does
not list, and `module_summary` must still find their files.
"""
from __future__ import annotations

from lynx.rag_manager import CodebaseRAG


def _rag(store_files, cached, excluded=()):
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag._ensure_bm25 = lambda: None
    rag._bm25_files = {p: (p.rsplit("/", 1)[-1], 2) for p in store_files}
    rag._bm25_excluded = set(excluded)
    rag._file_hashes = {p: {"sha256": "x", "chunks": 2} for p in cached}
    return rag


def test_files_in_the_store_count_before_the_cache_is_written():
    rag = _rag(["/src/query.py", "/src/builder.py"], cached=[])
    assert sorted(rag.indexed_files()) == ["/src/builder.py", "/src/query.py"]


def test_the_cache_and_the_store_are_merged():
    rag = _rag(["/src/a.py"], cached=["/src/b.py"])
    assert sorted(rag.indexed_files()) == ["/src/a.py", "/src/b.py"]


def test_files_the_config_now_excludes_are_left_out():
    rag = _rag(["/src/a.py", "/src/gen.py"], cached=["/src/gen.py"], excluded=["/src/gen.py"])
    assert rag.indexed_files() == ["/src/a.py"]
