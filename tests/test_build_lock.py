"""One build at a time, and no watcher write to a file while a build is on.

A build deletes a changed file's chunks and puts the new ones in a group
later; a watcher write to that file in between left it in the index twice.
The pass after lost events (1.13.0) starts right after a burst of changes,
when the watcher's own writes are still due, so the two met. And the hash
cache, saved by both, must never be read while the other changes it.
"""
from __future__ import annotations

import json
import threading
import time

from lynx.rag_manager import FILE_HASHES_SCHEMA_VERSION, CodebaseRAG


def _rag(tmp_path) -> CodebaseRAG:
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag.follower = False
    rag._build_lock = threading.RLock()
    rag._write_lock = threading.RLock()
    rag.building = False
    rag.embedding_model_name = "BAAI/bge-small-en-v1.5"
    rag.collection_name = "demo"
    rag.supported_extensions = frozenset({".py"})
    rag.file_hashes_file = tmp_path / "file_hashes.json"
    rag._file_hashes = {}
    return rag


def test_a_watcher_write_waits_for_the_build_on(tmp_path):
    rag = _rag(tmp_path)
    source = tmp_path / "a.py"
    source.write_text("x = 1\n", encoding="utf-8")
    order = []
    rag.is_supported = lambda p: True
    rag._compute_file_hash = lambda p: "new"
    rag._delete_file_chunks = lambda p: order.append("watcher write") or 0
    rag._build_nodes_for_file = lambda p: []
    building = threading.Event()

    def build():
        rag._build_index = lambda: (building.set(), time.sleep(0.3), order.append("build over"))
        rag._record_build_metadata = lambda: None
        rag.update(force=True)

    builder = threading.Thread(target=build)
    builder.start()
    assert building.wait(5)
    assert rag.building
    rag.update_file(str(source))
    builder.join(5)
    assert order == ["build over", "watcher write"]
    assert not rag.building


def test_two_builds_run_one_after_the_other(tmp_path):
    rag = _rag(tmp_path)
    running, overlap = [0], [False]

    def one_build():
        running[0] += 1
        overlap[0] |= running[0] > 1
        time.sleep(0.1)
        running[0] -= 1

    rag._build_index = one_build
    rag._record_build_metadata = lambda: None
    threads = [threading.Thread(target=rag.update, kwargs={"force": True}) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not overlap[0], "two builds ran at once"


def test_the_hash_cache_is_saved_from_a_copy_and_a_temporary_file_of_its_own(tmp_path, monkeypatch):
    import lynx.rag_manager as rm
    rag = _rag(tmp_path)
    rag._grammar_extensions = lambda: []
    rag._file_hashes = {"/src/a.py": {"sha256": "abc", "chunks": 1}}
    real_dumps, real_replace, temporaries = json.dumps, rm.os.replace, []

    def dumps(payload, **kw):
        # another thread records a file while this one encodes
        rag._file_hashes["/src/b.py"] = {"sha256": "def", "chunks": 2}
        return real_dumps(payload, **kw)

    def replace(src, dst):
        temporaries.append(src.name)
        return real_replace(src, dst)

    monkeypatch.setattr(rm.json, "dumps", dumps)
    monkeypatch.setattr(rm.os, "replace", replace)
    rag._save_file_hashes()
    saved = json.loads((tmp_path / "file_hashes.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == FILE_HASHES_SCHEMA_VERSION
    assert set(saved["files"]) == {"/src/a.py"}, "the live cache was encoded, not a copy"
    assert not list(tmp_path.glob("*.tmp"))
    assert len(temporaries) == 1 and temporaries[0] != "file_hashes.json.tmp", temporaries
    assert str(threading.get_ident()) in temporaries[0]


def test_a_reset_waits_for_the_build_on(tmp_path):
    rag = _rag(tmp_path)
    order = []
    rag._reset_collection = lambda: order.append("reset")
    rag._save_file_hashes = lambda: None
    rag._invalidate_bm25 = lambda: None
    building = threading.Event()
    rag._build_index = lambda: (building.set(), time.sleep(0.3), order.append("build over"))
    rag._record_build_metadata = lambda: None
    builder = threading.Thread(target=rag.update, kwargs={"force": True})
    builder.start()
    assert building.wait(5)
    rag.reset_index()
    builder.join(5)
    assert order == ["build over", "reset"]


def test_nothing_gets_in_between_a_resets_wipe_and_its_rebuild(tmp_path):
    import types
    from lynx.sources.codebase import CodebaseBackend
    order, wiped = [], threading.Event()
    rag = types.SimpleNamespace(_build_lock=threading.RLock())

    def reset_index():
        order.append("wipe")
        wiped.set()
        time.sleep(0.2)

    def update(force=False):
        time.sleep(0.2)
        order.append("rebuild")

    rag.reset_index, rag.update = reset_index, update
    backend = CodebaseBackend.__new__(CodebaseBackend)
    backend.name, backend.is_owner, backend.rag, backend.graph = "src", True, rag, None
    resetting = threading.Thread(target=backend.reset)
    resetting.start()
    assert wiped.wait(5)
    with rag._build_lock:                      # a watcher write
        order.append("watcher write")
    resetting.join(5)
    assert order == ["wipe", "rebuild", "watcher write"]
