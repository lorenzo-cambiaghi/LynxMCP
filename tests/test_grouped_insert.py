"""A build embeds pieces of many files per call, and a failed group still
leaves out only the file that fails.

lynx-zombie-bench E29/E33 (2026-11-22): 83% of an index build was ONNX
inference, and the index inserted one file at a time, so the embedder's sort
by length saw a file's 4-5 pieces. Grouping pieces of many files: 163.7 s to
52.6 s for 2,202 pieces, vectors equal. Builds a real index of small files
(real embedding model, as test_sha_incremental does).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _rag(code: Path, storage: Path):
    from lynx.rag_manager import CodebaseRAG
    return CodebaseRAG(
        codebase_path=str(code), rag_storage_path=str(storage), supported_extensions=[".txt"],
        embedding_model_name="BAAI/bge-small-en-v1.5", collection_name="src",
        search_mode="hybrid", rrf_k=60, candidate_pool_size=10,
    )


def _code(tmp: Path, n: int) -> Path:
    code = tmp / "code"
    code.mkdir(parents=True)
    for i in range(n):
        (code / f"f{i:02d}.txt").write_text(f"File {i} talks about topic number {i}.", encoding="utf-8")
    return code


def _hashed(storage: Path) -> set:
    raw = json.loads((storage / "file_hashes.json").read_text(encoding="utf-8"))
    return {Path(p).name for p in (raw.get("files") or {})}


@pytest.fixture
def inserts(monkeypatch):
    from llama_index.core import VectorStoreIndex
    real = VectorStoreIndex.insert_nodes
    calls = []

    def spy(self, nodes, **kw):
        files = {Path(n.metadata.get("file_path", "")).name for n in nodes}
        calls.append(files)
        if spy.fail_on and spy.fail_on in files:
            raise RuntimeError("the store refused this batch")
        return real(self, nodes, **kw)

    spy.fail_on = None
    monkeypatch.setattr(VectorStoreIndex, "insert_nodes", spy)
    spy.calls = calls
    return spy


def test_a_build_inserts_pieces_of_many_files_per_call(tmp_path, inserts):
    rag = _rag(_code(tmp_path, 30), tmp_path / "s")
    assert len(inserts.calls) == 1 and len(inserts.calls[0]) == 30
    assert rag._collection_count() == 30
    assert len(_hashed(tmp_path / "s")) == 30


def test_a_failing_file_in_a_group_is_the_only_one_left_out(tmp_path, inserts):
    inserts.fail_on = "f07.txt"
    rag = _rag(_code(tmp_path, 12), tmp_path / "s")
    assert rag._collection_count() == 11
    assert _hashed(tmp_path / "s") == {f"f{i:02d}.txt" for i in range(12)} - {"f07.txt"}


def test_the_embedder_sorts_a_whole_call_before_cutting_it_into_runs():
    from lynx.embeddings import DEFAULT_RUN_SIZE, OnnxEmbedding
    emb = OnnxEmbedding("BAAI/bge-small-en-v1.5")
    runs = []
    real = emb._runtime.encode

    def encode(texts, normalize):
        runs.append([len(t) for t in texts])
        return real(texts, normalize)

    emb._runtime.encode = encode
    texts = [("word " * ((i * 37) % 90 + 1)).strip() for i in range(100)]
    vecs = emb.get_text_embedding_batch(texts)
    assert len(vecs) == 100
    # LlamaIndex hands the whole call over, the embedder cuts it into runs
    # of similar length (each run starts where the previous one ended).
    assert len(runs) == -(-100 // DEFAULT_RUN_SIZE)
    assert all(max(a) <= min(b) for a, b in zip(runs, runs[1:]))


class _Killed(BaseException):
    """The process dying: not an Exception, so nothing in the build catches it."""


@pytest.mark.parametrize("after_insert", [False, True], ids=["killed before a group", "killed after a group went in"])
def test_a_build_cut_short_goes_on_from_where_it_stopped(tmp_path, inserts, monkeypatch, after_insert):
    """A first index of a large repository takes hours, and it used to be
    recorded only at the end: cut short, the next pass found no record, wiped
    what was there and started again. Each group is recorded once it is in."""
    from llama_index.core import VectorStoreIndex
    from lynx import rag_manager
    monkeypatch.setattr(rag_manager, "INSERT_GROUP_NODES", 5)
    code = _code(tmp_path, 30)
    real = VectorStoreIndex.insert_nodes
    calls = [0]

    def dying(self, nodes, **kw):
        calls[0] += 1
        if calls[0] == 3 and not after_insert:
            raise _Killed
        out = real(self, nodes, **kw)
        if calls[0] == 3:
            raise _Killed
        return out

    monkeypatch.setattr(VectorStoreIndex, "insert_nodes", dying)
    with pytest.raises(_Killed):
        _rag(code, tmp_path / "s")
    assert len(_hashed(tmp_path / "s")) >= 10, "the groups that went in are on record"

    monkeypatch.setattr(VectorStoreIndex, "insert_nodes", inserts)
    inserts.calls.clear()
    rag = _rag(code, tmp_path / "s")   # what is there is kept
    rag.update(force=True)             # `lynx build`: the rest
    sent = set().union(*inserts.calls)
    assert len(sent) == 20, f"only what was missing is embedded again, not {len(sent)} files"
    assert rag._collection_count() == 30, "no file twice, none missing"
    assert len(_hashed(tmp_path / "s")) == 30
