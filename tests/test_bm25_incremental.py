"""Incremental BM25 cache must equal a cold rebuild.

The watcher path now refreshes only the edited file's BM25 entries instead of
dropping the whole index and re-reading + re-tokenizing the entire collection
from ChromaDB on the next search. These tests pin the invariant: after an
incremental update/removal, the cache (and therefore every BM25 score) is
identical to cold-loading the equivalent final corpus from the store.

Model-free: we use `CodebaseRAG.__new__` to skip the HuggingFace embedding load
and a fake Chroma collection, so this runs fast under pytest.
"""
from __future__ import annotations

import os

from llama_index.core.schema import TextNode

from lynx.rag_manager import CodebaseRAG


class _FakeCollection:
    def __init__(self, docs):
        # docs: list of (id, content, meta_dict)
        self._docs = list(docs)

    def get(self, include=None):
        return {
            "ids": [d[0] for d in self._docs],
            "documents": [d[1] for d in self._docs],
            "metadatas": [d[2] for d in self._docs],
        }


class _FakeVS:
    def __init__(self, coll):
        self._collection = coll


def _meta(fp: str, sym: str) -> dict:
    return {
        "file_name": os.path.basename(fp),
        "file_path": fp,
        "symbol_name": sym,
        "symbol_kind": "function_definition",
        "language": "python",
        "start_line": 1,
        "end_line": 3,
    }


def _new_rag(store_docs) -> CodebaseRAG:
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag._bm25_docs = {}
    rag._bm25_meta = {}
    rag._bm25_okapi = None
    rag._bm25_doc_ids = []
    rag.vector_store = _FakeVS(_FakeCollection(store_docs))
    return rag


def _node(cid: str, text: str, fp: str, sym: str) -> TextNode:
    n = TextNode(text=text, metadata=_meta(fp, sym))
    n.id_ = cid
    return n


def _scores(rag: CodebaseRAG, query: str):
    """Sorted (id, rounded score) pairs for a BM25 lookup — order-independent."""
    rag._ensure_bm25()
    hits = rag._bm25_lookup(query, top_n=50)
    return sorted((h["id"], round(h["score"], 6)) for h in hits)


FP_A, FP_B = os.path.normpath("/repo/a.py"), os.path.normpath("/repo/b.py")

INITIAL = [
    ("a1", "def alpha(): return compute_total(items)", _meta(FP_A, "alpha")),
    ("b1", "def beta(): return parse_input(raw)", _meta(FP_B, "beta")),
]


def test_cold_load_populates_and_searches():
    # A 4-doc corpus so BM25 IDF is positive for a term in a single doc (on a
    # 2-doc corpus a term in 1 doc gets IDF≈0 — a BM25 small-corpus quirk).
    docs = [
        ("a1", "def alpha(): return compute_total(items)", _meta(FP_A, "alpha")),
        ("b1", "def beta(): return parse_input(raw)", _meta(FP_B, "beta")),
        ("c1", "def gamma(): return render(view)", _meta("/repo/c.py", "gamma")),
        ("d1", "def delta(): return persist(record)", _meta("/repo/d.py", "delta")),
    ]
    rag = _new_rag(docs)
    rag._ensure_bm25()
    assert set(rag._bm25_docs) == {"a1", "b1", "c1", "d1"}
    assert rag._bm25_okapi is not None
    hits = rag._bm25_lookup("compute total", top_n=10)
    assert hits and hits[0]["id"] == "a1"


def test_invalidate_clears_cache():
    rag = _new_rag(INITIAL)
    rag._ensure_bm25()
    assert rag._bm25_docs
    rag._invalidate_bm25()
    assert rag._bm25_docs == {} and rag._bm25_meta == {}
    assert rag._bm25_okapi is None and rag._bm25_doc_ids == []


def test_incremental_update_equals_cold_rebuild():
    rag = _new_rag(INITIAL)
    rag._ensure_bm25()  # warm

    # Edit a.py: its single chunk a1 is replaced by a2 with new content.
    nodes = [_node("a2", "def alpha_v2(): return compute_total(items) + extra(x)", FP_A, "alpha_v2")]
    rag._bm25_apply_file_update(FP_A, nodes)

    final = [
        ("a2", "def alpha_v2(): return compute_total(items) + extra(x)", _meta(FP_A, "alpha_v2")),
        ("b1", "def beta(): return parse_input(raw)", _meta(FP_B, "beta")),
    ]
    cold = _new_rag(final)
    cold._ensure_bm25()

    assert rag._bm25_docs == cold._bm25_docs
    assert rag._bm25_meta == cold._bm25_meta
    for q in ("compute total", "extra", "parse input", "alpha", "beta"):
        assert _scores(rag, q) == _scores(cold, q), f"score mismatch for {q!r}"


def test_incremental_multichunk_file_replacement():
    """A file whose chunk count changes (1 → 2) still matches a cold rebuild."""
    rag = _new_rag(INITIAL)
    rag._ensure_bm25()
    nodes = [
        _node("a2", "def alpha(): return compute_total(items)", FP_A, "alpha"),
        _node("a3", "def gamma(): return aggregate(values)", FP_A, "gamma"),
    ]
    rag._bm25_apply_file_update(FP_A, nodes)

    final = [
        ("a2", "def alpha(): return compute_total(items)", _meta(FP_A, "alpha")),
        ("a3", "def gamma(): return aggregate(values)", _meta(FP_A, "gamma")),
        ("b1", "def beta(): return parse_input(raw)", _meta(FP_B, "beta")),
    ]
    cold = _new_rag(final)
    cold._ensure_bm25()
    assert rag._bm25_docs == cold._bm25_docs
    assert rag._bm25_meta == cold._bm25_meta
    for q in ("aggregate values", "compute total", "parse input"):
        assert _scores(rag, q) == _scores(cold, q), f"score mismatch for {q!r}"


def test_incremental_removal_equals_cold_rebuild():
    rag = _new_rag(INITIAL)
    rag._ensure_bm25()
    rag._bm25_apply_file_removal(FP_A)

    cold = _new_rag([("b1", "def beta(): return parse_input(raw)", _meta(FP_B, "beta"))])
    cold._ensure_bm25()
    assert rag._bm25_docs == cold._bm25_docs
    assert set(rag._bm25_docs) == {"b1"}
    for q in ("parse input", "compute total"):
        assert _scores(rag, q) == _scores(cold, q)


def test_update_on_cold_cache_is_noop():
    """If the cache was never warmed, an incremental update does nothing — the
    next search cold-loads from the store (which already has the new chunks)."""
    rag = _new_rag(INITIAL)
    # No _ensure_bm25() yet → cache is cold.
    rag._bm25_apply_file_update(FP_A, [_node("a2", "whatever", FP_A, "x")])
    assert rag._bm25_docs == {}  # untouched; stays cold
