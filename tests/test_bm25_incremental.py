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
import threading

import pytest

from llama_index.core.schema import TextNode

from lynx.rag_manager import CodebaseRAG


class _FakeCollection:
    """Shaped like a chroma collection. Like SQLite under ChromaDB, one `get()`
    binds a variable per row it returns and refuses more than `max_vars`."""

    def __init__(self, docs, max_vars=None):
        # docs: list of (id, content, meta_dict)
        self._docs = list(docs)
        self.max_vars = max_vars
        self.gets = 0

    def count(self):
        return len(self._docs)

    def get(self, ids=None, where=None, limit=None, offset=None, include=None):
        self.gets += 1
        rows = self._docs
        if where:
            rows = [d for d in rows if all(d[2].get(k) == v for k, v in where.items())]
        start = offset or 0
        rows = rows[start:start + limit] if limit is not None else rows[start:]
        if self.max_vars is not None and len(rows) > self.max_vars:
            raise RuntimeError("Error executing plan: Internal error: error returned "
                               "from database: (code: 1) too many SQL variables")
        return {
            "ids": [d[0] for d in rows],
            "documents": [d[1] for d in rows],
            "metadatas": [d[2] for d in rows],
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
    rag._bm25_error = None
    rag._bm25_retry_at = 0.0
    # The cache is edited from the watcher thread and, on a follower, from the
    # refresh; the BM25 rebuild snapshots it under this lock before iterating.
    rag._write_lock = threading.RLock()
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


# ---------------------------------------------------------------------------
# A corpus past SQLite's variable limit, and a load that fails anyway
# ---------------------------------------------------------------------------

def _many(n: int) -> list:
    return [(f"c{i}", f"def handler_{i}(): return dispatch_{i}(event)",
             _meta(f"/repo/m{i % 40}.py", f"handler_{i}")) for i in range(n)]


def test_a_collection_past_the_sql_variable_limit_still_loads():
    # One get() for the whole collection binds a variable per row: past the
    # limit it raised, the lexical index stayed empty and every "hybrid"
    # search was dense-only. Here the limit is 250 and the corpus 1,200.
    docs = _many(1200)
    rag = _new_rag(docs)
    rag.vector_store._collection.max_vars = 250

    hits = rag._bm25_lookup("dispatch_777", top_n=3)

    assert len(rag._bm25_docs) == 1200
    assert rag.lexical_unavailable() is None
    assert hits and hits[0]["id"] == "c777"


def test_a_failed_load_is_reported_and_not_paid_for_on_every_query():
    rag = _new_rag(_many(50))
    rag.vector_store._collection.max_vars = 0          # every read fails

    assert rag._bm25_lookup("dispatch_7", top_n=3) == []
    first = rag.vector_store._collection.gets
    assert rag._bm25_lookup("dispatch_8", top_n=3) == []
    assert rag.vector_store._collection.gets == first, "retried on every query"
    assert "too many SQL variables" in rag.lexical_unavailable()

    rag.vector_store._collection.max_vars = None       # the index changes...
    rag._invalidate_bm25()
    assert rag._bm25_lookup("dispatch_7", top_n=3)[0]["id"] == "c7"
    assert rag.lexical_unavailable() is None


def _retrieving(rag):
    """Enough of a CodebaseRAG for `_retrieve_one`, with a canned dense half."""
    rag.reranker_config = None
    rag.candidate_pool_size = 30
    rag.rrf_k = 60
    rag.refresh_if_stale = lambda: None
    rag._dense_lookup = lambda query, n, embedding=None, files=None, **_: [
        {"id": "c3", "file": "m3.py", "file_path": "/repo/m3.py",
         "content": "def handler_3(): return dispatch_3(event)", "score": 0.7}]
    return rag


def test_hybrid_hits_say_when_their_lexical_half_was_missing():
    healthy = _retrieving(_new_rag(_many(50)))
    broken = _retrieving(_new_rag(_many(50)))
    broken.vector_store._collection.max_vars = 0

    assert {h["mode_effective"] for h in healthy._retrieve_one("dispatch_3", 3, "hybrid")} == {"hybrid"}
    assert {h["mode_effective"] for h in broken._retrieve_one("dispatch_3", 3, "hybrid")} == {"dense"}


def test_the_manager_names_each_source_answering_dense_only():
    from types import SimpleNamespace
    from lynx.source_manager import SourceManager

    broken = _new_rag(_many(10))
    broken.vector_store._collection.max_vars = 0
    broken._bm25_lookup("x", top_n=1)
    manager = SourceManager.__new__(SourceManager)
    manager.backends = {"good": SimpleNamespace(rag=_new_rag(_many(10))),
                        "bad": SimpleNamespace(rag=broken)}

    notice = manager.lexical_notice()

    assert "'bad'" in notice and "dense-only" in notice and "'good'" not in notice
    assert manager.lexical_notice(["good"]) == ""


# ---------------------------------------------------------------------------
# Filters choose the files before the ranking
# ---------------------------------------------------------------------------

TARGET = os.path.normpath("/repo/Shaders/SkinnedVoxelize.compute")


def _crowded():
    """Forty chunks that beat, on "kernel", the one file a filter will ask
    for; the rest of the corpus is about something else."""
    docs = [(f"c{i}", "kernel kernel kernel dispatch", _meta(f"/repo/k{i}.py", f"k{i}"))
            for i in range(40)]
    docs += [(f"f{i}", "alpha beta gamma", _meta(f"/repo/f{i}.py", f"f{i}"))
             for i in range(59)]
    docs.append(("target", "kernel helper", _meta(TARGET, "Rasterize")))
    return docs


def test_a_filter_picks_the_files_before_the_ranking_cuts_them():
    # Filtering the top 25 candidates afterwards answered "no results" for
    # files that were indexed but ranked 76th and 87th.
    rag = _retrieving(_new_rag(_crowded()))
    seen = {}

    def dense(query, n, embedding=None, files=None, **_):
        seen["files"] = files
        return []

    rag._dense_lookup = dense
    hits = rag._retrieve_one("kernel", 5, "hybrid", path_contains="SkinnedVoxelize")

    assert [h["id"] for h in hits] == ["target"]
    rule = seen["files"].filters[0]
    assert rule.operator.value == "in" and rule.value == [TARGET]


def test_a_filter_no_indexed_file_passes_asks_the_store_nothing():
    rag = _retrieving(_new_rag(_crowded()))
    rag._dense_lookup = lambda *a, **k: pytest.fail("queried the store")

    assert rag._retrieve_one("kernel", 5, "hybrid", path_contains="Nowhere") == []


def test_a_filtered_search_that_the_store_refuses_still_answers():
    # ChromaDB checks a filter's ids against the vector index this process
    # loaded; a store another process wrote to since can refuse with "Error
    # finding id". A follower catches up and retries; past that, the search
    # runs unfiltered on the wider pool and the filters cut it afterwards.
    rag = _new_rag(INITIAL)
    rag.collection_name, rag.follower = "src", True
    calls, reopened = [], []

    def retrieve(query, top_n, embedding, files):
        calls.append((top_n, files is not None))
        if files is not None:
            raise RuntimeError("Error executing plan: Internal error: Error finding id")
        return [{"id": "a1"}]

    rag._retrieve_dense = retrieve
    rag._reopen_store = lambda: reopened.append(1)

    hits = rag._dense_lookup("q", 5, files=object(), fallback_n=200)

    assert hits == [{"id": "a1"}]
    assert reopened == [1]
    assert calls == [(5, True), (5, True), (200, False)]


def test_a_search_during_the_warm_up_waits_for_it_instead_of_loading_twice():
    import time as _time

    rag = _new_rag(_many(10))
    collection = rag.vector_store._collection
    plain_get = collection.get

    def slow_get(*a, **k):
        _time.sleep(0.3)
        return plain_get(*a, **k)

    collection.get = slow_get
    threads = [threading.Thread(target=rag._ensure_bm25) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert collection.gets == 1
    assert len(rag._bm25_docs) == 10
