"""Cross-source reranking in `SourceManager.search_all`.

When the cross-encoder reranker is enabled, search_all reorders the MERGED
candidate pool by content relevance (so the final cross-source order isn't just
RRF-by-position). When disabled, it stays pure RRF (unchanged default).

Model-free: `SourceManager.__new__` + fake backends + a fake reranker, so no
embedding model and no cross-encoder download.
"""
from __future__ import annotations

from types import SimpleNamespace

from lynx.source_manager import SourceManager


class _FakeBackend:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5, **kw):
        return [dict(h) for h in self._hits[:top_k]]


class _FakeReranker:
    """Deterministic reranker: orders by content length desc, descending score."""
    def __init__(self, raises=False):
        self.raises = raises
        self.calls = []

    def rerank(self, query, results, top_k=None):
        if self.raises:
            raise RuntimeError("boom")
        self.calls.append((query, len(results)))
        ordered = sorted(results, key=lambda r: len(r.get("content", "")), reverse=True)
        out = []
        for i, r in enumerate(ordered):
            r = dict(r)
            r["original_score"] = r.get("score")
            r["score"] = float(100 - i)
            r["reranked"] = True
            out.append(r)
        return out[:top_k] if top_k is not None else out


def _mgr(*, enabled, backends, reranker=None):
    mgr = SourceManager.__new__(SourceManager)
    mgr.config = SimpleNamespace(search=SimpleNamespace(
        reranker=SimpleNamespace(enabled=enabled, model_name="fake"),
        candidate_pool_size=30,
        rrf_k=60,
    ))
    mgr.backends = backends
    mgr._reranker = reranker  # pre-set so _get_reranker won't import the real one
    return mgr


def _backends():
    return {
        "s1": _FakeBackend([
            {"id": "x1", "content": "aaaa", "file": "a.py", "file_path": "/a.py"},
            {"id": "x2", "content": "a", "file": "a.py", "file_path": "/a.py"},
        ]),
        "s2": _FakeBackend([
            {"id": "y1", "content": "aaaaaaaa", "file": "b.py", "file_path": "/b.py"},
        ]),
    }


def test_search_all_reranks_merged_pool_when_enabled():
    rr = _FakeReranker()
    mgr = _mgr(enabled=True, backends=_backends(), reranker=rr)
    out = mgr.search_all("q", top_k=2)

    # Reranker saw the deduped union of all source pools (3 hits).
    assert rr.calls == [("q", 3)]
    # Order follows the reranker (by content length): y1 (s2) then x1 (s1).
    assert [h["id"] for h in out] == ["y1", "x1"]
    assert out[0]["source"] == "s2" and out[1]["source"] == "s1"
    # Cross-source order really interleaves sources (not grouped per source).
    assert {h["source"] for h in out} == {"s1", "s2"}
    # Internal fusion key never leaks to callers.
    assert all("_fusion_id" not in h for h in out)


def test_search_all_uses_rrf_when_reranker_disabled():
    mgr = _mgr(enabled=False, backends=_backends())
    out = mgr.search_all("q", top_k=3)
    assert len(out) == 3
    # RRF scores are small reciprocal-rank sums, not the reranker's 0..100 scale.
    assert all(0.0 < h["score"] < 1.0 for h in out)
    assert all("_fusion_id" not in h for h in out)
    assert all(not h.get("reranked") for h in out)


def test_search_all_falls_back_to_rrf_on_rerank_error():
    rr = _FakeReranker(raises=True)
    mgr = _mgr(enabled=True, backends=_backends(), reranker=rr)
    out = mgr.search_all("q", top_k=3)
    # Did not blow up; produced RRF results instead.
    assert len(out) == 3
    assert all(0.0 < h["score"] < 1.0 for h in out)
    assert all("_fusion_id" not in h for h in out)
