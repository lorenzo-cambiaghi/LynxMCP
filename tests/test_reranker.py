"""Unit tests for the cross-encoder reranker.

We stub the heavy `CrossEncoder.predict` call so the tests run in
milliseconds, never touch the HF cache, and produce deterministic
ranking outputs. Each test asserts on shape/order/field preservation
without actually doing matrix multiplication.

Scenarios:
  1. Empty results → no-op, no model load
  2. Single result → no-op, no model load
  3. Multi-result rerank: order changes, scores updated, original_score preserved
  4. top_k cut: fewer results returned than fed in
  5. Lazy load: _model is None until first non-trivial rerank
  6. Long content gets truncated to max_input_chars (model never sees > N chars)
  7. predict() failure → fallback to original ordering, no crash
  8. All other result fields (file, file_path, symbol_name, ...) preserved
"""
from __future__ import annotations

import sys


def _fake_results(n: int, score_step: float = 0.01) -> list:
    """Build N synthetic results with monotonically decreasing scores.

    Each entry carries the full set of fields that CodebaseRAG.search
    actually returns, so we can assert that the reranker doesn't drop
    any of them on the way through.
    """
    return [
        {
            "id": f"chunk_{i}",
            "file": f"file_{i}.py",
            "file_path": f"/abs/path/file_{i}.py",
            "symbol_name": f"sym_{i}",
            "symbol_kind": "function",
            "language": "python",
            "start_line": i * 10,
            "end_line": i * 10 + 5,
            "content": f"Content of chunk {i}, useful for query matching.",
            "score": 0.1 - i * score_step,  # decreasing
        }
        for i in range(n)
    ]


class _FakeCrossEncoder:
    """Stub that returns a fixed score vector. Records calls for assertions."""

    def __init__(self, fixed_scores: list):
        self.fixed_scores = fixed_scores
        self.calls: list = []  # list of (query, [pairs]) tuples

    def predict(self, pairs):
        self.calls.append(("predict", list(pairs)))
        if len(pairs) != len(self.fixed_scores):
            raise AssertionError(
                f"stub mismatch: predict got {len(pairs)} pairs but stub holds "
                f"{len(self.fixed_scores)} scores"
            )
        return self.fixed_scores


class _FailingCrossEncoder:
    def predict(self, pairs):
        raise RuntimeError("simulated model failure")


def _install_fake_cross_encoder(rer, fake):
    """Bypass real model load by pre-setting `_model` on the Reranker."""
    rer._model = fake


def main() -> int:
    from lynx.reranker import Reranker, DEFAULT_RERANKER_MODEL, DEFAULT_MAX_INPUT_CHARS

    # ============================================================
    # 1. Empty results — no-op, no model load
    # ============================================================
    r = Reranker()
    out = r.rerank("any query", [])
    if out != []:
        print(f"[test] FAIL [1/8]: empty input should return []; got {out!r}")
        return 1
    if r._model is not None:
        print(f"[test] FAIL [1/8]: model should NOT be loaded on empty input")
        return 1
    print(f"[test] OK [1/8] empty input: no-op, model stays unloaded")

    # ============================================================
    # 2. Single result — no-op, no model load
    # ============================================================
    r = Reranker()
    one = _fake_results(1)
    out = r.rerank("any query", one)
    if out != one:
        print(f"[test] FAIL [2/8]: single result should pass through unchanged")
        return 2
    if r._model is not None:
        print(f"[test] FAIL [2/8]: model should NOT load for single result")
        return 2
    print(f"[test] OK [2/8] single result: no-op, model stays unloaded")

    # ============================================================
    # 3. Multi-result rerank: order changes, scores updated, original preserved
    # ============================================================
    r = Reranker()
    results = _fake_results(5)
    # Stub returns scores INVERTED relative to input — so the order
    # should completely flip after rerank.
    fake = _FakeCrossEncoder(fixed_scores=[1.0, 2.0, 3.0, 4.0, 5.0])
    _install_fake_cross_encoder(r, fake)
    out = r.rerank("query", results)
    if len(out) != 5:
        print(f"[test] FAIL [3/8]: expected 5 out, got {len(out)}")
        return 3
    # Highest stub-score is the LAST input chunk; should be on top now.
    if out[0]["id"] != "chunk_4":
        print(f"[test] FAIL [3/8]: top after rerank should be chunk_4, got {out[0]['id']}")
        return 3
    if out[-1]["id"] != "chunk_0":
        print(f"[test] FAIL [3/8]: bottom after rerank should be chunk_0, got {out[-1]['id']}")
        return 3
    # original_score preserved for diagnostics
    if "original_score" not in out[0]:
        print(f"[test] FAIL [3/8]: original_score missing from reranked result")
        return 3
    import math
    if not math.isclose(out[0]["original_score"], 0.06):  # chunk_4: 0.1 - 4 * 0.01
        print(f"[test] FAIL [3/8]: original_score wrong: {out[0]['original_score']}")
        return 3
    # New score is from the stub
    if out[0]["score"] != 5.0:
        print(f"[test] FAIL [3/8]: new score wrong: {out[0]['score']}")
        return 3
    # reranked flag set
    if not out[0].get("reranked"):
        print(f"[test] FAIL [3/8]: reranked flag missing")
        return 3
    # Stub was called exactly once
    if len(fake.calls) != 1:
        print(f"[test] FAIL [3/8]: predict called {len(fake.calls)} times, expected 1")
        return 3
    print(f"[test] OK [3/8] rerank reorders, updates score, preserves original_score, sets reranked")

    # ============================================================
    # 4. top_k cut: fewer results returned than fed in
    # ============================================================
    r = Reranker()
    results = _fake_results(10)
    fake = _FakeCrossEncoder(fixed_scores=[float(i) for i in range(10)])
    _install_fake_cross_encoder(r, fake)
    out = r.rerank("q", results, top_k=3)
    if len(out) != 3:
        print(f"[test] FAIL [4/8]: top_k=3 should return 3, got {len(out)}")
        return 4
    # Top 3 by stub score = indices 9, 8, 7
    if [o["id"] for o in out] != ["chunk_9", "chunk_8", "chunk_7"]:
        print(f"[test] FAIL [4/8]: top_k cut wrong: {[o['id'] for o in out]}")
        return 4
    print(f"[test] OK [4/8] top_k cut: only 3 best returned")

    # ============================================================
    # 5. Lazy load: _model stays None until non-trivial rerank
    # ============================================================
    r = Reranker()
    # Empty + single: no load
    r.rerank("q", [])
    r.rerank("q", _fake_results(1))
    if r._model is not None:
        print(f"[test] FAIL [5/8]: model loaded prematurely after no-op calls")
        return 5
    # Non-trivial: SHOULD attempt to load (we stub it so the real
    # cross-encoder download doesn't run in tests).
    fake = _FakeCrossEncoder(fixed_scores=[0.5, 0.4])
    _install_fake_cross_encoder(r, fake)
    r.rerank("q", _fake_results(2))
    # After our manual install + rerank: _model is set
    if r._model is None:
        print(f"[test] FAIL [5/8]: _model should be set after rerank")
        return 5
    print(f"[test] OK [5/8] lazy load: _model None until non-trivial rerank")

    # ============================================================
    # 6. Long content truncated to max_input_chars
    # ============================================================
    r = Reranker(max_input_chars=200)
    results = [{
        "id": "long", "file": "x.py", "file_path": "/x.py",
        "symbol_name": "s", "symbol_kind": "function", "language": "python",
        "start_line": 1, "end_line": 50, "score": 0.5,
        "content": "A" * 10_000,  # 10k chars
    }, {
        "id": "short", "file": "y.py", "file_path": "/y.py",
        "symbol_name": "s", "symbol_kind": "function", "language": "python",
        "start_line": 1, "end_line": 50, "score": 0.4,
        "content": "Short content.",
    }]
    fake = _FakeCrossEncoder(fixed_scores=[0.8, 0.2])
    _install_fake_cross_encoder(r, fake)
    r.rerank("query", results)
    # Inspect what the stub saw: the long doc text passed to predict
    # must be <= 200 chars
    pairs = fake.calls[0][1]
    long_doc_text = pairs[0][1]  # (query, doc) → take doc
    if len(long_doc_text) > 200:
        print(f"[test] FAIL [6/8]: long content not truncated; got {len(long_doc_text)} chars")
        return 6
    print(f"[test] OK [6/8] long content truncated to {len(long_doc_text)} chars (max=200)")

    # ============================================================
    # 7. predict() failure → fallback to original ordering, no crash
    # ============================================================
    r = Reranker()
    results = _fake_results(5)
    _install_fake_cross_encoder(r, _FailingCrossEncoder())
    out = r.rerank("query", results)
    # Should NOT raise, and should preserve the original ordering+scores
    if [o["id"] for o in out] != [f"chunk_{i}" for i in range(5)]:
        print(f"[test] FAIL [7/8]: failure fallback didn't preserve order")
        return 7
    # And NO "reranked" / "original_score" marker (because we couldn't actually rerank)
    if out[0].get("reranked"):
        print(f"[test] FAIL [7/8]: failure case wrongly marked as reranked")
        return 7
    # top_k still respected on the failure path
    out2 = r.rerank("query", results, top_k=2)
    if len(out2) != 2:
        print(f"[test] FAIL [7/8]: failure path ignored top_k; got {len(out2)} results")
        return 7
    print(f"[test] OK [7/8] predict failure: graceful fallback, top_k still honored")

    # ============================================================
    # 8. All metadata fields preserved
    # ============================================================
    r = Reranker()
    results = _fake_results(3)
    fake = _FakeCrossEncoder(fixed_scores=[0.5, 0.3, 0.7])
    _install_fake_cross_encoder(r, fake)
    out = r.rerank("q", results)
    expected_fields = {
        "id", "file", "file_path", "symbol_name", "symbol_kind",
        "language", "start_line", "end_line", "content", "score",
        "original_score", "reranked",
    }
    missing = expected_fields - set(out[0].keys())
    if missing:
        print(f"[test] FAIL [8/8]: fields missing after rerank: {missing}")
        return 8
    # No NEW unexpected fields beyond the 2 we add
    extras = set(out[0].keys()) - expected_fields
    if extras:
        print(f"[test] FAIL [8/8]: unexpected new fields: {extras}")
        return 8
    print(f"[test] OK [8/8] metadata: all original fields preserved + 2 new (original_score, reranked)")

    print("\n[test] === SUCCESS: reranker works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
