"""Cross-encoder reranker for hybrid search results.

Hybrid RRF fusion is fast and works well on average, but it ranks based
only on the rank position of each chunk in the two retrievers (dense and
sparse) — it never *looks* at the chunk content vs the query. A
cross-encoder model takes (query, chunk_content) as a pair and emits a
relevance score that does inspect the content, fixing many "the top
result is technically relevant but not the best answer" failures of
pure RRF.

We use `cross-encoder/ms-marco-MiniLM-L-6-v2` by default (~80MB, ~50ms
per query on CPU for 30 candidate chunks). It's the standard small-and-
fast reranker; bigger ones improve quality marginally at 5-10× cost.

Design notes:
  - **Lazy model load.** The 80MB download + RAM allocation happens on
    the first `rerank()` call, not at `__init__`. Users with reranker
    disabled pay zero cost; users with it enabled but never querying
    pay zero cost too.
  - **Preserves all result fields.** We only modify `score` (and stash
    the original RRF score as `original_score` for debugging).
  - **Chunk truncation.** Cross-encoders have a hard token limit (~512
    for MiniLM). We feed the first `max_length` characters of the
    chunk; on dense scientific text this is roughly the same as feeding
    the first ~500 tokens.
  - **Lazy import** of `sentence_transformers` so importing
    `lynx.reranker` doesn't pull in torch when nobody asks for it.
"""
from __future__ import annotations

import sys
from typing import Optional


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# Default cross-encoder model. Trained on MS-MARCO (web Q&A), surprisingly
# strong on code search too because identifier-rich queries look like
# question fragments to the model. Swap via config if you want a bigger /
# domain-specific reranker.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chars (not tokens) to feed the cross-encoder. The MiniLM tokenizer
# rounds about 4 chars to 1 token, so 1600 chars ≈ 400 tokens, comfortably
# below the model's 512-token context window after the query is prepended.
DEFAULT_MAX_INPUT_CHARS = 1600


class Reranker:
    """Wraps a `CrossEncoder` and applies it to a list of search results.

    Stateless w.r.t. queries — instantiate once per source, call
    `rerank()` once per search. The underlying model is loaded the first
    time `rerank()` runs, not in `__init__`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ):
        self.model_name = model_name
        self.device = device
        self.max_input_chars = int(max_input_chars)
        # `_model` stays None until the first rerank() call. We don't
        # type-annotate it as CrossEncoder because the import is lazy.
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load the cross-encoder on demand.

        Raises ImportError with an actionable message if sentence-
        transformers isn't installed (shouldn't happen — it's a hard
        dependency declared in pyproject — but defensive coding never
        hurts on a feature that ships as `enabled=false` by default).
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Reranker requires 'sentence-transformers'. Install with "
                "`pip install sentence-transformers` (it should already be "
                "pulled in by llama-index-embeddings-huggingface)."
            ) from e
        _log(f"[reranker] loading model {self.model_name!r} (device={self.device}) — first use")
        self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, results: list, top_k: Optional[int] = None) -> list:
        """Rerank `results` by cross-encoder relevance and return top_k.

        Each result dict keeps every field (file, content, symbol_name,
        etc.). The reranker:
          - replaces `score` with the cross-encoder score (a float in
            roughly [-10, 10] for MS-MARCO models — NOT comparable to
            the RRF scale)
          - sets `original_score` to the previous value, so callers can
            tell whether a result moved up or down

        No-op fast paths (no model load) when:
          - `results` is empty
          - `len(results) <= 1` (nothing to reorder)

        With non-trivial `top_k <= len(results)`, we still rerank because
        the cross-encoder can change the ordering of the top results
        (which is the whole point).
        """
        n = len(results)
        if n == 0 or n == 1:
            return results

        # Loading can fail too (model not in HF cache + offline mode,
        # bad model name, torch OOM, etc.). Treat it the same as a
        # predict failure: log + return original ranking so search
        # still works.
        try:
            self._ensure_loaded()
        except Exception as e:
            _log(f"[reranker] model load failed ({e}); falling back to original ranking")
            return results[:top_k] if top_k is not None else results

        pairs = [(query, self._prepare_text(r)) for r in results]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            # Don't break search if the model fails. Log and keep the
            # original ranking — users still get usable results.
            _log(f"[reranker] predict failed ({e}); falling back to original ranking")
            return results[:top_k] if top_k is not None else results

        # Pair scores with results, sort desc, slice.
        scored = list(zip(scores, results))
        scored.sort(key=lambda t: float(t[0]), reverse=True)

        out = []
        for new_score, r in scored:
            r_out = dict(r)
            r_out["original_score"] = r.get("score")
            r_out["score"] = float(new_score)
            r_out["reranked"] = True
            out.append(r_out)
        return out[:top_k] if top_k is not None else out

    def _prepare_text(self, result: dict) -> str:
        """Extract and truncate the text fed to the cross-encoder.

        We prefer `content` (the chunk body) but fall back to `text` in
        case a caller passes a different shape. Truncation to
        `max_input_chars` keeps the (query, doc) pair under the model's
        token window even on chunks of arbitrary length.
        """
        text = result.get("content") or result.get("text") or ""
        if len(text) > self.max_input_chars:
            text = text[: self.max_input_chars]
        return text
