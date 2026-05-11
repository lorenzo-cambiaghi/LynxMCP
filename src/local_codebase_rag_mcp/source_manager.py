"""Multi-source dispatcher.

Owns N `SourceBackend` instances (one per entry in `config.sources`) and
exposes:

  - Per-source operations (`search`, `deep_search`, `update`) routed by name.
  - Cross-source operations (`search_all`, `deep_search_all`) that run every
    backend and fuse results via Reciprocal Rank Fusion.
  - Lifecycle: starts each backend's watcher (if applicable).

The MCP server layer auto-generates `search_<name>` / `deep_search_<name>`
tools for each source by iterating `manager.backends`. No tool is hardcoded
to a specific source name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .sources import SOURCE_BACKENDS, SourceBackend


class SourceManager:
    """Holds N backends and dispatches operations to them."""

    def __init__(self, config: Any):
        self.config = config
        self.backends: dict[str, SourceBackend] = {}

        storage_root = Path(config.storage_path)
        storage_root.mkdir(parents=True, exist_ok=True)

        for name, src_cfg in config.sources.items():
            type_name = src_cfg["type"]
            backend_cls = SOURCE_BACKENDS.get(type_name)
            if backend_cls is None:
                raise ValueError(
                    f"Unknown source type {type_name!r} for source {name!r}. "
                    f"Supported types: {list(SOURCE_BACKENDS.keys())}"
                )

            per_source_storage = storage_root / name
            self.backends[name] = backend_cls(
                name=name,
                source_config=src_cfg,
                shared_config=config,
                storage_dir=per_source_storage,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_watchers(self) -> None:
        """Start the watcher for every backend that supports one. Safe to
        call multiple times — each backend is idempotent."""
        for backend in self.backends.values():
            try:
                backend.start_watcher()
            except Exception as e:
                print(
                    f"[manager] failed to start watcher for {backend.name!r}: {e}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Per-source dispatch
    # ------------------------------------------------------------------

    def get(self, source: str) -> SourceBackend:
        if source not in self.backends:
            raise KeyError(
                f"Unknown source {source!r}. Available: {list(self.backends)}"
            )
        return self.backends[source]

    def search(self, source: str, query: str, top_k: int = 5, **kw) -> list[dict]:
        return self.get(source).search(query, top_k=top_k, **kw)

    def deep_search(
        self, source: str, queries: list[str], top_k: int = 5, **kw
    ) -> dict:
        return self.get(source).deep_search(queries, top_k=top_k, **kw)

    def update(self, source: str, force: bool = False) -> None:
        self.get(source).update(force=force)

    # ------------------------------------------------------------------
    # Cross-source operations
    # ------------------------------------------------------------------

    def search_all(self, query: str, top_k: int = 5, **kw) -> list[dict]:
        """Run `query` against every source and fuse the rankings via RRF.

        Each result is tagged with `source` so the AI can tell which source
        produced it. RRF constant `k` comes from the shared search config so
        the fusion is consistent with the intra-source hybrid fusion.
        """
        pool_size = max(top_k, self.config.search.candidate_pool_size)
        per_source_results = []
        for name, backend in self.backends.items():
            try:
                hits = backend.search(query, top_k=pool_size, **kw)
            except Exception as e:
                print(
                    f"[manager] search on source {name!r} failed: {e}",
                    file=sys.stderr,
                )
                continue
            # Tag each hit with its source so we can present it later AND so
            # ids don't collide across sources during fusion.
            for h in hits:
                h["source"] = name
                # Make the id source-scoped to avoid accidental cross-source
                # id collisions (a chunk id like "node-123" could exist in
                # multiple sources).
                h["_fusion_id"] = f"{name}::{h.get('id') or h.get('file_path', '')}"
            per_source_results.append(hits)

        return self._rrf_fuse(per_source_results, self.config.search.rrf_k, top_k)

    def deep_search_all(
        self,
        queries: list[str],
        top_k: int = 5,
        *,
        min_score=None,
        min_results=None,
        **kw,
    ) -> dict:
        """Multi-query + multi-source fallback.

        Tries each query variant in order across ALL sources (RRF-fused).
        First variant whose fused top-K passes the weakness threshold wins.
        Threshold uses the same mode-aware logic as deep_search; the mode for
        threshold lookup is `self.config.search.mode` since fusion is mode-
        independent.
        """
        if not isinstance(queries, (list, tuple)) or not queries:
            raise ValueError("queries must be a non-empty list of strings")

        thresholds = self.config.search.deep.score_thresholds
        # Cross-source fusion uses RRF, so use the hybrid threshold (RRF scale)
        # for weakness detection unless the caller overrode it explicitly.
        if min_score is not None:
            threshold = float(min_score)
        else:
            threshold = float(thresholds.get("hybrid", 0.012))
        eff_min_results = int(min_results) if min_results is not None else 2

        best_results: list[dict] = []
        best_top: float = float("-inf")
        winning_idx = None

        for idx, variant in enumerate(queries):
            results = self.search_all(variant, top_k=top_k, **kw)
            top_score = results[0]["score"] if results else float("-inf")
            if top_score > best_top:
                best_top = top_score
                best_results = results
            passes = len(results) >= eff_min_results and top_score >= threshold
            if passes:
                winning_idx = idx + 1
                return {
                    "results": results,
                    "winning_variant_index": winning_idx,
                    "variants_tried": idx + 1,
                    "all_weak": False,
                }

        return {
            "results": best_results,
            "winning_variant_index": None,
            "variants_tried": len(queries),
            "all_weak": True,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self, rankings: list[list[dict]], k: int, top_k: int
    ) -> list[dict]:
        """RRF fusion across source-tagged ranking lists.

        Identical math to the intra-source RRF but uses `_fusion_id` (which
        embeds the source name) as the key so the same chunk id appearing in
        two sources is treated as two distinct documents.
        """
        fused_scores: dict = {}
        item_by_id: dict = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                fid = item.get("_fusion_id")
                if not fid:
                    continue
                fused_scores[fid] = fused_scores.get(fid, 0.0) + 1.0 / (k + rank + 1)
                item_by_id.setdefault(fid, item)
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        out = []
        for fid in sorted_ids:
            item = dict(item_by_id[fid])
            item["score"] = fused_scores[fid]
            item.pop("_fusion_id", None)
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        return [backend.status() for backend in self.backends.values()]
