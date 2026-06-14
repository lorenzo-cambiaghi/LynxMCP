"""Multi-source dispatcher.

Owns N `SourceBackend` instances (one per entry in `config.sources`) and
exposes:

  - Per-source operations (`search`, `deep_search`, `update`) routed by name.
  - Cross-source operations (`search_all`, `deep_search_all`) that run every
    backend and fuse results via Reciprocal Rank Fusion.
  - Lifecycle: starts each backend's watcher (if applicable).

The MCP server layer exposes a fixed tool set (`search`, `deep_search`,
`graph_query`, ...) that routes here via a `source` argument. No tool is
hardcoded to a specific source name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .sources import SOURCE_BACKENDS, SourceBackend


class SourceManager:
    """Holds N backends and dispatches operations to them."""

    def __init__(self, config: Any, *, probe_integrity: bool = True):
        self.config = config
        self.backends: dict[str, SourceBackend] = {}
        # Sources we could NOT bring up (corrupt index, construction failure).
        # Kept separate from `backends` so one bad source never takes down the
        # whole manager — the UI/CLI list them as "corrupt" with a reset hint.
        # name -> {name, type, path, error, storage_dir, crashed}
        self.broken: dict[str, dict] = {}
        self._probe_integrity = probe_integrity

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

            # Pre-flight integrity probe (out-of-process): a corrupt or
            # version-incompatible Chroma index can SEGFAULT on open/count,
            # which no try/except can catch — it would kill the whole host
            # process. Probing in a child means a crash there only fails the
            # probe; we mark the source broken and never touch the bad index.
            if probe_integrity and type_name == "codebase":
                from .integrity import check_index
                result = check_index(per_source_storage, name)
                if result["status"] == "corrupt":
                    self._register_broken(
                        name, type_name, src_cfg, per_source_storage,
                        result["detail"], crashed=result.get("crashed", False),
                    )
                    continue

            try:
                self.backends[name] = backend_cls(
                    name=name,
                    source_config=src_cfg,
                    shared_config=config,
                    storage_dir=per_source_storage,
                )
            except Exception as e:
                # Any failure to bring a source up (corrupt index caught
                # in-process, bad path, etc.) is isolated: register it as
                # broken and keep the other sources alive.
                from .errors import CorruptIndexError
                detail = e.detail if isinstance(e, CorruptIndexError) else f"{type(e).__name__}: {e}"
                self._register_broken(
                    name, type_name, src_cfg, per_source_storage, detail,
                )

    def _register_broken(self, name, type_name, src_cfg, storage_dir, detail,
                         *, crashed: bool = False) -> None:
        self.broken[name] = {
            "name": name,
            "type": type_name,
            "path": str(src_cfg.get("path") or src_cfg.get("url") or ""),
            "error": detail,
            "storage_dir": str(storage_dir),
            "crashed": crashed,
        }
        print(
            f"[manager] source {name!r}: index is corrupt — {detail} "
            f"Run `lynx reset --source {name}` (or use the dashboard's Reset "
            f"button); the data is disposable and rebuilds from your files.",
            file=sys.stderr,
        )

    def reset_source(self, name: str, *, rebuild: bool = True) -> dict:
        """Wipe a source's storage dir and (optionally) rebuild from scratch.

        Works whether the source is currently healthy or registered broken.
        The vector index holds only derived embeddings, so wiping it is safe;
        a rebuild re-reads the files on disk. Returns the fresh status() dict
        (or a minimal dict when rebuild=False).

        Two paths, because a LIVE Chroma client keeps the HNSW segment files
        open and Windows then refuses to delete them (WinError 32):

          - healthy source + rebuild → rebuild *in place* via the backend's own
            force-update (Chroma drops & recreates the collection through its
            API; no rmtree, no fight with the open handle).
          - broken source, or wipe-without-rebuild → delete the storage dir.
            A broken source was never opened, so nothing holds its files.
        """
        import shutil

        if name not in self.config.sources:
            raise KeyError(
                f"Unknown source {name!r}. Available: {list(self.config.sources)}"
            )

        src_cfg = self.config.sources[name]
        backend = self.backends.get(name)

        # Healthy + rebuild: clean rebuild without touching the filesystem out
        # from under the live Chroma client. backend.reset() empties the store
        # through Chroma's API + clears the SHA cache, then reindexes — a true
        # wipe-and-rebuild with no rmtree (which Windows blocks on the open
        # HNSW handle).
        if backend is not None and rebuild:
            backend.reset()
            status = backend.status()
            status.setdefault("health", "ok")
            return status

        # Otherwise we delete the storage dir. Drop the backend and stop its
        # watcher first; a broken source has no live backend at all.
        self.backends.pop(name, None)
        if backend is not None:
            try:
                backend.stop_watcher()
            except Exception:
                pass
            # Best-effort release of the Chroma client's file handles before
            # the delete (matters only for the rare healthy + no-rebuild call).
            backend = None
            import gc
            gc.collect()
        self.broken.pop(name, None)

        storage_dir = Path(self.config.storage_path) / name
        if storage_dir.exists():
            self._rmtree_with_retry(storage_dir)

        if not rebuild:
            return {"name": name, "type": src_cfg["type"], "health": "reset"}

        backend_cls = SOURCE_BACKENDS[src_cfg["type"]]
        backend = backend_cls(
            name=name,
            source_config=src_cfg,
            shared_config=self.config,
            storage_dir=storage_dir,
        )
        backend.update(force=True)
        self.backends[name] = backend
        status = backend.status()
        status.setdefault("health", "ok")
        return status

    @staticmethod
    def _rmtree_with_retry(path: Path, attempts: int = 5) -> None:
        """rmtree that tolerates Windows' lazy handle release (WinError 32)."""
        import shutil
        import time
        for i in range(attempts):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                if i == attempts - 1:
                    raise
                time.sleep(0.3)

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

    def search_batch(self, source: str, queries, top_k: int = 5, **kw) -> list:
        """Batch search: one result list per query. The codebase backend
        embeds all queries in a single model call."""
        return self.get(source).search_batch(queries, top_k=top_k, **kw)

    def deep_search(
        self, source: str, queries: list[str], top_k: int = 5, **kw
    ) -> dict:
        return self.get(source).deep_search(queries, top_k=top_k, **kw)

    def update(self, source: str, force: bool = False) -> None:
        self.get(source).update(force=force)

    # ------------------------------------------------------------------
    # Graph layer pass-through (opt-in per source)
    # ------------------------------------------------------------------
    # All of these raise ValueError if the named source doesn't have the
    # graph layer enabled, so callers (the MCP server, the CLI) can fail
    # fast with a clear message instead of returning empty results.

    def _require_graph(self, source: str):
        backend = self.get(source)
        graph = getattr(backend, "graph", None)
        if graph is None:
            raise ValueError(
                f"source {source!r} does not have the graph layer enabled "
                f"(set `graph: {{ enabled: true }}` in its config)"
            )
        return backend

    def get_callers(self, source: str, symbol: str, limit: int = 50) -> list:
        return self._require_graph(source).get_callers(symbol, limit=limit)

    def get_callees(self, source: str, symbol: str, limit: int = 50) -> list:
        return self._require_graph(source).get_callees(symbol, limit=limit)

    def get_subclasses(self, source: str, symbol: str, limit: int = 50) -> list:
        return self._require_graph(source).get_subclasses(symbol, limit=limit)

    def get_superclasses(self, source: str, symbol: str, limit: int = 50) -> list:
        return self._require_graph(source).get_superclasses(symbol, limit=limit)

    def get_imports(self, source: str, file_or_symbol: str, limit: int = 100) -> list:
        return self._require_graph(source).get_imports(file_or_symbol, limit=limit)

    def get_neighbors(
        self,
        source: str,
        symbol: str,
        relation_filter: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> list:
        return self._require_graph(source).get_neighbors(
            symbol, relation_filter=relation_filter, depth=depth, limit=limit
        )

    def shortest_path(self, source: str, src_symbol: str, target_symbol: str, max_hops: int = 8):
        return self._require_graph(source).shortest_path(
            src_symbol, target_symbol, max_hops=max_hops
        )

    def architectural_overview(self, source: str, top_n_gods: int = 10, min_community_size: int = 3) -> dict:
        return self._require_graph(source).architectural_overview(
            top_n_gods=top_n_gods, min_community_size=min_community_size
        )

    def surprising_connections(self, source: str, top_n: int = 5) -> list:
        return self._require_graph(source).surprising_connections(top_n=top_n)

    def graph_status(self, source: str) -> dict:
        return self._require_graph(source).graph_status()

    # ------------------------------------------------------------------
    # Combined tools pass-through (codebase sources only; graph optional)
    # ------------------------------------------------------------------
    # Unlike the pure-graph methods above, these work for any `codebase`
    # source (with or without graph layer enabled). They raise ValueError
    # if invoked on a non-codebase source so the AI client gets a clear
    # error instead of silent garbage.

    def _require_codebase(self, source: str):
        backend = self.get(source)
        if backend.type_name != "codebase":
            raise ValueError(
                f"source {source!r} is type={backend.type_name!r}; combined "
                f"tools (find_definition, find_usages, find_tests_for, "
                f"find_similar) are only available on codebase sources"
            )
        return backend

    def find_definition(self, source: str, symbol: str, limit: int = 10) -> list:
        return self._require_codebase(source).find_definition(symbol, limit=limit)

    def find_usages(self, source: str, symbol: str, limit: int = 50) -> list:
        return self._require_codebase(source).find_usages(symbol, limit=limit)

    def find_tests_for(
        self, source: str, symbol: str, limit: int = 20,
        test_path_pattern: str | None = None,
    ) -> list:
        return self._require_codebase(source).find_tests_for(
            symbol, limit=limit, test_path_pattern=test_path_pattern,
        )

    def find_similar(self, source: str, snippet: str, top_k: int = 10) -> list:
        return self._require_codebase(source).find_similar(snippet, top_k=top_k)

    def search_diff(
        self, source: str, query: str,
        base: str | None = None, top_k: int = 8, **kw,
    ) -> dict:
        """Pass-through to backend.search_diff (codebase only).

        Returns a dict (NOT a list) — see CodebaseBackend.search_diff
        for the shape. Raises ValueError on non-codebase sources.
        """
        return self._require_codebase(source).search_diff(
            query, base=base, top_k=top_k, **kw,
        )

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
        rows: list[dict] = []
        for backend in self.backends.values():
            st = backend.status()
            st.setdefault("health", "ok")
            rows.append(st)
        # Append sources that failed to load so the UI/CLI can surface them
        # (with a reset affordance) instead of silently hiding them.
        for info in self.broken.values():
            rows.append({
                "name": info["name"],
                "type": info["type"],
                "path": info["path"],
                "health": "corrupt",
                "error": info["error"],
                "chunk_count": None,
                "last_commit": None,
                "last_update": None,
                "drift_severity": None,
            })
        return rows
