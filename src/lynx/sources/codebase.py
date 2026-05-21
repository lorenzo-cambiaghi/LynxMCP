"""Codebase source backend.

Wraps the existing `CodebaseRAG` (hybrid retrieval, BM25 cache, drift detection,
file watcher, filters, deep_search) so it fits the `SourceBackend` interface.
This keeps the proven retrieval pipeline intact while exposing it through the
multi-source dispatcher.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from .base import SourceBackend


class CodebaseBackend(SourceBackend):
    """Indexes a directory of source files (code + markdown + config text)."""

    type_name = "codebase"

    def __init__(
        self,
        name: str,
        source_config: dict,
        shared_config: Any,
        storage_dir: Path,
    ):
        super().__init__(name, source_config, shared_config, storage_dir)

        # Imported here (not at module level) to keep the `sources` package
        # importable without pulling in the heavy llama-index stack until a
        # backend is actually instantiated.
        from ..rag_manager import CodebaseRAG

        self.rag = CodebaseRAG(
            codebase_path=str(source_config["path"]),
            rag_storage_path=str(self.storage_dir),
            supported_extensions=source_config["supported_extensions"],
            embedding_model_name=shared_config.embedding.model_name,
            # Source name doubles as the ChromaDB collection name so the data
            # is easy to identify with external Chroma tools.
            collection_name=name,
            search_mode=shared_config.search.mode,
            rrf_k=shared_config.search.rrf_k,
            candidate_pool_size=shared_config.search.candidate_pool_size,
        )

        # Watcher state — populated lazily by start_watcher().
        self._observer = None

        # Opt-in graph layer (call graph + import graph + analyzer).
        # Disabled by default; activated via `graph: { enabled: true }` in
        # the source's config. When enabled, every search/update/watcher
        # event also keeps the graph in sync.
        self.graph = None
        graph_cfg = source_config.get("graph") or {}
        if graph_cfg.get("enabled"):
            from ..graph import GraphLayer
            self.graph = GraphLayer(
                storage_dir=self.storage_dir / "graph",
                codebase_path=source_config["path"],
                supported_extensions=source_config["supported_extensions"],
                ignored_path_fragments=source_config.get("ignored_path_fragments") or [],
            )

    # ------------------------------------------------------------------
    # Search dispatch (thin delegation to the underlying CodebaseRAG)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
        **_ignored,
    ) -> list[dict]:
        return self.rag.search(
            query,
            top_k=top_k,
            file_glob=file_glob,
            extensions=extensions,
            path_contains=path_contains,
        )

    def deep_search(
        self,
        queries: list[str],
        top_k: int = 5,
        *,
        mode=None,
        file_glob=None,
        extensions=None,
        path_contains=None,
        min_score=None,
        min_results=None,
        return_all_variants: bool = False,
        **_ignored,
    ) -> dict:
        return self.rag.deep_search(
            queries=queries,
            top_k=top_k,
            mode=mode,
            file_glob=file_glob,
            extensions=extensions,
            path_contains=path_contains,
            min_score=min_score,
            min_results=min_results,
            return_all_variants=return_all_variants,
            score_thresholds=self.shared.search.deep.score_thresholds,
        )

    def update(self, force: bool = False) -> None:
        self.rag.update(force=force)
        # Keep the graph layer in sync with the same trigger. We do this
        # after the RAG update so a partial failure in the graph layer
        # doesn't leave the vector index half-built.
        if self.graph is not None:
            try:
                self.graph.rebuild(force=force)
            except Exception as e:
                # Failures here must not break search — log and move on.
                print(
                    f"[graph:{self.name}] rebuild failed (search still works): {e}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def start_watcher(self) -> None:
        """Start a watchdog Observer over this source's `path`.

        No-op when `watcher.enabled` is False in source_config. Idempotent:
        calling twice does not start two observers.
        """
        watcher_cfg = self.source_config.get("watcher") or {}
        if not watcher_cfg.get("enabled", True):
            return
        if self._observer is not None:
            return

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        debounce_seconds = float(watcher_cfg.get("debounce_seconds", 2.0))
        ignored_fragments = tuple(self.source_config.get("ignored_path_fragments") or ())
        rag = self.rag
        graph = self.graph
        source_name = self.name

        def _is_ignored(path: str) -> bool:
            return any(frag in path for frag in ignored_fragments)

        class _DebouncedHandler(FileSystemEventHandler):
            def __init__(self):
                super().__init__()
                self._pending: dict = {}
                self._lock = threading.Lock()

            def _flush(self, path: str, action: str):
                try:
                    if action == "delete":
                        rag.remove_file(path)
                        if graph is not None:
                            graph.remove_file(path)
                    else:
                        rag.update_file(path)
                        if graph is not None:
                            graph.update_file(path)
                except Exception as e:
                    print(
                        f"[watcher:{source_name}] flush failed for {path}: {e}",
                        file=sys.stderr,
                    )
                finally:
                    with self._lock:
                        self._pending.pop(path, None)

            def _schedule(self, path: str, action: str):
                if not path or _is_ignored(path):
                    return
                if action != "delete" and not rag.is_supported(path):
                    return
                with self._lock:
                    existing = self._pending.get(path)
                    if existing is not None:
                        existing.cancel()
                    t = threading.Timer(
                        debounce_seconds, self._flush, args=(path, action)
                    )
                    t.daemon = True
                    self._pending[path] = t
                    t.start()

            def on_modified(self, event):
                if not event.is_directory:
                    self._schedule(event.src_path, "update")

            def on_created(self, event):
                if not event.is_directory:
                    self._schedule(event.src_path, "update")

            def on_deleted(self, event):
                if not event.is_directory:
                    self._schedule(event.src_path, "delete")

            def on_moved(self, event):
                if event.is_directory:
                    return
                self._schedule(event.src_path, "delete")
                dest = getattr(event, "dest_path", None)
                if dest:
                    self._schedule(dest, "update")

        observer = Observer()
        observer.schedule(
            _DebouncedHandler(), str(self.source_config["path"]), recursive=True
        )
        observer.daemon = True
        observer.start()
        self._observer = observer
        print(
            f"[watcher:{source_name}] Watcher active on {self.source_config['path']}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        try:
            chunk_count = self.rag.vector_store._collection.count()
        except Exception:
            chunk_count = None

        drift = self.rag.check_config_drift()

        status = {
            "name": self.name,
            "type": self.type_name,
            "path": str(self.source_config.get("path", "")),
            "chunk_count": chunk_count,
            "last_commit": self.rag.metadata.get("last_commit"),
            "last_update": self.rag.metadata.get("last_update"),
            "drift_severity": drift["severity"] if drift else None,
        }
        if self.graph is not None:
            gstat = self.graph.status()
            status["graph"] = {
                "nodes": gstat["nodes"],
                "edges": gstat["edges"],
                "files_indexed": gstat["files_indexed"],
                "last_update": gstat["last_update"],
            }
        return status

    def drift_status_text(self) -> str:
        return self.rag.drift_status_text()

    def needs_update(self) -> bool:
        """Forwarded for the `get_rag_status` global tool."""
        return self.rag.needs_update()

    # ------------------------------------------------------------------
    # Graph layer pass-through (only meaningful when graph.enabled=true)
    # ------------------------------------------------------------------
    # The SourceManager / MCP server check `hasattr(backend, "graph") and
    # backend.graph is not None` before registering the graph tools, so
    # these methods are only invoked when the layer is active.

    def get_callers(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_callers as _q
        return _q(self.graph.graph, symbol, limit=limit)

    def get_callees(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_callees as _q
        return _q(self.graph.graph, symbol, limit=limit)

    def get_subclasses(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_subclasses as _q
        return _q(self.graph.graph, symbol, limit=limit)

    def get_superclasses(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_superclasses as _q
        return _q(self.graph.graph, symbol, limit=limit)

    def get_imports(self, file_or_symbol: str, limit: int = 100) -> list:
        from ..graph import get_imports as _q
        return _q(self.graph.graph, file_or_symbol, limit=limit)

    def get_neighbors(
        self,
        symbol: str,
        relation_filter: str | None = None,
        depth: int = 1,
        limit: int = 100,
    ) -> list:
        from ..graph import get_neighbors as _q
        return _q(self.graph.graph, symbol,
                  relation_filter=relation_filter, depth=depth, limit=limit)

    def shortest_path(self, source: str, target: str, max_hops: int = 8):
        from ..graph import shortest_path as _q
        return _q(self.graph.graph, source, target, max_hops=max_hops)

    def architectural_overview(self, top_n_gods: int = 10, min_community_size: int = 3) -> dict:
        from ..graph import god_nodes, communities
        return {
            "god_nodes": god_nodes(self.graph.graph, top_n=top_n_gods),
            "communities": communities(self.graph.graph, min_size=min_community_size),
            "status": self.graph.status(),
        }

    def surprising_connections(self, top_n: int = 5) -> list:
        from ..graph import surprising_connections as _q
        return _q(self.graph.graph, top_n=top_n)

    def graph_status(self) -> dict:
        return self.graph.status()
