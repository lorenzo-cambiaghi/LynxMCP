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
                    else:
                        rag.update_file(path)
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

        return {
            "name": self.name,
            "type": self.type_name,
            "path": str(self.source_config.get("path", "")),
            "chunk_count": chunk_count,
            "last_commit": self.rag.metadata.get("last_commit"),
            "last_update": self.rag.metadata.get("last_update"),
            "drift_severity": drift["severity"] if drift else None,
        }

    def drift_status_text(self) -> str:
        return self.rag.drift_status_text()

    def needs_update(self) -> bool:
        """Forwarded for the `get_rag_status` global tool."""
        return self.rag.needs_update()
