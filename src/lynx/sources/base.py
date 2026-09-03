"""Abstract base class for source backends.

A `SourceBackend` represents one indexable source (codebase, web doc set,
PDF set, ...). It owns its own ChromaDB collection in a per-source storage
subdir, exposes `search()` / `deep_search()` that may be type-specialized,
and (optionally) runs a watcher to keep its index live.

The same retrieval primitives (dense + BM25 + RRF fusion, drift detection,
filters, deep_search ladder) are available to every subclass via the
underlying `CodebaseRAG` instance. Subclasses only need to override the
methods whose behavior genuinely differs by type.
"""
from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SourceBackend(ABC):
    """One indexable source. Subclass per type (codebase, webdoc, pdf, ...)."""

    # Subclasses MUST set this. Used as the discriminator in config and the
    # key in `SOURCE_BACKENDS`.
    type_name: str = ""

    # Several Lynx processes may read one store; only one may write it. A
    # backend that has not claimed its store is a follower: no watcher, no
    # writes, and reads refreshed from the owner's. Backends that never call
    # `_claim_store` keep the old single-process behaviour by default.
    is_owner: bool = True

    # How often a follower re-checks whether the owner has gone away.
    _PROMOTION_INTERVAL_SEC = 5.0

    def __init__(
        self,
        name: str,
        source_config: dict,
        shared_config: Any,
        storage_dir: Path,
    ):
        """Construct the backend.

        Args:
            name: The source's identifier from `config.sources[name]`.
                  Used as the ChromaDB collection name and as the value the
                  MCP tools accept in their `source` argument.
            source_config: The per-source dict from `config.sources[name]`,
                  type-specific fields included.
            shared_config: The top-level `Config` object (embedding model,
                  search settings, storage root, etc.) shared across sources.
            storage_dir: Pre-computed `<storage_path>/<name>/`. The backend
                  owns this directory exclusively.
        """
        self.name = name
        self.source_config = source_config
        self.shared = shared_config
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._last_promotion_check = 0.0

    # ------------------------------------------------------------------
    # Store ownership (shared by every backend that writes an index)
    # ------------------------------------------------------------------

    def _claim_store(self) -> bool:
        """Try to become the process that writes this source's index.

        Call before building the inner engine, so `follower=not is_owner` can
        be passed to it. Sets `self.is_owner` and returns it.
        """
        from .. import ownership

        self.is_owner = ownership.claim(self.storage_dir, on_lost=self._stand_down)
        if self.is_owner:
            import atexit
            atexit.register(ownership.release, self.storage_dir)
        else:
            print(
                f"[owner] source {self.name!r}: {ownership.describe(self.storage_dir)}. "
                f"Searching it here; indexing stays with that process.",
                file=sys.stderr,
            )
        return self.is_owner

    def _release_store(self) -> None:
        """Give the index back, and go back to following it."""
        from .. import ownership

        if not self.is_owner:
            return
        ownership.release(self.storage_dir)
        self.is_owner = False
        self._set_follower(True)

    def _stand_down(self) -> None:
        """Stop being the writer, because the claim was taken from us.

        Reached when this process was stopped long enough for its heartbeat to
        go stale (a machine asleep, most often) and another session took the
        index over. Keeping our watcher running would put two writers on one
        store, which is the single thing ownership exists to prevent.
        """
        self.is_owner = False
        try:
            self.stop_watcher()
        except Exception as e:
            print(f"[owner] source {self.name!r}: could not stop the watcher: {e}",
                  file=sys.stderr)
        self.is_owner = False
        self._set_follower(True)
        self._last_promotion_check = 0.0    # re-check the new owner on next read

    def _set_follower(self, follower: bool) -> None:
        """Tell the inner engines whether they are reading someone else's store."""
        rag = getattr(self, "rag", None)
        if rag is not None:
            rag.follower = follower
        graph = getattr(self, "graph", None)
        if graph is not None:
            graph.follower = follower

    def _claim_if_abandoned(self) -> bool:
        """Take the index over when the process that owned it has exited.

        Without this, closing the window that happened to start first would
        leave every other session reading an index nobody keeps up to date:
        searches would keep working and silently go stale. One small file read
        plus a liveness check, throttled on top.
        """
        if self.is_owner:
            return False
        now = time.monotonic()
        if now - self._last_promotion_check < self._PROMOTION_INTERVAL_SEC:
            return False
        self._last_promotion_check = now

        from .. import ownership
        if ownership.owner_of(self.storage_dir) is not None:
            return False
        if not ownership.claim(self.storage_dir, on_lost=self._stand_down):
            return False                      # another session got there first

        # Catch up on whatever the departed owner wrote before we stop
        # following it: once `follower` is cleared, nothing checks again.
        self._force_refresh()
        import atexit
        atexit.register(ownership.release, self.storage_dir)
        self.is_owner = True
        self._set_follower(False)
        print(
            f"[owner] source {self.name!r}: the process that owned this index "
            f"has exited; taking over indexing here.",
            file=sys.stderr,
        )
        # Order matters: the watcher first, so nothing that happens from now
        # on is missed, then a scan for what happened before it started.
        self.start_watcher()
        self._catch_up_in_background()
        return True

    def catch_up(self) -> None:
        """Index whatever changed while nobody was watching. Owner only."""
        self.update(force=True)

    def _catch_up_in_background(self) -> None:
        """Run `catch_up` off the calling thread.

        Between the previous owner exiting and this session taking over, no
        watcher was listening, and a watcher only ever reports what happens
        after it starts. Without this, a file edited in that window stays
        invisible until someone runs a build by hand. It runs in the
        background because the caller is a search: correctness should not cost
        the user a stalled query.
        """
        def run():
            try:
                self.catch_up()
            except Exception as e:
                print(f"[owner] source {self.name!r}: catch-up scan failed "
                      f"(searches still work): {e}", file=sys.stderr)
            else:
                print(f"[owner] source {self.name!r}: caught up with the "
                      f"changes made while nobody was watching.",
                      file=sys.stderr)

        threading.Thread(target=run, name=f"lynx-catchup-{self.name}",
                         daemon=True).start()

    def _force_refresh(self) -> None:
        """Re-read the store now, ignoring the staleness throttle."""
        for engine in (getattr(self, "rag", None), getattr(self, "graph", None)):
            if engine is None:
                continue
            try:
                engine._last_staleness_check = 0.0
                engine.refresh_if_stale()
            except Exception as e:      # a stale read must never fail a query
                print(f"[owner] source {self.name!r}: refresh failed: {e}",
                      file=sys.stderr)

    def _before_read(self) -> None:
        """Run before ANY read on this source.

        Two jobs, both cheap and throttled: notice that the owner has written
        (so a follower does not answer from the snapshot it opened with), and
        notice that the owner is gone (so indexing does not simply stop).
        Every read entry point calls this; missing one is how a tool ends up
        quietly serving stale results while its neighbours are current.
        """
        if self.is_owner:
            return
        self._claim_if_abandoned()
        rag = getattr(self, "rag", None)
        if rag is not None:
            try:
                rag.refresh_if_stale()
            except Exception as e:
                print(f"[owner] source {self.name!r}: refresh failed: {e}",
                      file=sys.stderr)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[dict]:
        """Single-query search over this source. Returns a list of result dicts.

        Each dict has at minimum: `file`, `file_path`, `content`, `score`.
        Subclasses MAY include type-specific extra fields (e.g. `url`,
        `section`, `page`) — the formatter on the MCP side should tolerate
        their absence.
        """

    def search_batch(self, queries, top_k: int = 5, **kwargs) -> list:
        """Search multiple queries, returning one result list per query.

        Default: loop `search()` (correct, but no speed-up). The codebase
        backend overrides this to batch the query embedding into one model
        call — the real win for N > 1 (e.g. an external agent/script fanning
        a question across many rows of another data source)."""
        return [self.search(q, top_k=top_k, **kwargs) for q in queries]

    @abstractmethod
    def deep_search(
        self,
        queries: list[str],
        top_k: int = 5,
        **kwargs,
    ) -> dict:
        """Multi-query fallback search. Returns the response dict shape used
        by the `deep_search_*` MCP tool: `results`, `winning_variant_index`,
        `variants_tried`, `all_weak`, optional `per_variant`.
        """

    @abstractmethod
    def update(self, force: bool = False) -> None:
        """Rebuild this source's index. `force=True` triggers a full rebuild
        regardless of whether the underlying content has changed."""

    def start_watcher(self) -> None:
        """Start a background watcher if applicable.

        Default: no-op. Backends with file-system content (codebase) override
        this. Backends with manual / scheduled refresh (webdoc, pdf) leave it
        as no-op and rely on explicit `update()` calls instead.
        """

    def stop_watcher(self) -> None:
        """Stop a running watcher and release its handles, if any.

        Default: no-op. Overridden by backends that start an observer, so the
        manager can release file handles before deleting a source's storage
        (notably on Windows, where open handles block directory removal).
        """

    def reset(self) -> None:
        """Rebuild this source's index from scratch, in place.

        Default: a forced update. Backends with a wipeable vector store
        override this to first empty the store (so it's a true clean rebuild,
        not an incremental no-op) without deleting the storage directory — the
        latter is blocked on Windows while the store handle is open.
        """
        self.update(force=True)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def status(self) -> dict:
        """Snapshot of this source's state for `list_sources` / `get_rag_status`.

        Should include at minimum: `name`, `type`, `chunk_count`, `last_update`,
        `drift_severity` (None / "warning" / "critical").
        """

    def drift_status_text(self) -> str:
        """Human-readable drift summary (forwarded to MCP `get_rag_status`)."""
        return "No config drift detected."
