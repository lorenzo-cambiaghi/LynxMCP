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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SourceBackend(ABC):
    """One indexable source. Subclass per type (codebase, webdoc, pdf, ...)."""

    # Subclasses MUST set this. Used as the discriminator in config and the
    # key in `SOURCE_BACKENDS`.
    type_name: str = ""

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
                  Used as the ChromaDB collection name and as the suffix in
                  the auto-generated MCP tool names (`search_<name>`, etc.).
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
