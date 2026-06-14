"""Typed exceptions shared across lynx.

Kept dependency-free (stdlib only) so any module — including the low-level
`rag_manager` and the out-of-process `integrity` probe — can import it without
pulling in the heavy llama-index / chromadb stack.
"""
from __future__ import annotations


class CorruptIndexError(Exception):
    """A source's on-disk vector index is unreadable.

    Raised when ChromaDB cannot open or query a collection — typically because
    the index was written by an incompatible chromadb version, or the store was
    truncated by a process killed mid-write. The data is disposable (pure
    derived embeddings), so the remedy is always "wipe and rebuild":
    `lynx reset --source <name>` or the dashboard's Reset button.
    """

    def __init__(self, source: str, detail: str, storage_dir: str | None = None):
        self.source = source
        self.detail = detail
        self.storage_dir = storage_dir
        super().__init__(
            f"index for source {source!r} is corrupt or unreadable: {detail}. "
            f"Reset it with `lynx reset --source {source}` (data is disposable; "
            f"it rebuilds from your files)."
        )
