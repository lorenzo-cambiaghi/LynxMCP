"""
RAG Manager - 100% LOCAL.

No data ever leaves the host. No calls to external services.
- Embeddings: HuggingFace local model (default BAAI/bge-small-en-v1.5,
  ~130 MB, downloaded once and cached)
- Vector store: ChromaDB persistent on local disk
- LLM: Disabled (None) - we only do vector retrieval, no generation
- Telemetry: Disabled
"""

import logging
import os
import sys
import threading

# ============================================================================
# BLOCK ANY EXTERNAL COMMUNICATION AND SILENCE NOISY LOGS
# ============================================================================
# Disable LlamaIndex / HuggingFace anonymous telemetry.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Drop any OpenAI key so LlamaIndex cannot accidentally call OpenAI.
os.environ.pop("OPENAI_API_KEY", None)
# Silence HuggingFace progress bars and warnings: their output on stdout
# would corrupt the JSON-RPC stream MCP uses.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# OFFLINE MODE: the model is already cached, never contact HuggingFace again.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Silence the recurring "llama-index-readers-file package not found" warning:
# the default FlatReader handles plain-text files (code, markdown, JSON, etc.)
# correctly. The optional package adds PDF/Word/EPUB parsers we don't need.
logging.getLogger("llama_index.core.readers.file.base").setLevel(logging.ERROR)


def log(msg):
    """Write logs to stderr so we don't corrupt the JSON-RPC stream on stdout."""
    print(msg, file=sys.stderr)


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import json
from datetime import datetime


# Severity levels for config drift detection. See _build_config_snapshot
# and check_config_drift for what triggers each.
DRIFT_CRITICAL = "critical"
DRIFT_WARNING = "warning"

# Mapping of impacting field -> severity. Fields not in this mapping are
# considered runtime-only and do not trigger drift detection.
_DRIFT_SEVERITY = {
    "embedding_model_name": DRIFT_CRITICAL,  # vectors become incomparable
    "codebase_path": DRIFT_CRITICAL,         # entirely different code indexed
    "supported_extensions": DRIFT_WARNING,   # missing files / orphan vectors
    "collection_name": DRIFT_WARNING,        # points at a different collection
}


def _configure_local_only(embedding_model_name: str):
    """Configure LlamaIndex to run 100% offline.

    Must be called BEFORE any operation on the index.
    """
    # Local embeddings: small open-source model, runs on CPU.
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    # LLM = None: we don't want LlamaIndex to ever call a remote LLM.
    # We only use the vector retriever (similarity search); the AI client
    # connected via MCP is responsible for any generation.
    Settings.llm = None


def _norm(path) -> str:
    """Normalize a path the same way LlamaIndex stores it: absolute, OS-native separators."""
    return os.path.normpath(os.path.abspath(str(path)))


class CodebaseRAG:
    """Manages indexing and search over a codebase.

    Everything runs locally: ChromaDB on disk, HuggingFace embeddings on CPU.
    """

    def __init__(
        self,
        codebase_path,
        rag_storage_path: str,
        supported_extensions,
        embedding_model_name: str,
        collection_name: str,
    ):
        self.codebase_path = Path(codebase_path)
        self.storage_path = Path(rag_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Frozenset of supported file extensions (lowercase, with leading dot).
        self.supported_extensions = frozenset(
            ext.lower() for ext in supported_extensions
        )

        # Stored as instance state because they are part of the config snapshot
        # used for drift detection.
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name

        # Serialize index mutations between the watcher, explicit updates,
        # and the search path.
        self._write_lock = threading.Lock()

        # Configure LlamaIndex for fully offline operation.
        _configure_local_only(embedding_model_name)

        # Initialize ChromaDB (local, persistent vector database).
        self.vector_store = ChromaVectorStore(
            chroma_collection=self._get_or_create_collection()
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Metadata file tracks the last indexed git commit, update time,
        # and the config snapshot used to build the index.
        self.metadata_file = self.storage_path / "metadata.json"
        self.metadata = self._load_metadata()
        self.index = self._load_or_build_index()

        # After index is loaded, check whether the live config matches the
        # config that built it. Surfaces silent drift (e.g. embedding model
        # changed but old vectors still in store).
        self._check_and_log_drift()

    def _get_or_create_collection(self):
        """Open or create the persistent ChromaDB collection on local disk."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.storage_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return self._chroma_client.get_or_create_collection(self.collection_name)

    def _load_metadata(self):
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {"last_commit": None, "last_update": None, "config_snapshot": None}

    def _save_metadata(self):
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    def _reset_collection(self):
        """Drop the ChromaDB collection so a rebuild does not duplicate data."""
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception as e:
            log(f"warning: delete_collection failed (probably did not exist): {e}")
        collection = self._chroma_client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def _build_index(self):
        log(f"[rag] Indexing codebase from {self.codebase_path}...")

        # Wipe the collection before re-indexing to avoid duplicates.
        self._reset_collection()

        documents = SimpleDirectoryReader(
            self.codebase_path,
            exclude_hidden=True,
            recursive=True,
            required_exts=sorted(self.supported_extensions),
        ).load_data()

        log(f"[rag] Found {len(documents)} files")

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )

        return index

    def _load_or_build_index(self):
        """Load existing index or build a new one.

        Source of truth is the ChromaDB collection: if it already contains
        documents we reuse the index instead of rebuilding it on every start.
        """
        try:
            existing_count = self.vector_store._collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            log(f"[rag] ChromaDB already populated ({existing_count} chunks), reusing.")
            return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        # count == 0: before rebuilding (a destructive operation through
        # _reset_collection), distinguish "true first run" from "race / lock
        # contention with another process". If metadata records a previous
        # build, do not re-index: reuse the store as-is and let an explicit
        # update() call force a rebuild if the user really wants one.
        if self.metadata.get("last_commit"):
            log(
                "[rag] Empty collection but metadata indicates a previous build: "
                "skipping rebuild (likely contention with another process). "
                "Reusing existing store."
            )
            return VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        return self._build_index()

    def needs_update(self):
        """Return True if a new git commit has been made since the last indexing.

        Returns False (without raising) when the codebase is not a git repo
        or git integration is not desired.
        """
        try:
            from git import Repo
            repo = Repo(self.codebase_path)
            current_commit = repo.head.commit.hexsha
            last_commit = self.metadata.get("last_commit")
            if current_commit != last_commit:
                return True
        except Exception as e:
            # Not a git repo, or git not available: that's fine, just skip.
            log(f"[rag] git check skipped: {e}")
        return False

    def update(self, force=False):
        """Rebuild the index if there are git changes (or if force=True)."""
        if not force and not self.needs_update():
            log("[rag] Index already up to date.")
            return

        log("[rag] Updating codebase index...")
        self.index = self._build_index()

        try:
            from git import Repo
            repo = Repo(self.codebase_path)
            self.metadata["last_commit"] = repo.head.commit.hexsha
        except Exception as e:
            log(f"[rag] could not read git HEAD ({e}), saving last_commit='unknown'")
            self.metadata["last_commit"] = "unknown"

        self.metadata["last_update"] = datetime.now().isoformat()
        # A full rebuild aligns the index with the current config; refresh
        # the snapshot so subsequent drift checks compare against the right
        # baseline.
        self.metadata["config_snapshot"] = self._build_config_snapshot()
        self._save_metadata()

        log("[rag] Index updated successfully.")

    def search(self, query: str, top_k=5):
        """Semantic search over the codebase. Fully local, no external calls."""
        # No auto-update here: the watcher keeps the index live and any
        # git-based catch-up is handled on demand by update_codebase_index.
        # Avoids costly rebuilds during a search.
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)

        return [
            {
                "file": result.metadata.get("file_name", "unknown"),
                "content": result.get_content(),
                "score": result.score,
            }
            for result in results
        ]

    # ------------------------------------------------------------------
    # Incremental, per-file update (driven by the file watcher)
    # ------------------------------------------------------------------

    def is_supported(self, filepath) -> bool:
        """True if the file has a supported extension and lives inside the codebase."""
        p = Path(filepath)
        if p.suffix.lower() not in self.supported_extensions:
            return False
        try:
            p.resolve().relative_to(self.codebase_path.resolve())
        except (ValueError, OSError):
            return False
        return True

    def _delete_file_chunks(self, abs_path: str) -> int:
        """Delete all chunks belonging to a file from the collection.

        Returns the number of chunks removed.
        """
        try:
            col = self.vector_store._collection
            existing = col.get(where={"file_path": abs_path})
            ids = existing.get("ids") or []
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception as e:
            log(f"[rag] delete chunks failed for {abs_path}: {e}")
            return 0

    def update_file(self, filepath) -> bool:
        """Re-index a single file: drop the old chunks, insert the new ones."""
        abs_path = _norm(filepath)
        if not self.is_supported(abs_path):
            return False
        if not os.path.isfile(abs_path):
            # Modify followed by delete before flush: treat as remove.
            return self.remove_file(abs_path)

        with self._write_lock:
            removed = self._delete_file_chunks(abs_path)
            try:
                docs = SimpleDirectoryReader(input_files=[abs_path]).load_data()
            except Exception as e:
                log(f"[rag] read failed for {abs_path}: {e}")
                return False

            if not docs:
                log(f"[rag] no document extracted from {abs_path}")
                return False

            for doc in docs:
                try:
                    self.index.insert(doc)
                except Exception as e:
                    log(f"[rag] insert failed for {abs_path}: {e}")
                    return False

        log(f"[rag] Re-indexed {Path(abs_path).name} ({removed} old chunks dropped)")
        return True

    def remove_file(self, filepath) -> bool:
        """Remove all chunks of a file from the index (on delete or move)."""
        abs_path = _norm(filepath)
        if Path(abs_path).suffix.lower() not in self.supported_extensions:
            return False
        with self._write_lock:
            removed = self._delete_file_chunks(abs_path)
        if removed:
            log(f"[rag] Removed {removed} chunks for {Path(abs_path).name}")
        return removed > 0

    # ------------------------------------------------------------------
    # Config drift detection
    # ------------------------------------------------------------------

    def _build_config_snapshot(self) -> dict:
        """Snapshot of the config fields that, if changed, invalidate the index.

        See _DRIFT_SEVERITY for which fields are tracked. Fields like
        watcher.debounce_seconds, search.default_top_k, loading_timeout_seconds
        are intentionally omitted: changing them has no effect on the stored
        vectors.

        ignored_path_fragments is also omitted: it is currently consumed
        only by the file watcher in mcp_server.py, not by SimpleDirectoryReader,
        so it does not affect what is in the index.
        """
        return {
            "codebase_path": str(self.codebase_path.resolve()),
            "supported_extensions": sorted(self.supported_extensions),
            "embedding_model_name": self.embedding_model_name,
            "collection_name": self.collection_name,
        }

    def check_config_drift(self):
        """Compare the live config against the snapshot in metadata.json.

        Returns a dict shaped like:
            {
              "severity": "critical" | "warning",
              "changes": {field: {"old": ..., "new": ...}, ...}
            }
        or None when no drift is detected (or when there is no stored
        snapshot yet, in which case we silently record the current one).
        """
        current = self._build_config_snapshot()
        stored = self.metadata.get("config_snapshot")

        if stored is None:
            # No baseline yet: record the current snapshot silently.
            # Happens on first run after upgrading from a version without
            # drift detection.
            self.metadata["config_snapshot"] = current
            self._save_metadata()
            return None

        if stored == current:
            return None

        changes = {}
        worst = DRIFT_WARNING
        for field, new_value in current.items():
            old_value = stored.get(field)
            if old_value != new_value:
                changes[field] = {"old": old_value, "new": new_value}
                if _DRIFT_SEVERITY.get(field) == DRIFT_CRITICAL:
                    worst = DRIFT_CRITICAL

        if not changes:
            return None

        return {"severity": worst, "changes": changes}

    def _check_and_log_drift(self):
        """Run drift check and emit a warning to stderr if anything changed."""
        drift = self.check_config_drift()
        if drift is None:
            return
        sev = drift["severity"].upper()
        log(f"[drift] {sev}: config has changed since the index was built.")
        for field, delta in drift["changes"].items():
            log(f"[drift]   {field}: {delta['old']!r} -> {delta['new']!r}")
        if drift["severity"] == DRIFT_CRITICAL:
            log("[drift] Search results may be wrong. Run a full rebuild "
                "(update_codebase_index with force=True) to clear the warning.")
        else:
            log("[drift] The index may be stale. Consider a rebuild.")

    def drift_status_text(self) -> str:
        """Human-readable drift summary for inclusion in get_rag_status()."""
        drift = self.check_config_drift()
        if drift is None:
            return "No config drift detected."
        sev = drift["severity"].upper()
        lines = [f"Config drift: {sev}"]
        for field, delta in drift["changes"].items():
            lines.append(f"  - {field}: {delta['old']!r} -> {delta['new']!r}")
        if drift["severity"] == DRIFT_CRITICAL:
            lines.append(
                "  Search results may be wrong. Run update_codebase_index(force=True) to rebuild."
            )
        else:
            lines.append("  The index may be stale. Consider a rebuild.")
        return "\n".join(lines)
