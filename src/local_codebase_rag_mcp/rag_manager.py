"""
RAG Manager - 100% LOCAL.

No data ever leaves the host. No calls to external services.
- Embeddings: HuggingFace local model (default BAAI/bge-small-en-v1.5,
  ~130 MB, downloaded once and cached)
- Vector store: ChromaDB persistent on local disk
- LLM: Disabled (None) - we only do vector retrieval, no generation
- Telemetry: Disabled
"""

import fnmatch
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


import re
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi
from pathlib import Path
import json
from datetime import datetime


# ----------------------------------------------------------------------------
# Code-aware tokenizer for BM25.
# ----------------------------------------------------------------------------
# Goal: make queries like "IDamageable", "AStarPathFinder", "calculateSpacing"
# match identifiers in the code even when the casing differs slightly.
#
# Strategy: keep the FULL identifier as a token AND its constituent parts
# (CamelCase / snake_case / dot.notation split). Lowercase everything.
# This roughly doubles index size but dramatically improves recall on
# identifier-heavy queries.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"[^A-Za-z0-9]+")


def _tokenize_code(text: str) -> list:
    """Split text into BM25 tokens, preserving both whole identifiers and parts."""
    if not text:
        return []
    tokens = []
    for raw in _NON_WORD.split(text):
        if not raw:
            continue
        lower_raw = raw.lower()
        tokens.append(lower_raw)
        # Add CamelCase / Pascal pieces as separate tokens for partial matching.
        if any(c.isupper() for c in raw):
            for piece in _CAMEL_BOUNDARY.split(raw):
                piece_lower = piece.lower()
                if piece_lower and piece_lower != lower_raw:
                    tokens.append(piece_lower)
    return tokens


def _normalize_extension(ext: str) -> str:
    """Make sure an extension has a leading dot and is lowercased."""
    ext = ext.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


def _matches_filters(item: dict, file_glob, extensions, path_contains) -> bool:
    """Apply optional post-retrieval filters to a single result dict.

    All provided filters are AND-ed. None / empty filters are ignored.
    Both `file` (basename) and `file_path` (full path) are checked, so a
    glob like '*.cs' works regardless of which field carries the path.
    """
    file_name = item.get("file") or ""
    file_path = item.get("file_path") or ""

    if extensions:
        norm_exts = {_normalize_extension(e) for e in extensions}
        # Use file_name when available; fall back to file_path basename.
        candidate = file_name or os.path.basename(file_path)
        suffix = os.path.splitext(candidate)[1].lower()
        if suffix not in norm_exts:
            return False

    if path_contains:
        needle = path_contains
        if needle not in file_path and needle not in file_name:
            return False

    if file_glob:
        # fnmatch handles Unix-shell glob (*, ?, [seq]). Allow it to match
        # against both the basename and the full path so users can write
        # either '*.py' or '**/Bullet*/**'.
        if not (
            fnmatch.fnmatch(file_name, file_glob)
            or fnmatch.fnmatch(file_path, file_glob)
            # Path normalization: file_path uses OS separators; allow forward-
            # slash globs to still match by trying a normalized variant.
            or fnmatch.fnmatch(file_path.replace(os.sep, "/"), file_glob)
        ):
            return False

    return True


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
        search_mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_pool_size: int = 30,
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

        # Search behavior: dense / sparse / hybrid (RRF-fused).
        if search_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"search_mode must be one of 'hybrid' | 'dense' | 'sparse', got {search_mode!r}"
            )
        self.search_mode = search_mode
        self.rrf_k = int(rrf_k)
        self.candidate_pool_size = int(candidate_pool_size)

        # BM25 sparse index, lazily built from chunks living in ChromaDB.
        # _bm25_docs maps chunk_id -> tokenized content; the BM25Okapi object
        # is rebuilt on demand from this dict whenever it goes stale.
        self._bm25_docs: dict = {}
        self._bm25_meta: dict = {}  # chunk_id -> {"file": ..., "content": ...}
        self._bm25_okapi = None
        self._bm25_doc_ids: list = []

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

        # Collection contents changed wholesale: drop the BM25 cache so the
        # next search rebuilds it from the new ChromaDB state.
        self._invalidate_bm25()
        return index

    def _load_or_build_index(self):
        """Load existing index or build a new one.

        Source of truth is the ChromaDB collection: if it already contains
        documents we reuse the index instead of rebuilding it on every start.

        On a true first-run build we also persist metadata immediately, so
        subsequent code paths (drift detection, git-update detection) have a
        valid baseline without needing an extra explicit update() call.
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

        # Genuine first run: build, then persist metadata in the same step
        # so the next constructor call sees a complete state and so a
        # follow-up update(force=True) does not redundantly rebuild.
        index = self._build_index()
        self._record_build_metadata()
        return index

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

    def _record_build_metadata(self):
        """Persist the post-build metadata: git head, timestamp, config snapshot.

        Called from every code path that completes a fresh _build_index().
        Centralized here so that '_load_or_build_index' (first-run case) and
        'update()' (explicit rebuild case) cannot drift out of sync.
        """
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

    def update(self, force=False):
        """Rebuild the index if there are git changes (or if force=True)."""
        if not force and not self.needs_update():
            log("[rag] Index already up to date.")
            return

        log("[rag] Updating codebase index...")
        self.index = self._build_index()
        self._record_build_metadata()
        log("[rag] Index updated successfully.")

    def search(
        self,
        query: str,
        top_k=5,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
    ):
        """Hybrid (default) / dense / sparse search over the codebase.

        Mode is set at construction time via `search_mode`. In hybrid mode we
        fetch `candidate_pool_size` candidates from each retriever (dense and
        BM25) then fuse rankings via Reciprocal Rank Fusion.

        Optional filters (all AND-ed together) narrow the result set:
          - file_glob: fnmatch pattern matched against basename and full path.
          - extensions: iterable of extensions ('.py' or 'py'), case-insensitive.
          - path_contains: substring required in file path or filename.

        When any filter is present the underlying retrieval over-fetches a
        wider pool (5x top_k) so post-filtering still leaves enough results.
        """
        # No auto-update here: the watcher keeps the index live and any
        # git-based catch-up is handled on demand by update_codebase_index.
        return self._search_once(
            query,
            top_k=top_k,
            mode=self.search_mode,
            file_glob=file_glob,
            extensions=extensions,
            path_contains=path_contains,
        )

    def _search_once(
        self,
        query: str,
        top_k: int,
        mode: str,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
    ):
        """Single-pass retrieval helper shared by search() and deep_search().

        `mode` is explicit (rather than reading self.search_mode) so deep_search
        can override it per call without mutating instance state.
        """
        has_filters = bool(file_glob or extensions or path_contains)
        # Over-fetch when filtering: filtered-out candidates leave a smaller
        # pool, so we need extras to still produce top_k matches.
        fetch_n = top_k * 5 if has_filters else top_k

        if mode == "dense":
            results = self._dense_lookup(query, fetch_n)
        elif mode == "sparse":
            results = self._bm25_lookup(query, fetch_n)
        elif mode == "hybrid":
            pool = max(fetch_n, self.candidate_pool_size)
            dense_hits = self._dense_lookup(query, pool)
            sparse_hits = self._bm25_lookup(query, pool)
            results = self._rrf_fuse([dense_hits, sparse_hits], k=self.rrf_k, top_k=pool)
        else:
            raise ValueError(
                f"mode must be one of 'hybrid' | 'dense' | 'sparse', got {mode!r}"
            )

        if has_filters:
            results = [
                r for r in results
                if _matches_filters(r, file_glob, extensions, path_contains)
            ]

        return results[:top_k]

    def deep_search(
        self,
        queries,
        top_k: int = 5,
        *,
        mode=None,
        file_glob=None,
        extensions=None,
        path_contains=None,
        min_score=None,
        min_results=None,
        return_all_variants: bool = False,
        score_thresholds=None,
    ):
        """Multi-query fallback search.

        Tries each query in `queries` in order. The first variant whose result
        set passes the weakness threshold wins and is returned. If all variants
        fail the threshold, returns the *strongest weak set* with a marker so
        the caller can decide what to do.

        Args:
            queries: list[str] of query variants in priority order.
            top_k: results to return.
            mode: per-call override of retrieval mode ("dense"/"sparse"/"hybrid").
                  None = use self.search_mode.
            file_glob, extensions, path_contains: same semantics as search().
                  Applied to every variant.
            min_score: per-call override of the score threshold. None = use the
                  mode-specific default from score_thresholds.
            min_results: per-call override of the min-results threshold.
                  None = 2.
            return_all_variants: when True, the response dict includes the
                  per-variant result lists for debugging.
            score_thresholds: dict {mode: threshold} from config. Optional;
                  callers without config can omit it and rely on min_score.

        Returns:
            dict with shape:
              {
                "results": [...],          # the final ranked results
                "winning_variant_index": int or None,  # 1-based, None if all weak
                "variants_tried": int,
                "all_weak": bool,          # True if no variant crossed threshold
                "per_variant": [...],      # only when return_all_variants
              }
        """
        if not isinstance(queries, (list, tuple)):
            raise TypeError(f"queries must be a list of strings, got {type(queries).__name__}")
        if len(queries) == 0:
            raise ValueError("queries must contain at least one string")
        if not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError("every query must be a non-empty string")

        effective_mode = mode or self.search_mode
        if effective_mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(
                f"mode must be one of 'hybrid' | 'dense' | 'sparse', got {effective_mode!r}"
            )

        # Resolve the threshold for this call. Precedence:
        #   1. explicit min_score arg
        #   2. score_thresholds dict from config (mode-specific entry)
        #   3. built-in fallback defaults
        builtin_defaults = {"dense": 0.45, "hybrid": 0.012, "sparse": 3.0}
        if min_score is not None:
            threshold = float(min_score)
        elif score_thresholds and effective_mode in score_thresholds:
            threshold = float(score_thresholds[effective_mode])
        else:
            threshold = builtin_defaults[effective_mode]

        effective_min_results = int(min_results) if min_results is not None else 2

        per_variant = []
        best_results = []
        best_top_score = float("-inf")
        winning_idx = None

        for idx, variant in enumerate(queries):
            results = self._search_once(
                variant,
                top_k=top_k,
                mode=effective_mode,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
            )
            top_score = results[0]["score"] if results else float("-inf")
            passes = (
                len(results) >= effective_min_results
                and top_score >= threshold
            )

            if return_all_variants:
                per_variant.append({
                    "query": variant,
                    "results": list(results),
                    "top_score": top_score if results else None,
                    "passed_threshold": passes,
                })

            # Track the strongest weak set in case all variants fail.
            if top_score > best_top_score:
                best_top_score = top_score
                best_results = results

            if passes:
                response = {
                    "results": results,
                    "winning_variant_index": idx + 1,
                    "variants_tried": idx + 1,
                    "all_weak": False,
                }
                if return_all_variants:
                    response["per_variant"] = per_variant
                return response

        # All variants failed the threshold.
        response = {
            "results": best_results,
            "winning_variant_index": None,
            "variants_tried": len(queries),
            "all_weak": True,
        }
        if return_all_variants:
            response["per_variant"] = per_variant
        return response

    # ------------------------------------------------------------------
    # Retrieval primitives: dense (vectors), sparse (BM25), fusion (RRF)
    # ------------------------------------------------------------------

    def _dense_lookup(self, query: str, top_n: int) -> list:
        """Pure semantic retrieval via the LlamaIndex / ChromaDB pipeline."""
        retriever = self.index.as_retriever(similarity_top_k=top_n)
        results = retriever.retrieve(query)
        return [
            {
                "id": getattr(getattr(r, "node", None), "id_", None) or r.metadata.get("file_path", ""),
                "file": r.metadata.get("file_name", "unknown"),
                "file_path": r.metadata.get("file_path", ""),
                "content": r.get_content(),
                "score": r.score,
            }
            for r in results
        ]

    def _ensure_bm25(self):
        """Build (or rebuild) the BM25 index from the current ChromaDB content."""
        if self._bm25_okapi is not None:
            return
        # Pull every chunk currently in ChromaDB. Cheap for typical codebases
        # (a few thousand chunks); rare operation thanks to the lazy cache.
        try:
            data = self.vector_store._collection.get(include=["documents", "metadatas"])
        except Exception as e:
            log(f"[bm25] could not read ChromaDB collection: {e}")
            self._bm25_okapi = None
            return
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        self._bm25_docs = {}
        self._bm25_meta = {}
        for cid, content, meta in zip(ids, docs, metas):
            tokens = _tokenize_code(content or "")
            self._bm25_docs[cid] = tokens
            self._bm25_meta[cid] = {
                "file": (meta or {}).get("file_name", "unknown"),
                "file_path": (meta or {}).get("file_path", ""),
                "content": content or "",
            }
        self._bm25_doc_ids = list(self._bm25_docs.keys())
        if self._bm25_doc_ids:
            corpus = [self._bm25_docs[cid] for cid in self._bm25_doc_ids]
            self._bm25_okapi = BM25Okapi(corpus)
        else:
            self._bm25_okapi = None

    def _invalidate_bm25(self):
        """Mark the BM25 cache stale so the next search rebuilds it."""
        self._bm25_okapi = None

    def _bm25_lookup(self, query: str, top_n: int) -> list:
        """BM25 sparse retrieval over the in-memory tokenized corpus."""
        self._ensure_bm25()
        if self._bm25_okapi is None or not self._bm25_doc_ids:
            return []
        query_tokens = _tokenize_code(query)
        if not query_tokens:
            return []
        scores = self._bm25_okapi.get_scores(query_tokens)
        # Pair scores with doc ids, sort desc, take top_n.
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        out = []
        for i in ranked_idx:
            if scores[i] <= 0:
                # rank-bm25 returns 0 for documents without any matching token.
                # Skip them to keep the candidate list meaningful.
                continue
            cid = self._bm25_doc_ids[i]
            meta = self._bm25_meta.get(cid, {})
            out.append({
                "id": cid,
                "file": meta.get("file", "unknown"),
                "file_path": meta.get("file_path", ""),
                "content": meta.get("content", ""),
                "score": float(scores[i]),
            })
        return out

    def _rrf_fuse(self, rankings: list, k: int, top_k: int) -> list:
        """Reciprocal Rank Fusion: combine multiple rankings into one.

        Each document gets score = sum over rankings of 1 / (k + rank+1).
        """
        fused_scores: dict = {}
        item_by_id: dict = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking):
                cid = item.get("id")
                if not cid:
                    continue
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                # Keep the first metadata seen for this id; sources agree on file/content.
                item_by_id.setdefault(cid, item)
        sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        out = []
        for cid in sorted_ids:
            item = dict(item_by_id[cid])
            item["score"] = fused_scores[cid]  # fused score replaces raw per-source score
            out.append(item)
        return out

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

            # Chunks changed: invalidate BM25 cache (rebuilt lazily on next search).
            self._invalidate_bm25()

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
                self._invalidate_bm25()
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
