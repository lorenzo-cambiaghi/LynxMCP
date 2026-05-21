"""PDF source backend.

Scans a folder of `.pdf` files, extracts text page-by-page via
`pdf_extractor.extract_pdf`, dumps one Markdown per page in
`<storage>/<source>/_dump/<rel>/<pdf_stem>/page_NNNN.md`, then delegates
indexing (chunk, embed, BM25, hybrid retrieval, drift, SHA cache) to a
`CodebaseRAG` pointed at `_dump/`. Same architectural pattern as
`WebdocBackend` (fetch → dump → rag).

What we do NOT do:
  - OCR scanned PDFs (documented limit, no Tesseract dependency).
  - Decrypt password-protected PDFs (skipped with a warning row in
    `_extract_state.json`).
  - Watch the source folder by default — refresh is explicit via
    `lynx build --source <name>`. Watcher is opt-in via
    `watcher.enabled=true` because PDFs change rarely and re-extracting
    is expensive (10-30s for a typical document).
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import SourceBackend
from .pdf_extractor import extract_pdf, ExtractResult
from .pdf_dump import (
    write_page_dump,
    wipe_pdf_dump,
    load_state,
    save_state_atomic,
    PdfStateEntry,
)


def _log(msg: str) -> None:
    """Stderr only — stdout is the MCP JSON-RPC channel."""
    print(msg, file=sys.stderr)


# Same chunk size we use everywhere else for SHA-256.
_HASH_READ_CHUNK_BYTES = 1 << 20


def _canonical_path(abs_path: str) -> str:
    """Resolve a path the same way regardless of where it came from.

    Why: macOS exposes `/tmp` and `/var` as symlinks into `/private`.
    `Path.glob` returns paths verbatim with the prefix we passed in,
    while watchdog's FS events report the OS-canonical form. Without
    normalization the same physical file ends up under TWO keys in the
    extract state (one from bootstrap, one from the watcher), and the
    SHA cache misses on every watcher event.

    `os.path.realpath` also collapses any user-injected `..` segments,
    which is fine for our use case (we never want two different state
    rows to point at the same PDF).
    """
    try:
        return os.path.realpath(abs_path)
    except (OSError, ValueError):
        return os.path.normpath(abs_path)


def _sha256_of_file(abs_path: str) -> str:
    """SHA-256 of file bytes, empty string on I/O error."""
    h = hashlib.sha256()
    try:
        with open(abs_path, "rb") as f:
            while True:
                block = f.read(_HASH_READ_CHUNK_BYTES)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError as e:
        _log(f"[pdf] hash failed for {abs_path}: {e}")
        return ""


class PdfBackend(SourceBackend):
    """Indexes a directory of `.pdf` files via the extract-then-dump pattern."""

    type_name = "pdf"

    def __init__(
        self,
        name: str,
        source_config: dict,
        shared_config: Any,
        storage_dir: Path,
    ):
        super().__init__(name, source_config, shared_config, storage_dir)

        # Canonicalise the source root once. Every per-file path we then
        # derive from it (via glob) inherits the same prefix, so we don't
        # depend on watchdog's path-resolution behaviour matching ours.
        self.source_path = Path(_canonical_path(str(Path(source_config["path"]))))
        self.recursive = bool(source_config.get("recursive", True))
        self.file_glob = str(source_config.get("file_glob", "**/*.pdf"))

        ext = source_config["extractor"]
        self.backend_name: str = ext["backend"]
        self.max_file_mb: int = ext["max_file_mb"]
        self.max_pages: int = ext["max_pages_per_file"]
        self.skip_password: bool = ext["skip_password_protected"]
        self.skip_empty: bool = ext["skip_if_text_empty"]

        # On-disk layout: same skeleton as WebdocBackend.
        self.dump_dir = self.storage_dir / "_dump"
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.storage_dir / "_extract_state.json"

        # Single re-entrant lock guards every mutation (update, watcher
        # event, remove). Reads are lock-free because they go through
        # CodebaseRAG, which has its own locking.
        self._lock = threading.RLock()
        self._extract_state: dict = load_state(self.state_file)

        # Watcher state — only populated when start_watcher() runs.
        self._observer = None

        # Lazy-import the RAG: keeps `import lynx.sources.pdf` fast for
        # CLI utilities that don't actually need the embedding model.
        from ..rag_manager import CodebaseRAG
        self.rag = CodebaseRAG(
            codebase_path=str(self.dump_dir),
            rag_storage_path=str(self.storage_dir),
            supported_extensions=[".md"],
            embedding_model_name=shared_config.embedding.model_name,
            collection_name=name,
            search_mode=shared_config.search.mode,
            rrf_k=shared_config.search.rrf_k,
            candidate_pool_size=shared_config.search.candidate_pool_size,
            reranker_config=shared_config.search.reranker,
        )

        # Bootstrap: if the cache is empty AND there's at least one PDF
        # in the source folder, extract synchronously. Same UX-pattern as
        # GraphLayer — without this the first `lynx serve` after enabling
        # a new PDF source would return empty results until the user ran
        # `lynx build` manually.
        if not self._extract_state and self._has_source_pdfs():
            try:
                _log(f"[pdf:{name}] first-boot bootstrap of {self.source_path}")
                self.update(force=True)
            except Exception as e:
                # Never let bootstrap break server startup.
                _log(f"[pdf:{name}] bootstrap failed (search returns empty until "
                     f"`lynx build --source {name}`): {e}")

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_pdfs(self) -> list:
        """Walk the source folder and return absolute paths of candidate PDFs.

        Filters applied (AND-ed):
          - file_glob (default `**/*.pdf`) matched against the path
            relative to `source_path`, using `pathlib.Path.glob`
            semantics (`**` matches any number of dirs, `*` matches one
            segment, etc.). Note that Python's stdlib `fnmatch` does
            NOT support `**` — using `Path.glob` is what makes
            recursive globs actually work.
          - dot-prefixed directories pruned (matches CodebaseRAG / git
            convention) by post-filtering matched paths.
          - `recursive=False` restricts to top-level only by short-
            circuiting the file_glob to just its basename pattern.
        Sorted alphabetically so tests are deterministic.
        """
        if not self.source_path.is_dir():
            return []
        # When recursive=False, ignore any `**/` prefix the user added
        # and match against the top-level only. This keeps the option
        # intuitive: `recursive=False` is an honest "don't descend".
        glob_pat = self.file_glob if self.recursive else self.file_glob.rsplit("/", 1)[-1]
        candidates = self.source_path.glob(glob_pat)
        out: list = []
        root_str = str(self.source_path)
        for p in candidates:
            if not p.is_file():
                continue
            if p.suffix.lower() != ".pdf":
                continue
            # Skip anything under a dot-prefixed parent dir (mirrors the
            # `exclude_hidden=True` behavior of SimpleDirectoryReader and
            # of CodebaseRAG's own walker).
            try:
                rel_parts = p.relative_to(self.source_path).parts
            except ValueError:
                rel_parts = ()
            if any(part.startswith(".") for part in rel_parts):
                continue
            out.append(_canonical_path(str(p)))
        out.sort()
        return out

    def _has_source_pdfs(self) -> bool:
        """Cheap existence check used by the bootstrap path."""
        if not self.source_path.is_dir():
            return False
        try:
            return any(self._discover_pdfs())
        except Exception:
            return False

    def _rel_parts_for(self, abs_path: str) -> tuple:
        """Return `(rel_dir_path, pdf_stem)` used by the dump writer."""
        rel = Path(os.path.relpath(abs_path, str(self.source_path)))
        return (str(rel.parent).replace(os.sep, "/"), rel.stem)

    # ------------------------------------------------------------------
    # Per-PDF processing
    # ------------------------------------------------------------------

    def _wipe_pdf(self, abs_path: str) -> None:
        """Remove dump dir + state entry for one PDF. Caller persists."""
        entry = self._extract_state.get(abs_path)
        if entry is not None:
            try:
                wipe_pdf_dump(self.dump_dir, entry.dump_rel, entry.pdf_stem)
            except OSError as e:
                _log(f"[pdf:{self.name}] wipe failed for {abs_path}: {e}")
            self._extract_state.pop(abs_path, None)
        else:
            # The PDF may have been added then removed before state was
            # saved — derive the dump dir from current rel mapping.
            rel_dir, stem = self._rel_parts_for(abs_path)
            try:
                wipe_pdf_dump(self.dump_dir, rel_dir, stem)
            except OSError:
                pass

    def _process_one(self, abs_path: str, sha: str, when: str) -> None:
        """Extract one PDF + write per-page dumps + update state.

        Always upserts a state row (even on skip/error) so we record
        WHY a PDF didn't contribute and don't keep re-trying it on
        every rebuild. Bumping `max_file_mb` or `max_pages` invalidates
        the SHA equality check naturally (the file's still the same but
        the user's now asking for a different outcome) — handled by the
        caller via `force=True`.
        """
        rel_dir, stem = self._rel_parts_for(abs_path)
        result: ExtractResult = extract_pdf(
            abs_path,
            max_file_mb=self.max_file_mb,
            max_pages=self.max_pages,
            backend=self.backend_name,
            skip_if_text_empty=self.skip_empty,
        )

        if result.status == "ok":
            # Clear any previous dumps in case the page count shrank.
            try:
                wipe_pdf_dump(self.dump_dir, rel_dir, stem)
            except OSError:
                pass
            for page_num, page_text in result.pages:
                write_page_dump(
                    dump_root=self.dump_dir,
                    rel_path=rel_dir,
                    pdf_stem=stem,
                    page_num=page_num,
                    total_pages=result.total_pages,
                    pdf_abs_path=abs_path,
                    text=page_text,
                    extracted_at=when,
                    pdf_title=result.title,
                )
        else:
            # Non-ok: make sure no stale dump remains for this PDF.
            try:
                wipe_pdf_dump(self.dump_dir, rel_dir, stem)
            except OSError:
                pass

        self._extract_state[abs_path] = PdfStateEntry(
            sha256=sha,
            n_pages=result.total_pages,
            n_chars=result.n_chars,
            status=result.status,
            extracted_at=when,
            dump_rel=rel_dir,
            pdf_stem=stem,
            error=result.error,
        )

    # ------------------------------------------------------------------
    # SourceBackend API
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
            query, top_k=top_k,
            file_glob=file_glob, extensions=extensions, path_contains=path_contains,
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
            queries=queries, top_k=top_k, mode=mode,
            file_glob=file_glob, extensions=extensions, path_contains=path_contains,
            min_score=min_score, min_results=min_results,
            return_all_variants=return_all_variants,
            score_thresholds=self.shared.search.deep.score_thresholds,
        )

    def update(self, force: bool = False) -> None:
        """Re-process the source folder.

        With `force=False` (default), SHA-incremental: PDFs whose bytes
        haven't changed since the last successful run are skipped. This
        is the path the watcher uses (one PDF at a time, indirectly
        via `update_single_pdf`).

        With `force=True`, every PDF is re-extracted regardless of its
        cached SHA — used by `lynx build`, the bootstrap path, and when
        the user wants a clean re-extraction (e.g. after bumping
        `max_pages_per_file` or switching extractor backend).
        """
        with self._lock:
            now = datetime.now().isoformat(timespec="seconds")
            candidates = set(self._discover_pdfs())
            cached = set(self._extract_state.keys())
            removed = cached - candidates

            # Partition present files into changed / unchanged.
            changed: list = []
            unchanged: list = []
            for p in sorted(candidates):
                live_sha = _sha256_of_file(p)
                cached_entry = self._extract_state.get(p)
                if (
                    force
                    or cached_entry is None
                    or not live_sha
                    or live_sha != cached_entry.sha256
                ):
                    changed.append((p, live_sha))
                else:
                    unchanged.append(p)

            # Apply removals first so the dump dir is consistent before
            # we re-process the rest.
            for p in removed:
                self._wipe_pdf(p)

            n_ok = 0
            n_skipped = 0
            for p, sha in changed:
                try:
                    self._process_one(p, sha, now)
                except Exception as e:
                    # Per-file recovery so one bad PDF doesn't abort the
                    # whole rebuild. Mirror the diagnostic in the state
                    # row so it's visible from `lynx status`.
                    _log(f"[pdf:{self.name}] processing failed for {p}: {e}")
                    rel_dir, stem = self._rel_parts_for(p)
                    self._extract_state[p] = PdfStateEntry(
                        sha256=sha,
                        n_pages=0, n_chars=0,
                        status="error",
                        extracted_at=now,
                        dump_rel=rel_dir, pdf_stem=stem,
                        error=f"{type(e).__name__}: {e}",
                    )
                final = self._extract_state.get(p)
                if final and final.status == "ok":
                    n_ok += 1
                else:
                    n_skipped += 1

            save_state_atomic(self.state_file, self._extract_state)
            _log(
                f"[pdf:{self.name}] processed {len(changed)} files "
                f"({n_ok} ok, {n_skipped} skipped/error), "
                f"removed {len(removed)}, unchanged {len(unchanged)}"
            )

            # Finally, push the dump changes into ChromaDB.
            self.rag.update(force=force)

    # ------------------------------------------------------------------
    # Watcher (opt-in)
    # ------------------------------------------------------------------

    def _reprocess_single_pdf(self, abs_path: str) -> None:
        """Watcher entry point: re-extract one PDF and push the resulting
        per-page dump files into ChromaDB via `rag.update_file`.

        Wrapped in the same write lock as `update()` so a concurrent
        explicit rebuild can't race with the watcher.
        """
        # Canonicalise so we hit the same state key the bootstrap path
        # uses (see _canonical_path note about macOS /var vs /private/var).
        abs_path = _canonical_path(abs_path)
        with self._lock:
            now = datetime.now().isoformat(timespec="seconds")
            sha = _sha256_of_file(abs_path)
            cached = self._extract_state.get(abs_path)
            if cached and sha and sha == cached.sha256:
                # No-op: file hasn't actually changed (some editors save
                # the same bytes when you `:w` without changes).
                return
            try:
                self._process_one(abs_path, sha, now)
            except Exception as e:
                _log(f"[pdf:{self.name}] watcher reprocess failed for {abs_path}: {e}")
                return
            save_state_atomic(self.state_file, self._extract_state)

            # Push the per-page dump files into ChromaDB. The CodebaseRAG
            # watcher of `_dump/` is intentionally not used — we drive
            # rag updates directly so the timing is deterministic.
            rel_dir, stem = self._rel_parts_for(abs_path)
            target_dir = self.dump_dir / rel_dir / stem if rel_dir not in ("", ".") else self.dump_dir / stem
            if target_dir.is_dir():
                for md in target_dir.glob("page_*.md"):
                    try:
                        self.rag.update_file(str(md))
                    except Exception as e:
                        _log(f"[pdf:{self.name}] rag.update_file failed for {md}: {e}")

    def _remove_single_pdf(self, abs_path: str) -> None:
        """Watcher entry point: PDF was deleted from disk."""
        abs_path = _canonical_path(abs_path)
        with self._lock:
            # Capture dump files BEFORE wiping so we can tell rag to drop them.
            cached = self._extract_state.get(abs_path)
            dump_files: list = []
            if cached is not None:
                rel_dir, stem = cached.dump_rel, cached.pdf_stem
                target_dir = self.dump_dir / rel_dir / stem if rel_dir not in ("", ".") else self.dump_dir / stem
                if target_dir.is_dir():
                    dump_files = [str(p) for p in target_dir.glob("page_*.md")]
            self._wipe_pdf(abs_path)
            save_state_atomic(self.state_file, self._extract_state)
            for md_path in dump_files:
                try:
                    self.rag.remove_file(md_path)
                except Exception as e:
                    _log(f"[pdf:{self.name}] rag.remove_file failed for {md_path}: {e}")

    def start_watcher(self) -> None:
        """Start a watchdog Observer on the *source* PDF folder.

        We deliberately watch the source folder (the user-managed
        directory of `.pdf` files) and NOT `_dump/`: a watcher on
        `_dump/` would only see the secondary side-effect of our own
        writes, while the source folder is where actual user-initiated
        changes happen (new PDF dropped in, old PDF deleted).

        Off by default (`watcher.enabled=false`). Idempotent: calling
        twice doesn't spawn two observers.

        The handler is debounced (default 5s, larger than the codebase
        default 2s because reading a PDF is much slower than reading a
        Python file).
        """
        watcher_cfg = self.source_config.get("watcher") or {}
        if not watcher_cfg.get("enabled", False):
            return
        if self._observer is not None:
            return

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        debounce_seconds = float(watcher_cfg.get("debounce_seconds", 5.0))
        reprocess = self._reprocess_single_pdf
        remove = self._remove_single_pdf
        source_name = self.name
        dump_dir_str = str(self.dump_dir.resolve())

        class _Handler(FileSystemEventHandler):
            def __init__(self):
                super().__init__()
                self._pending: dict = {}
                self._h_lock = threading.Lock()

            def _interesting(self, path: str) -> bool:
                if not path or not path.lower().endswith(".pdf"):
                    return False
                # Defensive: ignore any path under our own dump dir, in
                # the (unusual) case _dump/ is nested inside the source.
                try:
                    return dump_dir_str not in str(Path(path).resolve())
                except OSError:
                    return True

            def _schedule(self, path: str, action: str):
                if not self._interesting(path):
                    return
                with self._h_lock:
                    existing = self._pending.get(path)
                    if existing is not None:
                        existing.cancel()
                    t = threading.Timer(
                        debounce_seconds, self._flush, args=(path, action)
                    )
                    t.daemon = True
                    self._pending[path] = t
                    t.start()

            def _flush(self, path: str, action: str):
                try:
                    if action == "delete":
                        remove(path)
                    else:
                        reprocess(path)
                except Exception as e:
                    _log(f"[watcher:{source_name}] flush failed for {path}: {e}")
                finally:
                    with self._h_lock:
                        self._pending.pop(path, None)

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
        observer.schedule(_Handler(), str(self.source_path), recursive=self.recursive)
        observer.daemon = True
        observer.start()
        self._observer = observer
        _log(f"[watcher:{source_name}] watching {self.source_path} for .pdf changes")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        try:
            chunk_count = self.rag.vector_store._collection.count()
        except Exception:
            chunk_count = None
        drift = None
        try:
            drift = self.rag.check_config_drift()
        except Exception:
            pass

        # Snapshot the state under the lock so a concurrent watcher
        # event can't mutate the dict while we iterate it 4 times.
        with self._lock:
            entries = list(self._extract_state.values())
        n_ok = sum(1 for e in entries if e.status == "ok")
        n_skipped = sum(
            1 for e in entries
            if e.status.startswith("skipped_") or e.status == "error"
        )
        total_pages = sum(e.n_pages for e in entries if e.status == "ok")
        last_extract_at = (max((e.extracted_at for e in entries), default="") or None)

        return {
            "name": self.name,
            "type": self.type_name,
            "path": str(self.source_path),
            "pdf_count": n_ok,
            "skipped_count": n_skipped,
            "total_pages_extracted": total_pages,
            "chunk_count": chunk_count,
            "extractor_backend": self.backend_name,
            "last_extract_at": last_extract_at,
            "last_update": self.rag.metadata.get("last_update"),
            "drift_severity": drift["severity"] if drift else None,
        }

    def drift_status_text(self) -> str:
        return self.rag.drift_status_text()

    def needs_update(self) -> bool:
        """Return True when there's a real drift between the source folder
        and the indexed state — surfaced via `get_rag_status` so the AI
        client / `lynx status` can suggest a rebuild.

        We diff three cheap things (no SHA computation): set of PDFs on
        disk vs set in the state cache, plus a per-file mtime sanity
        check. Heavy SHA verification stays inside `update(force=False)`
        — calling `needs_update()` shouldn't itself walk the bytes.
        """
        try:
            on_disk = set(self._discover_pdfs())
        except Exception:
            return False
        cached = set(self._extract_state.keys())
        if on_disk - cached:
            return True  # new PDFs added since last rebuild
        if cached - on_disk:
            return True  # PDFs removed since last rebuild
        # Same set on both sides — check mtimes against the recorded
        # extracted_at as a cheap "the file was edited" proxy. Real
        # change detection is SHA-based and happens at `update()` time.
        for p in on_disk:
            entry = self._extract_state.get(p)
            if entry is None:
                return True
            try:
                file_mtime = os.path.getmtime(p)
            except OSError:
                continue
            # extracted_at is ISO; compare with the file mtime parsed back
            try:
                from datetime import datetime as _dt
                extracted_dt = _dt.fromisoformat(entry.extracted_at)
                if file_mtime > extracted_dt.timestamp() + 1:  # +1s slack
                    return True
            except (ValueError, TypeError):
                continue
        return False
