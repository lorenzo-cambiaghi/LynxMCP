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
            # Opt-in cross-encoder reranker (default disabled; see
            # README "Reranking" section for cost/benefit).
            reranker_config=shared_config.search.reranker,
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

    # ------------------------------------------------------------------
    # Combined tools (graph + search hybrid)
    # ------------------------------------------------------------------
    # Unlike the pure-graph methods above, these work for ANY codebase
    # source — graph layer is used when present, with a search-only
    # fallback when it's not. The fallback gives lower-quality results
    # but keeps the tools usable for users who haven't opted into the
    # graph layer yet.

    def find_definition(self, symbol: str, limit: int = 10) -> list:
        """Find where `symbol` is defined.

        Primary: graph lookup (function/class nodes whose label matches
        `symbol`). Returns the precise file + line range from the AST.

        Fallback (graph off, or graph found nothing): BM25 sparse search
        for the literal symbol name. Less precise but usually finds the
        right file when the user gave an exact identifier.

        Each result carries `source` ("graph" or "search_bm25") so the
        AI client can communicate confidence to the user.
        """
        out: list = []
        seen_keys: set = set()

        # Primary path: graph layer
        if self.graph is not None:
            from ..graph import find_symbols
            G = self.graph.graph
            for nid in find_symbols(G, symbol):
                data = G.nodes[nid]
                if data.get("kind") not in ("function", "class"):
                    continue
                key = (data.get("file") or "", data.get("label") or "")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append({
                    "symbol": data.get("label", symbol),
                    "kind": data.get("kind"),
                    "file": data.get("file"),
                    "start_line": data.get("start_line"),
                    "end_line": data.get("end_line"),
                    "lang_key": data.get("lang_key"),
                    "source": "graph",
                })
                if len(out) >= limit:
                    return out

        # Fallback path: BM25 textual search. Even when graph is
        # available, fire this if the primary returned nothing — could
        # be a symbol from a language with no graph rule (markdown,
        # shaders, etc.) that the chunker still picked up.
        if not out:
            try:
                hits = self.rag.search(symbol, top_k=limit, file_glob=None,
                                       extensions=None, path_contains=None)
            except Exception:
                hits = []
            # If the RAG instance is in "sparse" mode this is already
            # exact-match-biased. Otherwise let the hybrid score guide it.
            for h in hits:
                key = (h.get("file_path") or h.get("file") or "",
                       h.get("symbol_name") or "")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append({
                    "symbol": h.get("symbol_name") or symbol,
                    "kind": h.get("symbol_kind"),
                    "file": h.get("file_path") or h.get("file"),
                    "start_line": h.get("start_line"),
                    "end_line": h.get("end_line"),
                    "lang_key": h.get("language"),
                    "score": h.get("score"),
                    "source": "search_bm25",
                })
                if len(out) >= limit:
                    break
        return out

    def find_usages(self, symbol: str, limit: int = 50) -> list:
        """Find every place that uses `symbol` — calls AND non-call
        references (typeof, generics, decorators, imports, comments
        mentioning the name).

        Strategy:
          1. If graph layer is on, start from `get_callers(symbol)` —
             precise structural callers from the call graph.
          2. Always also run a textual BM25 search for the symbol name
             to catch non-call references the graph misses (generics,
             type annotations, decorators, doc references, etc).
          3. Filter OUT chunks that ARE the definition of `symbol`
             (we want USES, not the definition itself).
          4. Dedupe by (file, line range).
        """
        out: list = []
        seen_keys: set = set()

        # 1. Structural callers from the graph
        if self.graph is not None:
            try:
                from ..graph import get_callers as _gc
                for edge in _gc(self.graph.graph, symbol, limit=limit):
                    src = edge.get("source") or {}
                    key = (src.get("file") or "", src.get("start_line") or 0)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    out.append({
                        "symbol": src.get("label"),
                        "kind": src.get("kind"),
                        "file": src.get("file"),
                        "start_line": src.get("start_line"),
                        "end_line": src.get("end_line"),
                        "edge_relation": "calls",
                        "source": "graph",
                    })
                    if len(out) >= limit:
                        return out
            except Exception:
                pass  # graph available but query failed — fall through to search

        # 2. Textual search for non-call uses (always runs)
        try:
            hits = self.rag.search(symbol, top_k=limit)
        except Exception:
            hits = []
        # Lowercase compare so "ApplyDamage" matches "applydamage" etc.
        symbol_lower = symbol.lower()
        for h in hits:
            sym_name = (h.get("symbol_name") or "").lower()
            # Skip the chunk that IS the definition (single-symbol chunk with
            # exact name match — the user wants USAGES, not the def).
            if sym_name == symbol_lower and h.get("symbol_kind") in (
                "function", "function_definition", "method_definition",
                "method_declaration", "class", "class_definition", "class_declaration",
            ):
                continue
            key = (h.get("file_path") or h.get("file") or "", h.get("start_line") or 0)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append({
                "symbol": h.get("symbol_name") or "",
                "kind": h.get("symbol_kind"),
                "file": h.get("file_path") or h.get("file"),
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "score": h.get("score"),
                "source": "search",
            })
            if len(out) >= limit:
                break
        return out

    # Default regex covering the most common test-path conventions:
    # Python pytest (`/tests/`, `/test/`, `_test.py`), JS/TS (`.test.`,
    # `.spec.`, `__tests__/`), C# xUnit/NUnit (`Test.cs`, `Tests.cs`),
    # Go (`_test.go`).
    _DEFAULT_TEST_PATH_PATTERN = (
        r"(^|/)(test|tests|spec|__tests__)/|"
        r"_test\.|\.test\.|\.spec\.|"
        r"Test\.cs$|Tests\.cs$"
    )

    def find_tests_for(
        self,
        symbol: str,
        limit: int = 20,
        test_path_pattern: str | None = None,
    ) -> list:
        """Find test chunks that exercise `symbol`.

        Over-fetches search results (`limit * 3`) then filters by a
        regex on `file_path` that covers the common test conventions.
        Caller can override `test_path_pattern` for custom layouts.
        """
        import re
        pattern = test_path_pattern or self._DEFAULT_TEST_PATH_PATTERN
        # Case-insensitive: file systems mix conventions ("Tests/", "tests/",
        # "Test.cs", "test.cs"). Matching case-sensitively would silently
        # drop tests on Linux/case-sensitive filesystems with unusual casing.
        rx = re.compile(pattern, re.IGNORECASE)
        try:
            hits = self.rag.search(symbol, top_k=limit * 3)
        except Exception:
            return []
        out: list = []
        for h in hits:
            fp = h.get("file_path") or h.get("file") or ""
            # Normalize Windows backslashes for the regex
            fp_normalized = fp.replace("\\", "/")
            if not rx.search(fp_normalized):
                continue
            out.append({
                "symbol": h.get("symbol_name") or "",
                "kind": h.get("symbol_kind"),
                "file": fp,
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "score": h.get("score"),
                "content": h.get("content"),
                "source": "search+test_filter",
            })
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # Diff-aware search (codebase only, requires git)
    # ------------------------------------------------------------------

    # Branches we try in order when the caller doesn't pin `base`. Picked
    # to cover the practical 95%: GitHub default since 2020 (`main`),
    # legacy default (`master`), GitFlow integration branch (`develop`).
    _DEFAULT_BASE_CANDIDATES = ("main", "master", "develop")

    def _resolve_diff_base(self, repo, base: str | None) -> str:
        """Return the branch name to diff against.

        If `base` is given, verify it exists and return it. If `base` is
        None, try `main`, `master`, `develop` in order; return the first
        that resolves. Raise ValueError with a clear message when nothing
        works — the AI client needs to be able to ask the user for a
        specific branch.
        """
        if base:
            try:
                repo.git.rev_parse("--verify", base)
                return base
            except Exception as e:
                raise ValueError(
                    f"search_diff: base branch {base!r} not found in repo: {e}"
                )
        for candidate in self._DEFAULT_BASE_CANDIDATES:
            try:
                repo.git.rev_parse("--verify", candidate)
                return candidate
            except Exception:
                continue
        raise ValueError(
            "search_diff: no default base branch found "
            f"(tried {list(self._DEFAULT_BASE_CANDIDATES)}). "
            "Pass `base=<branch_name>` explicitly."
        )

    def search_diff(
        self,
        query: str,
        base: str | None = None,
        top_k: int = 8,
        **kw,
    ) -> dict:
        """Search restricted to files added/modified vs `base` branch.

        Returns a dict (not a bare list) so the MCP client can see the
        resolved base + the file list alongside the hits — useful for
        diagnostics and for the AI to explain WHY a specific set of
        results was chosen:

            {
              "base": "main",
              "modified_files": ["src/a.py", "src/b.py"],
              "hits": [ ...standard search results... ],
            }

        Raises ValueError if the source folder isn't a git repo or if
        no base branch can be resolved.
        """
        try:
            from git import Repo, InvalidGitRepositoryError, NoSuchPathError
        except ImportError as e:
            raise RuntimeError(
                "search_diff requires GitPython. It should already be a "
                "Lynx dependency; install with `pip install gitpython`."
            ) from e

        try:
            repo = Repo(str(self.source_config["path"]))
        except (InvalidGitRepositoryError, NoSuchPathError) as e:
            raise ValueError(
                f"search_diff: source path {self.source_config['path']!r} "
                f"is not a git repository: {e}"
            )

        resolved_base = self._resolve_diff_base(repo, base)

        # `git diff --name-only --diff-filter=AMR A...B` lists files
        # Added, Modified, or Renamed in B (HEAD) relative to the
        # merge-base with A. We intentionally exclude Deleted files
        # (nothing left to search) but include Renames — the new path
        # is what's on disk and indexed, so a user who renamed
        # `old.py` → `new.py` still gets results from `new.py`.
        try:
            diff_out = repo.git.diff(
                "--name-only", "--diff-filter=AMR",
                f"{resolved_base}...HEAD",
            )
        except Exception as e:
            raise ValueError(
                f"search_diff: git diff failed (base={resolved_base!r}): {e}"
            )
        modified = [line.strip() for line in diff_out.splitlines() if line.strip()]

        if not modified:
            return {
                "base": resolved_base,
                "modified_files": [],
                "hits": [],
                "note": f"No files added/modified vs '{resolved_base}'.",
            }

        hits = self.rag.search(query, top_k=top_k, paths=modified, **kw)
        return {
            "base": resolved_base,
            "modified_files": modified,
            "hits": hits,
        }

    def find_similar(self, snippet: str, top_k: int = 10) -> list:
        """Find chunks structurally / semantically similar to `snippet`.

        Pure dense (semantic) search — BM25 would just bring back chunks
        that share identifiers, which is not the same thing. Truncates
        long snippets (>2000 chars) because the embedding model has a
        context window and over-long inputs muddy the vector.

        Filters out the chunk that IS the snippet (exact content match)
        so the user gets *other* similar code, not their own input back.
        """
        if not snippet or not snippet.strip():
            return []
        # Truncate at 2000 chars: anything longer dilutes the embedding
        # signal for BGE-small (max ~500 effective tokens).
        truncated = snippet[:2000]
        try:
            # Force dense mode regardless of source default — BM25 makes
            # no sense for a multi-line code snippet query.
            #
            # We go straight through `_search_once` with mode="dense"
            # rather than swapping `self.rag.search_mode` around: the
            # mutate-and-restore pattern is NOT thread-safe (two
            # concurrent find_similar calls can leak the override into
            # one another's view) and would also break any concurrent
            # plain search() running on the same backend.
            hits = self.rag._search_once(truncated, top_k=top_k + 1, mode="dense")
        except Exception:
            return []
        out: list = []
        # Strip whitespace once for the identity check
        snippet_stripped = truncated.strip()
        for h in hits:
            content = (h.get("content") or "").strip()
            if content == snippet_stripped:
                continue  # don't echo the query back
            out.append({
                "symbol": h.get("symbol_name") or "",
                "kind": h.get("symbol_kind"),
                "file": h.get("file_path") or h.get("file"),
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "score": h.get("score"),
                "content": h.get("content"),
                "lang_key": h.get("language"),
                "source": "search_dense",
            })
            if len(out) >= top_k:
                break
        return out
