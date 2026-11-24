"""Codebase source backend.

Wraps the existing `CodebaseRAG` (hybrid retrieval, BM25 cache, drift detection,
file watcher, filters, deep_search) so it fits the `SourceBackend` interface.
This keeps the proven retrieval pipeline intact while exposing it through the
multi-source dispatcher.
"""
from __future__ import annotations

import functools
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..file_kinds import TEST_PATH_PATTERN
from .base import SourceBackend


def _rel_paths(method):
    """Give every result of `method` its path inside the source (`rel_path`,
    and `from_rel_path` for a call site), next to the absolute one.

    The tools print the relative path: the absolute one repeats the same
    root on every line, a bare file name is ambiguous in a large tree. Done
    on what the backend returns, so the MCP tools, the CLI, the web UI and
    the HTTP API all get the same field, and JSON consumers keep `file`.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return self._attach_rel(method(self, *args, **kwargs))
    return wrapper


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

        # Several Lynx processes may read one index at the same time, but only
        # one may write it. The first to claim the store owns it and runs the
        # watcher; the rest open the same store as followers and pick the
        # owner's writes up on demand. Failing to claim never blocks reading.
        self._claim_store()

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
            secondary_weight=shared_config.search.secondary_weight,
            # Opt-in cross-encoder reranker (default disabled; see
            # README "Reranking" section for cost/benefit).
            reranker_config=shared_config.search.reranker,
            # Exclude vendored / build dirs from the vector index too — not
            # just the watcher and graph (they used to disagree, polluting
            # search with node_modules etc.).
            ignored_path_fragments=source_config.get("ignored_path_fragments") or [],
            ignored_globs=source_config.get("ignored_globs") or (),
            follower=not self.is_owner,
        )

        # Watcher state — populated lazily by start_watcher().
        self._observer = None
        self._handler = None

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
                ignored_globs=source_config.get("ignored_globs") or (),
                follower=not self.is_owner,
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
        mode=None,
        **_ignored,
    ) -> list[dict]:
        self._before_read()
        return self.rag.search(
            query,
            top_k=top_k,
            file_glob=file_glob,
            extensions=extensions,
            path_contains=path_contains,
            mode=mode,
        )

    def search_batch(
        self,
        queries,
        top_k: int = 5,
        *,
        file_glob=None,
        extensions=None,
        path_contains=None,
        **_ignored,
    ) -> list:
        """Search multiple queries with a single batched embedding call.
        Returns one result list per query (aligned to `queries`)."""
        self._before_read()
        return self.rag.search_batch(
            queries,
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
        self._before_read()
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
        # A follower must not write. Raising here (rather than in the engine)
        # keeps the graph rebuild below from running either.
        if not self.is_owner:
            from ..errors import StoreNotOwnedError
            raise StoreNotOwnedError(self.name, str(self.storage_dir))
        self.rag.update(force=force)
        # Keep the graph layer in sync with the same trigger. We do this
        # after the RAG update so a partial failure in the graph layer
        # doesn't leave the vector index half-built.
        self._rebuild_graph(from_scratch=False)

    def catch_up(self) -> None:
        """Index what changed while nobody was watching: a pass over every
        file's hash, in the vector index and in the graph. Owner only."""
        self.update(force=True)

    def _rebuild_graph(self, *, from_scratch: bool) -> None:
        """Bring the graph in line with the files on disk.

        `force` on `update` means "look at every file", which for the vector
        index is a pass over its SHA cache. The graph's equivalent is its
        delta rebuild, which yields the same graph as starting over (and
        starts over by itself when the extractor changed). Wiping it is for
        `reset` alone: it cost 17 s on a 3,500-file source where the delta
        takes about 2.5 s when nothing changed, most of it hashing files.
        """
        if self.graph is None:
            return
        try:
            self.graph.rebuild(force=from_scratch)
        except Exception as e:
            # Failures here must not break search — log and move on.
            print(
                f"[graph:{self.name}] rebuild failed (search still works): {e}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe this source's index + graph and rebuild from scratch, in place.

        Empties the Chroma collection through its API (no storage-dir delete, so
        it works while the client holds the store open) and clears the SHA cache
        so the rebuild re-indexes everything; the graph is rebuilt from scratch
        too."""
        if not self.is_owner:
            from ..errors import StoreNotOwnedError
            raise StoreNotOwnedError(self.name, str(self.storage_dir))
        self.rag.reset_index()
        self.rag.update(force=True)
        self._rebuild_graph(from_scratch=True)

    def stop_watcher(self) -> None:
        """Stop the watchdog Observer and give up ownership of the store.

        Idempotent. Releasing here (as well as at process exit) is what lets a
        `lynx build` hand the store back to a later session without waiting for
        the process to end.
        """
        observer = self._observer
        handler = getattr(self, "_handler", None)
        try:
            if observer is not None:
                observer.stop()
                observer.join(timeout=5)
        except Exception:
            pass
        finally:
            self._observer = None
            # Stopping the observer does not cancel the debounce timers it
            # already armed: without this they fire seconds later and write to
            # a store this process may no longer own.
            if handler is not None:
                try:
                    with handler._lock:
                        pending = list(handler._pending.values())
                        handler._pending.clear()
                    for timer in pending:
                        timer.cancel()
                except Exception:
                    pass
                self._handler = None
            self._release_store()

    def start_watcher(self) -> None:
        """Start a watchdog Observer over this source's `path`.

        No-op when `watcher.enabled` is False in source_config. Idempotent:
        calling twice does not start two observers.

        Also a no-op when another process owns this store: two watchers on one
        index would write the same chunks from two processes. The owner's
        watcher keeps the index current and this session reads the result.
        """
        watcher_cfg = self.source_config.get("watcher") or {}
        if not watcher_cfg.get("enabled", True):
            return
        if self._observer is not None:
            return
        if not self.is_owner:
            print(
                f"[watcher:{self.name}] not started: another Lynx process owns "
                f"this index and watches the files. Searches here follow its "
                f"updates.",
                file=sys.stderr,
            )
            return

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        debounce_seconds = float(watcher_cfg.get("debounce_seconds", 2.0))
        backend = self
        ignored_fragments = tuple(self.source_config.get("ignored_path_fragments") or ())
        ignored_globs = tuple(self.source_config.get("ignored_globs") or ())
        root = self.source_config["path"]
        rag = self.rag
        graph = self.graph
        source_name = self.name

        extensions = tuple(self.source_config.get("supported_extensions") or ())

        def _in_scope(path: str) -> bool:
            # The rule the index build walks the tree with, for one path.
            from ..fs_scan import in_scope
            return in_scope(path, root, extensions, ignored_fragments, ignored_globs)

        class _DebouncedHandler(FileSystemEventHandler):
            def __init__(self):
                super().__init__()
                self._pending: dict = {}
                self._lock = threading.Lock()

            def _flush(self, path: str, action: str):
                try:
                    if not backend.is_owner:
                        # Stood down (or stopped) between the event and this
                        # timer firing. Writing now would be the second writer
                        # ownership exists to prevent.
                        return
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
                if not path or not _in_scope(path):
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

        handler = _DebouncedHandler()
        self._handler = handler
        observer = Observer()
        observer.schedule(
            handler, str(self.source_config["path"]), recursive=True
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
        self._before_read()
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

    def _attach_rel(self, obj):
        """See `_rel_paths`. Walks result dicts, the nodes of graph edges
        (`source` / `target` / `node`, a path's `nodes`) and nested lists."""
        from ..fs_scan import to_rel_path
        root = self.source_config.get("path")

        def fix(item):
            if isinstance(item, list):
                for x in item:
                    fix(x)
                return
            if not isinstance(item, dict):
                return
            f = item.get("file")
            if isinstance(f, str) and f and "rel_path" not in item and os.path.isabs(f):
                item["rel_path"] = to_rel_path(f, root)
            ff = item.get("from_file")
            if isinstance(ff, str) and ff and "from_rel_path" not in item:
                item["from_rel_path"] = to_rel_path(ff, root)
            for key in ("source", "target", "node", "nodes", "definition", "called_by",
                        "calls", "superclasses", "subclasses", "tests", "callers",
                        "symbols", "imports", "god_nodes"):
                if isinstance(item.get(key), (dict, list)):
                    fix(item[key])

        fix(obj)
        return obj

    # ------------------------------------------------------------------
    # Graph layer pass-through (only meaningful when graph.enabled=true)
    # ------------------------------------------------------------------
    # The SourceManager / MCP server check `hasattr(backend, "graph") and
    # backend.graph is not None` before registering the graph tools, so
    # these methods are only invoked when the layer is active.

    @_rel_paths
    def get_callers(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_callers as _q
        return _q(self.graph.graph, symbol, limit=limit)

    @_rel_paths
    def get_callees(self, symbol: str, limit: int = 50) -> list:
        from ..graph import get_callees as _q
        return _q(self.graph.graph, symbol, limit=limit)

    @_rel_paths
    def get_subclasses(self, symbol: str, limit: int = 50, depth: int = 1) -> list:
        from ..graph import get_subclasses as _q
        return _q(self.graph.graph, symbol, limit=limit, depth=depth)

    @_rel_paths
    def get_superclasses(self, symbol: str, limit: int = 50, depth: int = 1) -> list:
        from ..graph import get_superclasses as _q
        return _q(self.graph.graph, symbol, limit=limit, depth=depth)

    def audit(self, path: str, per_principle: int = 5) -> str:
        """SOLID findings for the package at `path`, each with its fence (see
        lynx.audit.solid)."""
        from ..audit.solid import audit as _audit, render
        root = Path(str(self.source_config["path"]))
        findings, rel = _audit(self.graph.graph, root, path, repo=root)
        return render(rel, findings, per_principle=per_principle)

    @_rel_paths
    def get_imports(self, file_or_symbol: str, limit: int = 100) -> list:
        from ..graph import get_imports as _q
        return _q(self.graph.graph, file_or_symbol, limit=limit)

    @_rel_paths
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

    @_rel_paths
    def shortest_path(self, source: str, target: str, max_hops: int = 8):
        from ..graph import shortest_path as _q
        return _q(self.graph.graph, source, target, max_hops=max_hops)

    @_rel_paths
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

    def _definitions(self, symbol: str, limit: int) -> list:
        """[(node id or None, result dict)] for `symbol`'s definitions: graph
        nodes when the layer is on, chunks named after it otherwise."""
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
                out.append((nid, {
                    "symbol": data.get("label", symbol),
                    "kind": data.get("kind"),
                    "file": data.get("file"),
                    "start_line": data.get("start_line"),
                    "end_line": data.get("end_line"),
                    "lang_key": data.get("lang_key"),
                    "source": "graph",
                }))
                if len(out) >= limit:
                    return out

        # Fallback path: BM25 over the chunks. Even when graph is
        # available, fire this if the primary returned nothing — could
        # be a symbol from a language with no graph rule (markdown,
        # shaders, etc.) that the chunker still picked up. Chunks named
        # after the symbol come first; a mention is the best left after.
        if not out:
            try:
                hits = self.rag.lexical_search(symbol, top_k=max(limit * 3, 10))
            except Exception:
                hits = []
            leaf = _leaf(symbol)
            named = [h for h in hits if _leaf(h.get("symbol_name") or "") == leaf]
            for h in named or hits:
                key = (h.get("file_path") or h.get("file") or "",
                       h.get("symbol_name") or "")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append((None, {
                    "symbol": h.get("symbol_name") or symbol,
                    "kind": h.get("symbol_kind"),
                    "file": h.get("file_path") or h.get("file"),
                    "start_line": h.get("start_line"),
                    "end_line": h.get("end_line"),
                    "lang_key": h.get("language"),
                    "score": h.get("score"),
                    "source": "search_bm25",
                }))
                if len(out) >= limit:
                    break
        return out

    @staticmethod
    def _with_code(results: list, files, *, bodies: int = 1) -> list:
        """Attach the code: the body of the first `bodies` results, the
        signature and doc line of the rest."""
        from ..snippets import symbol_body, symbol_outline
        for i, r in enumerate(results):
            if i < bodies:
                body, cut = symbol_body(files, r.get("file"), r.get("start_line"),
                                        r.get("end_line"))
                if body:
                    r["body"], r["body_cut"] = body, cut
                    continue
            sig, doc = symbol_outline(files, r.get("file"), r.get("start_line"),
                                      r.get("end_line"), r.get("kind") or "")
            if sig:
                r["signature"] = sig
            if doc:
                r["doc"] = doc
        return results

    @_rel_paths
    def find_definition(self, symbol: str, limit: int = 10) -> list:
        """Find where `symbol` is defined, with its code.

        Primary: graph lookup (function/class nodes whose label matches
        `symbol`). Returns the precise file + line range from the AST.

        Fallback (graph off, or graph found nothing): BM25 sparse search
        for the literal symbol name. Less precise but usually finds the
        right file when the user gave an exact identifier.

        Each result carries `source` ("graph" or "search_bm25") so the
        AI client can communicate confidence to the user. The first one
        carries its body (`body`, capped; `body_cut` lines left out), the
        others their `signature` and `doc`, so the answer needs no read.
        """
        self._before_read()
        from ..snippets import FileLines
        results = [r for _nid, r in self._definitions(symbol, limit)]
        return self._with_code(results, FileLines())

    @_rel_paths
    def find_usages(self, symbol: str, limit: int = 50) -> list:
        """Find every place that uses `symbol` — calls AND non-call
        references (typeof, generics, decorators, imports, comments
        mentioning the name).

        Strategy:
          1. If graph layer is on, start from `get_callers(symbol)` —
             precise structural callers from the call graph — and, for a
             type, the types that inherit from it.
          2. Always also run a lexical (BM25) search for the name, and keep
             only the chunks that contain it as a whole word: the hybrid
             search used here before let in code that merely looked
             similar. Each keeps the lines it appears on.
          3. Filter OUT chunks that ARE the definition of `symbol`
             (we want USES, not the definition itself).
          4. Drop a textual hit inside a function the graph already listed.
        """
        self._before_read()
        out: list = []
        seen_keys: set = set()
        spans: list = []   # (file, start, end) already reported

        def add(entry: dict) -> bool:
            key = (_norm_path(entry.get("file")), entry.get("start_line") or 0)
            if key in seen_keys:
                return False
            seen_keys.add(key)
            spans.append((key[0], entry.get("start_line") or 0,
                          entry.get("end_line") or entry.get("start_line") or 0))
            out.append(entry)
            return True

        # 1. Structural callers (and subtypes) from the graph
        if self.graph is not None:
            try:
                from ..graph import get_callers as _gc, get_subclasses as _gs
                edges = [(e, "calls") for e in _gc(self.graph.graph, symbol, limit=limit)]
                edges += [(e, "inherits") for e in _gs(self.graph.graph, symbol, limit=limit)]
                for edge, relation in edges:
                    src = edge.get("source") or {}
                    entry = {
                        "symbol": src.get("label"),
                        "kind": src.get("kind"),
                        "file": src.get("file"),
                        "start_line": src.get("start_line"),
                        "end_line": src.get("end_line"),
                        "edge_relation": relation,
                        "source": "graph",
                    }
                    if relation == "inherits" and edge.get("base_kind"):
                        entry["base_kind"] = edge["base_kind"]
                    add(entry)
                    if len(out) >= limit:
                        return out
            except Exception:
                pass  # graph available but query failed — fall through to search

        # 2. Lexical search for non-call uses, verified word by word
        try:
            hits = self.rag.lexical_search(symbol, top_k=limit * 2)
        except Exception:
            hits = []
        leaf = _leaf(symbol)
        for h in hits:
            if _is_definition_of(h, leaf):
                continue
            lines = _lines_mentioning(h, leaf)
            if not lines:
                continue                     # BM25 matched parts of the name only
            path = _norm_path(h.get("file_path") or h.get("file"))
            if any(f == path and s <= lines[0] <= e for f, s, e in spans):
                continue                     # inside a caller already listed
            add({
                "symbol": h.get("symbol_name") or "",
                "kind": h.get("symbol_kind"),
                "file": h.get("file_path") or h.get("file"),
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "match_lines": lines,
                "score": h.get("score"),
                "source": "search",
            })
            if len(out) >= limit:
                break
        return out

    # The conventions live in file_kinds, shared with search ranking.
    _DEFAULT_TEST_PATH_PATTERN = TEST_PATH_PATTERN

    @_rel_paths
    def find_tests_for(
        self,
        symbol: str,
        limit: int = 20,
        test_path_pattern: str | None = None,
    ) -> list:
        """Find the tests that exercise `symbol`.

        1. With the graph on: every test function that reaches `symbol`
           through the call graph, up to four calls away. A test of
           `HandleHit` that ends up in `ApplyDamage` never names it, so no
           text search finds it; `depth` says how far it is.
        2. The tests that name it: a lexical search among test files only
           (the ranking happens there, not across the whole index), kept
           when the name is in the chunk as a whole word.

        Test files are told apart by a regex on the path covering the
        common conventions; `test_path_pattern` replaces it.
        """
        self._before_read()
        import re
        pattern = test_path_pattern or self._DEFAULT_TEST_PATH_PATTERN
        # Case-insensitive: file systems mix conventions ("Tests/", "tests/",
        # "Test.cs", "test.cs"). Matching case-sensitively would silently
        # drop tests on Linux/case-sensitive filesystems with unusual casing.
        rx = re.compile(pattern, re.IGNORECASE)

        def is_test(path) -> bool:
            return bool(path) and bool(rx.search(str(path).replace("\\", "/")))

        out: list = []
        spans: list = []
        if self.graph is not None:
            try:
                from ..graph import transitive_callers as _tc
                for c in _tc(self.graph.graph, symbol, max_depth=4, limit=500):
                    node = c.get("node") or {}
                    if node.get("kind") != "function" or not is_test(node.get("file")):
                        continue
                    out.append({
                        "symbol": node.get("label") or "",
                        "kind": node.get("kind"),
                        "file": node.get("file"),
                        "start_line": node.get("start_line"),
                        "end_line": node.get("end_line"),
                        "depth": c.get("depth"),
                        "source": "graph",
                    })
                    spans.append((_norm_path(node.get("file")), node.get("start_line") or 0,
                                  node.get("end_line") or 0))
                    if len(out) >= limit:
                        return out
            except Exception:
                pass

        try:
            hits = self.rag.lexical_search(symbol, top_k=limit * 3, path_filter=is_test)
        except Exception:
            return out
        leaf = _leaf(symbol)
        for h in hits:
            lines = _lines_mentioning(h, leaf)
            if not lines:
                continue
            path = _norm_path(h.get("file_path") or h.get("file"))
            if any(f == path and s <= lines[0] <= e for f, s, e in spans):
                continue
            out.append({
                "symbol": h.get("symbol_name") or "",
                "kind": h.get("symbol_kind"),
                "file": h.get("file_path") or h.get("file"),
                "start_line": h.get("start_line"),
                "end_line": h.get("end_line"),
                "match_lines": lines,
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
        self._before_read()
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

    @_rel_paths
    def find_similar(self, snippet: str, top_k: int = 10) -> list:
        """Find chunks structurally / semantically similar to `snippet`.

        Pure dense (semantic) search — BM25 would just bring back chunks
        that share identifiers, which is not the same thing. Truncates
        long snippets (>2000 chars) because the embedding model has a
        context window and over-long inputs muddy the vector.

        Filters out the chunk that IS the snippet (exact content match)
        so the user gets *other* similar code, not their own input back.
        """
        self._before_read()
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

    @_rel_paths
    def describe_symbol(
        self,
        symbol: str,
        *,
        file: str | None = None,
        callers_limit: int = 10,
        callees_limit: int = 10,
        tests_limit: int = 5,
        members_limit: int = 60,
        subtypes_limit: int = 15,
    ) -> dict:
        """One-shot context for a symbol — for an AI orienting around it.

        Composes the cheap primitives into a single round-trip instead of
        the agent issuing find_definition + graph callers + graph callees +
        find_tests_for + read_file separately:
          - definition  (precise file+line, graph or BM25 fallback; the first
                         one with its body, the others with their signature)
          - called_by   (graph callers — empty when the graph layer is off)
          - calls       (graph callees — empty when the graph layer is off)
          - tests       (tests that reach or name the symbol)
        For a type, its blueprint instead of a body a thousand lines long:
          - superclasses / subclasses (graph inheritance)
          - members     (what it contains, each with its signature)

        Definition + tests work without the graph; the rest needs it.
        """
        self._before_read()
        from ..snippets import FileLines, symbol_outline
        files = FileLines()
        # Every definition of the name, then the ones in `file` when given:
        # a name like `rig` (a helper in dozens of test files) used to come
        # back as the first three definitions found, none of them the one
        # the reader meant, with no word that others existed.
        every = self._definitions(symbol, 500)
        if file:
            wanted = file.replace("\\", "/").lower()
            chosen = [d for d in every if wanted in (d[1].get("file") or "").replace("\\", "/").lower()]
            every = chosen + [d for d in every if d not in chosen] if chosen else every
        defs = every[:3]
        from ..fs_scan import to_rel_path
        root = self.source_config.get("path")
        others = [to_rel_path(d[1].get("file"), root) for d in every[3:] if d[1].get("file")]
        top_nid, top = defs[0] if defs else (None, None)
        is_type = bool(top and top.get("kind") == "class")
        definition = [r for _nid, r in defs]
        # A type's body is its members, listed below with their signatures.
        self._with_code(definition, files, bodies=0 if is_type else 1)
        result = {
            "symbol": symbol,
            "graph_enabled": self.graph is not None,
            "definition": definition,
            "is_type": is_type,
            "called_by": [],
            "calls": [],
            "superclasses": [],
            "subclasses": [],
            "members": [],
            "tests": self.find_tests_for(symbol, limit=tests_limit),
            "other_definitions": len(others),
            "other_definition_files": sorted(set(others))[:10],
        }
        if self.graph is not None:
            if top_nid is not None:
                # The callers and callees of the definition shown, not of
                # every node that shares its name.
                from ..graph.query import callees_of_node, callers_of_node
                G = self.graph.graph
                result["called_by"] = callers_of_node(G, top_nid, limit=callers_limit)
                result["calls"] = callees_of_node(G, top_nid, limit=callees_limit)
            else:
                seed = top.get("symbol") if top else symbol
                result["called_by"] = self.get_callers(seed, limit=callers_limit)
                result["calls"] = self.get_callees(seed, limit=callees_limit)
            seed = top.get("symbol") if top else symbol
            if is_type:
                result["superclasses"] = self.get_superclasses(seed, limit=subtypes_limit)
                result["subclasses"] = self.get_subclasses(seed, limit=subtypes_limit)
                if top_nid is not None:
                    G = self.graph.graph
                    members = []
                    for _u, v, data in G.out_edges(top_nid, data=True):
                        if data.get("relation") != "contains":
                            continue
                        node = G.nodes[v]
                        members.append({
                            "label": node.get("label", v),
                            "kind": node.get("kind"),
                            "start_line": node.get("start_line"),
                            "end_line": node.get("end_line"),
                        })
                    members.sort(key=lambda m: m.get("start_line") or 0)
                    for m in members[:members_limit]:
                        sig, _doc = symbol_outline(files, top.get("file"), m.get("start_line"),
                                                   m.get("end_line"), m.get("kind") or "")
                        if sig:
                            m["signature"] = sig
                    result["members"] = members[:members_limit]
        return result

    @_rel_paths
    def impact_of(
        self,
        symbol: str,
        *,
        max_depth: int = 3,
        limit: int = 200,
        tests_limit: int = 10,
    ) -> dict:
        """Blast radius of changing `symbol`: who reaches it transitively
        through the call graph, plus the tests that exercise it.

        Needs the graph layer for the transitive callers (`graph_enabled` is
        False otherwise, with an empty `callers`); tests still resolve via
        search either way. Answers "if I change X, what could break and what
        should I re-run?".
        """
        self._before_read()
        result = {
            "symbol": symbol,
            "graph_enabled": self.graph is not None,
            "callers": [],
            "tests": self.find_tests_for(symbol, limit=tests_limit),
        }
        if self.graph is not None:
            from ..graph import transitive_callers as _q
            result["callers"] = _q(
                self.graph.graph, symbol, max_depth=max_depth, limit=limit,
            )
        return result

    # A fragment that names more files than this is not a module.
    _MODULE_MAX_FILES = 5

    def _module_files(self, fragment: str) -> list:
        """The indexed files `fragment` names (case-insensitive, either slash).
        A file whose path ends with it beats one that merely contains it, so
        `utils.py` is that file and not every `test_utils.py` too."""
        needle = (fragment or "").lower().replace("\\", "/").strip("/")
        if not needle:
            return []
        paths = sorted(self.rag.indexed_files())
        hits = [p for p in paths if needle in p.lower().replace("\\", "/")]
        exact = [p for p in hits
                 if p.lower().replace("\\", "/").endswith("/" + needle)
                 or p.lower().replace("\\", "/") == needle]
        return exact or hits

    @_rel_paths
    def module_summary(self, file_or_symbol: str, *, limit: int = 200) -> dict:
        """A file as a unit: the symbols it defines with their signatures and
        doc lines, what it imports, and which other files depend on it.

        Works without the graph layer too, from the file's own chunks; the
        graph adds the resolved imports and the dependents. Lets an AI grasp
        a unit without reading the whole file.
        """
        self._before_read()
        from ..snippets import FileLines, symbol_outline
        files = self._module_files(file_or_symbol)
        result = {
            "file": file_or_symbol,
            "graph_enabled": self.graph is not None,
            "files": [],
            "symbols": [],
            "imports": [],
            "dependent_files": [],
        }
        if not files:
            return result
        truncated = len(files) > self._MODULE_MAX_FILES
        files = files[:self._MODULE_MAX_FILES]
        root = self.source_config.get("path")
        from ..fs_scan import to_rel_path
        result["files"] = [to_rel_path(f, root) for f in files] + (["..."] if truncated else [])
        wanted = {_norm_path(f) for f in files}
        lines = FileLines()

        if self.graph is not None:
            from ..graph import get_imports
            G = self.graph.graph
            symbols = []
            symbol_ids, file_ids = [], []
            for _nid, data in G.nodes(data=True):
                if data.get("kind") == "file" and _norm_path(data.get("file")) in wanted:
                    file_ids.append(_nid)
                if data.get("kind") not in ("function", "class"):
                    continue
                if _norm_path(data.get("file")) not in wanted:
                    continue
                symbol_ids.append(_nid)
                symbols.append({
                    "label": data.get("label"),
                    "kind": data.get("kind"),
                    "file": data.get("file"),
                    "start_line": data.get("start_line"),
                    "end_line": data.get("end_line"),
                })
            symbols.sort(key=lambda s: (s.get("file") or "", s.get("start_line") or 0))
            symbols = symbols[:limit]
            imports = []
            for f in files:
                imports.extend(get_imports(G, f, limit=100))
            result["imports"] = imports
            # Dependent files: the files that import this one, and those whose
            # code calls a symbol defined here (its own file excluded). By
            # node, not by name: by name, a file defining `rig` listed the
            # callers of every `rig` in the project.
            dependents: set = set()
            for fid in file_ids:
                for u, _v, d in G.in_edges(fid, data=True):
                    if d.get("relation") in ("imports", "imports_from"):
                        f = G.nodes[u].get("file")
                        if f and _norm_path(f) not in wanted:
                            dependents.add(f)
            for nid in symbol_ids:
                for u, _v, d in G.in_edges(nid, data=True):
                    if d.get("relation") == "calls":
                        f = G.nodes[u].get("file")
                        if f and _norm_path(f) not in wanted:
                            dependents.add(f)
            # Relative, like `files`: 33 absolute paths of one deep checkout
            # were 3,500 characters of the same prefix (Sketchbook, 2026-11-21).
            result["dependent_files"] = sorted(to_rel_path(f, root) for f in dependents)
        else:
            from ..chunking import chunk_file
            symbols = []
            for f in files:
                text = lines.get(f)
                if text is None:
                    continue
                seen = set()
                for c in chunk_file(f, "\n".join(text)):
                    name = (c.get("symbol_name") or "").split("#part")[0]
                    if not name or name.startswith("<") or name in seen:
                        continue
                    seen.add(name)
                    symbols.append({
                        "label": name, "kind": c.get("symbol_kind"), "file": f,
                        "start_line": c.get("start_line"), "end_line": c.get("end_line"),
                    })
            symbols = symbols[:limit]

        for s in symbols:
            sig, doc = symbol_outline(lines, s.get("file"), s.get("start_line"),
                                      s.get("end_line"), s.get("kind") or "")
            if sig:
                s["signature"] = sig
            if doc:
                s["doc"] = doc
        result["symbols"] = symbols
        return result

    def repo_overview(self) -> dict:
        """Orientation map for this codebase: languages, manifests, frameworks,
        likely entry points, and build/test/run commands. Pure filesystem scan
        (no graph, no index needed) — the 'what is this and where do I start'
        answer for an AI landing in an unfamiliar repository.
        """
        self._before_read()
        from ..overview import build_overview
        return build_overview(self.source_config["path"])

    def export_graph(self, mode: str, target: str, *, depth: int = 2) -> dict:
        """Render a self-contained graph view (see graph/render.py).

        mode='symbol' → blast radius of `target` (a symbol name);
        mode='module' → hub view of `target` (a file path / fragment).
        Returns {content, suggested_name} on success, or {empty, reason} when
        the graph is off or the target isn't found.
        """
        self._before_read()
        if self.graph is None:
            return {"empty": True, "reason": "graph layer not enabled for this source"}
        import re as _re
        from .. import __version__
        from ..graph import build_symbol_view, build_module_view, render_html

        G = self.graph.graph
        root = self.source_config.get("path")
        if mode == "symbol":
            model = build_symbol_view(G, target, depth=depth, root=root)
            stem = target
        elif mode == "module":
            model = build_module_view(G, target, root=root)
            stem = os.path.basename(target)
        else:
            raise ValueError(f"mode must be 'symbol' or 'module', got {mode!r}")

        content = render_html(model, source=self.name, lynx_version=__version__)
        safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "graph"
        return {
            "empty": bool(model.get("empty")),
            "reason": model.get("reason"),
            "content": content,
            "suggested_name": f"{mode}_{safe}.html",
        }


def _norm_path(path) -> str:
    """One spelling per file, for comparing paths from the graph and the index."""
    return os.path.normcase(os.path.normpath(str(path))) if path else ""


def _leaf(symbol: str) -> str:
    """`Health.ApplyDamage` -> `applydamage`; `Foo#part2` -> `foo`."""
    return (symbol or "").split("#part")[0].split(".")[-1].strip().lower()


# Kinds of chunk that are no declaration of anything.
_NOT_DECLARATIONS = frozenset({"module_header", "text_window", ""})


def _is_definition_of(hit: dict, leaf: str) -> bool:
    """True for the chunk that declares the symbol: its name ends with it."""
    return ((hit.get("symbol_kind") or "") not in _NOT_DECLARATIONS
            and _leaf(hit.get("symbol_name") or "") == leaf)


def _lines_mentioning(hit: dict, leaf: str) -> list:
    """The file lines of a chunk where `leaf` appears as a whole word
    (case-insensitive): what makes a lexical hit a usage, not a chunk that
    shares a word with the name."""
    import re
    if not leaf:
        return []
    rx = re.compile(r"(?<!\w)" + re.escape(leaf) + r"(?!\w)", re.IGNORECASE)
    first = hit.get("start_line") or 1
    return [first + i for i, line in enumerate((hit.get("content") or "").splitlines())
            if rx.search(line)]
