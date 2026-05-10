"""
MCP server for the local-codebase-rag-mcp project.

The bulk of the work happens inside `run_server(config_path)`:
  - load config
  - construct CodebaseRAG in a background thread
  - start the file watcher once the RAG is ready
  - register MCP tools and call mcp.run() (blocking)

The stdout-redirect dance at module top must run BEFORE any heavy import,
so it is intentionally a side effect of importing this module — see the
comment by `_REAL_STDOUT_FD` for why.
"""

import os
import sys
import warnings
import threading

# Silence EVERYTHING before any library writes to stdout/stderr.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# CRITICAL for MCP: some libraries write to stdout during import/init
# (llama_index prints "LLM is explicitly disabled. Using MockLLM.";
# sentence_transformers logs the model load; etc.). In MCP stdio mode,
# stdout is the JSON-RPC channel and any spurious byte breaks the protocol
# and freezes tool calls. We save the real fd 1 and redirect it to fd 2 for
# the entire import/loading phase, then restore it just before mcp.run().
_REAL_STDOUT_FD = os.dup(1)
os.dup2(2, 1)

from mcp.server.fastmcp import FastMCP

from .config import load_config


def _restore_real_stdout():
    """Restore fd 1 to the real stdout, draining the Python buffer first."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.dup2(_REAL_STDOUT_FD, 1)


def run_server(config_path=None):
    """Boot the MCP server. Blocks on mcp.run() until the client disconnects.

    `config_path` (str | Path | None) is forwarded to load_config(); when
    None, the standard resolution chain (env var, then ./config.json) is used.
    """
    config = load_config(config_path=config_path)

    mcp = FastMCP("local-codebase-rag-mcp")

    # ----- shared state (closed over by the loader thread, the watcher,
    # and the @mcp.tool functions defined below) -----
    state = {
        "rag_instance": None,
        "rag_loading": False,
        "rag_error": None,
        "rag_ready": threading.Event(),
    }

    def _is_ignored(path: str) -> bool:
        return any(frag in path for frag in config.ignored_path_fragments)

    def _start_file_watcher(rag):
        """Start a watchdog Observer that reacts to changes inside the codebase."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        debounce_seconds = config.watcher.debounce_seconds

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
                    print(f"[watcher] flush failed for {path}: {e}", file=sys.stderr)
                finally:
                    with self._lock:
                        self._pending.pop(path, None)

            def _schedule(self, path: str, action: str):
                if not path or _is_ignored(path):
                    return
                # Let deletes through even if the file is gone; for updates
                # we rely on rag.is_supported() to filter by extension.
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
        observer.schedule(_DebouncedHandler(), str(config.codebase_path), recursive=True)
        observer.daemon = True
        observer.start()
        print(f"[watcher] Watcher active on {config.codebase_path}", file=sys.stderr)

    def _load_rag_background():
        """Load the RAG in a separate thread so we don't block the MCP handshake."""
        state["rag_loading"] = True
        try:
            from .rag_manager import CodebaseRAG
            state["rag_instance"] = CodebaseRAG(
                codebase_path=str(config.codebase_path),
                rag_storage_path=str(config.storage_path),
                supported_extensions=config.supported_extensions,
                embedding_model_name=config.embedding.model_name,
                collection_name=config.collection_name,
                search_mode=config.search.mode,
                rrf_k=config.search.rrf_k,
                candidate_pool_size=config.search.candidate_pool_size,
            )
        except Exception as e:
            state["rag_error"] = str(e)
        finally:
            state["rag_loading"] = False
            # Start the watcher BEFORE releasing rag_ready so the
            # "Watcher active" log is guaranteed to land on stderr before
            # mcp.run() takes the process.
            if state["rag_instance"] is not None and config.watcher.enabled:
                try:
                    _start_file_watcher(state["rag_instance"])
                except Exception as e:
                    print(f"[mcp] file watcher failed to start: {e}", file=sys.stderr)
            state["rag_ready"].set()

    def _get_rag():
        """Return the RAG manager once it's ready; otherwise wait up to the
        configured loading_timeout_seconds."""
        if state["rag_error"]:
            raise RuntimeError(f"RAG failed to load: {state['rag_error']}")
        if not state["rag_ready"].is_set():
            loaded = state["rag_ready"].wait(timeout=config.loading_timeout_seconds)
            if not loaded:
                raise TimeoutError("The RAG is still loading. Try again in a moment.")
        if state["rag_error"]:
            raise RuntimeError(f"RAG failed to load: {state['rag_error']}")
        return state["rag_instance"]

    # ----- MCP tools (close over `config` and `_get_rag`) -----

    @mcp.tool()
    def search_codebase(
        query: str,
        top_k: int | None = None,
        file_glob: str | None = None,
        extensions: list[str] | None = None,
        path_contains: str | None = None,
    ) -> str:
        """Semantic search over the indexed codebase, with optional filters.

        Use this to find functions, classes, patterns, or architectural context.
        Prefer a natural-language description of what the code does over an
        exact identifier - that's where semantic search beats grep.

        Filters (all optional, AND-ed together) let you scope the search to a
        subset of the codebase. They are post-filters: the underlying retriever
        over-fetches and the filters trim the result set, so very narrow filters
        on a large index may return fewer than top_k results.

        Args:
            query: Natural-language search query (e.g. "how damage is dispatched
                   between entities", "factory pattern for spawners",
                   "IBlobSerializer implementations").
            top_k: Number of results to return. Defaults to search.default_top_k
                   from config.json (typically 5).
            file_glob: Optional Unix-shell glob matched against the file name
                       and full path. Examples: "*.cs", "**/Bullet*/**/*.cs",
                       "**/tests/**".
            extensions: Optional list of file extensions to restrict results to.
                        Leading dots are normalized. Examples: [".py"],
                        ["ts", "tsx"]. Combine with `query` for "find X but
                        only in TypeScript files".
            path_contains: Optional substring required in the file path or name.
                           Example: "BulletSystem" returns only chunks whose path
                           contains "BulletSystem".
        """
        try:
            rag = _get_rag()
            effective_top_k = top_k if top_k is not None else config.search.default_top_k
            results = rag.search(
                query,
                top_k=effective_top_k,
                file_glob=file_glob,
                extensions=extensions,
                path_contains=path_contains,
            )

            active_filters = []
            if file_glob:
                active_filters.append(f"file_glob={file_glob!r}")
            if extensions:
                active_filters.append(f"extensions={list(extensions)!r}")
            if path_contains:
                active_filters.append(f"path_contains={path_contains!r}")
            filter_suffix = f" (filters: {', '.join(active_filters)})" if active_filters else ""

            if not results:
                return f"No results found for '{query}'{filter_suffix}."

            formatted = f"Found {len(results)} results for '{query}'{filter_suffix}:\n\n"
            for i, res in enumerate(results, 1):
                formatted += f"--- {i}. File: {res['file']} (Score: {res['score']:.2f}) ---\n"
                formatted += f"{res['content']}\n\n"
            return formatted
        except Exception as e:
            return f"Error during search: {str(e)}"

    @mcp.tool()
    def update_codebase_index(force: bool = False) -> str:
        """Force a full rebuild of the codebase index.

        Useful after a complex merge or when you suspect the live watcher has
        drifted. Day-to-day this is rarely needed: the file watcher keeps the
        index in sync incrementally.

        Args:
            force: If True, rebuild even when no new git commits are detected.
        """
        try:
            rag = _get_rag()
            rag.update(force=force)
            return "Index update completed successfully."
        except Exception as e:
            return f"Error during update: {str(e)}"

    @mcp.tool()
    def get_rag_status() -> str:
        """Return the current state of the RAG index.

        Reports git status (when git_integration is enabled), last update time,
        and any config drift detected since the last full rebuild. A drift
        means the live config no longer matches the config that built the
        current index - the most important case is a changed embedding model,
        which silently invalidates all stored vectors.
        """
        try:
            rag = _get_rag()
            needs_update = rag.needs_update()
            metadata = rag.metadata

            status = "Needs update" if needs_update else "Up to date"
            last_commit = metadata.get("last_commit", "None")
            last_update = metadata.get("last_update", "Never")
            drift_text = rag.drift_status_text()

            return (
                f"Status: {status}\n"
                f"Last commit: {last_commit}\n"
                f"Last update: {last_update}\n"
                f"\n{drift_text}"
            )
        except Exception as e:
            return f"Error reading status: {str(e)}"

    # ----- boot the loader, wait for ready, then hand off to mcp.run() -----

    threading.Thread(target=_load_rag_background, daemon=True).start()

    # Wait for the RAG to load before opening the JSON-RPC transport.
    # This guarantees that any spurious print() (e.g. llama_index's "MockLLM"
    # warning, sentence_transformers logs) lands on stderr and not on the
    # JSON-RPC channel that FastMCP will own a moment later.
    state["rag_ready"].wait(timeout=config.loading_timeout_seconds)

    # Loading phase done: restore fd 1 and start the transport.
    _restore_real_stdout()
    mcp.run()


if __name__ == "__main__":
    # Allow `python -m local_codebase_rag_mcp.server` for ad-hoc invocation.
    # Normal entry point goes through cli:main (`local-codebase-rag-mcp serve`).
    run_server()
