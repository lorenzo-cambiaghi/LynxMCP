"""
MCP server for the local-codebase-rag-mcp project.

Loads the RAG index in a background thread BEFORE handing the process to
FastMCP's JSON-RPC transport. Configuration comes from `config.json`
(see config.py for the resolution order).
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

from config import load_config

# Load config once at startup. Any path / setting comes from config.json.
CONFIG = load_config()

# Initialize the FastMCP server (instant, no heavy loading here).
mcp = FastMCP("local-codebase-rag-mcp")

# ============================================================================
# Background RAG loading
# ============================================================================
_rag_instance = None
_rag_loading = False
_rag_ready = threading.Event()
_rag_error = None


def _load_rag_background():
    """Load the RAG in a separate thread so we don't block the MCP handshake."""
    global _rag_instance, _rag_loading, _rag_error
    _rag_loading = True
    try:
        from rag_manager import CodebaseRAG
        _rag_instance = CodebaseRAG(
            codebase_path=str(CONFIG.codebase_path),
            rag_storage_path=str(CONFIG.storage_path),
            supported_extensions=CONFIG.supported_extensions,
            embedding_model_name=CONFIG.embedding.model_name,
            collection_name=CONFIG.collection_name,
            search_mode=CONFIG.search.mode,
            rrf_k=CONFIG.search.rrf_k,
            candidate_pool_size=CONFIG.search.candidate_pool_size,
        )
    except Exception as e:
        _rag_error = str(e)
    finally:
        _rag_loading = False
        # Start the watcher BEFORE releasing _rag_ready so the "Watcher active"
        # log is guaranteed to land on stderr before mcp.run() takes the
        # process.
        if _rag_instance is not None and CONFIG.watcher.enabled:
            try:
                _start_file_watcher(_rag_instance)
            except Exception as e:
                print(f"[mcp] file watcher failed to start: {e}", file=sys.stderr)
        _rag_ready.set()


# ============================================================================
# File watcher: incremental index updates on file save (debounced)
# ============================================================================


def _is_ignored(path: str) -> bool:
    return any(frag in path for frag in CONFIG.ignored_path_fragments)


def _start_file_watcher(rag):
    """Start a watchdog Observer that reacts to changes inside the codebase."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    debounce_seconds = CONFIG.watcher.debounce_seconds

    class _DebouncedHandler(FileSystemEventHandler):
        def __init__(self):
            super().__init__()
            self._pending: dict[str, threading.Timer] = {}
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
            # Let deletes through even if the file is gone; for updates we
            # rely on rag.is_supported() to filter by extension.
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
    observer.schedule(_DebouncedHandler(), str(CONFIG.codebase_path), recursive=True)
    observer.daemon = True
    observer.start()
    print(f"[watcher] Watcher active on {CONFIG.codebase_path}", file=sys.stderr)


# Kick off background loading immediately.
_loader_thread = threading.Thread(target=_load_rag_background, daemon=True)
_loader_thread.start()


def _get_rag():
    """Return the RAG manager once it's ready; otherwise wait up to the
    configured timeout (default 600s, override via loading_timeout_seconds)."""
    if _rag_error:
        raise RuntimeError(f"RAG failed to load: {_rag_error}")

    if not _rag_ready.is_set():
        loaded = _rag_ready.wait(timeout=CONFIG.loading_timeout_seconds)
        if not loaded:
            raise TimeoutError("The RAG is still loading. Try again in a moment.")

    if _rag_error:
        raise RuntimeError(f"RAG failed to load: {_rag_error}")

    return _rag_instance


@mcp.tool()
def search_codebase(query: str, top_k: int | None = None) -> str:
    """Semantic search over the indexed codebase.

    Use this to find functions, classes, patterns, or architectural context.
    Prefer a natural-language description of what the code does over an
    exact identifier - that's where semantic search beats grep.

    Args:
        query: Natural-language search query (e.g. "how damage is dispatched",
               "factory pattern for spawners").
        top_k: Number of results to return. Defaults to search.default_top_k
               from config.json (typically 5).
    """
    try:
        rag = _get_rag()
        effective_top_k = top_k if top_k is not None else CONFIG.search.default_top_k
        results = rag.search(query, top_k=effective_top_k)

        if not results:
            return f"No results found for '{query}'."

        formatted = f"Found {len(results)} results for '{query}':\n\n"
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
    and any config drift detected since the last full rebuild. A drift means
    the live config no longer matches the config that built the current
    index — the most important case is a changed embedding model, which
    silently invalidates all stored vectors.
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


def _restore_real_stdout():
    """Restore fd 1 to the real stdout, draining the Python buffer first."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os.dup2(_REAL_STDOUT_FD, 1)


if __name__ == "__main__":
    # Wait for the RAG to load before opening the JSON-RPC transport.
    # This guarantees that any spurious print() (e.g. llama_index's "MockLLM"
    # warning, sentence_transformers logs) lands on stderr and not on the
    # JSON-RPC channel that FastMCP will own a moment later. With an existing
    # index this is 10-15s; the first full build can take much longer.
    _rag_ready.wait(timeout=CONFIG.loading_timeout_seconds)

    # Loading phase done: restore fd 1 and start the transport.
    _restore_real_stdout()
    mcp.run()
