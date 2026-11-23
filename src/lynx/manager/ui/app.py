"""FastAPI application factory + uvicorn launcher for `lynx manager ui`.

Architecture
------------
- One single FastAPI app that serves HTML pages (Jinja2 templates) AND
  JSON API endpoints. The HTML pages use HTMX for interactivity so we
  don't need a separate single-page-app build pipeline.
- Static files (Tailwind, HTMX, custom CSS) are served from the
  `static/` sub-package, mounted at `/static`.
- The app is constructed via a factory (`create_app(config_path)`) so
  tests can build it without touching the network or the real config.
- `run_ui(args)` is the CLI entry point: it picks a free port, opens
  the browser asynchronously, and hands control to uvicorn.

Listening only on `127.0.0.1` by design — this is a personal
management tool, not a shared service. HTTPS / auth would only add
ceremony without value at this scope.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Port handling
# ---------------------------------------------------------------------------


def _find_free_port(preferred: int = 8765, attempts: int = 10) -> int:
    """Return the first free port starting at `preferred`.

    We test by trying to bind a socket to (127.0.0.1, port). If bind
    succeeds, the port is free; if it raises OSError, we move on.
    Falls back to letting the OS pick (port 0) after `attempts`.
    """
    for offset in range(attempts):
        port = preferred + offset
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            finally:
                s.close()
        except OSError:
            continue
    # Last resort: OS-assigned port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config_path: Optional[Path] = None):
    """Build the FastAPI app.

    `config_path` is stored on `app.state.config_path` so endpoints can
    re-load the config on demand (an edit through the UI may change it).
    The SourceManager is also lazy-loaded: created on first access via
    `_get_manager(app)` so the test client can construct the app
    without paying the embedding-model cost.
    """
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    pkg_dir = Path(__file__).parent
    static_dir = pkg_dir / "static"
    templates_dir = pkg_dir / "templates"

    app = FastAPI(
        title="LynxManager",
        description="Local web UI for Lynx: config, monitoring, playground.",
        version=_lynx_version(),
        # No /docs / /redoc — they're MCP-irrelevant noise on a local UI
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # State carried between requests. NEVER share mutable state across
    # routes via globals — always via app.state.
    app.state.config_path = config_path
    app.state.manager = None       # lazy
    app.state.manager_error = None
    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    # Cross-platform basename filter: split on BOTH separators so a Windows
    # config path (C:\...\config.json) renders as just the filename. Python's
    # str.split('/') alone leaves backslash paths whole, which overflowed the
    # sidebar.
    app.state.templates.env.filters["basename"] = (
        lambda p: str(p).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] if p else p
    )

    # Which code this UI runs. The version alone cannot say it: an editable
    # checkout reports the same number before and after its code changes.
    from ...ownership import code_identity
    app.state.templates.env.globals["lynx_identity"] = code_identity()

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes + pages
    from . import pages, routes
    pages.register(app)
    routes.register(app)

    return app


def _lynx_version() -> str:
    """Best-effort lookup of the installed package version."""
    from importlib.metadata import version
    # The distribution is `lynx-mcp` (bare `lynx` on PyPI is an unrelated
    # 2014 library); only an uninstalled source tree falls back to "dev".
    for dist in ("lynx-mcp", "lynx"):
        try:
            return version(dist)
        except Exception:
            continue
    return "dev"


def serving_info(storage_dir, is_owner) -> dict:
    """Who keeps an index current, and whether it runs this UI's code.

    `who` is "this UI" (we own it), "another" (a live claim names another
    process) or "nobody" (no live claim: no watcher runs). For "another",
    `same_code` compares the code hash in its claim with ours, which is the
    question behind "is the MCP server still on the old version?".
    """
    from ...ownership import code_identity, owner_of
    mine = code_identity()
    if is_owner:
        return {"who": "this UI", "same_code": True, "version": mine["version"],
                "code": mine["code"], "path": mine["path"]}
    try:
        claim = owner_of(storage_dir)
    except Exception:
        claim = None
    if not claim:
        return {"who": "nobody"}
    theirs = claim.get("lynx") or {}
    return {
        "who": "another", "pid": claim.get("pid"), "host": claim.get("host"),
        "since": claim.get("since"), "version": theirs.get("version"),
        "code": theirs.get("code"), "path": theirs.get("path"),
        "same_code": bool(theirs.get("code")) and theirs.get("code") == mine["code"],
    }


# One SourceManager at a time. FastAPI runs the plain route functions in a
# thread pool, and after a source is added the build request and the page's
# own requests arrive together. Each of them used to build a manager of its
# own and open the new index at the same moment; chromadb's per-path client
# cache then handed one a client still starting (AttributeError
# 'RustBindingsAPI' ... 'bindings') and another an entry already removed
# (KeyError), both reported as a corrupt index, so the first project added
# from the UI was never built. Found by scripts/manager_guide.py, 2026-11-23.
_MANAGER_LOCK = threading.RLock()


def _get_manager(app):
    """Lazy-construct the SourceManager. Cached on app.state so subsequent
    requests are cheap. Errors are stashed on app.state.manager_error so
    pages can surface them gracefully (instead of every request raising)."""
    if app.state.manager is not None:
        return app.state.manager
    with _MANAGER_LOCK:
        return _build_manager_locked(app)


def _build_manager_locked(app):
    if app.state.manager is not None:
        return app.state.manager  # built by the request we waited for
    if app.state.manager_error is not None:
        # We already tried and failed — don't retry on every request
        return None
    if app.state.config_path is None or not Path(app.state.config_path).exists():
        tried = (
            str(Path(app.state.config_path).resolve())
            if app.state.config_path is not None
            else "<none>"
        )
        app.state.manager_error = (
            f"No config file found at {tried}. "
            f"Restart the UI with --config PATH, set RAG_CONFIG_PATH, "
            f"or run `lynx manager init` to create one."
        )
        return None
    try:
        from ...config import load_config, configure_hf_offline
        cfg = load_config(Path(app.state.config_path))
        # Decide HF offline mode BEFORE the heavy imports freeze the env flags.
        configure_hf_offline(cfg)
        from ...source_manager import SourceManager
        app.state.manager = SourceManager(cfg)
        return app.state.manager
    except SystemExit as e:
        app.state.manager_error = f"Config validation failed (exit {e.code})."
        return None
    except Exception as e:
        app.state.manager_error = f"{type(e).__name__}: {e}"
        return None


def _drop_manager(app) -> None:
    """Forget the cached SourceManager, so the next request that needs one
    builds it lazily via `_get_manager` from the config on disk.

    The old manager hands its indexes back first. The new one claims them
    again, and would otherwise find a live claim with our own pid on it,
    take it for another Lynx process and refuse every build until the UI is
    restarted."""
    with _MANAGER_LOCK:
        if app.state.manager is not None:
            app.state.manager.close()
        app.state.manager = None
        app.state.manager_error = None


def _dispose_manager(app) -> None:
    """Drop the cached SourceManager AND release its ChromaDB file handles.

    `_drop_manager` alone is not enough: chromadb caches the
    client system per storage path (SharedSystemClient), so the SQLite and
    HNSW segment files stay open with zero references from our side.
    Measured on Windows: with a manager constructed, `rmtree` on a source's
    storage fails with WinError 32 on `data_level0.bin` even after dropping
    the reference and a gc pass; clearing the system cache is what actually
    closes the handles. Without this, the UI's own DELETE ?purge=true would
    always 409 against the UI's own lock — advising the user to stop the
    very process they're clicking in.

    The next request that needs the manager rebuilds it lazily via
    `_get_manager`, integrity probes included.
    """
    _drop_manager(app)
    SharedSystemClient = None
    for modpath in ("chromadb.api.shared_system_client", "chromadb.api.client"):
        try:
            mod = __import__(modpath, fromlist=["SharedSystemClient"])
            SharedSystemClient = getattr(mod, "SharedSystemClient")
            break
        except (ImportError, AttributeError):
            continue
    if SharedSystemClient is not None:
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass  # a failed cache clear degrades to the pre-fix 409, not a crash
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_ui(args) -> int:
    """Launch the web UI. Returns the exit code uvicorn produces (0 on
    clean shutdown via Ctrl+C)."""
    # Use the same resolution chain as `lynx serve` so the UI auto-picks
    # ./config.json (or $RAG_CONFIG_PATH) when --config is omitted. On
    # Windows the cwd handling is occasionally surprising — log what we
    # resolved so the user can verify at a glance.
    from ...config import resolve_config_path
    config_path = resolve_config_path(getattr(args, "config", None))
    if config_path.is_file():
        _log(f"[ui] using config: {config_path.resolve()}")
    else:
        _log(f"[ui] no config at {config_path.resolve()}; "
             f"launching with empty manager (pass --config or run "
             f"`lynx manager init` to create one)")

    requested_port = int(getattr(args, "port", 8765))
    host = str(getattr(args, "host", "127.0.0.1"))
    open_browser = not bool(getattr(args, "no_browser", False))

    port = _find_free_port(requested_port)
    if port != requested_port:
        _log(f"[ui] port {requested_port} busy, using {port}")

    app = create_app(config_path)
    # What `lynx update --restart-manager` needs to reopen this same manager
    # (it used to come back on the default config and port).
    app.state.relaunch_args = (
        (["--config", str(config_path.resolve())] if config_path.is_file() else [])
        + (["--port", str(requested_port)] if requested_port != 8765 else [])
        + (["--host", host] if host != "127.0.0.1" else []))
    url = f"http://{host}:{port}"

    # Pre-warm the SourceManager BEFORE uvicorn starts listening.
    # Without this the first GET / blocks for 5-30s while we load the
    # HuggingFace embedding model and open ChromaDB collections, and
    # the browser sits on a white page the whole time. Doing it here
    # means the terminal shows progress and the browser only opens
    # when the dashboard can actually render.
    if config_path.is_file():
        print(
            "[ui] loading embedding model + opening source collections "
            "(first launch can take 30s)...",
            file=sys.stderr,
        )
        t0 = time.time()
        _get_manager(app)
        elapsed = time.time() - t0
        if app.state.manager_error:
            print(f"[ui] WARNING: {app.state.manager_error}", file=sys.stderr)
            print(
                "[ui] the UI will still open, but the dashboard will show "
                "this error until you fix the config.",
                file=sys.stderr,
            )
        else:
            n_sources = len(app.state.manager.backends)
            print(
                f"[ui] manager ready in {elapsed:.1f}s "
                f"({n_sources} source{'' if n_sources == 1 else 's'}).",
                file=sys.stderr,
            )

    # Open browser only AFTER pre-warm completes. Still on a small
    # delay so uvicorn's bind has a moment to take effect.
    if open_browser:
        def _open_later():
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception as e:
                _log(f"[ui] couldn't auto-open browser: {e}")
        threading.Thread(target=_open_later, daemon=True).start()

    print(f"🐱 Lynx UI ready at {url}", file=sys.stderr)
    print(f"   Press Ctrl+C to stop.", file=sys.stderr)

    try:
        import uvicorn
    except ImportError as e:
        _log(f"[ui] uvicorn not installed: {e}")
        return 2

    # log_level="warning" keeps uvicorn quiet — our own server log lives
    # on stderr already.
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0
