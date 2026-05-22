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
        description="Local web UI for Lynx — config, monitoring, playground.",
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

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes + pages
    from . import pages, routes
    pages.register(app)
    routes.register(app)

    return app


def _lynx_version() -> str:
    """Best-effort lookup of the installed package version."""
    try:
        from importlib.metadata import version
        return version("lynx")
    except Exception:
        return "dev"


def _get_manager(app):
    """Lazy-construct the SourceManager. Cached on app.state so subsequent
    requests are cheap. Errors are stashed on app.state.manager_error so
    pages can surface them gracefully (instead of every request raising)."""
    if app.state.manager is not None:
        return app.state.manager
    if app.state.manager_error is not None:
        # We already tried and failed — don't retry on every request
        return None
    if app.state.config_path is None or not Path(app.state.config_path).exists():
        app.state.manager_error = "No config file (set --config or run `lynx manager init`)."
        return None
    try:
        from ...config import load_config
        from ...source_manager import SourceManager
        cfg = load_config(Path(app.state.config_path))
        app.state.manager = SourceManager(cfg)
        return app.state.manager
    except SystemExit as e:
        app.state.manager_error = f"Config validation failed (exit {e.code})."
        return None
    except Exception as e:
        app.state.manager_error = f"{type(e).__name__}: {e}"
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_ui(args) -> int:
    """Launch the web UI. Returns the exit code uvicorn produces (0 on
    clean shutdown via Ctrl+C)."""
    config_path = Path(args.config) if getattr(args, "config", None) else None

    requested_port = int(getattr(args, "port", 8765))
    host = str(getattr(args, "host", "127.0.0.1"))
    open_browser = not bool(getattr(args, "no_browser", False))

    port = _find_free_port(requested_port)
    if port != requested_port:
        _log(f"[ui] port {requested_port} busy, using {port}")

    app = create_app(config_path)
    url = f"http://{host}:{port}"

    # Open browser on a delay so uvicorn has time to start listening.
    if open_browser:
        def _open_later():
            time.sleep(0.6)
            try:
                webbrowser.open(url)
            except Exception as e:
                _log(f"[ui] couldn't auto-open browser: {e}")
        threading.Thread(target=_open_later, daemon=True).start()

    print(f"🦌 Lynx UI ready at {url}", file=sys.stderr)
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
