"""HTML-rendering endpoints for the web UI.

Phase 4 ships only the dashboard. Config editor (Phase 5), playground
(Phase 6), source detail (Phase 7), integrations (Phase 8) land in
their own files.

Each page is a thin wrapper: pull data via the manager, render a
Jinja2 template, return HTMLResponse. Long logic stays out of the
templates — keeps them readable and lets the routes module test the
data fetch separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Module-level FastAPI imports: with `from __future__ import annotations`
# every type hint is a string at function-definition time. FastAPI
# resolves these strings via the module globals, so `Request` must be
# importable from `pages.__dict__` — not just visible inside `register()`.
# Without this, FastAPI mistakes `request: Request` for a query parameter
# and every page returns HTTP 422.
from fastapi import Request
from fastapi.responses import HTMLResponse

from .app import _get_manager


def register(app) -> None:

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        """Single-page overview: sources cards + global stats + health card."""
        mgr = _get_manager(app)
        sources_data: list = []
        manager_error: Optional[str] = None

        if mgr is None:
            manager_error = app.state.manager_error or "Manager not initialized."
        else:
            from . import lock as lock_mod
            for st in mgr.list_sources():
                storage_path = Path(mgr.config.storage_path) / st["name"]
                st["locked"] = lock_mod.is_storage_locked(storage_path)
                sources_data.append(st)

        # Global stats (best-effort — missing fields tolerated)
        total_chunks = sum((s.get("chunk_count") or 0) for s in sources_data)
        total_sources = len(sources_data)

        # Lightweight doctor summary (top-level status only, no per-check
        # detail rendering on the dashboard — separate page for that).
        from .. import doctor as doc_mod
        config_path = Path(app.state.config_path) if app.state.config_path else None
        try:
            doctor_results = doc_mod.run_all_checks(config_path)
            worst = doc_mod._worst_status(doctor_results)
            doctor_counts = {
                "ok":    sum(1 for r in doctor_results if r.status == "ok"),
                "warn":  sum(1 for r in doctor_results if r.status == "warn"),
                "error": sum(1 for r in doctor_results if r.status == "error"),
            }
        except Exception:
            worst = "warn"
            doctor_counts = {"ok": 0, "warn": 0, "error": 0}
            doctor_results = []

        return app.state.templates.TemplateResponse(
            request, "dashboard.html",
            {
                "sources": sources_data,
                "total_chunks": total_chunks,
                "total_sources": total_sources,
                "doctor_counts": doctor_counts,
                "doctor_worst": worst,
                "doctor_results": [r.to_dict() for r in doctor_results],
                "manager_error": manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    # Phase 6: search playground — exercises all per-source tools the
    # MCP server exposes (search, find_definition, get_callers, ...).
    @app.get("/playground", response_class=HTMLResponse)
    def playground(request: Request):
        mgr = _get_manager(app)
        sources_meta: list = []
        if mgr is not None:
            for st in mgr.list_sources():
                sources_meta.append({
                    "name": st["name"],
                    "type": st["type"],
                    "graph": bool(st.get("graph")),
                })
        return app.state.templates.TemplateResponse(
            request, "playground.html",
            {
                "sources_meta": sources_meta,
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    # Phase 7-8 land in their own templates. Until then we surface the
    # sidebar links to clear placeholders so the user knows the page
    # exists and what CLI command does the same thing.

    # `/config` is fully implemented here (Phase 5). The remaining
    # entries below are placeholders that link the sidebar to a "this
    # arrives in phase N" page so the user knows the page exists.
    @app.get("/config", response_class=HTMLResponse)
    def config_editor(request: Request):
        config_path = app.state.config_path
        if config_path is None or not Path(config_path).exists():
            content = (
                "{\n  // No config.json loaded — run `lynx manager init` or\n"
                "  // pass --config to `lynx manager ui` to point at one.\n}"
            )
        else:
            try:
                content = Path(config_path).read_text(encoding="utf-8")
            except OSError as e:
                content = f"# Error reading config: {e}"
        return app.state.templates.TemplateResponse(
            request, "config.html",
            {
                "config_content": content,
                "config_path": str(config_path) if config_path else "(none)",
                "manager_error": app.state.manager_error,
            },
        )

    # Phase 7: sources list + per-source detail page (status + build).
    @app.get("/sources", response_class=HTMLResponse)
    def sources_index(request: Request):
        mgr = _get_manager(app)
        sources_data: list = []
        if mgr is not None:
            from . import lock as lock_mod
            for st in mgr.list_sources():
                storage_path = Path(mgr.config.storage_path) / st["name"]
                st["locked"] = lock_mod.is_storage_locked(storage_path)
                sources_data.append(st)
        return app.state.templates.TemplateResponse(
            request, "sources_list.html",
            {
                "sources": sources_data,
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    @app.get("/sources/{name}", response_class=HTMLResponse)
    def source_detail(request: Request, name: str):
        from fastapi import HTTPException
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        try:
            backend = mgr.get(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"source {name!r} not found")
        st = backend.status()
        from . import lock as lock_mod
        storage_path = Path(mgr.config.storage_path) / name
        st["locked"] = lock_mod.is_storage_locked(storage_path)
        return app.state.templates.TemplateResponse(
            request, "source_detail.html",
            {
                "source": st,
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    _PLACEHOLDERS = [
        ("/integrations",  "Integrations",  "Phase 8",
         "see README section 'Connect it to your AI client'"),
        ("/doctor",        "Doctor",        "Phase 5",
         "lynx manager doctor"),
    ]

    def _make_placeholder(path, title, phase, cli_hint):
        @app.get(path, response_class=HTMLResponse, name=f"placeholder_{path.strip('/')}")
        def _page(request: Request):
            return app.state.templates.TemplateResponse(
                request, "placeholder.html",
                {
                    "title": title,
                    "phase": phase,
                    "cli_hint": cli_hint,
                    "manager_error": app.state.manager_error,
                    "config_path": str(app.state.config_path) if app.state.config_path else None,
                },
            )
        return _page

    for path, title, phase, cli_hint in _PLACEHOLDERS:
        _make_placeholder(path, title, phase, cli_hint)
