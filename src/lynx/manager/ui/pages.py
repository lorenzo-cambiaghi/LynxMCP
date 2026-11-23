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
            from .app import serving_info
            for st in mgr.list_sources():
                storage_path = Path(mgr.config.storage_path) / st["name"]
                # Don't probe a corrupt store's SQLite — it's malformed and
                # would raise. It's reported as corrupt, not locked.
                if st.get("health") == "corrupt":
                    st["locked"] = False
                else:
                    st["locked"] = lock_mod.is_storage_locked(storage_path)
                st["serving"] = serving_info(storage_path, st.get("owner") is True
                                             and st.get("health", "ok") == "ok")
                sources_data.append(st)

        # Global stats (best-effort — missing fields tolerated)
        total_chunks = sum((s.get("chunk_count") or 0) for s in sources_data)
        total_sources = len(sources_data)

        # Feedback reports: what agents filed when the index couldn't answer
        # them. Read-only, local, and the one signal that tells the person
        # tuning sources WHY the index is falling short — it had a CLI
        # (`lynx manager feedback`) and no place in the UI at all, which is
        # backwards for something whose only audience is the index owner.
        feedback = {"total": 0, "recent": [], "by_source": {}}
        if mgr is not None:
            try:
                from .. import feedback as fb_mod
                records = fb_mod.load_feedback(
                    fb_mod.feedback_path_for(mgr.config.storage_path)
                )
                feedback = fb_mod.summarize_feedback(records, limit=5)
            except Exception:
                pass  # a missing or unreadable log must not blank the page

        # Which AI clients already have Lynx wired in (local file probe only,
        # never touches the network). Drives the 3-step onboarding strip.
        connected_clients: list[str] = []
        try:
            from ...client_connector import inspect_all_clients
            connected_clients = [
                st.spec.short_label for st in inspect_all_clients() if st.connected
            ]
        except Exception:
            pass

        return app.state.templates.TemplateResponse(
            request, "dashboard.html",
            {
                "sources": sources_data,
                "total_chunks": total_chunks,
                "total_sources": total_sources,
                "connected_clients": connected_clients,
                "feedback": feedback,
                "manager_error": manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    @app.get("/partials/health", response_class=HTMLResponse)
    def health_card(request: Request):
        """The dashboard's health card, loaded after the page.

        The doctor probes every index (up to 20 s each) and loads the lexical
        corpus: run inline, it held the whole dashboard until it finished,
        tens of seconds on a large index, to show three numbers. Now the
        page comes at once and the card names what is not OK.
        """
        from .. import doctor as doc_mod
        config_path = Path(app.state.config_path) if app.state.config_path else None
        try:
            results = doc_mod.run_all_checks(config_path)
            worst = doc_mod._worst_status(results)
            failed = None
        except Exception as e:
            results, worst, failed = [], "warn", f"{type(e).__name__}: {e}"
        counts = {s: sum(1 for r in results if r.status == s) for s in ("ok", "warn", "error")}
        problems = [r.to_dict() for r in results if r.status != "ok"]
        return app.state.templates.TemplateResponse(
            request, "_health_card.html",
            {"counts": counts, "worst": worst, "problems": problems, "failed": failed},
        )

    # Phase 6: search playground — exercises all per-source tools the
    # MCP server exposes (search, find_definition, get_callers, ...).
    @app.get("/playground", response_class=HTMLResponse)
    def playground(request: Request, source: str = ""):
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
                "selected_source": source,
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
                "{\n  // No config.json loaded: run `lynx manager init` or\n"
                "  // pass --config to `lynx manager ui` to point at one.\n}"
            )
        else:
            try:
                # utf-8-sig: a BOM read as plain utf-8 reaches the editor
                # textarea as a stray leading character the user then saves
                # back, making the config permanently unloadable.
                content = Path(config_path).read_text(encoding="utf-8-sig")
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
                # Don't probe a corrupt store's SQLite — it's malformed and
                # would raise. It's reported as corrupt, not locked.
                if st.get("health") == "corrupt":
                    st["locked"] = False
                else:
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

    # Phase 10: guided "Add source" flow — chooser + per-type form.
    # Routes registered BEFORE /sources/{name} so the literal "new"
    # segment matches the chooser instead of falling through to the
    # source-detail route.
    @app.get("/sources/new", response_class=HTMLResponse)
    def add_source_chooser(request: Request):
        return app.state.templates.TemplateResponse(
            request, "add_source_chooser.html",
            {
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    @app.get("/sources/new/{source_type}", response_class=HTMLResponse)
    def add_source_form(request: Request, source_type: str):
        from fastapi import HTTPException
        if source_type not in ("codebase", "webdoc", "pdf"):
            raise HTTPException(
                status_code=404,
                detail=f"unknown source type: {source_type!r}",
            )
        return app.state.templates.TemplateResponse(
            request, "add_source_form.html",
            {
                "source_type": source_type,
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
        from .app import serving_info
        storage_path = Path(mgr.config.storage_path) / name
        st["locked"] = lock_mod.is_storage_locked(storage_path)
        st["serving"] = serving_info(storage_path, getattr(backend, "is_owner", True))
        from . import jobs as jobs_mod
        from .routes import _render_job_widget
        job = (jobs_mod.has_running_job_for(f"build:{name}")
               or jobs_mod.has_running_job_for(f"reset:{name}"))
        return app.state.templates.TemplateResponse(
            request, "source_detail.html",
            {
                "source": st,
                "running_job": _render_job_widget(job, name) if job is not None else "",
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    # Phase 8: integrations snippets — wires Lynx into the user's AI client.
    @app.get("/integrations", response_class=HTMLResponse)
    def integrations_page(request: Request):
        from . import integrations as integ
        config_path = (
            Path(app.state.config_path) if app.state.config_path else None
        )
        mgr = _get_manager(app)

        # Derive source names + capability flags from the live manager so
        # the rules preview reflects what's actually configured.
        source_names: list[str] = []
        has_graph = False
        has_git = False
        if mgr is not None:
            try:
                cfg = mgr.config
                for name, sc in cfg.sources.items():
                    source_names.append(name)
                    if (sc.get("graph") or {}).get("enabled"):
                        has_graph = True
                    if (sc.get("git_integration") or {}).get("enabled"):
                        has_git = True
            except Exception:
                # Best-effort — never let the preview crash the page.
                pass

        rules_preview = (
            integ.render_rules_file(source_names, has_graph, has_git)
            if source_names else None
        )

        return app.state.templates.TemplateResponse(
            request, "integrations.html",
            {
                "clients": integ.build_integrations(config_path),
                "rules_preview": rules_preview,
                "source_names": source_names,
                "manager_error": app.state.manager_error,
                "config_path": str(config_path) if config_path else None,
            },
        )

    # Phase 9: the doctor page wraps `lynx manager doctor` — full
    # diagnostic report with re-run button.
    @app.get("/doctor", response_class=HTMLResponse)
    def doctor_page(request: Request):
        from .. import doctor as doc_mod
        config_path = Path(app.state.config_path) if app.state.config_path else None
        try:
            results = doc_mod.run_all_checks(config_path)
            worst = doc_mod._worst_status(results)
        except Exception as e:
            # Doctor itself should never crash, but better safe than sorry.
            results = []
            worst = "error"
            app.state.manager_error = (
                f"Doctor crashed: {type(e).__name__}: {e}"
            )
        counts = {
            "ok":    sum(1 for r in results if r.status == "ok"),
            "warn":  sum(1 for r in results if r.status == "warn"),
            "error": sum(1 for r in results if r.status == "error"),
        }
        return app.state.templates.TemplateResponse(
            request, "doctor.html",
            {
                "results": [r.to_dict() for r in results],
                "worst_status": worst,
                "counts": counts,
                "manager_error": app.state.manager_error,
                "config_path": str(app.state.config_path) if app.state.config_path else None,
            },
        )

    # No more placeholder routes — every sidebar page now has a real
    # implementation. List kept (empty) so future deferred features can
    # use the helper below without scaffolding it again.
    _PLACEHOLDERS: list = []

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
