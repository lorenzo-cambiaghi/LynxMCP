"""JSON API endpoints under `/api/*`.

Used by HTMX partials in the templates AND by anything external that
wants to poll the UI (the same FastAPI app serves both). Keep
endpoints focused: status, doctor, search. Mutations land in Phase 5+.
"""
from __future__ import annotations

from typing import Optional

# Module-level imports so FastAPI can resolve `Request` type annotations
# under `from __future__ import annotations`. Without this, FastAPI
# treats `request: Request` as a query parameter and returns 422.
from fastapi import Request

from .app import _get_manager


def register(app) -> None:
    """Attach the JSON routes to the given FastAPI app.

    Imported deps are local to keep test-side `create_app()` cheap.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    @app.get("/api/health")
    def api_health():
        """Simple liveness probe — UI uses this for the favicon tab status."""
        return {"status": "ok"}

    @app.get("/api/sources")
    def api_sources():
        """List configured sources with their status + lock state."""
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        out = []
        # Use list_sources which already aggregates the status dicts
        for st in mgr.list_sources():
            # Augment with lock detection — read-only sources can still
            # be searched, but the UI marks them visually.
            from pathlib import Path
            from . import lock as lock_mod
            storage_path = Path(mgr.config.storage_path) / st["name"]
            st["locked"] = lock_mod.is_storage_locked(storage_path)
            out.append(st)
        return {"sources": out}

    @app.get("/api/sources/{name}/status")
    def api_source_status(name: str):
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(status_code=503,
                                detail=app.state.manager_error or "manager not initialized")
        try:
            backend = mgr.get(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return backend.status()

    @app.get("/api/doctor")
    def api_doctor():
        """Re-run diagnostics on demand. Useful for the dashboard health card."""
        from pathlib import Path
        from .. import doctor as doc_mod
        config_path = Path(app.state.config_path) if app.state.config_path else None
        results = doc_mod.run_all_checks(config_path)
        return {
            "results": [r.to_dict() for r in results],
            "worst_status": doc_mod._worst_status(results),
        }

    @app.get("/api/config")
    def api_config():
        """Return the raw config file content."""
        from pathlib import Path
        if app.state.config_path is None:
            raise HTTPException(status_code=404, detail="no config path configured")
        p = Path(app.state.config_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"config not found at {p}")
        return {"path": str(p), "content": p.read_text(encoding="utf-8")}

    @app.put("/api/config")
    async def api_config_save(request: Request):
        """Save a new config.json. Validates BEFORE overwriting — invalid
        configs return 422 with the error message and never touch disk
        (besides writing to a tempfile to invoke `load_config`).

        Backups the previous content to `<config_path>.bak` after a
        successful save so the user can roll back manually if needed.

        Returns a small HTML fragment (toast) since this is invoked from
        the config editor page via HTMX. The fragment is also valid
        for non-HTMX callers — just slightly verbose.
        """
        import json
        from pathlib import Path
        import tempfile
        from fastapi.responses import HTMLResponse

        # Parse the JSON body. HTMX sends `{"content": "..."}` (handled in
        # the page's submit script — see config.html).
        try:
            payload = await request.json()
        except Exception as e:
            return HTMLResponse(
                f'<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                f'❌ Invalid request body: {e}</div>',
                status_code=400,
            )
        content = payload.get("content", "")
        if not content.strip():
            return HTMLResponse(
                '<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                '❌ Config content is empty.</div>',
                status_code=400,
            )

        # JSON syntax check first — we don't want to call `load_config`
        # on a fundamentally broken file.
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return HTMLResponse(
                f'<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                f'❌ <strong>JSON syntax error</strong>: {e.msg} '
                f'(line {e.lineno}, col {e.colno})</div>',
                status_code=422,
            )

        # Schema validation: write to a temp file and invoke load_config.
        # We chose this over duplicating validation logic so the UI and
        # CLI never disagree on what's valid.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as tf:
            tf.write(content)
            tmp_path = Path(tf.name)
        try:
            try:
                from ...config import load_config
                load_config(tmp_path)
            except SystemExit as e:
                return HTMLResponse(
                    f'<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                    f'❌ <strong>Schema validation failed</strong> (exit {e.code}). '
                    f'Check the terminal where you launched `lynx manager ui` for the '
                    f'detailed error message.</div>',
                    status_code=422,
                )
            except Exception as e:
                return HTMLResponse(
                    f'<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                    f'❌ <strong>Validation error</strong>: {type(e).__name__}: {e}</div>',
                    status_code=422,
                )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        # Validation passed — back up the old file and overwrite.
        if app.state.config_path is None:
            return HTMLResponse(
                '<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                '❌ No config path configured — launch UI with --config PATH.</div>',
                status_code=400,
            )
        target = Path(app.state.config_path)
        try:
            if target.exists():
                backup = target.with_suffix(target.suffix + ".bak")
                backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return HTMLResponse(
                f'<div class="p-3 bg-red-50 border border-red-200 rounded text-red-900 text-sm">'
                f'❌ Couldn\'t write config: {e}</div>',
                status_code=500,
            )

        # Invalidate the cached manager so next request reloads from disk
        app.state.manager = None
        app.state.manager_error = None

        return HTMLResponse(
            '<div class="p-3 bg-green-50 border border-green-200 rounded text-green-900 text-sm">'
            '✓ <strong>Config saved.</strong> '
            f'Backup at <code>{target.name}.bak</code>. '
            'Restart `lynx serve` to pick up the changes.'
            '</div>'
        )
