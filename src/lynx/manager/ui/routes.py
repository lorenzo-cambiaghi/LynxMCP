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

    # ------------------------------------------------------------------
    # Phase 6: search playground endpoints
    # ------------------------------------------------------------------
    # Each tool gets its own POST endpoint that returns an HTML partial
    # for HTMX to swap into the result div. Centralising error rendering
    # in `_err` keeps the per-tool handlers focused on dispatch + render.
    _register_playground_routes(app)


def _html_escape(s) -> str:
    """Minimal HTML escape so search results can include user code safely."""
    if s is None:
        return ""
    s = str(s)
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _err(msg: str, status: int = 400):
    """Toast-style HTML error for HTMX result panels."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        f'<div class="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-900">'
        f'❌ {_html_escape(msg)}</div>',
        status_code=status,
    )


def _empty(msg: str = "No results."):
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        f'<div class="p-3 bg-slate-50 border border-slate-200 rounded text-sm text-slate-500">'
        f'{_html_escape(msg)}</div>'
    )


def _render_hits(items, *, show_score: bool = True) -> str:
    """Render a list of search-shaped dicts (file_path, symbol, content,
    score) as compact cards. Used by search / find_* / search_diff."""
    if not items:
        return ('<div class="p-3 bg-slate-50 border border-slate-200 rounded '
                'text-sm text-slate-500">No results.</div>')
    parts = [f'<div class="text-xs text-slate-500 mb-2">{len(items)} result(s)</div>',
             '<div class="space-y-2">']
    for r in items:
        score = r.get("score")
        symbol = r.get("symbol_name") or ""
        kind = r.get("symbol_kind") or ""
        fp = r.get("file_path") or r.get("file") or ""
        line_start = r.get("start_line") or 0
        line_end = r.get("end_line") or 0
        loc = f"{fp}"
        if line_start:
            loc += f":{line_start}"
            if line_end and line_end != line_start:
                loc += f"-{line_end}"
        content = r.get("content") or ""
        snippet = content[:500] + ("…" if len(content) > 500 else "")

        badge = ""
        if symbol:
            label = symbol + (f" ({kind})" if kind else "")
            badge = (f'<span class="inline-block px-2 py-0.5 text-xs '
                     f'rounded bg-indigo-100 text-indigo-800 font-mono">'
                     f'{_html_escape(label)}</span>')
        score_badge = ""
        if show_score and score is not None:
            try:
                score_badge = (f'<span class="text-xs text-slate-500 ml-2">'
                               f'score={float(score):.3f}</span>')
            except (TypeError, ValueError):
                pass
        parts.append(
            f'<div class="p-3 bg-white border border-slate-200 rounded">'
            f'  <div class="flex items-center justify-between mb-1">'
            f'    <div>{badge}<span class="font-mono text-xs text-slate-600 ml-2">{_html_escape(loc)}</span></div>'
            f'    <div>{score_badge}</div>'
            f'  </div>'
            f'  <pre class="text-xs bg-slate-50 p-2 rounded overflow-x-auto whitespace-pre-wrap">'
            f'{_html_escape(snippet)}</pre>'
            f'</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def _render_simple_list(items, *, item_key: str | None = None) -> str:
    """Render a flat list (callers/callees) — items can be str OR dict."""
    if not items:
        return ('<div class="p-3 bg-slate-50 border border-slate-200 rounded '
                'text-sm text-slate-500">No results.</div>')
    parts = [f'<div class="text-xs text-slate-500 mb-2">{len(items)} item(s)</div>',
             '<ul class="space-y-1">']
    for it in items:
        if isinstance(it, dict):
            label = (it.get(item_key) if item_key else None) \
                or it.get("symbol") or it.get("name") or it.get("id") \
                or str(it)
            extra_bits = []
            if it.get("file_path"):
                line = it.get("start_line") or it.get("line") or ""
                loc = f"{it['file_path']}" + (f":{line}" if line else "")
                extra_bits.append(loc)
            if it.get("kind"):
                extra_bits.append(it["kind"])
            extra = (f' <span class="text-xs text-slate-500 font-mono">'
                     f'{_html_escape(" · ".join(extra_bits))}</span>'
                     if extra_bits else '')
        else:
            label = it
            extra = ""
        parts.append(
            f'<li class="p-2 bg-white border border-slate-200 rounded text-sm">'
            f'<span class="font-mono">{_html_escape(label)}</span>{extra}'
            f'</li>'
        )
    parts.append('</ul>')
    return "".join(parts)


def _register_playground_routes(app) -> None:
    """Per-tool POST endpoints used by playground.html via HTMX.

    All endpoints share the same request shape: form-encoded body with
    `source` always required, plus tool-specific fields.  They return
    HTML partials (not JSON) since HTMX swaps the response straight into
    the result div.

    Errors return 4xx/5xx with an HTML toast so HTMX still renders them.
    """
    from fastapi import Form
    from fastapi.responses import HTMLResponse

    from .app import _get_manager as _gm

    def _mgr_or_err():
        mgr = _gm(app)
        if mgr is None:
            return None, _err(
                app.state.manager_error or "Manager not initialized.",
                status=503,
            )
        return mgr, None

    @app.post("/api/playground/search")
    def pg_search(
        source: str = Form(...),
        query: str = Form(...),
        mode: str = Form(""),
        top_k: int = Form(5),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not query.strip():
            return _err("query is empty")
        try:
            # NOTE: backend.search() honours the configured mode; per-call
            # override would require plumbing through search_mode at
            # backend instantiation. Mode dropdown is reserved for a
            # future enhancement — kept in UI for visibility.
            hits = mgr.search(source, query, top_k=int(top_k))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_hits(hits))

    @app.post("/api/playground/find_definition")
    def pg_find_definition(
        source: str = Form(...),
        symbol: str = Form(...),
        limit: int = Form(10),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        try:
            hits = mgr.find_definition(source, symbol.strip(), limit=int(limit))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_hits(hits))

    @app.post("/api/playground/find_usages")
    def pg_find_usages(
        source: str = Form(...),
        symbol: str = Form(...),
        limit: int = Form(50),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        try:
            hits = mgr.find_usages(source, symbol.strip(), limit=int(limit))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_hits(hits))

    @app.post("/api/playground/find_tests_for")
    def pg_find_tests_for(
        source: str = Form(...),
        symbol: str = Form(...),
        limit: int = Form(20),
        test_path_pattern: str = Form(""),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        try:
            hits = mgr.find_tests_for(
                source, symbol.strip(), limit=int(limit),
                test_path_pattern=(test_path_pattern.strip() or None),
            )
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_hits(hits))

    @app.post("/api/playground/find_similar")
    def pg_find_similar(
        source: str = Form(...),
        snippet: str = Form(...),
        top_k: int = Form(10),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not snippet.strip():
            return _err("snippet is empty")
        try:
            hits = mgr.find_similar(source, snippet, top_k=int(top_k))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_hits(hits))

    @app.post("/api/playground/search_diff")
    def pg_search_diff(
        source: str = Form(...),
        query: str = Form(...),
        base: str = Form(""),
        top_k: int = Form(8),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not query.strip():
            return _err("query is empty")
        try:
            payload = mgr.search_diff(
                source, query,
                base=(base.strip() or None), top_k=int(top_k),
            )
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        # search_diff returns a dict — extract `results` + show meta info.
        results = payload.get("results", []) if isinstance(payload, dict) else []
        meta_bits = []
        if isinstance(payload, dict):
            for k in ("base", "head", "modified_files"):
                if payload.get(k) is not None:
                    v = payload[k]
                    if isinstance(v, (list, tuple)):
                        v = f"{len(v)} file(s)"
                    meta_bits.append(f"{k}={v}")
        header = (f'<div class="text-xs text-slate-600 mb-2">'
                  f'{_html_escape(" · ".join(meta_bits))}</div>'
                  if meta_bits else '')
        return HTMLResponse(header + _render_hits(results))

    @app.post("/api/playground/architectural_overview")
    def pg_arch_overview(
        source: str = Form(...),
        top_n_gods: int = Form(10),
        min_community_size: int = Form(3),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        try:
            payload = mgr.architectural_overview(
                source,
                top_n_gods=int(top_n_gods),
                min_community_size=int(min_community_size),
            )
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_arch_overview(payload))

    @app.post("/api/playground/get_callers")
    def pg_get_callers(
        source: str = Form(...),
        symbol: str = Form(...),
        limit: int = Form(50),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        try:
            items = mgr.get_callers(source, symbol.strip(), limit=int(limit))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_simple_list(items))

    @app.post("/api/playground/get_callees")
    def pg_get_callees(
        source: str = Form(...),
        symbol: str = Form(...),
        limit: int = Form(50),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        try:
            items = mgr.get_callees(source, symbol.strip(), limit=int(limit))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        return HTMLResponse(_render_simple_list(items))


def _render_arch_overview(payload) -> str:
    """Render the architectural_overview dict — god nodes + communities."""
    if not isinstance(payload, dict):
        return ('<div class="p-3 bg-slate-50 border border-slate-200 rounded '
                'text-sm text-slate-500">No data.</div>')

    parts = []
    gods = payload.get("god_nodes") or payload.get("gods") or []
    communities = payload.get("communities") or []

    if gods:
        parts.append('<div class="mb-4">')
        parts.append('<div class="font-semibold text-slate-800 text-sm mb-2">God nodes</div>')
        parts.append('<ul class="space-y-1">')
        for g in gods:
            if isinstance(g, dict):
                label = g.get("symbol") or g.get("name") or g.get("id") or "?"
                detail_bits = []
                for k in ("in_degree", "out_degree", "degree", "centrality"):
                    if g.get(k) is not None:
                        detail_bits.append(f"{k}={g[k]}")
                detail = f' <span class="text-xs text-slate-500">{" · ".join(detail_bits)}</span>' if detail_bits else ''
            else:
                label, detail = str(g), ""
            parts.append(
                f'<li class="p-2 bg-white border border-slate-200 rounded text-sm">'
                f'<span class="font-mono">{_html_escape(label)}</span>{detail}'
                f'</li>'
            )
        parts.append('</ul></div>')

    if communities:
        parts.append('<div>')
        parts.append(f'<div class="font-semibold text-slate-800 text-sm mb-2">'
                     f'Communities ({len(communities)})</div>')
        parts.append('<div class="space-y-2">')
        for i, c in enumerate(communities, start=1):
            if isinstance(c, dict):
                members = c.get("members") or c.get("nodes") or []
                label = c.get("label") or c.get("name") or f"Community {i}"
                size = len(members) if isinstance(members, (list, tuple)) else c.get("size", "?")
                preview = ", ".join(_html_escape(str(m)) for m in members[:8])
                if len(members) > 8:
                    preview += f", … (+{len(members) - 8} more)"
                parts.append(
                    f'<div class="p-2 bg-white border border-slate-200 rounded text-sm">'
                    f'  <div class="font-medium text-slate-700">{_html_escape(label)} '
                    f'    <span class="text-xs text-slate-500">({size} members)</span>'
                    f'  </div>'
                    f'  <div class="font-mono text-xs text-slate-600 mt-1">{preview}</div>'
                    f'</div>'
                )
            else:
                parts.append(
                    f'<div class="p-2 bg-white border border-slate-200 rounded text-sm">'
                    f'{_html_escape(str(c))}</div>'
                )
        parts.append('</div></div>')

    if not parts:
        return ('<div class="p-3 bg-slate-50 border border-slate-200 rounded '
                'text-sm text-slate-500">Graph layer returned empty result.</div>')
    return "".join(parts)
