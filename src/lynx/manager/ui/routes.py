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
from fastapi import Query, Request

from pydantic import BaseModel

from .app import _dispose_manager, _get_manager
from ...outline import doc_of, signature_for
from ..._format import (
    _format_deep_response,
    _format_describe_symbol,
    _format_impact,
    _format_module_summary,
    _format_repo_overview,
)


class BatchSearchRequest(BaseModel):
    """Body for POST /api/v1/search (multi-query). Module-level so FastAPI can
    resolve the annotation under `from __future__ import annotations`."""
    queries: list[str]
    source: Optional[str] = None
    top_k: int = 8


def _format_v1_hit(h: dict) -> dict:
    """Shape a backend hit into the stable /api/v1 JSON row."""
    return {
        "source": h.get("source", ""),
        "file": h.get("file", ""),
        "file_path": str(h.get("file_path", "")),
        "symbol": h.get("symbol_name", ""),
        "kind": h.get("symbol_kind", ""),
        "language": h.get("language", ""),
        "start_line": h.get("start_line") or 0,
        "end_line": h.get("end_line") or 0,
        "score": float(h.get("score") or 0.0),
        "content": h.get("content", ""),
    }


def _to_outline(row: dict) -> dict:
    """Drop the body, add a compact `signature` + `doc` for cheap triage. The
    row keeps `file_path` / `start_line` / `end_line`, so an agent reads the
    full body on demand instead of paying for every hit's body up front.
    Signature/doc derivation lives in `lynx.outline` (shared with the MCP
    `search(outline=true)` tool)."""
    content = row.get("content", "")
    out = {k: v for k, v in row.items() if k != "content"}
    out["signature"] = signature_for(content, row.get("kind", ""), row.get("language", ""))
    out["doc"] = doc_of(content, row.get("language", ""))
    return out


def _format_v1_edge(e: dict) -> dict:
    """Flatten a graph edge into a stable, JOIN-friendly /api/v1 JSON row.

    Edges come from the manager as `{source: node, target: node, relation,
    ...}`; Coral wants flat columns, so we expose `from_*` / `to_*` plus the
    call-site location (`from_file`/`from_line` on the edge itself)."""
    src = e.get("source") or {}
    tgt = e.get("target") or {}
    return {
        "relation": e.get("relation", ""),
        "base_kind": e.get("base_kind") or "",
        "confidence": e.get("confidence") or "",
        # For `imports` edges the imported module/path lives here (the target
        # node is often synthetic), so it's the meaningful column for imports.
        "module": e.get("module") or "",
        "from_symbol": src.get("label", ""),
        "from_kind": src.get("kind") or "",
        "from_file": src.get("file") or "",
        "from_start_line": src.get("start_line") or 0,
        "from_end_line": src.get("end_line") or 0,
        "to_symbol": tgt.get("label", ""),
        "to_kind": tgt.get("kind") or "",
        "to_file": tgt.get("file") or "",
        "to_start_line": tgt.get("start_line") or 0,
        "to_end_line": tgt.get("end_line") or 0,
        "call_site_file": e.get("from_file") or "",
        "call_site_line": e.get("from_line") or 0,
    }


def _rows_payload(rows: list, fmt: str, wrapper_key: str = "results"):
    """Render v1 rows as the default wrapped object (`{wrapper_key: [...]}`) or,
    when `fmt` is `ndjson`/`jsonl`, as newline-delimited JSON (one row per line).

    NDJSON drops straight into DuckDB (`read_json_auto(..., format='nd')`), `jq`
    and `pandas.read_json(lines=True)` without unwrapping. The default preserves
    the existing contract (Coral and other current consumers rely on it)."""
    if (fmt or "").strip().lower() in ("ndjson", "jsonl", "nd"):
        import json
        from fastapi.responses import Response
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        return Response(content=body, media_type="application/x-ndjson")
    return {wrapper_key: rows}


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

    # ------------------------------------------------------------------
    # Stable v1 JSON API — the integration surface for EXTERNAL tools
    # (e.g. the Coral source spec in integrations/coral/, scripts, CI).
    # Contract: versioned and additive-only — within /api/v1 fields may
    # be added but never renamed or removed. The unversioned /api/*
    # endpoints above remain UI-internal and may change freely.
    # ------------------------------------------------------------------

    @app.get("/api/v1/search")
    def api_v1_search(
        q: str,
        source: Optional[str] = None,
        top_k: int = 8,
        view: str = "full",
        fmt: str = Query("json", alias="format"),
    ):
        """Hybrid search as plain JSON rows.

        `source` omitted → all sources, RRF-fused (each row carries its
        source name). `top_k` clamped to [1, 50]. `format=ndjson` streams one
        row per line (DuckDB / jq / pandas friendly).

        `view=outline` drops each hit's `content` (the body) and returns a
        compact `signature` + `doc` instead — let an agent triage by signature
        and read the body on demand via `file_path`/`start_line`/`end_line`.
        Typically ~2.4x fewer tokens than the full bodies (see docs/OUTLINE.md).
        """
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        k = max(1, min(int(top_k), 50))
        try:
            if source:
                mgr.get(source)  # raises KeyError with available names
                hits = mgr.search(source, q, top_k=k)
                for h in hits:
                    h.setdefault("source", source)
            else:
                hits = mgr.search_all(q, top_k=k)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        results = [_format_v1_hit(h) for h in hits]
        if (view or "").strip().lower() == "outline":
            results = [_to_outline(r) for r in results]
        return _rows_payload(results, fmt)

    @app.post("/api/v1/search")
    def api_v1_search_batch(body: BatchSearchRequest):
        """Batch search: embed many queries in ONE model call.

        Body: `{"queries": [...], "source": "name"|null, "top_k": N}`.
        Returns `{"results": [{"query": "...", "hits": [...]}, ...]}` aligned to
        `queries`. For external multi-query consumers — e.g. an agent/script
        fanning one question across rows of another data source. (Coral calls
        the single-query GET per row and can't use this; see docs/CORAL.md.)

        A batch is atomic: queries are validated up front and, if the search
        fails, the whole call errors (no partial per-query error rows — keeps
        the response contract simple). Max 100 queries per call.
        """
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        queries = [q for q in (body.queries or []) if isinstance(q, str) and q.strip()]
        if not queries:
            raise HTTPException(
                status_code=400, detail="`queries` must be a non-empty list of strings"
            )
        if len(queries) > 100:
            raise HTTPException(status_code=400, detail="at most 100 queries per batch")
        k = max(1, min(int(body.top_k), 50))
        try:
            if body.source:
                mgr.get(body.source)  # raises KeyError with available names
                per_query = mgr.search_batch(body.source, queries, top_k=k)
                for hits in per_query:
                    for h in hits:
                        h.setdefault("source", body.source)
            else:
                # No source → fan across all sources per query (correct, but no
                # embedding batching across sources).
                per_query = [mgr.search_all(q, top_k=k) for q in queries]
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        results = [
            {"query": q, "hits": [_format_v1_hit(h) for h in hits]}
            for q, hits in zip(queries, per_query)
        ]
        return {"results": results}

    @app.get("/api/v1/graph")
    def api_v1_graph(
        operation: str,
        symbol: str,
        source: Optional[str] = None,
        relation: Optional[str] = None,
        depth: int = 1,
        limit: int = 50,
        fmt: str = Query("json", alias="format"),
    ):
        """Code knowledge-graph edges as plain JSON rows.

        Lets a SQL caller pivot from a `lynx.search` hit (a `symbol`) to its
        structural neighbourhood. `operation` is one of: `callers`, `callees`,
        `subclasses`, `superclasses`, `imports`, `neighbors`. `source` may be
        omitted when exactly one source has the graph layer. Symbol matching is
        fuzzy (case-insensitive substring), same as the `graph_query` MCP tool.
        Rows are flat edges (`from_*` → `to_*` with `relation`) so they JOIN
        cleanly with other Coral sources.
        """
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        op = (operation or "").strip().lower()
        valid_ops = {
            "callers", "callees", "subclasses", "superclasses",
            "imports", "neighbors",
        }
        if op not in valid_ops:
            raise HTTPException(
                status_code=400,
                detail=f"unknown operation {op!r}; expected one of {sorted(valid_ops)}",
            )
        if not symbol or not symbol.strip():
            raise HTTPException(status_code=400, detail="`symbol` is required")
        k = max(1, min(int(limit), 200))

        # Resolve the graph-enabled source (same policy as the MCP tool).
        graph_sources = [
            name for name, b in mgr.backends.items()
            if getattr(b, "graph", None) is not None
        ]
        if source is None:
            if len(graph_sources) == 1:
                source = graph_sources[0]
            elif not graph_sources:
                raise HTTPException(
                    status_code=404,
                    detail="no graph-enabled source is configured",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"multiple graph sources; pass `source` (one of {graph_sources})",
                )
        elif source not in graph_sources:
            raise HTTPException(
                status_code=404,
                detail=f"source {source!r} has no graph layer (graph sources: {graph_sources})",
            )

        try:
            if op == "callers":
                edges = mgr.get_callers(source, symbol, limit=k)
            elif op == "callees":
                edges = mgr.get_callees(source, symbol, limit=k)
            elif op == "subclasses":
                edges = mgr.get_subclasses(source, symbol, limit=k)
            elif op == "superclasses":
                edges = mgr.get_superclasses(source, symbol, limit=k)
            elif op == "imports":
                edges = mgr.get_imports(source, symbol, limit=k)
            else:  # neighbors
                edges = mgr.get_neighbors(
                    source, symbol,
                    relation_filter=relation,
                    depth=max(1, min(int(depth), 6)),
                    limit=k,
                )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _rows_payload([_format_v1_edge(e) for e in edges], fmt)

    @app.get("/api/v1/sources")
    def api_v1_sources(fmt: str = Query("json", alias="format")):
        """Configured sources as plain JSON rows (`format=ndjson` supported)."""
        mgr = _get_manager(app)
        if mgr is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.manager_error or "manager not initialized",
            )
        rows = [
            {
                "name": st.get("name", ""),
                "type": st.get("type", ""),
                "location": str(st.get("path") or st.get("url") or ""),
                "chunk_count": st.get("chunk_count") or 0,
                "last_update": st.get("last_update") or "",
            }
            for st in mgr.list_sources()
        ]
        return _rows_payload(rows, fmt, "sources")

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
        # utf-8-sig so a BOM-prefixed config (common when edited on Windows)
        # doesn't reach the editor textarea as a stray leading character that
        # the user then saves back, permanently.
        return {"path": str(p), "content": p.read_text(encoding="utf-8-sig")}

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

    # ------------------------------------------------------------------
    # Phase 7: build trigger + job polling
    # ------------------------------------------------------------------
    _register_build_routes(app)

    # ------------------------------------------------------------------
    # Phase 10: source CRUD + filesystem browser
    # ------------------------------------------------------------------
    # POST /api/sources, DELETE /api/sources/{name}, POST /api/sources/_detect
    # GET /api/fs/browse — all consumed by the new "Add source" UI flow.
    _register_source_crud_routes(app)

    # ------------------------------------------------------------------
    # Phase 8: integrations — MCP snippet + AI rules-file download
    # ------------------------------------------------------------------
    @app.get("/api/integrations/{client}/rules")
    def api_integrations_rules(client: str):
        """Return the generated rules-file content as a file download.

        Filename comes from the client's `rules_file` (CLAUDE.md /
        AGENTS.md / .cursor/rules/lynx.md), so the browser saves it
        with the right name out of the box.
        """
        from fastapi.responses import PlainTextResponse
        from . import integrations as integ
        c = integ.get_client(client)
        if c is None:
            raise HTTPException(status_code=404, detail=f"unknown client {client!r}")
        if not c.get("rules_file"):
            raise HTTPException(
                status_code=404,
                detail=f"client {client!r} doesn't have a recommended rules file",
            )
        mgr = _get_manager(app)
        source_names: list = []
        has_graph = False
        has_git = False
        if mgr is not None:
            try:
                for name, sc in mgr.config.sources.items():
                    source_names.append(name)
                    if (sc.get("graph") or {}).get("enabled"):
                        has_graph = True
                    if (sc.get("git_integration") or {}).get("enabled"):
                        has_git = True
            except Exception:
                pass
        content = integ.render_rules_for_sources(source_names, has_graph, has_git)
        # Use just the basename so the browser doesn't try to save
        # nested directories (`.cursor/rules/lynx.md` → `lynx.md`).
        from pathlib import Path
        filename = Path(c["rules_file"]).name
        return PlainTextResponse(
            content,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
            media_type="text/markdown",
        )


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


def _reports_dir(mgr):
    """Where exported graph views land — the same directory the MCP tool and
    `lynx graph export` write to, so a view is findable whichever produced it."""
    from ...config import reports_dir
    return reports_dir(mgr.config)


# A generated view's filename, and nothing else. The download endpoint takes
# a name straight from a URL, so it is matched against this rather than
# joined and hoped for: no separators, no dots to walk up with, .html only.
_REPORT_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9_.-]{1,120}\.html$")


def _render_tool_text(text: str) -> str:
    """Show a tool's own text output verbatim.

    Used for the tools whose answer is prose an agent reads — the composed
    ones (describe_symbol, impact, module_summary, repo_overview),
    deep_search, and every graph operation. Rendering them through the
    shared `_format_*` functions means the playground shows exactly what
    the agent receives, which is the point of a playground; a nicer
    UI-only view would display something no agent ever sees and would be a
    second rendering to keep in step.
    """
    return (
        '<pre class="p-3 bg-slate-50 border border-slate-200 rounded text-xs '
        'font-mono whitespace-pre-wrap overflow-x-auto max-h-[32rem]">'
        f'{_html_escape(text)}</pre>'
    )


# `_render_simple_list` and `_render_arch_overview` lived here to serve the
# three hand-wired graph panels. Consolidating those into one `graph_query`
# form left them without a caller: the graph now renders through the shared
# `_format_edge_lines` / overview text, the same output the agent gets.


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
        # search_diff returns a dict whose hits live under `hits`. This read
        # `results`, a key the backend has never produced, so the panel
        # rendered an empty list on every run — including the runs that
        # found something.
        results = payload.get("hits", []) if isinstance(payload, dict) else []
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
        # `note` is how the backend explains an empty answer ("no files
        # added/modified vs 'main'"); dropping it left the user staring at
        # a blank panel.
        if isinstance(payload, dict) and payload.get("note"):
            header += (f'<div class="text-xs text-slate-500 mb-2">'
                       f'{_html_escape(payload["note"])}</div>')
        return HTMLResponse(header + _render_hits(results))

    # ------------------------------------------------------------------
    # The tools whose answer IS a piece of prose the agent reads.
    #
    # These render the shared `_format_*` text in a <pre> rather than a
    # bespoke HTML view. The playground exists to show what your agent
    # gets when it calls a tool; a prettier UI-only rendering would show
    # something no agent ever sees, and would be a second thing to keep
    # correct. Hit lists keep their HTML rendering above — those are
    # scanned, not read.
    # ------------------------------------------------------------------

    def _tool_text(fn, *a, **kw):
        """Run a manager call and render its shared text, or an error."""
        try:
            return HTMLResponse(_render_tool_text(fn(*a, **kw)))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)

    @app.get("/api/reports/{name}")
    def api_report(name: str):
        """Serve a previously exported graph view.

        The name comes from a URL, so it is validated against a strict
        pattern AND the resolved path is checked to be inside the reports
        directory — the pattern alone would still be trusting `Path` to
        agree with it about what a separator is, which differs by platform.
        """
        from fastapi import HTTPException
        from fastapi.responses import FileResponse

        mgr, err = _mgr_or_err()
        if err: return err
        if not _REPORT_NAME_RE.match(name):
            raise HTTPException(status_code=404, detail="no such report")
        root = _reports_dir(mgr).resolve()
        target = (root / name).resolve()
        if target.parent != root or not target.is_file():
            raise HTTPException(status_code=404, detail="no such report")
        return FileResponse(target, media_type="text/html")

    @app.post("/api/playground/graph_query")
    def pg_graph_query(
        source: str = Form(...),
        operation: str = Form("callers"),
        symbol: str = Form(""),
        target: str = Form(""),
        relation: str = Form(""),
        depth: int = Form(1),
        limit: int = Form(50),
        max_hops: int = Form(8),
        top_n: int = Form(10),
        min_community_size: int = Form(3),
    ):
        """All ten graph operations behind one form.

        Replaces three hand-wired panels (get_callers, get_callees,
        architectural_overview) that between them covered less than a
        third of the operation set — the UI had no way to ask for
        subclasses, imports, a shortest path or the bridge edges. Same
        consolidation the MCP tool and `lynx graph query` already made,
        calling the same dispatcher.
        """
        mgr, err = _mgr_or_err()
        if err: return err
        from ...graph.dispatch import query_graph
        try:
            res = query_graph(
                mgr, source, operation,
                symbol=symbol.strip() or None,
                target=target.strip() or None,
                relation_filter=relation.strip() or None,
                depth=int(depth), limit=int(limit), max_hops=int(max_hops),
                top_n=int(top_n), min_community_size=int(min_community_size),
            )
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        if not res.ok:
            # A usage problem (missing symbol, unknown operation) — the
            # dispatcher already phrased it for a reader.
            return _err(res.text.removeprefix("Error: "), status=400)
        return HTMLResponse(_render_tool_text(res.text))

    @app.post("/api/playground/export_graph")
    def pg_export_graph(
        source: str = Form(...),
        target: str = Form(...),
        mode: str = Form("symbol"),
        depth: int = Form(2),
    ):
        """Render a shareable graph view and hand back a download link.

        The one tool that had a CLI and an MCP entry point but no UI —
        which was the wrong way round: it produces a self-contained HTML
        page whose whole purpose is to be opened in a browser and attached
        to a PR. Writes to the same reports directory the other two use, so
        a file exported here is where `lynx graph export` would have put it.
        """
        mgr, err = _mgr_or_err()
        if err: return err
        if not target.strip():
            return _err("target is empty")
        if mode not in ("symbol", "module"):
            return _err(f"mode must be 'symbol' or 'module', got {mode!r}")
        try:
            res = mgr.export_graph(source, mode, target.strip(),
                                   depth=int(depth))
        except KeyError as e:
            return _err(f"unknown source: {e}", status=404)
        except ValueError as e:
            return _err(str(e), status=400)
        except Exception as e:
            return _err(f"{type(e).__name__}: {e}", status=500)
        if res.get("empty"):
            return _empty(f"Nothing to export: {res.get('reason')}")

        out_path = _reports_dir(mgr) / res["suggested_name"]
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(res["content"], encoding="utf-8")
        except OSError as e:
            return _err(f"couldn't write the view: {e}", status=500)

        name = _html_escape(out_path.name)
        return HTMLResponse(
            '<div class="p-3 bg-green-50 border border-green-200 rounded '
            'text-sm text-green-900">'
            f'✓ Wrote <code class="font-mono">{_html_escape(str(out_path))}</code>'
            '<div class="mt-2">'
            f'<a href="/api/reports/{name}" target="_blank" '
            'class="px-3 py-1.5 bg-indigo-600 text-white rounded '
            'text-sm hover:bg-indigo-700 no-underline">Open the view →</a>'
            '</div></div>'
        )

    @app.post("/api/playground/describe_symbol")
    def pg_describe_symbol(
        source: str = Form(...),
        symbol: str = Form(...),
        callers_limit: int = Form(10),
        callees_limit: int = Form(10),
        tests_limit: int = Form(5),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        sym = symbol.strip()
        return _tool_text(
            lambda: _format_describe_symbol(sym, mgr.describe_symbol(
                source, sym, callers_limit=int(callers_limit),
                callees_limit=int(callees_limit), tests_limit=int(tests_limit),
            )),
        )

    @app.post("/api/playground/impact")
    def pg_impact(
        source: str = Form(...),
        symbol: str = Form(...),
        max_depth: int = Form(3),
        tests_limit: int = Form(10),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not symbol.strip():
            return _err("symbol is empty")
        sym = symbol.strip()
        return _tool_text(
            lambda: _format_impact(sym, mgr.impact_of(
                source, sym, max_depth=int(max_depth),
                tests_limit=int(tests_limit),
            )),
        )

    @app.post("/api/playground/module_summary")
    def pg_module_summary(
        source: str = Form(...),
        file: str = Form(...),
        limit: int = Form(40),
    ):
        mgr, err = _mgr_or_err()
        if err: return err
        if not file.strip():
            return _err("file is empty")
        target = file.strip()
        return _tool_text(
            lambda: _format_module_summary(
                target, mgr.module_summary(source, target, limit=int(limit)),
            ),
        )

    @app.post("/api/playground/repo_overview")
    def pg_repo_overview(source: str = Form(...)):
        mgr, err = _mgr_or_err()
        if err: return err
        return _tool_text(
            lambda: _format_repo_overview(mgr.repo_overview(source)),
        )

    @app.post("/api/playground/deep_search")
    def pg_deep_search(
        source: str = Form(...),
        queries: str = Form(...),
        top_k: int = Form(5),
    ):
        """Escalation search. The textarea takes one phrasing per line —
        the tool wants 2-4 genuinely different angles, and a line each is
        the clearest way to type them."""
        mgr, err = _mgr_or_err()
        if err: return err
        variants = [q.strip() for q in queries.splitlines() if q.strip()]
        if not variants:
            return _err("enter at least one query, one per line")
        return _tool_text(
            lambda: _format_deep_response(
                mgr.deep_search(source, queries=variants, top_k=int(top_k)),
                variants, f"source {source!r}", "",
            ),
        )



# ---------------------------------------------------------------------------
# Phase 7: source detail build endpoints
# ---------------------------------------------------------------------------


def _render_job_widget(job, source_name: str) -> str:
    """Render a job's current state as an HTML fragment.

    When the job is still running, the fragment includes an HTMX
    `hx-trigger="every 1s"` so the browser self-polls. When it's
    terminal (done/failed), polling stops naturally because the new
    fragment has no `hx-trigger`.
    """
    state = job.state
    duration = ""
    if job.started_at:
        end = job.ended_at or time.time()
        secs = end - job.started_at
        duration = f"{secs:.1f}s"

    if state == "running" or state == "queued":
        color = "bg-blue-50 border-blue-200 text-blue-900"
        icon = "⏳"
        msg = f"{state.capitalize()}… ({duration})"
        poll_attrs = (
            f'hx-get="/api/jobs/{job.id}/widget?source={_html_escape(source_name)}" '
            f'hx-trigger="every 1s" hx-swap="outerHTML"'
        )
    elif state == "done":
        color = "bg-green-50 border-green-200 text-green-900"
        icon = "✓"
        msg = f"Build complete in {duration}."
        poll_attrs = ""
    else:  # failed
        color = "bg-red-50 border-red-200 text-red-900"
        icon = "❌"
        msg = f"Build failed: {_html_escape(job.error or 'unknown error')}"
        poll_attrs = ""

    log_html = ""
    if job.log:
        log_html = (
            f'<details class="mt-2"><summary class="text-xs cursor-pointer">'
            f'Log ({len(job.log)} chars)</summary>'
            f'<pre class="mt-1 text-xs bg-white border border-slate-200 rounded p-2 '
            f'overflow-x-auto max-h-64">{_html_escape(job.log)}</pre>'
            f'</details>'
        )

    return (
        f'<div id="build-status" class="p-3 border rounded text-sm {color}" {poll_attrs}>'
        f'  <div class="flex items-center justify-between">'
        f'    <div>{icon} <strong>{_html_escape(msg)}</strong></div>'
        f'    <div class="text-xs opacity-70">job {job.id}</div>'
        f'  </div>'
        f'  {log_html}'
        f'</div>'
    )


def _register_build_routes(app) -> None:
    """POST /api/sources/{name}/build and the job polling endpoints."""
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse

    from .app import _get_manager as _gm
    from . import jobs as jobs_mod
    from . import lock as lock_mod

    @app.post("/api/sources/{name}/build")
    def api_source_build(name: str):
        mgr = _gm(app)
        if mgr is None:
            return _err(app.state.manager_error or "Manager not initialized.",
                        status=503)
        # Reject before kicking off the thread if another process holds
        # the SQLite write lock — two writers will corrupt the DB.
        try:
            backend = mgr.get(name)
        except KeyError:
            return _err(f"source {name!r} not found", status=404)

        from pathlib import Path
        storage_path = Path(mgr.config.storage_path) / name
        # Force a fresh probe — the 30s cache could be stale right after
        # the user shut down `lynx serve` and tried to build immediately.
        lock_mod.invalidate_cache(storage_path)
        if lock_mod.is_storage_locked(storage_path):
            return HTMLResponse(
                '<div id="build-status" class="p-3 border rounded text-sm '
                'bg-amber-50 border-amber-200 text-amber-900">'
                '🔒 <strong>Locked.</strong> Another process is writing to this '
                f'source. Stop <code>lynx serve</code> first, then retry.'
                '</div>',
                status_code=409,
            )
        # Also guard against double-clicking the button — refuse if we
        # already have a build job in flight for this source.
        existing = jobs_mod.has_running_job_for(f"build:{name}")
        if existing is not None:
            return HTMLResponse(
                _render_job_widget(existing, name),
                status_code=200,
            )

        # Kick off in a daemon thread. update() is blocking but typically
        # fast (seconds → minutes for big repos); the UI polls while it runs.
        jobs_mod.cleanup_old(max_age_sec=3600)

        def _target():
            mgr.update(name, force=True)
            # Invalidate the lock cache after a successful build so the
            # next dashboard render reflects the freshly-released lock.
            lock_mod.invalidate_cache(storage_path)

        job = jobs_mod.create_job(
            _target,
            label=f"build {name}",
            group=f"build:{name}",
            metadata={"source": name},
        )
        return HTMLResponse(_render_job_widget(job, name))

    @app.post("/api/sources/{name}/reset")
    def api_source_reset(name: str):
        """Wipe a (possibly corrupt) source's index and rebuild from scratch.

        Unlike build, this works on sources that failed to load — it goes
        through `manager.reset_source`, which deletes the storage dir before
        reconstructing the backend, so a corrupt index can be recovered without
        ever opening it. The data is disposable derived embeddings."""
        mgr = _gm(app)
        if mgr is None:
            return _err(app.state.manager_error or "Manager not initialized.",
                        status=503)
        if name not in mgr.config.sources:
            return _err(f"source {name!r} not found", status=404)

        existing = jobs_mod.has_running_job_for(f"reset:{name}")
        if existing is not None:
            return HTMLResponse(_render_job_widget(existing, name), status_code=200)

        jobs_mod.cleanup_old(max_age_sec=3600)
        from pathlib import Path
        storage_path = Path(mgr.config.storage_path) / name

        def _target():
            mgr.reset_source(name, rebuild=True)
            lock_mod.invalidate_cache(storage_path)

        job = jobs_mod.create_job(
            _target,
            label=f"reset {name}",
            group=f"reset:{name}",
            metadata={"source": name},
        )
        return HTMLResponse(_render_job_widget(job, name))

    @app.post("/api/manager/reload")
    def api_manager_reload():
        """Drop the cached manager so the next request rebuilds it, which
        re-runs the integrity probes. The 'index not verified' card uses this to
        re-check after the user has cleared whatever was holding the index (a
        concurrent build or a second Lynx process). Cheap here — the rebuild
        cost is paid on the page load that follows the redirect. Goes through
        _dispose_manager so the reload also RELEASES our ChromaDB handles,
        not just the Python reference — a reload that keeps the old locks
        alive defeats its own purpose."""
        _dispose_manager(app)
        return HTMLResponse("", headers={"HX-Redirect": "/"})

    @app.get("/api/jobs/{job_id}")
    def api_job_status(job_id: str):
        """JSON view of a job — for external callers / debugging."""
        j = jobs_mod.get_job(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        return jobs_mod.job_to_dict(j)

    @app.get("/api/jobs/{job_id}/widget")
    def api_job_widget(job_id: str, source: str = ""):
        """HTMX-friendly HTML view used by the source detail page polling."""
        j = jobs_mod.get_job(job_id)
        if j is None:
            return HTMLResponse(
                '<div id="build-status" class="p-3 border rounded text-sm '
                'bg-slate-50 border-slate-200 text-slate-500">'
                f'Job {_html_escape(job_id)} no longer in memory.</div>',
                status_code=404,
            )
        return HTMLResponse(_render_job_widget(j, source or (j.metadata.get("source") or "")))


# ---------------------------------------------------------------------------
# Phase 10: source CRUD + filesystem browser
# ---------------------------------------------------------------------------


# The config-mutation rules live in `manager/sources.py` (FastAPI-free) so
# `lynx source add` shares them without importing this module. Re-exported
# under their old private names: tests and any external caller that reached
# for `routes._apply_codebase_defaults` keep resolving.
from ..sources import (  # noqa: E402,F401 — F401: several are pure re-exports
    SOURCE_NAME_RE as _SOURCE_NAME_RE,
    DEFAULT_CODEBASE_IGNORES as _DEFAULT_CODEBASE_IGNORES,
    apply_codebase_defaults as _apply_codebase_defaults,
    load_config_dict as _load_config_dict,
    validate_and_write_config as _validate_and_write_config,
    add_source as _add_source,
    remove_source as _remove_source,
)

# Roots we never list on the filesystem browser — these are kernel
# virtual filesystems on Linux that are either huge, slow, or weird to
# walk and have no value to anyone picking a folder to index.
_FS_BROWSE_SKIP_DIRS = {"/proc", "/sys", "/dev"}

# The folder-picker breadcrumb can hand us several mangled shapes of a Windows
# drive reference that Path() then corrupts. Every one of them is *drive-
# relative* or *UNC-flavoured* when the user clearly meant an absolute path:
#   - Path("C:")      drive-relative -> resolves to the CWD on that drive, so
#                     the picker silently jumps into the server's own folder.
#   - Path("C:Users") drive-relative -> "<cwd>\Users", a non-existent path
#                     (the "path does not exist: C:Users" the picker shows).
#   - Path("//C:")    read as a UNC root -> "\\C:", a dead path from which no
#                     further navigation recovers ("irrecuperabile").
# A folder picker only ever points at absolute locations, so we force a root
# separator right after the drive letter. This makes the browser self-heal no
# matter what shape the breadcrumb sends (incl. a stale/cached frontend).
import re  # noqa: E402 — local to the FS browser; keeps module top focused

_DRIVE_WITH_SLASHES = re.compile(r"^[\\/]+([A-Za-z]:)(.*)$")
# Drive letter followed by something that is NOT a separator (incl. nothing):
# "C:", "C:Users", "C:foo\bar" — all drive-relative, none absolute.
_DRIVE_RELATIVE = re.compile(r"^([A-Za-z]:)(?![\\/])(.*)$")


def _normalize_browse_path(raw: str) -> str:
    """Repair slash-prefixed / drive-relative Windows drive paths from the picker.

    Leaves POSIX paths and ordinary absolute Windows paths untouched; only
    rewrites the drive-reference shapes that Path() would otherwise corrupt.
    """
    s = raw.strip()
    if not s:
        return s
    # "/C:foo", "//C:\foo", "\\C:\foo" -> "C:foo" / "C:\foo": drop the slashes
    # that wrongly precede the drive letter. (Genuine UNC paths like
    # "\\server\share" don't match — the char after the slashes isn't "X:".)
    m = _DRIVE_WITH_SLASHES.match(s)
    if m:
        s = m.group(1) + m.group(2)
    # "C:" -> "C:\", "C:Users" -> "C:\Users": insert the missing root separator
    # so a drive-relative reference can't drift to the server's CWD.
    m = _DRIVE_RELATIVE.match(s)
    if m:
        s = m.group(1) + "\\" + m.group(2)
    return s


def _toast_ok(html_body: str) -> str:
    """Wrap a success message in the green-toast div HTMX swaps in."""
    return (
        '<div class="p-3 bg-green-50 border border-green-200 rounded '
        'text-green-900 text-sm">'
        f'✓ {html_body}'
        '</div>'
    )


def _toast_err(html_body: str) -> str:
    """Wrap an error message in the red-toast div HTMX swaps in."""
    return (
        '<div class="p-3 bg-red-50 border border-red-200 rounded '
        'text-red-900 text-sm">'
        f'❌ {html_body}'
        '</div>'
    )


def _register_source_crud_routes(app) -> None:
    """POST/DELETE /api/sources/* and GET /api/fs/browse."""
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    def _require_config_path():
        if app.state.config_path is None:
            return None, HTMLResponse(
                _toast_err("No config path configured — launch the UI with --config PATH."),
                status_code=400,
            )
        from pathlib import Path
        p = Path(app.state.config_path)
        if not p.exists():
            return None, HTMLResponse(
                _toast_err(f"Config not found at {p} — run `lynx manager init` first."),
                status_code=404,
            )
        return p, None

    @app.post("/api/sources")
    async def api_add_source(request: Request):
        """Add a new source block to config.json.

        Body: {"name": "...", "block": {"type": "codebase"|"webdoc"|"pdf", ...}}
        On success: returns 200 + success toast + sets HX-Redirect header
        so HTMX navigates to the new source's detail page.
        """
        config_path, err = _require_config_path()
        if err: return err
        try:
            payload = await request.json()
        except Exception as e:
            return HTMLResponse(_toast_err(f"Invalid JSON body: {e}"), status_code=400)

        name = (payload.get("name") or "").strip()
        res = _add_source(config_path, name, payload.get("block"))
        if not res.ok:
            return HTMLResponse(_toast_err(_html_escape(res.message)),
                                status_code=res.status)

        # Invalidate cached manager so the new source becomes visible.
        app.state.manager = None
        app.state.manager_error = None

        # Tell HTMX to navigate to the new source detail page after
        # success. Falls back to swapping the toast if the caller isn't
        # using HTMX (vanilla fetch will see the body + header).
        return HTMLResponse(
            _toast_ok(f"<strong>{_html_escape(res.message)}</strong>"),
            status_code=200,
            headers={"HX-Redirect": f"/sources/{name}"},
        )

    @app.delete("/api/sources/{name}")
    def api_delete_source(name: str, purge: bool = False):
        """Remove a source from config.json. With `?purge=true`, also
        wipe its ChromaDB storage directory on disk.
        """
        config_path, err = _require_config_path()
        if err: return err
        if purge:
            # This process holds the source's ChromaDB files open through
            # the cached manager (and chromadb's per-path system cache), so
            # without releasing them first the rmtree fails against OUR OWN
            # lock and the 409 tells the user to stop... this very UI.
            _dispose_manager(app)
        res = _remove_source(config_path, name, purge=purge)
        if not res.ok:
            # A failed purge (409) changed nothing at all — the source is
            # still listed, so the user can free the lock and click again.
            return HTMLResponse(_toast_err(_html_escape(res.message)),
                                status_code=res.status)

        app.state.manager = None
        app.state.manager_error = None

        purge_msg = ""
        if res.purged_path:
            purge_msg = f" Storage at <code>{_html_escape(res.purged_path)}</code> wiped."

        return HTMLResponse(
            _toast_ok(f"<strong>{_html_escape(res.message)}</strong>{purge_msg}"),
            status_code=200,
            headers={"HX-Redirect": "/sources"},
        )

    @app.post("/api/sources/_detect")
    async def api_detect_source(request: Request):
        """Probe a folder: return top-N file extensions and git-repo flag.

        Body: {"path": "/abs/or/relative/dir"}
        Returns: {"path", "exists", "is_dir", "extensions": [...], "is_git": bool}
        """
        try:
            payload = await request.json()
        except Exception as e:
            return JSONResponse({"error": f"Invalid JSON: {e}"}, status_code=400)
        raw_path = (payload.get("path") or "").strip()
        if not raw_path:
            return JSONResponse({"error": "`path` is required"}, status_code=400)

        from pathlib import Path
        from . import detect
        p = Path(raw_path).expanduser()
        try:
            resolved = p.resolve()
        except OSError as e:
            return JSONResponse({"error": f"Couldn't resolve path: {e}"}, status_code=400)
        if not resolved.exists():
            return JSONResponse({
                "path": str(resolved),
                "exists": False,
                "is_dir": False,
                "extensions": [],
                "is_git": False,
            })
        if not resolved.is_dir():
            return JSONResponse({
                "path": str(resolved),
                "exists": True,
                "is_dir": False,
                "extensions": [],
                "is_git": False,
            })
        return JSONResponse({
            "path": str(resolved),
            "exists": True,
            "is_dir": True,
            "extensions": detect.detect_extensions(resolved, top_n=10),
            "is_git": detect.is_git_repo(resolved),
        })

    @app.get("/api/fs/browse")
    def api_fs_browse(path: str = ""):
        """Single-level folder listing for the folder-picker modal.

        Returns JSON {path, parent, entries: [{name, is_dir}]} where
        `entries` contains ONLY subdirectories (files are omitted to
        keep the tree small). Empty `path` defaults to the user's home
        directory.

        Safety:
        - Resolves the path (follows symlinks) so the response can't
          leak via `..` traversal beyond what the user could see anyway.
        - Skips kernel virtual roots (/proc, /sys, /dev) on Linux.
        - 404 on non-existent paths or non-directories.
        """
        from pathlib import Path
        raw = _normalize_browse_path(path)
        if not raw:
            target = Path.home()
        else:
            target = Path(raw).expanduser()
        try:
            resolved = target.resolve()
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"Couldn't resolve path: {e}")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"path does not exist: {resolved}")
        if not resolved.is_dir():
            raise HTTPException(status_code=404, detail=f"path is not a directory: {resolved}")

        entries = []
        try:
            for child in sorted(resolved.iterdir(), key=lambda c: c.name.lower()):
                # The skip-set guards Linux virtual FS roots; on macOS /
                # Windows the paths just won't match and nothing is skipped.
                if str(child) in _FS_BROWSE_SKIP_DIRS:
                    continue
                try:
                    if not child.is_dir():
                        continue
                except OSError:
                    # Permission / IO errors on a specific entry: just skip it.
                    continue
                entries.append({"name": child.name, "is_dir": True})
        except PermissionError:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied reading {resolved}",
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"OS error: {e}")

        # parent = None when we're at the filesystem root (Path('/').parent == Path('/'))
        parent = str(resolved.parent) if resolved.parent != resolved else None
        return {
            "path": str(resolved),
            "parent": parent,
            "entries": entries,
        }


# Hoist `time` so _render_job_widget can use it without re-importing per call.
import time  # noqa: E402  — bottom of file so module top stays focused
