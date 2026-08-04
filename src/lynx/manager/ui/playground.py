"""The playground endpoints — one panel per MCP tool.

Split out of `routes.py` for size: this is the surface that grows every
time a tool is added, and it was pushing the module past 1500 lines.

Two rules hold it together with the rest of Lynx:

  - the composed tools and the graph operations render their `_format_*`
    text VERBATIM, because a playground exists to show what your agent
    receives — a prettier UI-only view would display something no agent
    ever sees, and would be a second rendering to keep correct;
  - hit lists keep the HTML rendering in `render.py`; those are scanned,
    not read.
"""
from __future__ import annotations

from fastapi import Form

from .app import _get_manager
from .render import (
    _empty, _err, _html_escape, _render_hits, _render_tool_text,
)
from ..._format import (
    _format_deep_response,
    _format_describe_symbol,
    _format_impact,
    _format_module_summary,
    _format_repo_overview,
)


def _reports_dir(mgr):
    """Where exported graph views land — the same directory the MCP tool and
    `lynx graph export` write to, so a view is findable whichever produced it."""
    from ...config import reports_dir
    return reports_dir(mgr.config)


# A generated view's filename, and nothing else. The download endpoint takes
# a name straight from a URL, so it is matched against this rather than
# joined and hoped for: no separators, no dots to walk up with, .html only.
_REPORT_NAME_RE = __import__("re").compile(r"^[A-Za-z0-9_.-]{1,120}\.html$")


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



