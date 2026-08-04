"""HTML fragments the UI swaps in — pure presentation, no routes.

Split out of `routes.py`, which had grown past 1500 lines with the
rendering interleaved between the endpoints that use it. Nothing here
touches `app` or a manager: every function takes data and returns a
string, which is also what makes them readable in isolation.
"""
from __future__ import annotations


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
