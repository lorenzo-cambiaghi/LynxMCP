"""Self-contained graph views for humans.

Turns a slice of the code graph into a single, offline `.html` file (inline
SVG, no external CDN, no JavaScript) you can attach to a PR, email, or archive
for an audit. Two modes:

  - symbol : the blast radius of a symbol — callers above, callees below,
             laid out radially/by hop distance.
  - module : a file as a hub — what it imports (above) and which files depend
             on it (below), with its public symbols listed alongside.

Design notes in docs/DESIGN_graph_html_export.md. The renderer is deterministic
(seeded purely by sorted labels) so the same graph yields the same file.
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timezone


# Role -> fill colour. Roles drive both layout layer and legend.
_ROLE_COLOR = {
    "seed": "#4f46e5",       # indigo — the thing you asked about
    "caller": "#2563eb",     # blue   — reaches the seed
    "callee": "#059669",     # green  — reached by the seed
    "import": "#6b7280",     # gray   — modules this file imports
    "dependent": "#ea580c",  # orange — files that depend on this one
}

_NODE_W = 168
_NODE_H = 30
_H_GAP = 24
_ROW_GAP = 96
_MARGIN = 40
_MAX_LABEL = 28


# ---------------------------------------------------------------------------
# Graph slicing (directed BFS over the live nx.DiGraph)
# ---------------------------------------------------------------------------


def _node_view(G, nid: str, role: str, layer: int) -> dict:
    d = G.nodes[nid]
    return {
        "id": nid,
        "label": d.get("label", nid),
        "kind": d.get("kind"),
        "file": d.get("file"),
        "start_line": d.get("start_line"),
        "role": role,
        "layer": layer,
    }


def _bfs(G, seeds, reverse: bool, max_depth: int, limit: int):
    """Directed BFS. reverse=True walks `calls` in-edges (callers), False walks
    out-edges (callees). Yields (nid, depth), shallowest depth first, each node
    once. Self-loops can't occur (the builder drops them)."""
    visited = set(seeds)
    frontier = list(seeds)
    found = 0
    for depth in range(1, max_depth + 1):
        nxt = []
        for nid in frontier:
            edges = G.in_edges(nid, data=True) if reverse else G.out_edges(nid, data=True)
            for a, b, data in edges:
                if data.get("relation") != "calls":
                    continue
                other = a if reverse else b
                if other in visited:
                    continue
                visited.add(other)
                nxt.append(other)
                yield other, depth
                found += 1
                if found >= limit:
                    return
        frontier = nxt
        if not frontier:
            return


def build_symbol_view(G, symbol: str, *, depth: int = 2, root=None,
                      max_nodes: int = 150) -> dict:
    """Blast-radius view: seed + transitive callers (above) + callees (below)."""
    from .query import find_symbols

    depth = max(1, min(depth, 6))
    seeds = find_symbols(G, symbol)
    if not seeds:
        return {"empty": True, "reason": f"symbol {symbol!r} not found in the graph"}

    nodes: dict = {}
    for sid in seeds:
        nodes[sid] = _node_view(G, sid, "seed", 0)

    budget = max_nodes - len(nodes)
    truncated = False
    for nid, d in _bfs(G, seeds, reverse=True, max_depth=depth, limit=budget):
        nodes.setdefault(nid, _node_view(G, nid, "caller", -d))
    for nid, d in _bfs(G, seeds, reverse=False, max_depth=depth, limit=budget):
        if nid not in nodes:
            nodes[nid] = _node_view(G, nid, "callee", d)
    if len(nodes) >= max_nodes:
        truncated = True

    edges = []
    ids = set(nodes)
    for u, v, data in G.edges(data=True):
        if data.get("relation") == "calls" and u in ids and v in ids:
            edges.append({"src": u, "tgt": v})

    return {
        "title": f"Blast radius — {symbol}",
        "nodes": list(nodes.values()),
        "edges": edges,
        "sidebar": [],
        "truncated": truncated,
        "root": str(root) if root else None,
    }


def build_module_view(G, file_substring: str, *, root=None, max_nodes: int = 200) -> dict:
    """Hub view of a file: imports (above) + dependents (below), with the
    file's defined symbols listed in the sidebar."""
    from .query import nodes_in_file, get_imports, get_callers

    symbols = nodes_in_file(G, file_substring, limit=max_nodes)
    if not symbols:
        return {"empty": True, "reason": f"no symbols found for file {file_substring!r}"}

    own_file = symbols[0].get("file")
    seed_id = f"__file__:{file_substring}"
    nodes = {seed_id: {"id": seed_id, "label": os.path.basename(own_file or file_substring),
                       "kind": "file", "file": own_file, "start_line": None,
                       "role": "seed", "layer": 0}}
    edges = []

    # Imports (above): module nodes this file pulls in.
    for e in get_imports(G, file_substring, limit=100):
        tgt = e.get("target") or {}
        label = e.get("module") or tgt.get("label") or "?"
        iid = f"__import__:{label}"
        nodes.setdefault(iid, {"id": iid, "label": label, "kind": "module",
                               "file": None, "start_line": None,
                               "role": "import", "layer": -1})
        edges.append({"src": seed_id, "tgt": iid})

    # Dependents (below): distinct files that call a symbol defined here.
    dependents: set = set()
    for s in symbols:
        for e in get_callers(G, s["label"], limit=50):
            cf = (e.get("source") or {}).get("file")
            if cf and cf != own_file:
                dependents.add(cf)
    for df in sorted(dependents):
        did = f"__dep__:{df}"
        nodes[did] = {"id": did, "label": os.path.basename(df), "kind": "file",
                      "file": df, "start_line": None, "role": "dependent", "layer": 1}
        edges.append({"src": did, "tgt": seed_id})

    sidebar = [f"{s.get('label')}  [{s.get('kind')}]" for s in symbols]
    return {
        "title": f"Module — {os.path.basename(own_file or file_substring)}",
        "nodes": list(nodes.values()),
        "edges": edges,
        "sidebar": sidebar,
        "truncated": len(symbols) >= max_nodes,
        "root": str(root) if root else None,
    }


# ---------------------------------------------------------------------------
# Rendering (layered SVG → self-contained HTML)
# ---------------------------------------------------------------------------


def _rel(path, root) -> str:
    if not path:
        return ""
    if not root:
        return str(path)
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _trunc(label: str) -> str:
    return label if len(label) <= _MAX_LABEL else label[: _MAX_LABEL - 1] + "…"


def _layout(nodes):
    """Assign (x, y) to every node: one row per layer (sorted ascending so the
    most-negative layer is at the top), nodes spread left→right by sorted label
    within the row. Returns (positions, width, height)."""
    layers: dict = {}
    for n in nodes:
        layers.setdefault(n["layer"], []).append(n)
    ordered = sorted(layers)
    widest = max((len(v) for v in layers.values()), default=1)
    width = max(800, _MARGIN * 2 + widest * (_NODE_W + _H_GAP) - _H_GAP)
    height = _MARGIN * 2 + (len(ordered)) * (_NODE_H + _ROW_GAP) - _ROW_GAP

    pos = {}
    for row_i, layer in enumerate(ordered):
        row = sorted(layers[layer], key=lambda n: (n["label"].lower(), n["id"]))
        row_w = len(row) * (_NODE_W + _H_GAP) - _H_GAP
        x0 = (width - row_w) / 2
        y = _MARGIN + row_i * (_NODE_H + _ROW_GAP)
        for i, n in enumerate(row):
            pos[n["id"]] = (x0 + i * (_NODE_W + _H_GAP), y)
    return pos, width, height


def render_html(model: dict, *, source: str = "", lynx_version: str = "") -> str:
    """Render a view model to a single self-contained HTML string."""
    if model.get("empty"):
        body = f"<p class='empty'>Nothing to show: {_esc(model.get('reason'))}</p>"
        return _PAGE.format(title=_esc("Graph view"), meta="", body=body,
                            legend="", sidebar="")

    nodes = model["nodes"]
    pos, width, height = _layout(nodes)
    root = model.get("root")

    # Edges first (under nodes). Straight lines, centre-to-centre, arrowhead.
    seg = []
    pset = pos
    for e in model["edges"]:
        if e["src"] not in pset or e["tgt"] not in pset:
            continue
        x1, y1 = pset[e["src"]]
        x2, y2 = pset[e["tgt"]]
        seg.append(
            f'<line x1="{x1 + _NODE_W/2:.0f}" y1="{y1 + _NODE_H:.0f}" '
            f'x2="{x2 + _NODE_W/2:.0f}" y2="{y2:.0f}" '
            f'class="edge" marker-end="url(#arrow)"/>'
        )

    boxes = []
    roles_seen = set()
    for n in nodes:
        x, y = pset[n["id"]]
        color = _ROLE_COLOR.get(n["role"], "#334155")
        roles_seen.add(n["role"])
        loc = _rel(n.get("file"), root)
        if n.get("start_line"):
            loc = f"{loc}:{n['start_line']}" if loc else f"L{n['start_line']}"
        title = _esc(f"{n['label']}  ({n.get('kind') or '?'})" + (f"  {loc}" if loc else ""))
        boxes.append(
            f'<g class="node"><title>{title}</title>'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{_NODE_W}" height="{_NODE_H}" '
            f'rx="6" fill="{color}"/>'
            f'<text x="{x + _NODE_W/2:.0f}" y="{y + _NODE_H/2 + 4:.0f}" '
            f'text-anchor="middle" class="nlabel">{_esc(_trunc(n["label"]))}</text></g>'
        )

    svg = (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
        f'height="{height:.0f}" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>'
        + "".join(seg) + "".join(boxes) + "</svg>"
    )

    legend = "".join(
        f'<span class="lg"><i style="background:{_ROLE_COLOR[r]}"></i>{_esc(r)}</span>'
        for r in ("seed", "caller", "callee", "import", "dependent") if r in roles_seen
    )

    sidebar = ""
    if model.get("sidebar"):
        items = "".join(f"<li>{_esc(x)}</li>" for x in model["sidebar"])
        sidebar = f"<aside><h3>Defines ({len(model['sidebar'])})</h3><ul>{items}</ul></aside>"

    trunc_note = " · <b>truncated</b>" if model.get("truncated") else ""
    meta = (
        f"source: {_esc(source) or '—'} · root: {_esc(root) or '—'} · "
        f"{len(nodes)} nodes / {len(model['edges'])} edges{trunc_note} · "
        f"lynx {_esc(lynx_version) or '?'} · "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    return _PAGE.format(
        title=_esc(model["title"]), meta=meta, body=svg,
        legend=f'<div class="legend">{legend}</div>', sidebar=sidebar,
    )


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         color: #0f172a; background: #f8fafc; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid #e2e8f0; background: #fff; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .meta {{ color: #64748b; font-size: 12px; }}
  .wrap {{ display: flex; align-items: flex-start; gap: 16px; padding: 16px 24px; }}
  .canvas {{ flex: 1; overflow: auto; background: #fff; border: 1px solid #e2e8f0;
            border-radius: 8px; }}
  .edge {{ stroke: #94a3b8; stroke-width: 1.2; }}
  .nlabel {{ fill: #fff; font: 12px monospace; }}
  .node:hover rect {{ stroke: #0f172a; stroke-width: 2; }}
  .legend {{ margin-top: 8px; }}
  .lg {{ margin-right: 14px; font-size: 12px; color: #475569; }}
  .lg i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px;
          margin-right: 5px; vertical-align: middle; }}
  aside {{ width: 240px; flex: 0 0 240px; background: #fff; border: 1px solid #e2e8f0;
          border-radius: 8px; padding: 12px; max-height: 80vh; overflow: auto; }}
  aside h3 {{ margin: 0 0 8px; font-size: 13px; }}
  aside ul {{ margin: 0; padding-left: 16px; }}
  aside li {{ font: 12px monospace; color: #334155; }}
  .empty {{ padding: 40px; color: #64748b; }}
</style></head>
<body>
<header><h1>{title}</h1><div class="meta">{meta}</div>{legend}</header>
<div class="wrap"><div class="canvas">{body}</div>{sidebar}</div>
</body></html>
"""
