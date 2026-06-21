# Design — self-contained HTML export of the code graph

**Status:** draft for review (not implemented)
**Owner:** lcambiaghi
**Related:** `lynx/graph/query.py`, `impact`/`describe_symbol` tools, the
deterministic-SVG pattern already used for `docs/img/cost_savings.svg`.

---

## 1. Goal

Produce a **single, self-contained `.html` file** that visualizes a slice of
the code graph **for humans**: blast radius of a symbol, a module's
dependencies, an architecture snapshot. The file must be:

- **Offline / air-gapped** — no external CDN, no network at view time. (This is
  a hard requirement: the on-prem/regulated buyers who care most about a local
  code tool cannot load `d3` from a CDN.)
- **Distributable** — attach to a PR, email it, archive it for a compliance
  audit ("architecture at release X").
- **Zero build step** — one command (or one tool call) → one file you can
  double-click.

### Non-goals (v1)
- Not a live dashboard (the Manager UI already serves the interactive view).
- Not a whole-repo hairball dump — see §7 (scale).
- Not editing/annotation — read-only artifact.

---

## 2. Output modes

Ship **symbol-centric first** (small graphs, immediate value, lowest risk),
overview later.

| Mode | Shows | Primary user |
|---|---|---|
| **symbol** (v1) | Subgraph around one symbol: callers + callees up to N hops (the `impact` blast radius made visual) | Dev attaching "here's what this touches" to a PR |
| **module** (v2) | Symbols defined in a file + their imports + dependent files | Reviewer grasping a unit |
| **architecture** (v3) | Modules (communities) as clusters, hubs, cross-module edges, cycles in red | Architect / audit |

---

## 3. Surfaces

Two entry points over the same core (`lynx/graph/html_report.py`):

- **MCP tool** `export_graph_html` (codebase + graph sources):
  an agent can generate the file and hand back its path
  ("give me a visual of this module's dependencies for the team").
- **CLI** `lynx graph report` (for humans / CI):
  `lynx graph report --symbol GetVoxel --depth 2 --out blast.html`

Both resolve the graph the same way the existing graph tools do
(`backend.graph.graph`, `find_symbols`, `transitive_callers`, `get_callees`).

---

## 4. Data model

Reuse the existing dict shapes — no new graph code:

- **node**: `{id, label, kind, file, start_line, end_line, lang_key}`
  (from `query._node_to_dict`)
- **edge**: `{source_id, target_id, relation, confidence}`

The builder collects a **bounded** node/edge set for the chosen mode, then
hands it to the renderer. For the symbol mode:

```
seed = find_symbols(G, symbol)
nodes = seed ∪ callers(≤depth) ∪ callees(≤depth)   # capped, see §7
edges = calls-edges among `nodes`
```

Each node carries a role tag for coloring: `seed | caller | callee`.

---

## 5. Rendering approach

**v1: deterministic SVG, zero JavaScript.** Reuse the muscle already proven in
`cost_savings.svg`: compute a layout in Python, emit `<svg>` with `<circle>` /
`<line>` / `<text>`, wrap it in a minimal HTML shell with inline `<style>`.

Why SVG-first:
- Truly self-contained, **diffable**, printable, zero deps, zero CDN — the
  strongest fit for the air-gapped/compliance angle.
- No JS supply-chain or license to vendor.

**v2 (later, opt-in `--interactive`): a ~200-line vanilla-JS canvas renderer**
inlined into the file (hover, drag, click-to-open `file:line`). Still no
external lib — keeps the air-gap guarantee. Decision deferred; v1 ships value
without it.

> Rejected for v1: vendoring d3/cytoscape inline (+250KB–1MB and a license to
> track). Revisit only if the vanilla renderer proves too limiting.

---

## 6. File format & self-containment guarantees

- One `.html`, UTF-8, no external `<link>`/`<script src=>`.
- Structure: `<style>` (inline) + `<svg>` (the graph) + a legend + a small
  metadata header (source name, root, generated-at, lynx version, node/edge
  counts). Optionally an embedded `<script type="application/json">` data blob
  so v2's JS (or a downstream tool) can re-read the graph without re-running
  Lynx.
- **Escaping:** every label/file path goes through HTML-escaping (code symbols
  can contain `<`, `>`, `&`). This is a correctness *and* a safety requirement.

---

## 7. Scale handling (the part most viz tools get wrong)

A raw 10k-node graph is an unreadable hairball. Bound it up front:

- **symbol mode:** naturally small (N hops); hard-cap nodes (e.g. 150) and
  edges, with a visible "+N more (truncated)" note. Reuse the `limit` semantics
  already in `transitive_callers`.
- **architecture mode:** never raw nodes — aggregate to **modules**
  (communities) as the unit; edges = cross-module dependency counts.

If the cap is hit, the file says so explicitly rather than silently dropping
data (consistent with how the tools surface truncation today).

---

## 8. Layout algorithm

- **Deterministic** (seeded) so the same graph → the same file → clean diffs
  and reproducible audit artifacts.
- v1: a simple seeded force-directed / layered layout computed in Python
  (small graphs, so cost is negligible). For symbol mode, a **radial/layered**
  layout (seed in center, callers above, callees below, depth = ring) reads
  better than a generic force layout and is fully deterministic.

---

## 9. Privacy / security

- 100% local: no network at generate or view time (matches Lynx's promise).
- The file embeds **code structure** (symbols, file paths, line numbers) — NOT
  source bodies by default. Flag this in docs; offer `--no-paths` to redact
  absolute paths to repo-relative if a user wants to share externally.
- All embedded text HTML-escaped (§6).

---

## 10. Proposed API

**MCP tool**
```
export_graph_html(
    mode: "symbol" = "symbol",     # v1 only "symbol"
    symbol: str | None,            # required for mode=symbol
    source: str | None,            # omittable when one codebase source
    depth: int = 2,                # hops, 1..6
    out: str | None = None,        # path; default: <storage>/reports/<symbol>.html
) -> str   # human message + absolute path to the written file
```

**CLI**
```
lynx graph report --symbol GetVoxel --source game --depth 2 --out blast.html
```

Returns the path; non-zero exit on "symbol not found" / "graph not enabled".

---

## 11. Testing

- **Pure builder** (graph slice → render model): unit-test that the symbol
  slice contains the seed + expected callers/callees and respects caps. Uses
  the synthetic graph already built in `test_combined_tools._make_backend`.
- **Renderer**: assert the output is a single well-formed HTML string with no
  `src=`/`href=` pointing off-file (the air-gap invariant), that labels are
  escaped, and that the metadata header + legend are present. Pure string test,
  CI-friendly (pytest-style, like `test_result_quality.py`).
- **No browser/headless needed** for v1 (SVG is static).

---

## 12. Phasing

| Milestone | Scope | Effort |
|---|---|---|
| **M1** | `export_graph_html` symbol mode, SVG, CLI + MCP tool, tests | Small–Medium |
| **M2** | `--interactive` vanilla-JS renderer (still inline, no CDN) | Medium |
| **M3** | module mode | Small (reuses `module_summary`) |
| **M4** | architecture mode (communities + cycles), the audit artifact | Medium |

Ship **M1** first — demoable, fits the pitch, anchors the whole governance story.

---

## 13. Decisions to confirm before coding

1. **Tool name:** `export_graph_html` vs `graph_report` vs `graph_view`?
2. **Default output path:** `<storage>/reports/` vs current working dir?
3. **v1 layout:** radial/layered (recommended) vs generic force?
4. **Paths in output:** absolute by default with `--no-paths` opt-out — OK?
5. **Scope of M1:** symbol mode only, or also module mode in the first cut?
