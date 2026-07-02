# Lynx

[Lynx](https://github.com/lorenzo-cambiaghi/LynxMCP) is a **100% local** semantic
code-search engine (an MCP server) for codebases, library docs, and PDFs. This
source exposes its local API as SQL, so you can query your code by *behavior*
and **join the results with live data** from other Coral sources — without a
byte leaving your machine.

- `lynx.sources` — the sources you've indexed (codebases / docs sites / PDFs).
- `lynx.search(q => '...', source => '...')` — ranked hybrid (dense + BM25)
  search. `q` is required; `source` is optional (omit it to search every
  indexed source at once). Control how many results Lynx returns with SQL
  `LIMIT` — Coral maps it to Lynx's `top_k` (clamped to 50). Each row carries
  `source`, `file`, `file_path`, `symbol`, `kind`, `language`,
  `start_line`/`end_line`, `score`, and the code `content` itself.
- **Code-graph functions** — pivot from a `lynx.search` hit's `symbol` to its
  structural neighbourhood (who calls it, what it depends on, what breaks if it
  changes), then JOIN that blast radius with live data:
  `lynx.callers`, `lynx.callees`, `lynx.subclasses`, `lynx.superclasses`,
  `lynx.imports`, and `lynx.neighbors` — each called as
  `lynx.callers(symbol => '...')` with an optional `source`. Rows are flat edges
  (`from_*` → `to_*` with `relation`, plus the `call_site_*`); SQL `LIMIT` caps
  the edge count. Symbol matching is fuzzy (case-insensitive substring).
  Available only for **codebase sources that have the graph layer enabled**.

## Setup

1. Install Lynx and index at least one source (it serves the API on
   `127.0.0.1`):

   ```bash
   pipx install lynx-mcp                     # or: uv tool install lynx-mcp
   lynx manager init                         # downloads the embedding model + writes a default config (NO sources yet)
   lynx manager ui --port 8765 --no-browser  # serves the local API on 127.0.0.1:8765
   ```

   `lynx manager init` writes a default config with **no sources yet** — it
   does not index anything on its own. Open <http://127.0.0.1:8765> and use
   **+ Add your first source** to add a codebase, docs site, or PDF folder.
   Adding a source only writes its config; it starts at **0 chunks**. On the
   source detail page that opens, click **Rebuild index** and wait for it to
   finish — that step is what actually indexes the source. A source with 0
   chunks returns empty Coral results, so do this before validating. Leave the
   `lynx manager ui` process running afterwards; it's the API Coral talks to.
   (Prefer a config-based setup? Add the source to `config.json`, run
   `lynx build --source <name>`, then start `lynx manager ui`. Don't run a CLI
   build against a source while another process is writing to it.)

2. Register this source. The only input is `LYNX_PORT` (default `8765`):

   ```bash
   coral source add --file sources/community/lynx/manifest.yaml
   coral source test lynx
   ```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `LYNX_PORT` | no | `8765` | Port of the local Lynx API (`lynx manager ui --port 8765`). |

## Examples

Find code by behavior, not keywords:

```sql
SELECT file, symbol, score
FROM lynx.search(q => 'clamp a value between a min and max')
WHERE language = 'c_sharp'
LIMIT 5;
```

Broad context: pair the top code hits with the repo's open PRs. This is a
`CROSS JOIN` — every hit against every open PR — so keep the `LIMIT`s small
and treat it as a context dump, not a precise match:

```sql
SELECT s.file, s.symbol, s.score, p.html_url
FROM lynx.search(q => 'retry logic for payment webhooks') s
CROSS JOIN github.pulls p
WHERE p.owner = 'your-org' AND p.repo = 'your-repo' AND p.state = 'open'
ORDER BY s.score DESC
LIMIT 20;
```

List what you've indexed:

```sql
SELECT name, type, chunk_count FROM lynx.sources;
```

### From a hit to its blast radius (the graph functions)

Find the code behind a behaviour, then ask the graph who calls it — the set of
things a change would ripple into:

```sql
-- 1. locate the symbol
SELECT symbol FROM lynx.search(q => 'verify a webhook signature') LIMIT 1;
-- 2. who would a change to it affect?
SELECT from_symbol, from_file, from_start_line
FROM lynx.callers(symbol => 'WebhookVerifier.verify')
LIMIT 50;
```

`callees` (what it calls), `superclasses`/`subclasses` (inheritance, either
direction), and `imports` (the imported `module`) work the same way.
`neighbors` returns every incident edge and also takes an optional `relation`
filter and a `depth` (hops):

```sql
SELECT relation, from_symbol, to_symbol
FROM lynx.neighbors(symbol => 'PaymentService', relation => 'calls', depth => 2)
LIMIT 50;
```

The payoff is joining structural impact with live data — e.g. who *owns* the
code a change ripples into, by pairing callers with open PRs:

```sql
SELECT c.from_symbol, c.from_file, p.html_url
FROM lynx.callers(symbol => 'PaymentService.charge') c
CROSS JOIN github.pulls p
WHERE p.owner = 'your-org' AND p.repo = 'your-repo' AND p.state = 'open'
LIMIT 20;
```

## Live validation

Captured with coral 0.4.2 against a running Lynx API (`lynx manager ui --port
8765`) and one indexed codebase source. Everything runs on `127.0.0.1`; local
paths redacted.

```console
$ LYNX_PORT=8765 coral source add --file sources/community/lynx/manifest.yaml
Added source lynx (secrets: none)

  ✓ lynx connected successfully
  Secrets: none

    lynx (1 table)
    └─ sources
    Query tests
    2 declared · 2 passed · 0 failed

    ✓ SELECT name, type, chunk_count FROM lynx.sources
      1 row

    ✓ SELECT file, symbol, score FROM lynx.search(q => 'where passwords are hashed') LIMIT 5
      5 rows

$ coral source test lynx

  ✓ lynx connected successfully
  Secrets: none

    lynx (1 table)
    └─ sources
    Query tests
    2 declared · 2 passed · 0 failed

    ✓ SELECT name, type, chunk_count FROM lynx.sources
      1 row

    ✓ SELECT file, symbol, score FROM lynx.search(q => 'where passwords are hashed') LIMIT 5
      5 rows

$ coral sql "SELECT name, type, chunk_count FROM lynx.sources"
+-----------+----------+-------------+
| name      | type     | chunk_count |
+-----------+----------+-------------+
| framework | codebase | 12656       |
+-----------+----------+-------------+

$ coral sql "SELECT file, symbol, score FROM lynx.search(q => 'clamp a value between a min and max') WHERE language = 'c_sharp' LIMIT 5"
+---------------------------+-----------------------------------------------------------------------------+----------------------+
| file                      | symbol                                                                      | score                |
+---------------------------+-----------------------------------------------------------------------------+----------------------+
| Math.cs                   | Framework.Utility.Math.Clamp                                                | 0.01639344262295082  |
| SplineSamplingConfig.cs   | Framework.SplineUtility.Sampler.SplineSamplingConfig.ClampToValidRanges     | 0.016129032258064516 |
| WheelGripConfiguration.cs | Framework.Vehicle.WheelGripConfiguration.ClampGrip                          | 0.015873015873015872 |
| MinMaxValueAttributes.cs  | Framework.Utility.MinMaxCacheSegmentsAttribute.MinMaxCacheSegmentsAttribute | 0.015384615384615384 |
| InspectorValidation.cs    | Framework.UIEditorKit.InspectorValidation.Range                             | 0.014925373134328358 |
+---------------------------+-----------------------------------------------------------------------------+----------------------+
```

`LIMIT` sets how many candidates Lynx returns (its `top_k`); any `WHERE` then
filters that fetched set. With no `WHERE`, `LIMIT n` returns up to `n` rows,
capped at 50:

| Query | Rows returned |
|-------|---------------|
| `SELECT file FROM lynx.search(...) LIMIT 3`   | 3 |
| `SELECT file FROM lynx.search(...) LIMIT 12`  | 12 — exceeds Lynx's default 8, so `LIMIT` is driving `top_k` |
| `SELECT file FROM lynx.search(...) LIMIT 100` | 50 — Lynx clamps `top_k` to its `[1, 50]` ceiling |

### Graph functions

Same run, against the same indexed codebase source (it has the graph layer
enabled). `Math.Clamp` is a real symbol in that tree; matching is fuzzy, so the
short name resolves to `Framework.Utility.Math.Clamp`:

```console
$ coral sql "SELECT from_symbol, from_kind, from_start_line, relation FROM lynx.callers(symbol => 'Math.Clamp') LIMIT 5"
+-----------------------------------------------------------------+-----------+-----------------+----------+
| from_symbol                                                     | from_kind | from_start_line | relation |
+-----------------------------------------------------------------+-----------+-----------------+----------+
| Framework.Utility.Math.Clamp01                                  | function  | 167             | calls    |
| Framework.Animation.Editor.TimelinePanel.SetFrame               | function  | 164             | calls    |
| Framework.Animation.Editor.TimelinePanel.FrameFromPanelPosition | function  | 449             | calls    |
| Framework.Animation.AvatarMoveEvent.Validate                    | function  | 67              | calls    |
| Framework.Animation.Events.PhasedStateMoveEvent.Validate        | function  | 62              | calls    |
+-----------------------------------------------------------------+-----------+-----------------+----------+

$ coral sql "SELECT relation, from_symbol, to_symbol FROM lynx.neighbors(symbol => 'Math.Clamp') LIMIT 6"
+----------+-------------------------------------------------------------------------------------+------------------------------+
| relation | from_symbol                                                                         | to_symbol                    |
+----------+-------------------------------------------------------------------------------------+------------------------------+
| calls    | Framework.Voxel.VoxelWorldGI.CreateResources                                        | Framework.Utility.Math.Clamp |
| calls    | Framework.ElementsOnTrackV2.TrackMovement.UpdateMovement                            | Framework.Utility.Math.Clamp |
| calls    | Framework.SplineUtility.Smoothing.SmoothingSplinesSmoothing.CalculateOptimalLambda  | Framework.Utility.Math.Clamp |
| calls    | Framework.Track.Editor.RuntimePreviewGenerator.Padding                              | Framework.Utility.Math.Clamp |
| calls    | Framework.SplineUtility.Smoothing.SmoothingSplinesSmoothing.GetWrappedIndex         | Framework.Utility.Math.Clamp |
| calls    | Framework.Utility.Editor.CachedAnimationCurvePropertyDrawer.CreateSegmentsContainer | Framework.Utility.Math.Clamp |
+----------+-------------------------------------------------------------------------------------+------------------------------+
```

Each row is a real call edge from the codebase's graph layer — the inbound set
for `callers` is exactly the blast radius of a change to `Math.Clamp`. SQL
`LIMIT` caps the edge count (Lynx's `limit`, clamped to 200).

## Notes

- **Local-first.** The codebase and embeddings never leave your machine; only
  the live-data side of a join touches an API.
- The search string is a **literal** you pass — Coral resolves table-function
  arguments at plan time, so `lynx.search` is a joinable source, not a per-row
  enrichment of another table. For one search per row of another table, Lynx
  ships a batch endpoint (`POST /api/v1/search`) and a small Python helper.
- Requires the Lynx API to be running (`lynx manager ui`). See the
  [Lynx Coral guide](https://github.com/lorenzo-cambiaghi/LynxMCP/blob/main/docs/CORAL.md).
