# Steampipe plugin for Lynx — design spec

Design for a [Steampipe](https://steampipe.io) plugin that exposes Lynx's local
code search + code graph as SQL tables, joinable with Steampipe's large catalogue
of connectors (GitHub, Jira, AWS, …).

This is a **spec**, not an implementation — the plugin is Go and lives in its own
repo by convention (`steampipe-plugin-lynx`). Everything below maps onto Lynx's
already-shipped, tested `/api/v1` contract, so the Go code is a thin SQL skin
over HTTP.

## Why Steampipe (and how it differs from Coral)

Both turn Lynx into SQL. The key difference is **per-row correlation**:

- **Coral** resolves table-function arguments to constants at *plan* time, so
  `lynx.search` can't be driven by another table's column (you batch instead).
- **Steampipe** pushes WHERE quals *down* and runs a nested loop in joins, calling
  the API **once per qual value**. So this works directly:

  ```sql
  select t.key, h.file, h.symbol, h.score
  from jira_issue t
  join lynx_search h on h.query = t.summary      -- one Lynx call per issue
  where t.status = 'Open';
  ```

  Honest caveat: that's N API calls = N embedding passes. For large N it's still
  cheaper to use Lynx's batch endpoint (`POST /api/v1/search`) from a script; for
  interactive/moderate N, the Steampipe join is the nicest ergonomics.

## Prerequisite

A running local Lynx API (same as the Coral / DuckDB integrations):

```bash
lynx manager ui --port 8765 --no-browser
```

## Connection config (`~/.steampipe/config/lynx.spc`)

```hcl
connection "lynx" {
  plugin  = "lynx"
  # Base URL of the local Lynx API. Defaults to the env var LYNX_API or
  # http://127.0.0.1:8765.
  api_url = "http://127.0.0.1:8765"
}
```

The plugin reads NDJSON from the endpoints (`?format=ndjson`) — one row per
line, so streaming/decoding is trivial.

## Tables

Three tables, one per `/api/v1` endpoint. Column names/types come straight from
the JSON the endpoints already return.

### `lynx_source` — indexed sources

`GET /api/v1/sources`. No quals (lists everything).

| Column | Type | Source field |
|---|---|---|
| `name` | string | `name` |
| `type` | string | `type` (codebase \| webdoc \| pdf) |
| `location` | string | `location` |
| `chunk_count` | int | `chunk_count` |
| `last_update` | timestamp | `last_update` |

```sql
select name, type, chunk_count from lynx_source;
```

### `lynx_search` — semantic + lexical code search

`GET /api/v1/search`. The query string is a **required qual**; `source` and
`top_k` are optional quals.

**Key columns (quals):**

| Qual column | Require | → query param | Notes |
|---|---|---|---|
| `query` | required | `q` | what the code *does*, phrased behaviourally |
| `source` | optional | `source` | omit → search every source (RRF-fused) |
| `top_k` | optional | `top_k` | clamped server-side to [1, 50]; default 8 |

**Result columns** (one row per hit; `query`/`source`/`top_k` echo the quals):

| Column | Type | Source field |
|---|---|---|
| `source` | string | `source` |
| `file` | string | `file` |
| `file_path` | string | `file_path` |
| `symbol` | string | `symbol` |
| `kind` | string | `kind` |
| `language` | string | `language` |
| `start_line` | int | `start_line` |
| `end_line` | int | `end_line` |
| `score` | double | `score` |
| `content` | string | `content` |

```sql
select file, symbol, score
from lynx_search
where query = 'where the camera zoom is clamped'
  and source = 'framework'
  and top_k = 10;
```

### `lynx_graph` — code knowledge graph

`GET /api/v1/graph`. `operation` and `symbol` are **required quals**.

**Key columns (quals):**

| Qual column | Require | → query param | Notes |
|---|---|---|---|
| `operation` | required | `operation` | `callers`\|`callees`\|`subclasses`\|`superclasses`\|`imports`\|`neighbors` |
| `symbol` | required | `symbol` | fuzzy (case-insensitive substring) |
| `source` | optional | `source` | optional when one source has the graph layer |
| `relation_filter` | optional | `relation` | for `neighbors`: restrict to one edge relation (named distinctly from the `relation` result column below) |
| `depth` | optional | `depth` | for `neighbors`: 1–6 |
| `edge_limit` | optional | `limit` | max edges (server clamps to 200) |

**Result columns** (one flat edge per row):

| Column | Type | Source field |
|---|---|---|
| `relation` | string | `relation` |
| `base_kind` | string | `base_kind` |
| `confidence` | string | `confidence` |
| `module` | string | `module` (the imported path, for `imports`) |
| `from_symbol` | string | `from_symbol` |
| `from_kind` | string | `from_kind` |
| `from_file` | string | `from_file` |
| `from_start_line` | int | `from_start_line` |
| `from_end_line` | int | `from_end_line` |
| `to_symbol` | string | `to_symbol` |
| `to_kind` | string | `to_kind` |
| `to_file` | string | `to_file` |
| `to_start_line` | int | `to_start_line` |
| `to_end_line` | int | `to_end_line` |
| `call_site_file` | string | `call_site_file` |
| `call_site_line` | int | `call_site_line` |

```sql
-- Blast radius of a symbol, joined with GitHub ownership
select g.from_symbol as caller, g.from_file, o.owner
from lynx_graph g
left join github_codeowners o on o.path = g.from_file
where g.operation = 'callers' and g.symbol = 'ApplyDamage';
```

> Note on qual naming: `top_k`, `depth`, `edge_limit` are exposed as qual columns
> rather than relying on SQL `LIMIT`, because Steampipe's `LIMIT` is a row cap on
> the *result*, not a fetch budget for the upstream API. Keeping them explicit
> mirrors the `/api/v1` parameters one-to-one.

## Repo layout (`steampipe-plugin-lynx`)

```
steampipe-plugin-lynx/
├── main.go                     # plugin.Serve(PluginFunc: lynx.Plugin)
├── lynx/
│   ├── plugin.go               # Plugin(): TableMap + ConnectionConfigSchema + DefaultTransform
│   ├── connection_config.go    # lynxConfig{ APIURL *string }, GetConfig()
│   ├── client.go               # HTTP client: GET <api_url>/api/v1/... ?format=ndjson → []map
│   ├── table_lynx_source.go    # List hydrate → /api/v1/sources
│   ├── table_lynx_search.go    # List hydrate, KeyColumns{query required, source/top_k optional}
│   └── table_lynx_graph.go     # List hydrate, KeyColumns{operation+symbol required, ...}
├── config/lynx.spc             # example connection config (above)
├── docs/                       # Hub docs: index.md + one .md per table
├── Makefile                    # `make` → go build → ~/.steampipe/plugins/.../lynx.plugin
├── go.mod                      # module + steampipe-plugin-sdk/v5
└── README.md
```

## Implementation notes (for whoever writes the Go)

- **SDK:** `github.com/turbot/steampipe-plugin-sdk/v5` (current major). Tables use
  `*plugin.Table` with a `List: &plugin.ListConfig{ KeyColumns: ..., Hydrate: ... }`.
- **Quals → params:** in each List hydrate, read `d.EqualsQuals["query"]`,
  `d.EqualsQualString("source")`, etc., and build the URL query string. Mark
  required quals with `plugin.Required` in `KeyColumns`.
- **Streaming:** call the endpoint with `format=ndjson`, scan line by line, and
  `d.StreamListItem(ctx, row)` each decoded object. Respect `d.RowsRemaining(ctx)`.
- **Transform:** `transform.FromField("Field")` (or `FromGo()` with matching Go
  struct field names). Map snake_case JSON to columns directly.
- **Config default:** if `api_url` is unset, fall back to `LYNX_API` env then
  `http://127.0.0.1:8765`.
- **Errors:** a 404 from `/api/v1/graph` (no graph-enabled source) should surface
  as an empty result or a clear error, not a panic.

## Publishing

- Tag releases; the Steampipe Hub builds from the repo. Add `docs/index.md` and a
  per-table doc page (Hub convention).
- License Apache-2.0 to match Lynx.

## See also

- [docs/CORAL.md](../../docs/CORAL.md) — the other SQL surface (plan-time literal query).
- [docs/DUCKDB.md](../../docs/DUCKDB.md) — local analytics over the same `/api/v1` NDJSON.
- The endpoints this maps onto: `docs/CORAL.md#the-api-behind-it`.
