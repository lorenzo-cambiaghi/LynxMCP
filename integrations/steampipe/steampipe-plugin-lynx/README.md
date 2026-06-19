# steampipe-plugin-lynx

A [Steampipe](https://steampipe.io) plugin that exposes a local
[Lynx](https://github.com/lorenzo-cambiaghi/LynxMCP) instance — semantic code
search and the code knowledge graph — as SQL tables you can JOIN with
Steampipe's connectors (GitHub, Jira, AWS, …).

It is a thin SQL skin over Lynx's stable, already-shipped `/api/v1` HTTP API; the
design (tables, columns, qual → query-param mapping) is in
[`../DESIGN.md`](../DESIGN.md).

## Tables

| Table | Endpoint | Required quals |
|---|---|---|
| `lynx_source` | `GET /api/v1/sources` | — |
| `lynx_search` | `GET /api/v1/search` | `query` |
| `lynx_graph`  | `GET /api/v1/graph`  | `operation`, `symbol` |

## Why Steampipe (vs the Coral source)

Steampipe pushes WHERE quals **down** and runs a nested loop in joins, so it
calls the Lynx API **once per qual value** — `lynx_search` can be driven by
another table's column:

```sql
select t.key, h.file, h.symbol, h.score
from jira_issue t
join lynx_search h on h.query = t.summary   -- one Lynx call per issue
where t.status = 'Open';
```

(Coral resolves table-function args at plan time, so there you batch instead.
Honest caveat: N joined rows = N embedding passes; for large N, prefer Lynx's
`POST /api/v1/search` batch endpoint from a script.)

## Develop / build / test

Writing and compile-checking work anywhere (incl. Windows). **Running** needs the
Steampipe engine → macOS or Linux.

```bash
# 1. one-time: pull the SDK + transitive deps, write go.sum
go mod tidy

# 2. compile-check (Windows-friendly, no engine)
make build      # or: go build ./...

# 3. build + install as a local plugin (macOS/Linux)
make install    # -> ~/.steampipe/plugins/local/lynx/lynx.plugin

# 4. config: copy config/lynx.spc -> ~/.steampipe/config/lynx.spc
#    (start the backend first: lynx manager ui --port 8765 --no-browser)

# 5. query
steampipe query "select name, type, chunk_count from lynx_source;"
steampipe query "select file, symbol, score from lynx_search where query = 'where the camera zoom is clamped' and source = 'framework' and top_k = 10;"
steampipe query "select from_symbol, from_file from lynx_graph where operation = 'callers' and symbol = 'ApplyDamage';"
```

## Connection config

See [`config/lynx.spc`](config/lynx.spc). `api_url` is optional and falls back to
the `LYNX_API` env var, then `http://127.0.0.1:8765`.

## Layout

```
main.go                       plugin.Serve(PluginFunc: lynx.Plugin)
lynx/plugin.go                Plugin(): TableMap + config schema + default transform
lynx/connection_config.go     lynxConfig{ api_url } + GetConfig
lynx/client.go                HTTP client: GET /api/v1/... ?format=ndjson, stream rows
lynx/table_lynx_source.go     lynx_source
lynx/table_lynx_search.go     lynx_search  (query required; source, top_k optional)
lynx/table_lynx_graph.go      lynx_graph   (operation, symbol required; ...)
```

## License

Apache-2.0, to match Lynx.
