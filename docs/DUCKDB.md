# Lynx + DuckDB

Join Lynx's **local semantic code search** (and its code graph) with **any data
DuckDB can read** — Parquet, CSV, SQLite, Postgres, JSON — in one SQL engine, on
your machine. Lynx serves its results as NDJSON over a localhost HTTP API; DuckDB
reads that URL as a table.

This is the "total flexibility" path: shape and join the data in DuckDB, then
hand the result to whatever you like (an LLM, a notebook, a report).

## Prerequisites

1. A running Lynx API with at least one indexed source:

   ```bash
   lynx manager ui --port 8765 --no-browser   # serves 127.0.0.1:8765
   ```

2. DuckDB with the `httpfs` + `json` extensions (read remote URLs as tables):

   ```sql
   INSTALL httpfs; LOAD httpfs;
   INSTALL json;   LOAD json;
   ```

Everything stays local — `httpfs` only ever talks to `127.0.0.1`.

## The endpoints

Pass `format=ndjson` so each row is one JSON object on its own line. DuckDB reads
it with `read_ndjson_auto('url')` (or, equivalently,
`read_json('url', format='newline_delimited')`). URL-encode the query
(`spaces` → `%20`).

| Endpoint | Rows |
|---|---|
| `GET /api/v1/search?q=...&source=...&top_k=...&format=ndjson` | `source, file, file_path, symbol, kind, language, start_line, end_line, score, content` |
| `GET /api/v1/graph?operation=...&symbol=...&format=ndjson` | `relation, base_kind, confidence, module, from_symbol, from_file, …, to_symbol, to_file, …, call_site_file, call_site_line` |
| `GET /api/v1/sources?format=ndjson` | `name, type, location, chunk_count, last_update` |

## Recipes

**1. Code search as a table.**

```sql
SELECT file, symbol, score
FROM read_ndjson_auto(
  'http://127.0.0.1:8765/api/v1/search?q=where%20the%20camera%20zoom%20is%20clamped&top_k=20&format=ndjson'
)
WHERE language = 'c_sharp'
ORDER BY score DESC;
```

**2. Join code hits with local data** (e.g. a CSV/Parquet export of tickets).

```sql
WITH hits AS (
  SELECT * FROM read_ndjson_auto(
    'http://127.0.0.1:8765/api/v1/search?q=retry%20logic%20for%20payment%20webhooks&format=ndjson')
)
SELECT t.id, t.title, h.file, h.symbol, h.score
FROM read_csv_auto('tickets.csv') AS t
CROSS JOIN hits AS h            -- broad context pairing; keep top_k small
ORDER BY t.id, h.score DESC;
```

**3. Blast radius from the code graph** — pivot a search hit's symbol to who
calls it / what breaks if it changes, then join with ownership data.

```sql
SELECT from_symbol AS caller, to_symbol AS callee, call_site_file, call_site_line
FROM read_ndjson_auto(
  'http://127.0.0.1:8765/api/v1/graph?operation=callers&symbol=ApplyDamage&format=ndjson');
```

## Per-row correlation (one search per ticket)

A SQL `JOIN` can't drive `lynx.search` from another table's column — the query
string has to be a literal in the URL. For "for each ticket, find the most
relevant code", use the **batch** endpoint (`POST /api/v1/search`, all queries in
one embedding call) from a few lines of Python, then register the result back
into DuckDB:

```python
import duckdb, requests, pandas as pd

con = duckdb.connect()
tickets = con.execute(
    "SELECT id, title FROM read_csv_auto('tickets.csv')"
).fetchall()

resp = requests.post(
    "http://127.0.0.1:8765/api/v1/search",
    json={"queries": [t[1] for t in tickets], "source": "framework", "top_k": 3},
).json()

rows = [
    {"ticket_id": tickets[i][0], "query": r["query"], **hit}
    for i, r in enumerate(resp["results"])
    for hit in r["hits"]
]
con.register("correlations", pd.DataFrame(rows))
print(con.execute(
    "SELECT ticket_id, file, symbol, score "
    "FROM correlations ORDER BY ticket_id, score DESC"
).df())
```

That's the "DuckDB + Python helper" pattern: batch the per-row searches once,
then do all the joining/shaping in SQL.

## Notes

- **Local-first.** The codebase and embeddings never leave your machine; DuckDB
  only reads from `127.0.0.1`.
- The default (no `format`) returns the wrapped `{"results": [...]}` object used
  by Coral and other consumers; `format=ndjson` is purely additive.
- See also [CORAL.md](CORAL.md) for the SQL-source integration and the full
  `/api/v1` surface.
