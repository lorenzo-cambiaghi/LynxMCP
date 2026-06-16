# Using Lynx from Coral (SQL over your code search)

[Coral](https://github.com/withcoral/coral) gives agents a local SQL runtime
over live data sources (GitHub, Sentry, Datadog, Linear, ...). Lynx covers
the part Coral doesn't: **semantic retrieval over unstructured local content**
— your code, your library docs, your PDFs. Together you get to **correlate
code search with the tools your team already lives in**, all on your machine:

- 🔎 Find logic by behavior, not keywords — ranked code locations for a
  plain-language question.
- 🔁 Locate the code behind a feature and line it up against the repo's open
  PRs before you refactor.
- 🚨 Take the behavior from a Sentry alert and get the ranked code that
  implements it.
- 🎫 Pull open tickets from Coral and (with the Python helper) batch-search
  Lynx to map each to its likely code area.

With the source spec in [`integrations/coral/manifest.yaml`](../integrations/coral/manifest.yaml),
Lynx becomes a Coral schema, and your agent can JOIN code search with live
data in one SQL statement:

```sql
-- "We have a Sentry error about webhook retries — where does that live,
--  and is there an open PR touching it?"
SELECT s.file, s.symbol, s.score, p.html_url, p.state
FROM lynx.search(q => 'retry logic for payment webhooks') s
CROSS JOIN github.pulls p
WHERE p.owner = 'your-org' AND p.repo = 'your-repo' AND p.state = 'open'
ORDER BY s.score DESC
LIMIT 5
```

Every `lynx.search` row carries `file`, `file_path`, qualified `symbol`,
`kind`, `language`, `start_line`/`end_line`, `score`, and the code `content`
itself — so the agent cites precisely without extra reads.

## Setup

1. **Start the Lynx local API** (same process as the manager UI; binds
   127.0.0.1 only, nothing leaves the machine):

   ```bash
   lynx manager ui --port 8765 --no-browser
   ```

2. **Register the source in Coral** (the only input is `LYNX_PORT`, default
   `8765`). See Coral's
   [custom source guide](https://withcoral.com/docs/guides/write-a-custom-source):

   ```bash
   coral source add --file integrations/coral/manifest.yaml
   coral source test lynx          # checks connectivity to the running API
   ```

3. **Query it.** `lynx.sources` is a table; `lynx.search` is a ranked
   retrieval *function* — call it with `q => '...'` (optional `source => ...`),
   don't filter it with `WHERE`. Control the row count with SQL `LIMIT`
   (Coral maps it to Lynx's `top_k`, clamped to 50):

   ```sql
   SELECT name, type, chunk_count FROM lynx.sources;
   SELECT file, symbol, score
   FROM lynx.search(q => 'where passwords are hashed') LIMIT 5;
   ```

## The API behind it

The spec talks to Lynx's stable local JSON API (additive-only within v1):

- `GET /api/v1/search?q=...&source=...&top_k=...` — single query.
- `GET /api/v1/sources`
- `POST /api/v1/search` — **batch**: body `{"queries": [...], "source": ..., "top_k": ...}`,
  returns `{"results": [{"query": ..., "hits": [...]}, ...]}`. Embeds all queries in
  one model call, so it's faster than N single calls. For external multi-query
  consumers (e.g. an agent fanning one question across rows of another source);
  Coral can't use it — it calls the single-query `GET` per row.

You can use these endpoints directly from any tool — Coral is one consumer,
not a dependency.

## Build your own: row-driven search from Python

Coral can't drive `lynx.search` from another table's column (the per-row
correlation — its SQL resolves table-function arguments to constants at plan
time). You can, in a few lines: ask Coral for the rows, then batch them into
Lynx. [`integrations/coral/toolkit.py`](../integrations/coral/toolkit.py) gives
you two thin clients — **bricks, not a framework** — to compose whatever logic
you want:

```python
from toolkit import Lynx, Coral
lynx, coral = Lynx(port=8765), Coral(exe="coral")

rows = coral.sql("SELECT number, title FROM github.pulls WHERE state = 'open'")
hits = lynx.search_batch([r["title"] for r in rows], source="framework", top_k=3)
for row, res in zip(rows, hits):
    print(row["number"], "->", [h["file"] for h in res["hits"]])
```

`python integrations/coral/toolkit.py` runs a credential-free demo of exactly
this pattern (it uses an inline `VALUES` list as the stand-in for live data).

Stdlib only — nothing to `pip install`. These bricks live in the **repo**, not
in the installed `lynx-mcp` wheel: copy `toolkit.py` next to your own script (or
run it from `integrations/coral/`), then `from toolkit import Lynx, Coral`. Lynx
stays local; only the Coral side reaches live APIs.
