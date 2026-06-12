# Using Lynx from Coral (SQL over your code search)

[Coral](https://github.com/withcoral/coral) gives agents a local SQL runtime
over live data sources (GitHub, Sentry, Datadog, Linear, ...). Lynx covers
the part Coral doesn't: **semantic retrieval over unstructured local content**
— your code, your library docs, your PDFs.

With the source spec in [`integrations/coral/manifest.yaml`](../integrations/coral/manifest.yaml),
Lynx becomes a Coral schema, and your agent can JOIN code search with live
data in one SQL statement:

```sql
-- "We have a Sentry error about webhook retries — where does that live,
--  and is there an open PR touching it?"
SELECT s.file, s.symbol, s.score, p.html_url, p.state
FROM lynx.search s
CROSS JOIN github.pulls p
WHERE s.query = 'retry logic for payment webhooks'
  AND p.owner = 'your-org' AND p.repo = 'your-repo' AND p.state = 'open'
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

2. **Install the source in Coral** following Coral's
   [custom source guide](https://withcoral.com/docs/guides/write-a-custom-source),
   pointing it at `integrations/coral/manifest.yaml` from this repo.
   The only input is `LYNX_PORT` (default `8765`).

3. **Query it**:

   ```sql
   SELECT name, type, chunk_count FROM lynx.sources;
   SELECT file, symbol, score FROM lynx.search
   WHERE query = 'where passwords are hashed' LIMIT 5;
   ```

## The API behind it

The spec talks to Lynx's stable local JSON API (additive-only within v1):

- `GET /api/v1/search?q=...&source=...&top_k=...`
- `GET /api/v1/sources`

You can use these endpoints directly from any tool — Coral is one consumer,
not a dependency.
