# Lynx

**A 100% local MCP server for semantic code search — AST-aware chunking, hybrid BM25 + dense retrieval, and an optional code knowledge graph. Works with any MCP client (Claude Code, Cursor, Windsurf, Antigravity, ...).**

[![Tests](https://github.com/lorenzo-cambiaghi/LynxMCP/actions/workflows/test.yml/badge.svg)](https://github.com/lorenzo-cambiaghi/LynxMCP/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Glama score](https://glama.ai/mcp/servers/lorenzo-cambiaghi/LynxMCP/badges/score.svg)](https://glama.ai/mcp/servers/lorenzo-cambiaghi/LynxMCP)

Your AI assistant greps file names and guesses. Lynx gives it real retrieval over **your code, your library docs, and your PDFs** — without a single byte leaving your machine.

- **AST-aware indexing** — tree-sitter parses 18+ languages and indexes whole functions/classes, not arbitrary text windows.
- **Hybrid retrieval** — dense embeddings + code-tokenized BM25, fused with RRF; optional cross-encoder reranker.
- **Code knowledge graph (opt-in)** — who-calls-what, inheritance, imports: ask "what breaks if I change this?" and get the actual blast radius.
- **Joinable as SQL** — search and the graph are also served as rows over a local HTTP API, so you can correlate your code with tickets, PRs, or logs in [DuckDB](docs/DUCKDB.md) or [Coral](docs/CORAL.md) — no data leaves your machine.
- **Multi-source** — index codebases, public docs sites (fetched once, on demand; JS-rendered SPAs supported via optional headless Chromium), and PDFs side by side.
- **Live index** — a file watcher re-indexes saves in ~2s. No manual rebuild ritual.
- **Web manager UI** — `lynx manager ui` gives you guided setup, a query playground, diagnostics, and client config snippets.

*(Named after Lynceus, the Argonaut whose sharp eyes could find anything hidden.)*

## Quickstart

```bash
# 1. Install the CLI (isolated, no venv ritual)
pipx install lynx-mcp
#    or: uv tool install lynx-mcp

# 2. Create a config pointing at your project
lynx manager init

# 3. Build the index (downloads the ~130MB embedding model on first run)
lynx build
```

Then register Lynx in your MCP client (Claude Code shown; see the [full guide](docs/GUIDE.md) for Cursor, Antigravity, and generic stdio clients — or let `lynx manager ui` generate the snippet for you):

```json
{
  "mcpServers": {
    "lynx": {
      "command": "lynx",
      "args": ["serve", "--config", "/absolute/path/to/config.json"]
    }
  }
}
```

Prefer zero terminal? There are [double-click installers](https://github.com/lorenzo-cambiaghi/LynxMCP/releases) for macOS and Windows.

## The tools your AI gets

The tool set is **fixed** — it does not grow with the number of sources, so your client's tool list (and context window) stays small. Tools take a `source` argument where relevant.

| Tool | What it answers |
|------|-----------------|
| `search(query, source?)` | Primary hybrid search. Omit `source` to search every source at once (RRF-fused). |
| `deep_search(queries, source?)` | Escalation: tries multiple query phrasings until one passes a quality threshold. |
| `graph_query(operation, symbol?)` | `callers`, `callees`, `subclasses`, `superclasses`, `imports`, `neighbors`, `shortest_path`, `overview`, `surprising_connections`, `status`. |
| `find_definition(symbol)` | Where is X defined? (AST-precise when the graph is on, BM25 fallback otherwise.) |
| `find_usages(symbol)` | Every use of X — calls *and* non-call references (generics, decorators, docs). |
| `find_tests_for(symbol)` | Are there tests for X? |
| `find_similar(snippet)` | Does code like this already exist? |
| `search_diff(query, base?)` | Search only the files changed vs a base branch — built for code review. |
| `feedback(trying_to_do, tried, stuck)` | The agent files a report when the index couldn't answer — stored 100% locally, your signal for tuning sources. |
| `list_sources` / `get_rag_status` / `update_source_index` | Introspection and maintenance. |

All retrieval tools carry MCP `readOnlyHint` annotations (clients can auto-approve them), and the server ships its usage playbook in the MCP handshake (`instructions` + a `lynx://guide` resource) — your agent knows how to query well without any rules-file setup.

## How it works

```
your code ──► tree-sitter AST chunker ──► bge-small embeddings ──► ChromaDB
                                     └──► code-tokenized BM25 ─┐
query ───────────────────────────────────► RRF fusion ◄────────┘ ──► (optional reranker) ──► results
```

Everything runs locally: HuggingFace models are downloaded once, then Lynx switches to offline mode. No telemetry, no cloud index, no code upload. The only network access is the model download and the *explicit* `webdoc` fetch step you trigger yourself.

## Why not just let the agent grep?

Grep is great when you know the identifier. It fails when you (or the agent) know the *behavior*: "where do we clamp the camera zoom?" matches nothing literal. Agentic grep also burns tokens — every wrong file the agent opens is context spent. Lynx answers behavioral queries in one tool call with file + line + symbol citations, and the graph layer answers structural questions (callers, inheritance) that grep fundamentally cannot — polymorphic dispatch leaves no textual trace.

Honest counterpoint: on a small repo that fits in the agent's context, built-in tools are fine. Lynx pays off on large codebases, on framework docs your model's training data has gone stale on, and on repeated sessions where re-exploring from scratch is waste.

## Benchmarks (reproducible)

<img src="https://raw.githubusercontent.com/lorenzo-cambiaghi/LynxMCP/main/benchmarks/chart.svg" alt="Lynx vs agentic grep: -58% tokens to answer; 4 vs 101 tool calls to map a class hierarchy" width="980">

On the `django/` package of Django 5.2 (883 files, ~158k lines), 20 behavioral questions with known ground-truth files — full methodology, per-task results, and an intentionally *strong* grep baseline in [benchmarks/RESULTS.md](https://github.com/lorenzo-cambiaghi/LynxMCP/blob/main/benchmarks/RESULTS.md):

| | Agentic grep | Lynx |
|---|---|---|
| median tokens **to answer** (tool output + required follow-up read) | 4,150 | **1,725** |
| tool round-trips before the code is in context | 2+ | **1** (chunks included, with symbol + file:line + score) |
| hit@1 / MRR | 45% / 0.64 | 55% / 0.67 |
| *"what inherits from `Field`?"* — full descendant tree (100 classes) | **101 grep rounds** | **4 graph calls**, same recall, file:line per edge |

The ranking quality is comparable (Django's docstring-rich code is grep's best case — we say so in the report). The structural difference is not: every tool round-trip is a full model inference over the growing context, and class-relation questions force grep into one round per discovered class while `graph_query` reads resolved inheritance edges.

```bash
# reproduce
git clone --depth 1 --branch 5.2 https://github.com/django/django.git benchmarks/_target/django
python benchmarks/run_benchmark.py && python benchmarks/structural_demo.py
```

## Lynx + Coral: your code, joined with everything else

[Coral](https://github.com/withcoral/coral) turns your live tools — GitHub, Sentry, Jira, Linear — into one local SQL interface. Plug in Lynx ([source spec](integrations/coral/manifest.yaml)) and **your codebase becomes a queryable source too**: ask in plain language, get ranked code locations back, and **correlate them with the tools your team already lives in** — without a byte leaving your machine.

What that unlocks:

- 🔎 **Find logic by behavior, not keywords.** *"Where do we validate session tokens?"* returns the actual functions — file, symbol, line, score — even when nothing matches literally.
- 🔁 **Refactor without surprises.** Locate the code behind a feature and line it up against the repo's open PRs in one query — see who's already in there before you touch it.
- 🚨 **Triage crashes to code.** Take the behavior from a Sentry alert and get the ranked code locations; when your source exposes a file column, correlate them with the live issues.
- 🎫 **Turn a backlog into a map.** Pull your open tickets from Coral and — with the included Python helper — batch-search Lynx to surface the likely code area for each.
- 🔒 **100% local.** Repo and embeddings never leave your machine; only the live-data side hits an API.

Once the idea clicks, the syntax is just SQL:

```sql
-- ranked code for a behavioral question (C# only)
SELECT file, symbol, score
FROM lynx.search(q => 'where the camera zoom is clamped')
WHERE language = 'c_sharp'
ORDER BY score DESC
LIMIT 5;
```
```sql
-- top code matches for a question, next to the repo's open PRs
SELECT s.file, s.symbol, s.score, p.html_url
FROM lynx.search(q => 'retry logic for payment webhooks') s
CROSS JOIN github.pulls p
WHERE p.owner = 'your-org' AND p.repo = 'your-repo' AND p.state = 'open'
ORDER BY s.score DESC;
```

*The search string is a literal you pass (Coral resolves table-function arguments at plan time) — so it's code search as a **joinable** source, not a per-row enrichment. For one search per row of another table, use the batch endpoint + the Python helper. `lynx.sources` lists your indexed sources; `lynx.search(q => '…')` is the ranked search function (`source => '…'`, `top_k => N` to narrow it). Full setup in **[docs/CORAL.md](docs/CORAL.md)**.*

## Lynx + DuckDB: code search as a local SQL table

Lynx serves its search **and** its code graph as NDJSON over a local HTTP API, and [DuckDB](https://duckdb.org) reads that URL straight into a table. So you can JOIN your code with **anything DuckDB reads** — Parquet, CSV, SQLite, Postgres, a git log, a JSON log — in one engine, on your machine, with no plugin and no service to run.

- 🦆 **Zero setup.** `read_ndjson_auto('http://127.0.0.1:8765/api/v1/search?…')` is a table. No connector, no daemon.
- 🔗 **Join with any local data.** Cross code relevance with git churn, error logs, ownership, ticket exports — whatever you can read.
- 🧪 **Total flexibility.** Shape and filter in SQL, then hand a tiny, hyper-targeted context to an LLM or a notebook.

```sql
-- code search as a table
SELECT file, symbol, score
FROM read_ndjson_auto(
  'http://127.0.0.1:8765/api/v1/search?q=where%20we%20validate%20session%20tokens&format=ndjson')
ORDER BY score DESC;
```
```sql
-- regression hunting: code related to login that is ALSO churning in git
WITH churn AS (
  SELECT path, count(*) AS commits, max(date) AS last_modified
  FROM read_csv('churn.csv', header = false,
                columns = {'path': 'VARCHAR', 'date': 'DATE'})   -- from a one-line git log
  GROUP BY path
)
SELECT c.path, c.commits, h.symbol, h.score
FROM read_ndjson_auto(
       'http://127.0.0.1:8765/api/v1/search?q=user%20login%20and%20token%20validation&format=ndjson') h
JOIN churn c ON h.file = regexp_replace(c.path, '.*/', '')
WHERE c.commits >= 2
ORDER BY c.last_modified DESC, h.score DESC;
```

The code graph is one URL away too (`…/api/v1/graph?operation=callers&symbol=…`), so you can pivot a hit to its blast radius and join *that* with your data. Recipes for git freshness, error-log triage, and per-row batch search in **[docs/DUCKDB.md](docs/DUCKDB.md)**.

## Documentation

| | |
|---|---|
| [Full guide](docs/GUIDE.md) | Configuration, all source types (codebase / webdoc / PDF), retrieval internals, troubleshooting |
| [Manager UI](docs/GUIDE.md#lynxmanager--guided-setup-web-ui-diagnostics-new-in-v09) | Guided setup, playground, diagnostics |
| [Use Lynx from Coral](docs/CORAL.md) | SQL over your code search: `SELECT ... FROM lynx.search` joined with live GitHub/Sentry data |
| [Use Lynx from DuckDB](docs/DUCKDB.md) | `read_ndjson_auto('…/api/v1/search?format=ndjson')` — join code search + the code graph with any local data |
| [MCP recipes](docs/MCP_RECIPES.md) | Agent patterns combining Lynx with GitHub/Sentry/Jira MCP servers (triage, PR impact, ticket→code) |
| [PR impact analysis (GitHub Action)](integrations/github-action/) | On every PR, comment with the downstream callers + semantically related code, indexed locally on the runner |
| [Steampipe plugin (design spec)](integrations/steampipe/DESIGN.md) | Spec for a SQL plugin exposing `lynx_source`/`lynx_search`/`lynx_graph`, joinable with Steampipe's connectors — implementation TBD |
| [config.example.json](config.example.json) | Annotated example configuration |

## Status

Actively developed by one author; APIs may still move before 1.x stabilizes. Issues and PRs welcome — the test suite runs with `pytest` and CI must stay green. See [ROADMAP.md](ROADMAP.md) for what's under consideration (and what's explicitly *not* planned).

## License

[Apache 2.0](LICENSE)

<!-- MCP Registry ownership marker — must stay in the README published on
     PyPI so registry.modelcontextprotocol.io can verify the package.
mcp-name: io.github.lorenzo-cambiaghi/lynx
-->

