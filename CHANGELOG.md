# Changelog

## 1.6.0 — 2026-06-16

### Added
- **Objective-C AST chunking and code graph.** `.m` / `.mm` files (and `.h`
  headers that contain `@interface` / `@protocol`) are indexed as
  `@interface` / `@implementation` / `@protocol` methods with qualified symbol
  names, instead of falling back to plain-text windows. The **code knowledge
  graph** covers Objective-C too: `#import` edges, the message-send call graph
  (`[self foo]` resolves caller → callee), and inheritance (superclass
  `extends` + adopted-protocol `implements`). Shared `.h` headers still parse
  as C unless Objective-C markers are present.
- **Four more languages for AST-aware chunking: Bash/Shell, SQL, Scala, Lua.**
  Lynx now parses **18+ languages** with tree-sitter — `.sh`/`.bash`, `.sql`,
  `.scala`/`.sc`, and `.lua` files are indexed as whole functions / DDL objects
  (`CREATE TABLE`/`FUNCTION`/`VIEW`) with qualified symbol names, instead of
  falling back to plain-text windows. Add the extensions to a source's
  `supported_extensions` to index them (the manager UI auto-detects them).
  The optional **code knowledge graph** also covers the new languages:
  **Scala** with full call + inheritance (`extends`/`with`) edges, **Lua** and
  **Bash** with the call graph between defined functions. (SQL is
  search/chunk-only — its DDL has no call/inheritance/import structure.)
- **Self-host Docker image.** A `Dockerfile` to run Lynx on your own server
  (CPU-only torch, ~2.7 GB); the indexed code and embeddings stay on the
  machine — "local" = local to your server, not a hosted service.

### Changed
- README/docs: the Coral section now leads with cross-source use cases
  (correlate code search with live GitHub/Sentry/… data). Added `ROADMAP.md`
  (items under evaluation + explicit non-goals) and a Glama directory listing.

## 1.5.1 — 2026-06-15

### Changed
- **Per-parameter descriptions on every MCP tool.** Each tool's JSON input
  schema now documents its parameters (`source`, `top_k`, `query`, `symbol`,
  filters, etc.) via `Annotated[..., Field(description=...)]`, which the prose
  tool description alone didn't provide. Improves the in-client parameter hints
  and tool-definition quality scoring. Tool behavior is unchanged.

## 1.5.0 — 2026-06-15

### Added
- **Batch search API** — `POST /api/v1/search` accepts
  `{"queries": [...], "source": ..., "top_k": ...}` and embeds every query in a
  single model call, returning ranked hits per query
  (`{"results": [{"query": ..., "hits": [...]}, ...]}`). Results match the
  single-query `GET` endpoint: both now route through one shared retrieval core,
  and the batched query embedding is self-checked against the single-query path
  (so it's correct for any embedding model, not just BGE). For multi-query
  consumers fanning a question across rows of another data source.
- **Coral Python toolkit** (`integrations/coral/toolkit.py`) — two thin,
  stdlib-only clients (`Lynx`, `Coral`) so you can compose row-driven search and
  joins in a few lines of Python. Coral's SQL can't drive `lynx.search` from
  another table's column (table-function args resolve to constants at plan
  time); these bricks let you do it: query Coral for the rows, batch them into
  Lynx. Ships with a runnable, credential-free demo. See `docs/CORAL.md`.

## 1.4.1 — 2026-06-15

### Fixed
- **Coral integration docs.** The README examples used
  `lynx.search(q => other_table.column)`, but Coral resolves table-function
  arguments to constants at plan time, so row-driven (per-row) search isn't
  possible — `lynx.search`'s query must be a literal. Rewrote the README around
  the verified-valid pattern: a literal search query exposed as a SQL table you
  filter, sort, and join with live data. (No code change; the `/api/v1` API and
  the `manifest.yaml` were already correct.)

## 1.4.0 — 2026-06-14

### Added
- **Stable local JSON API** (`GET /api/v1/search`, `GET /api/v1/sources`)
  served by `lynx manager ui` — versioned, additive-only, 127.0.0.1 only.
  The integration surface for external tools.
- **Coral source spec** (`integrations/coral/manifest.yaml`): use Lynx from
  [Coral](https://github.com/withcoral/coral) as a SQL schema — join semantic
  code search with live GitHub / Sentry / Linear data in one query, e.g.
  `SELECT file, symbol, score FROM lynx.search(q => '...') LIMIT 5`. Recipe and
  worked examples in `docs/CORAL.md` and the README.

This release also carries everything from 1.3.1 below (antivirus startup-crash
fix, corrupt-index detection + `lynx reset`, UI fixes).

## 1.3.1 — 2026-06-14

### Fixed
- **Windows startup crash with HTTPS-inspecting antivirus.** Avast/AVG (and
  similar) inject `SSLKEYLOGFILE` pointing at a device path into every process;
  Python's `ssl` opens it through OpenSSL's file BIO on first TLS use, which
  aborts the bundled interpreter (`no OPENSSL_Applink`) mid-startup with no
  traceback. Lynx now strips `SSLKEYLOGFILE` before any TLS use — it runs
  offline and never needs TLS key logging. Opt out with
  `LYNX_KEEP_SSLKEYLOGFILE=1`.
- **Long config paths no longer overflow the sidebar.** The footer shows the
  config filename, with the full path on hover.

### Added
- **Corrupt-index detection, recovery, and `lynx reset`.** A
  version-incompatible or truncated ChromaDB index used to break the dashboard
  silently (blank page) or crash the process. Lynx now probes each index in a
  subprocess before opening it, so even a segfaulting store can't take down the
  UI/MCP server; the dashboard shows the source as **corrupt** (other sources
  keep working) with a **Reset** button, and the CLI gains
  `lynx reset --source <name>` to wipe and rebuild from scratch. The index is
  disposable derived data — a reset rebuilds it from your files.

## 1.3.0 — 2026-06-12

### Added
- **Usage playbook in the MCP handshake.** The server now sends compact
  `instructions` at initialize (source catalog, how to phrase queries, score
  interpretation, escalation ladder) and exposes the full playbook as the
  `lynx://guide` resource. Every MCP client gets this automatically — no
  rules file installation required (rules files remain available for
  stronger steering).
- **Tool annotations.** All tools now carry MCP annotations
  (`readOnlyHint` / `destructiveHint` / `idempotentHint` / `openWorldHint`),
  so clients can auto-approve the read-only retrieval tools.
- **`feedback` tool.** When the agent can't find what it needs, it can file
  a structured report (what it tried, where it got stuck) BEFORE giving up.
  Reports are appended to `rag_storage/_feedback/feedback.jsonl` — 100%
  local, never uploaded — and give the index owner a concrete signal for
  tuning sources, filters, and chunking.

## 1.2.0 — 2026-06-12

### Added
- **JS rendering for webdoc sources (opt-in).** Set `render_js: true` on a
  webdoc source to crawl SPA / client-side-rendered docs sites through
  headless Chromium (Playwright): pages are extracted *after* client-side
  rendering, and link discovery sees the post-JS DOM. Install the browser
  dependency with `lynx manager install webdoc-js`. New per-source tunables:
  `render_wait_until` (`load`/`domcontentloaded`/`networkidle`) and
  `render_timeout_seconds`. The default install stays browser-free.
- `lynx manager install` now supports extras with post-install steps
  (used by `webdoc-js` to download the Chromium binary).

## 1.1.2 — 2026-06-12

### Added
- `server.json` + `mcp-name` ownership marker in the README for listing on
  the official MCP Registry (registry.modelcontextprotocol.io).

## 1.1.1 — 2026-06-12

### Changed
- Depend on `llama-index-core` instead of the `llama-index` meta-package:
  drops the unused OpenAI bundle (`openai`, `llama-index-llms-openai`,
  `llama-index-embeddings-openai`) from the install. Lynx never calls any
  remote LLM — the install log now reflects that.
- Install instructions point at the published PyPI package
  (`pipx install lynx-mcp`); native installers bootstrap from PyPI too.

## 1.1.0 — 2026-06-12

### Breaking
- **Fixed MCP tool surface.** Tools no longer multiply per source
  (`search_<name>`, `get_callers_<name>`, ...). The server now exposes a
  constant set of ~11 tools that take a `source` argument:
  `search`, `deep_search`, `graph_query`, `find_definition`, `find_usages`,
  `find_tests_for`, `find_similar`, `search_diff`, `list_sources`,
  `get_rag_status`, `update_source_index`. With three sources this drops
  the registered tool count from 50+ to 11 and keeps client context lean.
  `search_all_sources` / `deep_search_all_sources` are folded into
  `search` / `deep_search` (omit `source` to fan out). The ten graph tools
  are folded into `graph_query(operation=...)`.
  If you generated an AI rules file with an older version, regenerate it
  from the manager UI.
- **Distribution renamed to `lynx-mcp`** (the bare `lynx` name on PyPI
  belongs to an unrelated project). The import package and the `lynx` CLI
  are unchanged.

### Fixed
- First run on a clean machine no longer hangs: HuggingFace offline mode
  is now enabled only when the configured models are already in the local
  cache, instead of being hard-coded on (which blocked the initial model
  download unless you knew about `lynx manager install --model`).
- `lynx manager install <extra>` no longer runs `pip install lynx[...]`,
  which resolved against the unrelated `lynx` package on PyPI. It now
  installs the extra's actual requirements.

### Changed
- README rewritten as a concise overview; the full manual moved to
  `docs/GUIDE.md`.
- CI now runs the pytest suite on Linux + Windows across Python 3.10/3.13
  (previously only installers were built).
