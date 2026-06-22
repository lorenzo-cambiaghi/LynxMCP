# Changelog

## 1.7.3 — 2026-06-22

Stabilization & bug-fixing release.

### Fixed
- **`ignored_path_fragments` now excludes files from the vector index too.**
  Previously the file watcher and the graph layer honored the configured ignore
  fragments (`node_modules`, `dist`, `build`, …) but the vector-index build did
  not — so vendored / build dirs were embedded into ChromaDB and BM25 and leaked
  into search results, while the watcher never refreshed them and the graph
  excluded them (the three layers disagreed). The index build now applies the
  same exclusion. The SHA partitioner self-heals on the next build: files that
  became ignored fall out of the candidate set and their chunks are dropped.
  Changing `ignored_path_fragments` is now reported as a WARNING-level config
  drift.
- **Multi-segment ignore fragments now match on Windows.** Fragment matching is
  separator-agnostic (both path and fragment normalized to `/`), so a fragment
  like `src/generated` is excluded regardless of OS path separators.

### Changed
- **Single source of truth for the codebase file walk** (`lynx.fs_scan`). The
  vector index (`rag_manager`) and the graph layer (`graph.builder`) shared two
  near-identical copies of the directory walk + extension/ignore filtering; they
  now both delegate to one module, which is what fixes the divergence above.
  Ignored directories are pruned during the walk (faster on big trees).

### Internal
- Removed an unreachable no-op branch in the cross-encoder reranker and refreshed
  its docstring.
- Added `tests/test_fs_scan.py` (pure, model-free) covering extension/hidden
  filtering, ignore-fragment exclusion, and separator-agnostic matching.

## 1.7.2 — 2026-06-22

### Added
- **Repository-comprehension tools** for AI agents working in an unfamiliar
  codebase:
  - **`repo_overview`** — the "what is this and where do I start" map: detected
    languages, frameworks, manifest files, likely entry points (main/CLI/server),
    and suggested build/test/run commands. Pure filesystem scan, no graph needed.
  - **`describe_symbol`** — DEFINITION + CALLED BY + CALLS + TESTS for a symbol
    in one call, instead of running find_definition + graph callers/callees +
    find_tests_for separately.
  - **`impact`** — the blast radius of a change: everything that reaches a symbol
    transitively through the call graph (with hop distance) + the tests to re-run.
  - **`module_summary`** (graph) — a file's public symbols + what it imports +
    which files depend on it.
- **`export_graph`** — render a slice of the code graph as a single, offline,
  self-contained file (inline SVG, no server, no CDN): a symbol's blast radius
  (`mode=symbol`) or a file as a hub (`mode=module`). MCP tool + CLI
  (`lynx graph export --symbol … | --module …`). Output goes to the new optional
  `reports_path` config key (default `<storage>/reports`).

### Changed
- **Retrieval deduplicates substantial duplicate chunks** — vendored or
  `build/`/`dist/` copies of the same file (identical bodies ≥120 chars) are
  collapsed to the highest-ranked hit. Short identical boilerplate is never
  merged, so distinct symbols that share a trivial body are all kept.
- **New codebase sources added via the UI get default ignore fragments**
  (`.git`, `.venv`, `node_modules`, `dist`, `build`, `target`, …) so build and
  dependency dirs don't pollute the index.

### Fixed
- **Graph: spurious self-loop call edges removed.** A member call `obj.Foo()`
  inside a method also named `Foo`, and a call between overloads (which collapse
  to one node id), both produced a bogus `Foo calls Foo` edge. Self-loop call
  edges are now dropped at both the extractor and the cross-file resolver.
- **Manager UI folder picker — Windows drive paths hardened end-to-end.** The
  backend now re-normalizes every drive shape (`C:`, `C:Users`, `/C:`, `//C:`,
  `\\C:`) to a real absolute path before resolving, so a corrupted breadcrumb
  segment can no longer send a dead path.

## 1.7.1 — 2026-06-21

### Added
- **Token savings, in money.** `benchmarks/savings_calculator.py` translates the
  measured per-query token saving into the monthly/yearly API bill it removes,
  for **all three** benchmarked codebases (Django/Python −58%, Json.NET/C# −77%,
  Guava/Java −86%) × every flagship model, at 2026-06 input prices —
  **Claude Fable 5** ($10/1M),
  **Claude Opus 4.8** ($5/1M), **GPT-5.5** ($5/1M), and more. Reports two figures
  per codebase: a fully-measured *floor* (tool-output delta only, zero
  assumptions) and a *realistic* figure that adds one eliminated grep round-trip
  re-billing a configurable context window. Measured deltas live in
  `benchmarks/measured.json`. Parametric by team size and usage (`--devs`,
  `--queries-per-dev-day`, `--avg-context-tokens`); emits a deterministic SVG
  chart (`docs/img/cost_savings.svg`, grouped by model and codebase, no matplotlib).
- **Editable price config + interactive calculator.** Model prices now live in
  `benchmarks/pricing.json` (Claude Fable 5 / Opus 4.8 / Sonnet 4.6 / Haiku 4.5,
  GPT-5.5 / GPT-5.4) — edit a price or add a model and both the CLI and the new
  `benchmarks/savings_calculator.html` pick it up. The HTML page lets a user
  choose the **codebase and model** from drop-downs, **override the $/1M price
  live**, and set their own team size/usage with no file edits; it works offline
  (embedded fallback) and reads `pricing.json` + `measured.json` live when served.
  CLI gains `--model` and `--input-price` to mirror the same per-model override.

- **Multi-language benchmarks — C# and Java.** `run_benchmark.py` is now
  language-parametric (target `ext` / `name` / `ref` in the tasks file; separate
  `--results-json` / `--results-md` so each language writes its own artifacts).
  Added `benchmarks/tasks_jsonnet.json` (15 questions over `JamesNK/Newtonsoft.Json`)
  and `benchmarks/tasks_guava.json` (15 questions over `google/guava`), both with
  ground-truth verified against the source. Results:
  - **C# / Json.NET** (`RESULTS_csharp.md`): sparser-comment code — Lynx wins
    **every** metric (hit@1 47% vs 33%, MRR 0.58 vs 0.47, **−77%** tokens to answer).
  - **Java / Guava** (`RESULTS_java.md`): Guava's self-documenting class names are
    grep's *best* case, so grep out-ranks (hit@1 73% vs 60%) — yet tokens to answer
    still collapse **−86%** (5,892 → 807). The honest cross-language takeaway:
    ranking parity swings with how self-documenting the code is, but the **token
    cost drops 58–86% every time** (1 call vs match-lines + a follow-up read).

### Changed
- **README is now outcome-first.** New commercial hero leads with what Lynx
  *saves* — the −58% / 2.4× token deltas and the dollar figures they translate to
  at frontier API prices — before the feature list. The "How it works" ASCII
  pipeline is now a rendered **Mermaid** architecture diagram, and the benchmark
  section now carries the C# (Json.NET) result next to Django.

### Fixed
- **Manager UI folder picker corrupted Windows paths.** The breadcrumb rebuilt a
  Windows drive segment as `\C:` / `/C:` (and rendered the root as `//C:`), so
  clicking the drive — or a child under it — sent a non-existent path to the
  backend (e.g. `path does not exist: C:\build`). The breadcrumb now treats a
  leading drive letter as the drive root (`C:\`), accepts either separator, and
  emits real resolvable absolute paths. POSIX paths are unaffected.

## 1.7.0 — 2026-06-19

### Added
- **Outline search — signatures instead of bodies, for cheap triage.** Both the
  MCP `search(query, outline=true)` tool and the HTTP
  `GET /api/v1/search?view=outline` endpoint return each hit's `signature` +
  first `doc` line instead of its body, so an agent triages candidates and reads
  the full code of only the one it picks (via `find_definition` or the cited
  `file:line`). The tool/param descriptions, the MCP handshake instructions, and
  the `lynx://guide` all tell the agent **when** to use it. Derivation is shared
  (`lynx/outline.py`) so both surfaces produce identical signatures. Measured
  **~2.4× fewer tokens** for the search step on a public repo (psf/requests:
  ≈59% triage; ≈53% even after reading the one chosen body) — see
  `docs/OUTLINE.md` for the data + chart, `benchmarks/outline_tokens.py` to
  reproduce. Query-time transform (no reindex, no `CHUNKER_VERSION` bump, no LLM);
  composes with `format=ndjson`; the default search is unchanged, so Coral /
  DuckDB are unaffected.

## 1.6.0 — 2026-06-19

### Added
- **Objective-C AST chunking and code graph.** `.m` / `.mm` files (and `.h`
  headers that contain `@interface` / `@protocol`) are indexed as
  `@interface` / `@implementation` / `@protocol` methods with qualified symbol
  names, instead of falling back to plain-text windows. The **code knowledge
  graph** covers Objective-C too: `#import` edges, the message-send call graph
  (`[self foo]` resolves caller → callee), and inheritance (superclass
  `extends` + adopted-protocol `implements`). Shared `.h` headers still parse
  as C unless Objective-C markers are present. Adding a language is additive, so
  `CHUNKER_VERSION` is unchanged and existing indexes are **not** auto-flagged —
  to pick up Objective-C on an already-indexed source, run
  `lynx build --source <name>` once to re-chunk its `.m`/`.h` files.
- **`GET /api/v1/graph` endpoint.** Exposes the code knowledge graph (`callers`,
  `callees`, `subclasses`, `superclasses`, `imports`, `neighbors`) as flat JSON
  rows over the stable v1 API, so an external consumer (e.g. a Coral graph
  source) can pivot from a `lynx.search` hit's symbol to its structural blast
  radius and JOIN it with live data. Additive to v1; no reindex.
- **`format=ndjson` on the v1 GET endpoints** (`search`, `sources`, `graph`).
  Returns one JSON row per line so the results drop straight into DuckDB
  (`read_json_auto(..., format='newline_delimited')`), `jq` and pandas without
  unwrapping. Opt-in; the default wrapped JSON is unchanged. See `docs/DUCKDB.md`
  for join recipes.
- **PR impact-analysis GitHub Action** (`integrations/github-action/`). On every
  pull request it indexes the repo on the runner (model + index cached) and
  comments with the cross-file callers of the changed symbols (blast radius) and
  the semantically related code, via the `/api/v1` endpoints. Runs 100% on the
  runner; a self-hosted Lynx is the lighter option for big monorepos.
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
