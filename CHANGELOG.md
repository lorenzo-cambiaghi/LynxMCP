# Changelog

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
