# local-codebase-rag-mcp

A self-hosted **MCP (Model Context Protocol) server** that gives any AI coding
assistant — Claude Code, Antigravity, Cursor, Continue.dev, Aider, etc. —
the ability to perform **semantic search over your codebase**.

Everything runs **100% locally**: no code, no embeddings, no queries ever
leave your machine. No API keys, no cloud dependencies, no recurring costs.

> **License:** [MIT](LICENSE)

---

## Table of contents

1. [What this does](#what-this-does)
2. [Why it exists](#why-it-exists)
3. [How it works (in 30 seconds)](#how-it-works-in-30-seconds)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Build the index for the first time](#build-the-index-for-the-first-time)
8. [Connect it to your AI client](#connect-it-to-your-ai-client)
   - [Claude Code (CLI)](#claude-code-cli)
   - [Claude Code extension for VS Code](#claude-code-extension-for-vs-code)
   - [Google Antigravity](#google-antigravity)
   - [Cursor](#cursor)
   - [Continue.dev / Aider / other MCP-compliant clients](#continuedev--aider--other-mcp-compliant-clients)
9. [Command-line interface](#command-line-interface)
10. [Verify the integration works](#verify-the-integration-works)
11. [The MCP tools you get](#the-mcp-tools-you-get)
12. [Keeping the index up to date](#keeping-the-index-up-to-date)
13. [Hybrid retrieval](#hybrid-retrieval)
14. [Config drift detection](#config-drift-detection)
15. [Architecture](#architecture)
16. [Troubleshooting](#troubleshooting)
17. [Contributing](#contributing)

---

## What this does

When an AI assistant has to answer questions about a codebase larger than its
context window, it has two options: read files almost at random hoping to find
what it needs, or ask you. Neither is great.

This server fixes that. It indexes your codebase into a local vector database
and exposes three MCP tools the assistant can call:

| Tool | What it does |
|---|---|
| `search_codebase(query, top_k)` | Semantic search. Returns the chunks most relevant to a natural-language query. |
| `update_codebase_index(force)` | Force a full rebuild of the index. |
| `get_rag_status()` | Report current state of the index. |

When you ask the AI *"how is damage handled in this codebase?"*, it calls
`search_codebase("damage handling")` and gets back the top-5 most relevant
code snippets — across files, regardless of naming conventions — and then
answers your question with that context.

## Why it exists

Modern AI IDEs (Cursor, Copilot, etc.) ship their own indexing, but with
trade-offs that are awkward for many teams:

- **They upload your code to vendor servers** to compute embeddings.
  For codebases under NDA, in regulated industries, or with sensitive IP,
  this is a hard blocker.
- **They are vendor-specific.** Switch IDE and you lose the index. Use
  multiple IDEs and you maintain multiple indexes.
- **Their retrieval is implicit.** The IDE decides when to retrieve and what.
  You can't deliberately ask "before writing this, search for anything
  similar that already exists."

This project addresses all three:

- **Zero data egress.** Embeddings are computed on your CPU using an open
  model. ChromaDB stores vectors on local disk. Nothing leaves the host.
- **Vendor-neutral.** Built on the open MCP protocol — works with any
  MCP-compliant client.
- **Explicit retrieval.** `search_codebase` is a tool the assistant (or you,
  through it) calls deliberately when needed.

## How it works (in 30 seconds)

```
Your code  --> chunked --> embedded (BGE-small, on CPU)  --> ChromaDB (local file)
                                                                   |
Your question  --> embedded --> top-K cosine similarity  ----------+
                                                                   |
                                                                   v
                                       Top-K relevant code snippets
                                       returned to the AI client via MCP
```

A file watcher keeps the index in sync as you edit (~2s after each save).

---

## Prerequisites

- **Python 3.10+** (3.12 or 3.13 recommended)
- **pip**
- ~500 MB of disk space for the embedding model (downloaded once on first run)
- An **MCP-compliant AI client**: Claude Code CLI, Claude Code extension for
  VS Code, Antigravity, Cursor, Continue.dev, Aider, etc.

> **First-run note:** the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB)
> is downloaded once from HuggingFace on the very first run. After that,
> offline mode is enforced — no network calls are ever made.

---

## Installation

The project is a standard Python package. Install it from a local clone in
editable mode (recommended while it is pre-PyPI):

```bash
git clone https://github.com/<your-username>/local-codebase-rag-mcp.git
cd local-codebase-rag-mcp
pip install -e .
```

This installs the package and exposes a `local-codebase-rag-mcp` console
command. (When PyPI publication lands, the install becomes a single
`pip install local-codebase-rag-mcp`.)

> On Python 3.14+ you may also need: `pip install "mcp[cli]"`

Dependencies installed automatically:

| Package | Why |
|---|---|
| `mcp` | The MCP SDK (FastMCP server framework) |
| `llama-index` | Document loading, chunking, retrieval pipeline |
| `llama-index-vector-stores-chroma` | Adapter for ChromaDB |
| `llama-index-embeddings-huggingface` | Local HuggingFace embeddings |
| `chromadb` | Persistent vector database (sqlite-backed) |
| `gitpython` | Optional: detect new commits for the rebuild fallback |
| `watchdog` | Cross-platform file system events for live updates |
| `rank-bm25` | Lexical (BM25) retrieval used by the hybrid search mode |

**Verify the install:**

```bash
local-codebase-rag-mcp --version
local-codebase-rag-mcp --help
```

> ### ⚠️ Two equivalent ways to invoke (read this once)
>
> Throughout this README every example uses the short form
> `local-codebase-rag-mcp ...`. The fully equivalent long form is
> `python -m local_codebase_rag_mcp ...`. Pick whichever works for you:
>
> | Use the short form... | Use `python -m ...` instead when... |
> |---|---|
> | The `local-codebase-rag-mcp` command is on your `PATH`. | The console script is missing or not on `PATH` (common on Windows: pip prints a warning at install time when `Scripts/` is not in `PATH`). |
> | You want shorter snippets in IDE config files. | You want a guaranteed-working invocation regardless of `PATH`. Recommended for IDE configs that you'll share or commit to dotfiles. |
>
> Side-by-side, all of these are interchangeable:
>
> ```bash
> local-codebase-rag-mcp --help
> python -m local_codebase_rag_mcp --help
>
> local-codebase-rag-mcp build --config /path/to/config.json
> python -m local_codebase_rag_mcp build --config /path/to/config.json
>
> local-codebase-rag-mcp serve --config /path/to/config.json
> python -m local_codebase_rag_mcp serve --config /path/to/config.json
> ```
>
> If you're on Windows and the short form fails with "command not found",
> just prefix with `python -m ` and substitute hyphens with underscores in
> the package name. That's the only difference.

---

## Configuration

Copy the example config and edit it to point at the codebase you want indexed:

```bash
cp config.example.json config.json
```

Open `config.json` and at minimum set `codebase_path`:

```json
{
  "codebase_path": "C:/path/to/your/codebase",
  "storage_path": "./rag_storage",
  "collection_name": "codebase",
  "loading_timeout_seconds": 600,
  "supported_extensions": [".py", ".md", ".js", ".ts"],
  "ignored_path_fragments": ["/.git/", "/node_modules/"],
  "watcher": { "enabled": true, "debounce_seconds": 2.0 },
  "embedding": { "model_name": "BAAI/bge-small-en-v1.5" },
  "git_integration": { "enabled": true },
  "search": { "default_top_k": 5 }
}
```

| Field | What it does |
|---|---|
| `codebase_path` | **Required.** Absolute path of the directory you want indexed. |
| `storage_path` | Where ChromaDB stores the vectors. Relative paths are resolved against the config file's directory. Default `./rag_storage`. |
| `collection_name` | ChromaDB collection name inside the storage. Useful when you want to keep multiple indexes in the same `storage_path`. Default `codebase`. |
| `loading_timeout_seconds` | Max time to wait for the first index build before MCP tool calls give up. Raise it for very large codebases on slow CPUs. Default `600`. |
| `supported_extensions` | File extensions to index. Anything else is ignored. |
| `ignored_path_fragments` | Substrings: any path containing one is skipped by the file watcher (use forward slashes — they are auto-normalized to OS-native). |
| `watcher.enabled` | Set to `false` to disable the live file watcher. |
| `watcher.debounce_seconds` | Time to wait after a save before re-indexing. Coalesces burst-saves. |
| `embedding.model_name` | Any HuggingFace sentence-transformer model. The default is small and CPU-friendly. **Changing this invalidates all existing vectors** — see [Config drift detection](#config-drift-detection). |
| `git_integration.enabled` | When `true`, `get_rag_status` reports the last indexed git commit. Optional. |
| `search.default_top_k` | Default number of chunks `search_codebase` returns when the caller does not pass `top_k`. |
| `search.mode` | `"hybrid"` (default), `"dense"`, or `"sparse"`. See [Hybrid retrieval](#hybrid-retrieval). |
| `search.rrf_k` | Reciprocal Rank Fusion constant (default `60`, the value from the original RRF paper). Only used in hybrid mode. |
| `search.candidate_pool_size` | How many candidates each retriever returns before fusion (default `30`). Larger = slower but more thorough. |

> `config.json` is **gitignored** — your local config is never committed.

**Want a different config location?** Three options, in priority order:

1. Pass `--config /absolute/path/to/config.json` to any CLI subcommand.
2. Set the environment variable `RAG_CONFIG_PATH` to an absolute path.
3. Default: `./config.json` in the current working directory.

For MCP IDE integration, option (1) is the recommended approach — see the
[Connect it to your AI client](#connect-it-to-your-ai-client) snippets.

---

## Build the index for the first time

The first index build can take a few minutes on a large codebase
(it scans every file and computes embeddings on CPU). To avoid making your
AI client wait the first time it connects, pre-build the index from a
terminal:

From the directory containing your `config.json`, run:

```bash
local-codebase-rag-mcp build
```

> If you get `command not found`, use the equivalent `python -m` form:
> `python -m local_codebase_rag_mcp build`. See
> [Two equivalent ways to invoke](#installation) for why.

You should see logs like:

```
[rag] Indexing codebase from C:\path\to\your\codebase...
[rag] Found 1247 files
[rag] Index updated successfully.
```

If your `config.json` lives elsewhere, use `--config`:

```bash
local-codebase-rag-mcp build --config /path/to/config.json
```

The same `build` command also handles **rebuilding** later (e.g. after
changing the embedding model) — it always does a full force rebuild, which
is the right thing for a CLI invocation.

> Note: starting the server with `local-codebase-rag-mcp serve` on a fresh
> install will also trigger an initial build implicitly. Pre-building is just
> a courtesy so the first MCP client doesn't have to wait minutes.

Once `rag_storage/` exists and is populated, every subsequent server start
is fast (a few seconds).

---

## Connect it to your AI client

The server speaks **MCP over stdio** — every modern AI client supports this
the same way: a `command` to launch the server, an array of `args`, and
optionally `env` and `cwd`.

In every example below, replace `C:/path/to/local-codebase-rag-mcp` with the
absolute path where you cloned this repo.

> Tip on Windows: in JSON, you can use either forward slashes (`C:/Users/...`)
> or escaped backslashes (`C:\\Users\\...`). Forward slashes are simpler.

All snippets below assume the package is installed (`pip install -e .` from
the repo, or eventually `pip install local-codebase-rag-mcp`) and that you
have a `config.json` somewhere on disk.

> **Tip — passing the config:** since an MCP client launches the server
> from an unpredictable working directory, always pass `--config` with an
> absolute path. Replace `C:/path/to/config.json` in every snippet with
> your actual path.

> **Tip — if the short command isn't on your `PATH`:** every snippet below
> uses `"command": "local-codebase-rag-mcp"`. If pip installed the console
> script outside your `PATH` (common on Windows — pip prints a warning at
> install time), swap the entry for the equivalent `python -m` form:
>
> ```json
> "command": "python",
> "args": ["-m", "local_codebase_rag_mcp", "serve", "--config", "C:/path/to/config.json"]
> ```
>
> The two are functionally identical. The `python -m` form is also a
> safer default for shared / committed config files, since it does not
> depend on the user's `PATH`.

### Claude Code (CLI)

Add the server with one command:

```bash
claude mcp add codebase-rag --scope user -- local-codebase-rag-mcp serve --config C:/path/to/config.json
```

Or edit `~/.claude.json` (or `%USERPROFILE%\.claude.json` on Windows) directly:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/path/to/config.json"]
    }
  }
}
```

Restart Claude Code. The next session will list `search_codebase`,
`update_codebase_index`, and `get_rag_status` among its available tools.

### Claude Code extension for VS Code

The VS Code extension reads the same `~/.claude.json` configuration as the
CLI — set it up once and it's available everywhere.

If you prefer per-workspace configuration, create a `.mcp.json` in your
workspace root:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/path/to/config.json"]
    }
  }
}
```

Reload VS Code. Open the Claude Code panel and verify the tools appear.

### Google Antigravity

Antigravity reads `mcp_config.json` from its user-data directory:

- Windows: `%USERPROFILE%\.gemini\antigravity\mcp_config.json`
- macOS: `~/.gemini/antigravity/mcp_config.json`
- Linux: `~/.gemini/antigravity/mcp_config.json`

Add an entry under `mcpServers`:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/path/to/config.json"],
      "type": "stdio"
    }
  }
}
```

Restart Antigravity. The new tools appear automatically.

### Cursor

Cursor reads `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/path/to/config.json"]
    }
  }
}
```

Restart Cursor. Settings → MCP should now list the server.

### Continue.dev / Aider / other MCP-compliant clients

Any MCP client takes the same three pieces of information:

- **Command:** `local-codebase-rag-mcp`
- **Args:** `["serve", "--config", "C:/path/to/config.json"]`
- **Transport:** `stdio`

Consult your client's documentation for where to put it.

### Multiple codebases (one server per repo)

The simplest way to index N codebases is N config files + N MCP server
entries:

```json
{
  "mcpServers": {
    "codebase-rag-frontend": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/configs/frontend.json"]
    },
    "codebase-rag-backend": {
      "command": "local-codebase-rag-mcp",
      "args": ["serve", "--config", "C:/configs/backend.json"]
    }
  }
}
```

Each appears as a separate MCP server in the IDE with its own three tools.
A single `pip install` covers all of them.

---

## Command-line interface

The same `local-codebase-rag-mcp` command exposes four subcommands. Useful
for debugging the index, scripting, or just querying the codebase without
opening an AI assistant.

```text
local-codebase-rag-mcp [--version] [-h] COMMAND ...

  serve   Run the MCP server (default if no command is given)
  build   Force a full rebuild of the index
  search  Run an ad-hoc search query (no MCP client needed)
  status  Show RAG status: git state, last update, config drift
```

> Every example in this section can be invoked equivalently as
> `python -m local_codebase_rag_mcp <subcommand> ...` if the short
> `local-codebase-rag-mcp` script is not on your `PATH`. See
> [Two equivalent ways to invoke](#installation) at the end of the
> Installation section.

Every subcommand accepts `--config PATH` (or honors `RAG_CONFIG_PATH`, or
falls back to `./config.json` in the current directory).

### `serve`

Runs the MCP server over stdio. This is what your IDE invokes:

```bash
local-codebase-rag-mcp serve --config /path/to/config.json
```

### `build`

Forces a full rebuild of the index. Use this for the first-time build,
after changing the embedding model, or whenever you want a known-clean
rebuild:

```bash
local-codebase-rag-mcp build --config /path/to/config.json
```

### `search`

Runs an ad-hoc search and prints the top results to your terminal. Useful
for "is the index actually finding this file?" debugging or for piping
into other commands.

```bash
local-codebase-rag-mcp search "how damage is dispatched" --top-k 3
local-codebase-rag-mcp search "IDamageable" --mode dense --ext .cs
local-codebase-rag-mcp search "auth" --glob "**/middleware/**" -k 5
```

Supported flags: `--top-k / -k`, `--mode {hybrid,dense,sparse}`, `--ext`
(repeatable, e.g. `--ext .py --ext .pyi`), `--glob`, `--path-contains`.

### `status`

Reports git state, last update time, and any config drift detected since
the last full rebuild:

```bash
$ local-codebase-rag-mcp status
Status:       Up to date
Last commit:  0edb21f9e07c5d64ba0e632606ae4ccb64b5c16a
Last update:  2026-05-10T13:45:27.492806

No config drift detected.
```

---

## Verify the integration works

After connecting the server to your client, ask the AI assistant:

> *"Use the search_codebase tool to find anything related to authentication
> in this codebase."*

You should see it invoke `search_codebase` and return relevant snippets.

To verify directly without an AI client, run either smoke test:

```bash
python tests/test_watch.py    # end-to-end test for the file watcher
python tests/test_drift.py    # end-to-end test for config drift detection
```

`test_watch.py` launches the server as a subprocess, creates / modifies /
deletes a temporary file inside your configured codebase, and asserts that
the watcher reacts. Expected output ends with:

```
[test] === SUCCESS: incremental watchdog updates work ===
```

`test_drift.py` constructs a `CodebaseRAG` directly, mutates the in-memory
config snapshot to simulate three drift scenarios (no-drift baseline,
warning-severity, critical-severity) and asserts each is detected
correctly. The on-disk `metadata.json` is backed up and restored, so your
real index is left untouched. Expected ending:

```
[test] === SUCCESS: drift detection works as expected ===
```

---

## The MCP tools you get

Once connected, the AI client sees three tools:

### `search_codebase(query, top_k=None, file_glob=None, extensions=None, path_contains=None)`

Semantic search. Returns the top-K most relevant code chunks, with file name
and similarity score. **Use natural language**, not exact identifiers — that's
where semantic search beats grep:

- ❌ `search_codebase("CalculateDistance")` — too lexical
- ✅ `search_codebase("calculate distance between two points")` — semantic;
  also matches `ComputeSpacing`, `MeasureGap`, etc.

**Optional filters** (all AND-ed together) let you scope the search to a
subset of the codebase. Useful when the AI knows roughly where to look:

| Filter | Type | Example | When to use |
|---|---|---|---|
| `file_glob` | `str` | `"**/Bullet*/**/*.cs"` | Most flexible: any Unix-shell glob pattern, matched against both basename and full path. |
| `extensions` | `list[str]` | `[".py", ".js"]` | Quick "only these languages" scope. Leading dots are normalized. |
| `path_contains` | `str` | `"BulletSystem"` | Plain substring required in the path. Easiest to spell when you know a folder name. |

Filters run as **post-filters**: the underlying retriever over-fetches a
wider pool (5× `top_k`) and the filters trim the result set. Very narrow
filters on a large index may return fewer than `top_k` results — the
formatted output mentions the active filters so you can tell.

Example calls (the AI invokes these via MCP):

```
search_codebase("how damage is dispatched", extensions=[".cs"])
search_codebase("test for serialization", file_glob="**/tests/**")
search_codebase("damage", path_contains="BulletSystem")
```

### `update_codebase_index(force: bool = False)`

Force a full rebuild. The watcher keeps things in sync incrementally, so
day-to-day this is rarely needed. Useful after a complex merge or if you
suspect drift.

### `get_rag_status()`

Reports current state of the index, including the last indexed git commit
(if `git_integration.enabled` is `true`).

---

## Keeping the index up to date

Two complementary mechanisms:

**1. Real-time file watcher (default).**
A `watchdog.Observer` reacts to every save, create, delete, and move inside
your codebase. Updates are debounced by 2 seconds (configurable) so burst
saves are coalesced. Only changed files are re-indexed — full rebuilds are
not needed.

**2. Optional git post-commit hook (fallback).**
If the server happens to be off when you make changes, a one-line git hook
keeps the index in sync at every commit. Add this to
`<your-codebase>/.git/hooks/post-commit` and `chmod +x` it:

```bash
#!/bin/bash
local-codebase-rag-mcp build --config /path/to/config.json >/dev/null 2>&1 &
```

Replace `/path/to/config.json` with your absolute config path. The `&`
backgrounds the rebuild so the commit returns immediately.

---

## Hybrid retrieval

By default, `search_codebase` runs in **hybrid mode**: it combines a dense
(semantic) retriever with a sparse (BM25, lexical) retriever and fuses the
two rankings using **Reciprocal Rank Fusion** (RRF, `k=60`).

The motivation is that dense and sparse retrieval have complementary
failure modes:

- **Dense semantic search** is excellent at "natural-language" queries
  ("how is damage dispatched between entities?") but can be weak on short
  identifier queries ("`AStarPathFinder`") because there is little context
  to embed.
- **BM25 lexical search** excels at exact identifier and CamelCase matches
  but cannot bridge synonyms ("calculate distance" vs `MeasureGap`).

A code-aware tokenizer keeps full identifiers AND splits CamelCase /
snake_case into parts, so `IDamageable` is indexed as `idamageable`, `i`,
and `damageable`. This gives BM25 a fair shot at partial-name queries.

Switch modes via `config.json` (`search.mode`):

- `"hybrid"` (default) — RRF over dense + BM25
- `"dense"` — semantic only (lower latency, weaker on identifiers)
- `"sparse"` — BM25 only (no semantic understanding, useful as a baseline)

To compare modes empirically on your codebase, run:

```bash
python test_hybrid_vs_dense.py
```

This is a **benchmark, not a regression test**: it prints two columns of
rankings (dense | hybrid) for a handful of canned queries so you can see
whether hybrid is helping or hurting on your specific corpus.

The BM25 index lives entirely in memory and is rebuilt lazily from the
ChromaDB collection on first query (and after any incremental update). For
a few thousand chunks the build takes about a second; query latency adds
~5-15 ms compared to dense alone.

---

## Config drift detection

Some `config.json` changes silently invalidate an existing index. The most
dangerous case: changing `embedding.model_name`. Old vectors were computed
in one embedding space, new queries in another — search returns junk and
nothing tells you.

To prevent this, the server records a snapshot of the index-affecting config
fields inside `metadata.json` after every full rebuild. On every startup
(and every call to `get_rag_status`) it compares the live config against
that snapshot. Differences are reported with a severity:

| Severity | Field | What changed | What it means |
|---|---|---|---|
| 🔴 **Critical** | `embedding.model_name` | Index was built with model A, config now requests model B | Vectors are not comparable. Search will return wrong results. **Rebuild.** |
| 🔴 **Critical** | `codebase_path` | Index points at a different folder than the live config | The index doesn't describe the codebase you're querying. **Rebuild.** |
| 🟡 **Warning** | `supported_extensions` | Extensions added or removed | Index may be missing files (additions) or contain orphan vectors (removals). |
| 🟡 **Warning** | `collection_name` | Different collection than the one that was built | You are pointing at a different (likely empty) collection. |

Where the warning shows up:

- **`get_rag_status()`** — included in the tool's output. The AI client sees it
  and can warn you in chat. This is the primary surface.
- **Server stderr at boot** — useful when you tail the logs.

The server **does not refuse to serve searches** when drift is detected. The
choice is yours: maybe you know the drift is intentional and you'll rebuild
later, or maybe you want to inspect the impact first.

To clear a drift warning, run a full rebuild via the CLI:

```bash
local-codebase-rag-mcp build --config /path/to/config.json
```

Or, while the server is running, ask the AI client to call
`update_codebase_index(force=True)`.

Fields **not** tracked (changing them does not trigger drift):
`watcher.*`, `git_integration.*`, `loading_timeout_seconds`,
`search.default_top_k`, `storage_path`, `ignored_path_fragments`. The first
four are runtime-only behavior; `storage_path` is the database itself
(changing it means a different database, not a drift); `ignored_path_fragments`
currently only filters file-watcher events, not the indexing pipeline.

---

## Architecture

```
+------------------------------------------------------------+
| AI client (Claude Code, Antigravity, Cursor, ...)          |
+--------------------------+---------------------------------+
                           | MCP / JSON-RPC over stdio
+--------------------------v---------------------------------+
| mcp_server.py (FastMCP)                                    |
|   tools: search_codebase / update_codebase_index / status  |
|   watchdog observer (debounced incremental updates)        |
+--------------------------+---------------------------------+
                           |
+--------------------------v---------------------------------+
| rag_manager.py (CodebaseRAG)                               |
|   - SimpleDirectoryReader  (LlamaIndex)                    |
|   - HuggingFaceEmbedding   (BAAI/bge-small-en-v1.5, CPU)   |
|   - update_file / remove_file (incremental)                |
+--------------------------+---------------------------------+
                           |
+--------------------------v---------------------------------+
| ChromaDB (persistent, local file)                          |
|   collection "codebase"                                    |
+--------------------------+---------------------------------+
                           ^
                           | scanned + watched
+--------------------------+---------------------------------+
| Your codebase (read-only from this server's perspective)   |
+------------------------------------------------------------+
```

Repository layout:

```
local-codebase-rag-mcp/
├── pyproject.toml           Package metadata + dependencies
├── README.md                You are here
├── LICENSE                  MIT
├── config.example.json      Template config (commit this)
├── config.json              Your local config (gitignored)
├── src/
│   └── local_codebase_rag_mcp/
│       ├── __init__.py      Package version
│       ├── __main__.py      Enables `python -m local_codebase_rag_mcp`
│       ├── cli.py           argparse-based CLI dispatcher
│       ├── server.py        FastMCP server + tool definitions + watcher
│       ├── rag_manager.py   CodebaseRAG: indexing, search, drift, BM25
│       └── config.py        Typed config loader
├── tests/
│   ├── conftest.py          pytest path shim for src/ layout
│   ├── test_watch.py        End-to-end smoke test for the watcher
│   ├── test_drift.py        End-to-end smoke test for drift detection
│   ├── test_filters.py      Smoke test for search-filter parameters
│   └── test_hybrid_vs_dense.py  Side-by-side dense vs hybrid benchmark
└── rag_storage/             ChromaDB collection (gitignored, auto-created)
```

### Key design decisions

- **Retrieval-only, no generative LLM.** This server returns chunks; your AI
  client does any reasoning or generation. Keeps the server simple and the
  privacy guarantee absolute.
- **stdout is sacred.** In MCP stdio mode, stdout carries JSON-RPC. Any
  spurious print breaks the protocol. The server redirects fd 1 → fd 2 at
  the OS level during import/load and restores it just before `mcp.run()`.
- **Watcher starts before the ready signal** so the "Watcher active" log
  reaches stderr before FastMCP takes over the process.
- **Search does not auto-rebuild.** The watcher keeps the index live, so
  there's no need to spend time on a git check at every query.
- **Offline enforcement.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  prevent any HuggingFace network call after the first model download.
  `OPENAI_API_KEY` is removed from the environment.

---

## Troubleshooting

**The first start is very slow.**
First-time indexing scans the whole codebase and computes embeddings on CPU.
Pre-build with the snippet in
[Build the index for the first time](#build-the-index-for-the-first-time).

**The MCP client receives broken responses.**
Something is writing to stdout after `_restore_real_stdout()`. Check that
any `print()` you add uses `file=sys.stderr`.

**A file change isn't being indexed.**
- Verify the file extension is in `supported_extensions`.
- Verify the path doesn't match any `ignored_path_fragments` substring.
- Wait at least `watcher.debounce_seconds` after the save.
- Check the server's stderr for `[rag] Re-indexed ...` or any error line.

**`config.json` not found.**
Either copy `config.example.json` to `config.json`, or set the
`RAG_CONFIG_PATH` environment variable to an absolute path.

**The server exits as soon as I run it from a terminal.**
Expected: in MCP stdio mode the server reads JSON-RPC from stdin; closing
stdin (Ctrl+D / EOF) terminates the process. To exercise the server
manually, use `python tests/test_watch.py` instead, or use the CLI:
`local-codebase-rag-mcp search "..."` runs a query without keeping a
server alive.

**The `local-codebase-rag-mcp` command is not on my PATH.**
On Windows, pip may install console scripts into a directory that is not
on `PATH` by default (it prints a warning at install time when this
happens). Two options:

- **Recommended:** use the equivalent
  `python -m local_codebase_rag_mcp ...` form everywhere. It does not
  depend on `PATH` and is interchangeable with the short command. Full
  details in [Two equivalent ways to invoke](#installation).
- Or add the printed install directory to `PATH` (typically something
  like `%USERPROFILE%\AppData\Local\Python\...\Scripts` on Windows).

**I want to change the embedding model.**
Set `embedding.model_name` in `config.json` to any HuggingFace
sentence-transformer model, then force a full rebuild — different models
produce non-comparable vectors. The next start will also flag a CRITICAL
drift if you forget:

```bash
local-codebase-rag-mcp build --config /path/to/config.json
```

---

## Contributing

Issues and pull requests are welcome. For non-trivial changes, please open
an issue first so we can discuss the approach.

When contributing code:

- After a clone, install in editable mode so the package and the
  `local-codebase-rag-mcp` command resolve:
  ```bash
  pip install -e .
  ```
- Run `python -m py_compile src/local_codebase_rag_mcp/*.py tests/*.py`
  to catch syntax errors.
- Run `python tests/test_watch.py`, `python tests/test_drift.py`, and
  `python tests/test_filters.py` to confirm the smoke tests still pass.
  (`tests/test_hybrid_vs_dense.py` is a side-by-side benchmark, not a
  pass/fail test.)
- Keep all comments, docstrings, and log messages in English.

---

## Privacy guarantees

- No calls to OpenAI, Anthropic, or any LLM provider.
- No telemetry (LlamaIndex, HuggingFace, ChromaDB are all configured to be
  silent).
- No code is uploaded anywhere.
- The embedding model is cached locally and offline mode is enforced after
  the first download.
- All vectors and metadata stay in `rag_storage/` on your disk.

If you find a leak, please open a security issue.
