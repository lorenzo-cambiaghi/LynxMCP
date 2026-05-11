# Lynx

```
           /\     /\           Lynceus (Latin: Lynx or Lynceus) was a hero in Greek mythology,
          {  `---'  }          one of the Argonauts who joined Jason's expedition, renowned
          {  O   O  }          for his extraordinary eyesight. Son of Aphareus and brother
          ~~>  V  <~~          of Idas, Lynceus was said to possess the ability to see through
           \  \|/  /           solid objects and perceive things at immense distances.
            `-----'____
            /     \    \_
           {       }\  )_\_   _
           |  \_/  |/ /  \_\_/ )
            \__/  /(_/     \__/
              (__/
```

A tool that finds the buried answer in your codebase deserves the name of
the man who saw it first.

**Lynx** is a self-hosted **MCP (Model Context Protocol) server** that gives any AI coding
assistant — Claude Code, Antigravity, Cursor, Continue.dev, Aider, etc. —
the ability to perform **semantic search over your code and library
documentation**.

Configure one or more **sources** (your codebase, library docs, references)
and the server exposes a dedicated set of MCP tools for each, plus
cross-source search. Everything runs **100% locally**: no code, no
embeddings, no queries ever leave your machine. No API keys, no cloud
dependencies, no recurring costs.

> **License:** [MIT](LICENSE)

---

## Table of contents

1. [What this does](#what-this-does)
2. [Why it exists](#why-it-exists)
3. [How it works (in 30 seconds)](#how-it-works-in-30-seconds)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Migrating from v1 (single-source) configs](#migrating-from-v1-single-source-configs)
8. [Build the index for the first time](#build-the-index-for-the-first-time)
9. [Multi-source: indexing code AND library documentation](#multi-source-indexing-code-and-library-documentation)
10. [Connect it to your AI client](#connect-it-to-your-ai-client)
    - [Claude Code (CLI)](#claude-code-cli)
    - [Claude Code extension for VS Code](#claude-code-extension-for-vs-code)
    - [Google Antigravity](#google-antigravity)
    - [Cursor](#cursor)
    - [Continue.dev / Aider / other MCP-compliant clients](#continuedev--aider--other-mcp-compliant-clients)
11. [Command-line interface](#command-line-interface)
12. [Verify the integration works](#verify-the-integration-works)
13. [The MCP tools you get](#the-mcp-tools-you-get)
14. [Get the most out of it: AI integration rules](#get-the-most-out-of-it-ai-integration-rules)
15. [Keeping the index up to date](#keeping-the-index-up-to-date)
16. [Hybrid retrieval](#hybrid-retrieval)
17. [Config drift detection](#config-drift-detection)
18. [Architecture](#architecture)
19. [Troubleshooting](#troubleshooting)
20. [Contributing](#contributing)

---

## What this does

When an AI assistant has to answer questions about a codebase larger than its
context window, it has two options: read files almost at random hoping to find
what it needs, or ask you. Neither is great.

This server fixes that. You configure N **sources** in `config.json` (a
codebase, a folder of library docs, a vendor reference dump, ...). For
each source the server auto-generates two MCP tools at boot:

| Tool generated per source `<name>` | What it does |
|---|---|
| `search_<name>(query, top_k, ...)` | **Default** semantic search over that source. Hybrid retrieval (dense + BM25 + RRF), one query. |
| `deep_search_<name>(queries, ...)` | **Fallback** for the same source. Tries multiple query variants in order, stops at the first strong result. Use when `search_<name>` returned weak or empty results. |

Plus the global, cross-source tools:

| Global tool | What it does |
|---|---|
| `list_sources()` | Enumerate configured sources with type, path, chunk count, drift status. |
| `search_all_sources(query, ...)` | Run the query against every source in parallel and fuse rankings via RRF. Use when you don't know which source has the answer. |
| `deep_search_all_sources(queries, ...)` | Multi-query × multi-source fallback. Use sparingly — runs N×M retrievals. |
| `update_source_index(source, force)` | Force a full rebuild of one source. |
| `get_rag_status(source?)` | Report state of one source or all. |

When you ask the AI *"how is damage handled in this codebase?"* with a
source named `myproject`, it calls `search_myproject("damage handling")`
and gets back the top-K most relevant code snippets — across files,
regardless of naming conventions — and answers your question with that
context. With multiple sources, it picks the right tool based on the
docstring of each `search_<name>`.

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
- **Explicit retrieval.** `search_<source>` tools are invoked deliberately
  by the assistant — you can also bias *when* via project rules files.

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
git clone https://github.com/<your-username>/lynx.git
cd lynx
pip install -e .
```

This installs the package and exposes a `lynx` console
command. (When PyPI publication lands, the install becomes a single
`pip install lynx`.)

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
lynx --version
lynx --help
```

> ### ⚠️ Two equivalent ways to invoke (read this once)
>
> Throughout this README every example uses the short form
> `lynx ...`. The fully equivalent long form is
> `python -m lynx ...`. Pick whichever works for you:
>
> | Use the short form... | Use `python -m ...` instead when... |
> |---|---|
> | The `lynx` command is on your `PATH`. | The console script is missing or not on `PATH` (common on Windows: pip prints a warning at install time when `Scripts/` is not in `PATH`). |
> | You want shorter snippets in IDE config files. | You want a guaranteed-working invocation regardless of `PATH`. Recommended for IDE configs that you'll share or commit to dotfiles. |
>
> Side-by-side, all of these are interchangeable:
>
> ```bash
> lynx --help
> python -m lynx --help
>
> lynx build --config /path/to/config.json
> python -m lynx build --config /path/to/config.json
>
> lynx serve --config /path/to/config.json
> python -m lynx serve --config /path/to/config.json
> ```
>
> If you're on Windows and the short form fails with "command not found",
> just prefix the invocation with `python -m`. That's the only difference.

---

## Configuration

The config is a single JSON file with **shared settings at the top level**
and a `sources` block that lists everything to index. Copy the example and
edit it:

```bash
cp config.example.json config.json
```

Minimum useful config (single source):

```json
{
  "config_version": 2,
  "storage_path": "./rag_storage",
  "loading_timeout_seconds": 600,
  "embedding": { "model_name": "BAAI/bge-small-en-v1.5" },
  "search": {
    "default_top_k": 8,
    "mode": "hybrid",
    "rrf_k": 60,
    "candidate_pool_size": 30,
    "deep": {
      "min_results": 2,
      "score_thresholds": { "dense": 0.45, "hybrid": 0.012, "sparse": 3.0 }
    }
  },
  "sources": {
    "myproject": {
      "type": "codebase",
      "path": "C:/path/to/your/codebase",
      "supported_extensions": [".py", ".md", ".js", ".ts"],
      "ignored_path_fragments": ["/.git/", "/node_modules/"],
      "watcher": { "enabled": true, "debounce_seconds": 2.0 },
      "git_integration": { "enabled": true }
    }
  }
}
```

### Top-level (shared across every source)

| Field | What it does |
|---|---|
| `config_version` | **Required.** Must be `2`. Use `lynx migrate-config` to upgrade an older config. |
| `storage_path` | Where per-source ChromaDB folders live (each at `<storage_path>/<source_name>/`). Relative paths are resolved against the config file's directory. Default `./rag_storage`. |
| `loading_timeout_seconds` | Max time to wait for the first index build before MCP tool calls give up. Default `600`. |
| `embedding.model_name` | Any HuggingFace sentence-transformer model. **Changing this invalidates all existing vectors across every source** — see [Config drift detection](#config-drift-detection). |
| `search.default_top_k` | Default number of chunks any `search_*` tool returns when `top_k` is not passed. |
| `search.mode` | `"hybrid"` (default), `"dense"`, or `"sparse"`. See [Hybrid retrieval](#hybrid-retrieval). |
| `search.rrf_k` | Reciprocal Rank Fusion constant (default `60`). |
| `search.candidate_pool_size` | Per-retriever candidate pool size (default `30`). |
| `search.deep.min_results` | For every `deep_search_*` tool: minimum results required for a variant to count as "strong" (default `2`). |
| `search.deep.score_thresholds` | Mode-specific "weak" thresholds. Defaults: `dense=0.45`, `hybrid=0.012`, `sparse=3.0`. |

### Per-source (one entry under `sources`)

Source names must match `^[a-zA-Z][a-zA-Z0-9_]{0,39}$` — letter, then letters
/ digits / underscores. The name is used **verbatim** in the auto-generated
tool names (`search_<name>`, `deep_search_<name>`), so pick something the AI
client will see clearly in its tool list (e.g. `myproject`, `unityDoc`,
`avalonia_docs`).

| Field (codebase type) | What it does |
|---|---|
| `type` | **Required.** `"codebase"` is the only type in M1. Future: `"webdoc"`, `"pdf"`. |
| `path` | **Required.** Absolute path of the directory to index. |
| `supported_extensions` | File extensions to include (e.g. `[".py", ".md"]`). Anything else is ignored. |
| `ignored_path_fragments` | Substrings: any path containing one is skipped by the watcher (forward slashes are auto-normalized). |
| `watcher.enabled` | Set to `false` to disable the live file watcher for this source. |
| `watcher.debounce_seconds` | Time to wait after a save before re-indexing. Default `2.0`. |
| `git_integration.enabled` | When `true`, drift / status reports include the last indexed git commit for this source. |

**Want a different config location?** Three options, in priority order:

1. Pass `--config /absolute/path/to/config.json` to any CLI subcommand.
2. Set the environment variable `RAG_CONFIG_PATH` to an absolute path.
3. Default: `./config.json` in the current working directory.

For MCP IDE integration, option (1) is the recommended approach — see the
[Connect it to your AI client](#connect-it-to-your-ai-client) snippets.

---

## Migrating from v1 (single-source) configs

The v0.1.x line used a flat schema with `codebase_path` at the top level and
no `sources` block. v0.2.0 introduced the multi-source schema documented
above. The migration is a one-liner:

```bash
lynx migrate-config --input config.json --source-name myproject
# writes config.v2.json next to the input
```

Review the generated file, then replace your old `config.json` with it (or
point your launcher at the new path via `--config` / `RAG_CONFIG_PATH`).
Your old `rag_storage/` is at the v1 layout (chroma data at the root) and
won't be read by v0.2 — delete it and run `build --source <name>` to rebuild
under the v2 per-source layout. A 1k-file codebase rebuilds in a few minutes.

The loader emits an explicit error pointing at this command if it sees a v1
config, so you'll never silently run with a stale schema.

---

## Build the index for the first time

The first index build can take a few minutes per source on a large codebase
(it scans every file and computes embeddings on CPU). To avoid making your
AI client wait the first time it connects, pre-build the index from a
terminal.

From the directory containing your `config.json`, run:

```bash
# When you have a single source, --source is optional:
lynx build

# When you have multiple sources, name the one you want:
lynx build --source myproject
lynx build --source unityDoc
```

> If you get `command not found`, use the equivalent `python -m` form:
> `python -m lynx build --source <name>`. See
> [Two equivalent ways to invoke](#installation) for why.

You should see logs like:

```
[rag] Indexing codebase from C:\path\to\your\codebase...
[rag] Found 1247 files
Source 'myproject' ready.
```

If your `config.json` lives elsewhere, use `--config`:

```bash
lynx build --config /path/to/config.json --source myproject
```

The same `build` command also handles **rebuilding** later (e.g. after
changing the embedding model) — it does a full force rebuild whenever the
source already has an index.

> Note: starting the server with `lynx serve` on a fresh
> install will also trigger an initial build of every source implicitly.
> Pre-building is just a courtesy so the first MCP client doesn't have to
> wait minutes.

Each source has its own subdirectory under `storage_path/`:

```
rag_storage/
├── myproject/
│   ├── chroma.sqlite3
│   └── metadata.json
└── unityDoc/
    ├── chroma.sqlite3
    └── metadata.json
```

Subsequent server starts are fast (seconds) once every configured source
has a populated `<source>/` subdir.

---

## Multi-source: indexing code AND library documentation

The primary use case for multi-source is keeping a local copy of library
documentation that the AI client doesn't know about — either because the
library updates faster than the model's knowledge cutoff (Unity, AvaloniaUI,
SDKs that ship every quarter) or because it's internal / niche / unindexed
on the public web.

Configure each one as its own source. Example for a Unity gamedev with
local copies of two documentation sets in addition to their code:

```json
{
  "config_version": 2,
  "storage_path": "./rag_storage",
  "embedding": { "model_name": "BAAI/bge-small-en-v1.5" },
  "search": { ... },
  "sources": {
    "myproject": {
      "type": "codebase",
      "path": "C:/projects/mygame/Assets/Scripts",
      "supported_extensions": [".cs", ".md", ".shader"],
      "watcher": { "enabled": true, "debounce_seconds": 2.0 }
    },
    "unityDoc": {
      "type": "codebase",
      "path": "C:/vendor-docs/unity-6.2-manual",
      "supported_extensions": [".md", ".html", ".txt"],
      "watcher": { "enabled": false, "debounce_seconds": 2.0 }
    },
    "avalonia": {
      "type": "codebase",
      "path": "C:/vendor-docs/avalonia-docs",
      "supported_extensions": [".md", ".rst"],
      "watcher": { "enabled": false, "debounce_seconds": 2.0 }
    }
  }
}
```

This boots the server with **six per-source tools** plus the globals:

```
search_myproject         deep_search_myproject
search_unityDoc          deep_search_unityDoc
search_avalonia          deep_search_avalonia

list_sources             search_all_sources       deep_search_all_sources
update_source_index      get_rag_status
```

Each `search_<name>` has a docstring that names the source by type and path,
so the AI client picks the right tool without you having to tell it. Pair
this with an AI integration rules file (see the [section below](#get-the-most-out-of-it-ai-integration-rules))
to bias the AI toward `search_unityDoc` for Unity API questions, etc.

> **Documentation source types are coming.** Today every source is
> `type: "codebase"` — fine for any folder of text files (markdown
> documentation included). Native `type: "webdoc"` (crawl + index a docs
> site) and `type: "pdf"` (extract + index PDF manuals) land in upcoming
> milestones. The architecture is already in place; only the new ingestors
> need adding.

---

## Connect it to your AI client

The server speaks **MCP over stdio** — every modern AI client supports this
the same way: a `command` to launch the server, an array of `args`, and
optionally `env` and `cwd`.

In every example below, replace `C:/path/to/lynx` with the
absolute path where you cloned this repo.

> Tip on Windows: in JSON, you can use either forward slashes (`C:/Users/...`)
> or escaped backslashes (`C:\\Users\\...`). Forward slashes are simpler.

All snippets below assume the package is installed (`pip install -e .` from
the repo, or eventually `pip install lynx`) and that you
have a `config.json` somewhere on disk.

> **Tip — passing the config:** since an MCP client launches the server
> from an unpredictable working directory, always pass `--config` with an
> absolute path. Replace `C:/path/to/config.json` in every snippet with
> your actual path.

> **Tip — if the short command isn't on your `PATH`:** every snippet below
> uses `"command": "lynx"`. If pip installed the console
> script outside your `PATH` (common on Windows — pip prints a warning at
> install time), swap the entry for the equivalent `python -m` form:
>
> ```json
> "command": "python",
> "args": ["-m", "lynx", "serve", "--config", "C:/path/to/config.json"]
> ```
>
> The two are functionally identical. The `python -m` form is also a
> safer default for shared / committed config files, since it does not
> depend on the user's `PATH`.

### Claude Code (CLI)

Add the server with one command:

```bash
claude mcp add codebase-rag --scope user -- lynx serve --config C:/path/to/config.json
```

Or edit `~/.claude.json` (or `%USERPROFILE%\.claude.json` on Windows) directly:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "lynx",
      "args": ["serve", "--config", "C:/path/to/config.json"]
    }
  }
}
```

Restart Claude Code. The next session will list one `search_<name>` /
`deep_search_<name>` pair for every configured source, plus the global
tools (`list_sources`, `search_all_sources`, `deep_search_all_sources`,
`update_source_index`, `get_rag_status`).

### Claude Code extension for VS Code

The VS Code extension reads the same `~/.claude.json` configuration as the
CLI — set it up once and it's available everywhere.

If you prefer per-workspace configuration, create a `.mcp.json` in your
workspace root:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "lynx",
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
      "command": "lynx",
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
      "command": "lynx",
      "args": ["serve", "--config", "C:/path/to/config.json"]
    }
  }
}
```

Restart Cursor. Settings → MCP should now list the server.

### Continue.dev / Aider / other MCP-compliant clients

Any MCP client takes the same three pieces of information:

- **Command:** `lynx`
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
      "command": "lynx",
      "args": ["serve", "--config", "C:/configs/frontend.json"]
    },
    "codebase-rag-backend": {
      "command": "lynx",
      "args": ["serve", "--config", "C:/configs/backend.json"]
    }
  }
}
```

Each appears as a separate MCP server in the IDE with its own three tools.
A single `pip install` covers all of them.

---

## Command-line interface

The same `lynx` command exposes four subcommands. Useful
for debugging the index, scripting, or just querying the codebase without
opening an AI assistant.

```text
lynx [--version] [-h] COMMAND ...

  serve   Run the MCP server (default if no command is given)
  build   Force a full rebuild of the index
  search  Run an ad-hoc search query (no MCP client needed)
  status  Show RAG status: git state, last update, config drift
```

> Every example in this section can be invoked equivalently as
> `python -m lynx <subcommand> ...` if the short
> `lynx` script is not on your `PATH`. See
> [Two equivalent ways to invoke](#installation) at the end of the
> Installation section.

Every subcommand accepts `--config PATH` (or honors `RAG_CONFIG_PATH`, or
falls back to `./config.json` in the current directory).

### `serve`

Runs the MCP server over stdio. This is what your IDE invokes:

```bash
lynx serve --config /path/to/config.json
```

### `build`

Forces a full rebuild of the index. Use this for the first-time build,
after changing the embedding model, or whenever you want a known-clean
rebuild:

```bash
lynx build --config /path/to/config.json
```

### `search`

Runs an ad-hoc search and prints the top results to your terminal. Useful
for "is the index actually finding this file?" debugging or for piping
into other commands.

```bash
lynx search "how damage is dispatched" --top-k 3
lynx search "IDamageable" --mode dense --ext .cs
lynx search "auth" --glob "**/middleware/**" -k 5
```

Supported flags: `--top-k / -k`, `--mode {hybrid,dense,sparse}`, `--ext`
(repeatable, e.g. `--ext .py --ext .pyi`), `--glob`, `--path-contains`.

### `status`

Reports git state, last update time, and any config drift detected since
the last full rebuild:

```bash
$ lynx status
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

You should see it invoke the appropriate `search_<name>` tool for your
source and return relevant snippets.

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

For each source `<name>` configured in `config.json`, the server
auto-registers **two tools** at boot. Add three sources → six per-source
tools. Plus the five globals. The AI client sees the whole set in its tool
list and picks based on the docstrings.

### Per-source tools (auto-generated)

#### `search_<name>(query, top_k=None, file_glob=None, extensions=None, path_contains=None)`

Semantic search over the source named `<name>`. Returns the top-K most
relevant chunks with file name, score, and (for cross-source results) the
`source` tag. **Use natural language**, not exact identifiers — that's
where semantic search beats grep:

- ❌ `search_myproject("CalculateDistance")` — too lexical
- ✅ `search_myproject("calculate distance between two points")` — semantic;
  also matches `ComputeSpacing`, `MeasureGap`, etc.

**Optional filters** (all AND-ed together) let you scope the search to a
subset of files within the source. Useful when the AI knows roughly where
to look:

| Filter | Type | Example | When to use |
|---|---|---|---|
| `file_glob` | `str` | `"**/Bullet*/**/*.cs"` | Most flexible: any Unix-shell glob pattern, matched against both basename and full path. |
| `extensions` | `list[str]` | `[".py", ".js"]` | Quick "only these languages" scope. Leading dots are normalized. |
| `path_contains` | `str` | `"BulletSystem"` | Plain substring required in the path. Easiest to spell when you know a folder name. |

Filters run as **post-filters**: the underlying retriever over-fetches a
wider pool (5× `top_k`) and the filters trim the result set. Very narrow
filters on a large index may return fewer than `top_k` results — the
formatted output mentions the active filters so you can tell.

#### `deep_search_<name>(queries, top_k=None, mode=None, file_glob=None, extensions=None, path_contains=None, min_score=None, min_results=None, return_all_variants=False)`

**Fallback** for the same source. Use **only** when `search_<name>`
returned weak or empty results, or when the user explicitly asks for a
more thorough search. Slower — when all variants fail it makes N
retrievals instead of 1.

Accepts an **ordered list of query variants**. The first variant whose
result set crosses the quality threshold wins. If none crosses, the
strongest weak set is returned with an explicit warning so the AI can
decide what to tell the user. Crucially:

- 1 string in the list ⇒ behaves like a single-query `search_<name>`.
- N strings ⇒ the server stops at the first strong result. Variants 2..N
  are only run when earlier variants were weak. **No cost when the first
  works.**

| Parameter | Type | Purpose |
|---|---|---|
| `queries` | `list[str]` | Required, ordered. Variant 1 is your best guess, variants 2+ are fallbacks. Use *genuinely different* phrasings, not paraphrases. |
| `top_k` | `int` | Results to return. Defaults to `search.default_top_k`. |
| `mode` | `"dense" \| "sparse" \| "hybrid" \| None` | Per-call retrieval mode override for this source. `"dense"` if hybrid noise hurts, `"sparse"` for exact identifier hunting. |
| `file_glob`, `extensions`, `path_contains` | same as `search_<name>` | Applied to every variant. |
| `min_score` | `float \| None` | Per-call threshold override. `None` = use mode-specific default. |
| `min_results` | `int \| None` | Per-call min-results override. `None` = `search.deep.min_results` (default 2). |
| `return_all_variants` | `bool` | Include a per-variant summary in the response for debugging. |

The output includes a header that tells the AI **which variant won**:

```
Found 5 results in source 'myproject' (variant 2/3 won (mode='dense')):
--- 1. File: IDamageable.cs (Score: 0.6234) ---
...
```

**Good query-variant design.** Bad variants are paraphrases of the same
phrasing; good variants approach the question from different angles
(literal identifier, semantic intent, usage angle). See the "AI integration
rules" section for examples.

### Global tools (always available, do not depend on source names)

#### `list_sources()`

Enumerate all configured sources with their type, path, chunk count, and
drift status. Useful for the AI to discover what's available without
inspecting the config:

```
Sources (3):
  - myproject (type: codebase, chunks: 3168)
      path: C:/projects/mygame/Assets/Scripts
  - unityDoc (type: codebase, chunks: 8421)
      path: C:/vendor-docs/unity-6.2-manual
  - avalonia (type: codebase, chunks: 1245)
      path: C:/vendor-docs/avalonia-docs
```

#### `search_all_sources(query, top_k=None, file_glob=None, extensions=None, path_contains=None)`

Run the query against **every** source in parallel and fuse the rankings
via Reciprocal Rank Fusion. Each result is tagged with its `source` so the
AI can tell which source produced it. Use when you don't know which source
has the answer (e.g. *"is `Spline` a class in my code or in a library
I'm using?"*).

Cost: linear in the number of sources. With 3 sources you get 3 retrievals
per call. With 10 sources, 10. Use direct `search_<name>` when you can.

#### `deep_search_all_sources(queries, top_k=None, ...)`

Multi-query × multi-source fallback. Combines the variant ladder with
cross-source fusion. Each variant is run on every source; results fused;
first variant that passes the threshold wins. **Use very sparingly** —
runs N×M retrievals in the worst case.

#### `update_source_index(source, force=False)`

Force a full rebuild of one source's index. Day-to-day the watcher keeps
things in sync; use this after a complex merge, a bulk rename, or when
drift detection flags a critical change.

#### `get_rag_status(source=None)`

Report state of the RAG index. Pass `source="<name>"` to inspect one;
omit to get the status of every source. Reports git state, last update
time, and config drift.

---

## Get the most out of it: AI integration rules

Installing the server and registering the MCP tools makes them *available*
to your AI assistant. It does not, on its own, make the assistant use them
strategically. By default most AI clients will only invoke
`search_<source>` when the query is obviously "find X" — they will not,
unprompted, search before implementing a new utility, a new interface,
or a new architectural pattern, and they may not realize a docs source
exists for the library they're about to misuse.

That habit (search before implement, pick the right source) is the
single highest-value workflow this tool unlocks. It is also the workflow
your AI client will skip unless you ask for it explicitly. The standard
way to ask is a project rules file the assistant reads at the start of
every session.

### Drop-in template

Below is a generic template. Replace `<myproject>` / `<unityDoc>` / etc.
with your actual source names. Copy it into whichever convention file
your AI client uses (table further down).

```markdown
# Code Reuse & Library Awareness

This project exposes a local MCP server with semantic search over
multiple sources. Use them BEFORE writing new code or guessing at
library APIs.

## Sources available

- `search_<myproject>` — our own codebase. Use to discover existing
  implementations before writing new utilities, interfaces, or patterns.
- `search_<unityDoc>` — Unity 6.x manual / API reference. Use whenever
  you're about to call a Unity API: the model's knowledge cutoff may
  predate the version actually in use.
- `search_<avalonia>` — AvaloniaUI docs. Use for any AvaloniaUI binding,
  control, or style question (the model doesn't know this library well).

When in doubt about *which* source has the answer, call
`search_all_sources(query)` once. It fuses results across every source
with RRF and tags each hit with its source.

## When to search first

Before implementing any of the following, search the codebase source:

- **Utility functions** — math, geometry, IO, parsing, formatting,
  helpers, extension methods.
- **Generic interfaces or components** — persistence, lifecycle,
  eventing, pooling, caching, registration, validation.
- **Architectural patterns** — factory, registry, observer, strategy,
  command, mediator.

Before invoking a library API, search the corresponding docs source.
Old habits (relying on training-data knowledge) are wrong for libraries
that update frequently.

For one-off code that lives in a single feature (gameplay scripting,
a single API endpoint, a one-shot script), normal judgement is fine.

## How to search

- **Semantic, not lexical.** Describe what the code *does*, not the
  name you would give it.
  - Bad: `search_<myproject>("AngleBetween")`
  - Good: `search_<myproject>("calculate angle between two vectors in degrees")`
- **Top-K**: the server default is `search.default_top_k` from config
  (typically 8). Pass `top_k=10` or higher for very open discovery
  queries; pass `top_k=3` when you only want the one canonical place.
- **Filters**: when you already know the rough area, narrow with
  `extensions=[...]`, `path_contains="..."`, or `file_glob="..."`.

## How to interpret scores

The default retrieval mode is **hybrid** (BM25 + dense, fused via
Reciprocal Rank Fusion with `k=60`). RRF scores are **NOT** cosine
similarity — they live on a different scale:

| Hybrid (RRF) score | Meaning |
|---|---|
| ~0.030–0.033 | Excellent. The chunk is near rank 0 in both retrievers. This is the practical maximum. |
| ~0.020–0.029 | Strong. Worth showing to the user. |
| ~0.012–0.019 | Decent. Useful for context but maybe not the canonical answer. |
| < 0.012 | Weak. Consider escalating to `deep_search_<source>`. |

Do **NOT** interpret a hybrid score of 0.03 as "low confidence" — that
is the top of the scale, not the bottom. If you ever see scores in the
range 0.3–0.7, the call was running in `mode="dense"` (per-call
override) and you're looking at cosine similarity, where `> 0.55` is
a good match and `< 0.45` is weak.

## When to escalate to `deep_search_<source>`

`search_<source>` is your default — fast, one query, hybrid retrieval.
Use it 90% of the time.

Escalate to the matching `deep_search_<source>` ONLY when:

1. `search_<source>` returned no results or all scores looked weak.
2. The user rejected the previous results ("non è quello che cercavo",
   "search more thoroughly", "look harder").
3. The user explicitly asks for an exhaustive search.

When you escalate, prepare 2–3 **genuinely different** query variants —
not paraphrases. Bad: `["IDamageable", "IDamageable interface", "the
IDamageable interface"]`. Good:

```
deep_search_<myproject>([
  "IDamageable interface",                              # literal terms
  "damage system contract",                             # semantic intent
  "where damage is dispatched between entities",        # usage angle
])
```

You can also override `mode` per-call:
- `mode="dense"` if previous hybrid result was diluted by lexical noise.
- `mode="sparse"` to chase an exact identifier or string.

Anti-pattern: do NOT default to `deep_search_<source>` "just in case".
If `search_<source>` already gave you what you need, you're done.

## How to report back

For each candidate above the threshold, present:

- File path and a one-line summary of what the chunk does.
- Verdict for the current task: **reusable as-is**, **extendable**,
  or **only superficially similar**.

Only after the user confirms that nothing existing fits, implement
from scratch.
```

### Where to put the rules file

| AI client | Convention file | Notes |
|---|---|---|
| Claude Code (CLI + VS Code extension) | `CLAUDE.md` (project root) or `.clauderules` | Both are auto-loaded. `CLAUDE.md` is the newer, cross-tool-friendly form. |
| Google Antigravity | `AGENTS.md` (project root) or files under `.agents/rules/` | `.agents/rules/` lets you split rules into multiple themed files. Antigravity ≥ v1.20.3 also reads `AGENTS.md`. |
| Cursor | Files under `.cursor/rules/` | Per-rule scoping with globs is supported. |
| Aider, Continue.dev, others | `AGENTS.md` (cross-tool standard) | Most modern agents support this convention. |

If you want the rule to apply across multiple AI clients in the same
project, the safest bet is `AGENTS.md` in the project root — it is
read by the largest set of agents.

### Tuning the strictness

The template above is a sensible default. Two knobs to consider:

**How aggressive to be about the "search first" requirement.**
The strict version says "MUST search before implementing X". A softer
version says "consider searching before implementing X". For shared
codebases with 5+ developers and a history of duplicated utilities, the
strict version pays for itself within a week. For solo projects or
prototypes the softer version avoids friction. Pick what matches your
duplication pain.

**When to suggest a forced rebuild.**
The file watcher keeps the index live for normal saves and refactors,
so day-to-day the AI never needs to think about freshness. A line in
your rules like:

> If `get_rag_status` reports `Needs update` or any drift warning,
> tell the user once at the start of the session — do not silently
> work against a stale index.

…is enough to surface the rare cases where a forced rebuild is needed
(complex merges, large bulk renames, embedding model changes) without
making every session noisy.

### A note on what to *avoid* in your rules

- Don't tell the AI to call `update_source_index(source, force=True)`
  routinely. A force rebuild on a large source takes minutes and re-embeds
  everything; it should be a deliberate user action, not a reflex. The
  `Needs update` line is the right surface.
- Don't ask the AI to call `search_<source>` for queries it can answer
  by reading a single known file. The tool's job is discovery; once you
  know what file to read, just read it.
- Don't default to `search_all_sources` for everything. It costs N
  retrievals per call. Use it only when you genuinely don't know which
  source has the answer; otherwise pick the specific `search_<source>`.
- Don't ask for "search the entire codebase exhaustively" — `top_k=10`
  is enough for almost any discovery query, and the filters
  (`extensions`, `path_contains`, `file_glob`) exist precisely to avoid
  this.

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
lynx build --config /path/to/config.json >/dev/null 2>&1 &
```

Replace `/path/to/config.json` with your absolute config path. The `&`
backgrounds the rebuild so the commit returns immediately.

---

## Hybrid retrieval

By default, every `search_<source>` tool runs in **hybrid mode**: it
combines a dense (semantic) retriever with a sparse (BM25, lexical)
retriever and fuses the
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
lynx build --config /path/to/config.json --source <name>
```

Or, while the server is running, ask the AI client to call
`update_source_index("<name>", force=True)`.

Fields **not** tracked (changing them does not trigger drift):
`watcher.*`, `git_integration.*`, `loading_timeout_seconds`,
`search.default_top_k`, `storage_path`, `ignored_path_fragments`. The first
four are runtime-only behavior; `storage_path` is the database itself
(changing it means a different database, not a drift); `ignored_path_fragments`
currently only filters file-watcher events, not the indexing pipeline.

---

## Architecture

```
+--------------------------------------------------------------------+
| AI client (Claude Code, Antigravity, Cursor, ...)                  |
+----------------------------+---------------------------------------+
                             | MCP / JSON-RPC over stdio
+----------------------------v---------------------------------------+
| server.py  (FastMCP)                                               |
|   - dynamically registers `search_<name>` / `deep_search_<name>`   |
|     for every entry in config.sources                              |
|   - global tools: list_sources / search_all_sources /              |
|     deep_search_all_sources / update_source_index / get_rag_status |
+----------------------------+---------------------------------------+
                             |
+----------------------------v---------------------------------------+
| source_manager.py  (SourceManager)                                 |
|   - per-source dispatch (KeyError on unknown source)               |
|   - cross-source RRF fusion for *_all_sources                      |
+----------+------------------------------------+--------------------+
           |                                    |
+----------v-----------+                +-------v-------------------+
| sources/codebase.py  | ... per type ..| sources/<future>.py       |
| CodebaseBackend      |                | WebdocBackend, PdfBackend |
|   wraps CodebaseRAG  |                |   (M2 / M3)               |
|   runs file watcher  |                +---------------------------+
+----------+-----------+
           |
+----------v---------------------------------------------------------+
| rag_manager.py (CodebaseRAG)                                       |
|   - SimpleDirectoryReader  (LlamaIndex)                            |
|   - HuggingFaceEmbedding   (BAAI/bge-small-en-v1.5, CPU)           |
|   - dense + BM25 + RRF fusion (hybrid retrieval)                   |
|   - drift detection + deep_search variant ladder                   |
|   - update_file / remove_file (incremental)                        |
+----------+---------------------------------------------------------+
           |
+----------v---------------------------------------------------------+
| ChromaDB — one collection per source, in its own subdir            |
|   rag_storage/<source_name>/chroma.sqlite3                         |
+----------+---------------------------------------------------------+
           ^
           | scanned + watched (codebase sources only)
+----------+---------------------------------------------------------+
| Your sources (one path or URL per entry in config.sources)         |
+--------------------------------------------------------------------+
```

Repository layout:

```
lynx/
├── pyproject.toml           Package metadata + dependencies
├── README.md                You are here
├── LICENSE                  MIT
├── config.example.json      Template config (commit this)
├── config.json              Your local config (gitignored)
├── src/
│   └── lynx/
│       ├── __init__.py        Package version
│       ├── __main__.py        Enables `python -m lynx`
│       ├── cli.py             argparse-based CLI dispatcher (incl. migrate-config)
│       ├── server.py          FastMCP server, dynamic per-source tool registration
│       ├── source_manager.py  SourceManager: per-source dispatch + cross-source RRF
│       ├── rag_manager.py     CodebaseRAG: hybrid retrieval, drift, BM25, deep_search
│       ├── config.py          v2 config loader with per-type validation
│       └── sources/           Per-type source backends
│           ├── __init__.py    SOURCE_BACKENDS registry
│           ├── base.py        SourceBackend abstract base class
│           └── codebase.py    CodebaseBackend (wraps CodebaseRAG + watcher)
├── tests/
│   ├── conftest.py            pytest path shim + per-source RAG helper
│   ├── test_watch.py          End-to-end smoke test for the watcher
│   ├── test_drift.py          End-to-end smoke test for drift detection
│   ├── test_filters.py        Smoke test for search-filter parameters
│   ├── test_deep_search.py    Smoke test for the deep_search fallback ladder
│   ├── test_multi_source.py   Smoke test for multi-source dispatch + isolation
│   └── test_hybrid_vs_dense.py  Side-by-side dense vs hybrid benchmark
└── rag_storage/               ChromaDB collections, one subdir per source (gitignored)
    ├── myproject/
    └── unityDoc/
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
`lynx search "..."` runs a query without keeping a
server alive.

**The `lynx` command is not on my PATH.**
On Windows, pip may install console scripts into a directory that is not
on `PATH` by default (it prints a warning at install time when this
happens). Two options:

- **Recommended:** use the equivalent
  `python -m lynx ...` form everywhere. It does not
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
lynx build --config /path/to/config.json
```

---

## Contributing

Issues and pull requests are welcome. For non-trivial changes, please open
an issue first so we can discuss the approach.

When contributing code:

- After a clone, install in editable mode so the package and the
  `lynx` command resolve:
  ```bash
  pip install -e .
  ```
- Run `python -m py_compile src/lynx/*.py src/lynx/sources/*.py tests/*.py`
  to catch syntax errors.
- Run `python tests/test_watch.py`, `python tests/test_drift.py`,
  `python tests/test_filters.py`, `python tests/test_deep_search.py`,
  and `python tests/test_multi_source.py` to confirm the smoke tests
  still pass.
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
