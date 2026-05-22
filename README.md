# LynxMCP

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

*A tool that finds the buried answer in your codebase deserves the name of the man who saw it first.*

### 🚀 Supercharge Your AI Assistant with "Super Sight"

Ever asked your AI assistant to fix a bug, only to watch it blindly guess because it can't "see" your entire project?
AI models are incredibly smart, but they suffer from **context limits**. They don't know your codebase, they haven't read the documentation for the latest version of your framework, and they often hallucinate the wrong APIs.

**LynxMCP** fixes this. It runs locally as an MCP server and gives your AI assistant three ways to interrogate your project the moment a question shows up — without uploading a single byte to the cloud.

### 🛠️ What you get

1. **🔍 Semantic + lexical search** (always on) — *"find the code that handles X"*. AST-aware chunking (13 languages) + hybrid dense + BM25 + RRF fusion. Beats `grep` because it understands what code *does*, not just how it's spelled.
2. **🌐 Library docs at hand** (always on) — point Lynx at a docs site (Unity, Avalonia, your in-house framework) and the AI searches the *real* current API instead of hallucinating one from its training data.
3. **📄 PDF documents** (always on, new in v0.7) — point Lynx at a folder of PDFs (manuals, RFCs, normative, white-papers) and it extracts the text page-by-page; results cite the source as `User Manual.pdf p.42`. Scanned PDFs and password-protected ones are auto-skipped with a clear warning.
4. **🧬 Structural code understanding** (opt-in, new in v0.6) — a **knowledge graph** of your codebase: *"who calls `IDamageable`?"*, *"what concrete classes extend `BaseController`?"*, *"how does `CheckoutFlow` reach `PaymentGateway`?"*, *"give me an architectural overview"*. Search finds the file; the graph finds the relationships.
5. **🎯 Code-aware combined queries** (new in v0.8) — `find_definition`, `find_usages`, `find_tests_for`, `find_similar`, `search_diff`. They combine the graph layer with hybrid search so the AI gets a direct answer to *"where is X defined?"*, *"who uses X?"*, *"are there tests for X?"*, *"is there code similar to this snippet?"*, *"what did I change vs main, and is anything else affected?"* — instead of you scrolling through search results.
6. **⚙️ Optional cross-encoder reranker** (opt-in, new in v0.8) — after the hybrid RRF, a small ~80MB local cross-encoder model re-scores the top-N candidates by actually *looking at* (query, chunk) content. ~+20-30% precision@1 on ambiguous queries for ~50ms extra latency. Off by default.
7. **🎛️ LynxManager** (new in v0.9) — a sub-command namespace that turns the painful parts of running Lynx into one-liners: `lynx manager init` is an interactive wizard that builds your `config.json` step-by-step (and writes the AI-client rules file for you), `lynx manager doctor` runs a full diagnostic (HF cache, drift, paths, extras, disk space), `lynx manager install` handles pip extras + explicit model downloads, and `lynx manager ui` opens a local web panel (FastAPI + HTMX + bundled Tailwind, `127.0.0.1` only) for dashboard, config editor, search playground, build trigger and integration snippets. No CDN, no cloud — fully offline like everything else in Lynx.

### 💡 See the Difference

**Scenario 1: You're using a game engine (like Unity)**
* 🔴 **Without Lynx:** You ask "How do I spawn a particle effect?" The AI suggests deprecated APIs from 3 years ago because its training data is old.
* 🟢 **With Lynx:** The AI transparently searches your *local, up-to-date Unity documentation*, finds the new particle system API, and writes perfectly working code on the first try.

**Scenario 2: Working on a massive enterprise codebase**
* 🔴 **Without Lynx:** "Where is the payment processing logic?" The AI starts randomly reading files, wasting tokens and your time.
* 🟢 **With Lynx:** The AI uses the Lynx tool to semantically search your code. It instantly finds `PaymentGateway.cs` deep in a folder it didn't know existed, reads the relevant chunks, and answers your question.

**Scenario 3: Refactoring a critical interface**
* 🔴 **Without Lynx:** "I want to rename `IDamageable.ApplyDamage`. What breaks?" The AI grep-searches the name and misses callers that pass it through delegates, callbacks, or polymorphic dispatch.
* 🟢 **With Lynx (graph layer enabled):** The AI calls `get_callers_myproject("ApplyDamage")` and `get_subclasses_myproject("IDamageable")`. It receives the complete dependency graph — every caller, every implementation, with file path + line number for each — and proposes a safe refactor with the full blast-radius in hand.

**Scenario 4: Working from a 300-page hardware manual (PDF)**
* 🔴 **Without Lynx:** "What's the I2C timing spec for this chip?" The AI either guesses or asks you to paste the relevant page. You scroll through 300 pages looking for "I²C".
* 🟢 **With Lynx (PDF source enabled):** The AI calls `search_manuals("I2C timing parameters")` and gets the exact paragraphs back with citation `Datasheet.pdf p.142`. Works for any vendor PDF (manuals, datasheets, RFCs, normative documents) — born-digital PDFs only, scanned ones are auto-skipped.

### ✨ Why you'll love it
* **🔒 100% Private & Local:** No code or queries ever leave your machine. No cloud, no API keys, no monthly fees.
* **🤝 Universal:** Works out-of-the-box with Cursor, Claude Code, Google Antigravity, Continue.dev, Aider, and any MCP-compliant client.
* **🧠 Smart:** AST-aware chunking + hybrid retrieval. Optional graph layer adds call/inheritance/import edges so the AI can reason about *structure*, not just similarity.
* **🌍 13 languages, one tool:** C#, Python, TypeScript/TSX, JavaScript, C/C++, Go, Rust, Java, Ruby, PHP, Kotlin, Swift — all parsed via tree-sitter (no LLM in the indexing pipeline).
* **⚡ Live updates:** A file watcher keeps both the search index and the knowledge graph in sync as you edit (~2s after save).

*(Ready to get technical? Read on to see [what this does](#what-this-does) and how it works under the hood 👇)*

> **License:** [Apache 2.0](LICENSE)

---

## Table of contents

1. [What this does](#what-this-does)
2. [Why it exists](#why-it-exists)
3. [How it works (in 30 seconds)](#how-it-works-in-30-seconds)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [LynxManager — guided setup, web UI, diagnostics (new in v0.9)](#lynxmanager--guided-setup-web-ui-diagnostics-new-in-v09)
7. [Configuration](#configuration)
8. [Migrating from v1 (single-source) configs](#migrating-from-v1-single-source-configs)
9. [Build the index for the first time](#build-the-index-for-the-first-time)
10. [Multi-source: indexing code AND library documentation](#multi-source-indexing-code-and-library-documentation)
11. [Webdoc sources](#webdoc-sources)
12. [PDF sources](#pdf-sources)
13. [Connect it to your AI client](#connect-it-to-your-ai-client)
    - [Claude Code (CLI)](#claude-code-cli)
    - [Claude Code extension for VS Code](#claude-code-extension-for-vs-code)
    - [Google Antigravity](#google-antigravity)
    - [Cursor](#cursor)
    - [Continue.dev / Aider / other MCP-compliant clients](#continuedev--aider--other-mcp-compliant-clients)
14. [Command-line interface](#command-line-interface)
15. [Verify the integration works](#verify-the-integration-works)
16. [The MCP tools you get](#the-mcp-tools-you-get)
17. [Get the most out of it: AI integration rules](#get-the-most-out-of-it-ai-integration-rules)
18. [Keeping the index up to date](#keeping-the-index-up-to-date)
19. [AST-aware chunking](#ast-aware-chunking)
20. [Hybrid retrieval](#hybrid-retrieval)
21. [Reranking (opt-in)](#reranking-opt-in)
22. [Graph layer (opt-in)](#graph-layer-opt-in)
23. [Config drift detection](#config-drift-detection)
24. [Architecture](#architecture)
25. [Troubleshooting](#troubleshooting)
26. [Contributing](#contributing)
27. [Privacy guarantees](#privacy-guarantees)

---

## What this does

When an AI assistant has to answer questions about a codebase larger than its
context window, it has two options: read files almost at random hoping to find
what it needs, or ask you. Neither is great.

This server fixes that. You configure N **sources** in `config.json` (a
codebase, a folder of library docs, a vendor reference dump, ...). For
each source the server auto-generates the right set of MCP tools at boot.

### Always-on: semantic + lexical search

For every source, two search tools:

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

### Opt-in: structural understanding (graph layer)

Search tells you *where* something is. The optional **graph layer** tells
you *how things are connected*. Add `graph: { enabled: true }` to a
codebase source in `config.json` and 9 extra tools per source come online:

| Graph tool per source `<name>` | What it answers |
|---|---|
| `get_callers_<name>(symbol)` | "Who calls X?" |
| `get_callees_<name>(symbol)` | "What does X call?" |
| `get_subclasses_<name>(symbol)` | "What concrete types extend / implement X?" |
| `get_superclasses_<name>(symbol)` | "What does X inherit from?" |
| `get_imports_<name>(file_or_symbol)` | "What does this file depend on?" |
| `get_neighbors_<name>(symbol, ...)` | "Show me everything around X" (BFS, optional relation filter) |
| `shortest_path_<name>(source, target)` | "How does A reach B in the call graph?" |
| `architectural_overview_<name>()` | "What are the main abstractions?" (god nodes + community detection) |
| `surprising_connections_<name>()` | Bridge edges between distant clusters (god-class antipatterns) |

Plus `graph_status_<name>()` for diagnostics. None of these are registered
when the flag is off — backward-compatible by default. See the [Graph
layer](#graph-layer-opt-in) section for how cross-file resolution,
inheritance edges, and persistence work.

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

**Two layers run side-by-side on every codebase source:**

```
Search layer (always on):
  Your code  --> chunked --> embedded (BGE-small, on CPU)  --> ChromaDB (local file)
                                                                     |
  Your question  --> embedded --> top-K cosine similarity  ----------+
                                                                     |
                                                                     v
                                       Top-K relevant code snippets
                                       returned to the AI client via MCP

Graph layer (opt-in, `graph: { enabled: true }`):
  Your code  --> tree-sitter walk --> nodes (classes, functions)
                                  --> edges (calls, inherits, imports, contains)
                                                                     |
  Your question (e.g. "who calls X?")  ------------------------------+
                                                                     |
                                                                     v
                                       NetworkX query results
                                       (callers, callees, subclasses, paths, ...)
```

Both layers parse the source files via the **same tree-sitter parsers**
(13 languages, sharing the parser cache), and both are kept in sync by
the same file watcher (~2s after each save). The graph layer is
backward-compatible: leave it off and nothing changes.

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

### Recommended: a one-line install with `pipx` or `uv tool`

If you just want to **use** Lynx (not hack on it), install it as a
standalone CLI tool. Both `pipx` and `uv tool` create an isolated
virtualenv per tool and drop the `lynx` command **on your system
PATH** so you can call it from any folder, in any shell, on any
platform — no `source .venv/bin/activate`, no `python -m`, no
absolute paths.

```bash
# pipx — the established "install a Python CLI globally" tool.
# macOS / Linux: `brew install pipx` (or `python -m pip install --user pipx`)
# Windows:       `py -m pip install --user pipx`
pipx install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git

# OR — uv (faster, same end result):
# Install uv first: see https://docs.astral.sh/uv/getting-started/installation/
uv tool install git+https://github.com/lorenzo-cambiaghi/LynxMCP.git
```

After either of those, `lynx` works from anywhere:

```bash
lynx --version
lynx manager ui     # opens the web panel
lynx manager init   # interactive setup wizard
```

(When LynxMCP lands on PyPI the source URL becomes just `lynx`, e.g.
`pipx install lynx`.)

To upgrade later: `pipx upgrade lynx` (or `uv tool upgrade lynx`).
To uninstall: `pipx uninstall lynx` (or `uv tool uninstall lynx`).

### Development install (editable from a clone)

Use this only if you want to **modify** Lynx's source code and have
your changes take effect immediately:

```bash
git clone https://github.com/lorenzo-cambiaghi/LynxMCP.git
cd LynxMCP

# pick ONE of these:
python -m venv .venv && source .venv/bin/activate     # macOS / Linux
python -m venv .venv && .venv\Scripts\activate        # Windows
# OR with uv:  uv venv && source .venv/bin/activate

pip install -e .     # or: uv pip install -e .
```

In this mode the `lynx` command exists at `.venv/bin/lynx` (or
`.venv\Scripts\lynx.exe` on Windows) and is on PATH **only while the
venv is activated**. If you forget to activate, you'll see
`command not found: lynx`. Three ways out:

1. Activate the venv: `source .venv/bin/activate` (then plain `lynx ...` works).
2. Use the full path: `.venv/bin/lynx manager ui`.
3. Use the explicit module form: `.venv/bin/python -m lynx manager ui`.

For everyday use, `pipx`/`uv tool` (above) is dramatically nicer —
it sidesteps the activate-the-venv ritual entirely.

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

## LynxManager — guided setup, web UI, diagnostics (new in v0.9)

If you'd rather not hand-edit a JSON file and read through the
configuration section that follows, `lynx manager` covers the entire
lifecycle in four commands. **This is the recommended on-ramp for new
users.** Everything stays local — no telemetry, no cloud, same privacy
guarantees as the rest of Lynx.

### `lynx manager init` — interactive setup wizard

```bash
lynx manager init
```

Walks you through every decision step by step: storage path, embedding
model (defaults to `BAAI/bge-small-en-v1.5`), search mode, optional
reranker, then a "add another source?" loop for codebase / webdoc / PDF
sources. For codebases it auto-detects file extensions, watcher
settings and git integration based on what's actually in the folder.
At the end it:

- writes a valid `config.json`,
- offers to download the embedding model (one-time `snapshot_download`),
- generates an AI-client rules file (`CLAUDE.md` / `AGENTS.md` /
  `.cursor/rules/lynx.md` depending on the client you pick),
- prints the exact MCP snippet to paste into your client.

Non-interactive mode (`--non-interactive`) is supported for scripting.

### `lynx manager doctor` — full diagnostic

```bash
lynx manager doctor
```

Runs six checks in parallel and prints a colour-coded report: Python
version, HuggingFace cache (per declared model), config file validity,
per-source state (path exists, ChromaDB collection healthy, drift
status, watcher path still there), optional extras (which are installed
vs available), and free disk space at the storage path. Exit code is
`0` (all green), `1` (warnings), or `2` (errors) so you can wire it
into CI / pre-deploy scripts.

### `lynx manager install` — extras + model download

```bash
# What extras are available / installed?
lynx manager install --list

# Install an extra (uses `pip install lynx[<name>]` under the hood)
lynx manager install pdf-fast

# Download the configured embedding model explicitly
lynx manager install --model

# ...and the reranker model too
lynx manager install --model --with-reranker
```

The `--model` path temporarily bypasses `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE` so you can force a download even when
runtime offline mode is on. Useful when prepping an air-gapped machine.

### `lynx manager ui` — local web panel

A small web app that runs **on your own machine** and lets you click
through everything Lynx can do — no command-line needed past the launch.
Useful when you've just set Lynx up and want to verify it actually works
before wiring it into your AI client.

#### Step 1 — launch it

**The fastest way: double-click the launcher in the repo.**

- **macOS**: double-click `LynxManager.command` in Finder. (First time
  only: right-click → Open → Open to clear the Gatekeeper warning, since
  the file isn't notarized.)
- **Windows**: double-click `LynxManager.bat` in Explorer. (First time
  only: SmartScreen may prompt — click "More info" → "Run anyway".)

A console window opens, the UI boots, and your browser is pointed at
`http://127.0.0.1:8765`. To stop it, press `Ctrl+C` in the console
window (or just close the window).

The launcher detects whichever install style you used (`pipx`, `uv tool`,
local `.venv`, or `python -m lynx`) and runs the matching one — so the
same file works for end-users and developers.

**Or, from a terminal in the folder where your `config.json` lives:**

```bash
lynx manager ui
```

When you don't pass `--config`, Lynx uses the same resolution chain as
`lynx serve`: explicit flag → `RAG_CONFIG_PATH` env var → `./config.json`
in the current working directory. On Windows that means PowerShell /
cmd.exe / Git Bash all need to be **launched from the folder that
contains `config.json`** (or you can pass the full path):

```bash
# Windows PowerShell / cmd.exe / Git Bash — point at it explicitly if
# you don't want to cd first:
python -m lynx manager ui --config C:\Users\you\projects\myrepo\config.json
```

> If your config sits elsewhere, point at it explicitly:
> `lynx manager ui --config /full/path/to/config.json`
>
> The terminal output tells you exactly which config file was picked
> up AND when the embedding model is finished loading — useful for
> catching "I thought I was in the right folder" mistakes and for
> understanding the brief startup pause on first launch:
> ```text
> [ui] using config: /home/you/myrepo/config.json
> [ui] loading embedding model + opening source collections (first launch can take 30s)...
> [ui] manager ready in 4.2s (2 sources).
> 🐱 Lynx UI ready at http://127.0.0.1:8765
> ```
> The browser only opens **after** the manager is ready — so the page
> loads instantly instead of hanging on a white screen while
> `BAAI/bge-small-en-v1.5` warms up in the background.

You'll see something like this in the terminal:

```text
🐱 Lynx UI ready at http://127.0.0.1:8765
   Press Ctrl+C to stop.
```

About half a second later **your default browser opens automatically**
on that URL and you land on the LynxManager dashboard.

> **Browser didn't open?** Some setups (SSH sessions, minimal Linux
> desktops, certain WSL configurations) can't launch a browser
> automatically. Just copy the URL — `http://127.0.0.1:8765` — and
> paste it into any browser tab on the same machine. It works the
> same way.
>
> **Port already in use?** Lynx tries the next 10 ports automatically
> (`8766`, `8767`, …) and prints whichever one it landed on. If you
> want a specific port, pass `--port 9000`.
>
> **Want to launch without opening a browser?** Use
> `lynx manager ui --no-browser` (handy when you're remoting in).
>
> **To stop it,** go back to the terminal and press `Ctrl+C`.

The UI listens **only on `127.0.0.1`** — it is not reachable from other
machines on the network. There's no login because there's no need: nobody
else can reach it.

#### Step 2 — a 5-minute tour for a brand-new user

Let's say you've just run `lynx manager init`, picked your codebase folder,
and now have no idea what you actually got. Here's what to do, in order:

1. **Dashboard (the page you land on).** You'll see one card per source.
   Each card shows the chunk count (how many text fragments are indexed),
   the last-update time, and a coloured badge if anything is off (`drift:
   warning` if files changed since the last build, `🔒 locked` if another
   process is using it). The "Health check" box at the top is your
   `lynx manager doctor` result rolled up into one line — green means
   everything's fine. **If a source shows `0 chunks`** you haven't built
   the index yet — see step 4.

2. **Click the source name** (e.g. `myproject`). You're now on the source
   detail page: full status, graph layer stats if you enabled it, and a
   big **Rebuild index** button.

3. **Open the Playground** (left sidebar → 🔎 Playground). Pick your
   source from the dropdown at the top and type a question into the
   Search tab — something like `"how does authentication work?"` or
   `"where do we parse JSON?"`. Hit Search. You'll get back a list of
   code chunks with file path, line range, and a relevance score. **This
   is exactly what your AI client will see** when it calls Lynx — no
   need to wire up an AI client just to find out whether Lynx is
   working.

4. **If you haven't built the index yet**, go back to the source detail
   page (left sidebar → 📚 Sources → click your source) and hit
   **Rebuild index**. A status panel appears and updates itself every
   second. Builds take seconds for small repos, a few minutes for
   bigger ones. When it says ✓ Build complete, return to the Playground
   and try the search again.

5. **Wire it into your AI client** (left sidebar → 🔌 Integrations).
   You'll see a card for Claude Code, Cursor, Antigravity, and generic
   stdio. **Copy the JSON snippet** with the button — it's already
   filled in with *your* Python interpreter and *your* config path —
   and paste it into the file shown on the card (e.g.
   `~/.claude/mcp_settings.json` for Claude Code). Also click
   **Download CLAUDE.md** (or `AGENTS.md` / `lynx.md`) to get an
   auto-generated rules file that teaches your AI when to call Lynx's
   tools. Drop the file in your repo root and restart the AI client.

6. **(Optional) Edit the config** from the Config tab if you want to
   add another source or tweak a setting — the editor validates with
   the exact same loader the CLI uses, and keeps a `.bak` of the
   previous version in case you want to roll back.

That's the whole product in five minutes. Everything else (graph tools,
diff search, multiple sources, reranker) is built on the same surface
— same dashboard, same playground, same lock detection.

#### What each page does (reference)

- **Dashboard** — per-source cards with chunk count, drift severity,
  lock badge (set when another process is writing to the same
  ChromaDB), plus a doctor summary.
- **Sources** — list + per-source detail page with a rebuild button
  that runs `manager.update(src, force=True)` in a daemon thread; the
  HTMX widget self-polls every second until the job is terminal.
  Refuses to start when the SQLite write lock is held by another
  process — guards against the classic "I left `lynx serve` running
  and the build corrupted the DB" footgun.
- **Playground** — tabbed forms for every per-source tool the MCP
  server exposes: hybrid search, `find_definition` / `find_usages` /
  `find_tests_for` / `find_similar`, `architectural_overview` /
  `get_callers` / `get_callees`, and `search_diff`. Faster than
  spinning up a client to validate that a query works.
- **Config** — JSON editor with backup-then-overwrite save and the
  exact same validation the CLI uses (validates to a tempfile before
  touching the real config).
- **Integrations** — per-client cards (Claude Code, Cursor,
  Antigravity, generic stdio) with the MCP JSON snippet pre-populated
  using *your* interpreter + *your* config path, copy-to-clipboard
  button, and one-click download of the generated `CLAUDE.md` /
  `AGENTS.md` / `lynx.md` rules file.

#### Notes on scope

Auth and HTTPS are explicitly out of scope — this is a personal
management tool, not a shared service. Bind is `127.0.0.1` only. To
disable the auto-browser-open use `--no-browser`. To pick a different
port use `--port`. If the port is busy it advances up to 10 slots
before letting the OS assign one.

---

## Configuration

> 💡 **New in v0.9:** prefer `lynx manager init` over editing JSON by
> hand — the wizard validates inputs as you go and writes the rules
> file for your AI client too. The reference below is still authoritative
> for advanced cases (multi-source, custom rerankers, watcher tuning).

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
| `type` | **Required.** `"codebase"` for a local directory of files. Other supported types: `"webdoc"` (see [Webdoc sources](#webdoc-sources)). Future: `"pdf"`. |
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

> **`type: "webdoc"` is now available** (since v0.4): fetch a public docs
> site on demand, dump the extracted main content to a local folder,
> and index it through the same hybrid pipeline as code. See the
> [Webdoc sources](#webdoc-sources) section for the config schema and
> the refresh model. `type: "pdf"` is on the roadmap.

---

## Webdoc sources

Indexes a public documentation site by crawling it from a starting URL.
For libraries that update faster than the model's training cutoff
(Unity, AvaloniaUI, AWS SDKs, any framework with a quarterly release
cadence) this is the difference between the AI suggesting deprecated
APIs and the AI knowing what's there today.

### Configuration

```json
"sources": {
  "unityDoc": {
    "type": "webdoc",
    "url": "https://docs.unity3d.com/Manual/index.html",
    "max_depth": 3,
    "max_pages": 1000,
    "same_origin_only": true,
    "include_url_patterns": ["/Manual/"],
    "exclude_url_patterns": ["/Manual/Obsolete"],
    "request_delay_seconds": 0.5,
    "user_agent": "Lynx-DocFetcher/0.4"
  }
}
```

| Field | What it does |
|---|---|
| `type` | **Required.** `"webdoc"`. |
| `url` | **Required.** Starting URL of the crawl. Must be `http://` or `https://`. |
| `max_depth` | BFS depth limit from the starting URL. Default `3`. |
| `max_pages` | Hard cap on the number of pages fetched. Default `500`. |
| `same_origin_only` | If `true` (default), the crawler stays on the seed URL's hostname. |
| `include_url_patterns` | List of substrings; a URL must contain **at least one** to be saved. The seed URL is always visited (so its links can be discovered) but only added to the dump if it matches. |
| `exclude_url_patterns` | List of substrings; any URL containing one is skipped entirely. |
| `request_delay_seconds` | Polite delay between requests. Default `0.5` (= 2 req/sec). |
| `user_agent` | Optional override. Default identifies as `Lynx-DocFetcher/<version>`. |

### Fetch and refresh

A webdoc source is **never auto-refreshed**: detecting "the upstream
site changed" without re-downloading is unreliable (ETag / Last-Modified
support varies wildly). Refresh is always explicit:

```bash
lynx build --source unityDoc
```

This:
1. Crawls the configured URL (respecting depth, origin, filters)
2. Extracts main content from each HTML page using `trafilatura`
   (drops nav / footer / sidebars)
3. Writes one `.md` file per crawled URL to `storage_path/<source>/_dump/`,
   with a YAML frontmatter block containing the original URL and
   fetched-at timestamp
4. Wipes any old dump first so URLs no longer present upstream don't
   leave stale chunks
5. Triggers a full reindex through the same chunker + embedding pipeline
   used by codebase sources

Run it again any time you want a fresh snapshot — typically after a
library version bump.

### What gets supported, what doesn't

| | Supported | Notes |
|---|---|---|
| Static HTML | ✅ | Sphinx, mkdocs, docusaurus, hand-written sites |
| Server-rendered SSR | ✅ | Anything that ships HTML directly |
| JS-rendered SPAs | ❌ | Would require a headless browser (~200 MB Chromium dep) — explicitly out of scope. ~10% of doc sites; for the 90% mainstream this isn't a problem. |
| Auth-gated docs | ❌ | The crawler sends a plain UA, no cookie / token support. PRs welcome if you need it. |
| PDFs / images | ❌ | The crawler skips any URL whose content-type isn't HTML. PDF support is planned as a separate `type: "pdf"` source. |
| robots.txt | ⚠️ Not consulted | Crawl is rate-limited and identifies itself, but currently doesn't parse robots.txt. Use `request_delay_seconds` to stay polite. |
| Partial / JS-rendered TOCs | ⚠️ Partial | If the site's index page lists its sub-pages via JavaScript (Unity 6 ScriptReference is one example), the crawler sees the static HTML only — no link discovery happens past the seed. Workaround: point at a non-index page, or use a sitemap URL as the seed. |

> **Windows TLS note.** Lynx ships with `truststore` as a dependency so the
> crawler reads the OS certificate store (Windows / macOS keychain / Linux
> ca-certificates) instead of certifi's Mozilla bundle. Without this, on a
> Windows machine inside a corporate / antivirus TLS proxy, the crawler
> would fail every fetch with `CERTIFICATE_VERIFY_FAILED`. If you've added
> custom trust to your Windows store, the crawler will respect it.

### Multi-source example: code + library docs together

```json
"sources": {
  "myproject": {
    "type": "codebase",
    "path": "C:/projects/mygame/Assets/Scripts",
    "supported_extensions": [".cs", ".md"]
  },
  "unityDoc": {
    "type": "webdoc",
    "url": "https://docs.unity3d.com/Manual/",
    "include_url_patterns": ["/Manual/"],
    "max_pages": 1500
  },
  "avalonia": {
    "type": "webdoc",
    "url": "https://docs.avaloniaui.net/",
    "max_pages": 800
  }
}
```

Boots the server with `search_myproject`, `search_unityDoc`,
`search_avalonia` (plus the corresponding `deep_search_*`), each with a
docstring that names the source's type — your AI client picks based on
the question.

---

## PDF sources

`type: "pdf"` indexes a folder of `.pdf` files (manuals, RFCs, normative
documents, white-papers, technical books). Lynx extracts text **page by
page**, writes one Markdown file per page under `_dump/`, then indexes
those via the same RAG pipeline used for codebases. Citations come back
as `User Manual.pdf p.42` thanks to YAML frontmatter on each page dump.

### Why a PDF source instead of converting upstream

Doing `pdftotext` + putting the result in a codebase source works, but
you lose page numbers, you have to re-convert manually whenever a PDF
changes, and you get a single giant blob of text per document. The PDF
source preserves page granularity, supports SHA-incremental refresh, and
auto-detects + skips files that aren't suitable.

### Configuration

```json
"sources": {
  "manuals": {
    "type": "pdf",
    "path": "C:/path/to/your/pdfs",
    "recursive": true,
    "file_glob": "**/*.pdf",
    "watcher": { "enabled": false, "debounce_seconds": 5.0 },
    "extractor": {
      "backend": "auto",
      "max_file_mb": 100,
      "max_pages_per_file": 5000,
      "skip_password_protected": true,
      "skip_if_text_empty": true
    }
  }
}
```

| Field | Default | What it does |
|---|---|---|
| `path` | **required** | Absolute path of the folder containing the .pdf files. |
| `recursive` | `true` | Walk sub-directories. Set `false` to limit to the top-level folder. |
| `file_glob` | `"**/*.pdf"` | `pathlib.Path.glob` pattern relative to `path`. Use e.g. `"datasheets/*.pdf"` to scope. |
| `watcher.enabled` | `false` | **OFF by default** because re-extracting a PDF is costly (10-30s) and PDFs change rarely. When `true`, watchdog observes the source folder; new / modified / deleted PDFs trigger an incremental re-extract. |
| `extractor.backend` | `"auto"` | `"auto"` prefers `pymupdf` if installed, else `pypdf`. Force one with `"pypdf"` or `"pymupdf"`. |
| `extractor.max_file_mb` | `100` | Skip PDFs larger than this. RAM-safety: pypdf loads the whole file at ~3-5× disk size. Raise carefully. |
| `extractor.max_pages_per_file` | `5000` | Skip pathological documents (50k-page legal dumps generate millions of chunks). |
| `extractor.skip_password_protected` | `true` | When `true` (only valid value today), encrypted PDFs are skipped with `status="skipped_password"` in the state file. |
| `extractor.skip_if_text_empty` | `true` | When `true`, PDFs with less than 100 total extracted characters are skipped as "probably scanned" (Lynx has no OCR). |

### What is supported

| PDF type | Result | Notes |
|---|---|---|
| **Born-digital** (Word/LaTeX/web → PDF) | ✅ Works | ~90% of real-world technical PDFs. |
| **Multi-column papers** (academic / IEEE) | ⚠️ OK with `pypdf`, **good with `pymupdf`** | Reading order can be wrong with the default. Install `lynx[pdf-fast]` for these. |
| **Tables and forms** | ✅ Extracted as flattened text | Searchable but structure not preserved. |
| **Password-protected** | ❌ Skipped | Status `skipped_password`. Decryption with a password store is out of scope (for now). |
| **Scanned PDFs** (no text layer) | ❌ Skipped | Status `skipped_empty`. **No OCR support** — Tesseract would require system binaries + ~200 MB of models, incompatible with the "100% local zero-deps" promise. Convert externally if you need them. |
| **Encrypted with DRM** | ⚠️ Sometimes works | DRM usually only blocks printing, not text extraction. Case-by-case. |
| **Files > 100 MB** | ❌ Skipped | Raise `extractor.max_file_mb` if you really need a 500 MB PDF. |

### Faster / better extraction (opt-in)

```bash
pip install lynx[pdf-fast]      # adds pymupdf (AGPL)
```

Then either let `extractor.backend: "auto"` pick it up, or pin it
explicitly with `"pymupdf"`. PyMuPDF is roughly **4× faster** than pypdf
and noticeably better on multi-column layouts. We keep it opt-in because
its AGPL license is incompatible with Lynx's Apache 2.0 distribution.

### Re-extracting

Same UX as webdoc: explicit by default.

```bash
# Add / modify / remove PDFs in the folder, then:
lynx build --source manuals
```

When `watcher.enabled=true`, the same logic runs automatically on every
file-system event (debounced 5s by default). Look in
`<storage>/manuals/_extract_state.json` for the SHA cache + per-file
status (`ok`, `skipped_*`, `error`).

### What you see in `lynx status`

```
$ lynx status --source manuals
=== Source: manuals (type: pdf) ===
  pdf_count:            42
  skipped_count:         3   ← 2 password-protected, 1 scanned
  total_pages_extracted: 8451
  chunk_count:          9234
  extractor_backend:    pypdf
  last_extract_at:      2026-05-21T10:00:00
```

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

The same `lynx` command exposes seven subcommands. Useful for debugging
the index, scripting, or just querying the codebase without opening an AI
assistant.

```text
lynx [--version] [-h] COMMAND ...

  serve         Run the MCP server (default if no command is given)
  build         Force a full rebuild of a source's index (search + graph)
  search        Run an ad-hoc search query (no MCP client needed)
  status        Show RAG status: git state, last update, config drift
  list-sources  Enumerate configured sources
  graph         Manage the per-source knowledge graph layer
                  graph build   Rebuild graph for a source
                  graph status  Show node / edge counts, by language / kind
  manager       Setup / install / diagnose / web UI (new in v0.9)
                  manager init     Interactive config wizard
                  manager doctor   Full diagnostic report
                  manager install  Extras + HF model download
                  manager ui       Local web panel (FastAPI + HTMX)
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

### `graph`

Manage the per-source knowledge graph layer (only meaningful when the
source has `graph: { enabled: true }` in `config.json`). Two
sub-commands:

```bash
# Rebuild the graph for a single source. --force wipes state first;
# without --force it does an SHA-incremental rebuild.
lynx graph build --source myproject
lynx graph build --source myproject --force

# Show counts and metadata. Without --source, shows every source that
# has the graph layer enabled.
lynx graph status
lynx graph status --source myproject
```

`lynx graph status` output:

```
=== Graph status: myproject ===
  schema_version:    2
  nodes:             8421
  edges:             14732
  files_indexed:     1284
  raw_calls_pending: 562
  last_update:       2026-05-21T09:14:02
  last_full_rebuild: 2026-05-20T18:02:11
  by_language:       {'c_sharp': 7912, 'python': 484, 'external': 25}
  by_kind:           {'function': 6201, 'class': 1992, 'file': 203, 'external': 25}
  by_relation:       {'contains': 6201, 'calls': 7488, 'inherits': 612, 'imports': 431}
```

Day-to-day you don't need this — the file watcher keeps the graph live
the same way it does for the search index, and `lynx build` rebuilds
both. Use `lynx graph build --force` after a graph schema bump or when
you suspect stale edges.

---

## Verify the integration works

After connecting the server to your client, ask the AI assistant:

> *"Use the search_codebase tool to find anything related to authentication
> in this codebase."*

You should see it invoke the appropriate `search_<name>` tool for your
source and return relevant snippets.

To verify directly without an AI client, run any of the unit tests
(they need no config, only the venv):

```bash
# Self-contained unit tests (no real codebase needed, ~5 seconds each)
python -m tests.test_tree_sitter         # AST chunker, 13 languages
python -m tests.test_graph_extractor     # single-file graph extraction
python -m tests.test_graph_builder       # build + persist + SHA cache + cross-file
python -m tests.test_graph_analyzer      # god_nodes / communities / surprising
python -m tests.test_graph_query         # get_callers / get_callees / shortest_path / ...
python -m tests.test_graph_integration   # config → backend → manager pass-through
python -m tests.test_graph_mcp_tools     # MCP tool registration end-to-end
python -m tests.test_pdf_extractor       # PDF extractor on synthetic PDFs (pypdf)
python -m tests.test_pdf_dump            # per-page Markdown writer + state cache
python -m tests.test_pdf_config          # type=pdf source validator
python -m tests.test_pdf_backend         # PdfBackend integration (stubbed CodebaseRAG)
```

Or the end-to-end smoke tests (these need a real `config.json` pointing
at a real codebase):

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
auto-registers **two search tools** at boot. If the source also has
`graph: { enabled: true }`, **10 graph tools** are registered too. Add
three sources with graph enabled → 36 per-source tools. Plus the five
globals. The AI client sees the whole set in its tool list and picks
based on the docstrings.

### Per-source search tools (always on, auto-generated)

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

### Graph tools (opt-in, per source)

When a codebase source has `graph: { enabled: true }`, the server
auto-registers **10 extra tools** for it (9 query tools + 1 status tool).
Names follow the same `<verb>_<source>` convention as the search tools so
the AI client picks the right one from its tool list:

| Tool | Answers | When to call |
|---|---|---|
| `get_callers_<name>(symbol, limit?)` | "Who calls X?" | Refactoring: blast-radius of a rename / signature change. |
| `get_callees_<name>(symbol, limit?)` | "What does X call?" | Understanding what a function actually does. |
| `get_subclasses_<name>(symbol, limit?)` | "What concrete types extend / implement X?" | Adding an interface method: which implementations need updating. |
| `get_superclasses_<name>(symbol, limit?)` | "What does X inherit from?" | Tracing where a method or field comes from. |
| `get_imports_<name>(file_or_symbol, limit?)` | "What does this file depend on?" | "Are we already using package Y?" |
| `get_neighbors_<name>(symbol, relation_filter?, depth?, limit?)` | Everything around X within N hops | When you don't yet know which direction matters. |
| `shortest_path_<name>(source, target, max_hops?)` | "How does A reach B?" | "Why does the checkout flow end up touching the audit logger?" |
| `architectural_overview_<name>(top_n_gods?, min_community_size?)` | God-nodes + communities + stats | Start an unfamiliar session: get a high-level map first. |
| `surprising_connections_<name>(top_n?)` | Top bridge edges by edge betweenness | Spot god-class / hidden-coupling antipatterns. |

Plus `graph_status_<name>()` for diagnostics (node / edge counts, by
language, by relation, last update timestamp).

Symbols are matched **fuzzy** (case-insensitive). Exact-leaf match wins
when one exists; otherwise substring matches are returned. Each result
includes `file`, `start_line`, `end_line` and a `confidence` flag
(`extracted` for intra-file, `resolved` / `ambiguous` for cross-file)
so the AI client can cite the source precisely.

Sample call inside the AI client:

```
get_callers_myproject("ApplyDamage")
→ Callers of 'ApplyDamage' in 'myproject':
    • HealthSystem.OnHit      [function] @ src/Health/HealthSystem.cs:L88-103
        --calls [resolved]--> IDamageable.ApplyDamage  [function] @ src/Damage/IDamageable.cs:L7
          at src/Health/HealthSystem.cs:L94
    • BulletImpact.Process    [function] @ src/Combat/BulletImpact.cs:L42-61
        --calls [resolved]--> IDamageable.ApplyDamage  [function] @ src/Damage/IDamageable.cs:L7
          at src/Combat/BulletImpact.cs:L55
```

See [Graph layer (opt-in)](#graph-layer-opt-in) for the build pipeline,
cross-file resolution policy, inheritance edges, and costs.

### Combined tools (always-on for codebase sources, new in v0.8)

Every `codebase` source automatically gets **4 combined tools** that
mix graph + search to answer questions a single tool can't:

| Tool | Answers | Uses graph? |
|---|---|---|
| `find_definition_<src>(symbol)` | "Where is X defined?" | Yes when enabled (precise file+line from AST); BM25 fallback otherwise. |
| `find_usages_<src>(symbol)` | "Who uses X?" — calls AND non-call refs (typeof, generics, decorators, doc mentions). | Yes when enabled (`get_callers` for structural callers); always also runs textual search to catch the rest. Deduped. |
| `find_tests_for_<src>(symbol, test_path_pattern?)` | "Are there tests for X?" | No graph; search + regex filter on standard test paths (`/tests/`, `_test.py`, `.spec.js`, `*Test.cs`, ...). |
| `find_similar_<src>(snippet, top_k?)` | "Is there code similar to this?" — semantic match on the snippet's embedding. | No graph; pure dense (BM25 ignored). Filters out byte-identical chunks. |

And one MORE tool, conditional on `git_integration.enabled=true`:

| Tool | Answers |
|---|---|
| `search_diff_<src>(query, base?, top_k?)` | "Search only in files I changed vs `main` (or whatever `base`)." Auto-detects `main` / `master` / `develop`; pass `base=` to override. Returns `{base, modified_files, hits}`. Killer for code review. |

Sample call inside the AI client:

```
find_definition_myproject("PaymentGateway")
→ Definitions of 'PaymentGateway':
  • PaymentGateway  [class]  @ src/Billing/PaymentGateway.cs:L12-89  (graph)

find_usages_myproject("ApplyDamage")
→ Usages of 'ApplyDamage':
  • HealthSystem.OnHit  @ src/Health/HealthSystem.cs:L94  (graph [calls])
  • bullet_handler      @ src/Combat/BulletImpact.cs:L42  (search)
  ...

search_diff_myproject("validation logic")
→ search_diff in 'myproject' vs base 'main':
  Modified files (3): src/checkout/discount.py, src/checkout/order.py, src/checkout/tests/test_order.py
  Hits (2):
    • src/checkout/discount.py:L42-67  apply_discount  score=0.0245
    • src/checkout/order.py:L120-138  validate_total  score=0.0198
```

Each result carries a `source` tag (`graph` / `search_bm25` / `search` /
`search_dense` / `search+test_filter`) so the AI client can communicate
confidence: a "graph" result is from the AST (precise); "search_bm25" is
a best-effort guess when the graph layer isn't enabled.

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

## When to use the graph layer instead of search

If the source `<myproject>` has `graph: { enabled: true }` in
`config.json`, you also have 9 structural tools available
(`get_callers_<myproject>`, `get_callees_<myproject>`,
`get_subclasses_<myproject>`, `get_superclasses_<myproject>`,
`get_imports_<myproject>`, `get_neighbors_<myproject>`,
`shortest_path_<myproject>`, `architectural_overview_<myproject>`,
`surprising_connections_<myproject>`).

The rule of thumb: **search finds files; the graph finds relationships.**

- Use **search** to answer "WHERE is the code that does X?".
- Use the **graph** to answer "WHO calls X?", "WHAT does X depend on?",
  "WHICH classes implement X?", "HOW does A reach B?".

Specifically:

- Before a refactor of `Foo.bar`: call `get_callers_<myproject>("bar")`
  to see the blast-radius before proposing changes.
- Before adding a method to an interface: call
  `get_subclasses_<myproject>("IFoo")` to find every implementation
  that will need updating.
- When starting an unfamiliar session: call
  `architectural_overview_<myproject>()` once to get god-nodes +
  communities + stats in one shot.
- When the user asks "what depends on this?" or "what uses this?": call
  the graph, NOT a fuzzy text search.

The graph is best-effort, not whole-program type inference. Each result
carries a `confidence` flag:
- `extracted` — intra-file, deterministic.
- `resolved` — single cross-file candidate, high confidence.
- `ambiguous` — multiple candidates with the same name; surface all of
  them to the user.

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

- **Precise location**: file + line range (e.g. `Container.cs:L555-580`)
  plus the qualified symbol name from the result header (e.g.
  `MyProject.DI.Container.TryInjectInterfaceReference`). Both come for
  free from the AST chunker — surface them so the user can jump straight
  to the code instead of grepping.
- One-line summary of what the chunk does.
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

## AST-aware chunking

Lynx chunks code at **syntactic boundaries** (one chunk per function /
method / class) instead of arbitrary token windows that cut across function
signatures. The result: embeddings represent meaningful units, BM25
identifier statistics aren't diluted by neighboring code, and search hits
come back as complete functions the AI can immediately reason about.

The chunker uses **tree-sitter** with per-language grammars bundled in
their own pip packages — no runtime downloads, no network calls (the
"100% local" guarantee holds). Supported languages out of the box:

| Extension | Language | Chunked on |
|---|---|---|
| `.cs` | C# | methods, constructors, properties, indexers, operators, delegates, events, enums (qualified by namespace + class) |
| `.py` | Python | functions, methods, decorated definitions (qualified by class) |
| `.ts`, `.tsx` | TypeScript | functions, methods, interfaces, type aliases, enums |
| `.js`, `.jsx`, `.mjs` | JavaScript | functions, methods, generators |
| `.cpp`, `.hpp`, `.cc`, `.cxx`, `.hxx` | C++ | functions, templates (inside namespace / class / struct) |
| `.c`, `.h` | C | functions, structs, enums, typedefs |
| `.go` | Go | functions, methods, type declarations |
| `.rs` | Rust | functions, structs, enums, typedefs (inside impl / mod / trait) |
| `.java` | Java | methods, constructors (inside class / interface / enum) |
| `.rb` | Ruby | methods, singleton methods (qualified by module / class) |
| `.php` | PHP | methods, free functions (qualified by namespace / class / interface / trait) |
| `.kt`, `.kts` | Kotlin | functions, secondary constructors (qualified by class / object) |
| `.swift` | Swift | functions, init, deinit, protocol-method signatures, property declarations (inside class / struct / extension / enum / protocol) |

For anything else (`.md`, `.txt`, `.json`, `.yaml`, `.hlsl`, `.shader`,
`.compute`, `.cginc`, …) Lynx falls back to a sentence-window splitter —
same behavior as v0.2.

**Chunk metadata** every retrieval result carries the AST context, so
the AI can cite precisely:

| Metadata field | Example | Notes |
|---|---|---|
| `symbol_name` | `MyFramework.Damage.HealthSystem.ApplyDamage` | Fully qualified; namespace + class + method |
| `symbol_kind` | `method_declaration` | The AST node type (or `text_window` for fallback) |
| `start_line`, `end_line` | `42`, `58` | 1-based inclusive |
| `language` | `c_sharp` | One of the supported keys above, or `text` for fallback |
| `chunker` | `tree_sitter` \| `sentence_splitter` | Which path produced this chunk |

**Oversized chunks** (e.g. a single 2000-line auto-generated method) get
split with the sentence-window splitter so no chunk exceeds ~8000
characters. Split pieces inherit the parent's symbol name with a `#partN`
suffix.

**`chunker_version` is in the drift snapshot.** Bumping the chunker logic
in a way that changes boundaries or metadata triggers a CRITICAL drift on
next start. The drift message lists `chunker_version: N -> M`; running
`lynx build --source <name>` clears it. Upgrading from any earlier
release to v0.5+ (where Ruby/PHP/Kotlin/Swift were added and
`CHUNKER_VERSION` bumped 3 → 4) will flag this automatically.

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

## Reranking (opt-in)

Hybrid RRF fusion is fast and works well on average — but it ranks
purely on *rank position* in each retriever (dense + sparse). It never
actually inspects whether a chunk's content answers the query.

A **cross-encoder reranker** takes the top-N RRF candidates and feeds
each `(query, chunk_content)` pair through a small transformer model
that DOES inspect both sides, producing a content-aware relevance
score. This typically improves **precision@1 by 20-30% on ambiguous
queries** with a one-time ~80MB model download and ~50ms of extra
latency per query.

### Enable it

Add to `config.json`:

```json
"search": {
  "reranker": {
    "enabled": true,
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_n_before_rerank": 30
  }
}
```

The model downloads from HuggingFace on the first query that needs it.
After that, it's fully offline (same `HF_HUB_OFFLINE=1` policy as the
embedding model).

### How it integrates

```
query → dense + BM25 → RRF fusion → top_n candidates
                                        ↓
                                  cross-encoder rerank
                                        ↓
                                  top_k returned
```

Each result keeps every field it had before, plus:
- `original_score` — the pre-rerank RRF score (for debugging /
  comparison)
- `reranked: true` — flag so the AI knows the score is on the
  cross-encoder scale, not RRF

### Cost

| Item | Cost |
|---|---|
| First query after server boot | ~2-3 s (one-time model load) |
| Every subsequent query | +30-100 ms on CPU |
| RAM | ~150 MB while loaded |
| Disk (HF cache) | ~80 MB |
| First-ever launch | ~80 MB download from HF |

Disable by setting `enabled: false` (or omit the block). Existing
indexes are unaffected — reranker doesn't touch ChromaDB.

### Other models

Swap `model_name` for a different cross-encoder. Trade-offs:

| Model | Size | Quality | Latency |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (default) | ~80 MB | Good | ~50 ms / 30 docs |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~140 MB | Better | ~90 ms / 30 docs |
| `BAAI/bge-reranker-base` | ~280 MB | Best general-purpose | ~200 ms / 30 docs |
| `BAAI/bge-reranker-large` | ~1.1 GB | Strongest | ~700 ms / 30 docs |

For code search specifically, the MiniLM defaults are surprisingly
competitive — try them first.

---

## Graph layer (opt-in)

Hybrid search excels at **"find me the code that does X"**. It struggles
with **structural questions** — *"who calls `IDamageable`?", "what does
`PaymentGateway` depend on?", "what's the shortest path from
`CheckoutController` to `PaymentGateway`?"* — because those are graph
queries, not similarity queries.

The graph layer answers them. It's an **opt-in** companion to the vector
store: when enabled on a `codebase` source, Lynx parses the same files
already chunked for retrieval and builds a small knowledge graph of
classes, functions, imports, calls, and inheritance relationships
(`extends` / `implements`). The graph lives in memory and persists to
JSON next to the ChromaDB collection.

### Enable it

Add `graph: { enabled: true }` to a codebase source in `config.json`:

```json
"sources": {
  "myproject": {
    "type": "codebase",
    "path": "C:/path/to/your/code",
    "supported_extensions": [".py", ".cs", ".ts"],
    "watcher": { "enabled": true },
    "graph": { "enabled": true }
  }
}
```

That's it. The graph is built automatically on the next `lynx build` (or
`lynx graph build --source myproject` for graph-only). The watcher keeps
it in sync the same way it does the RAG index.

### Languages supported

The graph extractor reuses the chunker's tree-sitter parsers, so any
language with an AST chunk rule also has a graph rule: **C#, Python,
TypeScript/TSX, JavaScript, C/C++, Go, Rust, Java, Ruby, PHP, Kotlin,
Swift**. Unsupported file types (markdown, shaders, JSON) are silently
skipped — they don't contribute to the graph but still feed search.

### The 10 MCP tools you get per source

When `graph.enabled=true` on source `myproject`, Lynx auto-registers these
extra tools on top of the usual `search_*` / `deep_search_*` (9 query
tools plus `graph_status_myproject()` for diagnostics):

| Tool | Answers |
|------|---------|
| `get_callers_myproject(symbol)` | "Who calls X?" |
| `get_callees_myproject(symbol)` | "What does X call?" |
| `get_subclasses_myproject(symbol)` | "What concrete types extend / implement X?" |
| `get_superclasses_myproject(symbol)` | "What does X inherit from?" |
| `get_imports_myproject(file_or_symbol)` | "What does this file depend on?" |
| `get_neighbors_myproject(symbol, relation_filter, depth)` | "Show me everything around X" |
| `shortest_path_myproject(source, target)` | "How does A reach B?" |
| `architectural_overview_myproject()` | "What are the main abstractions?" (god nodes + communities) |
| `surprising_connections_myproject()` | Bridge edges between distant communities (god-class antipatterns) |

Plus `graph_status_myproject()` for diagnostics.

### How cross-file calls and inheritance are resolved

Within a file, a call like `helper(x)` resolves immediately to the
locally-defined `def helper(...)`. Across files, the builder builds a
**global symbol index** keyed by lowercase identifier:

- **1 match** → edge with `confidence="resolved"`
- **>1 matches + direct call** → edges to all candidates, `confidence="ambiguous"`
- **>1 matches + member call (`obj.foo()`)** → skipped (too noisy without type info)
- **0 matches** → skipped (external / stdlib / dynamic call)

Inheritance bases (`class Foo : Bar`, `class Foo(Bar)`, `class Foo extends
Bar`, ...) use the same index with the same policy, minus the member-call
carve-out (base lists are always direct identifiers). Each `inherits`
edge also carries a `base_kind` attribute:

- `"extends"` — concrete superclass (Java/TS distinguish this at the AST level;
  Python uses it for every base by convention).
- `"implements"` — interface implementation, when the language exposes the
  distinction (Java `super_interfaces`, TS `implements_clause`, PHP
  `class_interface_clause`, Rust `impl Trait for Type`).
- `"extends_or_implements"` — language can't tell statically (C#, C++,
  Kotlin, Swift). Filter client-side by naming convention (`I*`) if needed.

This is best-effort, not whole-program type inference. The trade-off:
fast (sub-second on thousands of files), zero false negatives on simple
direct calls / single-class hierarchies, controlled false positives on
common names (e.g. two unrelated classes both called `BulletBase` — the
edges to both get `confidence="ambiguous"` so the AI client can flag it).

### What it costs

- **Disk**: ~1–5 MB JSON per 10k-file repo (nodes + edges + SHA cache).
- **RAM**: a NetworkX DiGraph rebuilt at startup — typically 20–80 MB.
- **Build time**: roughly equal to a second chunking pass. The graph
  layer shares the tree-sitter *parser cache* with the chunker (no
  duplicate parser instantiation), but each file is parsed twice — once
  for chunks, once for graph extraction — because the two passes use
  different AST traversal rules. An incremental SHA cache means unchanged
  files are skipped on subsequent runs of either layer.
- **Query latency**: sub-millisecond on graphs up to ~100k edges.
  `surprising_connections` is the heaviest (edge betweenness, ~seconds
  on large graphs; sampled automatically beyond 1500 nodes).

### Disabling it

Drop the `graph` block (or set `enabled: false`) and restart the server:
the 10 graph tools simply aren't registered. Search and deep_search work
exactly as before — the layer is genuinely optional.

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
|   - per-source search tools: search_<name> / deep_search_<name>    |
|   - per-source graph tools (opt-in): get_callers_<name>,           |
|     get_callees_<name>, get_subclasses_<name>,                     |
|     get_superclasses_<name>, get_imports_<name>,                   |
|     get_neighbors_<name>, shortest_path_<name>,                    |
|     architectural_overview_<name>, surprising_connections_<name>,  |
|     graph_status_<name>                                            |
|   - per-source combined tools (codebase only):                     |
|     find_definition_<name>, find_usages_<name>,                    |
|     find_tests_for_<name>, find_similar_<name>                     |
|   - per-source diff tool (when git_integration.enabled=true):      |
|     search_diff_<name>                                             |
|   - global tools: list_sources / search_all_sources /              |
|     deep_search_all_sources / update_source_index / get_rag_status |
+----------------------------+---------------------------------------+
                             |
+----------------------------v---------------------------------------+
| source_manager.py  (SourceManager)                                 |
|   - per-source dispatch (KeyError on unknown source)               |
|   - cross-source RRF fusion for *_all_sources                      |
|   - graph layer pass-through (raises if graph disabled)            |
+----------+----------------------------+----------------------------+
           |                            |
+----------v------------+ +--------v---------+ +-------v-----------+
| sources/codebase.py   | | sources/webdoc.py| | sources/pdf.py    |
| CodebaseBackend       | | WebdocBackend    | | PdfBackend        |
|   - file watcher      | |  - httpx crawler | |  - pypdf / pymupdf|
|     (updates both     | |  - trafilatura   | |  - per-page .md   |
|     RAG and graph)    | |    extraction    | |    dump + frontm. |
|   - holds CodebaseRAG | |  - writes .md    | |  - SHA cache for  |
|   - holds GraphLayer  | |    dump on fetch | |    incremental    |
|     (opt-in)          | |  - delegates to  | |  - delegates to   |
|                       | |    CodebaseRAG   | |    CodebaseRAG    |
+----+----------+-------+ +--------+---------+ +-------+-----------+
     |          |                  |  reads from _dump/  |
     |          |                       |
     |          |                       |
     |   +------v--------+              |
     |   | graph/ layer  |              |
     |   | (opt-in)      |              |
     |   |               |              |
     |   | extractor.py  | tree-sitter walk: classes, functions,
     |   |               | calls, inherits, imports, contains.
     |   |               | 13 langs, reuses chunking._get_parser.
     |   |               |
     |   | builder.py    | GraphLayer: nx.DiGraph in RAM + JSON on
     |   |               | disk. SHA-incremental rebuild. Cross-file
     |   |               | resolution against global symbol index.
     |   |               |
     |   | analyzer.py   | god_nodes (degree), communities
     |   |               | (greedy_modularity), surprising_connections
     |   |               | (edge_betweenness).
     |   |               |
     |   | query.py      | get_callers / get_callees / get_imports /
     |   |               | get_subclasses / get_superclasses /
     |   |               | get_neighbors / shortest_path.
     |   +------+--------+              |
     |          |                       |
     |  reads from path                 |
     +----+-----+----------+------------+
          |                |
+---------v----+    +------v-------------+
| Codebase RAG |    | Per-source ChromaDB collection + metadata     |
| pipeline:    |    | rag_storage/<source_name>/                    |
|              |    |   chroma.sqlite3       vectors                |
| chunking.    |--->|   metadata.json        drift snapshot         |
| chunk_file:  |    |   file_hashes.json     SHA-256 per file       |
| tree-sitter  |    |   graph/               (only if graph.enabled)|
| 13 langs +   |    |     nodes.json         all nodes              |
| SentenceSplit|    |     edges.json         all edges              |
|              |    |     raw_calls.json     unresolved calls       |
| HF embeddings|    |     raw_inherits.json  unresolved bases       |
| (BGE-small,  |    |     file_hashes.json   SHA-256 per file       |
|  CPU,offline)|    |     metadata.json      schema_version + ts    |
| ChromaDB,    |    |   _dump/               (webdoc only) .md      |
| BM25, RRF    |    |   _fetch_state.json    (webdoc only) state    |
+--------------+    +-----------------------------------------------+
                                ^
                                | scanned + watched (codebase) OR fetched (webdoc)
                +------------------------+
                | Your sources: local dir, or public docs site URL  |
                +------------------------+
```

Repository layout:

```
lynx/
├── pyproject.toml           Package metadata + dependencies
├── README.md                You are here
├── LICENSE                  Apache 2.0
├── config.example.json      Template config (commit this)
├── config.json              Your local config (gitignored)
├── src/
│   └── lynx/
│       ├── __init__.py        Package version
│       ├── __main__.py        Enables `python -m lynx`
│       ├── cli.py             argparse-based CLI dispatcher (incl. graph + migrate-config)
│       ├── server.py          FastMCP server, dynamic per-source tool registration
│       │                      (search + deep_search + 10 graph tools when opt-in)
│       ├── source_manager.py  SourceManager: per-source dispatch + cross-source RRF
│       │                      + graph layer pass-through
│       ├── rag_manager.py     CodebaseRAG: hybrid retrieval, drift, BM25, deep_search,
│       │                      per-file SHA-256 incremental cache
│       ├── chunking.py        AST-aware chunker (tree-sitter for 13 languages +
│       │                      SentenceSplitter fallback) with CHUNKER_VERSION,
│       │                      exposes `parse_file()` shared with graph layer
│       ├── config.py          v2 config loader with per-type validation
│       ├── sources/           Per-type source backends
│       │   ├── __init__.py    SOURCE_BACKENDS registry
│       │   ├── base.py        SourceBackend abstract base class
│       │   ├── codebase.py    CodebaseBackend (wraps CodebaseRAG + watcher
│       │   │                  + optional GraphLayer; 10 graph methods delegated)
│       │   ├── webdoc.py      WebdocBackend (crawl + extract + dump + reuse CodebaseRAG)
│       │   ├── pdf.py         PdfBackend (extract PDFs page-by-page + dump + reuse CodebaseRAG)
│       │   ├── pdf_extractor.py  pypdf / pymupdf backend selection, page-by-page extract
│       │   └── pdf_dump.py    Per-page .md writer + SHA-state cache
│       └── graph/             Opt-in knowledge graph layer (new in v0.5 / extended in v0.6)
│           ├── __init__.py    Public API: GraphLayer + 7 query funcs + 3 analyzers
│           ├── extractor.py   LangGraphRules + extract_file(): single-file walker,
│           │                  13 langs, classes / functions / imports / calls /
│           │                  base-lists (inherits)
│           ├── builder.py     GraphLayer: SHA cache, cross-file resolution,
│           │                  atomic JSON persistence, watcher sync, bootstrap
│           ├── analyzer.py    god_nodes, communities (greedy_modularity),
│           │                  surprising_connections (edge_betweenness)
│           └── query.py       get_callers, get_callees, get_subclasses,
│                              get_superclasses, get_imports, get_neighbors,
│                              shortest_path (fuzzy symbol matching)
├── tests/
│   ├── conftest.py                 pytest path shim + per-source RAG helper
│   ├── test_watch.py               End-to-end smoke test for the watcher
│   ├── test_drift.py               End-to-end smoke test for drift detection
│   ├── test_filters.py             Smoke test for search-filter parameters
│   ├── test_deep_search.py         Smoke test for the deep_search fallback ladder
│   ├── test_multi_source.py        Smoke test for multi-source dispatch + isolation
│   ├── test_sha_incremental.py     Smoke test for the per-file SHA rebuild cache
│   ├── test_tree_sitter.py         Unit tests for the AST chunker (13 languages)
│   ├── test_webdoc.py              Webdoc backend (crawl + extract) with mocked HTTP
│   ├── test_hybrid_vs_dense.py     Side-by-side dense vs hybrid benchmark
│   ├── test_graph_extractor.py     Graph extractor unit tests (10 scenarios, 13 langs)
│   ├── test_graph_builder.py       GraphLayer build / persist / SHA / cross-file
│   ├── test_graph_analyzer.py      god_nodes / communities / surprising_connections
│   ├── test_graph_query.py         get_callers / get_callees / shortest_path / ...
│   ├── test_graph_integration.py   config → backend → manager pass-through (stubbed RAG)
│   ├── test_graph_mcp_tools.py     _register_graph_tools end-to-end via FastMCP
│   ├── test_pdf_extractor.py       pypdf + reportlab synthetic PDFs (9 scenarios)
│   ├── test_pdf_dump.py            per-page .md writer + state cache round-trip
│   ├── test_pdf_config.py          type=pdf source validator
│   └── test_pdf_backend.py         PdfBackend integration with stubbed CodebaseRAG (9 scenarios)
└── rag_storage/                    ChromaDB collections, one subdir per source (gitignored)
    ├── myproject/                  (codebase source)
    │   ├── chroma.sqlite3          Vector store
    │   ├── metadata.json           Drift snapshot (config_snapshot + last_update)
    │   ├── file_hashes.json        Per-file SHA-256 cache for incremental rebuilds
    │   └── graph/                  Knowledge graph (only when graph.enabled=true)
    │       ├── nodes.json          One entry per class / function / file / external
    │       ├── edges.json          calls / inherits / imports / contains
    │       ├── raw_calls.json      Unresolved calls (re-resolved on update)
    │       ├── raw_inherits.json   Unresolved bases (re-resolved on update)
    │       ├── file_hashes.json    Graph-layer per-file SHA cache
    │       └── metadata.json       schema_version + last_update / last_full_rebuild
    ├── unityDoc/                   (webdoc source — same files as above PLUS:)
    │   ├── _dump/                  One .md per crawled URL (YAML frontmatter)
    │   └── _fetch_state.json       {url: {fetched_at, dump_file}}
    └── manuals/                    (pdf source — same Chroma files as above PLUS:)
        ├── _dump/                  One .md per extracted page, organised as
        │                           <rel>/<pdf_stem>/page_NNNN.md
        └── _extract_state.json     {pdf_abs_path: {sha256, n_pages, status, ...}}
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
  `OPENAI_API_KEY` is removed from the environment. The only network step
  is the explicit `webdoc` fetch, which the user triggers themselves —
  there is no implicit egress for `codebase` sources.
- **AST chunking, not token windows.** Code is split at function / method /
  class boundaries via tree-sitter (13 languages supported); other text falls
  back to the legacy sentence-window splitter. Each chunk carries a
  qualified `symbol_name` and a 1-based `start_line`/`end_line` range so the
  AI can cite precisely (`Container.cs:L555-580`). Bumping `CHUNKER_VERSION`
  invalidates all stored chunks via drift detection.
- **Webdoc refresh is always explicit.** A `type: "webdoc"` source never
  auto-re-fetches. Detecting "the upstream site changed" without
  re-downloading is unreliable across servers (ETag / Last-Modified support
  varies), so we don't try — the user runs `lynx build --source X` when
  they want a fresh snapshot. The fetch always wipes the dump first so
  removed pages don't leave stale chunks.
- **WebdocBackend reuses CodebaseRAG.** The only webdoc-specific code is the
  crawl + extract step; everything downstream (chunking, embedding, BM25,
  drift detection, SHA cache, hybrid search, deep_search) is exactly the
  pipeline that drives codebase sources, just pointed at the local dump
  folder. Adding `type: "pdf"` later follows the same pattern.
- **Per-file SHA-256 cache.** Each source's `file_hashes.json` stores the
  hash of every indexed file. On a force rebuild, unchanged files are
  skipped (no re-read, no re-embed); typical no-op rebuild is ~30× faster
  than a cold one. Snapshot mismatch (embedding model swap, chunker bump,
  extension list change) invalidates the cache and forces a full rebuild.

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
- Run the smoke tests:
  ```bash
  python tests/test_tree_sitter.py       # AST chunker (per-language)
  python tests/test_filters.py           # search filter parameters
  python tests/test_drift.py             # config drift detection
  python tests/test_deep_search.py       # deep_search fallback ladder
  python tests/test_multi_source.py      # multi-source dispatch + isolation
  python tests/test_sha_incremental.py   # per-file SHA rebuild cache
  python tests/test_webdoc.py            # webdoc crawler+extractor (mocked HTTP)
  python tests/test_watch.py             # watcher (subprocess; slowest)
  ```
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
