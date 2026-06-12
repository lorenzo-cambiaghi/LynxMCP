# Lynx

**A 100% local MCP server for semantic code search — AST-aware chunking, hybrid BM25 + dense retrieval, and an optional code knowledge graph. Works with any MCP client (Claude Code, Cursor, Windsurf, Antigravity, ...).**

[![Tests](https://github.com/lorenzo-cambiaghi/LynxMCP/actions/workflows/test.yml/badge.svg)](https://github.com/lorenzo-cambiaghi/LynxMCP/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

Your AI assistant greps file names and guesses. Lynx gives it real retrieval over **your code, your library docs, and your PDFs** — without a single byte leaving your machine.

- **AST-aware indexing** — tree-sitter parses 13+ languages and indexes whole functions/classes, not arbitrary text windows.
- **Hybrid retrieval** — dense embeddings + code-tokenized BM25, fused with RRF; optional cross-encoder reranker.
- **Code knowledge graph (opt-in)** — who-calls-what, inheritance, imports: ask "what breaks if I change this?" and get the actual blast radius.
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
| `list_sources` / `get_rag_status` / `update_source_index` | Introspection and maintenance. |

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

## Documentation

| | |
|---|---|
| [Full guide](docs/GUIDE.md) | Configuration, all source types (codebase / webdoc / PDF), retrieval internals, troubleshooting |
| [Manager UI](docs/GUIDE.md#lynxmanager--guided-setup-web-ui-diagnostics-new-in-v09) | Guided setup, playground, diagnostics |
| [config.example.json](config.example.json) | Annotated example configuration |

## Status

Actively developed by one author; APIs may still move before 1.x stabilizes. Issues and PRs welcome — the test suite runs with `pytest` and CI must stay green.

## License

[Apache 2.0](LICENSE)

<!-- MCP Registry ownership marker — must stay in the README published on
     PyPI so registry.modelcontextprotocol.io can verify the package.
mcp-name: io.github.lorenzo-cambiaghi/lynx
-->

