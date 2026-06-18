# Lynx PR impact analysis (GitHub Action)

On every pull request, comment with **what the change touches**:

- **Downstream callers** — the cross-file callers of the symbols the PR changes
  (the blast radius), from Lynx's code graph.
- **Semantically related code** — code elsewhere that is similar in behaviour to
  the change, from Lynx's hybrid search.

It runs **100% on the GitHub runner** against an indexed copy of your repo — your
code never leaves CI.

## Files

| File | What it is |
|---|---|
| `pr_impact.py` | The analysis script. Reads the PR's changed files, queries the local Lynx API (`/api/v1/search` + `/api/v1/graph`, NDJSON), prints a Markdown report. Pure stdlib. |
| `workflow.example.yml` | A ready workflow: index → start API → run the script → post a sticky comment. Copy it into your repo's `.github/workflows/`. |

## Setup

1. Copy `workflow.example.yml` to `.github/workflows/lynx-impact.yml` in your repo.
2. That's it — it uses the built-in `GITHUB_TOKEN` (the workflow already requests
   `pull-requests: write`). The script is fetched from this repo at run time; pin
   it to a tag/commit if you prefer.

The workflow writes a small `config.json` that registers the checkout as a
graph-enabled `codebase` source. Trim `supported_extensions` to your stack.

## The local-first tradeoff (read this)

Lynx indexes a **local** codebase and serves on `127.0.0.1`; a GitHub Action runs
in the cloud. So the index has to live somewhere:

- **Self-contained (this workflow):** build the index on the runner. The
  embedding model (~hundreds of MB) and `rag_storage` are cached between runs, so
  the **first PR is slow** (download + full index) and later ones are fast (cache
  restore + Lynx re-embeds only changed files). Best for small/medium repos.
- **Self-hosted Lynx (big monorepos):** run Lynx on your own box (there's a
  `Dockerfile` in the repo root), keep it indexed, and point the script at it —
  set `LYNX_API` to your instance and drop the install/build/start steps. CI does
  almost no work; the code stays on your server.

## How `pr_impact.py` works

```
git diff base...head            → changed code files
for each file:
  extract declared symbols      → best-effort per-language regex
  GET /api/v1/graph callers     → cross-file callers (blast radius)
  GET /api/v1/search            → semantically related code elsewhere
→ Markdown grouped per file (sticky comment, marker `<!-- lynx-impact-analysis -->`)
```

It is a **reviewer aid, not sound static analysis**: symbol names are extracted
heuristically then resolved fuzzily by Lynx. If the API is unreachable it degrades
gracefully (empty report, no failure).

Run it locally to see the output:

```bash
lynx manager ui --port 8765 --no-browser &      # an indexed, graph-enabled source
python pr_impact.py --source repo --base origin/main --head HEAD
# or feed files explicitly:
python pr_impact.py --source repo --changed-files "src/a.py src/b.py"
```

Env vars mirror the flags: `LYNX_API`, `LYNX_SOURCE`, `BASE_SHA`, `HEAD_SHA`,
`CHANGED_FILES`, `GITHUB_WORKSPACE` (repo root).

## See also

- [docs/MCP_RECIPES.md](../../docs/MCP_RECIPES.md) — the same correlation, agent-driven, in your editor.
- [docs/CORAL.md](../../docs/CORAL.md) / [docs/DUCKDB.md](../../docs/DUCKDB.md) — SQL / analytics over the same `/api/v1` endpoints.
