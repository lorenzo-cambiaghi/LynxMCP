# Lynx PR impact analysis (GitHub Action)

> On every pull request, a comment tells you **what your change might break and
> what else you should review** — based on what the code *means* and how it's
> *wired together*, not just matching text. It runs entirely inside your CI, so
> your code never leaves it.

## The idea, in plain words

When you open a PR, GitHub gives you a fresh throwaway computer to run checks on
(the "runner"). This action uses that computer to:

1. **copy your repo onto it**,
2. **install Lynx on it** and let Lynx read your code into a small searchable
   "map" (which functions exist, what calls what),
3. for the files your PR changes, ask Lynx two questions:
   - **who else uses this?** → what might break (the *blast radius*),
   - **what code is similar?** → what to review alongside it,
4. **post the answers as a comment** on your PR,
5. then **throw the computer away**.

Your code and the map live only on that throwaway machine. Nothing is sent to any
outside AI service. You need a GitHub account — that's it (Actions is free for
public repos; private repos get a monthly allowance).

```
        you open / update a PR
                  │
                  ▼
   ┌─────────────────────────────────────────────┐
   │  GitHub runner (a fresh throwaway machine)    │
   │                                               │
   │   ① checkout your repo                        │
   │   ② pip install lynx-mcp                       │
   │   ③ Lynx builds the index   ← cached, so this │
   │      (code map + embeddings)   is fast after  │
   │                                the first run   │
   │   ④ for the changed files:                    │
   │        • /api/v1/graph  → who calls this?     │
   │        • /api/v1/search → what's related?     │
   │   ⑤ write a Markdown report                   │
   └───────────────────────┬───────────────────────┘
                           │  (GitHub API)
                           ▼
              💬 sticky comment on the PR
                           │
                           ▼
              runner destroyed; code never left CI
```

## Why it's worth it

- **Semantic, not grep.** "What clamps the camera zoom" finds the method even if
  the PR never used those words.
- **Structural.** It walks the real call graph, so it shows the *downstream*
  callers a reviewer would otherwise miss.
- **Local.** Embeddings are computed on the runner; only the PR comment goes out
  (to GitHub's own API).

## Setup (one file)

1. Copy [`workflow.example.yml`](workflow.example.yml) into your repo at
   `.github/workflows/lynx-impact.yml`.
2. Set `LYNX_VERSION` (top of the file) to a published release — **1.6.0 or
   newer**, since the action uses the `/api/v1/graph` and `format=ndjson`
   endpoints introduced in 1.6.0.
3. Trim `supported_extensions` in the config step to your stack.

It uses the built-in `GITHUB_TOKEN` (the workflow already requests
`pull-requests: write`). The analysis script is fetched from this repo at run
time — pin it to a tag/commit if you prefer.

**What gets indexed:** every committed file whose extension is in
`supported_extensions`, minus dot-directories (`.git`, …). A fresh CI checkout
has no `node_modules` / `dist` / build artifacts (they're git-ignored), so those
aren't indexed. If your repo *commits* a large vendored or generated tree
(`vendor/`, `Pods/`, generated SDKs), point the source `path` at your source
subdirectory (e.g. `src/`) instead of `.` — Lynx's `ignored_path_fragments`
only affects the live file watcher, **not** a one-shot build.

---

## How the caching works (so it doesn't rebuild every time)

This is the part people worry about. Two mechanisms combine so a PR re-embeds
**only the files it changed**, not the whole repo.

**1. GitHub Actions cache** persists folders between runs:

```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/huggingface   # the embedding model (~hundreds of MB)
      rag_storage            # the index: ChromaDB + the per-file SHA cache
    key: lynx-index-${{ github.sha }}
    restore-keys: lynx-index-
```

`key` is what it *saves* under at the end of a run (unique per commit, so each
run writes a new entry). `restore-keys: lynx-index-` means that at the start —
when the exact key isn't found — it restores the **most recent** cache whose key
starts with `lynx-index-`. So each run starts from the previous index.

**2. Lynx's incremental indexing** is what makes the restore useful. Lynx keeps
`rag_storage/<source>/file_hashes.json`, a **SHA-256 per file**. On each build it
compares every file's current hash to the cached one and splits them into
`unchanged / changed / added / removed`:

- **unchanged (same hash) → keeps the existing chunks, re-embeds nothing**,
- changed / added → re-embeds only those,
- removed → drops them.

Full vs. incremental depends **only on whether that SHA cache is present** — not
on any flag. Even `lynx build` (which runs `update(force=True)`) does a *delta*
rebuild when the cache exists; with no changes it prints *"Index already in sync;
no work to do."*

So in practice:

| Run | What happens | Cost |
|---|---|---|
| **First PR** (cold cache) | download model + index everything | minutes |
| **Every later PR** | restore cache → re-embed only the diff's files | seconds to ~1 min |

### Caveats worth knowing

- **Branch scoping.** A cache is readable from its own branch, the PR's base
  branch, and the default branch — not from *other* feature branches. So a brand
  new branch's first PR restores `main`'s index (great — it starts warm) and then
  deltas. Multiple parallel branches don't share each other's caches.
- **Eviction.** GitHub keeps **10 GB of cache per repo** (evicted least-recently
  used) and drops anything **untouched for 7 days**. After a long gap, or on a
  very large repo, a run may fall back to a full index.
- **Pin the version.** `CHUNKER_VERSION` is part of the cache's validity. If a
  newer Lynx bumps it, the next run does a one-time full re-index. That's why the
  workflow pins `LYNX_VERSION` — bump it deliberately.
- **Config changes** (paths, `supported_extensions`) invalidate the snapshot and
  trigger a full rebuild — expected.

---

## The local-first tradeoff (big repos)

Lynx indexes a **local** codebase and serves on `127.0.0.1`; a GitHub-hosted
runner is just a cloud machine that does it for you. Two ways to run it:

- **Self-contained (this workflow):** build on the runner, cached as above. Best
  for small/medium repos. First PR slow, the rest fast.
- **Self-hosted (big monorepos / strict policies):** either a **self-hosted
  runner** (your own machine, registered with GitHub, where Lynx stays indexed),
  or Lynx running as a **service** on your server (there's a `Dockerfile` in the
  repo root) with the action pointed at it via `LYNX_API`. CI then does almost no
  work and the code never touches GitHub's runners.

## What `pr_impact.py` does

```
git diff base...head            → changed code files
for each file:
  extract declared symbols      → best-effort per-language regex
  GET /api/v1/graph callers     → cross-file callers (blast radius)
  GET /api/v1/search            → semantically related code elsewhere
→ Markdown grouped per file (sticky comment, marker `<!-- lynx-impact-analysis -->`)
```

It's a **reviewer aid, not sound static analysis**: symbol names are extracted
heuristically, then resolved fuzzily by Lynx. If the API is unreachable it
degrades gracefully (empty report, no failure).

Run it locally to preview the output:

```bash
lynx manager ui --port 8765 --no-browser &      # an indexed, graph-enabled source
python pr_impact.py --source repo --base origin/main --head HEAD
# or feed files explicitly:
python pr_impact.py --source repo --changed-files "src/a.py src/b.py"
```

Env vars mirror the flags: `LYNX_API`, `LYNX_SOURCE`, `BASE_SHA`, `HEAD_SHA`,
`CHANGED_FILES`, `GITHUB_WORKSPACE` (repo root).

## Files

| File | What it is |
|---|---|
| `pr_impact.py` | The analysis script (pure stdlib). |
| `workflow.example.yml` | The ready-to-copy workflow. |

## See also

- [docs/MCP_RECIPES.md](../../docs/MCP_RECIPES.md) — the same correlation, agent-driven, in your editor.
- [docs/CORAL.md](../../docs/CORAL.md) / [docs/DUCKDB.md](../../docs/DUCKDB.md) — SQL / analytics over the same `/api/v1` endpoints.
