# Lynx MCP recipes

Lynx is an MCP server, so an agent (Claude Code/Desktop, Cursor, Windsurf, VS
Code, …) can use Lynx's code tools **alongside other MCP servers** — GitHub,
Jira/Linear, Sentry, Slack, Postgres — and do the join in its own reasoning. No
SQL, no glue code: you describe the goal, the agent chains the tools.

This is the highest-leverage way to analyse **code in relation to other data**:
Lynx answers "where/what/who in the code", the companion server brings the
tickets / PRs / errors / owners, and the agent connects them.

## Lynx's tools (the code side)

| Tool | Answers |
|---|---|
| `search` | "where is the code that *does* X" (semantic + lexical hybrid). Omit `source` to search everything. |
| `deep_search` | same, heavier — escalate when `search` is weak/empty. |
| `find_definition` / `find_usages` | exact symbol definition / all call sites. |
| `find_tests_for` / `find_similar` | tests covering a symbol / structurally similar code. |
| `graph_query` | call graph + inheritance + imports: `callers`, `callees`, `subclasses`, `superclasses`, `imports`, `neighbors`, `shortest_path`, `overview`, `surprising_connections`, `status`. |
| `search_diff` | what changed *semantically* vs a git base (needs git integration). |

Companion servers referenced below (any equivalent works): **GitHub** (PRs,
issues, commits, CODEOWNERS, code-scanning alerts), **Sentry** (errors/
stacktraces), **Jira/Linear** (tickets), **Slack** (notify/ask), **Postgres/
SQLite** (app data).

## Setup

Add Lynx next to your other MCP servers in the host config (Lynx runs over
stdio via `lynx serve`):

```jsonc
{
  "mcpServers": {
    "lynx": { "command": "lynx", "args": ["serve"] }
    // …then add your GitHub / Sentry / Jira / Slack MCP servers alongside it,
    // each per its own install docs.
  }
}
```

Index at least one codebase source first (`lynx manager ui`), and enable the
graph layer on it for the `graph_query`-based recipes.

---

## Recipes

Each recipe lists the servers it touches, the flow (real tool names), and a
prompt you can paste.

### 1. Triage a production error → the code + its blast radius

**Servers:** Sentry · Lynx · GitHub

1. Sentry MCP → fetch the issue: message, culprit, top stack frames.
2. Lynx `search` with the failing behaviour ("where the webhook retry backoff is
   computed"), not the error string. Confirm with `find_definition`.
3. Lynx `graph_query operation=callers symbol=…` → what else reaches this code
   (the blast radius of a fix).
4. GitHub MCP → recent commits/PRs touching those files + the last author.

> **Prompt:** "Take Sentry issue PROJ-1234, find the code responsible with Lynx,
> show me everything that calls it, and who last changed those files on GitHub."

### 2. PR impact analysis (review helper)

**Servers:** GitHub · Lynx

1. GitHub MCP → the PR's changed files and the symbols they touch.
2. For each changed symbol: Lynx `find_usages` + `graph_query operation=callers`
   → who depends on it.
3. Lynx `find_tests_for` each changed symbol → is it covered?
4. Post a GitHub review comment summarising downstream impact + test gaps.

> **Prompt:** "For PR #481, use Lynx to list everything that uses the symbols it
> changes and whether they have tests, then leave a review comment with the
> blast radius."

### 3. Ticket → relevant code + scope estimate

**Servers:** Jira/Linear · Lynx

1. Jira/Linear MCP → the ticket description / acceptance criteria.
2. Lynx `search` (escalate to `deep_search` if weak) → the code areas involved.
3. Lynx `graph_query operation=neighbors depth=2` on the top hit → surface area.
4. Reply with the files to touch and a rough scope (isolated vs cross-cutting).

> **Prompt:** "Read LINEAR-555 and tell me which files implement this today and
> how big the change is, using Lynx to find and size the affected code."

### 4. Onboarding / architecture map

**Servers:** Lynx · GitHub

1. Lynx `graph_query operation=overview` → most-connected hubs + communities.
2. Lynx `graph_query operation=surprising_connections` → bridge points worth
   knowing.
3. Lynx `search` over a docs/PDF source for the "why" behind each cluster.
4. GitHub MCP → CODEOWNERS to attach an owner to each area.

> **Prompt:** "Give me an architecture tour of this repo: the core hubs and
> clusters from Lynx's graph, what each does, and who owns them per CODEOWNERS."

### 5. Security: where is a flagged API actually used?

**Servers:** GitHub · Lynx

1. GitHub MCP → open code-scanning / Dependabot alerts (the risky API/CVE).
2. Lynx `find_usages` / `search` → every call site of that API in the code.
3. Lynx `graph_query operation=callers` → which of them are reachable from
   entry points (prioritise those).
4. Output a ranked remediation list (call site → reachability → owner).

> **Prompt:** "For the top code-scanning alert on GitHub, use Lynx to find all
> call sites and rank them by how reachable they are from public entry points."

### 6. Semantic release notes

**Servers:** Lynx · GitHub

1. Lynx `search_diff` against the previous release tag → areas that changed in
   *behaviour*, not just lines.
2. GitHub MCP → the PR/commit titles in that range.
3. Draft notes grouped by area, each line citing the file + PR.

> **Prompt:** "Draft release notes since tag v1.5.0: use Lynx `search_diff` to
> find what changed behaviourally and match it to merged PRs on GitHub."

---

## Notes

- **Local-first.** Lynx never sends your code anywhere; only the companion
  server's side of a join touches a remote API.
- The agent decides the chaining — these prompts just steer it. If a step comes
  back weak, nudge it ("escalate to `deep_search`", "expand with
  `graph_query neighbors`").
- Prefer SQL or programmatic joins over agent reasoning? See [CORAL.md](CORAL.md)
  (SQL source) and [DUCKDB.md](DUCKDB.md) (local analytics over the `/api/v1`
  endpoints).
