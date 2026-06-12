# Lynx retrieval benchmark — Django 5.2

Target: `django/` package of [django 5.2](https://github.com/django/django.git) — 883 files, 157,865 lines, 10,027 indexed chunks (one-time index build: 587s on CPU; incremental afterwards via SHA cache).

20 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; see `run_benchmark.py` for the exact methodology (the grep baseline is deliberately generous).

| Metric | Agentic grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 95% | 85% |
| hit@1 (top result is correct) | 45% | 55% |
| MRR | 0.64 | 0.67 |
| median tool output (tokens) | 2,853 | 1,725 |
| median tokens **to answer** (incl. required follow-up read) | 4,150 | 1,725 |
| tool round-trips to reach code | 2 (grep, then read file) | 1 (chunks included) |
| median latency / query | 0.019s | 0.055s |

Notes:
- Lynx tool output already CONTAINS the relevant code chunks with file:line.
  The grep output is match lines only, so its 'to answer' figure adds ONE
  conservative follow-up read (150 lines around the first match in the top
  file); real agents often need several reads and several grep refinements.
- The grep baseline is intentionally strong (IDF-weighted multi-keyword
  ranking with ideal stopword removal — closer to BM25 than to what an
  agent's first `rg` attempt looks like).
- Django is a favorable corpus for lexical search: extensively docstring-ed,
  English identifiers everywhere. Codebases with sparser comments or
  non-obvious naming shift results further toward semantic retrieval.
- Tokens approximated as chars/4.

## Beyond ranking: what one tool call hands the model

The ranking metrics above understate the practical gap, because the two
outputs are not the same *kind* of thing:

| Per result | grep | Lynx |
|---|---|---|
| matching line(s) | ✅ | ✅ (whole AST chunk: the full function/class body) |
| file + line range of the enclosing symbol | ❌ | ✅ (`db/models/base.py L767-881`) |
| qualified symbol name (`Model._save_table`) | ❌ | ✅ — citable directly in the answer |
| relevance score (when to stop trusting results) | ❌ | ✅ |
| works when the query shares NO words with the code | ❌ | ✅ (dense embeddings) |

And there is a whole class of questions grep cannot answer at any number
of calls, that Lynx answers in ONE (`graph_query` / `find_usages`):
*who calls X? what implements this interface? what breaks if I change
this?* — polymorphic dispatch leaves no textual trace, so grepping the
method name finds the definition and the textual mentions, not the
runtime callers through base classes.

## The round-trip economy

Tool execution time is noise (both pipelines answer in well under a
second). The real cost is that **every tool round-trip is a full model
inference over the entire growing context**: seconds of wall-clock and
the whole conversation re-billed in input tokens, every time.

| Typical flow | model inferences before the code is in context |
|---|---|
| grep: search → read top file → (often) refine search → read again | 3-5 |
| Lynx `search`: chunks arrive with the first response | **1** |
| grep: reconstruct callers of a symbol (N textual hits to triage) | 2 + N reads, dispatch still missed |
| Lynx `graph_query("callers")`: resolved edges with file:line | **1** |

## Per-task results

| # | Task | grep rank | Lynx rank |
|---|---|---|---|
| 1 | where are password hashes upgraded to a stronger algorithm after a suc | 1 | 1 |
| 2 | code that generates the SQL for select for update row locking | 1 | 2 |
| 3 | logic that decides whether saving a model instance performs an INSERT  | 1 | 1 |
| 4 | how conflicting migrations on the same app are detected | 3 | 1 |
| 5 | where the CSRF token from the request is compared against the stored s | 1 | 1 |
| 6 | how a case insensitive contains filter is turned into a SQL LIKE expre | — | — |
| 7 | template engine logic that resolves a dotted variable name against dic | 1 | 1 |
| 8 | where signed values are validated and their signature checked for tamp | 2 | 1 |
| 9 | how the development server restarts itself when a source file changes | 2 | 4 |
| 10 | where the password reset token is generated and checked for expiry | 3 | 2 |
| 11 | middleware that caches whole responses per URL and serves them on late | 4 | 3 |
| 12 | logic that streams large uploaded files to a temporary file on disk in | 2 | 1 |
| 13 | implementation of prefetching related objects to avoid N plus one quer | 1 | 1 |
| 14 | how an incoming request path is matched against the configured URL pat | 2 | — |
| 15 | where session data is serialized signed and loaded back from the sessi | 4 | 4 |
| 16 | how datetimes are converted to the currently active timezone | 1 | 1 |
| 17 | where bulk inserts are split into batches before hitting the database | 1 | 1 |
| 18 | middleware that protects against clickjacking by setting the X-Frame-O | 3 | 1 |
| 19 | how static files are located across the installed applications | 4 | — |
| 20 | logic that detects which model changes need new migrations to be gener | 1 | 2 |

## Structural queries: class relations (where grep dies)

Question: *"what inherits from `Field` (django.db.models) — i.e. what breaks if I change it?"*

| | grep (regex + manual closure) | Lynx graph_query |
|---|---|---|
| direct subclasses found | 49 | 49 (each with file + line) |
| full descendant tree | 100 classes | 100 classes, 4 levels |
| tool calls required | **101 grep rounds** (one per discovered class) | **1 per level (4 total)** — or 1 `get_neighbors(depth=N)` |
| wall clock (this harness; a real `rg` would be faster — the call count would not change) | 125.4s | 0.39s |
| symbol metadata (file:line, kind) | none — more reads needed | included in every edge |
| descendants the regex closure missed | 0 | — |
| found by grep but not in graph (false positives / unresolved) | 0 | — |

The grep numbers are the BEST case: they assume the agent writes a correct multi-name regex on the first try and never loses track of the frontier across 101 rounds. Each of those rounds is a full model inference in a real agent loop. One-time graph build for this corpus: ~1s with tree-sitter (incremental afterwards).
