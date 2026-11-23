# Lynx retrieval benchmark: django

Target: `django` of [django 5.2](https://github.com/django/django.git): 883 files, 157,865 lines, 10,976 indexed chunks (index build: 302s on CPU, incremental afterwards).

20 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; `run_benchmark.py` has the exact method.

| Metric | Keyword grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 95% | 90% |
| hit@1 (top result is correct) | 45% | 70% |
| MRR | 0.64 | 0.79 |
| median tool output, tokens | 2,853 | 1,587 |
| median tokens with one follow-up read of the top file | 4,150 | 1,587 |
| median latency per query | 0.023s | 0.062s |

Notes:
- The grep baseline ranks files by IDF-weighted counts of the question's
  content words, stopwords removed: closer to BM25 than to a first `rg`.
- Its output is up to 5 matching lines per keyword for each top file; the
  follow-up figure adds one 150-line read of the top file. Lynx's output
  is the top pieces with their code. Both simulate one retrieval, not an
  agent session: agents grep with narrower patterns, and read the files
  they change whatever found them.
- Tokens approximated as chars/4.

## What one result carries

| Per result | grep | Lynx |
|---|---|---|
| matching line(s) | yes | the whole piece: the function or class body |
| file and line range of the enclosing symbol | no | yes (`db/models/base.py L767-881`) |
| qualified symbol name (`Model._save_table`) | no | yes |
| relevance score | no | yes |
| a match when the question shares no word with the code | no | possible, through the dense embeddings |

Questions about structure (who calls X, what breaks if it changes) go to
the code graph (`find_usages`, `impact`, `graph_query`). A call through a
base class or an interface names the base type at the call site, so a
grep for the concrete method finds its definition and its mentions, not
those callers.

## Per-task results

| # | Task | grep rank | Lynx rank |
|---|---|---|---|
| 1 | where are password hashes upgraded to a stronger algorithm after a suc | 1 | 1 |
| 2 | code that generates the SQL for select for update row locking | 1 | 2 |
| 3 | logic that decides whether saving a model instance performs an INSERT  | 1 | 1 |
| 4 | how conflicting migrations on the same app are detected | 3 | 1 |
| 5 | where the CSRF token from the request is compared against the stored s | 1 | 1 |
| 6 | how a case insensitive contains filter is turned into a SQL LIKE expre | miss | 1 |
| 7 | template engine logic that resolves a dotted variable name against dic | 1 | 1 |
| 8 | where signed values are validated and their signature checked for tamp | 2 | 1 |
| 9 | how the development server restarts itself when a source file changes | 2 | 3 |
| 10 | where the password reset token is generated and checked for expiry | 3 | 1 |
| 11 | middleware that caches whole responses per URL and serves them on late | 4 | 1 |
| 12 | logic that streams large uploaded files to a temporary file on disk in | 2 | 2 |
| 13 | implementation of prefetching related objects to avoid N plus one quer | 1 | 1 |
| 14 | how an incoming request path is matched against the configured URL pat | 2 | 1 |
| 15 | where session data is serialized signed and loaded back from the sessi | 4 | miss |
| 16 | how datetimes are converted to the currently active timezone | 1 | 1 |
| 17 | where bulk inserts are split into batches before hitting the database | 1 | 1 |
| 18 | middleware that protects against clickjacking by setting the X-Frame-O | 3 | 1 |
| 19 | how static files are located across the installed applications | 4 | miss |
| 20 | logic that detects which model changes need new migrations to be gener | 1 | 2 |

## Structural queries: class relations

Question: *"what inherits from `Field` (django.db.models), that is, what can a change to it reach?"*

| | grep (regex and manual closure) | Lynx `graph_query subclasses` |
|---|---|---|
| direct subclasses found | 49 | 49, each with file and line |
| full descendant tree | 100 classes | 100 classes, 4 levels |
| calls, one per class found | 101 grep rounds | 101 calls (one symbol per call) |
| calls, one per level | 5 grep rounds (the level's names in one alternation) | not available: the tool takes one symbol |
| wall clock in this harness | 115.4s | 0.50s |
| what each match carries | a matching line; the file has to be read for the rest | a resolved edge with file, line and kind |
| descendants the regex closure missed | 0 | |
| found by grep but not in the graph | 0 | |

On this question grep needs fewer calls than the graph tool, because a regex can name a whole level at once. What the graph adds is edges resolved at index time, with no comment or string matches to sort out.
