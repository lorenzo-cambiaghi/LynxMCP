# Lynx retrieval benchmark — Guava

Target: `guava/src/com/google/common` of [Guava master](https://github.com/google/guava.git) — 606 files, 180,517 lines, 12,901 indexed chunks (one-time index build: 525s on CPU).

15 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; see `run_benchmark.py` for the exact methodology (the grep baseline is deliberately generous).

| Metric | Agentic grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 93% | 80% |
| hit@1 (top result is correct) | 73% | 60% |
| MRR | 0.81 | 0.70 |
| median tool output (tokens) | 3,991 | 807 |
| median tokens **to answer** (incl. required follow-up read) | 5,892 | 807 |
| tool round-trips to reach code | 2 (grep, then read file) | 1 (chunks included) |
| median latency / query | 0.022s | 0.051s |

Notes:
- Lynx tool output already CONTAINS the relevant code chunks with file:line.
  The grep output is match lines only, so its 'to answer' figure adds ONE
  conservative follow-up read (150 lines around the first match in the top
  file); real agents often need several reads and several grep refinements.
- The grep baseline is intentionally strong (IDF-weighted multi-keyword
  ranking with ideal stopword removal — closer to BM25 than to what an
  agent's first `rg` attempt looks like).
- A well-documented, English-identifier corpus is a favorable case for
  lexical search. Codebases with sparser comments or non-obvious naming
  shift results further toward semantic retrieval.
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
| 1 | limiting the rate of operations by handing out permits smoothed over t | 1 | 1 |
| 2 | an in-memory cache that loads missing values and evicts entries by siz | 1 | 1 |
| 3 | splitting a string on a separator, optionally trimming and dropping em | 1 | 1 |
| 4 | joining a sequence of values into one string with a separator | 2 | 2 |
| 5 | a probabilistic set membership test that can have false positives but  | 3 | — |
| 6 | mapping a hash value to one of n buckets so that adding a bucket moves | 1 | 2 |
| 7 | argument checks that throw an exception when a precondition is not met | 1 | 1 |
| 8 | measuring elapsed time with start, stop and read of the duration | 1 | 1 |
| 9 | a map that also lets you look up keys by their value | — | — |
| 10 | walking the nodes of a graph in breadth first or depth first order | 1 | 2 |
| 11 | computing percentiles or quantiles of a collection of numbers | 1 | 1 |
| 12 | percent encoding a string so it is safe inside a url | 3 | 1 |
| 13 | dispatching posted events to all registered subscriber methods | 1 | — |
| 14 | configuring a cache's maximum size and expiration before building it | 1 | 1 |
| 15 | building an immutable list and copying elements into it | 1 | 1 |
