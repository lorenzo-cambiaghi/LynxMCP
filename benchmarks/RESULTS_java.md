# Lynx retrieval benchmark: Guava

Target: `guava/src/com/google/common` of [Guava master](https://github.com/google/guava.git): 606 files, 180,517 lines, 14,189 indexed chunks (index build: 355s on CPU, incremental afterwards).

15 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; `run_benchmark.py` has the exact method.

| Metric | Keyword grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 93% | 87% |
| hit@1 (top result is correct) | 73% | 67% |
| MRR | 0.81 | 0.76 |
| median tool output, tokens | 3,991 | 1,316 |
| median tokens with one follow-up read of the top file | 5,892 | 1,316 |
| median latency per query | 0.024s | 0.083s |

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
| 1 | limiting the rate of operations by handing out permits smoothed over t | 1 | 1 |
| 2 | an in-memory cache that loads missing values and evicts entries by siz | 1 | 2 |
| 3 | splitting a string on a separator, optionally trimming and dropping em | 1 | 1 |
| 4 | joining a sequence of values into one string with a separator | 2 | miss |
| 5 | a probabilistic set membership test that can have false positives but  | 3 | 2 |
| 6 | mapping a hash value to one of n buckets so that adding a bucket moves | 1 | 1 |
| 7 | argument checks that throw an exception when a precondition is not met | 1 | 1 |
| 8 | measuring elapsed time with start, stop and read of the duration | 1 | 1 |
| 9 | a map that also lets you look up keys by their value | miss | miss |
| 10 | walking the nodes of a graph in breadth first or depth first order | 1 | 1 |
| 11 | computing percentiles or quantiles of a collection of numbers | 1 | 1 |
| 12 | percent encoding a string so it is safe inside a url | 3 | 1 |
| 13 | dispatching posted events to all registered subscriber methods | 1 | 3 |
| 14 | configuring a cache's maximum size and expiration before building it | 1 | 1 |
| 15 | building an immutable list and copying elements into it | 1 | 1 |
