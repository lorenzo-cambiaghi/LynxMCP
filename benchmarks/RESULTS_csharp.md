# Lynx retrieval benchmark: Json.NET (Newtonsoft.Json)

Target: `Src/Newtonsoft.Json` of [Json.NET (Newtonsoft.Json) master](https://github.com/JamesNK/Newtonsoft.Json.git): 240 files, 69,132 lines, 3,932 indexed chunks (index build: 123s on CPU, incremental afterwards).

15 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; `run_benchmark.py` has the exact method.

| Metric | Keyword grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 67% | 73% |
| hit@1 (top result is correct) | 33% | 67% |
| MRR | 0.47 | 0.70 |
| median tool output, tokens | 4,900 | 1,083 |
| median tokens with one follow-up read of the top file | 6,590 | 1,083 |
| median latency per query | 0.012s | 0.029s |

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
| 1 | where circular references between objects are detected so serializatio | 1 | 1 |
| 2 | how a type's members are inspected to build the contract metadata used | 2 | 1 |
| 3 | reading json one token at a time from a text stream, handling string e | 1 | 1 |
| 4 | writing json to a text output with proper escaping of quotes and contr | 2 | 1 |
| 5 | turning incoming json back into objects, choosing a constructor and po | miss | miss |
| 6 | selecting tokens from a document with a path expression like store boo | 2 | miss |
| 7 | converting an enumeration value to and from its text name during conve | miss | 1 |
| 8 | deep merging the contents of one json object into another | 1 | miss |
| 9 | validating a json document against a schema while it is being read | 1 | 1 |
| 10 | parsing and formatting date and time values in iso 8601 format | miss | 1 |
| 11 | resolving a type name string back to the concrete type for polymorphic | miss | 1 |
| 12 | reading the binary bson wire format into json tokens | 1 | 1 |
| 13 | the naming strategy that lowercases the first character of property na | 4 | 2 |
| 14 | converting primitive values between types such as string to number wit | miss | miss |
| 15 | generating a json schema document from a type's structure | 3 | 1 |
