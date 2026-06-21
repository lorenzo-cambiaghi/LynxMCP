# Lynx retrieval benchmark — Json.NET (Newtonsoft.Json)

Target: `Src/Newtonsoft.Json` of [Json.NET (Newtonsoft.Json) master](https://github.com/JamesNK/Newtonsoft.Json.git) — 240 files, 69,132 lines, 3,724 indexed chunks (one-time index build: 186s on CPU).

15 natural-language tasks with known ground-truth files. Both pipelines return a ranked file list; see `run_benchmark.py` for the exact methodology (the grep baseline is deliberately generous).

| Metric | Agentic grep (simulated) | Lynx hybrid search |
|---|---|---|
| hit@5 (ground truth in top 5) | 67% | 73% |
| hit@1 (top result is correct) | 33% | 47% |
| MRR | 0.47 | 0.58 |
| median tool output (tokens) | 4,900 | 1,540 |
| median tokens **to answer** (incl. required follow-up read) | 6,590 | 1,540 |
| tool round-trips to reach code | 2 (grep, then read file) | 1 (chunks included) |
| median latency / query | 0.01s | 0.027s |

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
| 1 | where circular references between objects are detected so serializatio | 1 | 1 |
| 2 | how a type's members are inspected to build the contract metadata used | 2 | 1 |
| 3 | reading json one token at a time from a text stream, handling string e | 1 | — |
| 4 | writing json to a text output with proper escaping of quotes and contr | 2 | 1 |
| 5 | turning incoming json back into objects, choosing a constructor and po | — | 1 |
| 6 | selecting tokens from a document with a path expression like store boo | 2 | — |
| 7 | converting an enumeration value to and from its text name during conve | — | 1 |
| 8 | deep merging the contents of one json object into another | 1 | 2 |
| 9 | validating a json document against a schema while it is being read | 1 | 1 |
| 10 | parsing and formatting date and time values in iso 8601 format | — | 4 |
| 11 | resolving a type name string back to the concrete type for polymorphic | — | — |
| 12 | reading the binary bson wire format into json tokens | 1 | 2 |
| 13 | the naming strategy that lowercases the first character of property na | 4 | 2 |
| 14 | converting primitive values between types such as string to number wit | — | — |
| 15 | generating a json schema document from a type's structure | 3 | 1 |
