# Search before you write

This project is indexed by Lynx, a local MCP server. Its sources:
`<project>` (our code) and `<library-docs>` (the docs of a library we use).
Every tool takes `source`; leave it out to search all of them.

## Before writing code

- Does it already exist? Call `search` and describe what the code does:
  "clamp camera zoom between two limits". Identifiers work too:
  "MeshVoxelizer.Voxelize". Do this before writing any utility, math helper,
  interface, or pooling, caching, registry or event code.
- Is my draft a duplicate? Call `find_similar` with the draft, or with
  the function you are about to copy. It returns the parallel
  implementations. In practice a score of about 0.6 or more has meant a
  real twin.
- When you know the name: `find_definition`, `find_usages`, `describe_symbol`.
  They bring the code with them, so there is no file to open afterwards.
- Before opening a file you don't know, `module_summary` lists what it
  defines, with signatures and doc lines.
- Who calls it, and what breaks if it changes: `graph_query` (callers,
  subclasses, neighbors) and `impact`. Text search cannot see these.
- Before calling a library API, search `<library-docs>`: the model's
  knowledge may predate the version this project uses.

Show the user what you found before writing something new next to it, and
say whether it can be reused as is, extended, or only looks similar.

## Narrowing

- `path_contains`, `extensions`, `file_glob` choose the files before the
  ranking, so a filter never hides a match that exists.
- `outline=true` returns signatures only: use it for broad questions or a
  large `top_k`, then read the one body you need.

## Reading `search` scores

They fuse a lexical and a semantic ranking, so they are small:

| Score | Meaning |
|---|---|
| about 0.03 | First in both rankings. The best a result gets. |
| 0.02 to 0.03 | Strong. |
| 0.012 to 0.02 | Related: read it before you rely on it. |
| below 0.012 | Weak. Rephrase, or call `deep_search` with two or three genuinely different phrasings. |

If a note above the results says the lexical index is unavailable and the
results are dense-only, scores stop near 0.016 and identifier queries suffer.
Tell the user once: `lynx manager doctor` shows why.

`find_similar` scores come from embedding distance (higher is closer), not
from the fused ranking above.

## Not for this

- A file you already know: read it.
- Exact strings, such as an error message or a config key: grep is fine.
- Rebuilding the index: the watcher keeps it current. Call
  `update_source_index` only when the user asks or a drift warning says so.
