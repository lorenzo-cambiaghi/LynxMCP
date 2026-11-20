# Roadmap

This is a list of things we're considering, not commitments or a timeline.
Priorities follow real usage and feedback: items here may change, ship
later, or be dropped. Lynx works fully today without any of them. Issues and
PRs that make a case for (or against) these are welcome.

## Under evaluation

### Vector store: evaluate LanceDB as the local backend
ChromaDB serves Lynx well today, but we've hit cross-version index-format
fragility: a version-incompatible index could blank the dashboard or crash
the process, which is why Lynx now ships an out-of-process integrity probe and
`lynx reset`. [LanceDB](https://lancedb.com) is embedded, file-based, no
server, Rust, with a stable versioned on-disk format and more headroom for
large indexes. That is a better fit for the local-first design.

- Qdrant and the others run as a separate server or daemon. Adding
  infrastructure cuts against the "one command, 100% local, no services"
  promise. Embedded and file-based is the bar.
- It is not done yet because it is a real migration: a vector-store backend
  rewrite, a reindex for existing users, and retesting. Worth it only if
  Chroma's stability or scale becomes a recurring pain in practice, not as a
  speculative rewrite.

### Embeddings: an optional "quality" preset
The default stays `bge-small-en-v1.5` (384-dim, ~130 MB, fast on CPU).
That is what keeps the install light and the on-save re-index near-instant
without a GPU. For users with more RAM or compute who want higher ranking
quality, we're considering documenting a vetted step-up preset
(e.g. `bge-base-en-v1.5` or `bge-m3`).

- The embedding model is already swappable via `embedding.model_name`,
  as long as the repo ships the ONNX export Lynx runs (`onnx/model.onnx`;
  the BGE family does, `nomic-embed-text-v1.5` does not in its main repo).
  This would only recommend an alternative and document the trade-offs.
- Changing the model means a full reindex (different vector
  space), and bigger models are slower on CPU and produce larger indexes.
- Lynx is hybrid (dense + BM25 + RRF), so the dense model doesn't
  carry retrieval alone. The marginal gain from a larger embedder is smaller
  here than in a dense-only system.

## On the radar (additive, lower-risk)

### More language parsers
Lynx ships tree-sitter grammars for 19+ languages today (Python, TypeScript,
JavaScript, C#, C, C++, Go, Rust, Java, Ruby, PHP, Kotlin, Swift, Bash, SQL,
Scala, Lua, Objective-C, GDScript). Each grammar is a small wheel bundled in
the install, with no runtime download, and adding one is usually a matter of
declaring which node types are functions and classes. Demand decides what
comes next; open an issue with the language you need and it moves up.

- Zig and HTML/CSS are candidates. Elixir and HCL take more work: every
  construct is the same node type (`call`, `block`), so they need a rule that
  reads the first identifier.
- Dart and R are on hold: their community grammars don't expose a parser
  usable with the current tree-sitter.
- A grammar without wheels for every platform is installed only where it has
  them, and elsewhere its files are indexed as text. GDScript is the first.

## Non-goals

Some things we don't plan to do:

- No hosted or cloud version, no telemetry, no code upload. "100% local"
  is the point, not a phase. (The self-host Docker image runs *your* server,
  on *your* machine; see the `Dockerfile`.)
- No per-source tool explosion. The MCP tool surface stays fixed and small,
  so your client's context window doesn't grow with the number of sources.
