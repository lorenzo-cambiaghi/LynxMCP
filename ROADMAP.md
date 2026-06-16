# Roadmap

**This is a list of things we're *considering*, not commitments or a timeline.**
Priorities are driven by real usage and feedback — items here may change, ship
later, or be dropped. Lynx works fully today without any of them. Issues and
PRs that make a case for (or against) these are very welcome.

## Under evaluation

### Vector store — evaluate LanceDB as the local backend
ChromaDB serves Lynx well today, but we've hit **cross-version index-format
fragility** (a version-incompatible index could blank the dashboard or crash
the process — the reason Lynx now ships an out-of-process integrity probe and
`lynx reset`). [LanceDB](https://lancedb.com) is embedded, file-based, no
server, Rust, with a stable versioned on-disk format and more headroom for
large indexes — a better fit for the local-first design.

- **Why not Qdrant/others:** they run as a separate server/daemon. Adding
  infrastructure cuts against the "one command, 100% local, no services"
  promise. Embedded + file-based is the bar.
- **Cost / why it's not done yet:** a real migration (vector-store backend
  rewrite + a reindex for existing users + retesting). Worth it only if
  Chroma's stability or scale becomes a recurring pain in practice — not a
  speculative rewrite.

### Embeddings — an optional "quality" preset
The default stays **`bge-small-en-v1.5`** (384-dim, ~130 MB, fast on CPU) —
that's what keeps the install light and the on-save re-index near-instant
without a GPU. For users with more RAM/compute who want higher ranking
quality, we're considering **blessing and documenting a step-up preset**
(e.g. `nomic-embed-text-v1.5` or `bge-m3`).

- The embedding model is **already swappable** via `embedding.model_name` —
  this would just recommend a vetted alternative and document the trade-offs.
- **Caveat:** changing the model means a full reindex (different vector
  space), and bigger models are slower on CPU and produce larger indexes.
- Note: Lynx is **hybrid** (dense + BM25 + RRF), so the dense model doesn't
  carry retrieval alone — the marginal gain from a larger embedder is smaller
  here than in a dense-only system.

## Non-goals

To set expectations honestly, some things we **don't** plan to do:

- **No hosted / cloud version, no telemetry, no code upload.** "100% local"
  is the point, not a phase. (The self-host Docker image runs *your* server,
  on *your* machine — see the `Dockerfile`.)
- **No per-source tool explosion.** The MCP tool surface stays fixed and small
  so your client's context window doesn't grow with the number of sources.
