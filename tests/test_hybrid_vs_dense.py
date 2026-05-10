"""A/B comparison: dense vs hybrid retrieval on the configured codebase.

Constructs two CodebaseRAG instances against the same index, one in dense
mode and one in hybrid mode, runs the same set of queries through both,
and prints the rankings side by side. Used as a one-off sanity check that
hybrid actually improves identifier-heavy queries on real data.

This is a benchmark, not a regression test: it does not assert anything,
it only reports.
"""

import sys

from local_codebase_rag_mcp.config import load_config
from local_codebase_rag_mcp.rag_manager import CodebaseRAG


# (label, query) tuples covering different retrieval regimes.
QUERIES = [
    ("exact-identifier",    "IDamageable"),
    ("camelcase-identifier", "AStarPathFinder"),
    ("natural-language",    "how damage is dispatched between entities"),
    ("mixed",               "where IBlobSerializer is implemented"),
    ("snake-case-ish",      "modular bullet movement system"),
]

TOP_K = 5


def _build(mode: str) -> CodebaseRAG:
    cfg = load_config()
    return CodebaseRAG(
        codebase_path=str(cfg.codebase_path),
        rag_storage_path=str(cfg.storage_path),
        supported_extensions=cfg.supported_extensions,
        embedding_model_name=cfg.embedding.model_name,
        collection_name=cfg.collection_name,
        search_mode=mode,
        rrf_k=cfg.search.rrf_k,
        candidate_pool_size=cfg.search.candidate_pool_size,
    )


def _format_ranking(results) -> list:
    return [f"{r['file']}  ({r['score']:.3f})" for r in results]


def main() -> int:
    print("[bench] Loading dense engine...")
    dense_rag = _build("dense")
    print("[bench] Loading hybrid engine...")
    hybrid_rag = _build("hybrid")

    width = 50
    print()
    print("=" * (width * 2 + 6))
    print(f"  {'DENSE'.ljust(width)}  |  {'HYBRID (BM25 + RRF, k=60)'.ljust(width)}")
    print("=" * (width * 2 + 6))

    for label, query in QUERIES:
        print()
        print(f"Query [{label}]: {query!r}")
        print("-" * (width * 2 + 6))
        d = _format_ranking(dense_rag.search(query, top_k=TOP_K))
        h = _format_ranking(hybrid_rag.search(query, top_k=TOP_K))
        rows = max(len(d), len(h))
        for i in range(rows):
            left = d[i] if i < len(d) else ""
            right = h[i] if i < len(h) else ""
            marker = " " if (left or "").split("  ")[0] == (right or "").split("  ")[0] else "*"
            print(f"{marker} {left.ljust(width)}  |  {right.ljust(width)}")

    print()
    print("Legend: '*' = the two columns disagree at this rank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
