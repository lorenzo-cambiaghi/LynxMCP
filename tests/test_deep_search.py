"""Smoke test for `deep_search()` — the multi-query fallback search.

Constructs CodebaseRAG once against the configured index and exercises the
following scenarios in sequence (one embedding-model load total):

  1. Single variant, strong query  - winning_variant_index == 1, all_weak False
  2. Multi-variant, first wins      - stops at variant 1, does NOT try 2/3
  3. Multi-variant, first weak      - falls through to the strong fallback
  4. All variants weak              - returns strongest weak set, all_weak True
  5. mode='dense' override          - retrieval runs in dense mode even when
                                      server default is hybrid
  6. return_all_variants            - response includes per_variant block
  7. Invalid input                  - empty list / non-string entries raise

Read-only: the index is not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

from local_codebase_rag_mcp.config import load_config
from local_codebase_rag_mcp.rag_manager import CodebaseRAG


HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
CONFIG_FILE = REPO_ROOT / "config.json"

# Queries chosen for predictable behavior against the user's Unity codebase
# (1413 files / 3168 chunks). The "good" queries are domain-relevant; the
# "nonsense" tokens are invented and should produce empty / very weak results.
STRONG_QUERY = "serialization interfaces and blob storage"
NONSENSE_QUERY = "qwertyxyz_zzz_nonsense_token_unlikely_to_match_anything_42"
NONSENSE_QUERY_2 = "absolutely_invented_marker_string_xyzzy_plugh"


def _build() -> CodebaseRAG:
    cfg = load_config(CONFIG_FILE)
    return CodebaseRAG(
        codebase_path=str(cfg.codebase_path),
        rag_storage_path=str(cfg.storage_path),
        supported_extensions=cfg.supported_extensions,
        embedding_model_name=cfg.embedding.model_name,
        collection_name=cfg.collection_name,
        search_mode=cfg.search.mode,
        rrf_k=cfg.search.rrf_k,
        candidate_pool_size=cfg.search.candidate_pool_size,
    )


def main() -> int:
    print("[test] Constructing CodebaseRAG (one embedding-model load)...")
    rag = _build()
    cfg = load_config(CONFIG_FILE)
    thresholds = cfg.search.deep.score_thresholds

    # ----- 1. single variant, strong query -----
    resp = rag.deep_search([STRONG_QUERY], top_k=5, score_thresholds=thresholds)
    if resp["winning_variant_index"] != 1:
        print(f"[test] FAIL [1/7]: expected winning_variant_index=1, got {resp['winning_variant_index']}")
        return 1
    if resp["all_weak"]:
        print(f"[test] FAIL [1/7]: strong query unexpectedly flagged as all_weak")
        return 1
    if resp["variants_tried"] != 1:
        print(f"[test] FAIL [1/7]: variants_tried should be 1, got {resp['variants_tried']}")
        return 1
    if not resp["results"]:
        print("[test] FAIL [1/7]: strong query returned no results — is the index built?")
        return 1
    print(f"[test] OK [1/7] single strong variant won at index 1, {len(resp['results'])} results")

    # ----- 2. multi-variant, first wins (should NOT try variants 2 and 3) -----
    resp = rag.deep_search(
        [STRONG_QUERY, NONSENSE_QUERY, NONSENSE_QUERY_2],
        top_k=5,
        score_thresholds=thresholds,
    )
    if resp["winning_variant_index"] != 1:
        print(f"[test] FAIL [2/7]: expected variant 1 to win, got {resp['winning_variant_index']}")
        return 2
    if resp["variants_tried"] != 1:
        print(f"[test] FAIL [2/7]: variants_tried should be 1 (stop on first win), got {resp['variants_tried']}")
        return 2
    print(f"[test] OK [2/7] first variant won, fallback variants 2 and 3 NOT tried")

    # ----- 3. multi-variant, first weak, fallback strong -----
    # Use mode='dense' + explicit min_score so the threshold semantics are
    # corpus-independent. BGE-small has a non-zero similarity floor on
    # nonsense input (~0.45-0.50 cosine), so we set the threshold at 0.55
    # which strong queries clear (~0.60-0.70) and nonsense does not.
    resp = rag.deep_search(
        [NONSENSE_QUERY, STRONG_QUERY],
        top_k=5,
        mode="dense",
        min_score=0.55,
    )
    if resp["winning_variant_index"] != 2:
        print(f"[test] FAIL [3/7]: expected variant 2 to win after variant 1 was weak, got {resp['winning_variant_index']}")
        return 3
    if resp["variants_tried"] != 2:
        print(f"[test] FAIL [3/7]: variants_tried should be 2, got {resp['variants_tried']}")
        return 3
    if resp["all_weak"]:
        print(f"[test] FAIL [3/7]: variant 2 should have crossed threshold")
        return 3
    print(f"[test] OK [3/7] fallback to variant 2 worked after variant 1 was weak (mode=dense, min_score=0.55)")

    # ----- 4. all variants weak -----
    # Same dense+explicit-threshold approach: both nonsense tokens should fail.
    resp = rag.deep_search(
        [NONSENSE_QUERY, NONSENSE_QUERY_2],
        top_k=5,
        mode="dense",
        min_score=0.55,
    )
    if not resp["all_weak"]:
        print(f"[test] FAIL [4/7]: all-nonsense queries should be flagged all_weak=True")
        return 4
    if resp["winning_variant_index"] is not None:
        print(f"[test] FAIL [4/7]: winning_variant_index should be None when all weak, got {resp['winning_variant_index']}")
        return 4
    if resp["variants_tried"] != 2:
        print(f"[test] FAIL [4/7]: should try every variant when none pass, got {resp['variants_tried']}")
        return 4
    print(f"[test] OK [4/7] all-weak case: {resp['variants_tried']} variants tried, all_weak flag set")

    # ----- 5. mode='dense' override -----
    # Force dense mode for this call regardless of the configured default.
    # We don't assert on score values (mode-dependent), only that the call
    # completes and respects the override (no crash / mode confusion).
    resp = rag.deep_search(
        [STRONG_QUERY],
        top_k=5,
        mode="dense",
        score_thresholds=thresholds,
    )
    # Dense scores are cosine ~0.3-0.7; if the value looks like an RRF score
    # (~0.01-0.03) we know the override was ignored.
    if resp["results"]:
        top_score = resp["results"][0]["score"]
        if top_score < 0.1:
            print(f"[test] FAIL [5/7]: mode='dense' override appears ignored (score {top_score:.4f} is in RRF range)")
            return 5
    print(f"[test] OK [5/7] mode='dense' override accepted and produced dense-scale scores")

    # ----- 6. return_all_variants=True -----
    # Same dense+explicit-threshold setup as test 3 for determinism.
    resp = rag.deep_search(
        [NONSENSE_QUERY, STRONG_QUERY],
        top_k=5,
        mode="dense",
        min_score=0.55,
        return_all_variants=True,
    )
    if "per_variant" not in resp:
        print(f"[test] FAIL [6/7]: per_variant missing when return_all_variants=True")
        return 6
    pv = resp["per_variant"]
    if len(pv) != 2:
        print(f"[test] FAIL [6/7]: per_variant should have 2 entries, got {len(pv)}")
        return 6
    if pv[0]["passed_threshold"]:
        print(f"[test] FAIL [6/7]: nonsense variant should NOT have passed threshold")
        return 6
    if not pv[1]["passed_threshold"]:
        print(f"[test] FAIL [6/7]: strong variant should have passed threshold")
        return 6
    print(f"[test] OK [6/7] return_all_variants exposed both variants with correct passed_threshold flags")

    # ----- 7. invalid inputs raise -----
    failures = []
    try:
        rag.deep_search([], top_k=5)
        failures.append("empty list did not raise")
    except ValueError:
        pass
    try:
        rag.deep_search([""], top_k=5)
        failures.append("empty string entry did not raise")
    except ValueError:
        pass
    try:
        rag.deep_search([STRONG_QUERY], top_k=5, mode="not-a-real-mode")
        failures.append("invalid mode did not raise")
    except ValueError:
        pass
    try:
        rag.deep_search("a single string not a list", top_k=5)
        failures.append("non-list queries argument did not raise")
    except TypeError:
        pass
    if failures:
        print("[test] FAIL [7/7]:", "; ".join(failures))
        return 7
    print("[test] OK [7/7] invalid inputs raise as expected (empty list, empty string, bad mode, str-not-list)")

    print("\n[test] === SUCCESS: deep_search works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
