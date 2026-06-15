"""Script-style test: search_batch() must agree with search() per query.

Guards the batched-search optimization on three fronts:
  1. the batched query embedding matches the single-query path (catches the
     query-vs-text-embedding class of bug, which causes *large* divergences),
  2. the shared `_retrieve_one` core keeps single and batch in lockstep,
  3. the one-time embedding self-check engages and passes.

hybrid (RRF, rank-based) and sparse come out bit-identical; dense is score-
sensitive, so batched float inference (padding) can reorder near-ties at the
~1e-6 level — we allow at most one boundary swap there but require the same
top-1 and near-total overlap (a real bug shifts the *files*, not one boundary).

Read-only. Needs a built index (run after `lynx build`):
    python tests/test_search_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from conftest import build_rag_from_first_source

HERE = Path(__file__).parent
CONFIG_FILE = HERE.parent / "config.json"

QUERIES = [
    "serialization interfaces and blob storage",
    "camera zoom clamp",
    "object pool allocation",
    "rotation matrix helper",
    "spline bullet movement",
    "voxel grid neighbor lookup",
]


def _sig(hits):
    return [(h.get("file_path"), h.get("symbol_name")) for h in hits]


def main() -> int:
    print("[test] Constructing CodebaseRAG (one embedding-model load)...")
    cfg, name, rag = build_rag_from_first_source(CONFIG_FILE)
    rag.search("warm-up", top_k=5)

    fails = 0
    for mode in ("hybrid", "dense", "sparse"):
        single = [rag._search_once(q, 5, mode) for q in QUERIES]
        batch = rag.search_batch(QUERIES, top_k=5, mode=mode)
        if len(batch) != len(QUERIES):
            print(f"[FAIL] mode={mode}: batch returned {len(batch)} lists for "
                  f"{len(QUERIES)} queries")
            fails += 1
            continue
        for i, q in enumerate(QUERIES):
            s, b = _sig(single[i]), _sig(batch[i])
            if not s:
                continue  # nothing to compare (e.g. sparse on a weak query)
            if mode in ("hybrid", "sparse"):
                ok = s == b  # rank-based / deterministic → bit-identical
            else:  # dense: allow one float-noise boundary reorder
                ok = s[0] == b[0] and len(set(s) & set(b)) >= len(s) - 1
            if not ok:
                fails += 1
                print(f"[FAIL] mode={mode} q={q!r}\n   single={s}\n   batch ={b}")

    if rag._batch_embed_ok is not True:
        print(f"[FAIL] batched-embedding self-check did not pass "
              f"(flag={rag._batch_embed_ok!r}) — batch fell back to per-query")
        fails += 1

    if fails:
        print(f"[test] FAIL: {fails} problem(s)")
        return 1
    print("[test] OK: search_batch agrees with search across hybrid/dense/sparse; "
          "batched embedding verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
