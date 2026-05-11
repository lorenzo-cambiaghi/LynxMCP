"""Smoke test for per-file SHA-256 incremental rebuilds.

Builds a CodebaseRAG over a synthetic 3-file source, then walks through:

  1. First build           → file_hashes.json created with 3 entries + snapshot
  2. No-op rebuild         → 0 changed (short-circuit in _build_index)
  3. Modify one file       → 1 changed, SHA updated for that file only
  4. Delete one file       → 1 removed, entry dropped from cache
  5. Add one file          → 1 added, new entry in cache
  6. Snapshot mismatch     → cache invalidated; rebuild treats all as added
  7. Watcher no-op save    → update_file returns False, no SHA change

Tempdir teardown at the end. Read-only against the user's real index.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _make_codebase(root: Path, files: dict) -> Path:
    """Create a small codebase under root. files = {basename: content}."""
    src = root / "code"
    src.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (src / fname).write_text(content, encoding="utf-8")
    return src


def _norm(path) -> str:
    """Normalize a path the same way the production code does."""
    import os
    return os.path.normpath(os.path.abspath(str(path)))


def _load_hashes(storage_dir: Path) -> dict:
    """Read the file_hashes.json file and return its `files` dict (or {})."""
    fh = storage_dir / "file_hashes.json"
    if not fh.exists():
        return {}
    raw = json.loads(fh.read_text(encoding="utf-8"))
    return raw.get("files") or {}


def _load_hash_envelope(storage_dir: Path) -> dict:
    """Read the full file_hashes.json envelope (schema_version, config_snapshot, files)."""
    fh = storage_dir / "file_hashes.json"
    return json.loads(fh.read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag-sha-"))
    print(f"[test] tempdir: {tmp}")

    rag = None
    try:
        # ----- Build synthetic 3-file codebase -----
        code = _make_codebase(tmp, {
            "alpha.txt": "Alpha file. Talks about widgets and their colors.",
            "beta.txt":  "Beta file. Covers acoustic sensors and underwater calibration.",
            "gamma.txt": "Gamma file. Gamma describes pathfinding heuristics for A-star.",
        })
        storage_dir = tmp / "rag_storage" / "src"

        from lynx.rag_manager import CodebaseRAG

        def _make_rag(embedding_model="BAAI/bge-small-en-v1.5"):
            return CodebaseRAG(
                codebase_path=str(code),
                rag_storage_path=str(storage_dir),
                supported_extensions=[".txt"],
                embedding_model_name=embedding_model,
                collection_name="src",
                search_mode="hybrid",
                rrf_k=60,
                candidate_pool_size=10,
            )

        # ============================================================
        # 1. First build creates file_hashes.json with 3 entries + snapshot
        # ============================================================
        print("[test] [1/7] first build...")
        rag = _make_rag()
        hashes = _load_hashes(storage_dir)
        if len(hashes) != 3:
            print(f"[test] FAIL [1/7]: expected 3 hash entries, got {len(hashes)}")
            return 1
        envelope = _load_hash_envelope(storage_dir)
        if envelope.get("schema_version") != 1:
            print(f"[test] FAIL [1/7]: bad schema_version {envelope.get('schema_version')}")
            return 1
        if envelope.get("config_snapshot", {}).get("collection_name") != "src":
            print(f"[test] FAIL [1/7]: snapshot collection_name mismatch")
            return 1
        for fname in ("alpha.txt", "beta.txt", "gamma.txt"):
            abs_p = _norm(code / fname)
            if abs_p not in hashes:
                print(f"[test] FAIL [1/7]: missing hash entry for {fname}")
                return 1
            if not hashes[abs_p].get("sha256"):
                print(f"[test] FAIL [1/7]: empty SHA for {fname}")
                return 1
        print(f"[test] OK [1/7] first build wrote file_hashes.json with 3 entries + valid snapshot")

        # Snapshot SHAs and chunk count for comparison in later steps.
        original_hashes = dict(hashes)
        original_chunk_count = rag.vector_store._collection.count()
        if original_chunk_count < 3:
            print(f"[test] FAIL [1/7]: expected >= 3 chunks in ChromaDB, got {original_chunk_count}")
            return 1

        # ============================================================
        # 2. No-op rebuild: update(force=True) on unchanged disk → short-circuit
        # ============================================================
        print("[test] [2/7] no-op rebuild on unchanged disk...")
        rag.update(force=True)
        hashes = _load_hashes(storage_dir)
        # Every SHA should be unchanged.
        for abs_p, entry in original_hashes.items():
            if hashes.get(abs_p, {}).get("sha256") != entry["sha256"]:
                print(f"[test] FAIL [2/7]: SHA changed for {abs_p} despite no edit")
                return 2
        # Chunk count must be identical (no re-embed work).
        if rag.vector_store._collection.count() != original_chunk_count:
            print(f"[test] FAIL [2/7]: chunk count changed despite no edits")
            return 2
        print("[test] OK [2/7] no-op rebuild preserved all SHAs and chunks")

        # ============================================================
        # 3. Modify one file → that file's SHA must update; the others must not
        # ============================================================
        print("[test] [3/7] modify alpha.txt, rebuild...")
        time.sleep(0.01)  # ensure last_indexed_at timestamps differ visibly
        (code / "alpha.txt").write_text("Alpha file v2. Now talks about widgets, sizes, AND opacity.")
        rag.update(force=True)
        hashes = _load_hashes(storage_dir)
        alpha_p = _norm(code / "alpha.txt")
        beta_p  = _norm(code / "beta.txt")
        gamma_p = _norm(code / "gamma.txt")
        if hashes[alpha_p]["sha256"] == original_hashes[alpha_p]["sha256"]:
            print(f"[test] FAIL [3/7]: alpha SHA did not change after edit")
            return 3
        if hashes[beta_p]["sha256"] != original_hashes[beta_p]["sha256"]:
            print(f"[test] FAIL [3/7]: beta SHA changed unexpectedly")
            return 3
        if hashes[gamma_p]["sha256"] != original_hashes[gamma_p]["sha256"]:
            print(f"[test] FAIL [3/7]: gamma SHA changed unexpectedly")
            return 3
        print("[test] OK [3/7] only the modified file's SHA changed")

        # ============================================================
        # 4. Delete one file → entry dropped from cache, chunks gone
        # ============================================================
        print("[test] [4/7] delete beta.txt, rebuild...")
        (code / "beta.txt").unlink()
        rag.update(force=True)
        hashes = _load_hashes(storage_dir)
        if beta_p in hashes:
            print(f"[test] FAIL [4/7]: beta still in cache after deletion")
            return 4
        # Other two must still be there.
        if alpha_p not in hashes or gamma_p not in hashes:
            print(f"[test] FAIL [4/7]: alpha/gamma evaporated unexpectedly")
            return 4
        print("[test] OK [4/7] deleted file removed from cache")

        # ============================================================
        # 5. Add a new file → new entry, others untouched
        # ============================================================
        print("[test] [5/7] add delta.txt, rebuild...")
        (code / "delta.txt").write_text("Delta file. Brand-new content on observability.")
        delta_p = _norm(code / "delta.txt")
        old_alpha_sha = hashes[alpha_p]["sha256"]
        old_gamma_sha = hashes[gamma_p]["sha256"]
        rag.update(force=True)
        hashes = _load_hashes(storage_dir)
        if delta_p not in hashes:
            print(f"[test] FAIL [5/7]: new file not added to cache")
            return 5
        if hashes[alpha_p]["sha256"] != old_alpha_sha:
            print(f"[test] FAIL [5/7]: alpha SHA changed despite no edit")
            return 5
        if hashes[gamma_p]["sha256"] != old_gamma_sha:
            print(f"[test] FAIL [5/7]: gamma SHA changed despite no edit")
            return 5
        print("[test] OK [5/7] new file added without disturbing the others")

        # ============================================================
        # 6. Snapshot mismatch → cache invalidated, all-new treatment
        # ============================================================
        # Simulate a config change (e.g. user swapped embedding model) by
        # corrupting the stored snapshot directly. Then re-instantiate
        # CodebaseRAG — _load_file_hashes should refuse the cache.
        print("[test] [6/7] corrupt config_snapshot, re-instantiate, expect cache rejection...")
        fh_path = storage_dir / "file_hashes.json"
        envelope = _load_hash_envelope(storage_dir)
        envelope["config_snapshot"]["embedding_model_name"] = "fake/different-model-v999"
        fh_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

        # Release the live RAG so we can construct a new one cleanly.
        del rag
        gc.collect()

        rag = _make_rag()
        # On instantiation, _load_file_hashes sees the mismatch and returns {}.
        # We can verify by inspecting the in-memory hashes BEFORE any rebuild.
        if rag._file_hashes:
            print(f"[test] FAIL [6/7]: stale cache survived snapshot mismatch "
                  f"({len(rag._file_hashes)} entries loaded)")
            return 6
        # And a force-rebuild repopulates a fresh, snapshot-correct cache.
        rag.update(force=True)
        hashes = _load_hashes(storage_dir)
        envelope = _load_hash_envelope(storage_dir)
        if envelope["config_snapshot"]["embedding_model_name"] != "BAAI/bge-small-en-v1.5":
            print(f"[test] FAIL [6/7]: live snapshot not restored after rebuild")
            return 6
        if len(hashes) != 3:  # alpha, gamma, delta (beta was deleted)
            print(f"[test] FAIL [6/7]: expected 3 entries post-rebuild, got {len(hashes)}")
            return 6
        print("[test] OK [6/7] cache invalidated, full rebuild restored a valid one")

        # ============================================================
        # 7. Watcher no-op save: same content → no SHA change, return False
        # ============================================================
        print("[test] [7/7] no-op save via update_file...")
        # Re-save alpha.txt with the EXACT same bytes it already has.
        same_content = (code / "alpha.txt").read_text(encoding="utf-8")
        before_alpha_sha = hashes[alpha_p]["sha256"]
        before_alpha_indexed_at = hashes[alpha_p]["last_indexed_at"]
        (code / "alpha.txt").write_text(same_content, encoding="utf-8")
        result = rag.update_file(str(code / "alpha.txt"))
        hashes_after = _load_hashes(storage_dir)
        if result is not False:
            print(f"[test] FAIL [7/7]: update_file returned {result!r} on no-op save (expected False)")
            return 7
        if hashes_after[alpha_p]["sha256"] != before_alpha_sha:
            print(f"[test] FAIL [7/7]: SHA changed despite identical content")
            return 7
        if hashes_after[alpha_p]["last_indexed_at"] != before_alpha_indexed_at:
            print(f"[test] FAIL [7/7]: last_indexed_at advanced despite no-op")
            return 7
        print("[test] OK [7/7] no-op save short-circuited; SHA and timestamp untouched")

        print("\n[test] === SUCCESS: SHA-based incremental indexing works as expected ===")
        return 0

    finally:
        try:
            del rag
        except Exception:
            pass
        gc.collect()
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print(f"[test] warning: tempdir cleanup failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
