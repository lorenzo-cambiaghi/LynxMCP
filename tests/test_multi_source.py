"""Smoke test for the multi-source architecture.

Builds a fresh v2 config in a tempdir with TWO independent codebase sources
(two synthetic small folders with distinct content), then validates:

  1. SourceManager constructs both backends
  2. list_sources reports both with correct chunk counts and per-source storage
  3. Per-source search returns ONLY documents from that source
  4. Per-source ChromaDB lives in its own storage_path/<name>/ subdir
  5. Cross-source search_all returns hits from BOTH sources, each tagged with `source`
  6. deep_search_all winning_variant_index / all_weak semantics work cross-source
  7. KeyError on unknown source name

The test writes everything (sources + storage) under a single tempdir so the
user's real `rag_storage/` and `config.json` are never touched. Tempdir is
cleaned up at the end.
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _make_synthetic_source(root: Path, name: str, files: dict) -> Path:
    """Create a directory under `root` with the given files dict ({filename: text})."""
    src = root / name
    src.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (src / fname).write_text(content, encoding="utf-8")
    return src


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag-multi-src-"))
    print(f"[test] tempdir: {tmp}")

    try:
        # ----- Build two synthetic source folders -----
        alpha_dir = _make_synthetic_source(
            tmp,
            "alpha_src",
            {
                "alpha_overview.txt": (
                    "Alpha is a system for managing widgets. "
                    "Widgets have colors, sizes, and unique identifiers."
                ),
                "alpha_widget.txt": (
                    "Widget API: create_widget(color, size) returns Widget. "
                    "All widget operations are thread-safe and lock-free."
                ),
            },
        )
        beta_dir = _make_synthetic_source(
            tmp,
            "beta_src",
            {
                "beta_overview.txt": (
                    "Beta handles underwater volcanology measurements. "
                    "Sensors record temperature, pressure, and acoustic data."
                ),
                "beta_sensor.txt": (
                    "SensorArray.read() returns a TimeSeries of measurements. "
                    "Calibration is required every hundred dive cycles."
                ),
            },
        )

        # ----- Build a v2 config that points at both -----
        storage_path = tmp / "rag_storage"
        cfg_dict = {
            "config_version": 2,
            "storage_path": str(storage_path),
            "loading_timeout_seconds": 60,
            "embedding": {"model_name": "BAAI/bge-small-en-v1.5"},
            "search": {
                "default_top_k": 5,
                "mode": "hybrid",
                "rrf_k": 60,
                "candidate_pool_size": 30,
                "deep": {
                    "min_results": 1,
                    "score_thresholds": {
                        "dense": 0.45,
                        "hybrid": 0.012,
                        "sparse": 3.0,
                    },
                },
            },
            "sources": {
                "alpha": {
                    "type": "codebase",
                    "path": str(alpha_dir),
                    "supported_extensions": [".txt"],
                    "ignored_path_fragments": [],
                    "watcher": {"enabled": False, "debounce_seconds": 2.0},
                    "git_integration": {"enabled": False},
                },
                "beta": {
                    "type": "codebase",
                    "path": str(beta_dir),
                    "supported_extensions": [".txt"],
                    "ignored_path_fragments": [],
                    "watcher": {"enabled": False, "debounce_seconds": 2.0},
                    "git_integration": {"enabled": False},
                },
            },
        }
        cfg_path = tmp / "config.json"
        cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

        # ----- Construct SourceManager -----
        from local_codebase_rag_mcp.config import load_config
        from local_codebase_rag_mcp.source_manager import SourceManager

        cfg = load_config(cfg_path)
        mgr = SourceManager(cfg)

        # ----- 1. both backends constructed -----
        if set(mgr.backends.keys()) != {"alpha", "beta"}:
            print(f"[test] FAIL [1/7]: expected backends {{'alpha','beta'}}, got {set(mgr.backends.keys())}")
            return 1
        print(f"[test] OK [1/7] both backends constructed: {sorted(mgr.backends.keys())}")

        # ----- 2. list_sources reports both with chunk counts -----
        listed = mgr.list_sources()
        if len(listed) != 2:
            print(f"[test] FAIL [2/7]: list_sources returned {len(listed)} entries, expected 2")
            return 2
        for entry in listed:
            if entry["chunk_count"] is None or entry["chunk_count"] < 1:
                print(f"[test] FAIL [2/7]: source {entry['name']!r} has no chunks ({entry['chunk_count']})")
                return 2
        print(f"[test] OK [2/7] list_sources reports both, chunks: " +
              ", ".join(f"{e['name']}={e['chunk_count']}" for e in listed))

        # ----- 3. per-source search returns only that source's files -----
        a_hits = mgr.search("alpha", "widget operations", top_k=5)
        b_hits = mgr.search("beta", "sensor calibration", top_k=5)
        if not a_hits or not b_hits:
            print(f"[test] FAIL [3/7]: per-source search returned empty: alpha={len(a_hits)} beta={len(b_hits)}")
            return 3
        # Verify file names belong to the right source
        for h in a_hits:
            if not h["file"].startswith("alpha_"):
                print(f"[test] FAIL [3/7]: alpha search returned non-alpha file: {h['file']}")
                return 3
        for h in b_hits:
            if not h["file"].startswith("beta_"):
                print(f"[test] FAIL [3/7]: beta search returned non-beta file: {h['file']}")
                return 3
        print(f"[test] OK [3/7] per-source search scoping respected (alpha→alpha files, beta→beta files)")

        # ----- 4. per-source ChromaDB lives in its own subdir -----
        alpha_storage = storage_path / "alpha"
        beta_storage = storage_path / "beta"
        if not (alpha_storage / "chroma.sqlite3").is_file():
            print(f"[test] FAIL [4/7]: missing {alpha_storage}/chroma.sqlite3")
            return 4
        if not (beta_storage / "chroma.sqlite3").is_file():
            print(f"[test] FAIL [4/7]: missing {beta_storage}/chroma.sqlite3")
            return 4
        # Sanity: storage_path itself should NOT have its own chroma.sqlite3 at the root
        # (that would be the v1 layout — proves isolation is real).
        if (storage_path / "chroma.sqlite3").is_file():
            print(f"[test] FAIL [4/7]: unexpected top-level chroma.sqlite3 at {storage_path} (v1 layout leak)")
            return 4
        print(f"[test] OK [4/7] per-source storage isolated under {storage_path.name}/<source>/")

        # ----- 5. cross-source search_all returns tagged results from BOTH -----
        all_hits = mgr.search_all("widget volcanology", top_k=10)
        if not all_hits:
            print(f"[test] FAIL [5/7]: search_all returned 0 results")
            return 5
        sources_seen = {h.get("source") for h in all_hits}
        if "alpha" not in sources_seen or "beta" not in sources_seen:
            print(f"[test] FAIL [5/7]: search_all did not return both sources, saw {sources_seen}")
            return 5
        for h in all_hits:
            if "source" not in h:
                print(f"[test] FAIL [5/7]: cross-source hit missing 'source' tag: {h.get('file')}")
                return 5
        print(f"[test] OK [5/7] search_all returned hits from both sources, all tagged")

        # ----- 6. deep_search_all semantics -----
        resp = mgr.deep_search_all(
            ["widget operations", "sensor calibration"],
            top_k=5,
        )
        if not resp["results"]:
            print(f"[test] FAIL [6/7]: deep_search_all returned no results")
            return 6
        # First variant should be enough to cross the threshold (widgets live in alpha)
        if resp["all_weak"]:
            print(f"[test] FAIL [6/7]: deep_search_all flagged all_weak unexpectedly")
            return 6
        if resp["winning_variant_index"] != 1:
            print(f"[test] FAIL [6/7]: expected variant 1 to win, got {resp['winning_variant_index']}")
            return 6
        print(f"[test] OK [6/7] deep_search_all picked variant 1, no all_weak")

        # ----- 7. unknown source raises KeyError -----
        try:
            mgr.get("nonexistent")
            print(f"[test] FAIL [7/7]: mgr.get('nonexistent') did not raise")
            return 7
        except KeyError:
            pass
        try:
            mgr.search("nonexistent", "anything", top_k=1)
            print(f"[test] FAIL [7/7]: mgr.search on nonexistent did not raise")
            return 7
        except KeyError:
            pass
        print(f"[test] OK [7/7] unknown source raises KeyError")

        print("\n[test] === SUCCESS: multi-source dispatch, isolation, and cross-source work ===")
        return 0

    finally:
        # Release ChromaDB file handles before rmtree (Windows locks otherwise).
        try:
            del mgr  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print(f"[test] warning: tempdir cleanup failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
