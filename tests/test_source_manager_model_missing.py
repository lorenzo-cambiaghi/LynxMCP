"""Unit tests for SourceManager's model-missing classification.

When a source backend fails to init because the embedding/reranker model
isn't in the HF cache (and couldn't be downloaded), that is NOT a corrupt
index — so the user must NOT be told to `lynx reset` (which rebuilds but
still needs the model → loop). These tests exercise the two helpers that
encode that distinction without constructing real backends.

Run: `python tests/test_source_manager_model_missing.py`
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _fake_config(embed="BAAI/bge-small-en-v1.5", reranker_enabled=False,
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return SimpleNamespace(
        embedding=SimpleNamespace(model_name=embed),
        search=SimpleNamespace(
            reranker=SimpleNamespace(
                enabled=reranker_enabled, model_name=reranker_model
            )
        ),
    )


def main() -> int:
    from lynx.source_manager import SourceManager

    saved = os.environ.pop("HF_HUB_CACHE", None)
    tmp = Path(tempfile.mkdtemp(prefix="lynx-sm-"))
    try:
        # Empty cache dir → nothing is "cached".
        os.environ["HF_HUB_CACHE"] = str(tmp)

        mgr = SourceManager.__new__(SourceManager)  # skip __init__/backends
        mgr.config = _fake_config()
        mgr.broken = {}

        # 1. embedding missing → reported; reranker disabled → not reported
        missing = mgr._required_models_missing()
        if missing != ["BAAI/bge-small-en-v1.5"]:
            print(f"[test] FAIL [1/3]: missing={missing!r}")
            return 1
        print("[test] OK [1/3] _required_models_missing reports uncached embedding")

        # 2. reranker enabled → both reported
        mgr.config = _fake_config(reranker_enabled=True)
        missing = mgr._required_models_missing()
        if len(missing) != 2 or "cross-encoder" not in missing[1]:
            print(f"[test] FAIL [2/3]: missing={missing!r}")
            return 2
        print("[test] OK [2/3] enabled reranker is included when uncached")

        # 3. _register_model_unavailable: clear, non-corrupt message
        mgr.config = _fake_config()
        mgr.broken = {}
        mgr._register_model_unavailable(
            "src1", "codebase", {"path": "/p"}, tmp / "src1",
            ["BAAI/bge-small-en-v1.5"],
        )
        entry = mgr.broken.get("src1")
        if entry is None:
            print("[test] FAIL [3/3]: source not registered broken")
            return 3
        err = entry["error"].lower()
        if "model not available" not in err:
            print(f"[test] FAIL [3/3]: error missing model phrasing: {entry['error']!r}")
            return 3
        if "corrupt" in err or "reset" in err:
            print(f"[test] FAIL [3/3]: error wrongly mentions corrupt/reset: {entry['error']!r}")
            return 3
        if entry["crashed"] is not False:
            print(f"[test] FAIL [3/3]: crashed should be False: {entry!r}")
            return 3
        print("[test] OK [3/3] _register_model_unavailable: clear, non-corrupt")

        print("\n[test] === SUCCESS: model-missing classification works ===")
        return 0
    finally:
        os.environ.pop("HF_HUB_CACHE", None)
        if saved is not None:
            os.environ["HF_HUB_CACHE"] = saved
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    sys.exit(main())
