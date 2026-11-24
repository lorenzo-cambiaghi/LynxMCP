"""Run every script suite: the tests/test_*.py files with a `main()`, which
conftest.py keeps out of pytest. CI calls this after pytest.

    python tests/run_script_suites.py [--list]

A new suite is picked up with the probe conftest.py uses, so the workflow
never lists file names. Left out: the suites that open the machine's real
configuration (test_watch.py, test_deep_search.py, test_search_batch.py)
and, when the embedding model is not on the machine (CI does not download
it), the suites that need it. Exit code 1 when any suite fails, with its
last lines.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL_CONFIG = {"test_watch.py", "test_deep_search.py", "test_search_batch.py"}
# Suites that embed text with the real model: the ones that failed in a run
# with an empty model cache and HF_HUB_OFFLINE=1 (2026-11-25), while all 24
# passed with the model.
NEED_MODEL = {"test_drift.py", "test_filters.py", "test_hybrid_vs_dense.py", "test_multi_source.py",
              "test_sha_incremental.py", "test_webdoc.py"}


def suites() -> list:
    return sorted(p for p in HERE.glob("test_*.py")
                  if "def main(" in p.read_text(encoding="utf-8") and p.name not in REAL_CONFIG)


def model_available() -> bool:
    """The default embedding model is in the local cache (checked offline,
    so nothing is downloaded)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        sys.path.insert(0, str(HERE.parent / "src"))
        from lynx.model_files import resolve_model_files
        resolve_model_files("BAAI/bge-small-en-v1.5")
        return True
    except Exception:
        return False


def main() -> int:
    picked = suites()
    if not model_available():
        picked = [p for p in picked if p.name not in NEED_MODEL]
    if "--list" in sys.argv:
        print("\n".join(p.name for p in picked))
        return 0
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(HERE.parent / "src"), os.environ.get("PYTHONPATH")])))
    failed = []
    for p in picked:
        t = time.time()
        r = subprocess.run([sys.executable, str(p)], cwd=str(HERE.parent), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=1800)
        took = time.time() - t
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {p.name} ({took:.0f} s)", flush=True)
        if r.returncode != 0:
            failed.append(p.name)
            tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
            print("\n".join("     " + line for line in tail), flush=True)
    skipped = len(suites()) - len(picked)
    print(f"{len(picked) - len(failed)} of {len(picked)} script suites passed"
          + (f"; {skipped} need the embedding model, not on this machine" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
