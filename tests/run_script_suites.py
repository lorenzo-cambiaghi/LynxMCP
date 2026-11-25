"""Run every script suite: the tests/test_*.py files with a `main()`, which
conftest.py keeps out of pytest. CI calls this after pytest.

    python tests/run_script_suites.py [--list]

A new suite is picked up with the probe conftest.py uses, so the workflow
never lists file names. Left out: the suites that work on the machine's
real configuration and its index, and, when the embedding model is not on
the machine (CI does not download it), the suites that need it. Exit code 1
when any suite fails, with its last lines.

Every suite runs with RAG_CONFIG_PATH naming a file that does not exist: a
suite that reaches for the default configuration fails, instead of opening
the real indexes (2026-11-24: test_drift, test_filters and
test_hybrid_vs_dense did, from ./config.json, until CI had no such file).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Suites that need a real, configured index: they read ./config.json.
REAL_CONFIG = {"test_watch.py", "test_deep_search.py", "test_search_batch.py",
               "test_drift.py", "test_filters.py", "test_hybrid_vs_dense.py"}
# Suites that embed text with the real model: the ones that failed in a run
# with an empty model cache and HF_HUB_OFFLINE=1 (2026-11-25), while all
# passed with the model.
NEED_MODEL = {"test_multi_source.py", "test_sha_incremental.py", "test_webdoc.py"}


def suite_env() -> dict:
    """The environment of a suite: the source tree first, no default
    configuration to fall back on, and UTF-8 output (a Windows console in
    cp1252 made a suite fail on printing a check mark)."""
    no_config = Path(tempfile.mkdtemp(prefix="lynx-suites-")) / "config.json"
    return dict(os.environ,
                PYTHONPATH=os.pathsep.join(filter(None, [str(HERE.parent / "src"), os.environ.get("PYTHONPATH")])),
                RAG_CONFIG_PATH=str(no_config), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def suites() -> list:
    return sorted(p for p in HERE.glob("test_*.py")
                  if re.search(r"(?m)^def main\(", p.read_text(encoding="utf-8")) and p.name not in REAL_CONFIG)


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
    # A failing suite's last lines may hold characters the console cannot show.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    picked = suites()
    if not model_available():
        picked = [p for p in picked if p.name not in NEED_MODEL]
    if "--list" in sys.argv:
        print("\n".join(p.name for p in picked))
        return 0
    env = suite_env()
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
