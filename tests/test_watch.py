"""End-to-end smoke test for the file watcher.

Launches the MCP server, then creates / modifies / deletes a small file inside
the configured codebase and asserts that the watcher reacts as expected on
stderr.

Reads the codebase path from `config.json` (same loader as the server), so the
test follows whatever the server is configured to index.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from local_codebase_rag_mcp.config import load_config

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
CONFIG_FILE = REPO_ROOT / "config.json"
CONFIG = load_config(CONFIG_FILE)
# Bind to the FIRST codebase-type source. The smoke test exercises the
# watcher wiring, which is the same for every codebase source.
_FIRST_SOURCE_NAME = next(iter(CONFIG.sources))
_FIRST_SOURCE = CONFIG.sources[_FIRST_SOURCE_NAME]
CODEBASE = Path(_FIRST_SOURCE["path"])
DEBOUNCE_SECONDS = _FIRST_SOURCE["watcher"]["debounce_seconds"]
TEST_FILE = CODEBASE / "__rag_watch_smoketest.md"

# Slack on top of the watcher's debounce so we don't race the timer.
DEBOUNCE_WAIT = max(DEBOUNCE_SECONDS + 1.5, 3.5)


def wait_for(stderr_path: Path, needle: str, timeout: float, since: int = 0) -> bool:
    """Wait for `needle` to appear in stderr starting from byte offset `since`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stderr_path.exists():
            data = stderr_path.read_bytes()[since:]
            if needle.encode("utf-8", errors="ignore") in data:
                return True
        time.sleep(0.2)
    return False


def main() -> int:
    if TEST_FILE.exists():
        TEST_FILE.unlink()

    log_path = HERE / "_test_watch_stderr.log"
    if log_path.exists():
        log_path.unlink()

    print(f"[test] Launching server (python -m local_codebase_rag_mcp serve), "
          f"stderr -> {log_path.name}")
    log_handle = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "local_codebase_rag_mcp",
            "serve", "--config", str(CONFIG_FILE),
        ],
        stdin=subprocess.PIPE,  # keep mcp.run() alive (otherwise EOF -> exit)
        stdout=subprocess.DEVNULL,
        stderr=log_handle,
        cwd=str(REPO_ROOT),
    )

    try:
        # 1) Wait until the watcher signals it's active (RAG fully loaded).
        print("[test] Waiting for 'Watcher active'...")
        if not wait_for(log_path, "Watcher active", timeout=300):
            print("[test] FAIL: watcher did not start within 5 minutes")
            return 1
        print("[test] OK: watcher started")

        # 2) Create a file -> expect a 'Re-indexed' line.
        offset = log_path.stat().st_size
        TEST_FILE.write_text(
            "# RAG smoketest\nUnique marker XYZZY-CREATE-12345.\n",
            encoding="utf-8",
        )
        print(f"[test] Created {TEST_FILE.name}, waiting for update...")
        if not wait_for(log_path, "Re-indexed", timeout=DEBOUNCE_WAIT * 2, since=offset):
            print("[test] FAIL: no 'Re-indexed' after create")
            return 2
        print("[test] OK: create triggered re-indexing")

        # 3) Modify the file -> another 'Re-indexed'.
        offset = log_path.stat().st_size
        TEST_FILE.write_text(
            "# RAG smoketest\nUnique marker XYZZY-MODIFY-99999.\n",
            encoding="utf-8",
        )
        print(f"[test] Modified {TEST_FILE.name}, waiting for update...")
        if not wait_for(log_path, "Re-indexed", timeout=DEBOUNCE_WAIT * 2, since=offset):
            print("[test] FAIL: no 'Re-indexed' after modify")
            return 3
        print("[test] OK: modify triggered re-indexing")

        # 4) Delete the file -> expect a 'Removed'.
        offset = log_path.stat().st_size
        TEST_FILE.unlink()
        print(f"[test] Deleted {TEST_FILE.name}, waiting for delete...")
        if not wait_for(log_path, "Removed", timeout=DEBOUNCE_WAIT * 2, since=offset):
            print("[test] FAIL: no 'Removed' after delete")
            return 4
        print("[test] OK: delete triggered removal")

        print("\n[test] === SUCCESS: incremental watchdog updates work ===")
        return 0
    finally:
        try:
            if TEST_FILE.exists():
                TEST_FILE.unlink()
        except Exception:
            pass
        print("[test] Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_handle.close()
        if log_path.exists():
            tail = log_path.read_bytes()[-2000:].decode("utf-8", errors="replace")
            print("\n[test] --- last lines of server stderr ---")
            print(tail)


if __name__ == "__main__":
    sys.exit(main())
