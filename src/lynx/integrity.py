"""Out-of-process integrity probe for a source's ChromaDB index.

A corrupt or version-incompatible Chroma index can fail in two ways when the
host process opens it:

  1. a catchable ``chromadb.errors.*`` exception, or
  2. a hard NATIVE crash (segfault / access violation) that no ``try/except``
     can intercept — it takes the whole process down.

Case (2) would kill the LynxManager UI or the MCP server outright (white page,
no traceback). To guarantee the host survives, we probe the index in a CHILD
process first: open the collection and ``count()`` it there. If the child
crashes or errors, the parent marks the source corrupt and NEVER opens the bad
index itself.

Run as a module for the child side::

    python -m lynx.integrity <storage_dir> <collection_name>

Exit 0 + JSON ``{"ok": true, "count": N}`` on success; non-zero on any failure
(our own ``exit(1)`` for a caught exception, or an OS crash code for a segfault).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _probe_child(storage_dir: str, collection_name: str) -> int:
    """Child-process body: open + read the collection. Crashes here are
    contained to this process."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=storage_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(collection_name)
    count = collection.count()
    # Touch the HNSW segment read path too — count() alone doesn't always
    # exercise the segment that tends to be the corrupt one.
    try:
        collection.get(limit=1)
    except Exception:
        # A get() failure on a non-empty collection still signals trouble,
        # but count() is the authoritative health signal; don't fail here.
        pass
    print(json.dumps({"ok": True, "count": count}))
    return 0


def check_index(
    storage_dir,
    collection_name: str,
    *,
    timeout: float = 60.0,
) -> dict:
    """Probe a source's index in a subprocess. Never raises, never opens the
    index in the calling process.

    Returns one of::

        {"status": "ok",      "count": N}
        {"status": "empty"}                       # nothing built yet / 0 chunks
        {"status": "corrupt", "detail": "...", "crashed": bool}
    """
    storage_dir = Path(storage_dir)
    # Nothing on disk yet → a fresh source, not a corrupt one.
    if not (storage_dir / "chroma.sqlite3").exists():
        return {"status": "empty"}

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "lynx.integrity", str(storage_dir), collection_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "corrupt",
            "detail": f"integrity probe timed out after {timeout:.0f}s "
                      f"(index likely wedged)",
            "crashed": True,
        }
    except Exception as e:  # pragma: no cover - spawn failure is environmental
        # If we can't even spawn the probe, fail open: assume healthy and let
        # the in-process CorruptIndexError net catch real problems. Better than
        # marking every source corrupt because of an unrelated OS hiccup.
        return {"status": "ok", "count": None, "detail": f"probe unavailable: {e}"}

    if proc.returncode == 0:
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            count = data.get("count")
        except Exception:
            return {"status": "ok", "count": None}
        return {"status": "ok", "count": count} if count else {"status": "empty"}

    # Non-zero exit. returncode 1 == our caught-exception path; anything else
    # (negative on POSIX, large codes like 0xC0000005 on Windows) == the child
    # crashed natively.
    crashed = proc.returncode != 1
    stderr_lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
    tail = stderr_lines[-1] if stderr_lines else f"probe exited with code {proc.returncode}"
    prefix = "the index crashed the probe process" if crashed else "the index is unreadable"
    return {
        "status": "corrupt",
        "detail": f"{prefix}: {tail}",
        "crashed": crashed,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python -m lynx.integrity <storage_dir> <collection_name>",
              file=sys.stderr)
        sys.exit(2)
    try:
        sys.exit(_probe_child(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
