"""ChromaDB concurrent-access detection.

SQLite (the backend ChromaDB uses) supports concurrent READS but only
one WRITER at a time. When `lynx serve` is running in the background to
feed an MCP client, the web UI must NOT try to write to the same
ChromaDB or both processes start getting `SQLITE_BUSY` errors.

This module provides a cheap probe: `is_storage_locked(path)` opens
ChromaDB on the storage directory, does a tiny write attempt (creating
+ deleting a temp metadata key on a "_lynx_probe" collection), and
returns True iff the write fails with a locking error.

Results are cached for 30 seconds because the probe itself is non-zero
cost (~50ms cold) and the lock state changes slowly (you don't start
and stop `lynx serve` every second).
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# Per-path cache: {path_str: (locked: bool, checked_at: float)}
_CACHE: dict = {}
_CACHE_TTL_SEC = 30.0


def is_storage_locked(storage_path: Path) -> bool:
    """Return True iff another process holds a write lock on the
    ChromaDB at `storage_path` (or the path doesn't yet exist).

    For a not-yet-built source the result is False (we can write the
    moment a build creates the DB). The cache prevents stampedes when
    the UI hits this on every render.
    """
    key = str(storage_path.resolve()) if storage_path.exists() else str(storage_path)
    now = time.time()
    cached = _CACHE.get(key)
    if cached is not None:
        locked, ts = cached
        if now - ts < _CACHE_TTL_SEC:
            return locked
    locked = _probe(storage_path)
    _CACHE[key] = (locked, now)
    return locked


def invalidate_cache(storage_path: Path | None = None) -> None:
    """Drop cached results. Called after a successful write so the next
    probe re-checks immediately. With `storage_path=None`, drops all."""
    if storage_path is None:
        _CACHE.clear()
        return
    key = str(storage_path.resolve()) if storage_path.exists() else str(storage_path)
    _CACHE.pop(key, None)


def _probe(storage_path: Path) -> bool:
    """The actual probe. Returns True if write is currently blocked."""
    if not storage_path.exists():
        return False  # nothing to lock yet
    chroma_sqlite = storage_path / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        return False  # source not built yet — nobody can hold a write lock

    # Try to open SQLite directly and do a BEGIN IMMEDIATE — this is the
    # cheapest way to ask "could I write right now?" without going through
    # ChromaDB's full client which is slow to initialize.
    #
    # `timeout=0.1` keeps the probe responsive: if another writer is
    # holding the lock, we get SQLITE_BUSY after 100ms instead of the
    # default 5 second wait.
    try:
        conn = sqlite3.connect(str(chroma_sqlite), timeout=0.1)
    except sqlite3.OperationalError as e:
        _log(f"[lock] probe failed to open {chroma_sqlite}: {e}")
        return True  # treat "can't even open" as locked
    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError as e:
        # "database is locked" / "database is busy"
        msg = str(e).lower()
        if "lock" in msg or "busy" in msg:
            return True
        # Some other operational error — log and treat as locked to be safe
        _log(f"[lock] unexpected SQLite error on probe: {e}")
        return True
    finally:
        conn.close()
