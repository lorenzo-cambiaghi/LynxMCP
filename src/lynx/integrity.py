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

There is a third failure mode that is NOT corruption: ChromaDB's Rust core
allows only one process to actively use a persist directory. A second process
can *open* the store, but its first real read (``count()``) blocks
indefinitely on the first process's lock — no timeout, no error. So when
`lynx serve` is running, any probe of a store it holds can never finish. We
detect that case up front (see ``_store_usage``) and report ``in_use``
instead of burning the full timeout budget and alarming the user.

Run as a module for the child side::

    python -m lynx.integrity <storage_dir> <collection_name> [<max_lifetime_s>]

Exit 0 + JSON ``{"ok": true, "count": N}`` on success; non-zero on any failure
(our own ``exit(1)`` for a caught exception, or an OS crash code for a segfault).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
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


def _self_destruct_after(seconds: float) -> None:
    """Orphan insurance for the probe child.

    The parent kills us when its timeout fires — but only if the parent is
    still alive. If it dies first (Ctrl-C, closed terminal, killed session),
    Windows does not reap the child, and a probe blocked on another process's
    store lock would linger forever, holding a handle on the store and making
    every LATER probe look wedged too. A daemon timer hard-exits this process
    once the parent's budget is over. Exit code 3 is never observed by a live
    parent: our deadline is strictly later than the parent's kill."""
    timer = threading.Timer(seconds, os._exit, args=(3,))
    timer.daemon = True
    timer.start()


# ---------------------------------------------------------------------------
# Store-usage detection: is another process already holding this store?
# ---------------------------------------------------------------------------

def _sqlite_handle_is_held(sqlite_path: Path) -> bool:
    """Windows only: True iff ANY process (this one included) has an open
    handle on the file. Asking for the file with zero sharing fails with
    ERROR_SHARING_VIOLATION exactly when another handle exists."""
    import ctypes
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    ERROR_SHARING_VIOLATION = 32

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [
        ctypes.c_wchar_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    k32.CreateFileW.restype = ctypes.c_void_p
    invalid_handle = ctypes.c_void_p(-1).value

    handle = k32.CreateFileW(str(sqlite_path), GENERIC_READ, 0, None,
                             OPEN_EXISTING, 0, None)
    if handle in (None, invalid_handle):
        return ctypes.get_last_error() == ERROR_SHARING_VIOLATION
    k32.CloseHandle(ctypes.c_void_p(handle))
    return False


def _handle_holders(sqlite_path: Path):
    """PIDs of the processes holding the file open, via the Windows Restart
    Manager (the same machinery Explorer uses for "file in use by ...").
    Returns a set of PIDs, or None when it can't be determined."""
    import ctypes
    from ctypes import wintypes

    try:
        rm = ctypes.WinDLL("RstrtMgr")
    except OSError:
        return None

    class _RM_UNIQUE_PROCESS(ctypes.Structure):
        _fields_ = [("dwProcessId", wintypes.DWORD),
                    ("ProcessStartTime", wintypes.FILETIME)]

    class _RM_PROCESS_INFO(ctypes.Structure):
        _fields_ = [("Process", _RM_UNIQUE_PROCESS),
                    ("strAppName", ctypes.c_wchar * 256),
                    ("strServiceShortName", ctypes.c_wchar * 64),
                    ("ApplicationType", ctypes.c_int),
                    ("AppStatus", wintypes.ULONG),
                    ("dwSessionId", wintypes.DWORD),
                    ("bRestartable", wintypes.BOOL)]

    ERROR_MORE_DATA = 234
    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)  # CCH_RM_SESSION_KEY + 1
    if rm.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return None
    try:
        path = ctypes.c_wchar_p(str(sqlite_path))
        if rm.RmRegisterResources(session, 1, ctypes.byref(path),
                                  0, None, 0, None) != 0:
            return None
        needed = wintypes.UINT(0)
        count = wintypes.UINT(0)
        reason = wintypes.DWORD()
        rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count),
                          None, ctypes.byref(reason))
        if rc == 0:
            return set()
        infos = None
        while rc == ERROR_MORE_DATA:
            count = wintypes.UINT(needed.value)
            infos = (_RM_PROCESS_INFO * count.value)()
            rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count),
                              infos, ctypes.byref(reason))
        if rc != 0 or infos is None:
            return None
        return {infos[i].Process.dwProcessId for i in range(count.value)}
    finally:
        rm.RmEndSession(session)


def _store_usage(storage_dir: Path) -> str:
    """Classify who currently holds the store's sqlite open.

    Returns one of:

      "free"    — no open handles; the probe can run normally.
      "self"    — only THIS process holds it (e.g. the manager UI reloaded and
                  ChromaDB's module-level system cache still has the client).
                  A child probe would deadlock against our own lock, and the
                  store demonstrably opens fine here — treat as healthy.
      "other"   — another process holds it (usually a running `lynx serve`).
                  A probe can never finish; report `in_use` instead.
      "unknown" — no way to tell (POSIX, or detection failed). Fall back to
                  the probe-and-timeout path.
    """
    if os.name != "nt":
        return "unknown"
    sqlite_path = storage_dir / "chroma.sqlite3"
    try:
        if not _sqlite_handle_is_held(sqlite_path):
            return "free"
    except Exception:
        return "unknown"
    holders = _handle_holders(sqlite_path)
    if holders is None:
        # Someone holds a handle but we can't attribute it. In practice that
        # someone is `lynx serve`; saying "in use" beats hanging for minutes.
        return "other"
    if any(pid != os.getpid() for pid in holders):
        return "other"
    return "self" if holders else "free"


def check_index(
    storage_dir,
    collection_name: str,
    *,
    timeout: float = 60.0,
) -> dict:
    """Probe a source's index in a subprocess. Never raises, never opens the
    index in the calling process.

    Returns one of::

        {"status": "ok",         "count": N}
        {"status": "empty"}                          # nothing built yet / 0 chunks
        {"status": "corrupt",    "detail": "...", "crashed": bool}
        {"status": "in_use",     "detail": "..."}    # another Lynx process holds
                                                     # the store — healthy, just
                                                     # not verifiable right now
        {"status": "unverified", "detail": "..."}    # probe timed out — NOT proof
                                                     # of corruption (slow disk,
                                                     # large index, or another
                                                     # Lynx process holding it)

    A genuinely corrupt or version-incompatible index makes the child exit
    non-zero (caught exception) or crash natively — those are the only signals
    we treat as ``corrupt``. A store held open by another process (typically
    `lynx serve`) can never be probed: ChromaDB's core lock makes the child's
    ``count()`` block forever, so we detect that case up front and report
    ``in_use`` without spawning anything. A residual *timeout* is ambiguous:
    a healthy probe returns in ~1s, so we retry once with a larger budget and,
    if it still doesn't finish, report ``unverified`` — the host still won't
    open the index (crash-safety is preserved), but we don't tell the user to
    wipe a possibly-healthy one.
    """
    storage_dir = Path(storage_dir)
    # Nothing on disk yet → a fresh source, not a corrupt one.
    if not (storage_dir / "chroma.sqlite3").exists():
        return {"status": "empty"}

    usage = _store_usage(storage_dir)
    if usage == "other":
        return {
            "status": "in_use",
            "detail": "the index is open in another running Lynx process "
                      "(usually `lynx serve`) and can't be verified or opened "
                      "here until that process exits; it is not corrupt",
            "crashed": False,
        }
    if usage == "self":
        # Already open and in use by this very process — it can't fail a
        # fresh-open probe, and a child probe would block on our own lock.
        return {"status": "ok", "count": None,
                "detail": "index already open in this process"}

    # Two attempts: a transient lock/race usually clears by the retry, and the
    # second, longer budget gives a legitimately slow open room to finish.
    budgets = (timeout, timeout * 2)
    timed_out_after = 0.0
    for budget in budgets:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "lynx.integrity",
                 str(storage_dir), collection_name,
                 # Child's self-destruct deadline: strictly after our kill,
                 # so it only ever fires for orphans (see _self_destruct_after).
                 str(int(budget) + 30)],
                capture_output=True,
                text=True,
                timeout=budget,
            )
        except subprocess.TimeoutExpired:
            timed_out_after = budget
            continue  # retry with a larger budget before concluding anything
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
        # crashed natively. Either way this IS a real corruption signal.
        crashed = proc.returncode != 1
        stderr_lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        tail = stderr_lines[-1] if stderr_lines else f"probe exited with code {proc.returncode}"
        prefix = "the index crashed the probe process" if crashed else "the index is unreadable"
        return {
            "status": "corrupt",
            "detail": f"{prefix}: {tail}",
            "crashed": crashed,
        }

    # Every attempt timed out. Maybe a holder appeared after our up-front
    # check (e.g. `lynx serve` started meanwhile) — reclassify if so.
    if _store_usage(storage_dir) == "other":
        return {
            "status": "in_use",
            "detail": "the index is open in another running Lynx process "
                      "(usually `lynx serve`) and can't be verified or opened "
                      "here until that process exits; it is not corrupt",
            "crashed": False,
        }
    # Still nobody visible holding it. NOT proof of corruption — most often the
    # index is large, on a slow/AV-scanned disk, or locked by another Lynx
    # process we couldn't attribute. Surface it as such.
    return {
        "status": "unverified",
        "detail": f"could not verify the index within {timed_out_after:.0f}s — it may "
                  f"be large, on a slow disk, or locked by another running Lynx "
                  f"process; this is not necessarily corruption",
        "crashed": False,
    }


if __name__ == "__main__":
    # This runs as a fresh subprocess (not through cli.main), and an
    # HTTPS-inspecting antivirus re-injects SSLKEYLOGFILE into every new
    # process. Strip it here too so the probe can never itself abort on the
    # OPENSSL_Applink crash and get misread as a corrupt index.
    from .config import sanitize_tls_keylog_env
    sanitize_tls_keylog_env()

    if len(sys.argv) < 3:
        print("usage: python -m lynx.integrity <storage_dir> <collection_name> "
              "[<max_lifetime_s>]",
              file=sys.stderr)
        sys.exit(2)
    _self_destruct_after(float(sys.argv[3]) if len(sys.argv) > 3 else 600.0)
    try:
        sys.exit(_probe_child(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
