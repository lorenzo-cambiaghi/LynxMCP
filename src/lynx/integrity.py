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
import time
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
        sample = collection.get(limit=1, include=["embeddings"])
    except Exception:
        # A get() failure on a non-empty collection still signals trouble,
        # but count() is the authoritative health signal; don't fail here.
        sample = None

    # And exercise the KNN path, which is the one searches actually use and the
    # one that fails on a vector segment out of step with the metadata: count()
    # and get() read sqlite and stay green while every query raises "Error
    # querying knn" / "Error finding id". A probe that never queries reports a
    # search-dead index as healthy — which is how this went unnoticed.
    if count:
        vectors = (sample or {}).get("embeddings")
        dim = len(vectors[0]) if vectors is not None and len(vectors) else None
        if dim:
            # Raising here exits non-zero via __main__ = the "corrupt" verdict.
            collection.query(query_embeddings=[[0.0] * dim], n_results=1)

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
    usage = _classify_store_usage(storage_dir)
    if usage in ("other", "self"):
        # A transient handle — an AV scan, a backup tool, the search indexer,
        # or one of our own short-lived sqlite probes — shouldn't brand the
        # store as held. Any holder that matters (a serve's Chroma client, our
        # own cached one) easily survives this pause; a scanner's does not.
        time.sleep(0.15)
        usage = _classify_store_usage(storage_dir)
    return usage


def store_in_use_by_other_process(storage_dir) -> bool:
    """Guard for destructive operations (reset / wipe): True when ANOTHER
    process currently holds the store open. Deleting files a live process
    holds open half-succeeds on Windows — the unlocked files go first, then
    the delete fails on the held sqlite — turning a healthy index into a
    genuinely broken one."""
    return _store_usage(Path(storage_dir)) == "other"


def _claim_based_usage(storage_dir: Path) -> str:
    """Who holds this store, according to the ownership claim on disk.

    Only Windows can ask the OS which process has a file open. Everywhere else
    this is the answer, because a Lynx process that opens a store leaves a
    claim naming its pid (see lynx/ownership.py). Without it, the guards that
    refuse to wipe or heal a store somebody is using were Windows-only, and a
    `lynx reset` on macOS or Linux would delete files out from under a running
    server.
    """
    from . import ownership

    owner = ownership.owner_of(storage_dir)
    if owner is None:
        return "unknown"
    return "self" if owner.get("pid") == os.getpid() else "other"


def _classify_store_usage(storage_dir: Path) -> str:
    """Single-shot classification behind `_store_usage` (which re-checks to
    filter transient holders)."""
    if os.name != "nt":
        return _claim_based_usage(storage_dir)
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


def _probe_once(storage_dir, collection_name: str, budget: float):
    """Run the child probe once. A verdict dict, or None if it timed out."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "lynx.integrity",
             str(storage_dir), collection_name,
             # Child's self-destruct deadline: strictly after our kill, so it
             # only ever fires for orphans (see _self_destruct_after).
             str(int(budget) + 30)],
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return None
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
    return {"status": "corrupt", "detail": f"{prefix}: {tail}", "crashed": crashed}


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
                                                     # the store AND a probe did
                                                     # not finish — healthy, just
                                                     # not verifiable right now
        {"status": "unverified", "detail": "..."}    # probe timed out — NOT proof
                                                     # of corruption (slow disk,
                                                     # large index, or another
                                                     # Lynx process holding it)

    A genuinely corrupt or version-incompatible index makes the child exit
    non-zero (caught exception) or crash natively — those are the only signals
    we treat as ``corrupt``. A store held open by another process (typically
    `lynx serve`) is probed like any other, on a shorter budget: ChromaDB does
    let a second process read a live store, so sharing one index across
    sessions is the normal case, and ``in_use`` is now only what we answer when
    such a store also fails to respond. A residual *timeout* is ambiguous: a
    healthy probe returns in ~1s, so we retry once with a larger budget and, if
    it still doesn't finish, report ``unverified`` — the host still won't open
    the index (crash-safety is preserved), but we don't tell the user to wipe a
    possibly-healthy one.
    """
    storage_dir = Path(storage_dir)
    # Nothing on disk yet → a fresh source, not a corrupt one.
    if not (storage_dir / "chroma.sqlite3").exists():
        return {"status": "empty"}

    usage = _store_usage(storage_dir)
    if usage == "other":
        # Held by another Lynx process. That used to end the check here,
        # because a second reader was believed to block forever on Chroma's
        # lock. Measured on 1.5.9 against a live `lynx serve`, it does not: a
        # second process opens the store, counts it and runs a KNN query in
        # about a second. So probe it like any other store and let several
        # sessions share one index (see lynx/ownership.py). The budget is kept
        # short: if a machine or version really does block, we fall back to the
        # old `in_use` answer in seconds rather than minutes.
        result = _probe_once(storage_dir, collection_name,
                             budget=min(timeout, 15.0))
        if result is not None:
            return result
        return {
            "status": "in_use",
            "detail": "the index is open in another running Lynx process "
                      "(usually `lynx serve`) and did not answer a probe here; "
                      "it is not corrupt",
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
        result = _probe_once(storage_dir, collection_name, budget)
        if result is not None:
            return result
        timed_out_after = budget

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


# ---------------------------------------------------------------------------
# WAL wedge: fingerprint + surgical heal
# ---------------------------------------------------------------------------

def inspect_wal(storage_dir) -> dict | None:
    """Read-only WAL fingerprint of a store, via plain sqlite.

    Never goes through a Chroma client: on the state this looks for, any
    Chroma client deadlocks forever at zero CPU — that's the very failure
    being diagnosed. A process killed mid-write can leave pending rows in
    ``embeddings_queue`` with a segment's ``max_seq_id`` behind the queue
    tail; chromadb (observed on 1.5.9) then hangs on the first read, from
    any process, even on a byte-for-byte copy of the store.

    Returns None when there is no store, else a dict:

        pending_ops       rows still in embeddings_queue
        unapplied_ops     rows above the lowest segment watermark, i.e. the
                          ones at least one segment has not consumed yet
        oldest_pending_s  age in seconds of the oldest pending row (0 if none)
        lagging_segments  segments whose max_seq_id trails the queue tail
        stale_locks       rows in acquire_write. Chroma leaves one behind per
                          open (a healthy store accumulates them), so this is
                          bookkeeping, not evidence of a killed writer.
        affected_files    file_path values named by the UNAPPLIED rows — the
                          files whose chunks a lagging segment never received.
        wedged            True on the deadlock FINGERPRINT: pending ops older
                          than 10 minutes with a lagging segment. Necessary,
                          not sufficient: a clean build whose size is not a
                          multiple of the HNSW sync threshold legitimately
                          ends in this exact state (queue rows the vector
                          segment replays on the next open), and stays there
                          while idle. The store either answers a probe in
                          about a second (healthy) or hangs on it forever
                          (wedged) — so callers MUST confirm with
                          ``check_index`` before acting on this flag. Skip
                          the verdict when another process holds the store.
        discarding_writes True on a second, quieter wedge: the queue is EMPTY
                          while segments still hold a high watermark. Because
                          `embeddings_queue.seq_id` is an INTEGER PRIMARY KEY
                          with no AUTOINCREMENT, an emptied queue restarts
                          numbering at 1 — and every segment sitting at N > 1
                          treats those rows as already applied and drops them.
                          The store then ACCEPTS every write and stores none,
                          without raising. Purging the queue and leaving the
                          watermarks behind (what `heal_wal` used to do) is
                          exactly how a store gets here.
        watermark         highest segment watermark (0 when clean)
    """
    import sqlite3

    sqlite_path = Path(storage_dir) / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None

    info = {"pending_ops": 0, "unapplied_ops": 0, "oldest_pending_s": 0.0,
            "lagging_segments": 0, "stale_locks": 0, "affected_files": [],
            "wedged": False, "watermark": 0, "discarding_writes": False}
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        def one(sql, default=0):
            # Tables differ across chroma versions — a missing one just
            # means "no signal", not an error.
            try:
                row = db.execute(sql).fetchone()
                return row[0] if row and row[0] is not None else default
            except sqlite3.Error:
                return default

        # The sequence anchor (see _anchor_queue_sequence) is an inert row that
        # lives in the queue on purpose: it must never read as a pending write,
        # or every anchored store would look permanently wedged.
        info["pending_ops"] = one(
            f"SELECT count(*) FROM embeddings_queue"
            f" WHERE id != '{SEQ_ANCHOR_ID}'")
        if info["pending_ops"]:
            info["oldest_pending_s"] = float(one(
                f"SELECT (julianday('now') - julianday(min(created_at))) * 86400"
                f" FROM embeddings_queue WHERE id != '{SEQ_ANCHOR_ID}'"))
            info["lagging_segments"] = one(
                "SELECT count(*) FROM max_seq_id WHERE seq_id <"
                " (SELECT max(seq_id) FROM embeddings_queue)")
            # Rows every segment has already consumed are just waiting for
            # chroma's log purge; only the ones above the lowest watermark
            # are genuinely unapplied. Naming the files of ALL queued rows
            # (as this used to) made a heal re-index files whose chunks were
            # perfectly fine. No segment row at all = nothing applied yet.
            floor = one("SELECT min(seq_id) FROM max_seq_id", default=0)
            info["unapplied_ops"] = one(
                f"SELECT count(*) FROM embeddings_queue"
                f" WHERE id != '{SEQ_ANCHOR_ID}' AND seq_id > {int(floor)}")
            files = set()
            try:
                for (meta,) in db.execute(
                        f"SELECT metadata FROM embeddings_queue"
                        f" WHERE id != '{SEQ_ANCHOR_ID}' AND seq_id > {int(floor)}"):
                    try:
                        m = json.loads(meta) if meta else {}
                    except ValueError:
                        continue
                    fp = m.get("file_path") or (m.get("metadata") or {}).get("file_path")
                    if fp:
                        files.add(fp)
            except sqlite3.Error:
                pass
            info["affected_files"] = sorted(files)
        info["stale_locks"] = one("SELECT count(*) FROM acquire_write")
        info["watermark"] = one("SELECT max(seq_id) FROM max_seq_id")
        queue_tail = one("SELECT max(seq_id) FROM embeddings_queue")
    finally:
        db.close()

    info["wedged"] = bool(
        info["pending_ops"]
        and info["lagging_segments"]
        and info["oldest_pending_s"] > 600
    )
    # Queue numbering below the segments' watermark = every future write is
    # numbered under what they already consumed, so it is accepted and dropped.
    # Silent, permanent, and invisible to count() — the index just stops moving.
    # An anchored queue (tail == watermark) is precisely the cured state, which
    # is why the comparison is against the tail and not against zero.
    info["discarding_writes"] = bool(
        not info["pending_ops"] and info["watermark"] > queue_tail
    )
    return info


def heal_wal(storage_dir) -> dict:
    """Surgically unwedge a store: purge the pending queue and the stale
    write locks, then drop the affected files from ``file_hashes.json`` so
    the next build / watcher pass re-indexes them — nothing is silently
    lost, and no full rebuild is needed.

    Refuses while ANY process (this one included) holds the store: the
    purge must not race a live Chroma client. Verified against the real
    wedge of 2026-07-04: the integrity probe went from an infinite hang to
    ``ok`` the moment the queue was purged.

    This is surgery, not hygiene: on a HEALTHY store the queued rows are the
    tail the vector segment replays on its next open, and purging them
    throws those vectors away (they come back only through a re-index). The
    caller is expected to have confirmed the wedge with ``check_index``
    first — see ``inspect_wal`` on why the fingerprint alone is not proof.
    """
    import sqlite3

    storage_dir = Path(storage_dir)
    sqlite_path = storage_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"no store at {sqlite_path}")
    usage = _store_usage(storage_dir)
    if usage in ("other", "self"):
        holder = ("another running Lynx process (usually `lynx serve`)"
                  if usage == "other" else "this very process")
        raise RuntimeError(
            f"the store is currently open in {holder} — stop it and retry; "
            f"healing must not race a live client"
        )

    fingerprint = inspect_wal(storage_dir) or {}
    affected = fingerprint.get("affected_files", [])

    db = sqlite3.connect(str(sqlite_path))
    try:
        purged_ops = db.execute("DELETE FROM embeddings_queue").rowcount
        try:
            purged_locks = db.execute("DELETE FROM acquire_write").rowcount
        except sqlite3.Error:
            purged_locks = 0
        # Order matters: fast-forward first (a lagging segment blocks every
        # later write), then anchor the queue above the resulting watermark.
        forwarded = _fast_forward_lagging_segments(db)
        anchored = _anchor_queue_sequence(db)
        db.commit()
    finally:
        db.close()

    dropped = _drop_from_file_hashes(storage_dir, affected)
    return {"purged_ops": purged_ops, "purged_locks": purged_locks,
            "forwarded_segments": forwarded, "anchored_seq": anchored,
            "reindex_files": dropped}


SEQ_ANCHOR_ID = "__lynx_seq_anchor__"


def _fast_forward_lagging_segments(db) -> int:
    """Bring segments left behind the queue's history up to the leading one.

    A segment consumes the WAL from its own watermark forward. If the rows it
    still needs have been purged (which is what unwedging a queue does), they
    are gone: the segment can never reach the tail and simply stops consuming —
    and every later write waits on it, so indexing crawls at a few chunks per
    minute while the vector index stays frozen. Observed exactly so: a metadata
    segment at 64116 and a vector segment stuck at 63966 since the day it was
    interrupted.

    Moving it forward is the only way out, and it is honest: the chunks whose
    vectors were lost stay in the metadata WITHOUT a vector, where
    `inspect_coverage` finds them and `heal_coverage` re-indexes their files.
    Nothing is silently dropped — it is queued for rebuild.

    Only safe with an empty queue (nothing left to legitimately apply).
    Returns the number of segments moved.
    """
    import sqlite3

    try:
        pending = db.execute(
            "SELECT count(*) FROM embeddings_queue WHERE id != ?",
            (SEQ_ANCHOR_ID,)).fetchone()[0]
        if pending:
            return 0
        top = db.execute("SELECT max(seq_id) FROM max_seq_id").fetchone()[0]
        if not top:
            return 0
        cur = db.execute("UPDATE max_seq_id SET seq_id = ? WHERE seq_id < ?",
                         (top, top))
        return cur.rowcount or 0
    except sqlite3.Error:
        return 0


def _anchor_queue_sequence(db) -> int:
    """Make an emptied queue resume numbering ABOVE the segments' watermarks.

    ``embeddings_queue.seq_id`` is an INTEGER PRIMARY KEY with no AUTOINCREMENT:
    sqlite hands out ``max(seq_id) + 1``, so an emptied queue restarts at 1. A
    segment parked at 64111 then discards every new row as already-applied —
    the store accepts writes and stores nothing, forever, without raising.

    The obvious repair (lower the watermarks) is WRONG and was verified to be
    so: a vector segment carries a persisted HNSW built up to its watermark, so
    telling it "you are at 0" asks it to replay tens of thousands of operations
    the WAL no longer holds. It then falls behind the queue tail — the exact
    fingerprint `inspect_wal` calls wedged — and chromadb deadlocks at zero CPU
    on the next read.

    So instead of moving the segments down to the queue, we move the queue up
    to the segments: one sentinel row at ``max(watermark)``. Every segment has
    already consumed that seq (rows are read with ``seq_id >`` watermark), so
    it is inert — and the next real write lands at watermark + 1, where the
    segments expect it. Returns the anchored seq (0 when nothing was needed).
    """
    import sqlite3

    try:
        pending = db.execute(
            "SELECT count(*) FROM embeddings_queue WHERE id != ?",
            (SEQ_ANCHOR_ID,)).fetchone()[0]
        if pending:
            # Rows still waiting: the numbering is in use and consistent.
            return 0
        watermark = db.execute(
            "SELECT max(seq_id) FROM max_seq_id").fetchone()[0] or 0
        tail = db.execute(
            "SELECT max(seq_id) FROM embeddings_queue").fetchone()[0] or 0
        if watermark <= tail:
            return 0
        db.execute(
            "INSERT OR REPLACE INTO embeddings_queue "
            "(seq_id, operation, topic, id) VALUES (?, 0, ?, ?)",
            (watermark, "lynx:seq-anchor", SEQ_ANCHOR_ID))
        return watermark
    except sqlite3.Error:
        return 0


# ---------------------------------------------------------------------------
# Coverage drift: the SHA cache claims files the index does not hold
# ---------------------------------------------------------------------------

def inspect_coverage(storage_dir) -> dict | None:
    """Read-only audit of the store's SHA cache against what the index HOLDS.

    ``file_hashes.json`` is what ``_partition_files`` trusts to decide which
    files need work: a file listed there with a matching SHA is classified
    *unchanged* and skipped forever. So an entry written for a file whose
    chunks never made it into the index makes that file permanently invisible
    to search — no error, no retry, and `count()` stays healthy because the
    other files are fine. (Real incident 2026-07-03 on the `framework` source:
    202 such files, unnoticed for 25 days.)

    This compares the two, via plain sqlite — never a Chroma client, which
    would contend with a running `lynx serve` for the store lock.

    It also catches the mirror-image damage one layer down: chunks that ARE in
    the metadata but whose vector never reached the HNSW segment (an insert
    interrupted between the two). Those are what make every search die with
    "Error querying knn" / "Error finding id" while count() reports a full,
    healthy index — the metadata hands out ids the vector index cannot resolve.

    Returns None when there is no store, else a dict::

        cached_files       entries in file_hashes.json
        indexed_files      distinct file_path values present in the index
        phantom_files      cached as indexed, absent from the index, and not
                           recorded as legitimately chunk-less (`chunks: 0`)
        empty_files        cached with chunks == 0 — expected to be absent
        unvectorized       chunks in the metadata with no vector (0 = clean)
        unvectorized_files the files those chunks belong to
        drifted            True when either kind of damage is present
    """
    import sqlite3

    storage_dir = Path(storage_dir)
    sqlite_path = storage_dir / "chroma.sqlite3"
    hashes_path = storage_dir / "file_hashes.json"
    if not sqlite_path.exists():
        return None

    def norm(p):
        return os.path.normcase(os.path.normpath(str(p)))

    indexed = set()
    unvectorized_files: set = set()
    unvectorized = 0
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata"
            " WHERE key = 'file_path' AND string_value IS NOT NULL"
        )
        indexed = {norm(r[0]) for r in rows}
        vector_ids = _vector_segment_ids(storage_dir)
        if vector_ids is not None:
            orphan_ids = [eid for (eid,) in
                          db.execute("SELECT embedding_id FROM embeddings")
                          if eid not in vector_ids]
            # Not every chunk missing from the HNSW is damage. Chroma persists
            # the vector segment on a threshold, so the tail of a fresh build
            # legitimately sits in the queue waiting to be applied — it lands on
            # the next open. Damage is when the rows are NOT in the queue any
            # more: then the vectors are gone and only a re-index brings them
            # back. Without this distinction the check would cry wolf after
            # every single build, which is the fastest way to be ignored.
            recoverable = _recoverable_from_queue(db)
            unvectorized = max(0, len(orphan_ids) - recoverable)
            if unvectorized:
                unvectorized_files = _files_of_embeddings(db, orphan_ids)
    except sqlite3.Error:
        # An unreadable metadata table is a different failure (corruption);
        # `check_index` is the tool for that. Don't guess about coverage.
        return None
    finally:
        db.close()

    entries = {}
    if hashes_path.exists():
        try:
            data = json.loads(hashes_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                entries = _hash_cache_files(data)
        except (OSError, ValueError):
            entries = {}

    phantom, empty = [], []
    for path, meta in entries.items():
        if norm(path) in indexed:
            continue
        # `chunks: 0` means the file legitimately produced nothing. Entries
        # written before that field existed carry None and are treated as
        # "expected present" — which is what surfaces the historical damage.
        if isinstance(meta, dict) and meta.get("chunks") == 0:
            empty.append(path)
        else:
            phantom.append(path)

    return {
        "cached_files": len(entries),
        "indexed_files": len(indexed),
        "phantom_files": sorted(phantom),
        "empty_files": sorted(empty),
        "unvectorized": unvectorized,
        "unvectorized_files": sorted(unvectorized_files),
        "drifted": bool(phantom) or bool(unvectorized),
    }


def _vector_segment_ids(storage_dir: Path):
    """The chunk ids the persisted HNSW segment actually knows, or None when
    that can't be read (no segment yet, or a Chroma build that stores its
    mapping elsewhere — then we simply make no claim rather than a wrong one).

    Read from the segment's own ``index_metadata.pickle`` rather than through a
    client: opening the store is exactly what deadlocks against a live serve.
    """
    import pickle

    try:
        for child in Path(storage_dir).iterdir():
            meta = child / "index_metadata.pickle"
            if child.is_dir() and meta.is_file():
                with open(meta, "rb") as fh:
                    data = pickle.load(fh)
                mapping = data.get("id_to_label") if isinstance(data, dict) else None
                if isinstance(mapping, dict):
                    return set(mapping)
    except Exception:
        return None
    return None


def _recoverable_from_queue(db) -> int:
    """How many not-yet-vectorized rows the WAL can still deliver on its own:
    queue rows above the lowest segment watermark. They need no repair — the
    next open applies them."""
    import sqlite3

    try:
        floor = db.execute("SELECT min(seq_id) FROM max_seq_id").fetchone()[0]
        if floor is None:
            return 0
        return db.execute(
            "SELECT count(*) FROM embeddings_queue WHERE seq_id > ? AND id != ?",
            (floor, SEQ_ANCHOR_ID)).fetchone()[0] or 0
    except sqlite3.Error:
        return 0


def _files_of_embeddings(db, embedding_ids) -> set:
    """file_path values of the given chunk ids. Chunked to stay under sqlite's
    variable limit — a broken batch can easily be thousands of ids."""
    import sqlite3

    files = set()
    batch_size = 500
    for start in range(0, len(embedding_ids), batch_size):
        batch = embedding_ids[start:start + batch_size]
        placeholders = ",".join("?" * len(batch))
        try:
            rows = db.execute(
                "SELECT DISTINCT em.string_value FROM embedding_metadata em"
                " JOIN embeddings e ON e.id = em.id"
                " WHERE em.key = 'file_path' AND em.string_value IS NOT NULL"
                f" AND e.embedding_id IN ({placeholders})",
                batch,
            )
            files.update(r[0] for r in rows)
        except sqlite3.Error:
            continue
    return files


def heal_coverage(storage_dir) -> dict:
    """Drop the phantom entries from the SHA cache so the next build / watcher
    pass re-indexes exactly those files — no full rebuild, same surgical shape
    as `heal_wal`.

    Refuses while any process holds the store: a live `lynx serve` keeps the
    cache in memory and would write it straight back over our edit.
    """
    storage_dir = Path(storage_dir)
    usage = _store_usage(storage_dir)
    if usage in ("other", "self"):
        holder = ("another running Lynx process (usually `lynx serve`)"
                  if usage == "other" else "this very process")
        raise RuntimeError(
            f"the store is currently open in {holder} — stop it and retry; "
            f"the in-memory cache would overwrite the repair"
        )

    report = inspect_coverage(storage_dir)
    if not report:
        raise FileNotFoundError(f"no readable store at {storage_dir}")
    # Both damages heal the same way: forget the file, and the next pass
    # re-indexes it — which for the unvectorized ones also deletes the
    # dangling chunks first (`_delete_file_chunks` runs before the insert).
    targets = list(report["phantom_files"]) + list(report["unvectorized_files"])
    dropped = _drop_from_file_hashes(storage_dir, targets)
    return {
        "phantom": len(report["phantom_files"]),
        "unvectorized": report["unvectorized"],
        "unvectorized_files": len(report["unvectorized_files"]),
        "reindex_files": dropped,
        "indexed_files": report["indexed_files"],
    }


def _hash_cache_files(data: dict) -> dict:
    """The per-file mapping inside a loaded ``file_hashes.json``.

    The on-disk shape is ``{"schema_version": N, "config_snapshot": {...},
    "files": {abs_path: {...}}}``; a legacy cache was the flat mapping itself.
    Returning the right sub-dict (by identity, so callers can mutate it) is
    what makes the healers actually touch the entries — iterating the top level
    of the current schema only ever sees the three envelope keys, so a heal
    reported success while dropping nothing."""
    files = data.get("files")
    return files if isinstance(files, dict) else data


def _drop_from_file_hashes(storage_dir: Path, files) -> list:
    """Remove `files` from the store's SHA cache so the next build re-indexes
    them. Path-normalized matching — the cache may use either slash style."""
    hashes_path = storage_dir / "file_hashes.json"
    if not files or not hashes_path.exists():
        return []

    def norm(p):
        return os.path.normcase(os.path.normpath(str(p)))

    wanted = {norm(f) for f in files}
    try:
        data = json.loads(hashes_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    entries = _hash_cache_files(data)
    dropped = [k for k in entries if norm(k) in wanted]
    for k in dropped:
        del entries[k]
    if dropped:
        hashes_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dropped


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
