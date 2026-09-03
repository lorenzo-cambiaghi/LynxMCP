"""Which Lynx process owns a source's index.

ChromaDB lets several processes READ one store at the same time. Measured on
1.5.9 against a live `lynx serve`: a second process opens the store, counts it
and runs a KNN query in about a second, and both processes stay healthy. What
must not happen twice is WRITING, because two watchers indexing the same files
would race on the same write-ahead log.

So exactly one process claims a source as **owner**: it runs the watcher and
performs every write. Any other session opens the same store as a **follower**:
it searches normally and picks up the owner's writes on demand, but never
writes.

The claim is a small JSON file in the source's storage directory, and it is
deliberately advisory:

* the owner rewrites it every `_HEARTBEAT_SEC`, so a claim nobody has touched
  for `_STALE_AFTER_SEC` is abandoned even if its pid is still alive. That
  covers the two cases a pid check alone gets wrong: a pid the OS recycled onto
  an unrelated process, and an owner that stopped running without exiting (a
  laptop asleep for an hour is the ordinary version of this);
* an owner whose claim is taken from it while it sleeps is told, through the
  `on_lost` callback, so it can stand down instead of becoming a second writer;
* every write is an atomic replace, so a reader never sees half a claim, and
  claiming ends with a read-back: when two processes start at the same instant
  they both write, and exactly one sees itself afterwards;
* failing to claim never blocks a session from reading.
"""
from __future__ import annotations

import errno
import json
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path

OWNER_FILE = "owner.json"

# The owner refreshes its claim this often...
_HEARTBEAT_SEC = 30.0
# ...and a claim nobody has refreshed for this long is up for grabs. Five
# missed beats: generous enough that a busy machine never loses an index it is
# still indexing, short enough that a killed session frees it within minutes.
_STALE_AFTER_SEC = 150.0
# Time given to a competing writer to land before we read back who won.
_SETTLE_SEC = 0.15

# path -> (stop_event, thread). Module-level so `release` can find them.
_HEARTBEATS: dict = {}
_HEARTBEATS_LOCK = threading.Lock()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return platform.node() or "unknown"


def _key(storage_dir) -> str:
    p = Path(storage_dir)
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def pid_alive(pid: int) -> bool:
    """True when a process with this pid exists.

    Errs on the side of "alive": a claim we cannot disprove is left alone, so
    the worst case is a session that reads instead of owning, never two owners.
    """
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True          # exists, owned by someone else
        except OSError:
            return True
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_INVALID_PARAMETER = 87

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # "invalid parameter" is how Windows says "no such pid"; anything else
        # (access denied on a service, say) means it is there.
        return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# Reading and writing the claim
# ---------------------------------------------------------------------------

def read_claim(storage_dir) -> dict | None:
    """The claim on disk, whatever its state. None when there is none."""
    path = Path(storage_dir) / OWNER_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_claim(storage_dir, payload: dict) -> bool:
    """Put `payload` in place atomically. A reader never sees a partial claim."""
    storage_dir = Path(storage_dir)
    path = storage_dir / OWNER_FILE
    tmp = storage_dir / f"{OWNER_FILE}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as e:
        if e.errno not in (errno.EACCES, errno.EROFS, errno.EPERM):
            _log(f"[owner] could not write {path}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _payload() -> dict:
    return {"pid": os.getpid(), "host": _host(), "beat": time.time(),
            "since": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _is_ours(claim) -> bool:
    return bool(claim and claim.get("pid") == os.getpid()
                and claim.get("host") == _host())


def owner_of(storage_dir) -> dict | None:
    """The LIVE owner of this store, or None when nobody owns it.

    A claim is live only while it is being refreshed. That is what makes a
    recycled pid, or an owner suspended with the machine, release the index
    instead of holding it forever.
    """
    claim = read_claim(storage_dir)
    if not claim:
        return None
    beat = claim.get("beat")
    if not isinstance(beat, (int, float)):
        beat = 0.0
    # Clocks move: a claim stamped in the future is treated as fresh, never as
    # infinitely stale.
    if time.time() - beat > _STALE_AFTER_SEC:
        return None
    if claim.get("host") != _host():
        # Shared storage: we cannot check that pid, so the heartbeat is the
        # only evidence we have, and it says the owner is alive.
        return claim
    pid = claim.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        return None
    return claim


def owned_by_this_process(storage_dir) -> bool:
    return _is_ours(read_claim(storage_dir))


# ---------------------------------------------------------------------------
# Claiming, keeping, releasing
# ---------------------------------------------------------------------------

def claim(storage_dir, on_lost=None) -> bool:
    """Try to become the owner of this store. True when we are.

    `on_lost` is called if the claim is ever taken from us while we still hold
    it, which happens when this process was stopped long enough for its
    heartbeat to go stale. The caller is expected to stand down there rather
    than keep writing alongside the new owner.
    """
    storage_dir = Path(storage_dir)
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log(f"[owner] cannot use {storage_dir}: {e}; running as follower")
        return False

    if owner_of(storage_dir) is not None:
        return False                      # someone alive holds it

    # Nobody live holds it: stake ours and read back who actually landed last.
    # Two processes starting together both write, and exactly one sees itself.
    if not _write_claim(storage_dir, _payload()):
        return False
    time.sleep(_SETTLE_SEC)
    if not _is_ours(read_claim(storage_dir)):
        return False

    _start_heartbeat(storage_dir, on_lost)
    return True


def _start_heartbeat(storage_dir, on_lost=None) -> None:
    """Keep our claim fresh until we release it or lose it."""
    key = _key(storage_dir)
    stop = threading.Event()

    def beat():
        while not stop.wait(_HEARTBEAT_SEC):
            claim_now = read_claim(storage_dir)
            if not _is_ours(claim_now):
                # Someone took it while we were not running. Standing down is
                # the only safe answer: two writers on one store is the thing
                # this whole file exists to prevent.
                _log(f"[owner] lost the claim on {storage_dir}; standing down")
                if on_lost is not None:
                    try:
                        on_lost()
                    except Exception as e:
                        _log(f"[owner] stand-down handler failed: {e}")
                break
            _write_claim(storage_dir, _payload())
        with _HEARTBEATS_LOCK:
            if _HEARTBEATS.get(key, (None, None))[0] is stop:
                _HEARTBEATS.pop(key, None)

    thread = threading.Thread(target=beat, name=f"lynx-owner-{Path(storage_dir).name}",
                              daemon=True)
    with _HEARTBEATS_LOCK:
        previous = _HEARTBEATS.get(key)
        if previous is not None:
            previous[0].set()
        _HEARTBEATS[key] = (stop, thread)
    thread.start()


def release(storage_dir) -> None:
    """Give up ownership, but only if the claim on disk is still ours."""
    key = _key(storage_dir)
    with _HEARTBEATS_LOCK:
        entry = _HEARTBEATS.pop(key, None)
    if entry is not None:
        entry[0].set()
    if not owned_by_this_process(storage_dir):
        return
    try:
        (Path(storage_dir) / OWNER_FILE).unlink()
    except OSError:
        pass


def describe(storage_dir) -> str:
    """One line naming the live owner, for messages the user reads."""
    claim = owner_of(storage_dir)
    if claim is None:
        return "no other Lynx process owns this index"
    where = "" if claim.get("host") == _host() else f" on {claim.get('host')}"
    return (f"another Lynx process owns this index (pid {claim.get('pid')}"
            f"{where}, since {claim.get('since')})")
