"""Tiny in-memory job tracker for long-running UI actions.

Right now the only consumer is the "build source" button on the source
detail page: clicking it kicks off `manager.update(source, force=True)`
in a daemon thread so the HTTP request returns immediately. The HTMX
widget then polls `/api/jobs/<id>` every second to render progress.

Why not Celery / RQ / arq? The UI is a personal management tool, not
a multi-user service. Jobs are per-process and don't need to survive
restarts — if the user kills the UI mid-build, the partial write is
fine (ChromaDB handles it) and they can just re-trigger. Pulling in a
broker would be ceremony without value.

State is in-memory, no persistence. `cleanup_old(max_age_sec)` evicts
finished jobs so the dict can't grow unbounded for a long-running UI.
"""
from __future__ import annotations

import io
import threading
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Job:
    """A single tracked unit of work."""
    id: str
    label: str                                # human-readable: "build demo"
    state: str = "queued"                     # queued | running | done | failed
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None
    # Captured stdout+stderr during the job. Stored as a single string
    # rather than a list — easier to render in <pre>, and the build
    # output is small (< few KB).
    log: str = ""
    # Optional grouping key — e.g. source name — so callers can ask
    # "is any job already running for source X?" before spawning a new one.
    group: Optional[str] = None
    metadata: dict = field(default_factory=dict)


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def create_job(
    target: Callable[[], None],
    *,
    label: str,
    group: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Job:
    """Spawn `target()` in a daemon thread and return the Job handle.

    The thread captures stdout+stderr into `job.log` so the UI can show
    what the build actually printed. On exception, the traceback is
    stored in `job.error` and `job.log`.
    """
    job = Job(
        id=str(uuid.uuid4())[:8],   # short id is plenty for in-memory dict
        label=label,
        group=group,
        metadata=metadata or {},
    )
    with _JOBS_LOCK:
        _JOBS[job.id] = job

    def _runner():
        job.state = "running"
        job.started_at = time.time()
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                target()
            job.state = "done"
        except Exception as e:
            job.state = "failed"
            job.error = f"{type(e).__name__}: {e}"
            buf.write("\n--- traceback ---\n")
            buf.write(traceback.format_exc())
        finally:
            job.log = buf.getvalue()
            job.ended_at = time.time()

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{job.id}")
    t.start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def has_running_job_for(group: str) -> Optional[Job]:
    """Return the first running/queued job whose `group` matches."""
    with _JOBS_LOCK:
        for j in _JOBS.values():
            if j.group == group and j.state in ("queued", "running"):
                return j
    return None


def cleanup_old(max_age_sec: float = 3600.0) -> int:
    """Drop finished jobs older than `max_age_sec`. Returns count dropped.

    Called opportunistically by `create_job` so we never need a separate
    sweeper thread.
    """
    cutoff = time.time() - max_age_sec
    dropped = 0
    with _JOBS_LOCK:
        for jid in list(_JOBS.keys()):
            j = _JOBS[jid]
            if j.state in ("done", "failed") and (j.ended_at or 0) < cutoff:
                _JOBS.pop(jid, None)
                dropped += 1
    return dropped


def job_to_dict(j: Job) -> dict:
    """Serialise for JSON responses (timestamps stay floats — the JS
    side formats them with `new Date(t * 1000)`)."""
    return {
        "id": j.id,
        "label": j.label,
        "state": j.state,
        "started_at": j.started_at,
        "ended_at": j.ended_at,
        "duration_sec": (
            (j.ended_at or time.time()) - j.started_at
            if j.started_at else None
        ),
        "error": j.error,
        "log": j.log,
        "group": j.group,
        "metadata": j.metadata,
    }
