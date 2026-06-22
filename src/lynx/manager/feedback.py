"""`lynx manager feedback` — read & summarize the local feedback log.

The MCP `feedback` tool appends a JSON line to
`<storage_path>/_feedback/feedback.jsonl` whenever an agent couldn't find what
it needed (100% local, never uploaded). That log was write-only: nothing read
it back. This command closes the loop — it turns the collected reports into a
summary the index owner can act on (how many reports, over what span, what
agents were trying to do, and where they got stuck). Note: each report records
the sources configured at that time (not the one that failed — the log doesn't
carry that), so the per-source counts are "present at report time", not blame.

The parsing/summarizing helpers are pure (no config, no I/O beyond reading the
given file) so they're trivially testable; `run_feedback` just wires the active
config's storage_path to them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .ansi import bold, dim, heading, bullet


def feedback_path_for(storage_path) -> Path:
    """Where the MCP `feedback` tool writes its log (mirror of server.py)."""
    return Path(storage_path) / "_feedback" / "feedback.jsonl"


def load_feedback(path: Path) -> list:
    """Parse the JSONL feedback log into a list of record dicts.

    Missing file → []. Malformed lines are skipped (the log is append-only and
    a half-written final line shouldn't break the reader)."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def summarize_feedback(records: list, limit: int = 10) -> dict:
    """Aggregate the raw records into a summary.

    Returns:
      total       — number of reports
      first_at    — earliest `at` timestamp (or None)
      last_at     — latest `at` timestamp (or None)
      by_source   — {source_name: count} across every report that named it
      recent      — the last `limit` records (chronological, newest last)
    """
    total = len(records)
    ats = [r.get("at") for r in records if r.get("at")]
    by_source: dict = {}
    for r in records:
        for s in r.get("sources") or []:
            by_source[s] = by_source.get(s, 0) + 1
    recent = records[-limit:] if limit and limit > 0 else list(records)
    return {
        "total": total,
        "first_at": min(ats) if ats else None,
        "last_at": max(ats) if ats else None,
        "by_source": by_source,
        "recent": recent,
    }


def format_summary(summary: dict) -> str:
    """Human-readable, colored rendering of a feedback summary."""
    total = summary["total"]
    if total == 0:
        return (
            "No feedback recorded yet.\n"
            + dim("Agents call the `feedback` tool when the index can't answer; "
                  "nothing has been logged so far.")
        )

    lines = [heading(f"Feedback reports: {total}")]
    span_from = summary.get("first_at") or "?"
    span_to = summary.get("last_at") or "?"
    lines.append(dim(f"Span: {span_from} → {span_to}"))

    by_source = summary.get("by_source") or {}
    if by_source:
        ranked = sorted(by_source.items(), key=lambda kv: kv[1], reverse=True)
        lines.append("")
        # Each report records the sources configured at that moment (the agent's
        # search spans them all), so this is "present when the report was filed",
        # NOT "the source that failed" — the log doesn't carry that.
        lines.append(bold("Sources configured when reports were filed:"))
        for name, count in ranked:
            lines.append(bullet(f"{name}: {count}"))

    recent = summary.get("recent") or []
    if recent:
        lines.append("")
        lines.append(bold(f"Recent (showing {len(recent)} of {total}, newest last):"))
        for r in recent:
            at = r.get("at") or "?"
            trying = (r.get("trying_to_do") or "").strip() or "(unspecified)"
            lines.append(bullet(f"[{at}] {trying}"))
            tried = (r.get("tried") or "").strip()
            stuck = (r.get("stuck") or "").strip()
            if tried:
                lines.append(dim(f"      tried: {tried}"))
            if stuck:
                lines.append(dim(f"      stuck: {stuck}"))
    return "\n".join(lines)


def _storage_path_from_config(config_path: Path) -> Path:
    """Read just `storage_path` from the config JSON, resolved relative to the
    config file (mirrors `config._resolve_path` with the same default).

    We deliberately do NOT go through `load_config`: that validates every
    source (e.g. each codebase path must exist), and reading a local log
    shouldn't fail just because one source folder has since moved.
    """
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    sp = raw.get("storage_path", "./rag_storage")
    p = Path(sp)
    if not p.is_absolute():
        p = (config_path.resolve().parent / p).resolve()
    return p


def run_feedback(args) -> int:
    """CLI entry point for `lynx manager feedback`."""
    import sys

    from ..config import resolve_config_path

    config_path = resolve_config_path(getattr(args, "config", None))
    if not config_path.is_file():
        print(
            f"error: config file not found at {config_path}. "
            f"Pass --config PATH, set RAG_CONFIG_PATH, or run `lynx manager init`.",
            file=sys.stderr,
        )
        return 1
    try:
        storage_path = _storage_path_from_config(config_path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"error: could not read storage_path from {config_path}: {e}",
              file=sys.stderr)
        return 1

    path = feedback_path_for(storage_path)
    records = load_feedback(path)
    limit = getattr(args, "limit", 10) or 10
    summary = summarize_feedback(records, limit=limit)

    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(format_summary(summary))
    if records:
        print(dim(f"\nLog: {path}"))
    return 0
