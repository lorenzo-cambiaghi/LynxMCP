"""Tests for `lynx manager feedback` — the reader/summarizer of the local
feedback log written by the MCP `feedback` tool.

Model-free: pure parsing/summarizing helpers plus an integration test of
`run_feedback` against a minimal temp config (empty `sources` is valid, so no
embedding model is loaded).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from lynx.manager import feedback as fb


def _write_log(path: Path, records: list, *, trailing_garbage: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records]
    if trailing_garbage:
        lines.append("{ this is not valid json")  # half-written final line
        lines.append("")                            # blank line
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rec(at, trying, sources, tried="grep", stuck="nothing matched"):
    return {"at": at, "trying_to_do": trying, "tried": tried,
            "stuck": stuck, "sources": sources}


def test_load_feedback_missing_file(tmp_path):
    assert fb.load_feedback(tmp_path / "nope.jsonl") == []


def test_load_feedback_skips_malformed(tmp_path):
    p = tmp_path / "feedback.jsonl"
    _write_log(p, [_rec("2026-06-01", "find auth", ["app"]),
                   _rec("2026-06-02", "find retry", ["app", "docs"])],
               trailing_garbage=True)
    records = fb.load_feedback(p)
    assert len(records) == 2
    assert records[0]["trying_to_do"] == "find auth"


def test_summarize_feedback_aggregates():
    records = [
        _rec("2026-06-01T10:00", "a", ["app"]),
        _rec("2026-06-03T10:00", "b", ["app", "docs"]),
        _rec("2026-06-02T10:00", "c", ["docs"]),
    ]
    s = fb.summarize_feedback(records, limit=2)
    assert s["total"] == 3
    assert s["first_at"] == "2026-06-01T10:00"
    assert s["last_at"] == "2026-06-03T10:00"
    assert s["by_source"] == {"app": 2, "docs": 2}
    # recent = last `limit`, chronological order preserved (newest last)
    assert [r["trying_to_do"] for r in s["recent"]] == ["b", "c"]


def test_format_summary_empty():
    out = fb.format_summary(fb.summarize_feedback([]))
    assert "No feedback recorded yet" in out


def test_format_summary_renders_sections():
    records = [_rec("2026-06-01", "find where we clamp zoom", ["game"])]
    out = fb.format_summary(fb.summarize_feedback(records))
    assert "Feedback reports: 1" in out
    assert "game" in out
    assert "find where we clamp zoom" in out
    assert "nothing matched" in out  # the `stuck` line


def _minimal_config(tmp_path: Path) -> Path:
    cfg = {
        "config_version": 2,
        "storage_path": str(tmp_path / "rag_storage"),
        "sources": {},  # empty sources is valid (no model load)
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_run_feedback_integration(tmp_path, capsys):
    config_path = _minimal_config(tmp_path)
    log = fb.feedback_path_for(tmp_path / "rag_storage")
    _write_log(log, [_rec("2026-06-01", "find session validation", ["app"])])

    rc = fb.run_feedback(SimpleNamespace(config=str(config_path), limit=10, json=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Feedback reports: 1" in out
    assert "find session validation" in out


def test_run_feedback_json(tmp_path, capsys):
    config_path = _minimal_config(tmp_path)
    log = fb.feedback_path_for(tmp_path / "rag_storage")
    _write_log(log, [_rec("2026-06-01", "x", ["app"])])

    rc = fb.run_feedback(SimpleNamespace(config=str(config_path), limit=10, json=True))
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 1 and data["by_source"] == {"app": 1}


def test_run_feedback_no_log_yet(tmp_path, capsys):
    config_path = _minimal_config(tmp_path)  # no feedback.jsonl written
    rc = fb.run_feedback(SimpleNamespace(config=str(config_path), limit=10, json=False))
    assert rc == 0
    assert "No feedback recorded yet" in capsys.readouterr().out


def test_run_feedback_missing_config(tmp_path, capsys):
    rc = fb.run_feedback(SimpleNamespace(
        config=str(tmp_path / "nope.json"), limit=10, json=False))
    assert rc == 1
    assert "config file not found" in capsys.readouterr().err
