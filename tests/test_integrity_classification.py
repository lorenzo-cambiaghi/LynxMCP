"""`check_index` must tell a TIMEOUT apart from real corruption.

A timed-out integrity probe is ambiguous (large index, slow/AV-scanned disk,
or another Lynx process holding the store) and must NOT be branded `corrupt` —
that would push the user to wipe a likely-healthy index. Only a non-zero child
exit (caught exception or native crash) is a genuine corruption signal.
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from lynx import integrity


def _seed_index(tmp_path):
    """check_index returns early on a missing store, so give it a file."""
    (tmp_path / "chroma.sqlite3").write_text("not really sqlite, but present")
    return tmp_path


def test_timeout_is_unverified_and_retried(tmp_path, monkeypatch):
    storage = _seed_index(tmp_path)
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

    monkeypatch.setattr(integrity.subprocess, "run", fake_run)
    result = integrity.check_index(storage, "framework", timeout=1.0)

    assert result["status"] == "unverified"
    assert result.get("crashed") is False
    # A transient lock/race should get a second chance before we give up.
    assert calls["n"] == 2
    # The message must not scare the user into a reset.
    assert "corrupt" in result["detail"]  # only as "...not necessarily corruption"
    assert "not necessarily" in result["detail"]


def test_native_crash_is_corrupt(tmp_path, monkeypatch):
    storage = _seed_index(tmp_path)
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=-11, stdout="", stderr="boom"),
    )
    result = integrity.check_index(storage, "framework")
    assert result["status"] == "corrupt"
    assert result["crashed"] is True


def test_caught_exception_is_corrupt_not_crashed(tmp_path, monkeypatch):
    storage = _seed_index(tmp_path)
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="CorruptIndexError: bad segment"),
    )
    result = integrity.check_index(storage, "framework")
    assert result["status"] == "corrupt"
    assert result["crashed"] is False


def test_healthy_index_is_ok(tmp_path, monkeypatch):
    storage = _seed_index(tmp_path)
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout=json.dumps({"ok": True, "count": 9629}), stderr=""),
    )
    result = integrity.check_index(storage, "framework")
    assert result["status"] == "ok"
    assert result["count"] == 9629


def test_missing_store_is_empty(tmp_path, monkeypatch):
    # No chroma.sqlite3 → fresh source, never even spawns the probe.
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    result = integrity.check_index(tmp_path, "framework")
    assert result["status"] == "empty"
