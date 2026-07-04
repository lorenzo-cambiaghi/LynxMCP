"""`check_index` must tell a TIMEOUT apart from real corruption.

A timed-out integrity probe is ambiguous (large index, slow/AV-scanned disk,
or another Lynx process holding the store) and must NOT be branded `corrupt` —
that would push the user to wipe a likely-healthy index. Only a non-zero child
exit (caught exception or native crash) is a genuine corruption signal.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

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


# ---------------------------------------------------------------------------
# "in use by another Lynx process" detection
#
# ChromaDB's core lock means a probe of a store `lynx serve` holds BLOCKS
# forever in count() — no timeout can verify it. check_index must recognise
# the situation up front and report `in_use` (healthy, just busy) instead of
# burning 3 minutes and alarming the user with an unverified/corrupt card.
# ---------------------------------------------------------------------------

def test_store_held_by_other_process_is_in_use_without_probing(tmp_path, monkeypatch):
    storage = _seed_index(tmp_path)
    monkeypatch.setattr(integrity, "_store_usage", lambda p: "other")
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    result = integrity.check_index(storage, "framework")
    assert result["status"] == "in_use"
    assert "not corrupt" in result["detail"]


def test_store_held_by_self_is_ok_without_probing(tmp_path, monkeypatch):
    # E.g. the manager UI reloaded: ChromaDB's system cache still holds the
    # store in THIS process. A child probe would deadlock on our own lock,
    # and a store we already use can't fail a fresh-open check.
    storage = _seed_index(tmp_path)
    monkeypatch.setattr(integrity, "_store_usage", lambda p: "self")
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    result = integrity.check_index(storage, "framework")
    assert result["status"] == "ok"


def test_holder_appearing_after_timeout_reclassifies_as_in_use(tmp_path, monkeypatch):
    # `lynx serve` starts while the probe is running: the up-front check said
    # free, both attempts time out, and the post-timeout re-check must land on
    # in_use rather than the scarier unverified.
    storage = _seed_index(tmp_path)
    usages = iter(["free", "other"])
    monkeypatch.setattr(integrity, "_store_usage", lambda p: next(usages))
    monkeypatch.setattr(
        integrity.subprocess, "run",
        lambda cmd, **kw: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))),
    )
    result = integrity.check_index(storage, "framework", timeout=0.1)
    assert result["status"] == "in_use"


@pytest.mark.skipif(os.name != "nt", reason="handle-sharing probe is Windows-only")
def test_windows_handle_probe_sees_open_handles(tmp_path):
    storage = _seed_index(tmp_path)
    assert integrity._store_usage(storage) == "free"
    with open(storage / "chroma.sqlite3"):
        # An open handle in this very process must classify as self, not other.
        assert integrity._store_usage(storage) == "self"
    assert integrity._store_usage(storage) == "free"


def test_probe_child_self_destructs_when_orphaned():
    # Orphan insurance: a probe child whose parent died must hard-exit on its
    # own once its lifetime is over, instead of lingering forever holding the
    # store open (which would wedge every later probe too).
    code = ("from lynx.integrity import _self_destruct_after; import time; "
            "_self_destruct_after(0.5); time.sleep(30)")
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", code], timeout=20)
    assert proc.returncode == 3
    assert time.monotonic() - t0 < 15
