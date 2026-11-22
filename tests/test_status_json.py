"""`lynx status --json` and `lynx list-sources --json`: one parseable object.

They were the only inspection commands without --json (2026-11-22 review):
a script that asks which sources exist and whether they are current had to
parse text written for people.
"""
from __future__ import annotations

import json
import os

import pytest


class _Backend:
    def __init__(self, name, stale):
        self.name, self._stale = name, stale

    def needs_update(self):
        os.write(1, b"noise from a backend")   # below Python, as libraries do
        return self._stale

    def drift_status_text(self):
        return f"drift of {self.name}"


class _Manager:
    def __init__(self):
        self.backends = {"api": _Backend("api", True), "docs": _Backend("docs", False)}
        self.broken = {"old": {"name": "old", "type": "codebase", "path": "/x", "health": "corrupt",
                               "error": "bad index"}}

    def list_sources(self):
        rows = [{"name": n, "type": "codebase", "path": f"/{n}", "chunk_count": 3, "health": "ok"}
                for n in self.backends]
        rows.append({"name": "old", "type": "codebase", "path": "/x", "health": "corrupt",
                     "error": "bad index", "chunk_count": None})
        return rows


class _Config:
    sources = {"api": {}, "docs": {}, "old": {}}


@pytest.fixture
def cli(monkeypatch):
    import lynx.cli as cli_mod

    def build(_path):
        os.write(1, b"[manager] loading sources")  # what a real build prints
        return _Config(), _Manager()

    monkeypatch.setattr(cli_mod, "_build_manager", build)
    return cli_mod


def _payload(capfd):
    return json.loads(capfd.readouterr().out)


def test_status_json_is_one_object_with_every_source(cli, capfd):
    assert cli.main(["status", "--json"]) == 0
    out = _payload(capfd)
    assert out["ok"] is True and out["operation"] == "status"
    rows = {r["name"]: r for r in out["sources"]}
    assert set(rows) == {"api", "docs", "old"}
    assert rows["api"]["needs_update"] is True and rows["docs"]["needs_update"] is False
    assert rows["api"]["drift"] == "drift of api"
    assert rows["old"]["health"] == "corrupt" and "needs_update" not in rows["old"]


def test_status_json_scopes_to_one_source(cli, capfd):
    assert cli.main(["status", "--json", "--source", "docs"]) == 0
    assert [r["name"] for r in _payload(capfd)["sources"]] == ["docs"]


def test_status_json_reports_an_unknown_source(cli, capfd):
    assert cli.main(["status", "--json", "--source", "ghost"]) == 1
    out = _payload(capfd)
    assert out["ok"] is False and "ghost" in out["error"]


def test_list_sources_json_is_one_object(cli, capfd):
    assert cli.main(["list-sources", "--json"]) == 0
    out = _payload(capfd)
    assert out["operation"] == "list-sources"
    assert [r["name"] for r in out["sources"]] == ["api", "docs", "old"]
