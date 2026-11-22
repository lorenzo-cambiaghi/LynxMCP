"""The web manager's routes after the 2026-11-22 review: what each fix
promises, asserted on the route, not on the layout.

The review found: the folder scan of Add source listed every extension (a
Unity project's .meta and .png) and walked the whole tree; a build already
running was invisible on the source page, and its widget repeated one id;
reset could wipe an index a build was writing; the config editor wrote
without the lock or an atomic replace and echoed errors unescaped; an
update with no output crashed, and the restart lost --config and --port;
the connect control drawn after a click lost "Active in"; a connected but
undetected client was drawn twice; the Playground ignored Mode and had no
identifiers, outline or describe_symbol `file`; the per-source Search link
did not select the source.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class _Status(dict):
    pass


class _Backend:
    type_name = "codebase"
    is_owner = True

    def __init__(self, name):
        self.name = name
        self.graph = None
        self.source_config = {"path": "/tmp/demo", "git_integration": {"enabled": False}}

    def status(self):
        return _Status(name=self.name, type="codebase", path="/tmp/demo", chunk_count=1,
                       last_commit=None, last_update=None, drift_severity=None)


class _Config:
    def __init__(self, storage):
        self.storage_path = storage
        self.sources = {"demo": {}, "other": {}}


class _Manager:
    def __init__(self, storage):
        self.config = _Config(storage)
        self.backends = {"demo": _Backend("demo"), "other": _Backend("other")}
        self.broken = {}
        self.calls = []

    def get(self, name):
        return self.backends[name]

    def list_sources(self):
        return [b.status() for b in self.backends.values()]

    def search(self, *a, **kw):
        self.calls.append(("search", a, kw))
        return [{"file_path": "/tmp/demo/a.ts", "symbol_name": "pickActor", "start_line": 1,
                 "end_line": 2, "score": 0.5, "content": "function pickActor() {}"}]

    def describe_symbol(self, *a, **kw):
        self.calls.append(("describe_symbol", a, kw))
        return {}

    def lexical_notice(self, names=None):
        return ""

    def close(self):
        pass


@pytest.fixture
def ui(tmp_path):
    from lynx.manager.ui.app import create_app
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"config_version": 2, "storage_path": str(tmp_path / "storage"),
                               "sources": {}}), encoding="utf-8")
    app = create_app(cfg)
    mgr = _Manager(tmp_path / "storage")
    app.state.manager = mgr
    client = TestClient(app)
    client.app_ref, client.mgr, client.cfg = app, mgr, cfg
    return client


@pytest.fixture
def running_build():
    """A build job for `demo` that runs until the test lets it go."""
    from lynx.manager.ui import jobs as jobs_mod
    release = threading.Event()
    job = jobs_mod.create_job(lambda: release.wait(10), label="build demo", group="build:demo",
                              metadata={"source": "demo"})
    yield job
    release.set()


# --- Add source: the folder scan -----------------------------------------------

def test_the_folder_scan_lists_readable_formats_and_the_engines_folders(ui, tmp_path):
    proj = tmp_path / "unity"
    for rel in ("Assets/Player.cs", "Assets/Player.cs.meta", "Assets/hero.png",
                "Assets/hero.prefab", "ProjectSettings/ProjectVersion.txt", "Library/cache.cs"):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text("x", encoding="utf-8")
    data = ui.post("/api/sources/_detect", json={"path": str(proj)}).json()
    assert ".cs" in data["extensions"]
    assert not {".meta", ".png", ".prefab"} & set(data["extensions"])
    assert any("Library" in f for f in data["ignored_path_fragments"])


# --- builds and resets ---------------------------------------------------------------

def test_the_source_page_shows_a_build_already_running(ui, running_build):
    html = ui.get("/sources/demo").text
    assert f'id="job-{running_build.id}"' in html
    assert html.count('id="build-slot"') == 1


def test_reset_shows_the_running_build_instead_of_wiping_its_index(ui, running_build):
    from lynx.manager.ui import jobs as jobs_mod
    html = ui.post("/api/sources/demo/reset").text
    assert f'id="job-{running_build.id}"' in html
    assert jobs_mod.has_running_job_for("reset:demo") is None


def test_two_job_widgets_never_share_an_id():
    from lynx.manager.ui.routes import _render_job_widget

    class _J:
        def __init__(self, i):
            self.id, self.state, self.started_at, self.ended_at = i, "running", None, None
            self.error, self.log, self.metadata, self.label = None, "", {}, "x"

    a, b = _render_job_widget(_J("a1"), "demo"), _render_job_widget(_J("b2"), "other")
    assert 'id="job-a1"' in a and 'id="job-b2"' in b


# --- the config editor -----------------------------------------------------------------

def test_the_config_editor_keeps_the_text_and_a_backup(ui):
    original = ui.cfg.read_text(encoding="utf-8")
    text = json.dumps({"config_version": 2, "storage_path": str(Path(ui.cfg).parent / "storage"),
                       "sources": {}}, indent=4) + "\n"
    r = ui.put("/api/config", json={"content": text})
    assert r.status_code == 200, r.text
    assert ui.cfg.read_text(encoding="utf-8") == text
    assert Path(str(ui.cfg) + ".bak").read_text(encoding="utf-8") == original
    assert not Path(str(ui.cfg) + ".tmp").exists()


def test_the_config_editor_escapes_what_it_echoes(ui, monkeypatch):
    # A validation error that quotes the config (a source name, a path)
    # comes back into the page: it must arrive as text, not markup.
    import lynx.config as config

    def fail(path):
        raise ValueError("source '<script>x</script>' is invalid")

    monkeypatch.setattr(config, "load_config", fail)
    r = ui.put("/api/config", json={"content": '{"config_version": 2, "sources": {}}'})
    assert r.status_code == 422
    assert "<script>" not in r.text and "&lt;script&gt;" in r.text


# --- update ----------------------------------------------------------------------------

def test_an_update_with_no_output_is_not_a_crash(ui, monkeypatch):
    import lynx.updater as updater
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(updater, "run_upgrade", lambda *a, **k: (0, ""))
    r = ui.post("/api/system/update")
    assert r.status_code == 200 and "Updated" in r.text


def test_the_manager_reopens_with_its_own_config_and_port(ui, monkeypatch):
    import subprocess
    import lynx.manager.ui.routes as routes
    seen = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.append(cmd))
    monkeypatch.setattr(subprocess, "CREATE_NEW_CONSOLE", 0, raising=False)
    monkeypatch.setattr(threading, "Timer", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    import lynx.updater as updater
    import lynx.manager.ui.jobs as jobs
    monkeypatch.setattr(updater, "blocking_processes", lambda *a, **k: [])
    monkeypatch.setattr(jobs, "any_running_job", lambda *a, **k: None)
    ui.app_ref.state.relaunch_args = ["--config", "C:/x/config.json", "--port", "9000"]
    ui.post("/api/system/update")
    assert seen, "the update console was not started"
    cmd = seen[-1]
    assert cmd[cmd.index("--restart-manager") + 1:] == [
        "--manager-arg", "--config", "--manager-arg", "C:/x/config.json",
        "--manager-arg", "--port", "--manager-arg", "9000"]


# --- integrations ------------------------------------------------------------------------

def test_the_control_after_a_click_keeps_where_lynx_is_active(ui, monkeypatch):
    import lynx.client_connector as cc
    from lynx.client_connector import ClientStatus, ConnectResult

    def fake_connect(key, **kw):
        return ConnectResult(ok=True, client_key=key, message="Connected.")

    monkeypatch.setattr(cc, "connect_client", fake_connect)
    monkeypatch.setattr(cc, "inspect_client", lambda spec, **k: ClientStatus(
        spec=spec, config_path=None, detected=True, connected=True, locations=("user",)))
    html = ui.post("/api/integrations/cursor/connect").text
    assert html.count('id="connect-ctrl-cursor"') == 1
    assert "Active in" in html and "all projects" in html


def test_a_connected_client_not_detected_is_drawn_once(ui, monkeypatch):
    import lynx.manager.ui.integrations as integ
    from lynx.client_connector import ClientStatus
    monkeypatch.setattr(integ, "inspect_client", lambda spec, **k: ClientStatus(
        spec=spec, config_path=None, detected=(spec.key != "cursor"), connected=(spec.key == "cursor")))
    html = ui.get("/integrations").text
    assert "Connected" in html
    assert html.count('id="connect-ctrl-cursor"') == 1


# --- playground -----------------------------------------------------------------------------

def test_playground_search_sends_mode_and_identifiers(ui):
    ui.post("/api/playground/search", data={"source": "demo", "query": "where a click picks a person",
                                            "mode": "dense", "identifiers": "pickActor, labelHover"})
    name, args, kw = ui.mgr.calls[-1]
    assert args[1] == "where a click picks a person pickActor  labelHover"
    assert kw.get("mode") == "dense"


def test_playground_outline_renders_signatures(ui):
    html = ui.post("/api/playground/search", data={"source": "demo", "query": "x", "outline": "1"}).text
    assert "OUTLINE (signatures only" in html
    assert "OUTLINE" not in ui.post("/api/playground/search", data={"source": "demo", "query": "x"}).text


def test_playground_describe_symbol_takes_a_file(ui):
    ui.post("/api/playground/describe_symbol", data={"source": "demo", "symbol": "rig", "file": "input.test"})
    name, _args, kw = ui.mgr.calls[-1]
    assert name == "describe_symbol" and kw.get("file") == "input.test"


def test_playground_preselects_the_source_of_the_link(ui):
    html = ui.get("/playground?source=other").text
    assert 'value="other"\n              selected' in html or 'value="other" selected' in html


# --- the mode reaches the retrieval ------------------------------------------------------------

def test_a_per_call_mode_overrides_the_configured_one():
    from lynx.rag_manager import CodebaseRAG
    seen = {}

    class _Self:
        search_mode = "hybrid"

        def _search_once(self, query, top_k, mode, **kw):
            seen["mode"] = mode
            return []

    CodebaseRAG.search(_Self(), "q", mode="sparse")
    assert seen["mode"] == "sparse"
    CodebaseRAG.search(_Self(), "q")
    assert seen["mode"] == "hybrid"
    with pytest.raises(ValueError):
        CodebaseRAG.search(_Self(), "q", mode="fuzzy")


def test_the_config_editor_writes_through_the_shared_locked_writer(ui, monkeypatch):
    import lynx.manager.sources as sources
    seen = []
    real_lock, real_write = sources.config_lock, sources.write_config_text

    def lock(path, *a, **k):
        seen.append("lock")
        return real_lock(path, *a, **k)

    def write(content, path):
        seen.append("write")
        return real_write(content, path)

    monkeypatch.setattr(sources, "config_lock", lock)
    monkeypatch.setattr(sources, "write_config_text", write)
    text = json.dumps({"config_version": 2, "storage_path": str(Path(ui.cfg).parent / "storage"),
                       "sources": {}}) + "\n"
    assert ui.put("/api/config", json={"content": text}).status_code == 200
    assert seen == ["lock", "write"]
