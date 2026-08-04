"""The web UI playground — the panel where you try a tool by hand.

It had drifted into the smallest of the three surfaces: nine tools of
seventeen, and a Graph tab wired to three of the ten graph operations, so
there was no way to ask the UI for subclasses, imports, a shortest path or
the bridge edges. Meanwhile the only test covering any of it lived in
`test_manager_ui_app.py`, which has a `main()` and is therefore excluded
from collection — it runs when somebody remembers, which is how
`search_diff` came to read a key the backend has never returned and render
an empty list on every single run without anyone noticing.

So: pytest-collected, and asserting the two things that matter.

  - The panels answer with the tool's OWN text. The composed tools and the
    graph operations render `_format_*` output verbatim, because a
    playground exists to show what your agent receives; a prettier UI-only
    view would display something no agent ever sees.
  - Every graph operation is reachable, not just the three that used to
    have their own form.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class _Layer:
    def __init__(self, graph=None):
        self.graph = graph


class _Backend:
    def __init__(self, graph=True):
        self.type_name = "codebase"
        self.graph = _Layer() if graph else None
        self.source_config = {"path": "/tmp/demo",
                              "git_integration": {"enabled": True}}


class _Manager:
    """Returns canned payloads and records what was asked of it."""

    def __init__(self, **returns):
        self.backends = {"demo": _Backend()}
        self.calls = []
        self._returns = returns

    def _record(self, name, *a, **kw):
        self.calls.append((name, a, kw))
        return self._returns.get(name, [])

    def get(self, name):
        return self.backends[name]

    def list_sources(self):
        # The page renders a source picker from this.
        return [{"name": "demo", "type": "codebase", "chunk_count": 1,
                 "path": "/tmp/demo"}]

    # graph
    def get_callers(self, *a, **kw): return self._record("get_callers", *a, **kw)
    def get_callees(self, *a, **kw): return self._record("get_callees", *a, **kw)
    def get_subclasses(self, *a, **kw): return self._record("get_subclasses", *a, **kw)
    def get_superclasses(self, *a, **kw): return self._record("get_superclasses", *a, **kw)
    def get_imports(self, *a, **kw): return self._record("get_imports", *a, **kw)
    def get_neighbors(self, *a, **kw): return self._record("get_neighbors", *a, **kw)

    def shortest_path(self, *a, **kw):
        self._record("shortest_path", *a, **kw)
        return None

    def architectural_overview(self, *a, **kw):
        self._record("architectural_overview", *a, **kw)
        return {"status": {}, "god_nodes": [], "communities": []}

    def surprising_connections(self, *a, **kw):
        return self._record("surprising_connections", *a, **kw)

    def graph_status(self, *a, **kw):
        self._record("graph_status", *a, **kw)
        return {"schema_version": 2, "nodes": 5, "edges": 2,
                "files_indexed": 3, "raw_calls_pending": 0,
                "last_update": "x", "last_full_rebuild": "y",
                "by_language": {}, "by_kind": {}, "by_relation": {}}

    # composed
    def describe_symbol(self, *a, **kw):
        self._record("describe_symbol", *a, **kw)
        return self._returns.get("describe_symbol", {})

    def impact_of(self, *a, **kw):
        self._record("impact_of", *a, **kw)
        return self._returns.get("impact_of", {})

    def module_summary(self, *a, **kw):
        self._record("module_summary", *a, **kw)
        return self._returns.get("module_summary", {})

    def repo_overview(self, *a, **kw):
        self._record("repo_overview", *a, **kw)
        return self._returns.get("repo_overview", {})

    def search_diff(self, *a, **kw):
        self._record("search_diff", *a, **kw)
        return self._returns.get("search_diff", {})

    def export_graph(self, *a, **kw):
        self._record("export_graph", *a, **kw)
        return self._returns.get("export_graph", {"empty": True,
                                                  "reason": "nothing here"})

    def deep_search(self, *a, **kw):
        self._record("deep_search", *a, **kw)
        return self._returns.get("deep_search", {
            "results": [], "variants_tried": 1,
            "winning_variant_index": 0, "all_weak": False,
        })


def _make_app(tmp_path: Path, manager):
    from lynx.manager.ui.app import create_app

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp_path / "storage"),
        "sources": {},
    }), encoding="utf-8")
    app = create_app(cfg)
    app.state.manager = manager   # skip the lazy build; we're testing routes
    return app


@pytest.fixture
def client(tmp_path):
    mgr = _Manager()
    app = _make_app(tmp_path, mgr)
    c = TestClient(app)
    c.manager = mgr
    return c


# ----------------------------------------------------------------------
# graph_query: one form, all ten operations
# ----------------------------------------------------------------------


@pytest.mark.parametrize("operation,method", [
    ("callers", "get_callers"),
    ("callees", "get_callees"),
    ("subclasses", "get_subclasses"),
    ("superclasses", "get_superclasses"),
    ("imports", "get_imports"),
    ("neighbors", "get_neighbors"),
    ("shortest_path", "shortest_path"),
    ("overview", "architectural_overview"),
    ("surprising_connections", "surprising_connections"),
    ("status", "graph_status"),
])
def test_every_graph_operation_is_reachable(client, operation, method):
    """Three of these had a form; the other seven had no UI at all."""
    r = client.post("/api/playground/graph_query", data={
        "source": "demo", "operation": operation,
        "symbol": "helper", "target": "other",
    })

    assert r.status_code == 200, r.text
    assert client.manager.calls[0][0] == method


def test_graph_query_renders_the_tool_s_own_text(client):
    """Verbatim `_format_*` output — the point of a playground is to show
    what the agent gets, not a prettier thing only humans ever see."""
    from lynx.graph.dispatch import query_graph

    r = client.post("/api/playground/graph_query",
                    data={"source": "demo", "operation": "callers",
                          "symbol": "helper"})

    expected = query_graph(_Manager(), "demo", "callers", symbol="helper").text
    assert "<pre" in r.text
    for line in expected.splitlines():
        assert line.strip() in r.text.replace("&#x27;", "'").replace("&quot;", '"')


def test_graph_query_reports_a_usage_error_as_400(client):
    """A missing symbol is the user's mistake, not a server fault."""
    r = client.post("/api/playground/graph_query",
                    data={"source": "demo", "operation": "callers"})

    assert r.status_code == 400
    assert "symbol" in r.text.lower()
    assert client.manager.calls == []


def test_graph_query_rejects_an_unknown_operation(client):
    r = client.post("/api/playground/graph_query",
                    data={"source": "demo", "operation": "nope"})

    assert r.status_code == 400
    assert "unknown operation" in r.text.lower()


def test_the_replaced_endpoints_are_gone(client):
    """The three hand-wired panels folded into graph_query; leaving them
    behind would mean two ways to ask the same question, drifting."""
    for path in ("get_callers", "get_callees", "architectural_overview"):
        r = client.post(f"/api/playground/{path}",
                        data={"source": "demo", "symbol": "x"})
        assert r.status_code == 404, path


# ----------------------------------------------------------------------
# The composed tools that had no panel at all
# ----------------------------------------------------------------------


def test_describe_symbol_panel(client):
    r = client.post("/api/playground/describe_symbol",
                    data={"source": "demo", "symbol": "helper",
                          "callers_limit": "3"})

    assert r.status_code == 200
    name, args, kwargs = client.manager.calls[0]
    assert name == "describe_symbol" and args[1] == "helper"
    assert kwargs["callers_limit"] == 3


def test_impact_panel(client):
    r = client.post("/api/playground/impact",
                    data={"source": "demo", "symbol": "helper",
                          "max_depth": "4"})

    assert r.status_code == 200
    name, _args, kwargs = client.manager.calls[0]
    assert name == "impact_of" and kwargs["max_depth"] == 4


def test_module_summary_panel(client):
    r = client.post("/api/playground/module_summary",
                    data={"source": "demo", "file": "main.py", "limit": "7"})

    assert r.status_code == 200
    name, args, kwargs = client.manager.calls[0]
    assert name == "module_summary" and args[1] == "main.py"
    assert kwargs["limit"] == 7


def test_repo_overview_panel(client):
    r = client.post("/api/playground/repo_overview", data={"source": "demo"})

    assert r.status_code == 200
    assert client.manager.calls[0][0] == "repo_overview"


def test_composed_panels_render_the_shared_text(client, tmp_path):
    """Same assertion as for the graph: the panel shows the tool's text."""
    from lynx._format import _format_repo_overview

    mgr = _Manager(repo_overview={"root": "/repo", "file_count": 3,
                                  "languages": [{"language": "Python",
                                                 "files": 3}]})
    c = TestClient(_make_app(tmp_path / "b", mgr))
    r = c.post("/api/playground/repo_overview", data={"source": "demo"})

    expected = _format_repo_overview(mgr._returns["repo_overview"])
    assert "Python (3)" in expected          # guard the fixture itself
    assert "Python (3)" in r.text


def test_empty_symbol_is_rejected(client):
    for path, field in (("describe_symbol", "symbol"), ("impact", "symbol"),
                        ("module_summary", "file")):
        r = client.post(f"/api/playground/{path}",
                        data={"source": "demo", field: "   "})
        assert r.status_code == 400, path


# ----------------------------------------------------------------------
# deep_search
# ----------------------------------------------------------------------


def test_deep_search_splits_the_textarea_into_variants(client):
    r = client.post("/api/playground/deep_search", data={
        "source": "demo",
        "queries": "player health\n\n  damage logic  \nHP lifecycle\n",
        "top_k": "4",
    })

    assert r.status_code == 200
    _name, _args, kwargs = client.manager.calls[0]
    assert kwargs["queries"] == ["player health", "damage logic", "HP lifecycle"]
    assert kwargs["top_k"] == 4


def test_deep_search_needs_at_least_one_query(client):
    r = client.post("/api/playground/deep_search",
                    data={"source": "demo", "queries": "\n  \n"})

    assert r.status_code == 400
    assert client.manager.calls == []


# ----------------------------------------------------------------------
# search_diff: the panel that never showed anything
# ----------------------------------------------------------------------


def test_search_diff_renders_the_hits(client, tmp_path):
    """It read `payload["results"]`; the backend returns `hits`. The panel
    rendered "No results." on every run, including the ones that found
    something — invisible because nothing tested it."""
    mgr = _Manager(search_diff={
        "base": "main",
        "modified_files": ["a.py"],
        "hits": [{"file": "a.py", "file_path": "/repo/a.py",
                  "content": "def changed(): pass", "score": 0.9,
                  "symbol_name": "changed", "start_line": 1, "end_line": 2}],
    })
    c = TestClient(_make_app(tmp_path / "c", mgr))

    r = c.post("/api/playground/search_diff",
               data={"source": "demo", "query": "changed", "top_k": "5"})

    assert r.status_code == 200
    assert "changed" in r.text
    assert "No results" not in r.text


def test_search_diff_shows_the_note_when_nothing_changed(client, tmp_path):
    """The backend explains an empty answer; dropping the note left the
    user staring at a blank panel."""
    mgr = _Manager(search_diff={
        "base": "main", "modified_files": [], "hits": [],
        "note": "No files added/modified vs 'main'.",
    })
    c = TestClient(_make_app(tmp_path / "d", mgr))

    r = c.post("/api/playground/search_diff",
               data={"source": "demo", "query": "anything"})

    assert "No files added/modified" in r.text


# ----------------------------------------------------------------------
# The page itself
# ----------------------------------------------------------------------


def test_the_page_offers_every_panel(client):
    """A route with no form is unreachable; a form with no route 404s.

    Coverage against the real tool list is asserted in
    test_surface_parity.py — this only checks the page renders them.
    """
    r = client.get("/playground")

    assert r.status_code == 200
    for endpoint in ("search", "deep_search", "find_definition",
                     "find_usages", "find_tests_for", "find_similar",
                     "describe_symbol", "impact", "module_summary",
                     "repo_overview", "graph_query", "export_graph",
                     "search_diff"):
        assert f"/api/playground/{endpoint}" in r.text, endpoint


def test_the_graph_form_lists_every_operation(client):
    from lynx.graph.dispatch import OPERATIONS

    r = client.get("/playground")

    for op in OPERATIONS:
        assert f'value="{op}"' in r.text, f"{op} missing from the form"


# ----------------------------------------------------------------------
# export_graph — the tool that had a CLI and an MCP entry point but no UI,
# which was backwards for something whose output is a web page.
# ----------------------------------------------------------------------


def _exporting_manager(tmp_path, content="<html>graph</html>"):
    mgr = _Manager(export_graph={"content": content,
                                 "suggested_name": "blast_helper.html"})
    mgr.config = type("_Cfg", (), {
        "storage_path": str(tmp_path / "storage"), "reports_path": None,
    })()
    return mgr


def test_export_writes_the_view_and_links_to_it(tmp_path):
    from fastapi.testclient import TestClient

    mgr = _exporting_manager(tmp_path)
    c = TestClient(_make_app(tmp_path / "e", mgr))

    r = c.post("/api/playground/export_graph",
               data={"source": "demo", "target": "helper", "mode": "symbol",
                     "depth": "3"})

    assert r.status_code == 200
    _name, args, kwargs = mgr.calls[0]
    assert args[1] == "symbol" and args[2] == "helper"
    assert kwargs["depth"] == 3
    # Written where the MCP tool and `lynx graph export` write.
    written = tmp_path / "storage" / "reports" / "blast_helper.html"
    assert written.is_file()
    assert "/api/reports/blast_helper.html" in r.text


def test_the_written_view_is_downloadable(tmp_path):
    from fastapi.testclient import TestClient

    mgr = _exporting_manager(tmp_path, content="<html>the blast radius</html>")
    c = TestClient(_make_app(tmp_path / "f", mgr))
    c.post("/api/playground/export_graph",
           data={"source": "demo", "target": "helper"})

    r = c.get("/api/reports/blast_helper.html")

    assert r.status_code == 200
    assert "the blast radius" in r.text


@pytest.mark.parametrize("name", [
    "../config.json",
    "..%2Fconfig.json",
    "sub/dir.html",
    "notes.txt",
    "no-extension",
])
def test_the_download_endpoint_refuses_to_walk_out(tmp_path, name):
    """The name comes straight from a URL. Both the pattern and the
    resolved-parent check have to hold, because a pattern alone trusts
    `Path` to agree about what a separator is — which differs by OS."""
    from fastapi.testclient import TestClient

    mgr = _exporting_manager(tmp_path)
    c = TestClient(_make_app(tmp_path / "g", mgr))

    r = c.get(f"/api/reports/{name}")

    assert r.status_code == 404, f"{name!r} was served"


def test_export_reports_an_empty_result(tmp_path):
    from fastapi.testclient import TestClient

    mgr = _Manager(export_graph={"empty": True, "reason": "no such symbol"})
    mgr.config = type("_Cfg", (), {"storage_path": str(tmp_path / "s"),
                                   "reports_path": None})()
    c = TestClient(_make_app(tmp_path / "h", mgr))

    r = c.post("/api/playground/export_graph",
               data={"source": "demo", "target": "ghost"})

    assert r.status_code == 200
    assert "no such symbol" in r.text


def test_export_rejects_a_bad_mode(client):
    r = client.post("/api/playground/export_graph",
                    data={"source": "demo", "target": "helper", "mode": "nope"})

    assert r.status_code == 400
    assert client.manager.calls == []


# ----------------------------------------------------------------------
# The dashboard's feedback card — the other tool the UI didn't serve
# ----------------------------------------------------------------------


def test_the_dashboard_shows_feedback_reports(tmp_path):
    import json as _json

    from fastapi.testclient import TestClient

    mgr = _Manager()
    storage = tmp_path / "storage"
    mgr.config = type("_Cfg", (), {"storage_path": str(storage),
                                   "reports_path": None})()
    fb = storage / "_feedback"
    fb.mkdir(parents=True)
    (fb / "feedback.jsonl").write_text(
        _json.dumps({"at": "2026-08-04T10:00:00",
                     "trying_to_do": "find the retry policy",
                     "tried": "search, deep_search",
                     "stuck": "nothing about backoff in the index",
                     "sources": ["demo"]}) + "\n",
        encoding="utf-8",
    )
    c = TestClient(_make_app(tmp_path / "i", mgr))

    r = c.get("/")

    assert r.status_code == 200
    assert "find the retry policy" in r.text
    assert "nothing about backoff" in r.text


def test_the_dashboard_is_silent_without_feedback(tmp_path):
    from fastapi.testclient import TestClient

    mgr = _Manager()
    mgr.config = type("_Cfg", (), {"storage_path": str(tmp_path / "empty"),
                                   "reports_path": None})()
    c = TestClient(_make_app(tmp_path / "j", mgr))

    r = c.get("/")

    assert r.status_code == 200
    assert "Feedback from your AI clients" not in r.text


def test_an_unreadable_feedback_log_does_not_blank_the_page(tmp_path):
    """Best-effort by design: a corrupt log is not a reason to lose the
    dashboard."""
    from fastapi.testclient import TestClient

    mgr = _Manager()
    storage = tmp_path / "storage"
    fb = storage / "_feedback"
    fb.mkdir(parents=True)
    (fb / "feedback.jsonl").write_text("{not json at all\n", encoding="utf-8")
    mgr.config = type("_Cfg", (), {"storage_path": str(storage),
                                   "reports_path": None})()
    c = TestClient(_make_app(tmp_path / "k", mgr))

    assert c.get("/").status_code == 200
