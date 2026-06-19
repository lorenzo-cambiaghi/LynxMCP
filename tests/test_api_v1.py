"""Tests for the stable /api/v1/* JSON surface (the external integration
contract used by the Coral source spec in integrations/coral/).

Uses FastAPI's TestClient with a fake manager injected on app.state —
no embedding model, no ChromaDB.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lynx.manager.ui.app import create_app


class FakeManager:
    def __init__(self):
        self.config = SimpleNamespace(storage_path="/tmp/x")
        # "code" has the graph layer, "docs" does not.
        self.backends = {
            "code": SimpleNamespace(graph=object()),
            "docs": SimpleNamespace(graph=None),
        }

    def get(self, source):
        if source not in self.backends:
            raise KeyError(f"Unknown source {source!r}")
        return self.backends[source]

    def search(self, source, q, top_k=8, **kw):
        return [{
            "file": "a.py", "file_path": "/repo/a.py",
            "symbol_name": "Foo.bar", "symbol_kind": "method",
            "language": "python", "start_line": 10, "end_line": 20,
            "score": 0.031,
            "content": 'def bar(self, x):\n    """Return x, doubled."""\n    return x * 2',
        }][:top_k]

    def search_batch(self, source, queries, top_k=8, **kw):
        return [self.search(source, q, top_k=top_k) for q in queries]

    def search_all(self, q, top_k=8, **kw):
        hits = self.search("code", q, top_k=top_k)
        for h in hits:
            h["source"] = "code"
        return hits

    def list_sources(self):
        return [
            {"name": "code", "type": "codebase", "path": "/repo",
             "chunk_count": 123, "last_update": "2026-06-12T10:00:00"},
            {"name": "docs", "type": "webdoc", "url": "https://example.com",
             "chunk_count": None, "last_update": None},
        ]

    # --- graph layer (edge-returning ops mirror the real SourceManager) ---
    def _edge(self, frm, to, relation, **extra):
        e = {
            "source": {"label": frm, "kind": "method",
                       "file": "/repo/a.py", "start_line": 10, "end_line": 20},
            "target": {"label": to, "kind": "method",
                       "file": "/repo/b.py", "start_line": 5, "end_line": 8},
            "relation": relation,
            "confidence": "extracted",
            "from_file": "/repo/a.py", "from_line": 15,
        }
        e.update(extra)
        return e

    def get_callers(self, source, symbol, limit=50):
        return [self._edge("Caller.run", symbol, "calls")][:limit]

    def get_callees(self, source, symbol, limit=50):
        return [self._edge(symbol, "Helper.do", "calls")][:limit]

    def get_subclasses(self, source, symbol, limit=50):
        return [self._edge("Derived", symbol, "inherits", base_kind="extends")][:limit]

    def get_superclasses(self, source, symbol, limit=50):
        return [self._edge(symbol, "Base", "inherits", base_kind="extends")][:limit]

    def get_imports(self, source, file_or_symbol, limit=100):
        return [self._edge(file_or_symbol, "os", "imports", module="os")][:limit]

    def get_neighbors(self, source, symbol, relation_filter=None, depth=1, limit=100):
        es = [self._edge(symbol, "Helper.do", "calls"),
              self._edge("Caller.run", symbol, "inherits", base_kind="extends")]
        if relation_filter:
            es = [e for e in es if e["relation"] == relation_filter]
        return es[:limit]


@pytest.fixture()
def client():
    app = create_app(Path("nonexistent-config.json"))
    app.state.manager = FakeManager()
    return TestClient(app)


def test_v1_search_single_source(client):
    r = client.get("/api/v1/search", params={"q": "where is bar", "source": "code"})
    assert r.status_code == 200
    rows = r.json()["results"]
    assert rows[0]["symbol"] == "Foo.bar"
    assert rows[0]["start_line"] == 10
    assert rows[0]["score"] == pytest.approx(0.031)
    assert rows[0]["source"] == "code"
    assert "def bar" in rows[0]["content"]


def test_v1_search_all_sources_when_source_omitted(client):
    r = client.get("/api/v1/search", params={"q": "anything"})
    assert r.status_code == 200
    assert r.json()["results"][0]["source"] == "code"


def test_v1_search_unknown_source_is_404(client):
    r = client.get("/api/v1/search", params={"q": "x", "source": "nope"})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_v1_search_requires_query(client):
    assert client.get("/api/v1/search").status_code == 422


def test_v1_sources_rows(client):
    r = client.get("/api/v1/sources")
    assert r.status_code == 200
    rows = r.json()["sources"]
    assert {row["name"] for row in rows} == {"code", "docs"}
    code = next(row for row in rows if row["name"] == "code")
    assert code["location"] == "/repo"
    assert code["chunk_count"] == 123
    docs = next(row for row in rows if row["name"] == "docs")
    assert docs["location"] == "https://example.com"
    assert docs["chunk_count"] == 0  # None normalized


def test_v1_search_batch_post(client):
    r = client.post(
        "/api/v1/search",
        json={"queries": ["q1", "q2"], "source": "code", "top_k": 2},
    )
    assert r.status_code == 200
    rows = r.json()["results"]
    assert [row["query"] for row in rows] == ["q1", "q2"]
    assert rows[0]["hits"][0]["symbol"] == "Foo.bar"
    assert rows[0]["hits"][0]["source"] == "code"


def test_v1_search_batch_all_sources_when_source_omitted(client):
    r = client.post("/api/v1/search", json={"queries": ["q1"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["hits"][0]["source"] == "code"


def test_v1_search_batch_empty_queries_is_400(client):
    assert client.post("/api/v1/search", json={"queries": []}).status_code == 400


def test_v1_search_batch_unknown_source_is_404(client):
    r = client.post("/api/v1/search", json={"queries": ["x"], "source": "nope"})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_v1_unavailable_manager_is_503():
    app = create_app(Path("nonexistent-config.json"))
    app.state.manager = None
    app.state.manager_error = "boom"
    c = TestClient(app)
    assert c.get("/api/v1/search", params={"q": "x"}).status_code == 503
    assert c.get("/api/v1/sources").status_code == 503
    assert c.get(
        "/api/v1/graph", params={"operation": "callers", "symbol": "x"}
    ).status_code == 503


# ---------------------------------------------------------------------------
# /api/v1/graph
# ---------------------------------------------------------------------------


def test_v1_graph_callers_flattened_rows(client):
    r = client.get(
        "/api/v1/graph",
        params={"operation": "callers", "symbol": "Foo.bar", "source": "code"},
    )
    assert r.status_code == 200
    row = r.json()["results"][0]
    assert row["relation"] == "calls"
    assert row["from_symbol"] == "Caller.run"
    assert row["to_symbol"] == "Foo.bar"
    assert row["from_file"] == "/repo/a.py"
    assert row["to_file"] == "/repo/b.py"
    assert row["call_site_line"] == 15


def test_v1_graph_source_omitted_uses_single_graph_source(client):
    # Only "code" has the graph layer, so `source` can be omitted.
    r = client.get("/api/v1/graph", params={"operation": "callees", "symbol": "Foo.bar"})
    assert r.status_code == 200
    assert r.json()["results"][0]["to_symbol"] == "Helper.do"


def test_v1_graph_nongraph_source_is_404(client):
    r = client.get(
        "/api/v1/graph",
        params={"operation": "callers", "symbol": "x", "source": "docs"},
    )
    assert r.status_code == 404
    assert "docs" in r.json()["detail"]


def test_v1_graph_unknown_source_is_404(client):
    r = client.get(
        "/api/v1/graph",
        params={"operation": "callers", "symbol": "x", "source": "nope"},
    )
    assert r.status_code == 404


def test_v1_graph_unknown_operation_is_400(client):
    r = client.get("/api/v1/graph", params={"operation": "frobnicate", "symbol": "x"})
    assert r.status_code == 400


def test_v1_graph_requires_symbol(client):
    assert client.get(
        "/api/v1/graph", params={"operation": "callers"}
    ).status_code == 422


def test_v1_graph_neighbors_relation_filter(client):
    r = client.get(
        "/api/v1/graph",
        params={"operation": "neighbors", "symbol": "Foo.bar", "relation": "calls"},
    )
    assert r.status_code == 200
    rows = r.json()["results"]
    assert rows and all(row["relation"] == "calls" for row in rows)


def test_v1_graph_inherits_carries_base_kind(client):
    r = client.get(
        "/api/v1/graph", params={"operation": "superclasses", "symbol": "Derived"}
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["base_kind"] == "extends"


def test_v1_graph_imports_exposes_module(client):
    # The imported module string lives in `module` (the target node is often
    # synthetic for imports), so it must survive the flattening.
    r = client.get(
        "/api/v1/graph", params={"operation": "imports", "symbol": "a.py"}
    )
    assert r.status_code == 200
    row = r.json()["results"][0]
    assert row["relation"] == "imports"
    assert row["module"] == "os"


# ---------------------------------------------------------------------------
# format=ndjson (DuckDB / jq / pandas friendly)
# ---------------------------------------------------------------------------


def test_v1_search_ndjson(client):
    r = client.get(
        "/api/v1/search",
        params={"q": "x", "source": "code", "format": "ndjson"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert rows[0]["symbol"] == "Foo.bar"


def test_v1_graph_ndjson(client):
    r = client.get(
        "/api/v1/graph",
        params={"operation": "callers", "symbol": "Foo.bar", "format": "ndjson"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert rows[0]["relation"] == "calls"


def test_v1_sources_ndjson(client):
    r = client.get("/api/v1/sources", params={"format": "ndjson"})
    assert r.status_code == 200
    names = {json.loads(ln)["name"] for ln in r.text.splitlines() if ln.strip()}
    assert names == {"code", "docs"}


def test_v1_default_format_stays_wrapped(client):
    # Regression: the default JSON keeps its envelope (Coral's rows_path relies
    # on `results` / `sources`).
    assert "results" in client.get(
        "/api/v1/search", params={"q": "x", "source": "code"}
    ).json()
    assert "sources" in client.get("/api/v1/sources").json()


# ---------------------------------------------------------------------------
# view=outline (signature triage; body fetched on demand)
# ---------------------------------------------------------------------------


def test_v1_search_outline_drops_body_for_signature(client):
    r = client.get(
        "/api/v1/search",
        params={"q": "x", "source": "code", "view": "outline"},
    )
    assert r.status_code == 200
    row = r.json()["results"][0]
    assert "content" not in row                       # body dropped
    assert row["signature"] == "def bar(self, x)"     # compact declaration
    assert row["doc"] == "Return x, doubled."         # in-chunk docstring
    # navigation fields kept so the agent can read the real body on demand
    assert row["file_path"] == "/repo/a.py"
    assert row["start_line"] == 10 and row["end_line"] == 20


def test_v1_search_default_view_keeps_body(client):
    row = client.get(
        "/api/v1/search", params={"q": "x", "source": "code"}
    ).json()["results"][0]
    assert "content" in row and "signature" not in row


def test_v1_search_outline_composes_with_ndjson(client):
    r = client.get(
        "/api/v1/search",
        params={"q": "x", "source": "code", "view": "outline", "format": "ndjson"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    row = json.loads(r.text.splitlines()[0])
    assert "content" not in row and row["signature"] == "def bar(self, x)"
