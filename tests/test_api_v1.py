"""Tests for the stable /api/v1/* JSON surface (the external integration
contract used by the Coral source spec in integrations/coral/).

Uses FastAPI's TestClient with a fake manager injected on app.state —
no embedding model, no ChromaDB.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lynx.manager.ui.app import create_app


class FakeManager:
    def __init__(self):
        self.config = SimpleNamespace(storage_path="/tmp/x")
        self.backends = {"code": object(), "docs": object()}

    def get(self, source):
        if source not in self.backends:
            raise KeyError(f"Unknown source {source!r}")
        return self.backends[source]

    def search(self, source, q, top_k=8, **kw):
        return [{
            "file": "a.py", "file_path": "/repo/a.py",
            "symbol_name": "Foo.bar", "symbol_kind": "method",
            "language": "python", "start_line": 10, "end_line": 20,
            "score": 0.031, "content": "def bar(): ...",
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
