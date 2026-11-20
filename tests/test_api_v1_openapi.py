"""The OpenAPI document of /api/v1, as Coral 1.x imports it.

Coral builds a Provider from this document; its materializer
(`coral-catalog/src/openapi/materialize.rs`, v1.10) refuses a document
without exactly one absolute server URL, without `info.title` and
`info.version`, or outside OpenAPI 3.0/3.1. Each `operationId` becomes the
tool's call path, and the 200 schema becomes its TypeScript return type, so
the models must describe the rows the routes really return.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lynx.manager.ui import api_v1
from lynx.manager.ui.app import create_app
from lynx.manager.ui.routes import _format_v1_edge, _format_v1_hit, _to_outline

from test_api_v1 import FakeManager

BASE = "http://127.0.0.1:8765"


@pytest.fixture()
def client():
    app = create_app(Path("nonexistent-config.json"))
    app.state.manager = FakeManager()
    return TestClient(app, base_url=BASE)


@pytest.fixture()
def doc(client):
    r = client.get("/api/v1/openapi.json")
    assert r.status_code == 200
    return r.json()


def _operations(doc):
    for path, ops in doc["paths"].items():
        for method, op in ops.items():
            yield path, method, op


def test_the_document_is_one_coral_accepts(doc):
    assert re.match(r"^3\.[01]\.\d+$", doc["openapi"])
    assert doc["info"]["title"] == "Lynx" and doc["info"]["version"]
    assert doc["servers"] == [{"url": BASE}]          # exactly one, absolute
    # An empty list, not an absent field: Coral then knows no credential is
    # needed instead of asking the owner for a sign-in method.
    assert doc["security"] == []
    assert "securitySchemes" not in doc.get("components", {})


def test_the_server_is_the_address_the_caller_reached(client):
    r = client.get("/api/v1/openapi.json", headers={"host": "localhost:9000"})
    assert r.json()["servers"] == [{"url": "http://localhost:9000"}]


def test_only_the_v1_routes_are_published(client, doc):
    assert all(path.startswith("/api/v1/") for path in doc["paths"])
    assert "/api/v1/openapi.json" not in doc["paths"]
    assert client.get("/openapi.json").status_code == 404     # UI routes stay private


def test_every_operation_has_its_call_path(doc):
    ids = {(method, path): op.get("operationId") for path, method, op in _operations(doc)}
    assert ids == {
        ("get", "/api/v1/search"): "search.query",
        ("post", "/api/v1/search"): "search.batch",
        ("get", "/api/v1/graph"): "graph.query",
        ("get", "/api/v1/sources"): "sources.list",
    }


def test_every_operation_returns_a_typed_200(doc):
    for path, method, op in _operations(doc):
        schema = op["responses"]["200"]["content"]["application/json"]["schema"]
        assert schema.get("$ref", "").startswith("#/components/schemas/"), (method, path)


def test_texts_fit_coral_presentation_rules(doc):
    # Non-empty, at most 1,024 bytes for a summary, no control characters
    # other than whitespace (types.rs valid_presentation_text).
    def ok(text):
        return text.strip() and not any(c < " " and c not in "\t\n\r" for c in text)

    for path, method, op in _operations(doc):
        assert ok(op["summary"]) and len(op["summary"].encode()) <= 1024, (method, path)
        assert ok(op["description"]) and len(op["description"].encode()) <= 16 * 1024


def test_format_is_hidden_and_enums_are_listed(client, doc):
    params = {(path, p["name"]): p for path, method, op in _operations(doc)
              for p in op.get("parameters", [])}
    assert not any(name == "format" for _path, name in params)
    assert params[("/api/v1/graph", "operation")]["schema"]["enum"] == list(api_v1.GRAPH_OPERATIONS)
    assert params[("/api/v1/search", "view")]["schema"]["enum"] == list(api_v1.SEARCH_VIEWS)
    assert not any(name == "body" for _path, name in params)  # Coral puts the body there
    # Hidden from the schema, still served: DuckDB and jq ask for it by hand.
    r = client.get("/api/v1/sources", params={"format": "ndjson"})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    # An unknown operation is still the route's own 400, not a 422.
    assert client.get("/api/v1/graph", params={"operation": "frobnicate",
                                                "symbol": "X"}).status_code == 400


def _fields(model) -> set:
    return set(model.model_fields)


def test_the_models_describe_the_rows_the_routes_return(client):
    hit = _format_v1_hit({"file": "a.py", "content": "def f():\n    pass"})
    assert set(hit) == _fields(api_v1.SearchHit) - {"signature", "doc"}
    assert set(_to_outline(hit)) == _fields(api_v1.SearchHit) - {"content"}
    edge = _format_v1_edge({"source": {}, "target": {}, "relation": "calls"})
    assert set(edge) == _fields(api_v1.GraphEdge)
    sources = client.get("/api/v1/sources").json()["sources"]
    assert sources and all(set(row) == _fields(api_v1.SourceInfo) for row in sources)
