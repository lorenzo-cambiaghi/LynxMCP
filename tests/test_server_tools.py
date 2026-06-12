"""Unit tests for the fixed MCP tool surface in server.py.

The core property under test: the number of registered tools is CONSTANT
in the number of sources. The per-source naming scheme this replaced
(~17 tools per source) blew client tool limits as soon as a user added a
second or third source.

Uses a fake manager (no embedding model, no ChromaDB) — these tests are
about registration and routing, not retrieval quality.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp.server.fastmcp import FastMCP

from lynx.server import (
    _register_search_tools,
    _register_global_tools,
    _register_graph_tools,
    _register_combined_tools,
    _resolve_source,
)


class FakeBackend:
    def __init__(self, type_name="codebase", graph=None, git=False, path="/x"):
        self.type_name = type_name
        self.graph = graph
        self.source_config = {
            "path": path,
            "git_integration": {"enabled": git},
        }


class FakeManager:
    """Records calls; returns canned results shaped like the real ones."""

    def __init__(self, backends):
        self.backends = backends
        self.config = SimpleNamespace(
            search=SimpleNamespace(default_top_k=5)
        )
        self.calls = []

    def get(self, source):
        if source not in self.backends:
            raise KeyError(f"Unknown source {source!r}")
        return self.backends[source]

    def search(self, source, query, top_k=5, **kw):
        self.calls.append(("search", source, query))
        return [{"file": "a.py", "score": 0.9, "content": "x"}]

    def search_all(self, query, top_k=5, **kw):
        self.calls.append(("search_all", query))
        return [{"file": "a.py", "score": 0.9, "content": "x", "source": "one"}]

    def deep_search(self, source, queries, top_k=5, **kw):
        self.calls.append(("deep_search", source))
        return {
            "results": [], "variants_tried": len(queries),
            "winning_variant_index": None, "all_weak": True,
        }

    def deep_search_all(self, queries, top_k=5, **kw):
        self.calls.append(("deep_search_all",))
        return {
            "results": [], "variants_tried": len(queries),
            "winning_variant_index": None, "all_weak": True,
        }

    def find_definition(self, source, symbol, limit=10):
        self.calls.append(("find_definition", source, symbol))
        return []


def _register_everything(manager):
    mcp = FastMCP("test")
    _register_search_tools(mcp, manager)
    _register_global_tools(mcp, manager)
    if any(b.graph is not None for b in manager.backends.values()):
        _register_graph_tools(mcp, manager)
    if any(b.type_name == "codebase" for b in manager.backends.values()):
        _register_combined_tools(mcp, manager)
    return mcp


def _tools(mcp):
    return mcp._tool_manager._tools


def test_tool_count_constant_in_number_of_sources():
    one = FakeManager({"a": FakeBackend(graph=object(), git=True)})
    three = FakeManager({
        "a": FakeBackend(graph=object(), git=True),
        "b": FakeBackend(graph=object(), git=True),
        "c": FakeBackend(type_name="webdoc"),
    })
    n_one = len(_tools(_register_everything(one)))
    n_three = len(_tools(_register_everything(three)))
    assert n_one == n_three, "tool count must not grow with sources"
    assert n_three <= 12, f"tool surface too large: {n_three}"


def test_expected_fixed_tool_names():
    mgr = FakeManager({"a": FakeBackend(graph=object(), git=True)})
    names = set(_tools(_register_everything(mgr)))
    assert names == {
        "search", "deep_search", "list_sources", "update_source_index",
        "get_rag_status", "graph_query", "find_definition", "find_usages",
        "find_tests_for", "find_similar", "search_diff",
    }


def test_conditional_tools_skipped_without_capability():
    # webdoc-only config: no graph_query, no find_*, no search_diff.
    mgr = FakeManager({"docs": FakeBackend(type_name="webdoc")})
    names = set(_tools(_register_everything(mgr)))
    assert names == {
        "search", "deep_search", "list_sources", "update_source_index",
        "get_rag_status",
    }


def test_descriptions_carry_source_catalog():
    mgr = FakeManager({
        "mygame": FakeBackend(graph=object()),
        "docs": FakeBackend(type_name="webdoc"),
    })
    tools = _tools(_register_everything(mgr))
    for name in ("search", "deep_search"):
        desc = tools[name].description
        assert desc and "mygame" in desc and "docs" in desc


def test_search_routes_to_one_source_or_all():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    tools = _tools(_register_everything(mgr))

    tools["search"].fn(query="q", source="a")
    assert mgr.calls[-1] == ("search", "a", "q")

    tools["search"].fn(query="q")  # no source → fan out
    assert mgr.calls[-1] == ("search_all", "q")

    out = tools["search"].fn(query="q", source="nope")
    assert "Error" in out and "nope" in out


def test_find_definition_defaults_single_codebase_source():
    mgr = FakeManager({
        "code": FakeBackend(),
        "docs": FakeBackend(type_name="webdoc"),
    })
    tools = _tools(_register_everything(mgr))
    tools["find_definition"].fn(symbol="Foo")  # source omitted
    assert mgr.calls[-1] == ("find_definition", "code", "Foo")


def test_find_definition_ambiguous_source_lists_candidates():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    tools = _tools(_register_everything(mgr))
    out = tools["find_definition"].fn(symbol="Foo")
    assert "Error" in out and "a" in out and "b" in out


# ---------------------------------------------------------------------------
# _resolve_source unit tests
# ---------------------------------------------------------------------------


def test_resolve_source_explicit_valid():
    mgr = FakeManager({"a": FakeBackend(), "b": FakeBackend()})
    assert _resolve_source(mgr, "b") == "b"


def test_resolve_source_unknown_raises_with_available():
    mgr = FakeManager({"a": FakeBackend()})
    with pytest.raises(ValueError, match="unknown source"):
        _resolve_source(mgr, "zzz")


def test_resolve_source_predicate_mismatch():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc"), "b": FakeBackend()})
    with pytest.raises(ValueError, match="does not support"):
        _resolve_source(mgr, "a", predicate=lambda b: b.type_name == "codebase")


def test_resolve_source_none_unambiguous():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc"), "b": FakeBackend()})
    got = _resolve_source(mgr, None, predicate=lambda b: b.type_name == "codebase")
    assert got == "b"


def test_resolve_source_none_no_candidates():
    mgr = FakeManager({"a": FakeBackend(type_name="webdoc")})
    with pytest.raises(ValueError, match="no configured"):
        _resolve_source(mgr, None, predicate=lambda b: b.type_name == "codebase")
