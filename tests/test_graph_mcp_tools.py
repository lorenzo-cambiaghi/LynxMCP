"""Verifies that `_register_graph_tools` exposes the single `graph_query`
tool, that its description carries the operation catalog + source names,
and that each operation calls through to SourceManager / GraphLayer.

Reuses the stubbed-RAG trick from test_graph_integration so we don't
have to load the HuggingFace embedding model. Pytest-style: the manager
and the FastMCP instance are built once per module via a fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_codebase(root: Path, files: dict) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _stub_rag():
    """Same trick as test_graph_integration — stub CodebaseRAG so we don't
    need HuggingFace running."""
    from lynx.rag_manager import CodebaseRAG

    class _StubCollection:
        def count(self):
            return 0

    class _StubVS:
        def __init__(self):
            self._collection = _StubCollection()

    def stub_init(self, **kwargs):
        self.codebase_path = Path(kwargs["codebase_path"])
        self.storage_path = Path(kwargs["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVS()

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.check_config_drift = lambda self: None
    CodebaseRAG.drift_status_text = lambda self: "No drift."
    CodebaseRAG.needs_update = lambda self: False


@pytest.fixture(scope="module")
def graph_mcp(tmp_path_factory):
    """A FastMCP with graph tools registered over a tiny synthetic codebase.

    Codebase: util.py defines `helper` + an inheritance chain
    (Base <- Derived) used to exercise subclasses / superclasses.
    """
    tmp = tmp_path_factory.mktemp("lynx-mcp-tools")
    code = tmp / "code"
    code.mkdir()
    _make_codebase(code, {
        "util.py": (
            "def helper(x):\n    return x + 1\n"
            "class Base:\n    pass\n"
        ),
        "main.py": (
            "from util import helper, Base\n"
            "def go():\n    return helper(1)\n"
            "class Derived(Base):\n    pass\n"
        ),
    })

    _stub_rag()

    from lynx.config import load_config
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {
            "demo": {
                "type": "codebase",
                "path": str(code),
                "supported_extensions": [".py"],
                "graph": {"enabled": True},
                "watcher": {"enabled": False},
            }
        }
    }), encoding="utf-8")
    cfg = load_config(cfg_path)

    from lynx.source_manager import SourceManager
    mgr = SourceManager(cfg)
    mgr.get("demo").graph.rebuild(force=True)

    from mcp.server.fastmcp import FastMCP
    from lynx.server import _register_graph_tools

    mcp = FastMCP("test")
    _register_graph_tools(mcp, mgr)
    return mcp


def _tool(graph_mcp):
    # FastMCP exposes registered tools via _tool_manager._tools (dict)
    return graph_mcp._tool_manager._tools["graph_query"]


def test_single_graph_query_tool_with_rich_description(graph_mcp):
    registered = list(graph_mcp._tool_manager._tools.keys())
    assert registered == ["graph_query"], registered

    # The description MUST be non-empty (f-string "docstrings" silently
    # produce empty descriptions) and must carry both the routing info the
    # client needs: the operation catalog and the source name.
    desc = _tool(graph_mcp).description
    assert desc and len(desc) > 100
    assert "demo" in desc
    for op in (
        "callers", "callees", "subclasses", "superclasses", "imports",
        "neighbors", "shortest_path", "overview", "surprising_connections",
        "status",
    ):
        assert op in desc, f"operation {op!r} missing from description"


def test_callers_operation(graph_mcp):
    out = _tool(graph_mcp).fn(operation="callers", symbol="helper")
    assert "Callers of 'helper'" in out
    assert "helper" in out


def test_source_defaults_to_only_graph_source(graph_mcp):
    # `source` omitted → resolves to 'demo' since it's the only graph source.
    out = _tool(graph_mcp).fn(operation="status")
    assert "Graph status for 'demo'" in out


def test_unknown_operation_lists_valid_ones(graph_mcp):
    out = _tool(graph_mcp).fn(operation="explode", symbol="helper")
    assert "unknown operation" in out
    assert "callers" in out


def test_missing_symbol_is_reported(graph_mcp):
    out = _tool(graph_mcp).fn(operation="callers")
    assert "requires `symbol`" in out


def test_overview_operation(graph_mcp):
    out = _tool(graph_mcp).fn(operation="overview", top_n=5, min_community_size=2)
    assert "Architectural overview" in out
    assert "God nodes" in out
    assert "Communities" in out


def test_shortest_path_operation(graph_mcp):
    tool = _tool(graph_mcp)
    # No path → user-friendly text, not a stack trace.
    out = tool.fn(operation="shortest_path", symbol="helper", target="nonexistent", max_hops=5)
    assert "No directed path" in out or "Error" in out
    # Real path
    out = tool.fn(operation="shortest_path", symbol="go", target="helper", max_hops=5)
    assert "Path from 'go'" in out
    assert "helper" in out


def test_subclasses_and_superclasses(graph_mcp):
    tool = _tool(graph_mcp)

    out = tool.fn(operation="subclasses", symbol="Base")
    assert "Subclasses of 'Base'" in out
    assert "Derived" in out, "cross-file inheritance not resolved"
    assert "inherits" in out

    out = tool.fn(operation="superclasses", symbol="Derived")
    assert "Superclasses of 'Derived'" in out
    assert "Base" in out
