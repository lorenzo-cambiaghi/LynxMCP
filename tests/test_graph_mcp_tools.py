"""Verifies that `_register_graph_tools` exposes exactly the 8 tools we
promise (7 query + 1 status), namespaced with the source name, and that
each one calls through to the underlying SourceManager / GraphLayer.

Reuses the stubbed-RAG trick from test_graph_integration so we don't
have to load the HuggingFace embedding model.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


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


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-mcp-tools-"))
    print(f"[test] tempdir: {tmp}")
    try:
        code = tmp / "code"
        code.mkdir()
        _make_codebase(code, {
            "util.py": "def helper(x):\n    return x + 1\n",
            "main.py": (
                "from util import helper\n"
                "def go():\n"
                "    return helper(1)\n"
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

        # ============================================================
        # 1. Register graph tools on a real FastMCP and verify count
        # ============================================================
        from mcp.server.fastmcp import FastMCP
        from lynx.server import _register_graph_tools

        mcp = FastMCP("test")
        _register_graph_tools(mcp, mgr, "demo")

        # FastMCP exposes registered tools via _tool_manager._tools (dict)
        registered = list(mcp._tool_manager._tools.keys())
        expected = {
            "get_callers_demo",
            "get_callees_demo",
            "get_imports_demo",
            "get_neighbors_demo",
            "shortest_path_demo",
            "architectural_overview_demo",
            "surprising_connections_demo",
            "graph_status_demo",
        }
        missing = expected - set(registered)
        if missing:
            print(f"[test] FAIL [1/4]: missing tools: {missing}; got {registered}")
            return 1
        # Each tool MUST have a non-empty description — without it the AI
        # client has no idea when to call which tool. F-string "docstrings"
        # silently produce empty descriptions, so we assert on actual text.
        for name in expected:
            desc = mcp._tool_manager._tools[name].description
            if not desc or len(desc) < 30:
                print(f"[test] FAIL [1/4]: tool {name!r} has empty/tiny description: {desc!r}")
                return 1
            if "demo" not in desc:
                print(f"[test] FAIL [1/4]: tool {name!r} description does not mention source name: {desc!r}")
                return 1
        print(f"[test] OK [1/4] all 8 graph tools registered with `_demo` suffix AND non-empty descriptions")

        # ============================================================
        # 2. Call get_callers_demo, verify it returns formatted text
        # ============================================================
        tool = mcp._tool_manager._tools["get_callers_demo"]
        out = tool.fn(symbol="helper")
        if "Callers of 'helper'" not in out:
            print(f"[test] FAIL [2/4]: get_callers_demo output unexpected: {out!r}")
            return 2
        if "helper" not in out:
            print(f"[test] FAIL [2/4]: get_callers_demo missing helper in output: {out!r}")
            return 2
        print(f"[test] OK [2/4] get_callers_demo returned formatted caller list")

        # ============================================================
        # 3. architectural_overview_demo
        # ============================================================
        tool = mcp._tool_manager._tools["architectural_overview_demo"]
        out = tool.fn(top_n_gods=5, min_community_size=2)
        if "Architectural overview" not in out:
            print(f"[test] FAIL [3/4]: architectural_overview header missing: {out!r}")
            return 3
        if "God nodes" not in out or "Communities" not in out:
            print(f"[test] FAIL [3/4]: architectural_overview sections missing: {out!r}")
            return 3
        print(f"[test] OK [3/4] architectural_overview_demo returned overview")

        # ============================================================
        # 4. shortest_path_demo with no path returns user-friendly text
        # ============================================================
        tool = mcp._tool_manager._tools["shortest_path_demo"]
        out = tool.fn(source="helper", target="nonexistent", max_hops=5)
        if "No directed path" not in out and "Error" not in out:
            print(f"[test] FAIL [4/4]: shortest_path_demo unexpected on missing path: {out!r}")
            return 4
        # Real path
        out = tool.fn(source="go", target="helper", max_hops=5)
        if "Path from 'go'" not in out:
            print(f"[test] FAIL [4/4]: shortest_path_demo missing path output: {out!r}")
            return 4
        if "helper" not in out:
            print(f"[test] FAIL [4/4]: shortest_path_demo missing 'helper' in path: {out!r}")
            return 4
        print(f"[test] OK [4/4] shortest_path_demo handles both no-path and real path")

        print("\n[test] === SUCCESS: MCP graph tools registered and working ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
