"""Pytest-style regression tests for the result-quality + comprehension
helpers added alongside the `describe_symbol` tool.

These are pure-function tests (no index, no model load) so they run in CI
under `pytest tests -q` — unlike the main()-runner smoke tests, which the
conftest keeps out of collection.

Covers:
  - _dedup_by_content: identical bodies collapse, order/highest-rank preserved
  - _apply_codebase_defaults: new codebase sources get default ignores
  - _format_describe_symbol: every section renders, graph-off hint shown
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# _dedup_by_content (rag_manager) — collapse byte-identical chunk bodies
# ---------------------------------------------------------------------------


def test_dedup_collapses_identical_bodies_keeping_first():
    from lynx.rag_manager import _dedup_by_content

    results = [
        {"id": "a", "file": "src/foo.py", "content": "def f():\n    return 1", "score": 0.9},
        {"id": "b", "file": "build/foo.py", "content": "def f():\n    return 1", "score": 0.8},
        {"id": "c", "file": "src/bar.py", "content": "def g():\n    return 2", "score": 0.7},
    ]
    out = _dedup_by_content(results)
    assert [r["id"] for r in out] == ["a", "c"]  # 'b' (build copy) dropped
    assert out[0]["file"] == "src/foo.py"        # highest-ranked survivor kept


def test_dedup_ignores_surrounding_whitespace():
    from lynx.rag_manager import _dedup_by_content

    out = _dedup_by_content([
        {"id": "a", "content": "x = 1\n"},
        {"id": "b", "content": "  x = 1  "},
    ])
    assert len(out) == 1


def test_dedup_keeps_distinct_bodies_and_empty_by_id():
    from lynx.rag_manager import _dedup_by_content

    out = _dedup_by_content([
        {"id": "a", "content": ""},
        {"id": "b", "content": ""},   # empty bodies must NOT collapse together
        {"id": "c", "content": "real"},
    ])
    assert [r["id"] for r in out] == ["a", "b", "c"]


def test_dedup_empty_input():
    from lynx.rag_manager import _dedup_by_content

    assert _dedup_by_content([]) == []


# ---------------------------------------------------------------------------
# _apply_codebase_defaults (manager.ui.routes) — default ignore fragments
# ---------------------------------------------------------------------------


def test_codebase_source_gets_default_ignores():
    from lynx.manager.ui.routes import _apply_codebase_defaults, _DEFAULT_CODEBASE_IGNORES

    block = {"type": "codebase", "path": "/x"}
    out = _apply_codebase_defaults(block)
    assert out["ignored_path_fragments"] == _DEFAULT_CODEBASE_IGNORES
    # The defaults cover the build/vendored dirs that cause duplicate hits.
    for frag in ("/node_modules/", "/build/", "/dist/", "/.git/"):
        assert frag in out["ignored_path_fragments"]


def test_explicit_ignores_are_preserved():
    from lynx.manager.ui.routes import _apply_codebase_defaults

    block = {"type": "codebase", "path": "/x", "ignored_path_fragments": ["/only_this/"]}
    out = _apply_codebase_defaults(block)
    assert out["ignored_path_fragments"] == ["/only_this/"]


def test_non_codebase_block_untouched():
    from lynx.manager.ui.routes import _apply_codebase_defaults

    block = {"type": "webdoc", "url": "https://example.com"}
    out = _apply_codebase_defaults(block)
    assert "ignored_path_fragments" not in out


# ---------------------------------------------------------------------------
# _format_describe_symbol (server) — section rendering
# ---------------------------------------------------------------------------


def test_describe_symbol_formatter_renders_all_sections():
    from lynx.server import _format_describe_symbol

    d = {
        "symbol": "GetVoxel",
        "graph_enabled": True,
        "definition": [
            {"symbol": "VoxelWorld.GetVoxel", "kind": "function",
             "file": "VoxelWorld.cs", "start_line": 91, "end_line": 91, "source": "graph"},
        ],
        "called_by": [
            {"source": {"label": "VoxelRenderer.Draw", "file": "VoxelRenderer.cs",
                        "start_line": 10, "end_line": 20}, "confidence": "resolved"},
        ],
        "calls": [
            {"target": {"label": "VoxelChunk.GetLocal", "file": "VoxelChunk.cs",
                        "start_line": 40, "end_line": 41}, "confidence": "resolved"},
        ],
        "tests": [
            {"symbol": "GetVoxel_returns_air", "file": "tests/VoxelTests.cs",
             "start_line": 5, "end_line": 9},
        ],
    }
    text = _format_describe_symbol("GetVoxel", d)
    for section in ("DEFINITION:", "CALLED BY:", "CALLS:", "TESTS:"):
        assert section in text
    assert "VoxelRenderer.Draw" in text   # caller rendered
    assert "VoxelChunk.GetLocal" in text  # callee rendered


def test_describe_symbol_formatter_graph_off_hint():
    from lynx.server import _format_describe_symbol

    d = {
        "symbol": "helper", "graph_enabled": False,
        "definition": [{"symbol": "helper", "kind": "function",
                        "file": "util.py", "start_line": 1, "end_line": 2, "source": "search_bm25"}],
        "called_by": [], "calls": [],
        "tests": [],
    }
    text = _format_describe_symbol("helper", d)
    # When the graph is off, the call sections say how to enable it.
    assert "enable the graph layer" in text


def test_describe_symbol_formatter_empty():
    from lynx.server import _format_describe_symbol

    d = {"symbol": "nope", "graph_enabled": True,
         "definition": [], "called_by": [], "calls": [], "tests": []}
    text = _format_describe_symbol("nope", d)
    assert "No information found" in text
