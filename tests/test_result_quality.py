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


# A body long enough (>= _DEDUP_MIN_CHARS) to be treated as a confident
# duplicate when it appears twice — i.e. a real (vendored/build) file copy.
_LONG_BODY = (
    "def transform(records):\n"
    "    out = []\n"
    "    for r in records:\n"
    "        if r.valid:\n"
    "            out.append(normalize(r))\n"
    "    return out\n"
)


def test_dedup_collapses_substantial_duplicate_keeping_first():
    from lynx.rag_manager import _dedup_by_content

    results = [
        {"id": "a", "file": "src/foo.py", "content": _LONG_BODY, "score": 0.9},
        {"id": "b", "file": "build/foo.py", "content": _LONG_BODY, "score": 0.8},
        {"id": "c", "file": "src/bar.py", "content": _LONG_BODY + "# variant\n", "score": 0.7},
    ]
    out = _dedup_by_content(results)
    assert [r["id"] for r in out] == ["a", "c"]  # 'b' (build copy) dropped
    assert out[0]["file"] == "src/foo.py"        # highest-ranked survivor kept


def test_dedup_keeps_short_identical_boilerplate():
    from lynx.rag_manager import _dedup_by_content

    # Distinct symbols that happen to share a trivial body must NOT collapse.
    out = _dedup_by_content([
        {"id": "a", "content": "=> base.GetHashCode();"},
        {"id": "b", "content": "=> base.GetHashCode();"},
    ])
    assert [r["id"] for r in out] == ["a", "b"]


def test_dedup_ignores_surrounding_whitespace_on_long_bodies():
    from lynx.rag_manager import _dedup_by_content

    out = _dedup_by_content([
        {"id": "a", "content": _LONG_BODY},
        {"id": "b", "content": "  " + _LONG_BODY + "  "},
    ])
    assert len(out) == 1


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


# ---------------------------------------------------------------------------
# build_overview (overview) — filesystem orientation map
# ---------------------------------------------------------------------------


def test_build_overview_python_django_and_node(tmp_path):
    from lynx.overview import build_overview

    (tmp_path / "manage.py").write_text("import django\n")
    (tmp_path / "requirements.txt").write_text("Django>=5.0\nrequests\n")
    app = tmp_path / "app"
    app.mkdir()
    (app / "views.py").write_text("def index(request): ...\n")
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"react": "^18.0.0"}, '
        '"scripts": {"build": "webpack", "test": "jest", "dev": "vite"}}'
    )
    # A pruned dir that must NOT inflate counts or leak entry points.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "main.py").write_text("should be ignored\n")

    ov = build_overview(tmp_path)

    langs = {l["language"] for l in ov["languages"]}
    assert "Python" in langs
    assert "Django" in ov["frameworks"]
    assert "React" in ov["frameworks"]
    assert "requirements.txt" in ov["manifests"]
    assert "package.json" in ov["manifests"]
    entry_files = {e["file"] for e in ov["entry_points"]}
    assert "manage.py" in entry_files
    # node_modules pruned: its main.py must not appear as an entry point.
    assert not any("node_modules" in e["file"] for e in ov["entry_points"])
    assert "npm run build" in ov["commands"]["build"]
    assert "pytest" in ov["commands"]["test"]
    assert "npm run dev" in ov["commands"]["run"]


def test_build_overview_nonexistent_dir():
    from lynx.overview import build_overview

    ov = build_overview("/no/such/path/anywhere")
    assert "error" in ov


# ---------------------------------------------------------------------------
# impact / module_summary / repo_overview formatters
# ---------------------------------------------------------------------------


def test_format_impact_renders_callers_by_depth_and_tests():
    from lynx.server import _format_impact

    d = {
        "symbol": "GetVoxel", "graph_enabled": True,
        "callers": [
            {"node": {"label": "Renderer.Draw", "file": "Renderer.cs",
                      "start_line": 5, "end_line": 9}, "depth": 1, "confidence": "resolved"},
            {"node": {"label": "Game.Loop", "file": "Game.cs",
                      "start_line": 1, "end_line": 3}, "depth": 2, "confidence": "extracted"},
        ],
        "tests": [{"symbol": "VoxelTests", "file": "tests/VoxelTests.cs",
                   "start_line": 1, "end_line": 4}],
    }
    text = _format_impact("GetVoxel", d)
    assert "Renderer.Draw" in text and "[d1]" in text
    assert "Game.Loop" in text and "[d2]" in text
    assert "TESTS to re-run" in text and "VoxelTests" in text


def test_format_impact_graph_off_hint():
    from lynx.server import _format_impact

    d = {"symbol": "x", "graph_enabled": False, "callers": [], "tests": []}
    assert "enable the graph layer" in _format_impact("x", d)


def test_format_module_summary_sections():
    from lynx.server import _format_module_summary

    d = {
        "file": "VoxelWorld.cs", "graph_enabled": True,
        "symbols": [{"label": "VoxelWorld.GetVoxel", "kind": "method",
                     "file": "VoxelWorld.cs", "start_line": 91, "end_line": 91}],
        "imports": [{"module": "UnityEngine", "target": {}}],
        "dependent_files": ["Renderer.cs", "Game.cs"],
    }
    text = _format_module_summary("VoxelWorld.cs", d)
    assert "DEFINES (1)" in text and "VoxelWorld.GetVoxel" in text
    assert "IMPORTS:" in text and "UnityEngine" in text
    assert "DEPENDED ON BY (2 file(s)" in text and "Renderer.cs" in text


def test_format_module_summary_graph_off():
    from lynx.server import _format_module_summary

    d = {"file": "x.py", "graph_enabled": False,
         "symbols": [], "imports": [], "dependent_files": []}
    assert "needs the graph layer" in _format_module_summary("x.py", d)


def test_format_repo_overview_sections():
    from lynx.server import _format_repo_overview

    d = {
        "root": "/proj", "file_count": 42,
        "languages": [{"language": "C#", "files": 30}],
        "frameworks": ["Unity"],
        "manifests": ["proj.csproj"],
        "entry_points": [{"file": "Program.cs", "hint": ".NET program entrypoint"}],
        "commands": {"build": ["dotnet build"], "test": ["dotnet test"], "run": []},
    }
    text = _format_repo_overview(d)
    assert "LANGUAGES: C# (30)" in text
    assert "FRAMEWORKS: Unity" in text
    assert "Program.cs" in text
    assert "build: dotnet build" in text


# ---------------------------------------------------------------------------
# graph/render — self-contained graph views
# ---------------------------------------------------------------------------


def _tiny_graph():
    import networkx as nx
    G = nx.DiGraph()
    G.add_node("s", label="Target", kind="function", file="/repo/a.py", start_line=1)
    G.add_node("c", label="Caller", kind="function", file="/repo/a.py", start_line=5)
    G.add_node("e", label="Callee<T>", kind="function", file="/repo/b.py", start_line=9)
    G.add_edge("c", "s", relation="calls", confidence="resolved")
    G.add_edge("s", "e", relation="calls", confidence="resolved")
    return G


def test_build_symbol_view_roles_and_layers():
    from lynx.graph.render import build_symbol_view

    m = build_symbol_view(_tiny_graph(), "Target", depth=2, root="/repo")
    roles = {n["label"]: n["role"] for n in m["nodes"]}
    layers = {n["label"]: n["layer"] for n in m["nodes"]}
    assert roles["Target"] == "seed" and layers["Target"] == 0
    assert roles["Caller"] == "caller" and layers["Caller"] < 0   # above
    assert roles["Callee<T>"] == "callee" and layers["Callee<T>"] > 0  # below
    assert len(m["edges"]) == 2


def test_build_symbol_view_missing_symbol():
    from lynx.graph.render import build_symbol_view

    m = build_symbol_view(_tiny_graph(), "Nope", depth=2)
    assert m.get("empty") and "not found" in m["reason"]


def test_render_html_is_self_contained_and_escaped():
    from lynx.graph.render import build_symbol_view, render_html

    m = build_symbol_view(_tiny_graph(), "Target", depth=2, root="/repo")
    out = render_html(m, source="proj", lynx_version="9.9.9")

    assert out.lower().startswith("<!doctype html>")
    # Air-gap invariant: no external resource fetches at view time.
    assert "src=" not in out
    assert "<link" not in out
    assert "<script" not in out      # v1 is pure SVG, no JS
    # Labels are HTML-escaped (Callee<T> must not appear raw).
    assert "Callee&lt;T&gt;" in out
    assert "Callee<T>" not in out
    # Paths rendered repo-relative (root stripped).
    assert "/repo/a.py" not in out
    assert "a.py" in out


def test_render_svg_standalone_and_deterministic():
    from lynx.graph.render import build_symbol_view, render_svg

    m = build_symbol_view(_tiny_graph(), "Target", depth=2, root="/repo")
    a = render_svg(m)
    b = render_svg(m)
    assert a == b                       # deterministic: no timestamps/randomness
    assert a.startswith("<svg")
    assert "<style>" in a               # self-styled (renders as a bare image)
    assert "UTC" not in a               # no timestamp baked into the image
    assert "<script" not in a and "src=" not in a


# ---------------------------------------------------------------------------
# graph/query — new traversal primitives (transitive_callers, nodes_in_file)
# ---------------------------------------------------------------------------


def test_transitive_callers_depth_and_ordering():
    import networkx as nx
    from lynx.graph.query import transitive_callers

    G = nx.DiGraph()
    for nid in ("t", "c", "b", "a"):
        G.add_node(nid, label=nid.upper(), kind="function", file="x.py", start_line=1)
    # a -> b -> c -> t  (so t's transitive callers are c@1, b@2, a@3)
    G.add_edge("c", "t", relation="calls")
    G.add_edge("b", "c", relation="calls")
    G.add_edge("a", "b", relation="calls")
    G.add_edge("t", "t", relation="calls")  # self-loop must be ignored

    res = transitive_callers(G, "T", max_depth=3)
    by_label = {r["node"]["label"]: r["depth"] for r in res}
    assert by_label == {"C": 1, "B": 2, "A": 3}
    # Depth cap respected.
    shallow = transitive_callers(G, "T", max_depth=1)
    assert {r["node"]["label"] for r in shallow} == {"C"}


def test_nodes_in_file_filters_kind_and_path():
    import networkx as nx
    from lynx.graph.query import nodes_in_file

    G = nx.DiGraph()
    G.add_node("f1", label="Foo", kind="class", file="/repo/foo.py", start_line=1)
    G.add_node("f2", label="Foo.bar", kind="method", file="/repo/foo.py", start_line=3)
    G.add_node("file", label="foo.py", kind="file", file="/repo/foo.py")
    G.add_node("ext", label="os", kind="external", file=None)
    G.add_node("other", label="Baz", kind="class", file="/repo/baz.py", start_line=1)

    labels = {n["label"] for n in nodes_in_file(G, "foo.py")}
    assert labels == {"Foo", "Foo.bar"}     # file/external excluded; baz.py excluded


def test_build_module_view_hub():
    import networkx as nx
    from lynx.graph.render import build_module_view

    G = nx.DiGraph()
    # service.py defines Service.run; main.py calls it (a dependent).
    G.add_node("svc", label="Service.run", kind="method",
               file="/repo/service.py", start_line=2)
    G.add_node("main", label="main", kind="function",
               file="/repo/main.py", start_line=1)
    G.add_edge("main", "svc", relation="calls", confidence="resolved")

    m = build_module_view(G, "service.py", root="/repo")
    assert not m.get("empty")
    roles = {n["role"] for n in m["nodes"]}
    assert "seed" in roles and "dependent" in roles
    assert any("Service.run" in s for s in m["sidebar"])
