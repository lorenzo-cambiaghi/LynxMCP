"""Every MCP tool is reachable from the CLI *and* from the web UI —
asserted, not claimed.

The question this file answers is the one that started the work: "is all
of Lynx reachable without an MCP client?" For a long time the honest
answer was no, twice over — ten of the seventeen tools were agent-only,
and the UI's playground had quietly settled at nine with a Graph tab wired
to three of the ten graph operations. Nothing in the repo noticed either
time. Enumerating a surface by hand to check is exactly the kind of audit
that gets done once and then rots.

So the tools are registered for real (the same four `_register_*` calls
`run_server` makes) and every registered name is looked up in two explicit
mappings, in both directions. Adding a tool without a CLI command or
without a UI panel fails here, with the tool's name in the message.

The mappings are written out rather than derived: most CLI names are a
mechanical `_` → `-`, but a handful are deliberately not
(`update_source_index` is `lynx build`, `graph_query` is a sub-command),
and several tools are served by a whole UI page rather than a playground
panel. Writing those out is what makes an unmapped newcomer impossible to
miss.

The first version of this file covered the CLI only — written right after
completing the CLI by hand, and not for the surface that had already
drifted twice. That asymmetry is the reason the UI mapping exists.
"""
from __future__ import annotations

import pytest


class _Backend:
    def __init__(self, type_name="codebase", graph=True, git=True):
        self.type_name = type_name
        self.graph = object() if graph else None
        self.source_config = {
            "path": "/tmp/demo",
            "git_integration": {"enabled": git},
        }


class _Manager:
    """Only what the registrars read while building tool descriptions."""

    def __init__(self):
        self.backends = {"demo": _Backend()}

    class _Config:
        storage_path = "/tmp/storage"
        reports_path = None

    config = _Config()

    def list_sources(self):
        return [{"name": "demo", "type": "codebase", "chunk_count": 1}]


@pytest.fixture(scope="module")
def registered_tools():
    """Every tool name `run_server` would register for a codebase source
    with the graph layer and git enabled — i.e. the full surface."""
    from mcp.server.fastmcp import FastMCP

    from lynx.server import (
        _register_combined_tools, _register_global_tools,
        _register_graph_tools, _register_search_tools,
    )

    mcp = FastMCP("parity")
    manager = _Manager()
    _register_search_tools(mcp, manager)
    _register_global_tools(mcp, manager)
    _register_graph_tools(mcp, manager)
    _register_combined_tools(mcp, manager, has_graph=True)
    return set(mcp._tool_manager._tools)


# tool name -> the command that answers the same question.
# `None` means deliberately CLI-less, with the reason spelled out.
TOOL_TO_CLI = {
    "search": "search",
    "deep_search": "deep-search",
    "list_sources": "list-sources",
    "update_source_index": "build",
    "get_rag_status": "status",
    "graph_query": "graph query",
    "export_graph": "graph export",
    "find_definition": "find-definition",
    "find_usages": "find-usages",
    "find_tests_for": "find-tests-for",
    "find_similar": "find-similar",
    "describe_symbol": "describe-symbol",
    "impact": "impact",
    "repo_overview": "repo-overview",
    "module_summary": "module-summary",
    "search_diff": "search-diff",
    # An agent files a feedback report when the index couldn't answer it;
    # a human typing at a terminal has no such gap to report. The CLI reads
    # the log instead, with `lynx manager feedback`.
    "feedback": None,
}


# tool name -> how the web UI serves it. A `/api/playground/<name>` endpoint
# for the tools you try by hand; a page for the ones that are a view rather
# than a query. `None` would mean deliberately absent — nothing is, now.
TOOL_TO_UI = {
    "search": "playground:search",
    "deep_search": "playground:deep_search",
    "find_definition": "playground:find_definition",
    "find_usages": "playground:find_usages",
    "find_tests_for": "playground:find_tests_for",
    "find_similar": "playground:find_similar",
    "describe_symbol": "playground:describe_symbol",
    "impact": "playground:impact",
    "module_summary": "playground:module_summary",
    "repo_overview": "playground:repo_overview",
    "graph_query": "playground:graph_query",
    "search_diff": "playground:search_diff",
    "export_graph": "playground:export_graph",
    # Served by a page, not a form: these are things you look at.
    "list_sources": "page:/sources",
    "get_rag_status": "page:/sources",
    "update_source_index": "page:/sources",   # the per-source Build button
    "feedback": "page:/",                     # the dashboard's report card
}


def test_every_registered_tool_is_mapped(registered_tools):
    """A tool added without a line here fails, by name."""
    unmapped = sorted(registered_tools - set(TOOL_TO_CLI))
    assert not unmapped, (
        f"tools with no CLI mapping: {unmapped} — add the command, or map it "
        f"to None with the reason it is agent-only"
    )


def test_the_mapping_has_no_ghosts(registered_tools):
    """The other direction: a renamed or removed tool must not leave a
    stale row here quietly asserting nothing."""
    ghosts = sorted(set(TOOL_TO_CLI) - registered_tools)
    assert not ghosts, f"mapped tools that no longer exist: {ghosts}"


def test_every_mapped_command_exists(registered_tools):
    """The commands really are reachable — parsed and dispatched, not just
    written down in this table."""
    from lynx.cli import _DISPATCH, _build_parser

    parser = _build_parser()
    top_level = {}
    for action in parser._subparsers._group_actions:
        top_level.update(action.choices)

    for tool, command in sorted(TOOL_TO_CLI.items()):
        if command is None:
            continue
        head, _, sub = command.partition(" ")
        assert head in top_level, f"{tool}: `lynx {head}` does not exist"
        assert head in _DISPATCH, f"{tool}: `lynx {head}` has no dispatch entry"
        if sub:
            sub_choices = {}
            for action in top_level[head]._subparsers._group_actions:
                sub_choices.update(action.choices)
            assert sub in sub_choices, f"{tool}: `lynx {command}` does not exist"


def test_the_whole_tool_surface_is_covered(registered_tools):
    """The headline claim, stated as a number so a regression is loud:
    every tool but `feedback` is reachable from a terminal."""
    reachable = {t for t in registered_tools if TOOL_TO_CLI.get(t) is not None}
    assert reachable == registered_tools - {"feedback"}
    assert len(registered_tools) == 17, (
        f"the tool surface changed ({len(registered_tools)} tools); "
        f"re-check the CLI mapping"
    )


# ----------------------------------------------------------------------
# The same guarantee for the web UI
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ui_page_source(tmp_path_factory):
    """The rendered playground and dashboard HTML, plus the app's routes."""
    import json

    from fastapi.testclient import TestClient

    from lynx.manager.ui.app import create_app

    tmp = tmp_path_factory.mktemp("parity-ui")
    cfg = tmp / "config.json"
    cfg.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {},
    }), encoding="utf-8")
    app = create_app(cfg)
    app.state.manager = _Manager()
    client = TestClient(app)
    return {
        "playground": client.get("/playground").text,
        "routes": {r.path for r in app.routes},
    }


def test_every_registered_tool_has_a_ui_mapping(registered_tools, ui_page_source):
    unmapped = sorted(registered_tools - set(TOOL_TO_UI))
    assert not unmapped, (
        f"tools with no UI mapping: {unmapped} — add a playground panel, or "
        f"map it to the page that already serves it"
    )


def test_the_ui_mapping_has_no_ghosts(registered_tools):
    ghosts = sorted(set(TOOL_TO_UI) - registered_tools)
    assert not ghosts, f"mapped tools that no longer exist: {ghosts}"


def test_every_mapped_ui_target_exists(registered_tools, ui_page_source):
    """The panel is really wired: the endpoint is a route AND the page
    posts to it. A route with no form is unreachable; a form with no route
    404s on click — the playground had both kinds of rot before."""
    for tool, target in sorted(TOOL_TO_UI.items()):
        if target is None:
            continue
        kind, _, name = target.partition(":")
        if kind == "playground":
            path = f"/api/playground/{name}"
            assert path in ui_page_source["routes"], f"{tool}: {path} is not a route"
            assert path in ui_page_source["playground"], (
                f"{tool}: the playground page has no form posting to {path}"
            )
        else:
            assert name in ui_page_source["routes"], f"{tool}: page {name} missing"


def test_the_graph_form_offers_every_operation(ui_page_source):
    """`graph_query` maps to one panel, so its coverage is the operation
    list inside that form — three of ten used to be wired."""
    from lynx.graph.dispatch import OPERATIONS

    for op in OPERATIONS:
        assert f'value="{op}"' in ui_page_source["playground"], op


def test_both_surfaces_cover_the_same_tools(registered_tools):
    """Stated as an equality so 'the CLI is complete' can never again be
    true while the UI quietly isn't."""
    cli = {t for t in registered_tools if TOOL_TO_CLI.get(t) is not None}
    ui = {t for t in registered_tools if TOOL_TO_UI.get(t) is not None}
    assert ui == registered_tools
    assert cli == registered_tools - {"feedback"}
