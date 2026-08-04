"""Every MCP tool has a CLI command — asserted, not claimed.

The question this file answers is the one that started the work: "is all
of Lynx reachable from a terminal?" For a long time the honest answer was
no — ten of the seventeen tools were agent-only, and nothing in the repo
noticed. Enumerating the tools by hand to check is exactly the kind of
audit that gets done once and then rots.

So the tools are registered for real (the same four `_register_*` calls
`run_server` makes) and every registered name is looked up in an explicit
mapping to its command. Adding a tool without a CLI now fails here, with
the tool's name in the message.

The mapping is explicit rather than derived: most names are a mechanical
`_` → `-`, but a handful are deliberately not (`update_source_index` is
`lynx build`, `graph_query` is a sub-command of `graph`), and writing
those out is what makes an unmapped newcomer impossible to miss.
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
