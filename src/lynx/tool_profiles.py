"""Tool profiles: which MCP tools a session gets, and why that is a setting.

Every tool definition an MCP server exposes rides along in the client's
context on every turn. Lynx's seventeen tools cost about 8,100 tokens per
session before this module existed, against a saving of roughly 2,400 tokens
per retrieval compared to agentic grep: a session that searched three times
had not yet paid for its own tool list. Most sessions need a handful of tools,
so the tool set is layered:

  core      the five you reach for in an ordinary coding session
  standard  core plus the precise lookups, duplicate detection, graph query,
            repo orientation and diff-scoped search (default)
  full      everything, including escalation search, feedback, maintenance and export

A profile is a ceiling on what gets registered, never a promise: a tool that
needs the graph layer or git integration is still registered only when a
source supports it. `include` and `exclude` fine-tune a profile by name.

Standard library only: `config.py` imports this at load time.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

# Every tool `run_server` can register, in the order the docs list them.
ALL_TOOLS: Tuple[str, ...] = (
    "search",
    "prompt_hook",
    "read_guard",
    "deep_search",
    "graph_query",
    "find_definition",
    "find_usages",
    "find_tests_for",
    "find_similar",
    "describe_symbol",
    "impact",
    "module_summary",
    "repo_overview",
    "export_graph",
    "search_diff",
    "feedback",
    "list_sources",
    "get_rag_status",
    "update_source_index",
)

CORE_TOOLS: Tuple[str, ...] = (
    "search",
    # Not for the model: Claude Code's UserPromptSubmit hook calls it with the
    # prompt and puts the map in front of the first turn (prompt_hook.py). In
    # every profile, since the hook fails when the tool is not there.
    "prompt_hook",
    # Not for the model either: the PreToolUse hook on Read (read_guard.py).
    "read_guard",
    "describe_symbol",
    # A file as signatures and doc lines: what an agent reads instead of
    # opening a file it does not know yet. Works without the graph.
    "module_summary",
    "find_usages",
    "impact",
)

STANDARD_TOOLS: Tuple[str, ...] = CORE_TOOLS + (
    "find_definition",
    # "Does this already exist?" asked with a draft in hand: the best answer
    # Lynx has to duplicated code, and it used to be reachable only from the
    # command line or the full profile.
    "find_similar",
    "graph_query",
    "repo_overview",
    "search_diff",
)

# Tools a client should load with the session instead of deferring them
# behind a tool search. Claude Code defers every MCP tool by default: the model
# sees only the names, has to fetch a schema before a call, and Grep and Glob,
# always loaded, win by default. `search` is the way in for finding code, so it
# is loaded; the instructions name the others, which the model fetches when it
# needs them (on the Sketchbook bench it loaded find_usages, describe_symbol and
# module_summary in one tool search). find_similar was here too until
# 2026-11-21: its schema rode in every turn of sessions that never wrote a
# helper.
ALWAYS_LOAD: Tuple[str, ...] = ("search",)

PROFILES = {
    "core": CORE_TOOLS,
    "standard": STANDARD_TOOLS,
    "full": ALL_TOOLS,
}

DEFAULT_PROFILE = "standard"

# Tools whose registration depends on what the configured sources support.
# Mirrors the gating in `server.run_server`; `tests/test_tool_profiles.py`
# asserts the two agree, so this table cannot drift silently.
_NEEDS_CODEBASE = frozenset({
    "find_definition", "find_usages", "find_tests_for", "find_similar",
    "describe_symbol", "impact", "repo_overview", "module_summary",
    "export_graph", "search_diff",
})
_NEEDS_GRAPH = frozenset({"graph_query", "export_graph"})
_NEEDS_GIT = frozenset({"search_diff"})


class ToolProfileError(ValueError):
    """An unknown profile or tool name in the configuration."""


def validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ToolProfileError(
            f"unknown tool profile {profile!r}; choose one of {', '.join(PROFILES)}"
        )
    return profile


def validate_tool_names(names: Iterable[str], *, what: str) -> Tuple[str, ...]:
    out = []
    for n in names:
        if n not in ALL_TOOLS:
            raise ToolProfileError(
                f"unknown tool {n!r} in tools.{what}; known tools: {', '.join(ALL_TOOLS)}"
            )
        if n not in out:
            out.append(n)
    return tuple(out)


def available_tools(*, has_codebase: bool, has_graph: bool, has_git: bool) -> List[str]:
    """The tools `run_server` would register for these capabilities, before
    any profile is applied."""
    out = []
    for name in ALL_TOOLS:
        if name in _NEEDS_CODEBASE and not has_codebase:
            continue
        if name in _NEEDS_GRAPH and not has_graph:
            continue
        if name in _NEEDS_GIT and not has_git:
            continue
        out.append(name)
    return out


def select_tools(
    available: Iterable[str],
    profile: str = DEFAULT_PROFILE,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Tuple[List[str], List[str]]:
    """Split `available` into (kept, dropped) for a profile.

    `include` adds tools to the profile, `exclude` removes them; exclude wins
    when a name is in both. Order follows `ALL_TOOLS`.
    """
    validate_profile(profile)
    wanted: Set[str] = set(PROFILES[profile])
    wanted.update(validate_tool_names(include, what="include"))
    wanted.difference_update(validate_tool_names(exclude, what="exclude"))
    avail = set(available)
    kept = [n for n in ALL_TOOLS if n in avail and n in wanted]
    dropped = [n for n in ALL_TOOLS if n in avail and n not in wanted]
    # Anything registered under a name this module does not know is kept:
    # a profile must never silently hide a tool it cannot reason about.
    kept += sorted(avail - set(ALL_TOOLS))
    return kept, dropped
