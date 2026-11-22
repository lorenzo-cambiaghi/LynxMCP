"""AI client integration helpers — pure functions, no FastAPI deps.

Each supported client gets a small dict describing:
- `key`        — URL slug used in /api/integrations/<key>/rules.
- `label`      — human-readable name shown on the page card.
- `config_file`— path the user pastes the snippet into (informational).
- `rules_file` — recommended path for the AI rules file (relative to repo
  root); `None` for clients that don't read one.
- `snippet`    — JSON snippet pre-populated with the current interpreter
  and config path.
- `notes`      — short paragraph rendered under the snippet (optional).

The rules file a user pastes into CLAUDE.md / AGENTS.md /
.cursor/rules/lynx.md comes from `render_rules_file` below, a short pointer.
The full playbook is `render_rules_for_sources`, served as the `lynx://guide`
MCP resource.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from ...client_connector import (
    CLIENTS as _CLIENT_SPECS,
    get_client_spec,
    inspect_client,
    render_snippet,
)


def _spec_to_dict(spec, config_path: Optional[Path] = None, *, with_status: bool = False) -> dict:
    entry = {
        "key": spec.key,
        "label": spec.label,
        "short_label": spec.short_label,
        "config_file": spec.config_file_hint,
        "rules_file": spec.rules_file,
        "notes": spec.notes,
        "supports_auto_connect": spec.supports_auto_connect,
    }
    if with_status:
        st = inspect_client(spec)
        entry["detected"] = st.detected
        entry["connected"] = st.connected
        entry["locations"] = list(st.locations)
        entry["resolved_config_path"] = str(st.config_path) if st.config_path else None
        entry["status_error"] = st.error
        entry["snippet"] = render_snippet(
            spec, config_path=config_path, python_exe=sys.executable,
        )
    return entry


def build_integrations(config_path: Optional[Path]) -> list[dict]:
    """Return the per-client list with snippet, live status, and rules paths filled in."""
    return [_spec_to_dict(spec, config_path, with_status=True) for spec in _CLIENT_SPECS]


def get_client(key: str) -> Optional[dict]:
    spec = get_client_spec(key)
    if spec is None:
        return None
    return _spec_to_dict(spec, with_status=False)


def render_rules_file(
    source_names: list[str],
    has_graph: bool,
    has_git: bool,
    tools=None,
) -> str:
    """The body a user pastes into CLAUDE.md / AGENTS.md / .cursor/rules.

    A pointer, not a manual. The client already receives the server's
    handshake instructions and every tool description, and a rules file is
    loaded into the context of every call too: the long playbook this used to
    be (`render_rules_for_sources`, now only the `lynx://guide` resource)
    repeated them, measured at 887 tokens per call in Claude Code against
    about 220 for these few lines (Sketchbook bench, 2026-11-21). What stays
    is what a client without handshake instructions still needs: that Lynx
    is there, the source names, and when to reach for it.
    """
    has = (lambda name: True) if tools is None else (lambda name: name in set(tools))
    names = ", ".join(f"`{n}`" for n in source_names) or "(none yet)"
    label = "source" if len(source_names) == 1 else "sources"
    lines = [
        "# Lynx",
        "",
        f"This project is indexed by Lynx (MCP server `lynx`; {label}: {names}).",
    ]
    if has("search"):
        lines.append("To find where something happens, call `search` first with a plain "
                     "description ('where the player picks a seat to enter') instead of "
                     "grep or opening whole files, then read only the lines it points to.")
    lookups = [f"`{n}`" for n in ("find_usages", "find_definition", "describe_symbol") if has(n)]
    if lookups:
        lines.append(f"For a name you already know: {', '.join(lookups)}.")
    structural = [f"`{n}`" for n in ("graph_query", "impact") if has(n)]
    if has_graph and structural:
        lines.append(f"Who calls X, what breaks if X changes: {', '.join(structural)}.")
    if has("find_similar"):
        lines.append("Before writing a new helper: `find_similar` on your draft.")
    lines.append("The full playbook is the `lynx://guide` resource.")
    lines.append("")
    return "\n".join(lines)


def render_rules_for_sources(
    source_names: list[str],
    has_graph: bool,
    has_git: bool,
    tools=None,
) -> str:
    """Build the AI-client rules-file body for the configured sources.

    Tailored to which capabilities are actually available: graph-layer
    tool names appear only when at least one source has the graph layer
    enabled, and diff-search tool names appear only when at least one
    source has git_integration on. `tools`, when given, is the set the
    running server registered (its tool profile); tools outside it are not
    mentioned, so the playbook never recommends a call that cannot be made.
    """
    has = (lambda name: True) if tools is None else (lambda name: name in set(tools))
    lines: list[str] = [
        "# Code Reuse & Library Awareness",
        "",
        "This project exposes a local MCP server (Lynx) with semantic search",
        "over the configured sources. Use the tools BEFORE writing new code",
        "or guessing at library APIs.",
        "",
        "## Sources available",
        "",
    ]
    for n in source_names:
        lines.append(f"- `{n}` — pass it as the `source` argument of the tools below.")
    lines.append("")
    if has("search"):
        lines.append("- `search(query, source=...)` — semantic + lexical hybrid search.")
        lines.append("- `search(query, outline=true)` — same ranked hits, signatures only "
                     "(no bodies). Cheap triage for broad/exploratory queries or a large "
                     "top_k: scan the signatures, then read just the one body you need "
                     "by its file:line. Use the default full search when you'll work "
                     "with the code right away.")
    if has("deep_search"):
        lines.append("- `deep_search(queries, source=...)` — fallback for ambiguous queries.")
    understand = []
    if has("describe_symbol"):
        understand.append("- `describe_symbol(symbol)`: its code, callers, callees and tests in one "
                          "call; for a class, its bases, subtypes and member signatures.")
    if has("module_summary"):
        understand.append("- `module_summary(file)`: every symbol of a file with its signature and "
                          "doc line. Read it instead of opening a file you don't know.")
    if has("impact"):
        understand.append("- `impact(symbol)` — transitive blast radius of a change, plus the tests to re-run.")
    if has("repo_overview"):
        understand.append("- `repo_overview()` — languages, entry points and build/test commands of an unfamiliar repo.")
    if understand:
        lines.append("")
        lines.append("## Understand before you change")
        lines.append("")
        lines.extend(understand)
    if len(source_names) > 1:
        lines.append("")
        lines.append("When unsure which source has the answer, omit `source` to "
                     "search ALL sources at once.")
    lines.append("")
    lines.append("## When to search")
    lines.append("")
    lines.append("Before implementing any utility, interface, or pattern,")
    lines.append("search the codebase source first to avoid duplication.")
    lines.append("Before invoking a library API, search the relevant docs source")
    lines.append("if one exists — your training data may predate the version in use.")
    structural = []
    if has("find_definition"):
        structural.append("- `find_definition(symbol)` — where is X defined?")
    if has("find_usages"):
        structural.append("- `find_usages(symbol)` — who calls X? typeof / generics included.")
    if has("find_tests_for"):
        structural.append("- `find_tests_for(symbol)` — are there tests for X?")
    if has("find_similar"):
        structural.append("- `find_similar(snippet)` — is there code similar to this?")
    if has_graph and has("graph_query"):
        structural.append("- `graph_query(operation, symbol)` — operations: callers, callees,")
        structural.append("  subclasses, superclasses, imports, neighbors, shortest_path,")
        structural.append("  overview (god nodes + communities), surprising_connections, status.")
    if has_graph and structural:
        lines.append("")
        lines.append("## Code-aware structural queries (graph layer enabled)")
        lines.append("")
        lines.extend(structural)
    if has_git and has("search_diff"):
        lines.append("")
        lines.append("## Diff-aware search (git_integration enabled)")
        lines.append("")
        lines.append("- `search_diff(query)` — search only in files changed vs `main`.")
        lines.append("  Use during code review: 'I changed the discount logic — what else uses the same formula?'")
    lines.append("")
    lines.append("## How to search")
    lines.append("")
    lines.append("Describe what the code *does*, not its name. Semantic search")
    lines.append("beats `grep` precisely because it bridges synonyms.")
    lines.append("")
    return "\n".join(lines)
