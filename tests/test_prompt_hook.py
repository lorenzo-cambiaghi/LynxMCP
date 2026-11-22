"""The `prompt_hook` tool: the Lynx map Claude Code's UserPromptSubmit hook
puts in front of the model's first turn.

What is asserted is the contract with Claude Code, measured by probes on
2026-11-21: an `mcp_tool` hook's text reaches the model only as hook JSON
(`hookSpecificOutput.additionalContext`), and `{}` adds nothing. A prompt
must never get an error or a loading message in its context, so every
failure answers `{}`.
"""
from __future__ import annotations

import asyncio
import json

from mcp.server.fastmcp import FastMCP

from lynx import prompt_hook as ph
from lynx.server import _register_search_tools


def _hit(path, line, sig="def f(a, b):", source="demo"):
    return {"rel_path": path, "start_line": line, "end_line": line + 9,
            "content": sig + "\n    return a", "symbol_kind": "function",
            "language": "python", "source": source}


# ---------------------------------------------------------------------------
# the pure part
# ---------------------------------------------------------------------------

def test_one_query_per_sentence_and_short_ones_dropped():
    prompt = ("Bug: the player takes over a car an AI drives.\n\n"
              "Fix it. A seat someone already occupies must stay theirs, every time.\n"
              "Thanks!")
    assert ph.sentences(prompt) == [
        "the player takes over a car an AI drives.",
        "A seat someone already occupies must stay theirs, every time.",
    ]


def test_the_number_and_length_of_queries_are_capped():
    prompt = " ".join(f"Sentence number {i} has five words." for i in range(20))
    assert len(ph.sentences(prompt)) == ph.MAX_SENTENCES
    long = "word " * 500 + "."
    assert len(ph.sentences(long)[0]) == ph.MAX_QUERY_CHARS


def test_fusion_takes_each_sentence_best_hit_first_without_duplicates():
    a = [_hit("a.py", 1), _hit("shared.py", 5), _hit("a2.py", 1)]
    b = [_hit("shared.py", 5), _hit("b.py", 1)]
    fused = ph.fuse([a, b])
    assert [h["rel_path"] for h in fused] == ["a.py", "shared.py", "b.py", "a2.py"]


def test_fusion_drops_agent_rules_files_and_stops_at_the_limit():
    hits = [_hit("CLAUDE.md", 1), _hit("docs/AGENTS.md", 1)] + [_hit(f"m{i}.py", 1) for i in range(30)]
    fused = ph.fuse([hits])
    assert len(fused) == ph.MAX_HITS
    assert not any(h["rel_path"].endswith(".md") for h in fused)


def test_the_answer_is_hook_json_with_one_line_per_place():
    out = json.loads(ph.build("Where does the player enter a car seat?",
                              lambda q: [_hit("src/car.py", 40, "def enter(seat):")],
                              lambda h: h["content"].splitlines()[0]))
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "UserPromptSubmit"
    lines = spec["additionalContext"].splitlines()
    assert lines[0] == ph.HEADER
    assert lines[1] == "- src/car.py L40-49  def enter(seat):"


def test_nothing_to_map_adds_nothing():
    assert ph.build("yes", lambda q: [_hit("x.py", 1)], lambda h: "") == "{}"
    assert ph.build("Where does the player enter a car?", lambda q: [], lambda h: "") == "{}"


# ---------------------------------------------------------------------------
# the registered tool
# ---------------------------------------------------------------------------

class _Backend:
    type_name = "codebase"
    graph = None
    source_config = {"path": "/tmp/demo"}


class _Manager:
    def __init__(self, search):
        self.backends = {"demo": _Backend()}
        self._search = search

    def search(self, source, query, top_k=5, **kw):
        return self._search(query)

    def search_all(self, query, top_k=5, **kw):  # pragma: no cover - one source here
        return self._search(query)


def _call(manager, prompt):
    mcp = FastMCP("t")
    _register_search_tools(mcp, manager)
    content = asyncio.run(mcp.call_tool("prompt_hook", {"prompt": prompt}))
    blocks = content[0] if isinstance(content, tuple) else content
    return blocks[0].text


def test_the_tool_maps_the_prompt_through_the_index():
    seen = []

    def search(query):
        seen.append(query)
        return [_hit("src/ts/Character.ts", 596, "public findVehicleToEnter(wantsToDrive: boolean): void")]

    out = json.loads(_call(_Manager(search), "Fix it. A seat someone occupies must stay theirs."))
    assert seen == ["A seat someone occupies must stay theirs."]
    assert "src/ts/Character.ts L596-605  public findVehicleToEnter" in \
        out["hookSpecificOutput"]["additionalContext"]


def test_an_index_still_loading_adds_nothing_instead_of_an_error():
    def search(query):
        raise RuntimeError("Lynx is still opening its indexes. Retry this call in a few seconds.")

    assert _call(_Manager(search), "Where does the player enter a car seat?") == "{}"


def test_search_folds_guessed_identifiers_into_the_query():
    seen = []

    class _M(_Manager):
        def search(self, source, query, top_k=5, **kw):
            seen.append(query)
            return []

        def search_all(self, query, top_k=5, **kw):
            seen.append(query)
            return []

        def lexical_notice(self, names=None):
            return ""

    class _Cfg:
        class search:
            default_top_k = 5

    m = _M(lambda q: [])
    m.config = _Cfg()
    mcp = FastMCP("t")
    _register_search_tools(mcp, m)
    asyncio.run(mcp.call_tool("search", {"query": "where a click picks a person", "identifiers": "pickActor, labelHover"}))
    assert seen == ["where a click picks a person pickActor  labelHover"]
