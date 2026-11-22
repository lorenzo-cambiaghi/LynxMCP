"""describe_symbol on a name defined in many files: the reader can pick the
definition (`file`), is told the others exist, and gets the callers of the
definition shown, not of every node with that name.

Found on The Lost Storm (2026-11-22): `rig`, a helper defined in 15 test
files, came back as the first three definitions found, none the one asked
about, with no word that others existed, and with callers from other files.
"""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from lynx._format import _format_describe_symbol
from lynx.graph.query import callers_of_node, get_callers
from lynx.sources.codebase import CodebaseBackend


def _graph():
    G = nx.DiGraph()
    files = {"parse": "/repo/sim/parse-rigs.ts", "auction": "/repo/src/auction.test.ts",
             "input": "/repo/src/input.test.ts", "audio": "/repo/src/audio.test.ts",
             "save": "/repo/src/save.test.ts"}
    for key, path in files.items():
        G.add_node(f"rig_{key}", label="rig", kind="function", file=path, start_line=10, end_line=20)
    G.add_node("opened", label="opened", kind="function", file=files["auction"], start_line=30, end_line=35)
    G.add_node("clickOn", label="clickOn", kind="function", file=files["input"], start_line=40, end_line=45)
    G.add_edge("opened", "rig_auction", relation="calls", confidence="extracted")
    G.add_edge("clickOn", "rig_input", relation="calls", confidence="extracted")
    return G


def _backend(G):
    defs = [(nid, {"symbol": "rig", "kind": "function", "file": G.nodes[nid]["file"],
                   "start_line": 10, "end_line": 20, "source": "graph"})
            for nid in ("rig_parse", "rig_auction", "rig_audio", "rig_input", "rig_save")]
    fake = SimpleNamespace(
        graph=SimpleNamespace(graph=G),
        source_config={"path": "/repo"},
        _before_read=lambda: None,
        _definitions=lambda symbol, limit: defs[:limit],
        _with_code=lambda results, files, bodies=1: results,
        find_tests_for=lambda symbol, limit=5: [],
        _attach_rel=lambda result: result,
    )
    fake.get_callers = lambda symbol, limit=10: get_callers(G, symbol, limit=limit)
    fake.get_callees = lambda symbol, limit=10: []
    return fake


def _callers(d):
    return sorted(e["source"]["label"] for e in d["called_by"])


def test_the_callers_are_those_of_the_definition_shown():
    G = _graph()
    assert sorted(e["source"]["label"] for e in get_callers(G, "rig")) == ["clickOn", "opened"], \
        "by name, the callers of every rig are united"
    assert [e["source"]["label"] for e in callers_of_node(G, "rig_input")] == ["clickOn"]
    d = CodebaseBackend.describe_symbol(_backend(G), "rig")
    assert d["definition"][0]["file"].endswith("parse-rigs.ts")
    assert _callers(d) == [], "parse-rigs' rig has no caller; the test helpers' callers are not its"


def test_file_picks_the_definition_and_its_callers():
    d = CodebaseBackend.describe_symbol(_backend(_graph()), "rig", file="input.test")
    assert d["definition"][0]["file"].endswith("src/input.test.ts")
    assert _callers(d) == ["clickOn"]


def test_the_other_definitions_are_named_and_the_formatter_says_how_to_pick():
    d = CodebaseBackend.describe_symbol(_backend(_graph()), "rig")
    assert d["other_definitions"] == 2
    assert d["other_definition_files"] == ["src/input.test.ts", "src/save.test.ts"]
    text = _format_describe_symbol("rig", d)
    assert "2 other definition(s)" in text and "src/input.test.ts" in text and "`file`" in text


def test_a_file_that_matches_nothing_keeps_the_usual_order():
    d = CodebaseBackend.describe_symbol(_backend(_graph()), "rig", file="nowhere")
    assert d["definition"][0]["file"].endswith("parse-rigs.ts")
