"""`lynx graph query` — the knowledge graph from a terminal.

The graph layer was reachable two ways: agents got the full `graph_query`
MCP tool, humans got `lynx graph build|status|export` and no way to *ask*
anything. The dispatch that renders an operation now lives in
`lynx/graph/dispatch.py` and both front-ends call it, so what you read in
the terminal is byte-for-byte what the model reads.

Tests run against a fake SourceManager — the point is the dispatch and the
CLI wiring, and a real manager would drag in the embedding model for
questions that never touch the vector index.

Covered:
  1. cli._GRAPH_OPERATIONS stays in sync with dispatch.OPERATIONS
  2. every operation reaches the matching SourceManager method, with the
     CLI's flags passed through (limit, depth, relation, top_n, ...)
  3. operations that need a symbol say so instead of querying for None
  4. shortest_path reports a missing --target, and renders a found path
  5. an unknown operation lists the valid ones
  6. the CLI exits non-zero when the dispatcher reports an error in-band
"""
from __future__ import annotations

import json

import pytest

from lynx.graph.dispatch import (
    OPERATIONS, SYMBOL_OPS, query_graph, run_graph_query,
)


class _FakeManager:
    """Records calls; returns the shapes SourceManager's graph methods return."""

    def __init__(self, edges=None, path=None, surprises=None):
        self.calls = []
        self._edges = edges if edges is not None else []
        self._path = path
        self._surprises = surprises if surprises is not None else []

    def _record(self, name, **kw):
        self.calls.append((name, kw))
        return self._edges

    def get_callers(self, source, symbol, limit=50):
        return self._record("get_callers", source=source, symbol=symbol, limit=limit)

    def get_callees(self, source, symbol, limit=50):
        return self._record("get_callees", source=source, symbol=symbol, limit=limit)

    def get_subclasses(self, source, symbol, limit=50):
        return self._record("get_subclasses", source=source, symbol=symbol, limit=limit)

    def get_superclasses(self, source, symbol, limit=50):
        return self._record("get_superclasses", source=source, symbol=symbol, limit=limit)

    def get_imports(self, source, symbol, limit=50):
        return self._record("get_imports", source=source, symbol=symbol, limit=limit)

    def get_neighbors(self, source, symbol, relation_filter=None, depth=1, limit=50):
        return self._record("get_neighbors", source=source, symbol=symbol,
                            relation_filter=relation_filter, depth=depth, limit=limit)

    def shortest_path(self, source, symbol, target, max_hops=8):
        self.calls.append(("shortest_path", {"source": source, "symbol": symbol,
                                             "target": target, "max_hops": max_hops}))
        return self._path

    def architectural_overview(self, source, top_n_gods=10, min_community_size=3):
        self.calls.append(("architectural_overview", {
            "source": source, "top_n_gods": top_n_gods,
            "min_community_size": min_community_size}))
        return {
            "status": {"nodes": 5, "edges": 2, "files_indexed": 3,
                       "last_update": "2026-08-03T10:00:00"},
            "god_nodes": [{"label": "hub", "degree": 4, "in_degree": 3,
                           "out_degree": 1, "file": "hub.py"}],
            "communities": [{"id": 0, "name": "core", "size": 3,
                             "by_language": {"python": 3},
                             "members_sample": ["a", "b", "c"]}],
        }

    def surprising_connections(self, source, top_n=10):
        self.calls.append(("surprising_connections", {"source": source, "top_n": top_n}))
        return self._surprises

    def graph_status(self, source):
        self.calls.append(("graph_status", {"source": source}))
        return {
            "schema_version": 2, "nodes": 5, "edges": 2, "files_indexed": 3,
            "raw_calls_pending": 1, "raw_inherits_pending": 0,
            "last_update": "2026-08-03T10:00:00",
            "last_full_rebuild": "2026-08-03T09:59:00",
            "by_language": {"python": 5}, "by_kind": {"function": 2},
            "by_relation": {"calls": 1},
        }


def _edge():
    return {
        "source": {"label": "go", "kind": "function", "file": "main.py",
                   "start_line": 2, "end_line": 3},
        "target": {"label": "helper", "kind": "function", "file": "util.py",
                   "start_line": 1, "end_line": 2},
        "relation": "calls", "confidence": "resolved",
    }


# ----------------------------------------------------------------------


def test_cli_operation_list_matches_dispatch():
    """cli.py hard-codes the list to keep tree-sitter out of `lynx serve`'s
    startup path. This test is what makes that duplication safe."""
    from lynx.cli import _GRAPH_OPERATIONS
    assert list(_GRAPH_OPERATIONS) == list(OPERATIONS)


@pytest.mark.parametrize("op,method", [
    ("callers", "get_callers"),
    ("callees", "get_callees"),
    ("subclasses", "get_subclasses"),
    ("superclasses", "get_superclasses"),
    ("imports", "get_imports"),
])
def test_edge_operations_delegate(op, method):
    mgr = _FakeManager(edges=[_edge()])
    out = run_graph_query(mgr, "demo", op, symbol="helper", limit=7)

    assert mgr.calls == [(method, {"source": "demo", "symbol": "helper", "limit": 7})]
    assert "helper" in out and "go" in out


@pytest.mark.parametrize("op,method", [
    ("callers", "get_callers"),
    ("callees", "get_callees"),
    ("subclasses", "get_subclasses"),
    ("superclasses", "get_superclasses"),
    ("imports", "get_imports"),
])
def test_only_the_needed_method_is_touched(op, method):
    """The five edge operations share one dispatch table. Built as a table
    of BOUND methods it evaluated all five attributes to serve one call, so
    a manager exposing a subset died on an AttributeError for a method the
    request never needed — surfacing as `{"error": "AttributeError: ... has
    no attribute 'get_callees'"}` on a perfectly valid `--op callers`.
    Found while reproducing something else; nothing failed when it broke.

    Each manager here implements exactly one of the five.
    """
    calls = []

    def _only(self, source, symbol, limit=50):
        calls.append(method)
        return []

    mgr = type("_OneMethodManager", (), {method: _only})()

    res = query_graph(mgr, "demo", op, symbol="x")

    assert res.ok is True
    assert calls == [method]


def test_neighbors_passes_relation_and_depth():
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", "neighbors", symbol="helper",
                          relation_filter="calls", depth=3, limit=5)

    name, kw = mgr.calls[0]
    assert name == "get_neighbors"
    assert kw == {"source": "demo", "symbol": "helper",
                  "relation_filter": "calls", "depth": 3, "limit": 5}
    assert "relation='calls'" in out and "depth=3" in out


@pytest.mark.parametrize("op", sorted(SYMBOL_OPS))
def test_symbol_operations_refuse_a_missing_symbol(op):
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", op)

    assert out.startswith("Error:") and "requires `symbol`" in out
    assert mgr.calls == []  # never queried with symbol=None


def test_shortest_path_requires_target():
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", "shortest_path", symbol="go")
    assert out.startswith("Error:") and "requires `target`" in out
    assert mgr.calls == []


def test_shortest_path_renders_nodes():
    mgr = _FakeManager(path={"hops": 1, "nodes": [
        {"label": "go", "kind": "function", "file": "main.py",
         "start_line": 2, "end_line": 3},
        {"label": "helper", "kind": "function", "file": "util.py",
         "start_line": 1, "end_line": 2},
    ]})
    out = run_graph_query(mgr, "demo", "shortest_path", symbol="go",
                          target="helper", max_hops=4)

    assert mgr.calls[0][1]["max_hops"] == 4
    assert "1 hops" in out and "go" in out and "helper" in out


def test_shortest_path_reports_no_route():
    mgr = _FakeManager(path=None)
    out = run_graph_query(mgr, "demo", "shortest_path", symbol="a", target="b")
    assert "No directed path" in out
    assert not out.startswith("Error:")  # a real answer, not a usage problem


def test_overview_passes_tuning_arguments():
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", "overview", top_n=3, min_community_size=2)

    assert mgr.calls[0] == ("architectural_overview", {
        "source": "demo", "top_n_gods": 3, "min_community_size": 2})
    assert "God nodes" in out and "hub" in out and "core" in out


def test_overview_alias_still_dispatches():
    """'architectural_overview' was the pre-consolidation tool name."""
    mgr = _FakeManager()
    run_graph_query(mgr, "demo", "architectural_overview")
    assert mgr.calls[0][0] == "architectural_overview"


def test_surprising_connections_empty_is_not_an_error():
    mgr = _FakeManager(surprises=[])
    out = run_graph_query(mgr, "demo", "surprising_connections")
    assert "No surprising connections" in out
    assert not out.startswith("Error:")


def test_status_renders_counts():
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", "status")
    assert mgr.calls == [("graph_status", {"source": "demo"})]
    assert "Nodes:             5" in out
    assert "Raw inherits pending: 0" in out


def test_unknown_operation_lists_the_valid_ones():
    mgr = _FakeManager()
    out = run_graph_query(mgr, "demo", "nope")
    assert out.startswith("Error: unknown operation")
    for op in OPERATIONS:
        assert op in out


def test_operation_is_case_and_space_insensitive():
    mgr = _FakeManager()
    run_graph_query(mgr, "demo", "  CALLERS ", symbol="x")
    assert mgr.calls[0][0] == "get_callers"


# ----------------------------------------------------------------------
# CLI wiring: flags → dispatch → exit code
# ----------------------------------------------------------------------


class _FakeBackend:
    graph = object()  # non-None is what marks a source as graph-enabled


def _patch_manager(monkeypatch, mgr):
    """Stand in for `_build_manager`, which would load the embedding model."""
    import lynx.cli as cli

    mgr.backends = {"demo": _FakeBackend()}
    monkeypatch.setattr(cli, "_build_manager", lambda config_path: (None, mgr))
    return cli


def test_cli_query_passes_flags_and_exits_zero(monkeypatch, capsys):
    mgr = _FakeManager(edges=[_edge()])
    cli = _patch_manager(monkeypatch, mgr)

    rc = cli.main(["graph", "query", "--op", "neighbors", "--symbol", "helper",
                   "--relation", "calls", "--depth", "2", "--limit", "9"])

    assert rc == 0
    assert mgr.calls[0][1] == {"source": "demo", "symbol": "helper",
                               "relation_filter": "calls", "depth": 2, "limit": 9}
    assert "helper" in capsys.readouterr().out


def test_cli_query_exits_nonzero_on_dispatch_error(monkeypatch, capsys):
    mgr = _FakeManager()
    cli = _patch_manager(monkeypatch, mgr)

    # `callers` without --symbol: the dispatcher reports it in the text it
    # returns, so the exit code has to be derived from that.
    rc = cli.main(["graph", "query", "--op", "callers"])

    assert rc == 1
    assert "requires `symbol`" in capsys.readouterr().out


def test_cli_query_rejects_unknown_operation_at_parse_time(monkeypatch):
    mgr = _FakeManager()
    cli = _patch_manager(monkeypatch, mgr)

    with pytest.raises(SystemExit) as e:
        cli.main(["graph", "query", "--op", "nope"])
    assert e.value.code == 2
    assert mgr.calls == []


# ----------------------------------------------------------------------
# Structured result: the exit code must not come from sniffing the text.
#
# The first cut did `return 1 if out.startswith("Error:") else 0` on a
# string written to be read by a language model — a contract that any
# rewording silently breaks.
# ----------------------------------------------------------------------


def test_usage_problems_are_flagged_not_spelled(monkeypatch):
    res = query_graph(_FakeManager(), "demo", "callers")
    assert res.ok is False
    # Same keys as a success payload: a JSON consumer reads `.operation`
    # and `.source` whichever way the call went.
    assert res.data == {"operation": "callers", "source": "demo",
                        "error": "operation 'callers' requires `symbol`."}


def test_an_empty_answer_is_still_an_answer():
    """"no callers" is a result, not a failure — it must not exit non-zero."""
    res = query_graph(_FakeManager(edges=[]), "demo", "callers", symbol="x")
    assert res.ok is True
    assert res.data["count"] == 0


def test_result_text_matches_the_text_facade():
    """The MCP tool goes through run_graph_query; the CLI through
    query_graph. They must not drift apart."""
    mgr_a, mgr_b = _FakeManager(edges=[_edge()]), _FakeManager(edges=[_edge()])
    assert (query_graph(mgr_a, "demo", "callers", symbol="helper").text
            == run_graph_query(mgr_b, "demo", "callers", symbol="helper"))


@pytest.mark.parametrize("op,key", [
    ("callers", "edges"),
    ("status", "nodes"),
    ("overview", "god_nodes"),
    ("surprising_connections", "connections"),
    ("shortest_path", "path"),
])
def test_payloads_are_json_serializable(op, key):
    res = query_graph(_FakeManager(edges=[_edge()]), "demo", op,
                      symbol="helper", target="other")
    assert key in json.loads(json.dumps(res.data, default=str))


@pytest.mark.parametrize("op", [
    "callers", "neighbors", "shortest_path", "overview",
    "surprising_connections", "status",
])
def test_every_payload_names_its_operation_and_source(op):
    """`status` and `overview` return a dict of their own and used to be
    handed back bare, so a script couldn't read `.operation` uniformly."""
    res = query_graph(_FakeManager(edges=[_edge()]), "demo", op,
                      symbol="helper", target="other")
    assert res.data["operation"] == op
    assert res.data["source"] == "demo"


def test_status_payload_keeps_its_top_level_keys():
    """The envelope is merged, not nested: `jq .nodes` must keep working."""
    res = query_graph(_FakeManager(), "demo", "status")
    assert res.data["nodes"] == 5 and res.data["by_kind"] == {"function": 2}


# ----------------------------------------------------------------------
# "no results" vs "that symbol isn't in the graph"
# ----------------------------------------------------------------------


def _manager_knowing(*labels, edges=None):
    """A fake manager whose graph contains nodes with the given labels."""
    import networkx as nx

    G = nx.DiGraph()
    for i, label in enumerate(labels):
        G.add_node(f"n{i}", label=label, kind="function",
                   file="mod.py", start_line=1, end_line=2)

    mgr = _FakeManager(edges=edges)
    layer = type("_Layer", (), {"graph": G})()
    backend = type("_Backend", (), {"graph": layer})()
    mgr.get = lambda source: backend
    return mgr


def test_unknown_symbol_says_so():
    res = query_graph(_manager_knowing("helper"), "demo", "callers",
                      symbol="ghost")
    assert res.matched is False
    assert "nothing in this graph is called 'ghost'" in res.text
    assert res.ok is True  # a correct answer to a question about nothing


def test_known_symbol_with_no_edges_gets_no_hint():
    res = query_graph(_manager_knowing("helper"), "demo", "callers",
                      symbol="helper")
    assert res.matched is True
    assert "nothing in this graph" not in res.text
    assert "(no results)" in res.text


def test_hint_is_skipped_when_the_graph_cannot_be_reached():
    """A stubbed or half-built manager must not turn a valid empty answer
    into a wrong claim that the symbol doesn't exist."""
    res = query_graph(_FakeManager(edges=[]), "demo", "callers", symbol="x")
    assert res.matched is None
    assert "nothing in this graph" not in res.text


def test_shortest_path_names_the_endpoint_that_is_missing():
    res = query_graph(_manager_knowing("go"), "demo", "shortest_path",
                      symbol="go", target="ghost")
    assert "nothing in this graph is called 'ghost'" in res.text


def test_no_hint_when_results_exist():
    res = query_graph(_manager_knowing("helper", edges=[_edge()]), "demo",
                      "callers", symbol="helper")
    assert "nothing in this graph" not in res.text
    # Results imply the seed resolved: report True rather than a null the
    # JSON consumer would have to special-case.
    assert res.matched is True and res.data["matched"] is True


# ----------------------------------------------------------------------
# --json
# ----------------------------------------------------------------------


def test_cli_json_output_is_pure_json(monkeypatch, capsys):
    mgr = _FakeManager(edges=[_edge()])
    cli = _patch_manager(monkeypatch, mgr)

    rc = cli.main(["graph", "query", "--op", "callers", "--symbol", "helper",
                   "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "callers"
    assert payload["count"] == 1
    assert payload["edges"][0]["relation"] == "calls"


def test_cli_json_output_on_usage_error(monkeypatch, capsys):
    mgr = _FakeManager()
    cli = _patch_manager(monkeypatch, mgr)

    rc = cli.main(["graph", "query", "--op", "callers", "--json"])

    assert rc == 1
    assert "requires" in json.loads(capsys.readouterr().out)["error"]


def test_cli_json_status_payload(monkeypatch, capsys):
    mgr = _FakeManager()
    cli = _patch_manager(monkeypatch, mgr)

    cli.main(["graph", "query", "--op", "status", "--json"])
    assert json.loads(capsys.readouterr().out)["nodes"] == 5


@pytest.mark.parametrize("argv", [
    ["--op", "callers", "--symbol", "helper"],   # a result
    ["--op", "status"],                          # a dict-shaped payload
    ["--op", "callers"],                         # a usage error
    ["--op", "status", "--source", "ghost"],     # a resolver error
])
def test_every_json_object_carries_ok_operation_source(monkeypatch, capsys, argv):
    """The contract a script leans on: `ok`, `operation` and `source` on
    every object, regardless of which operation ran or whether it worked.
    `source` is None exactly when resolving it was the failure."""
    cli = _patch_manager(monkeypatch, _FakeManager(edges=[_edge()]))
    rc = cli.main(["graph", "query", *argv, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is (rc == 0)
    assert payload["operation"] == argv[1]
    assert "source" in payload
    assert (payload["source"] is None) == ("ghost" in argv)


# ----------------------------------------------------------------------
# Source resolution: a helper that called sys.exit left --json with an
# empty stdout and a non-zero code — nothing for a script to parse, and
# the one failure of this command that didn't produce an object.
# ----------------------------------------------------------------------


def test_resolver_returns_errors_instead_of_exiting():
    import argparse

    from lynx.cli import _resolve_graph_source

    mgr = _FakeManager()
    mgr.backends = {"demo": _FakeBackend()}

    name, err = _resolve_graph_source(mgr, argparse.Namespace(source=None))
    assert (name, err) == ("demo", None)

    name, err = _resolve_graph_source(mgr, argparse.Namespace(source="ghost"))
    assert name is None and "unknown source" in err


def test_resolver_reports_when_no_source_has_a_graph():
    import argparse

    from lynx.cli import _resolve_graph_source

    mgr = _FakeManager()
    mgr.backends = {}
    name, err = _resolve_graph_source(mgr, argparse.Namespace(source=None))
    assert name is None and "graph layer" in err


def test_unknown_source_still_produces_json(monkeypatch, capsys):
    cli = _patch_manager(monkeypatch, _FakeManager())

    rc = cli.main(["graph", "query", "--source", "ghost", "--op", "status",
                   "--json"])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)  # an object, not silence
    assert payload["ok"] is False
    assert "unknown source" in payload["error"]
    assert payload["operation"] == "status"
    assert payload["source"] is None  # resolving it WAS the failure


# ----------------------------------------------------------------------
# Unexpected exceptions: the one path where "--json = exactly one object"
# used to be false. The MCP tool wraps the dispatch in try/except; the CLI
# didn't — a manager-level error (graph mid-rebuild, corrupt state) meant
# empty stdout, a traceback, exit 1, nothing to parse. Reproduced live.
# ----------------------------------------------------------------------


class _ExplodingManager(_FakeManager):
    def get_callers(self, *a, **kw):
        raise RuntimeError("graph JSON corrotto a metà rebuild")


def test_unexpected_exception_still_produces_json(monkeypatch, capsys):
    cli = _patch_manager(monkeypatch, _ExplodingManager())

    rc = cli.main(["graph", "query", "--op", "callers", "--symbol", "x",
                   "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]
    assert "corrotto" in payload["error"]
    assert payload["operation"] == "callers"
    assert payload["source"] == "demo"  # resolution had succeeded


def test_unexpected_exception_in_text_mode_goes_to_stderr(monkeypatch, capsys):
    """Text mode: a one-line error on stderr, empty stdout, exit 1 — not a
    traceback dumped at the user."""
    cli = _patch_manager(monkeypatch, _ExplodingManager())

    rc = cli.main(["graph", "query", "--op", "callers", "--symbol", "x"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "RuntimeError" in captured.err


def test_unknown_source_in_text_mode_goes_to_stderr(monkeypatch, capsys):
    cli = _patch_manager(monkeypatch, _FakeManager())

    rc = cli.main(["graph", "query", "--source", "ghost", "--op", "status"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown source" in captured.err
    assert captured.out == ""


def test_source_without_a_graph_layer_is_rejected(monkeypatch, capsys):
    mgr = _FakeManager()
    mgr.backends = {"demo": _FakeBackend(), "plain": type("_B", (), {})()}
    import lynx.cli as cli
    monkeypatch.setattr(cli, "_build_manager", lambda config_path: (None, mgr))

    rc = cli.main(["graph", "query", "--source", "plain", "--op", "status",
                   "--json"])

    assert rc == 2
    assert "no graph layer" in json.loads(capsys.readouterr().out)["error"]


def test_cli_json_survives_a_library_greeting_on_stdout(monkeypatch, capfd):
    """Loading the manager imports llama_index, which writes "LLM is
    explicitly disabled. Using MockLLM." to file descriptor 1 — underneath
    Python's stdout, where `capsys` cannot see it. The first version of
    --json passed its unit tests (fake manager, no greeting) and produced
    unparseable output on a real index. capfd captures at the fd level,
    which is the only way this regression stays caught."""
    import os

    import lynx.cli as cli
    mgr = _FakeManager(edges=[_edge()])
    mgr.backends = {"demo": _FakeBackend()}

    def _noisy_build(config_path):
        os.write(1, b"LLM is explicitly disabled. Using MockLLM.\n")
        return None, mgr

    monkeypatch.setattr(cli, "_build_manager", _noisy_build)

    rc = cli.main(["graph", "query", "--op", "callers", "--symbol", "helper",
                   "--json"])
    out, err = capfd.readouterr()

    assert rc == 0
    assert json.loads(out)["count"] == 1   # stdout parses: nothing else on it
    assert "MockLLM" in err                # the greeting was rerouted, not lost


def test_text_mode_keeps_the_library_output_visible(monkeypatch, capfd):
    """Muting is for --json only — in text mode those lines are ordinary
    progress output and hiding them would be a different bug."""
    import os

    import lynx.cli as cli
    mgr = _FakeManager(edges=[_edge()])
    mgr.backends = {"demo": _FakeBackend()}

    def _noisy_build(config_path):
        os.write(1, b"loading model...\n")
        return None, mgr

    monkeypatch.setattr(cli, "_build_manager", _noisy_build)
    cli.main(["graph", "query", "--op", "callers", "--symbol", "helper"])

    assert "loading model" in capfd.readouterr().out
