"""The retrieval / navigation subcommands — `lynx find-definition` and the
nine others that used to exist only as MCP tools.

Before these, an agent could ask where a symbol is defined, who calls it,
what tests cover it and what breaks if it changes; a human at a terminal,
or a CI script, could not. Ten of the seventeen tools had no CLI at all.

What these tests actually guard:

  - **Names map 1:1 with the MCP tools.** The list `cli.py` routes on is
    duplicated (importing `query_cli` at parse time would cost every
    `lynx serve` the formatting layer), so the parity assertion is what
    keeps the duplication honest.
  - **The rendering is shared, not reimplemented.** Each text assertion
    compares the command's stdout against the very `_format_*` function
    the MCP tool calls. A second renderer that drifts is the failure mode
    this file exists to prevent — `lynx search` had exactly that, printing
    strictly less than the model received.
  - **`--json` holds the contract the rest of the CLI now holds**: exactly
    one object on stdout in every case, carrying ok/operation/source.

Everything runs against a fake SourceManager: these commands are wiring,
and a real manager would drag in the embedding model for questions that
never touch the vector index.
"""
from __future__ import annotations

import json

import pytest

from lynx.query_cli import resolve_backend


class _Backend:
    def __init__(self, type_name="codebase", git=True):
        self.type_name = type_name
        self.source_config = {"git_integration": {"enabled": git}}


class _FakeManager:
    """Records every call; returns whatever the test handed it."""

    def __init__(self, backends=None, **returns):
        self.backends = backends if backends is not None else {"demo": _Backend()}
        self.calls = []
        self._returns = returns

    def _record(self, name, *a, **kw):
        self.calls.append((name, a, kw))
        if name in self._returns and isinstance(self._returns[name], Exception):
            raise self._returns[name]
        return self._returns.get(name, [])

    # --- the ten methods the commands reach for -----------------------
    def find_definition(self, *a, **kw): return self._record("find_definition", *a, **kw)
    def find_usages(self, *a, **kw): return self._record("find_usages", *a, **kw)
    def find_tests_for(self, *a, **kw): return self._record("find_tests_for", *a, **kw)
    def find_similar(self, *a, **kw): return self._record("find_similar", *a, **kw)
    def describe_symbol(self, *a, **kw): return self._record("describe_symbol", *a, **kw) or {}
    def impact_of(self, *a, **kw): return self._record("impact_of", *a, **kw) or {}
    def repo_overview(self, *a, **kw): return self._record("repo_overview", *a, **kw) or {}
    def module_summary(self, *a, **kw): return self._record("module_summary", *a, **kw) or {}
    def search_diff(self, *a, **kw): return self._record("search_diff", *a, **kw) or {}
    def deep_search(self, *a, **kw): return self._record("deep_search", *a, **kw) or _DEEP
    def deep_search_all(self, *a, **kw): return self._record("deep_search_all", *a, **kw) or _DEEP
    def search(self, *a, **kw): return self._record("search", *a, **kw)
    def search_all(self, *a, **kw): return self._record("search_all", *a, **kw)
    def get(self, name): return self.backends[name]


_DEEP = {"results": [], "variants_tried": 1, "winning_variant_index": 0,
         "all_weak": False}


class _Config:
    """Just enough config for the commands that read one."""
    class _Search:
        default_top_k = 5
    search = _Search()
    sources = {"demo": {}, "other": {}}


@pytest.fixture
def cli(monkeypatch):
    """`lynx.cli` with the manager replaced by a recording fake."""
    import lynx.cli as cli_mod

    mgr = _FakeManager()
    monkeypatch.setattr(cli_mod, "_build_manager",
                        lambda config_path: (_Config(), mgr))
    cli_mod.manager = mgr  # handle for the tests
    return cli_mod


def _mgr(cli):
    return cli.manager


# ----------------------------------------------------------------------
# The duplication that makes the routing cheap
# ----------------------------------------------------------------------


def test_cli_command_list_matches_query_cli():
    """`cli._QUERY_COMMANDS` is a copy of `query_cli.COMMANDS`, kept out of
    an import so `lynx serve` doesn't pay for the formatting layer. This is
    what stops the copy from rotting."""
    from lynx.cli import _QUERY_COMMANDS
    from lynx.query_cli import COMMANDS

    assert list(_QUERY_COMMANDS) == list(COMMANDS)


def test_every_command_is_routable():
    """A name in the list but missing from the parser would only fail when
    a user typed it."""
    from lynx.cli import _DISPATCH, _build_parser
    from lynx.query_cli import COMMANDS

    parser = _build_parser()
    registered = set()
    for action in parser._subparsers._group_actions:
        registered |= set(action.choices)

    for name in COMMANDS:
        assert name in registered, f"{name} has no parser"
        assert name in _DISPATCH, f"{name} has no dispatch entry"


def test_names_map_onto_the_mcp_tools():
    """`find_definition` must be `find-definition`, not some third
    spelling — the tool table doubles as the CLI reference."""
    from lynx.query_cli import COMMANDS

    mcp_names = {
        "deep_search", "find_definition", "find_usages", "find_tests_for",
        "find_similar", "describe_symbol", "impact", "repo_overview",
        "module_summary", "search_diff",
    }
    assert {c.replace("-", "_") for c in COMMANDS} == mcp_names


# ----------------------------------------------------------------------
# Each command reaches the right method with the flags it was given
# ----------------------------------------------------------------------


@pytest.mark.parametrize("argv,method,expected_kw", [
    (["find-definition", "helper", "--limit", "3"],
     "find_definition", {"limit": 3}),
    (["find-usages", "helper", "--limit", "7"],
     "find_usages", {"limit": 7}),
    (["find-tests-for", "helper", "--limit", "4",
      "--test-path-pattern", "/spec/"],
     "find_tests_for", {"limit": 4, "test_path_pattern": "/spec/"}),
    (["describe-symbol", "helper", "--callers-limit", "2",
      "--callees-limit", "3", "--tests-limit", "4"],
     "describe_symbol", {"callers_limit": 2, "callees_limit": 3,
                         "tests_limit": 4}),
    (["impact", "helper", "--max-depth", "5", "--tests-limit", "6"],
     "impact_of", {"max_depth": 5, "tests_limit": 6}),
    (["module-summary", "main.py", "--limit", "9"],
     "module_summary", {"limit": 9}),
    (["search-diff", "damage", "--base", "develop", "--top-k", "3"],
     "search_diff", {"base": "develop", "top_k": 3}),
])
def test_flags_reach_the_manager(cli, argv, method, expected_kw):
    rc = cli.main([*argv, "--source", "demo"])

    assert rc == 0
    name, args, kwargs = _mgr(cli).calls[0]
    assert name == method
    assert args[0] == "demo"        # the resolved source comes first
    for k, v in expected_kw.items():
        assert kwargs[k] == v, f"{k} was {kwargs.get(k)!r}, expected {v!r}"


def test_find_similar_takes_a_snippet(cli):
    cli.main(["find-similar", "def helper(x): return x", "--top-k", "3",
              "--source", "demo"])

    name, args, kwargs = _mgr(cli).calls[0]
    assert name == "find_similar"
    assert args[1] == "def helper(x): return x"
    assert kwargs["top_k"] == 3


def test_find_similar_reads_a_file(cli, tmp_path):
    """Quoting a multi-line snippet on a shell command line is miserable;
    --file is the terminal-native way in."""
    snippet = tmp_path / "s.py"
    snippet.write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")

    cli.main(["find-similar", "--file", str(snippet), "--source", "demo"])

    _name, args, _kw = _mgr(cli).calls[0]
    assert "return x + 1" in args[1]


def test_find_similar_without_a_snippet_is_an_error(cli, capsys):
    rc = cli.main(["find-similar", "--source", "demo"])
    assert rc == 1
    assert "empty snippet" in capsys.readouterr().err


def test_find_similar_reads_a_file_with_a_bom(cli, tmp_path):
    snippet = tmp_path / "s.py"
    snippet.write_text("﻿def helper(x):\n    return x\n", encoding="utf-8")

    cli.main(["find-similar", "--file", str(snippet), "--source", "demo"])

    _name, args, _kw = _mgr(cli).calls[0]
    assert args[1].startswith("def helper")


def test_repo_overview_needs_no_arguments(cli):
    assert cli.main(["repo-overview", "--source", "demo"]) == 0
    assert _mgr(cli).calls[0][0] == "repo_overview"


# ----------------------------------------------------------------------
# The text is the MCP tool's text, not a second rendering
# ----------------------------------------------------------------------


def test_find_definition_text_is_the_shared_renderer(cli, capsys, monkeypatch):
    from lynx._format import _format_definition_results

    results = [{"symbol": "helper", "kind": "function", "file": "util.py",
                "start_line": 1, "end_line": 2, "source": "graph"}]
    monkeypatch.setattr(_mgr(cli), "_returns", {"find_definition": results})

    cli.main(["find-definition", "helper", "--source", "demo"])

    assert (capsys.readouterr().out.strip()
            == _format_definition_results("helper", results).strip())


def test_search_diff_text_is_the_shared_renderer(cli, capsys, monkeypatch):
    """This renderer lived inline in the MCP tool until the CLI needed it;
    the extraction is only worth anything if both really use it."""
    from lynx._format import _format_search_diff

    out = {"base": "main", "modified_files": ["a.py"],
           "hits": [{"file_path": "a.py", "start_line": 1, "end_line": 2,
                     "score": 0.5, "symbol_name": "helper"}]}
    monkeypatch.setattr(_mgr(cli), "_returns", {"search_diff": out})

    cli.main(["search-diff", "damage", "--source", "demo"])

    assert (capsys.readouterr().out.strip()
            == _format_search_diff("demo", out).strip())


def test_empty_results_render_the_tool_s_own_wording(cli, capsys):
    cli.main(["find-definition", "nothing", "--source", "demo"])
    assert "No definition found for 'nothing'." in capsys.readouterr().out


# ----------------------------------------------------------------------
# --json
# ----------------------------------------------------------------------


@pytest.mark.parametrize("argv,operation", [
    (["find-definition", "helper"], "find_definition"),
    (["find-usages", "helper"], "find_usages"),
    (["find-tests-for", "helper"], "find_tests_for"),
    (["find-similar", "code"], "find_similar"),
    (["describe-symbol", "helper"], "describe_symbol"),
    (["impact", "helper"], "impact"),
    (["repo-overview"], "repo_overview"),
    (["module-summary", "main.py"], "module_summary"),
    (["search-diff", "damage"], "search_diff"),
    (["deep-search", "a", "b"], "deep_search"),
])
def test_json_payloads_carry_the_contract(cli, capsys, argv, operation):
    rc = cli.main([*argv, "--source", "demo", "--json"])

    payload = json.loads(capsys.readouterr().out)  # stdout must be pure JSON
    assert rc == 0
    assert payload["ok"] is True
    assert payload["operation"] == operation
    assert payload["source"] == "demo"


def test_json_carries_the_results(cli, capsys, monkeypatch):
    results = [{"file": "util.py", "symbol": "helper"}]
    monkeypatch.setattr(_mgr(cli), "_returns", {"find_usages": results})

    cli.main(["find-usages", "helper", "--source", "demo", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1 and payload["results"] == results


# ----------------------------------------------------------------------
# Failures still produce one object
# ----------------------------------------------------------------------


def test_unknown_source_is_reported_as_json(cli, capsys):
    rc = cli.main(["find-definition", "helper", "--source", "ghost", "--json"])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["source"] is None
    assert "unknown source" in payload["error"]


def test_unknown_source_in_text_mode_goes_to_stderr(cli, capsys):
    rc = cli.main(["find-definition", "helper", "--source", "ghost"])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == "" and "unknown source" in captured.err


def test_non_codebase_source_is_refused(cli, capsys):
    _mgr(cli).backends = {"docs": _Backend(type_name="webdoc")}

    rc = cli.main(["find-definition", "helper", "--json"])

    assert rc == 2
    assert "codebase source" in json.loads(capsys.readouterr().out)["error"]


def test_search_diff_refuses_a_source_without_git(cli, capsys):
    _mgr(cli).backends = {"demo": _Backend(git=False)}

    rc = cli.main(["search-diff", "damage", "--json"])

    assert rc == 2
    assert "git-enabled" in json.loads(capsys.readouterr().out)["error"]


def test_an_unexpected_error_is_still_one_object(cli, capsys, monkeypatch):
    monkeypatch.setattr(_mgr(cli), "_returns",
                        {"impact_of": RuntimeError("graph half-written")})

    rc = cli.main(["impact", "helper", "--source", "demo", "--json"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]
    assert payload["operation"] == "impact" and payload["source"] == "demo"


def test_an_unexpected_error_in_text_mode_is_one_line(cli, capsys, monkeypatch):
    monkeypatch.setattr(_mgr(cli), "_returns",
                        {"impact_of": RuntimeError("graph half-written")})

    rc = cli.main(["impact", "helper", "--source", "demo"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "RuntimeError" in captured.err and "Traceback" not in captured.err


# ----------------------------------------------------------------------
# resolve_backend
# ----------------------------------------------------------------------


def test_resolve_backend_defaults_to_the_only_candidate():
    mgr = _FakeManager()
    from lynx.query_cli import _is_codebase

    assert resolve_backend(mgr, None, _is_codebase, "codebase source") == ("demo", None)


def test_resolve_backend_refuses_to_guess_between_many():
    from lynx.query_cli import _is_codebase

    mgr = _FakeManager(backends={"a": _Backend(), "b": _Backend()})
    name, err = resolve_backend(mgr, None, _is_codebase, "codebase source")

    assert name is None and "specify --source" in err


def test_resolve_backend_reports_when_none_qualify():
    from lynx.query_cli import _has_git

    mgr = _FakeManager(backends={"a": _Backend(git=False)})
    name, err = resolve_backend(mgr, None, _has_git, "git-enabled codebase source")

    assert name is None and "no git-enabled codebase source" in err


# ----------------------------------------------------------------------
# deep-search
# ----------------------------------------------------------------------


def test_deep_search_fuses_every_source_by_default(cli):
    cli.main(["deep-search", "a", "b"])

    name, _args, kwargs = _mgr(cli).calls[0]
    assert name == "deep_search_all"
    assert kwargs["queries"] == ["a", "b"]
    assert kwargs["only"] is None


def test_deep_search_targets_one_source(cli):
    cli.main(["deep-search", "a", "--source", "demo", "--mode", "dense"])

    name, args, kwargs = _mgr(cli).calls[0]
    assert name == "deep_search" and args[0] == "demo"
    assert kwargs["mode"] == "dense"


def test_deep_search_scopes_a_subset(cli):
    _mgr(cli).backends = {"demo": _Backend(), "other": _Backend()}

    cli.main(["deep-search", "a", "-s", "demo", "-s", "other"])

    name, _args, kwargs = _mgr(cli).calls[0]
    assert name == "deep_search_all"
    assert kwargs["only"] == ["demo", "other"]


def test_deep_search_rejects_an_unknown_source(cli, capsys):
    rc = cli.main(["deep-search", "a", "-s", "ghost", "-s", "demo", "--json"])

    assert rc == 2
    assert "ghost" in json.loads(capsys.readouterr().out)["error"]


# ----------------------------------------------------------------------
# search: the additions that closed the last gaps with the MCP tool
# ----------------------------------------------------------------------


def test_search_scopes_a_subset_of_sources(cli):
    _mgr(cli).backends = {"demo": _Backend(), "other": _Backend()}

    cli.main(["search", "q", "-s", "demo", "-s", "other"])

    name, _args, kwargs = _mgr(cli).calls[0]
    assert name == "search_all"
    assert kwargs["only"] == ["demo", "other"]


def test_search_rejects_an_unknown_source_in_a_subset(cli, capsys):
    rc = cli.main(["search", "q", "-s", "demo", "-s", "ghost"])

    assert rc == 2
    assert "ghost" in capsys.readouterr().err


def test_search_all_still_works(cli):
    cli.main(["search", "q", "--source", "ALL"])
    assert _mgr(cli).calls[0][0] == "search_all"


def test_search_outline_uses_the_outline_renderer(cli, capsys, monkeypatch):
    from lynx._format import _build_filter_suffix, _format_outline_results

    results = [{"file": "util.py", "content": "def helper(x):\n    return x",
                "symbol_name": "helper", "start_line": 1, "end_line": 2,
                "score": 0.5}]
    monkeypatch.setattr(_mgr(cli), "_returns", {"search": results})

    cli.main(["search", "q", "--source", "demo", "--outline"])

    expected = _format_outline_results(
        "q", results, "source 'demo'", _build_filter_suffix(None, None, None))
    assert capsys.readouterr().out.strip() == expected.strip()


def test_search_json_is_pure_json(cli, capsys, monkeypatch):
    results = [{"file": "util.py", "score": 0.5}]
    monkeypatch.setattr(_mgr(cli), "_returns", {"search": results})

    rc = cli.main(["search", "q", "--source", "demo", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0 and payload["ok"] is True
    assert payload["operation"] == "search" and payload["results"] == results
