"""Unit tests for the find_definition / find_usages / find_tests_for /
find_similar combined tools on CodebaseBackend.

We stub both `CodebaseRAG.__init__` (to skip HF embedding load) and
`CodebaseRAG.search` (to return deterministic results). We DON'T stub
the graph layer — we build a small real GraphLayer over a synthetic
codebase to exercise the graph-on path.

Scenarios:
  1. find_definition graph-on: returns graph result with kind=function
  2. find_definition graph-off: falls back to search, returns same file
  3. find_definition graph-on but symbol not in graph: still searches
  4. find_usages graph-on: get_callers + textual search merged, deduped
  5. find_usages graph-off: search-only fallback works
  6. find_tests_for: matches /tests/ dir and _test.py files
  7. find_tests_for: custom test_path_pattern overrides default
  8. find_similar: dense mode forced; identical chunk filtered out
  9. find_similar: empty / whitespace snippet returns []
 10. describe_symbol graph-on: definition + callers + tests composed in one call
 11. describe_symbol formatter renders every section
 12. describe_symbol graph-off: no call data, definition + tests still work
 13. impact_of graph-on: transitive callers + tests (blast radius)
 14. module_summary graph-on: defined symbols + imports for a file
 15. repo_overview: language + file-count detection from a filesystem scan
 16. export_graph: self-contained symbol view rendered from the real graph
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _stub_rag(search_returns=None):
    """Replace CodebaseRAG.__init__ + .search with deterministic stubs.

    `search_returns` is the list of dicts the stubbed `search()` will
    return regardless of query/filters. Tests set this per-scenario.
    """
    from lynx.rag_manager import CodebaseRAG

    class _StubColl:
        def count(self): return 0

    class _StubVS:
        def __init__(self): self._collection = _StubColl()

    state = {"search_returns": list(search_returns or [])}

    def stub_init(self, **kw):
        self.codebase_path = Path(kw["codebase_path"])
        self.storage_path = Path(kw["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVS()
        # Search mode the stub honors (find_similar swaps it temporarily).
        self.search_mode = kw.get("search_mode", "hybrid")

    def stub_search(self, query, top_k=5, **kw):
        return list(state["search_returns"])  # copy so callers can mutate

    def stub_search_once(self, query, top_k, mode, **kw):
        # find_similar bypasses .search() and goes straight to
        # _search_once so it can pass mode="dense" without mutating
        # instance state. We mirror that here so the stub still works.
        return list(state["search_returns"])

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.search = stub_search
    CodebaseRAG._search_once = stub_search_once
    CodebaseRAG.update = lambda self, force=False: None
    CodebaseRAG.update_file = lambda self, p: None
    CodebaseRAG.remove_file = lambda self, p: None
    CodebaseRAG.check_config_drift = lambda self: None
    return state  # return dict so tests can mutate `search_returns`


def _shared(reranker_enabled=False):
    return SimpleNamespace(
        embedding=SimpleNamespace(model_name="stub"),
        search=SimpleNamespace(
            mode="hybrid", rrf_k=60, candidate_pool_size=30,
            deep=SimpleNamespace(score_thresholds={"hybrid": 0.012}),
            reranker=SimpleNamespace(enabled=reranker_enabled, model_name="", top_n_before_rerank=30),
        ),
    )


def _make_backend(tmp, name="proj", graph_enabled=True):
    """Build a CodebaseBackend over a tiny synthetic codebase.

    The codebase contains:
      - util.py: def helper(x): return x+1
      - service.py: from util import helper; class Service: def run(self): return helper(1)
      - tests/test_service.py: def test_service(): assert Service().run() == 2
    """
    code = tmp / "code"
    code.mkdir(parents=True)
    (code / "util.py").write_text("def helper(x):\n    return x + 1\n")
    (code / "service.py").write_text(
        "from util import helper\n"
        "class Service:\n"
        "    def run(self):\n"
        "        return helper(1)\n"
    )
    (code / "tests").mkdir()
    (code / "tests" / "test_service.py").write_text(
        "from service import Service\n"
        "def test_service():\n"
        "    s = Service()\n"
        "    assert s.run() == 2\n"
    )

    from lynx.sources.codebase import CodebaseBackend
    cfg = {
        "type": "codebase",
        "path": code,
        "supported_extensions": frozenset({".py"}),
        "ignored_path_fragments": [],
        "watcher": {"enabled": False},
        "git_integration": {"enabled": False},
        "graph": {"enabled": graph_enabled},
    }
    return CodebaseBackend(
        name=name, source_config=cfg, shared_config=_shared(),
        storage_dir=tmp / f"storage_{name}",
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-combined-"))
    print(f"[test] tempdir: {tmp}")
    try:
        rag_state = _stub_rag()

        # ============================================================
        # 1. find_definition graph-on
        # ============================================================
        b = _make_backend(tmp / "t1", name="t1", graph_enabled=True)
        out = b.find_definition("helper")
        if not out:
            print(f"[test] FAIL [1/9]: find_definition('helper') returned empty (graph on)")
            return 1
        if not any(r["source"] == "graph" and r.get("file", "").endswith("util.py")
                   and r.get("kind") == "function" for r in out):
            print(f"[test] FAIL [1/9]: no graph match for helper in util.py: {out}")
            return 1
        print(f"[test] OK [1/9] find_definition graph-on: helper → util.py via graph")

        # ============================================================
        # 2. find_definition graph-off → fallback to search
        # ============================================================
        b2 = _make_backend(tmp / "t2", name="t2", graph_enabled=False)
        # Make search return a synthetic hit pointing at util.py
        rag_state["search_returns"] = [{
            "id": "x", "symbol_name": "helper", "symbol_kind": "function",
            "file": "util.py", "file_path": str((tmp / "t2" / "code" / "util.py")),
            "start_line": 1, "end_line": 2, "score": 0.5,
            "language": "python", "content": "def helper(x): return x+1",
        }]
        out = b2.find_definition("helper")
        if not out:
            print(f"[test] FAIL [2/9]: graph-off fallback returned empty")
            return 2
        if out[0]["source"] != "search_bm25":
            print(f"[test] FAIL [2/9]: expected source=search_bm25, got {out[0]}")
            return 2
        if not (out[0]["file"] or "").endswith("util.py"):
            print(f"[test] FAIL [2/9]: file mismatch: {out[0]}")
            return 2
        print(f"[test] OK [2/9] find_definition graph-off: fallback to search_bm25")

        # ============================================================
        # 3. find_definition graph-on but symbol not in graph
        # ============================================================
        # (still uses b from t1 with graph on; "doesnotexist" not in graph)
        rag_state["search_returns"] = []
        out_empty = b.find_definition("doesnotexist")
        if out_empty != []:
            print(f"[test] FAIL [3/9]: expected [] for unknown symbol, got {out_empty}")
            return 3
        # With fallback search returning something, should pick it up
        rag_state["search_returns"] = [{
            "id": "y", "symbol_name": "doesnotexist", "symbol_kind": "function",
            "file": "elsewhere.py", "file_path": "/elsewhere.py",
            "start_line": 1, "end_line": 1, "score": 0.3,
            "language": "python", "content": "...",
        }]
        out2 = b.find_definition("doesnotexist")
        if not out2 or out2[0]["source"] != "search_bm25":
            print(f"[test] FAIL [3/9]: fallback didn't activate for not-in-graph symbol")
            return 3
        print(f"[test] OK [3/9] find_definition graph-on but missing → fallback")

        # ============================================================
        # 4. find_usages graph-on: graph callers + textual search
        # ============================================================
        # Symbol "helper" — graph should say Service.run is a caller.
        # Set the search stub to return Service.run too, so we test dedupe.
        srv_path = str(tmp / "t1" / "code" / "service.py")
        rag_state["search_returns"] = [
            {
                "id": "c1", "symbol_name": "Service.run", "symbol_kind": "function_definition",
                "file": "service.py", "file_path": srv_path,
                "start_line": 3, "end_line": 4, "score": 0.7,
                "language": "python", "content": "def run(self): return helper(1)",
            },
        ]
        out = b.find_usages("helper")
        # Expect at least one entry from graph (Service.run as caller)
        if not any(r["source"] == "graph" for r in out):
            print(f"[test] FAIL [4/9]: no graph caller found for helper: {out}")
            return 4
        # Dedupe: shouldn't have TWO entries for the same (file, line) pair
        keys = [(r["file"], r["start_line"]) for r in out]
        if len(keys) != len(set(keys)):
            print(f"[test] FAIL [4/9]: duplicates in find_usages output: {keys}")
            return 4
        print(f"[test] OK [4/9] find_usages graph-on: caller + dedupe vs search")

        # ============================================================
        # 5. find_usages graph-off: search-only fallback
        # ============================================================
        rag_state["search_returns"] = [
            {
                "id": "u1", "symbol_name": "Service.run", "symbol_kind": "function_definition",
                "file": "service.py", "file_path": "/x/service.py",
                "start_line": 3, "end_line": 4, "score": 0.7,
                "language": "python", "content": "uses helper here",
            },
        ]
        out = b2.find_usages("helper")
        if not out:
            print(f"[test] FAIL [5/9]: graph-off find_usages returned empty")
            return 5
        if not all(r["source"] == "search" for r in out):
            print(f"[test] FAIL [5/9]: expected all source=search, got {out}")
            return 5
        print(f"[test] OK [5/9] find_usages graph-off: search-only fallback works")

        # ============================================================
        # 6. find_tests_for: matches /tests/ dir
        # ============================================================
        rag_state["search_returns"] = [
            {
                "id": "tA", "symbol_name": "test_service", "symbol_kind": "function_definition",
                "file": "test_service.py", "file_path": "/proj/tests/test_service.py",
                "start_line": 1, "end_line": 5, "score": 0.6,
                "content": "def test_service(): ...",
            },
            {
                "id": "tB", "symbol_name": "Service.run", "symbol_kind": "function_definition",
                "file": "service.py", "file_path": "/proj/src/service.py",
                "start_line": 1, "end_line": 5, "score": 0.55,
                "content": "def run(self): ...",
            },
            {
                "id": "tC", "symbol_name": "spec_helper", "symbol_kind": "function_definition",
                "file": "x.spec.js", "file_path": "/proj/x.spec.js",
                "start_line": 1, "end_line": 5, "score": 0.5,
                "content": "describe('helper', ...);",
            },
        ]
        out = b.find_tests_for("anysymbol")
        files = [r["file"] for r in out]
        # tA matches /tests/ → IN; tB does NOT → OUT; tC matches .spec. → IN
        if not any("test_service.py" in f for f in files):
            print(f"[test] FAIL [6/9]: /tests/ path didn't match")
            return 6
        if any("/src/service.py" in f for f in files):
            print(f"[test] FAIL [6/9]: non-test file leaked through")
            return 6
        if not any("x.spec.js" in f for f in files):
            print(f"[test] FAIL [6/9]: .spec. file didn't match")
            return 6
        print(f"[test] OK [6/9] find_tests_for: /tests/ + .spec. matched, non-tests excluded")

        # ============================================================
        # 7. find_tests_for: custom pattern overrides default
        # ============================================================
        out = b.find_tests_for("anysymbol", test_path_pattern=r"\.spec\.js$")
        files = [r["file"] for r in out]
        if not any("x.spec.js" in f for f in files):
            print(f"[test] FAIL [7/9]: custom pattern didn't match")
            return 7
        if any("test_service.py" in f for f in files):
            print(f"[test] FAIL [7/9]: default-only file leaked with custom pattern: {files}")
            return 7
        print(f"[test] OK [7/9] find_tests_for: custom pattern overrides default")

        # ============================================================
        # 8. find_similar: dense mode forced; identical chunk filtered
        # ============================================================
        rag_state["search_returns"] = [
            {
                "id": "s1", "symbol_name": "exact", "symbol_kind": "function_definition",
                "file": "a.py", "file_path": "/x/a.py", "start_line": 1, "end_line": 1,
                "score": 0.95, "language": "python",
                "content": "def helper(x): return x+1",  # identical to snippet
            },
            {
                "id": "s2", "symbol_name": "other", "symbol_kind": "function_definition",
                "file": "b.py", "file_path": "/x/b.py", "start_line": 1, "end_line": 1,
                "score": 0.85, "language": "python",
                "content": "def computeIncrement(n): return n+1",
            },
        ]
        snippet = "def helper(x): return x+1"
        out = b.find_similar(snippet, top_k=5)
        ids = [r["symbol"] for r in out]
        if "exact" in ids:
            print(f"[test] FAIL [8/9]: identical chunk should be filtered: {ids}")
            return 8
        if "other" not in ids:
            print(f"[test] FAIL [8/9]: non-identical similar chunk missing: {ids}")
            return 8
        if not all(r["source"] == "search_dense" for r in out):
            print(f"[test] FAIL [8/9]: source tag wrong: {[r['source'] for r in out]}")
            return 8
        print(f"[test] OK [8/9] find_similar: dense-only, identical filtered")

        # ============================================================
        # 9. find_similar: empty / whitespace snippet → []
        # ============================================================
        if b.find_similar("") != []:
            print(f"[test] FAIL [9/9]: empty snippet should return []")
            return 9
        if b.find_similar("   \n\t  ") != []:
            print(f"[test] FAIL [9/9]: whitespace snippet should return []")
            return 9
        print(f"[test] OK [9/9] find_similar: empty/whitespace returns [] without crashing")

        # ============================================================
        # 10. describe_symbol graph-on: definition + callers + tests in one
        # ============================================================
        # `helper` is defined in util.py and called by Service.run (graph).
        # find_tests_for pulls from the search stub — point it at a /tests/ file.
        rag_state["search_returns"] = [{
            "id": "td", "symbol_name": "test_service", "symbol_kind": "function_definition",
            "file": "test_service.py", "file_path": "/proj/tests/test_service.py",
            "start_line": 1, "end_line": 5, "score": 0.6,
            "content": "def test_service(): ...",
        }]
        d = b.describe_symbol("helper")
        if not d.get("graph_enabled"):
            print(f"[test] FAIL [10/12]: graph_enabled should be True: {d}")
            return 10
        if not any((r.get("file") or "").endswith("util.py") for r in d["definition"]):
            print(f"[test] FAIL [10/12]: definition missing util.py: {d['definition']}")
            return 10
        caller_labels = [(e.get("source") or {}).get("label", "") for e in d["called_by"]]
        if not any(lbl.endswith("run") for lbl in caller_labels):
            print(f"[test] FAIL [10/12]: Service.run not among callers: {caller_labels}")
            return 10
        if not d["tests"]:
            print(f"[test] FAIL [10/12]: tests section empty: {d}")
            return 10
        print(f"[test] OK [10/12] describe_symbol graph-on: definition + callers + tests")

        # ============================================================
        # 11. describe_symbol formatter renders all sections
        # ============================================================
        from lynx.server import _format_describe_symbol
        text = _format_describe_symbol("helper", d)
        for section in ("DEFINITION:", "CALLED BY:", "CALLS:", "TESTS:"):
            if section not in text:
                print(f"[test] FAIL [11/12]: formatter missing {section!r}:\n{text}")
                return 11
        print(f"[test] OK [11/12] describe_symbol formatter renders every section")

        # ============================================================
        # 12. describe_symbol graph-off: no call data, definition+tests still work
        # ============================================================
        rag_state["search_returns"] = [{
            "id": "dd", "symbol_name": "helper", "symbol_kind": "function",
            "file": "util.py", "file_path": str((tmp / "t2" / "code" / "util.py")),
            "start_line": 1, "end_line": 2, "score": 0.5,
            "language": "python", "content": "def helper(x): return x+1",
        }]
        d_off = b2.describe_symbol("helper")
        if d_off.get("graph_enabled"):
            print(f"[test] FAIL [12/12]: graph_enabled should be False on graph-off backend")
            return 12
        if d_off["called_by"] or d_off["calls"]:
            print(f"[test] FAIL [12/12]: call data should be empty without graph: {d_off}")
            return 12
        if not d_off["definition"]:
            print(f"[test] FAIL [12/12]: definition should still resolve via search: {d_off}")
            return 12
        print(f"[test] OK [12/12] describe_symbol graph-off: no calls, definition still works")

        # ============================================================
        # 13. impact_of graph-on: transitive callers + tests
        # ============================================================
        # Service.run calls helper → it's a depth-1 caller of helper.
        rag_state["search_returns"] = [{
            "id": "it", "symbol_name": "test_service", "symbol_kind": "function_definition",
            "file": "test_service.py", "file_path": "/proj/tests/test_service.py",
            "start_line": 1, "end_line": 5, "score": 0.6,
            "content": "def test_service(): ...",
        }]
        imp = b.impact_of("helper", max_depth=3)
        if not imp.get("graph_enabled"):
            print(f"[test] FAIL [13/15]: graph_enabled should be True: {imp}")
            return 13
        caller_labels = [(c.get("node") or {}).get("label", "") for c in imp["callers"]]
        if not any(lbl.endswith("run") for lbl in caller_labels):
            print(f"[test] FAIL [13/15]: Service.run not a transitive caller of helper: {caller_labels}")
            return 13
        if not all(isinstance(c.get("depth"), int) for c in imp["callers"]):
            print(f"[test] FAIL [13/15]: callers missing depth: {imp['callers']}")
            return 13
        if not imp["tests"]:
            print(f"[test] FAIL [13/15]: tests should be populated: {imp}")
            return 13
        print(f"[test] OK [13/15] impact_of graph-on: transitive callers + tests")

        # ============================================================
        # 14. module_summary graph-on: symbols + imports for service.py
        # ============================================================
        ms = b.module_summary("service.py")
        if not ms.get("graph_enabled"):
            print(f"[test] FAIL [14/15]: graph_enabled should be True: {ms}")
            return 14
        sym_labels = [s.get("label", "") for s in ms["symbols"]]
        if not any("Service" in lbl for lbl in sym_labels):
            print(f"[test] FAIL [14/15]: Service symbols missing from service.py summary: {sym_labels}")
            return 14
        if not ms["imports"]:
            print(f"[test] FAIL [14/15]: service.py should import util/helper: {ms}")
            return 14
        print(f"[test] OK [14/15] module_summary graph-on: symbols + imports")

        # ============================================================
        # 15. repo_overview: filesystem scan of the synthetic codebase
        # ============================================================
        ov = b.repo_overview()
        langs = {l["language"] for l in ov.get("languages", [])}
        if "Python" not in langs:
            print(f"[test] FAIL [15/15]: Python not detected in overview: {ov}")
            return 15
        if ov.get("file_count", 0) < 3:
            print(f"[test] FAIL [15/15]: expected >=3 files scanned: {ov}")
            return 15
        print(f"[test] OK [15/15] repo_overview: languages + file_count from fs scan")

        # ============================================================
        # 16. export_graph: self-contained symbol view from the real graph
        # ============================================================
        res = b.export_graph("symbol", "helper", depth=2)
        if res.get("empty"):
            print(f"[test] FAIL [16/16]: export_graph reported empty: {res}")
            return 16
        content = res.get("content") or ""
        if not content.lower().startswith("<!doctype html>"):
            print(f"[test] FAIL [16/16]: output is not a standalone HTML doc")
            return 16
        if "helper" not in content:
            print(f"[test] FAIL [16/16]: seed symbol not present in the view")
            return 16
        if "src=" in content or "<script" in content:
            print(f"[test] FAIL [16/16]: output is not self-contained (external/script refs)")
            return 16
        if not res.get("suggested_name", "").endswith(".html"):
            print(f"[test] FAIL [16/16]: suggested_name missing/!.html: {res.get('suggested_name')}")
            return 16
        print(f"[test] OK [16/16] export_graph: self-contained symbol view rendered")

        print("\n[test] === SUCCESS: combined tools work as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
