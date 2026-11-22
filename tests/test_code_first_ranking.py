"""Code ranks above the tests, data and docs that describe it in the same
words, and a file's path and symbol name count in the lexical search.

Measured on The Lost Storm (lynx-zombie-bench E17, E18): over 76 issue titles
nobody in the bench wrote, the right file first 4 times out of 76 before, 19
with the weight; MRR 0.185 to 0.362. What is asserted here is the mechanism,
not those numbers.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lynx import prompt_hook as ph
from lynx.config import _secondary_weight
from lynx.file_kinds import is_secondary, is_test_path
from lynx.rag_manager import CodebaseRAG, lexical_tokens


@pytest.mark.parametrize("path", [
    "src/input.test.ts", "src/pick.spec.ts", "src/__tests__/pick.ts", "test/support.ts",
    "tests/test_server.py", "pkg/server_test.go", "Assets/Tests/PlayerTests.cs", "test_player.gd",
])
def test_test_files_are_recognised_by_the_usual_conventions(path):
    assert is_test_path(path)
    assert is_secondary(path)


@pytest.mark.parametrize("path", ["data/npcs.json", "docs/content.md", "README.md", "notes.txt"])
def test_data_and_docs_are_secondary(path):
    assert not is_test_path(path)
    assert is_secondary(path)


@pytest.mark.parametrize("path", ["src/input.ts", "sim/bot.ts", "src/latest.ts", "Assets/Scripts/Player.cs", "contest.py"])
def test_code_is_not(path):
    assert not is_secondary(path)


def _rag(weight, root="/repo"):
    return SimpleNamespace(secondary_weight=weight, codebase_path=root)


def _hit(rel, score):
    return {"file_path": f"/repo/{rel}", "score": score, "id": rel}


def test_a_test_that_outscores_the_code_by_less_than_half_ranks_after_it():
    results = [_hit("src/input.test.ts", 0.030), _hit("data/npcs.json", 0.028), _hit("src/input.ts", 0.020)]
    ranked = CodebaseRAG._rank_code_first(_rag(0.5), results)
    assert [r["id"] for r in ranked] == ["src/input.ts", "src/input.test.ts", "data/npcs.json"]
    assert {r["id"] for r in ranked} == {r["id"] for r in results}, "nothing is dropped"


def test_a_test_far_ahead_of_the_code_stays_ahead():
    results = [_hit("src/input.test.ts", 0.050), _hit("src/input.ts", 0.020)]
    ranked = CodebaseRAG._rank_code_first(_rag(0.5), results)
    assert [r["id"] for r in ranked] == ["src/input.test.ts", "src/input.ts"]


def test_weight_one_leaves_the_order_alone():
    results = [_hit("src/input.test.ts", 0.030), _hit("src/input.ts", 0.020)]
    assert CodebaseRAG._rank_code_first(_rag(1.0), results) == results


def test_a_project_under_a_folder_called_test_is_still_code():
    results = [_hit("src/a.ts", 0.020)]
    rag = _rag(0.5, root="/repo")
    results[0]["file_path"] = "/home/me/test/repo/src/a.ts"
    rag.codebase_path = "/home/me/test/repo"
    assert CodebaseRAG._rank_code_first(rag, results)[0]["score"] == 0.020


def test_the_lexical_document_holds_the_file_path_and_the_symbol_name():
    tokens = lexical_tokens("const x = 1;", "C:\\games\\storm\\src\\render\\overlays.ts", "Overlays.update#part2")
    for word in ("render", "overlays", "update", "part2"):
        assert word in tokens
    assert tokens[:3] == ["const", "x", "1"], "the body comes first"


def test_the_prompt_map_leaves_tests_data_and_docs_out():
    hits = [{"rel_path": "src/input.test.ts", "start_line": 1, "end_line": 9, "content": "x"},
            {"rel_path": "data/npcs.json", "start_line": 1, "end_line": 9, "content": "x"},
            {"rel_path": "docs/content.md", "start_line": 1, "end_line": 9, "content": "x"},
            {"rel_path": "src/input.ts", "start_line": 240, "end_line": 260, "content": "x"},
            {"rel_path": "src/render/overlays.ts", "start_line": 500, "end_line": 530, "content": "x"}]
    asked = []

    def search(query):
        asked.append(query)
        return hits

    out = json.loads(ph.build("Every click on the ground opens a dialogue.", search, lambda h: "sig"))
    lines = out["hookSpecificOutput"]["additionalContext"].splitlines()[1:]
    assert [line.split()[1] for line in lines] == ["src/input.ts", "src/render/overlays.ts"]


@pytest.mark.parametrize("value", [0, -0.5, 1.5, "half"])
def test_a_secondary_weight_outside_zero_to_one_is_refused(value):
    with pytest.raises(ValueError):
        _secondary_weight(value)
