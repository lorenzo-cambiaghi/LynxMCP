"""How much code one `search` answer may hold."""
from __future__ import annotations


def _hit(i, size):
    body = "\n".join(f"line {n} " + "x" * 60 for n in range(size // 68 + 1))
    return {"rel_path": f"src/f{i}.py", "symbol_name": f"f{i}", "start_line": 1,
            "end_line": body.count("\n") + 1, "score": 0.03, "content": body}


def test_a_long_body_is_cut_and_says_which_lines_are_left():
    from lynx._format import RESULT_BODY_CHARS, _format_one_result

    hit = _hit(1, RESULT_BODY_CHARS * 2)
    text = _format_one_result(1, hit)
    assert len(text) < RESULT_BODY_CHARS + 400
    assert "more lines, cut at" in text and "read src/f1.py:L" in text
    small = _hit(2, 500)
    assert small["content"] in _format_one_result(1, small)


def test_past_the_budget_the_rest_comes_as_outlines():
    from lynx._format import RESPONSE_BODY_CHARS, _format_search_results

    hits = [_hit(i, 3900) for i in range(1, 9)]
    text = _format_search_results("q", hits, "source 'x'", "")
    shown = text.count("line 20 ")          # only a full body reaches line 20
    assert 0 < shown < len(hits)
    assert "Output budget" in text
    assert len(text) < RESPONSE_BODY_CHARS + 3 * 4000
    # Every hit is still ranked and cited.
    for i in range(1, 9):
        assert f"--- {i}. src/f{i}.py" in text
