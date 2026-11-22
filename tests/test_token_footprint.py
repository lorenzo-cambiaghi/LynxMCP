"""What Lynx adds to the context of an agent: answers and the rules file.

Every character a tool returns is written into the client's prompt cache
once and re-read on every later call of the session, so the size of an
answer is a cost, not just a style. These pin the compact shapes adopted on
2026-11-21 after the Sketchbook bench, where Claude Code read 40,700
characters through Lynx (more than the 33,700 of its own grep path) because
the defaults returned whole bodies and a 9,700-character module summary.
"""
from __future__ import annotations


def _hit(i, size, symbol=None):
    body = f"def f{i}(a, b):\n" + "\n".join(f"    line_{n} = " + "x" * 60
                                           for n in range(size // 72 + 1))
    return {"rel_path": f"src/f{i}.py", "symbol_name": symbol or f"f{i}", "start_line": 1,
            "end_line": body.count("\n") + 1, "score": 0.03, "content": body,
            "symbol_kind": "function_definition", "language": "python"}


def test_the_mcp_search_answer_carries_one_body_and_the_rest_as_signatures():
    from lynx._format import SEARCH_BODY_CHARS, _format_search_results

    hits = [_hit(i, 3000) for i in range(1, 6)]
    text = _format_search_results("q", hits, "source 'x'", "", SEARCH_BODY_CHARS)

    assert text.count("more lines, cut at") == 1          # the best hit, capped
    assert "Output budget" in text
    assert len(text) < SEARCH_BODY_CHARS + 5 * 200
    for i in range(1, 6):                                 # every hit still cited
        assert f"--- {i}. src/f{i}.py" in text


def test_short_bodies_share_the_budget():
    from lynx._format import SEARCH_BODY_CHARS, _format_search_results

    hits = [_hit(i, 300) for i in range(1, 6)]
    text = _format_search_results("q", hits, "source 'x'", "", SEARCH_BODY_CHARS)

    # Five 300-character bodies fit in 2,000: no hit is reduced to an outline.
    assert "Output budget" not in text and all(h["content"] in text for h in hits)


def test_the_cli_keeps_the_full_view():
    from lynx._format import RESPONSE_BODY_CHARS, _format_search_results

    hits = [_hit(i, 3000) for i in range(1, 5)]
    text = _format_search_results("q", hits, "source 'x'", "")
    assert RESPONSE_BODY_CHARS > 4 * 3000 and "Output budget" not in text


def _class_summary():
    members = [
        {"label": "Character", "kind": "class", "start_line": 31, "end_line": 989,
         "signature": "export class Character extends THREE.Object3D implements IWorldEntity"},
        {"label": "Character.setPosition", "kind": "function", "start_line": 190, "end_line": 204,
         "signature": "public setPosition(x: number, y: number, z: number): void"},
        {"label": "Character.setState", "kind": "function", "start_line": 184, "end_line": 188,
         "signature": "public setState(state: ICharacterState): void",
         "doc": "Set state to the player."},
        {"label": "Character.speed", "kind": "field", "start_line": 40, "end_line": 40},
    ]
    return {"file": "src/Character.ts", "graph_enabled": True, "symbols": members,
            "imports": [{"module": "three"}, {"module": "../core/KeyBinding"}],
            "dependent_files": ["src/World.ts"], "files": ["src/Character.ts"]}


def test_module_summary_is_one_line_per_symbol():
    from lynx._format import _format_module_summary

    text = _format_module_summary("src/Character.ts", _class_summary())
    lines = text.splitlines()

    assert "  L31-989    export class Character extends THREE.Object3D implements IWorldEntity" in lines
    # Members sit under their class: no `Character.` prefix, no `public`, no kind.
    assert "    L190-204   setPosition(x: number, y: number, z: number): void" in lines
    assert "    L184-188   setState(state: ICharacterState): void  ↳ Set state to the player." in lines
    assert "    L40        speed [field]" in lines
    assert "IMPORTS: three, ../core/KeyBinding" in lines
    assert "DEPENDED ON BY (1 file(s), imports and calls): src/World.ts" in lines
    # The file is not repeated next to itself in the heading.
    assert lines[0] == "Module summary for 'src/Character.ts':"


def test_module_summary_size_per_symbol():
    # The layout it replaces spent about 200 characters per symbol.
    from lynx._format import _format_module_summary

    d = _class_summary()
    d["symbols"] = [d["symbols"][0]] + [
        {"label": f"Character.method{i}", "kind": "function", "start_line": 100 + i * 10,
         "end_line": 108 + i * 10, "signature": f"public method{i}(value: number): void"}
        for i in range(40)
    ]
    text = _format_module_summary("src/Character.ts", d)
    assert len(text) / 41 < 60


def test_module_summary_lists_a_few_dependents_and_counts_the_rest():
    from lynx._format import _MAX_DEPENDENTS, _format_module_summary

    d = _class_summary()
    d["dependent_files"] = [f"src/states/State{i:02d}.ts" for i in range(33)]
    line = next(ln for ln in _format_module_summary("src/Character.ts", d).splitlines()
                if ln.startswith("DEPENDED ON BY"))
    assert line.startswith("DEPENDED ON BY (33 file(s), imports and calls): src/states/State00.ts")
    assert line.endswith(f" and {33 - _MAX_DEPENDENTS} more")
    assert "State32" not in line


def test_the_rules_file_is_a_pointer_not_the_playbook():
    from lynx.manager.ui.integrations import render_rules_file, render_rules_for_sources

    short = render_rules_file(["game"], has_graph=True, has_git=True)
    assert "`game`" in short and "`search`" in short and "lynx://guide" in short
    assert "`graph_query`" in short
    assert len(short) < 800
    # The playbook is still there, as the lynx://guide resource.
    assert len(render_rules_for_sources(["game"], True, True)) > 3 * len(short)


def test_the_rules_file_names_only_tools_that_exist():
    from lynx.manager.ui.integrations import render_rules_file

    core = render_rules_file(["game"], has_graph=False, has_git=False,
                             tools=["search", "describe_symbol", "module_summary",
                                    "find_usages", "impact"])
    assert "find_definition" not in core and "find_similar" not in core
    assert "graph_query" not in core and "impact" not in core    # no graph
