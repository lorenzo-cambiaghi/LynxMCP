"""A TypeScript function longer than the embedding window gives its named
inner functions pieces of their own, and split pieces carry their own lines.

Why (lynx-zombie-bench E16, E21): the 512-token window of the embedding
model is about 1,440 characters of code; `pickActors`, a closure in a
358-line `createInput`, was not found by its own name, and the three parts
of that function all said lines 94-451.
"""
from __future__ import annotations

from lynx.chunking import LONG_FUNCTION_CHARS, MAX_CHUNK_CHARS, chunk_file


def _long_function(inner: str, filler_lines: int) -> str:
    filler = "\n".join(f"  const v{i} = compute({i}, 'padding padding padding');" for i in range(filler_lines))
    return (
        "export function createInput(canvas: Surface): Input {\n"
        f"{filler}\n"
        f"{inner}\n"
        "  canvas.addEventListener('pointerdown', (event) => follow(event));\n"
        "  return { pick: pickActors };\n"
        "}\n"
    )


INNER = (
    "  const pickActors = (): Candidate | null => {\n"
    "    const labelId = hooks.labelHover();\n"
    "    return pickActor(candidates, point, labelId);\n"
    "  };\n"
    "  function follow(event: PointerEvent): void {\n"
    "    track(event);\n"
    "  }"
)


def _chunks(src: str, name: str = "input.ts"):
    return {c["symbol_name"]: c for c in chunk_file(f"/repo/src/{name}", src)}


def test_a_long_function_gives_its_inner_functions_pieces_of_their_own():
    src = _long_function(INNER, filler_lines=40)
    assert len(src) > LONG_FUNCTION_CHARS
    chunks = _chunks(src)
    pick = chunks["createInput.pickActors"]
    lines = src.splitlines()
    assert lines[pick["start_line"] - 1].strip().startswith("const pickActors")
    assert lines[pick["end_line"] - 1].strip() == "};"
    assert "pickActor(candidates" in pick["text"]
    assert "createInput.follow" in chunks
    outer = chunks["createInput"]
    # The body of an inner function is indexed once, in its own piece.
    assert "pickActor(candidates" not in outer["text"]
    assert "const pickActors = (): Candidate | null => { ..." in outer["text"]
    assert "addEventListener" in outer["text"], "the rest of the outer function stays"
    assert "line_map" not in outer


def test_a_short_function_stays_whole():
    src = _long_function(INNER, filler_lines=2)
    assert len(src) < LONG_FUNCTION_CHARS
    chunks = _chunks(src)
    assert "createInput.pickActors" not in chunks
    assert "pickActor(candidates" in chunks["createInput"]["text"]


def test_the_parts_of_a_split_piece_carry_their_own_lines():
    src = _long_function(INNER, filler_lines=MAX_CHUNK_CHARS // 40)
    lines = src.splitlines()
    parts = [c for name, c in _chunks(src).items() if name.startswith("createInput#part")]
    assert len(parts) >= 2
    spans = [(p["start_line"], p["end_line"]) for p in parts]
    assert spans[0][0] == 1 and spans[-1][1] == len(lines), spans
    assert spans[0][1] < spans[-1][1] and spans[-1][0] > 1, "not every part spans the whole function"
    for p in parts:
        first = p["text"].strip().splitlines()[0].strip()
        assert first in lines[p["start_line"] - 1] or lines[p["start_line"] - 1].strip() in first


def test_python_keeps_its_long_functions_whole():
    body = "\n".join(f"    v{i} = compute({i}, 'padding padding padding')" for i in range(60))
    src = f"def outer():\n{body}\n    def inner():\n        return 1\n    return inner\n"
    names = set(_chunks(src, "mod.py"))
    assert "outer.inner" not in names
