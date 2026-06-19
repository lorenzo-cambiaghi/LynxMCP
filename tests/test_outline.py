"""Unit tests for `lynx.outline` — signature/doc derivation shared by the HTTP
`/api/v1/search?view=outline` endpoint and the MCP `search(outline=true)` tool.

Pure functions, no deps — these lock the behavior the token saving relies on.
"""
from __future__ import annotations

from lynx.outline import signature_of, doc_of, signature_for, _preview_line


def test_signature_brace_language_cuts_at_brace():
    cs = (
        "public static float Clamp(float a, float min, float max)\n"
        "        {\n            return Mathf.Clamp(a, min, max);\n        }"
    )
    assert signature_of(cs, "c_sharp") == "public static float Clamp(float a, float min, float max)"


def test_signature_python_cuts_at_colon():
    py = 'def bar(self, x):\n    """Return x, doubled."""\n    return x * 2'
    assert signature_of(py, "python") == "def bar(self, x)"


def test_signature_multiline_header_collapses_whitespace():
    py = "def iter_content(\n    self, chunk_size=1, decode_unicode=False\n):\n    pass"
    assert signature_of(py) == "def iter_content( self, chunk_size=1, decode_unicode=False )"


def test_signature_ignores_type_hint_colon():
    # the `:` in `a: int` must NOT be mistaken for the body opener
    assert signature_of("def f(a: int) -> int:\n    return a", "python") == "def f(a: int) -> int"


def test_doc_extracts_first_docstring_line():
    py = 'def bar(self):\n    """Return x, doubled.\n\n    Longer text."""\n    return 1'
    assert doc_of(py, "python") == "Return x, doubled."


def test_doc_empty_without_in_chunk_docstring():
    assert doc_of("def bar(): return 1", "python") == ""
    assert doc_of("public void Foo() { }", "c_sharp") == ""   # C# /// precedes the node


def test_signature_for_text_window_is_a_clean_preview():
    # a brace in the middle of a TEXT chunk must not truncate it to a fragment
    text = "376 + if (bindToShader) Shader.SetGlobalVector(x) { foo }"
    assert signature_for(text, "text_window") == text
    # but a real code chunk still cuts at the body opener
    assert signature_for("void Foo() { body }", "method") == "void Foo()"


def test_preview_line_first_nonempty_collapsed():
    assert _preview_line("\n\n  hello   world  \nmore") == "hello world"


def test_empty_content():
    assert signature_of("") == ""
    assert doc_of("") == ""
    assert signature_for("", "method") == ""
