"""Outline view of a search hit — a compact, body-free `signature` + `doc` for
cheap triage. Shared by the HTTP `/api/v1/search?view=outline` endpoint and the
MCP `search(outline=true)` tool so the two produce identical signatures.

Query-time only: derived from the chunk text (+ its AST kind), no reindex, no
re-parse, no LLM. See docs/OUTLINE.md for the rationale and the measured saving.

A chunk starts with the doc comment written above its declaration (`///`,
`/** */`, `//`, `#`): the chunker keeps it, because it is the one place the
code says in words what it does. `signature_of` steps over it, `doc_of`
reads it.
"""
from __future__ import annotations

import re

# One leading comment line: `//`, `///`, `#` (Python, Ruby, shell), `--`
# (Lua, SQL), or a line inside a `/* ... */` block.
_LINE_COMMENT = re.compile(r"^\s*(?://|#(?=\s|#|$)|--)")
_BLOCK_OPEN = re.compile(r"^\s*/\*")


def split_leading_comment(content: str) -> tuple:
    """(comment block, rest): the comment lines a chunk opens with, and the
    code after them. Leaves the content alone when it opens with code."""
    if not content:
        return "", ""
    lines = content.split("\n")
    i, in_block = 0, False
    while i < len(lines):
        line = lines[i]
        if in_block:
            if "*/" in line:
                in_block = False
            i += 1
            continue
        if not line.strip():
            if i == 0:
                i += 1
                continue
            break
        if _BLOCK_OPEN.match(line):
            in_block = "*/" not in line.split("/*", 1)[1]
            i += 1
            continue
        if _LINE_COMMENT.match(line):
            i += 1
            continue
        break
    if i == 0:
        return "", content
    return "\n".join(lines[:i]), "\n".join(lines[i:])


def signature_of(content: str, language: str = "") -> str:
    """Best-effort one-line declaration: the chunk text up to the body opener
    (`{` for brace languages, a line-ending `:` for Python/Ruby-style headers),
    whitespace-collapsed, after any leading doc comment. Falls back to the
    first non-empty line."""
    if not content:
        return ""
    _comment, code = split_leading_comment(content)
    code = code if code.strip() else content
    cut = len(code)
    brace = code.find("{")
    if brace != -1:
        cut = min(cut, brace)
    m = re.search(r":\s*(?:\n|$)", code)   # `def foo(...):` end, not a type hint
    if m:
        cut = min(cut, m.start())
    sig = re.sub(r"\s+", " ", code[:cut]).strip()
    if not sig:                                # opener on the very first char
        sig = _preview_line(code)
    return sig[:400]


# Markup a doc comment carries that is not its sentence: comment markers,
# XML doc tags, Javadoc's leading `*`.
_DOC_MARKER = re.compile(r"^\s*(?:/\*\*?|\*/|\*(?!/)|///?|#+|--)\s?")
_XML_TAG = re.compile(r"</?\s*(?:summary|remarks|para|returns|value|example)\b[^>]*>")
_INLINE_TAG = re.compile(r"<(?:see|seealso|paramref|typeparamref)\s+\w+\s*=\s*\"([^\"]*)\"\s*/?>")
_JAVADOC_INLINE = re.compile(r"\{@\w+\s+([^}]*)\}")
_BLOCK_TAG = re.compile(r"^\s*(?:@\w+|<(?:param|typeparam|exception|returns)\b|:param|:return|Args:|Returns:)")


def _first_doc_line(comment: str) -> str:
    """The first sentence-bearing line of a comment block, markup removed."""
    for raw in comment.split("\n"):
        line = _DOC_MARKER.sub("", raw).replace("*/", "").strip()
        if not line or _BLOCK_TAG.match(line):
            continue
        line = _INLINE_TAG.sub(lambda m: m.group(1).split(".")[-1], line)
        line = _JAVADOC_INLINE.sub(lambda m: m.group(1).strip(), line)
        line = _XML_TAG.sub("", line).strip()
        line = re.sub(r"</?c>|</?code>", "", line).strip()
        if line:
            return line[:200]
    return ""


def doc_of(content: str, language: str = "") -> str:
    """First line of a chunk's documentation: the doc comment above the
    declaration (`///` XML docs, Javadoc/JSDoc, Go/Rust `//`, `#`), or a
    Python-style docstring inside the body."""
    if not content:
        return ""
    comment, code = split_leading_comment(content)
    for quote in ('"""', "'''"):
        m = re.search(re.escape(quote) + r"(.*?)" + re.escape(quote), code, re.S)
        if m and m.group(1).strip():
            return m.group(1).strip().splitlines()[0][:200]
    if comment:
        return _first_doc_line(comment)
    return ""


def doc_above(lines: list, index: int) -> str:
    """The doc line of the comment block ending right above `lines[index]`
    (0-based): what `doc_of` would read had the chunk started there. Used
    where a symbol's lines come from the graph, which points at the
    declaration itself."""
    j = index
    while j > 0:
        prev = lines[j - 1]
        stripped = prev.strip()
        if not stripped:
            break
        if (_LINE_COMMENT.match(prev) or stripped.startswith(("/*", "*", "*/"))
                or stripped.endswith("*/")):
            j -= 1
            continue
        break
    if j == index:
        return ""
    return _first_doc_line("\n".join(lines[j:index]))


def _preview_line(content: str) -> str:
    """First non-empty line, whitespace-collapsed — used for non-AST chunks."""
    line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    return re.sub(r"\s+", " ", line)[:200]


def signature_for(content: str, kind: str = "", language: str = "") -> str:
    """Signature for a hit, dispatching on its AST `kind`. Non-AST chunks
    (`text_window`, from plain-text / unsupported files) have no real
    signature, so we return a clean first-line preview instead of a mid-text
    fragment."""
    if kind == "text_window":
        return _preview_line(content)
    return signature_of(content, language)
