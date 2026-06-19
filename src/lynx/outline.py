"""Outline view of a search hit — a compact, body-free `signature` + `doc` for
cheap triage. Shared by the HTTP `/api/v1/search?view=outline` endpoint and the
MCP `search(outline=true)` tool so the two produce identical signatures.

Query-time only: derived from the chunk text (+ its AST kind), no reindex, no
re-parse, no LLM. See docs/OUTLINE.md for the rationale and the measured saving.
"""
from __future__ import annotations

import re


def signature_of(content: str, language: str = "") -> str:
    """Best-effort one-line declaration: the chunk text up to the body opener
    (`{` for brace languages, a line-ending `:` for Python/Ruby-style headers),
    whitespace-collapsed. Falls back to the first non-empty line."""
    if not content:
        return ""
    cut = len(content)
    brace = content.find("{")
    if brace != -1:
        cut = min(cut, brace)
    m = re.search(r":\s*(?:\n|$)", content)   # `def foo(...):` end, not a type hint
    if m:
        cut = min(cut, m.start())
    sig = re.sub(r"\s+", " ", content[:cut]).strip()
    if not sig:                                # opener on the very first char
        sig = _preview_line(content)
    return sig[:400]


def doc_of(content: str, language: str = "") -> str:
    """First line of an in-chunk docstring / leading comment, if any. Python &
    co. keep the docstring inside the body (so it lands in the chunk); C#-style
    `///` doc precedes the node and won't appear here."""
    if not content:
        return ""
    for quote in ('"""', "'''"):
        m = re.search(re.escape(quote) + r"(.*?)" + re.escape(quote), content, re.S)
        if m and m.group(1).strip():
            return m.group(1).strip().splitlines()[0][:200]
    m = re.search(r"^\s*///\s?(.+)$", content, re.M)
    if m:
        return m.group(1).strip()[:200]
    return ""


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
