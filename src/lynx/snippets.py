"""Code read back from disk for the tools that answer in one call.

`describe_symbol`, `find_definition` and `module_summary` used to answer
with `file:L42-88` and nothing else, so the next thing an agent did was
always a `read_file` of those lines: one more model round-trip, the whole
conversation billed again. They now carry the code, or its signatures,
read here. The graph knows where a symbol is; the file says what it is.

Files are read whole, once per call (`FileLines`), and sliced: a class with
thirty members costs one read. A file changed since it was indexed can have
moved lines; the watcher re-indexes it within seconds, so the window is
short, and the worst case is a slice a few lines off.
"""
from __future__ import annotations

import os

from .outline import doc_above, doc_of, signature_for

# The body handed back with a definition. Most functions fit (the median
# chunk is under 300 characters); a long one is cut, and the cut says which
# lines are left.
BODY_CHARS = 2500


class FileLines:
    """Lines of the files one call reads, each file read once."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def get(self, path: str):
        """The file's lines, or None when it cannot be read."""
        if not path:
            return None
        key = os.path.normcase(os.path.normpath(path))
        if key not in self._cache:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    self._cache[key] = f.read().split("\n")
            except OSError:
                self._cache[key] = None
        return self._cache[key]


def _lead_in(lines: list, index: int) -> int:
    """Index of the first line of the comment / decorator block directly above
    `lines[index]`: the graph points at `def` or `public void`, and the doc
    comment or the `@decorator` on top belong to the same declaration."""
    j = index
    while j > 0:
        s = lines[j - 1].strip()
        if not s:
            break
        if s.startswith(("//", "/*", "*", "#", "@", "--")) or s.endswith("*/"):
            j -= 1
            continue
        break
    return j


def symbol_body(files: FileLines, path: str, start: int, end: int,
                max_chars: int = BODY_CHARS) -> tuple:
    """(text, lines left out) of a symbol, its doc comment included, cut at a
    line break past `max_chars`. ("", 0) when the file is gone or the range
    is unknown."""
    lines = files.get(path)
    if not lines or not start:
        return "", 0
    first = _lead_in(lines, start - 1)
    last = min(len(lines), end or start)
    chunk = lines[first:last]
    out, size = [], 0
    for i, line in enumerate(chunk):
        if size + len(line) > max_chars and out:
            return "\n".join(out), len(chunk) - i
        out.append(line)
        size += len(line) + 1
    return "\n".join(out), 0


def symbol_outline(files: FileLines, path: str, start: int, end: int,
                   kind: str = "") -> tuple:
    """(signature, doc) of the symbol at those lines; ("", "") if unreadable."""
    lines = files.get(path)
    if not lines or not start:
        return "", ""
    # The declaration header rarely needs more than a few lines; the cap
    # keeps a class's members cheap to sign one by one.
    text = "\n".join(lines[start - 1:min(len(lines), end or start, start + 40)])
    sig = signature_for(text, kind)
    doc = doc_of(text) or doc_above(lines, start - 1)
    return sig, doc
