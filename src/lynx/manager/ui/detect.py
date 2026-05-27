"""Filesystem-introspection helpers used by the "Add source" UI flow.

The web UI calls these from `POST /api/sources/_detect` to populate the
codebase form with smart defaults (top-N file extensions, git presence)
when the user picks a folder.

Both helpers are pure stdlib + filesystem-only — no FastAPI, no network,
safe to import from anywhere.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path


# Directories we never recurse into when scanning for extensions. Mirrors
# the defaults baked into the codebase source config so what the user
# sees in the form matches what `lynx build` will actually index.
_IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".idea", ".vscode", "dist", "build", "target", ".next",
}
_IGNORED_EXTS = {".pyc", ".so", ".dylib", ".dll", ".class", ".o"}


def detect_extensions(folder: Path, top_n: int = 10) -> list:
    """Walk `folder` and return the top-N file extensions by count.

    Skips dot-prefixed dirs (e.g. `.git`, `.venv`) and common compiled
    artifacts (`.pyc`, `.so`, ...). Returns lower-case extensions with
    the leading dot, ordered by frequency (most common first).

    Empty list if nothing matches (e.g. a folder of binaries only).
    """
    counter: Counter = Counter()
    for root, dirs, files in os.walk(folder):
        # Prune in place — os.walk respects mutation of `dirs`
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _IGNORED_DIRS]
        for f in files:
            ext = Path(f).suffix.lower()
            if not ext or ext in _IGNORED_EXTS:
                continue
            counter[ext] += 1
    return [ext for ext, _ in counter.most_common(top_n)]


def is_git_repo(folder: Path) -> bool:
    """Return True if `folder` (or any ancestor) contains a `.git/` entry.

    Mirrors `git rev-parse --is-inside-work-tree` behavior without
    invoking the git binary — works on bare-bones systems and stays fast.
    """
    folder = folder.resolve()
    while True:
        if (folder / ".git").exists():
            return True
        if folder.parent == folder:  # reached fs root
            return False
        folder = folder.parent
