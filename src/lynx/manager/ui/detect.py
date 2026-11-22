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


# Directories we never recurse into when scanning for extensions. Mirrors the
# default ignores a NEW codebase source gets at creation time (see
# `routes._DEFAULT_CODEBASE_IGNORES`), so the extensions the form shows match
# what `lynx build` will actually index. NB: these defaults apply to sources
# created via the UI; sources written by hand in config.json keep whatever
# `ignored_path_fragments` they declare (possibly none).
_IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".idea", ".vscode", "dist", "build", "target", ".next", "obj",
}
# `.import` and `.uid` are Godot's per-asset and per-script sidecar files:
# as many as the scripts, and nothing to search in them.
_IGNORED_EXTS = {".pyc", ".so", ".dylib", ".dll", ".class", ".o", ".import", ".uid"}


def detect_extensions(
    folder: Path,
    top_n: int | None = 10,
    *,
    allowed=None,
    extra_ignored_dirs=(),
    max_files: int | None = None,
) -> list:
    """Walk `folder` and return the top-N file extensions by count.

    Skips dot-prefixed dirs (e.g. `.git`, `.venv`) and common compiled
    artifacts (`.pyc`, `.so`, ...). Returns lower-case extensions with
    the leading dot, ordered by frequency (most common first).

    `allowed` keeps only those extensions (zero-config mode passes the
    formats Lynx can read, so a Unity project's `.meta` and `.png` never
    make the list). `extra_ignored_dirs` adds directory names to prune and
    `max_files` stops the walk early on huge trees. `top_n=None` returns
    every extension found.

    Empty list if nothing matches (e.g. a folder of binaries only).
    """
    counter: Counter = Counter()
    pruned = _IGNORED_DIRS | set(extra_ignored_dirs)
    seen = 0
    for root, dirs, files in os.walk(folder):
        # Prune in place: os.walk respects mutation of `dirs`
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in pruned]
        for f in files:
            seen += 1
            ext = os.path.splitext(f)[1].lower()
            if not ext or ext in _IGNORED_EXTS:
                continue
            if allowed is not None and ext not in allowed:
                continue
            counter[ext] += 1
        if max_files is not None and seen >= max_files:
            break
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
