"""Shared filesystem walk for codebase sources.

ONE source of truth for "which files belong to a codebase source". Both the
vector-index build (`rag_manager.CodebaseRAG`) and the graph layer
(`graph.builder.GraphLayer`) call into here, so the two can never disagree on
the file set — previously each had its own copy of the walk and they drifted
(the graph honored `ignored_path_fragments`, the vector index did not, so e.g.
`node_modules` ended up embedded into ChromaDB while the graph and watcher
excluded it).

Rules (mirror `SimpleDirectoryReader(exclude_hidden=True)` plus the ignores):
  - skip dot-prefixed directories and files;
  - keep only files whose extension is in `extensions`;
  - skip any path containing one of `ignored_fragments`;
  - skip any file matching one of `ignored_globs` (see `glob_hit`).

Fragment matching is **separator-agnostic**: both the path and each fragment are
normalized to forward slashes before the substring test, so a fragment works
regardless of how it was stored (the config normalizes fragments to OS-native
separators) or which OS the path came from.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable


def normalize_extensions(extensions: Iterable[str]) -> frozenset:
    """Lowercased, leading-dot extension set."""
    return frozenset(
        (e if e.startswith(".") else f".{e}").lower() for e in extensions
    )


def normalize_fragments(fragments: Iterable[str]) -> tuple:
    """Forward-slash, stripped, empties dropped — so matching is OS-agnostic."""
    out = []
    for frag in fragments or ():
        f = str(frag).replace("\\", "/").strip()
        if f:
            out.append(f)
    return tuple(out)


def _frag_hit(path: str, frags: tuple) -> bool:
    needle = str(path).replace("\\", "/")
    return any(frag in needle for frag in frags)


def normalize_globs(globs: Iterable[str]) -> tuple:
    """Forward-slash, stripped, empties dropped; lowercased on Windows,
    where case does not tell two files apart."""
    out = []
    for glob in globs or ():
        g = str(glob).replace("\\", "/").strip()
        if g:
            out.append(g.lower() if os.name == "nt" else g)
    return tuple(out)


def glob_hit(path: str, root, globs: tuple) -> bool:
    """True if the file at `path` matches one of `globs` (normalized).

    A glob without a slash is matched against the file name
    (`20??-??-??-*.txt`); one with a slash against the path relative to the
    source root, written with forward slashes, where a leading `**/` also
    matches at the root (`**/Generated/*.cs`). `*` crosses directories, as in
    fnmatch. Globs filter files only: to skip a whole directory, an ignored
    fragment also stops the walk from descending into it.
    """
    if not globs:
        return False
    try:
        rel = os.path.relpath(str(path), str(root)).replace("\\", "/")
    except ValueError:            # another drive: not under this source
        return False
    if os.name == "nt":
        rel = rel.lower()
    name = rel.rsplit("/", 1)[-1]
    for g in globs:
        if "/" not in g:
            if fnmatch.fnmatchcase(name, g):
                return True
        elif fnmatch.fnmatchcase(rel, g) or (
                g.startswith("**/") and fnmatch.fnmatchcase(rel, g[3:])):
            return True
    return False


def is_ignored(path: str, ignored_fragments: Iterable[str],
               ignored_globs: Iterable[str] = (), root=None) -> bool:
    """True if `path` contains an ignored fragment (separator-agnostic) or,
    given the source `root`, matches an ignored glob.

    Used by the file watcher to decide whether a single FS event is in scope,
    and by search to drop chunks of files the config excludes but the index
    still holds until the next build.
    """
    frags = normalize_fragments(ignored_fragments)
    if frags and _frag_hit(path, frags):
        return True
    globs = normalize_globs(ignored_globs)
    return bool(globs) and root is not None and glob_hit(path, root, globs)


def in_scope(path, root, extensions: Iterable[str],
             ignored_fragments: Iterable[str] = (),
             ignored_globs: Iterable[str] = ()) -> bool:
    """Whether one file belongs to the source, by the rules
    `list_candidate_files` applies to the whole tree: under the root, no
    dot-prefixed directory or name below it, a supported extension, no ignored
    fragment or glob.

    The watcher and search ask this about single paths. When the watcher had
    its own, looser rule, it indexed files under `.claude/` that every build
    then dropped again.
    """
    try:
        rel = os.path.relpath(str(path), str(root))
    except ValueError:            # another drive
        return False
    parts = Path(rel).parts
    if not parts or parts[0] == ".." or any(p.startswith(".") for p in parts):
        return False
    if Path(str(path)).suffix.lower() not in normalize_extensions(extensions):
        return False
    return not is_ignored(path, ignored_fragments, ignored_globs, root)


def list_candidate_files(
    codebase_path,
    extensions: Iterable[str],
    ignored_fragments: Iterable[str] = (),
    ignored_globs: Iterable[str] = (),
) -> list:
    """Absolute, sorted list of files to index for a codebase source.

    Ignored directories are pruned during the walk (we never descend into them),
    which both speeds up the scan on big trees (`node_modules`, `build/`) and
    yields the exact same file set as a per-file filter would (substring match
    on a directory implies the same match on every file beneath it).
    """
    exts = normalize_extensions(extensions)
    frags = normalize_fragments(ignored_fragments)
    globs = normalize_globs(ignored_globs)
    root = str(codebase_path)
    out: list = []
    for dirpath, dirs, files in os.walk(root):
        # exclude_hidden=True: prune dot dirs in place so os.walk skips them.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        # Prune ignored directories early (avoids descending into them).
        if frags:
            dirs[:] = [
                d for d in dirs
                if not _frag_hit(os.path.join(dirpath, d), frags)
            ]
        for f in files:
            if f.startswith("."):
                continue
            if Path(f).suffix.lower() not in exts:
                continue
            full = os.path.normpath(os.path.abspath(os.path.join(dirpath, f)))
            if frags and _frag_hit(full, frags):
                continue
            if globs and glob_hit(full, root, globs):
                continue
            out.append(full)
    out.sort()
    return out
