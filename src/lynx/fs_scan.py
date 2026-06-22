"""Shared filesystem walk for codebase sources.

ONE source of truth for "which files belong to a codebase source". Both the
vector-index build (`rag_manager.CodebaseRAG`) and the graph layer
(`graph.builder.GraphLayer`) call into here, so the two can never disagree on
the file set — previously each had its own copy of the walk and they drifted
(the graph honored `ignored_path_fragments`, the vector index did not, so e.g.
`node_modules` ended up embedded into ChromaDB while the graph and watcher
excluded it).

Rules (mirror `SimpleDirectoryReader(exclude_hidden=True)` plus ignore-fragments):
  - skip dot-prefixed directories and files;
  - keep only files whose extension is in `extensions`;
  - skip any path containing one of `ignored_fragments`.

Fragment matching is **separator-agnostic**: both the path and each fragment are
normalized to forward slashes before the substring test, so a fragment works
regardless of how it was stored (the config normalizes fragments to OS-native
separators) or which OS the path came from.
"""
from __future__ import annotations

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


def is_ignored(path: str, ignored_fragments: Iterable[str]) -> bool:
    """True if `path` contains any ignored fragment (separator-agnostic).

    Used by the file watcher to decide whether a single FS event is in scope.
    """
    frags = normalize_fragments(ignored_fragments)
    return bool(frags) and _frag_hit(path, frags)


def list_candidate_files(
    codebase_path,
    extensions: Iterable[str],
    ignored_fragments: Iterable[str] = (),
) -> list:
    """Absolute, sorted list of files to index for a codebase source.

    Ignored directories are pruned during the walk (we never descend into them),
    which both speeds up the scan on big trees (`node_modules`, `build/`) and
    yields the exact same file set as a per-file filter would (substring match
    on a directory implies the same match on every file beneath it).
    """
    exts = normalize_extensions(extensions)
    frags = normalize_fragments(ignored_fragments)
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
            out.append(full)
    out.sort()
    return out
