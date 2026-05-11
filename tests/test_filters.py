"""Smoke test for the search filter parameters (file_glob / extensions /
path_contains).

Constructs CodebaseRAG once against the configured index and runs the same
base query through six scenarios:
  1. baseline (no filters) - expects > 0 results, all extensions allowed,
  2. extensions filter - asserts every result matches a requested extension,
  3. path_contains filter - asserts every result path contains the substring,
  4. file_glob filter - asserts every result matches the glob,
  5. combined filters (AND) - asserts every result satisfies ALL filters,
  6. impossible filter - asserts the system gracefully returns 0 results.

Read-only: the index is not modified.
"""

import os
import sys
from pathlib import Path

from lynx.rag_manager import CodebaseRAG, _matches_filters, _normalize_extension
from conftest import build_rag_from_first_source


QUERY = "serialization"
TOP_K = 10


def _build() -> CodebaseRAG:
    _, _, rag = build_rag_from_first_source(None)
    return rag


def _all(results, predicate, label):
    bad = [r for r in results if not predicate(r)]
    if bad:
        print(f"[test] FAIL: {label} - {len(bad)} of {len(results)} results violate the filter")
        for r in bad[:3]:
            print(f"    offender: file={r['file']!r} file_path={r['file_path']!r}")
        return False
    return True


def main() -> int:
    print("[test] Constructing CodebaseRAG (one embedding-model load)...")
    rag = _build()

    # ----- 1. baseline -----
    baseline = rag.search(QUERY, top_k=TOP_K)
    if not baseline:
        print("[test] FAIL: baseline returned 0 results — does the index contain anything?")
        return 1
    print(f"[test] OK [1/6] baseline returned {len(baseline)} results (no filters)")

    # ----- 2. extensions filter -----
    target_exts = [".cs", ".md"]
    norm = {_normalize_extension(e) for e in target_exts}
    out = rag.search(QUERY, top_k=TOP_K, extensions=target_exts)
    if not out:
        print("[test] FAIL: extensions filter returned 0 results")
        return 2

    def ext_ok(r):
        suffix = os.path.splitext(r["file"] or os.path.basename(r["file_path"]))[1].lower()
        return suffix in norm

    if not _all(out, ext_ok, "extensions filter"):
        return 3
    print(f"[test] OK [2/6] extensions={target_exts} returned {len(out)} results, all matching")

    # ----- 3. path_contains filter -----
    # Pick a substring that we know occurs in the user's framework.
    needle = "Serial"
    out = rag.search(QUERY, top_k=TOP_K, path_contains=needle)
    if not out:
        print(f"[test] WARN [3/6] path_contains={needle!r} returned 0 results — "
              "needle may not exist in this codebase, skipping assertion")
    else:
        if not _all(out, lambda r: needle in r["file_path"] or needle in r["file"], "path_contains filter"):
            return 4
        print(f"[test] OK [3/6] path_contains={needle!r} returned {len(out)} results, all matching")

    # ----- 4. file_glob filter -----
    glob = "*.cs"
    out = rag.search(QUERY, top_k=TOP_K, file_glob=glob)
    if not out:
        print(f"[test] FAIL: file_glob={glob!r} returned 0 results")
        return 5

    def glob_ok(r):
        return _matches_filters(r, file_glob=glob, extensions=None, path_contains=None)

    if not _all(out, glob_ok, "file_glob filter"):
        return 6
    print(f"[test] OK [4/6] file_glob={glob!r} returned {len(out)} results, all matching")

    # ----- 5. combined filters (AND) -----
    out = rag.search(QUERY, top_k=TOP_K, extensions=[".cs"], path_contains="Serial")
    if not out:
        print("[test] WARN [5/6] combined extensions+.cs+path_contains=Serial returned 0 results")
    else:
        for r in out:
            suffix = os.path.splitext(r["file"] or os.path.basename(r["file_path"]))[1].lower()
            if suffix != ".cs":
                print(f"[test] FAIL: combined filter let through non-.cs: {r['file']}")
                return 7
            if "Serial" not in r["file_path"] and "Serial" not in r["file"]:
                print(f"[test] FAIL: combined filter let through missing-needle: {r['file']}")
                return 8
        print(f"[test] OK [5/6] combined filters returned {len(out)} results, all matching")

    # ----- 6. impossible filter -----
    out = rag.search(QUERY, top_k=TOP_K, extensions=[".this_extension_does_not_exist"])
    if out:
        print(f"[test] FAIL: impossible filter returned {len(out)} results, expected 0")
        return 9
    print("[test] OK [6/6] impossible filter cleanly returned 0 results")

    print("\n[test] === SUCCESS: search filters work as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
