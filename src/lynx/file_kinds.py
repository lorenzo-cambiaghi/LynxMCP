"""Which indexed files are the code itself, and which are tests, data or docs.

A question about behaviour ("where does a click pick the person under the
pointer") is answered by the code; tests, JSON data and Markdown tell the same
story in plain words, so they win on the words and crowd it out. Measured on
The Lost Storm (214k lines of TypeScript, 889 test files; lynx-zombie-bench
E17 and E18): over 76 issue titles nobody in the bench wrote, halving the
fused score of those files took the right file from 1st place 4 times to 19,
top 5 from 28 to 38, MRR from 0.185 to 0.362.

Standard library only. Paths are relative to the source root: an absolute path
would mark a whole project as tests when it lives under a folder called
`test/`.
"""
from __future__ import annotations

import re

# The conventions `find_tests_for` has always used: Python pytest (`tests/`,
# `test/`, `_test.py`), JS/TS (`.test.`, `.spec.`, `__tests__/`), C#
# xUnit/NUnit (`Test.cs`, `Tests.cs`), Go (`_test.go`), Godot GUT
# (`test_player.gd`).
TEST_PATH_PATTERN = (
    r"(^|/)(test|tests|spec|__tests__)/|"
    r"_test\.|\.test\.|\.spec\.|"
    r"(^|/)test_[^/]*$|"
    r"Test\.cs$|Tests\.cs$"
)
_TEST = re.compile(TEST_PATH_PATTERN, re.IGNORECASE)

# Data and prose. Measured: .json and .md; the other prose formats behave the
# same way and join them.
SECONDARY_SUFFIXES = (".json", ".md", ".mdx", ".rst", ".txt")


def is_test_path(rel_path: str) -> bool:
    return bool(rel_path) and bool(_TEST.search(rel_path.replace("\\", "/")))


def is_secondary(rel_path: str) -> bool:
    """A test, data or doc file: kept in results, ranked after the code."""
    if not rel_path:
        return False
    return is_test_path(rel_path) or rel_path.lower().endswith(SECONDARY_SUFFIXES)
