"""The read guard: a Read of more than READ_LINES lines of an indexed file is
refused once with the file's outline; short reads, ranges, other files and
the second ask go through.

Probed with Claude Code on 2026-11-22: the deny reason reaches the model as
the tool's error and it then reads the narrower range.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from lynx import read_guard as rg
from lynx.server import _register_search_tools


def _decide(total, offset="", limit="", seen=None):
    seen = set() if seen is None else seen

    def already(key):
        if key in seen:
            return True
        seen.add(key)
        return False
    return rg.decide("src/input.ts", total, offset, limit, lambda: "L10-20 function pick()", already)


def test_a_whole_long_file_is_refused_with_its_outline():
    out = json.loads(_decide(462))
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PreToolUse" and spec["permissionDecision"] == "deny"
    assert "lines 1-462 of src/input.ts" in spec["permissionDecisionReason"]
    assert "L10-20 function pick()" in spec["permissionDecisionReason"]


def test_short_files_and_narrow_ranges_go_through():
    assert _decide(rg.READ_LINES) == "{}"
    assert _decide(462, offset="240", limit="40") == "{}"


def test_a_wide_range_is_refused_too():
    assert json.loads(_decide(800, offset="100", limit="480"))["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_same_read_asked_again_goes_through():
    seen = set()
    assert json.loads(_decide(462, seen=seen))["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert _decide(462, seen=seen) == "{}"


def test_the_span_follows_the_read_tool():
    assert rg.requested_span(462, "", "") == (1, 462)
    assert rg.requested_span(462, "100", "480") == (100, 462)
    assert rg.requested_span(462, "not a number", "50") == (1, 50)


class _Rag:
    def __init__(self, root):
        self.codebase_path = root
        self.supported_extensions = {".ts"}
        self.ignored_path_fragments = ["/node_modules/"]

    def is_supported(self, path):
        p = Path(path)
        try:
            p.resolve().relative_to(Path(self.codebase_path).resolve())
        except ValueError:
            return False
        return p.suffix in self.supported_extensions


class _Backend:
    type_name = "codebase"

    def __init__(self, root):
        self.rag = _Rag(root)
        self.source_config = {"path": str(root)}


class _Manager:
    def __init__(self, root):
        self.backends = {"demo": _Backend(root)}

    def module_summary(self, source, rel):
        return {"file": rel, "files": [rel], "symbols": [
            {"name": "pick", "kind": "function", "start_line": 10, "end_line": 20, "signature": "function pick()"}],
            "imports": [], "dependent_files": []}


def _call(manager, mcp=None, **args):
    if mcp is None:
        mcp = FastMCP("t")
        _register_search_tools(mcp, manager)
    content = asyncio.run(mcp.call_tool("read_guard", args))
    blocks = content[0] if isinstance(content, tuple) else content
    return blocks[0].text


def test_the_tool_refuses_a_long_indexed_file_and_ignores_the_rest(tmp_path):
    (tmp_path / "src").mkdir()
    long_file = tmp_path / "src" / "input.ts"
    long_file.write_text("\n".join(f"const v{i} = {i};" for i in range(400)), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("\n".join("x" for _ in range(400)), encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    vendored = tmp_path / "node_modules" / "lib.ts"
    vendored.write_text("\n".join("x" for _ in range(400)), encoding="utf-8")
    m = _Manager(tmp_path)

    out = json.loads(_call(m, file_path=str(long_file), offset="", limit=""))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "src/input.ts" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert _call(m, file_path=str(tmp_path / "notes.txt"), offset="", limit="") == "{}"
    assert _call(m, file_path=str(vendored), offset="", limit="") == "{}"
    assert _call(m, file_path=str(long_file), offset="100", limit="50") == "{}"


def test_a_failure_lets_the_read_run(tmp_path):
    class _Broken(_Manager):
        @property
        def backends(self):
            raise RuntimeError("Lynx is still opening its indexes")

        @backends.setter
        def backends(self, value):
            pass
    assert _call(_Broken(tmp_path), file_path=str(tmp_path / "x.ts"), offset="", limit="") == "{}"


def test_bash_reads_follow_cd_and_skip_filtered_output(tmp_path):
    base = str(tmp_path)
    reads = rg.bash_reads('cd src/ts; sed -n 590,760p Character.ts; cat a.ts | grep x; head -n 300 b.ts', base)
    names = [(Path(r[0]).name, r[1], r[2], r[3]) for r in reads]
    assert names == [("Character.ts", 590, 760, None), ("b.ts", 1, 300, None)]
    assert rg.span_of(("f", None, None, 50), 400) == (351, 400)


def test_the_tool_refuses_a_wide_bash_read_of_an_indexed_file(tmp_path):
    (tmp_path / "src").mkdir()
    long_file = tmp_path / "src" / "input.ts"
    long_file.write_text("\n".join(f"const v{i} = {i};" for i in range(400)), encoding="utf-8")
    m = _Manager(tmp_path)
    server = FastMCP("t")
    _register_search_tools(server, m)
    out = json.loads(_call(m, server, command=f'cd "{tmp_path.as_posix()}" && cat src/input.ts'))
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny" and "sed -n" in reason
    assert _call(m, server, command=f'cd "{tmp_path.as_posix()}" && sed -n 10,60p src/input.ts') == "{}"
    assert _call(m, server, command=f'cd "{tmp_path.as_posix()}" && cat src/input.ts') == "{}", "asked again: through"
    assert _call(m, server, command="ls -la") == "{}"
