"""`obj.f()` across files, in every language the graph reads.

Found by a reviewer on 2026-11-22: the rule that keeps a TypeScript member
call on the methods of the files it imports (commit 658a935) applied to every
language, and only TS/JS resolve imports to files, so C#, Python, GDScript
and the rest lost every cross-file member call (`a.Foo()` in C#).
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layer(code: Path, storage: Path, exts) -> GraphLayer:
    layer = GraphLayer(storage_dir=storage / "graph", codebase_path=code, supported_extensions=list(exts))
    layer.rebuild(force=True)
    return layer


def _calls(layer) -> set:
    G = layer.graph
    return {(G.nodes[u].get("label"), G.nodes[v].get("label"), d.get("confidence"))
            for u, v, d in G.edges(data=True) if d.get("relation") == "calls"}


def test_a_csharp_member_call_reaches_a_method_in_another_file(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "a.cs", "class A { public void Foo() {} }\n")
    _write(code / "b.cs", "class B { void Bar(A a) { a.Foo(); } }\n")
    assert ("B.Bar", "A.Foo", "resolved") in _calls(_layer(code, tmp_path / "s", [".cs"]))


def test_a_gdscript_member_call_reaches_a_class_name_script(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "mover.gd", "class_name Mover\nfunc move():\n\tpass\n")
    _write(code / "player.gd", "extends Node\nvar m: Mover\nfunc _ready():\n\tm.move()\n")
    assert ("player._ready", "Mover.move", "resolved") in _calls(_layer(code, tmp_path / "s", [".gd"]))


def test_a_python_member_call_reaches_a_method_in_another_file(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "pkg" / "__init__.py", "")
    _write(code / "pkg" / "engine.py", "class Engine:\n    def start(self):\n        pass\n")
    _write(code / "pkg" / "runner.py", "def run(e):\n    e.start()\n")
    assert ("run", "Engine.start", "resolved") in _calls(_layer(code, tmp_path / "s", [".py"]))


def test_of_several_definitions_the_one_in_the_callers_own_file_is_called(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "a.cs", "class A { public void Run() {} }\n")
    _write(code / "b.cs", "class B { public void Run() {} void Go(B b) { b.Run(); } }\n")
    calls = _calls(_layer(code, tmp_path / "s", [".cs"]))
    assert ("B.Go", "B.Run", "resolved") in calls or ("B.Go", "B.Run", "extracted") in calls
    assert not any(src == "B.Go" and dst == "A.Run" for src, dst, _c in calls)


def test_a_java_call_on_a_receiver_is_a_member_call(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "Store.java", "class Store { void add(int x) {} }\n")
    _write(code / "Bag.java", "class Bag { void add(int x) {} }\n")
    _write(code / "User.java", "class User { void go(java.util.List<Integer> l) { l.add(2); } }\n")
    calls = _calls(_layer(code, tmp_path / "s", [".java"]))
    # Two methods named add and nothing to choose between them: a member call
    # is skipped, where a plain call would link both as ambiguous.
    assert not any(src == "User.go" for src, _dst, _c in calls)


def test_a_typescript_member_call_still_needs_an_import(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "heap.ts", "export class MinHeap { push(x: number) {} }\n")
    _write(code / "user.ts", "export function go(a: number[]) { a.push(1); }\n")
    calls = _calls(_layer(code, tmp_path / "s", [".ts"]))
    assert not any(src == "go" for src, _dst, _c in calls)
