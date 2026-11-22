"""Imports through incremental updates: every path gives the graph a full
rebuild gives.

Found by a second review on 2026-11-22, each case reproduced before the fix:
the delta rebuild (`rebuild(force=False)`) neither unresolved the imports of
a deleted file nor adopted the imports waiting for an added one; a tsconfig
alias with no wildcard, and a specifier differing in case from the file on
Windows, were filtered out before resolution; `from . import late` left
nothing to adopt when late.py appeared, and turned into a stub a full
rebuild does not make when it went; a barrel that changed what it re-exports
left its importers' calls as they were; an absolute Python import resolved
to a file outside the indexed folder; `d.keys()` in Python reached a free
function `keys` of a module it never imported.
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


def _sigs(layer):
    G = layer.graph
    nodes = {(nid, d.get("kind"), d.get("label")) for nid, d in G.nodes(data=True)}
    edges = {(u, v, d.get("relation"), d.get("confidence"), d.get("module")) for u, v, d in G.edges(data=True)}
    return nodes, edges


def _assert_same_as_full(layer, code: Path, tmp: Path, exts, tag: str) -> None:
    full = _layer(code, tmp / f"full_{tag}", exts)
    got, want = _sigs(layer), _sigs(full)
    assert got[0] == want[0], (tag, "nodes", got[0] ^ want[0])
    assert got[1] == want[1], (tag, "edges", got[1] ^ want[1])


# --- the delta rebuild ----------------------------------------------------

def test_delta_rebuild_after_a_delete_equals_a_full_rebuild(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "a.ts", "export function f() {}\n")
    _write(code / "b.ts", "import { f } from './a';\nexport function g() { f(); }\n")
    layer = _layer(code, tmp_path / "s", [".ts"])
    (code / "a.ts").unlink()
    layer.rebuild(force=False)
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "delete")


def test_delta_rebuild_after_an_add_equals_a_full_rebuild(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "b.ts", "import { f } from './a';\nexport function g() { f(); }\n")
    layer = _layer(code, tmp_path / "s", [".ts"])
    _write(code / "a.ts", "export function f() {}\n")
    layer.rebuild(force=False)
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "add")


# --- TS/JS through update_file ----------------------------------------------

def test_an_exact_tsconfig_alias_is_adopted_when_its_file_appears(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "tsconfig.json", '{"compilerOptions": {"paths": {"~lib": ["lib/index.ts"]}}}\n')
    _write(code / "main.ts", "import { util } from '~lib';\nexport function go() { util(); }\n")
    layer = _layer(code, tmp_path / "s", [".ts"])
    _write(code / "lib" / "index.ts", "export function util() {}\n")
    layer.update_file(str(code / "lib" / "index.ts"))
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "alias")


def test_a_specifier_in_another_case_follows_the_file_system(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "main.ts", "import { w } from './widget';\nexport function go() { w(); }\n")
    layer = _layer(code, tmp_path / "s", [".ts"])
    _write(code / "Widget.ts", "export function w() {}\n")
    layer.update_file(str(code / "Widget.ts"))
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "case")


def test_a_barrel_that_changes_what_it_re_exports_updates_its_importers_calls(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "input" / "heap.ts", "export class MinHeap { push(x: number) {} }\n")
    _write(code / "input" / "index.ts", "export const nothing = 0;\n")
    _write(code / "main.ts", "import { MinHeap } from './input';\nexport function go(h: MinHeap) { h.push(1); }\n")
    layer = _layer(code, tmp_path / "s", [".ts"])
    _write(code / "input" / "index.ts", "export * from './heap';\n")
    layer.update_file(str(code / "input" / "index.ts"))
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "barrel_add")
    _write(code / "input" / "index.ts", "export const nothing = 0;\n")
    layer.update_file(str(code / "input" / "index.ts"))
    _assert_same_as_full(layer, code, tmp_path, [".ts"], "barrel_remove")


# --- Python through update_file and remove_file -------------------------------

def _py_project(code: Path) -> None:
    _write(code / "pyproject.toml", "[project]\nname = 'x'\n")
    _write(code / "app" / "__init__.py", "")
    _write(code / "app" / "engine.py", "class Engine:\n    def start(self):\n        pass\n")
    _write(code / "app" / "runner.py", "from . import engine, late\n\ndef run():\n    engine.Engine().start()\n")


def test_from_package_import_a_submodule_that_appears_later(tmp_path: Path):
    code = tmp_path / "code"
    _py_project(code)
    layer = _layer(code, tmp_path / "s", [".py"])
    _write(code / "app" / "late.py", "def thing():\n    pass\n")
    layer.update_file(str(code / "app" / "late.py"))
    _assert_same_as_full(layer, code, tmp_path, [".py"], "py_create")


def test_from_package_import_a_submodule_that_goes_away(tmp_path: Path):
    code = tmp_path / "code"
    _py_project(code)
    layer = _layer(code, tmp_path / "s", [".py"])
    (code / "app" / "engine.py").unlink()
    layer.remove_file(str(code / "app" / "engine.py"))
    _assert_same_as_full(layer, code, tmp_path, [".py"], "py_delete")


def test_delta_rebuild_of_python_creates_and_deletes(tmp_path: Path):
    code = tmp_path / "code"
    _py_project(code)
    layer = _layer(code, tmp_path / "s", [".py"])
    (code / "app" / "engine.py").unlink()
    _write(code / "app" / "late.py", "def thing():\n    pass\n")
    layer.rebuild(force=False)
    _assert_same_as_full(layer, code, tmp_path, [".py"], "py_delta")


def test_an_absolute_import_does_not_resolve_outside_the_indexed_folder(tmp_path: Path):
    _write(tmp_path / "helper.py", "def h():\n    pass\n")
    code = tmp_path / "sub"
    _write(code / "main.py", "import helper\n\ndef go():\n    helper.h()\n")
    from lynx.graph.extractor import _make_id
    G = _layer(code, tmp_path / "s", [".py"]).graph
    targets = [v for u, v, d in G.edges(data=True) if d.get("relation") == "imports"]
    # The package stub, as for any module not in the index, not a stub named
    # after the file found outside it.
    assert targets == [_make_id("helper")]


def test_a_python_member_call_does_not_reach_a_free_function_of_a_module_it_never_imported(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "a.py", "def keys():\n    return []\n")
    _write(code / "b.py", "def f(d):\n    return d.keys()\n")
    _write(code / "c.py", "import a\n\ndef g():\n    return a.keys()\n")
    G = _layer(code, tmp_path / "s", [".py"]).graph
    calls = {(G.nodes[u].get("label"), G.nodes[v].get("label"))
             for u, v, d in G.edges(data=True) if d.get("relation") == "calls"}
    assert ("f", "keys") not in calls
    assert ("g", "keys") in calls
