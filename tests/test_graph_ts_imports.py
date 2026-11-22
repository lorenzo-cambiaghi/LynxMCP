"""TypeScript relative imports point at the files they name, and stay right
through incremental updates (the graph after any edit equals a full rebuild).

Found on The Lost Storm (2026-11-22): all 5,663 project imports were
"external" stubs, 5,534 of them files in the index, so the graph could not say
which file imports which.
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layer(code: Path, storage: Path) -> GraphLayer:
    layer = GraphLayer(storage_dir=storage / "graph", codebase_path=code, supported_extensions=[".ts"])
    layer.rebuild(force=True)
    return layer


def _sigs(layer):
    nodes = {(nid, d.get("kind"), d.get("label")) for nid, d in layer.graph.nodes(data=True)}
    edges = {(u, v, d.get("relation"), d.get("confidence")) for u, v, d in layer.graph.edges(data=True)}
    return nodes, edges


def _same_as_full(layer, code: Path, tmp: Path, tag: str) -> None:
    gt = _layer(code, tmp / f"gt_{tag}")
    assert _sigs(layer) == _sigs(gt), tag


def _imports_of(layer, file_label: str) -> dict:
    G = layer.graph
    src = next(n for n, d in G.nodes(data=True) if d.get("kind") == "file" and d.get("label") == file_label)
    return {d.get("module"): G.nodes[v].get("kind") for _u, v, d in G.out_edges(src, data=True)
            if d.get("relation") == "imports_from"}


def _project(code: Path) -> None:
    _write(code / "src" / "pick.ts", "export function pickActor(): number { return 1; }\n")
    _write(code / "src" / "render" / "overlays.ts", "export function show(): void {}\n")
    _write(code / "sim" / "index.ts", "export const x = 1;\n")
    _write(code / "src" / "input.ts", (
        "import { pickActor } from './pick.ts';\n"
        "import { show } from './render/overlays.js';\n"
        "import { x } from '../sim';\n"
        "import { Vector3 } from 'three';\n"
        "export function createInput(): void { pickActor(); show(); }\n"))


def test_relative_imports_point_at_files_and_packages_stay_external(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    layer = _layer(code, tmp_path / "s")
    assert _imports_of(layer, "src.input") == {
        "./pick.ts": "file", "./render/overlays.js": "file", "../sim": "file", "three": "external"}


def test_re_indexing_an_imported_file_keeps_who_imports_it(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    layer = _layer(code, tmp_path / "s")
    _write(code / "src" / "pick.ts", "export function pickActor(): number { return 2; }\n")
    layer.update_file(str(code / "src" / "pick.ts"))
    assert _imports_of(layer, "src.input")["./pick.ts"] == "file"
    _same_as_full(layer, code, tmp_path, "update")


def test_deleting_an_imported_file_leaves_the_import_unresolved_as_a_rebuild_would(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    layer = _layer(code, tmp_path / "s")
    (code / "src" / "pick.ts").unlink()
    layer.remove_file(str(code / "src" / "pick.ts"))
    assert _imports_of(layer, "src.input")["./pick.ts"] == "external"
    _same_as_full(layer, code, tmp_path, "remove")


def test_a_file_created_later_takes_the_imports_waiting_for_it(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    (code / "src" / "pick.ts").unlink()
    layer = _layer(code, tmp_path / "s")
    assert _imports_of(layer, "src.input")["./pick.ts"] == "external"
    _write(code / "src" / "pick.ts", "export function pickActor(): number { return 1; }\n")
    layer.update_file(str(code / "src" / "pick.ts"))
    assert _imports_of(layer, "src.input")["./pick.ts"] == "file"
    _same_as_full(layer, code, tmp_path, "create")


def _call_targets(layer, caller_label: str) -> dict:
    G = layer.graph
    src = next(n for n, d in G.nodes(data=True) if d.get("label") == caller_label and d.get("kind") == "function")
    return {G.nodes[v].get("file").replace("\\", "/").rsplit("/", 1)[-1]: d.get("confidence")
            for _u, v, d in G.out_edges(src, data=True) if d.get("relation") == "calls"}


def test_a_name_defined_twice_resolves_to_the_file_the_caller_imports(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "a.ts", "export function helper(): number { return 1; }\n")
    _write(code / "b.ts", "export function helper(): number { return 2; }\n")
    _write(code / "main.ts", "import { helper } from './b.ts';\nexport function run(): number { return helper(); }\n")
    _write(code / "other.ts", "export function go(): number { return helper(); }\n")
    layer = _layer(code, tmp_path / "s")
    assert _call_targets(layer, "run") == {"b.ts": "resolved"}
    assert _call_targets(layer, "go") == {"a.ts": "ambiguous", "b.ts": "ambiguous"}, "nothing imported: every candidate"

    _write(code / "main.ts", "import { helper } from './a.ts';\nexport function run(): number { return helper(); }\n")
    layer.update_file(str(code / "main.ts"))
    assert _call_targets(layer, "run") == {"a.ts": "resolved"}
    _same_as_full(layer, code, tmp_path, "import_switch")


def test_a_member_call_lands_only_on_a_method_of_a_class_the_caller_can_see(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "util.ts", "export function keys(): string[] { return []; }\n")
    _write(code / "heap.ts", "export class Heap { push(x: number): void {} }\n")
    _write(code / "queue.ts", "export class Queue { push(x: number): void {} }\n")
    _write(code / "main.ts", (
        "import { Heap } from './heap.ts';\n"
        "export function run(o: object, h: Heap, a: number[]): void { Object.keys(o); h.push(1); a.push(2); }\n"))
    _write(code / "other.ts", "export function go(a: number[]): void { a.push(1); }\n")
    layer = _layer(code, tmp_path / "s")
    assert _call_targets(layer, "run") == {"heap.ts": "resolved"}
    assert _call_targets(layer, "go") == {}, "an array's push is not a project method"


def test_a_call_to_a_local_closure_is_not_a_call_to_another_file(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "rigs.ts", "export function lift(): number { return 1; }\n")
    _write(code / "input.ts", (
        "export function createInput(): void {\n"
        "  const lift = (b: string): void => { release(b); };\n"
        "  lift('a');\n"
        "}\n"
        "export function release(b: string): void {}\n"))
    layer = _layer(code, tmp_path / "s")
    assert _call_targets(layer, "createInput") == {"input.ts": "extracted"}, "release, called inside the closure"
