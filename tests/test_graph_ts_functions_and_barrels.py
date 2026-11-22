"""TS/JS functions written as values, barrels and tsconfig aliases.

Found by a reviewer on 2026-11-22: `export const f = () => ...` had no node,
so a codebase written with arrow functions had no callers or callees for
them; `export * from "./y"` was not an import, so a file importing a barrel
did not reach what it re-exports; `@/x` from tsconfig `paths` was a package.
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layer(code: Path, storage: Path, exts=(".ts",)) -> GraphLayer:
    layer = GraphLayer(storage_dir=storage / "graph", codebase_path=code, supported_extensions=list(exts))
    layer.rebuild(force=True)
    return layer


def _sigs(layer):
    nodes = {(nid, d.get("kind"), d.get("label")) for nid, d in layer.graph.nodes(data=True)}
    edges = {(u, v, d.get("relation"), d.get("confidence")) for u, v, d in layer.graph.edges(data=True)}
    return nodes, edges


def _calls(layer) -> set:
    G = layer.graph
    return {(G.nodes[u].get("label"), G.nodes[v].get("label"))
            for u, v, d in G.edges(data=True) if d.get("relation") == "calls"}


def _import_kinds(layer, file_name: str) -> dict:
    G = layer.graph
    src = next(n for n, d in G.nodes(data=True)
               if d.get("kind") == "file" and Path(d.get("file") or "").name == file_name)
    return {d.get("module"): (G.nodes[v].get("kind"), Path(G.nodes[v].get("file") or "").name)
            for _u, v, d in G.out_edges(src, data=True) if d.get("relation") == "imports_from"}


def test_functions_written_as_values_are_nodes_with_their_calls(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "pick.ts", "export const pickActor = (x: number) => x + 1;\n")
    _write(code / "main.ts", (
        "import { pickActor } from './pick';\n"
        "export const createInput = () => { pickActor(1); };\n"
        "const helper = function () { createInput(); };\n"
        "export class Rig { press = () => { helper(); }; size = 2; }\n"))
    layer = _layer(code, tmp_path / "s")
    labels = {d.get("label") for _n, d in layer.graph.nodes(data=True) if d.get("kind") == "function"}
    assert {"pickActor", "createInput", "helper", "Rig.press"} <= labels
    assert "Rig.size" not in labels
    calls = _calls(layer)
    assert ("createInput", "pickActor") in calls
    assert ("helper", "createInput") in calls
    assert ("Rig.press", "helper") in calls


def test_a_member_call_reaches_a_class_through_a_barrel(tmp_path: Path):
    code = tmp_path / "code"
    _write(code / "input" / "heap.ts", "export class MinHeap { push(x: number) {} }\n")
    _write(code / "input" / "index.ts", "export { MinHeap } from './heap';\nexport * from './heap';\n")
    _write(code / "main.ts", (
        "import { MinHeap } from './input';\n"
        "export function go(h: MinHeap) { h.push(1); }\n"))
    layer = _layer(code, tmp_path / "s")
    assert _import_kinds(layer, "index.ts")["./heap"] == ("file", "heap.ts")
    assert ("go", "MinHeap.push") in _calls(layer)


def _aliased(code: Path) -> None:
    _write(code / "tsconfig.json", (
        "{\n  // comments and trailing commas, as tsconfig allows\n"
        "  \"compilerOptions\": {\n    \"baseUrl\": \".\",\n"
        "    \"paths\": { \"@/*\": [\"src/*\"], \"~lib\": [\"lib/index.ts\"], },\n  },\n}\n"))
    _write(code / "src" / "render" / "overlays.ts", "export function show() {}\n")
    _write(code / "lib" / "index.ts", "export function util() {}\n")
    _write(code / "shared" / "time.ts", "export function now() { return 0; }\n")
    _write(code / "src" / "main.ts", (
        "import { show } from '@/render/overlays';\n"
        "import { util } from '~lib';\n"
        "import { now } from 'shared/time';\n"
        "import * as THREE from 'three';\n"
        "export function start() { show(); util(); now(); }\n"))


def test_tsconfig_paths_and_base_url_point_at_files_and_packages_stay_external(tmp_path: Path):
    code = tmp_path / "code"
    _aliased(code)
    imports = _import_kinds(_layer(code, tmp_path / "s"), "main.ts")
    assert imports["@/render/overlays"] == ("file", "overlays.ts")
    assert imports["~lib"] == ("file", "index.ts")
    assert imports["shared/time"] == ("file", "time.ts")
    assert imports["three"][0] == "external"


def test_aliased_imports_stay_right_through_updates(tmp_path: Path):
    code = tmp_path / "code"
    _aliased(code)
    layer = _layer(code, tmp_path / "s")
    target = code / "src" / "render" / "overlays.ts"
    target.unlink()
    layer.remove_file(str(target))
    assert _import_kinds(layer, "main.ts")["@/render/overlays"][0] == "external"
    assert _sigs(layer) == _sigs(_layer(code, tmp_path / "gt_delete"))
    _write(target, "export function show() { return 1; }\n")
    layer.update_file(str(target))
    assert _import_kinds(layer, "main.ts")["@/render/overlays"] == ("file", "overlays.ts")
    assert _sigs(layer) == _sigs(_layer(code, tmp_path / "gt_create"))
