"""Python imports point at the project files they name, and stay right through
incremental updates (the graph after any edit equals a full rebuild).

Found by a reviewer on 2026-11-22: the fix that made TS/JS imports point at
files left Python as it was, every import an "external" stub, the project's
own modules included (338 of 338 in Lynx's own source).
"""
from __future__ import annotations

from pathlib import Path

from lynx.graph import GraphLayer


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _layer(code: Path, storage: Path) -> GraphLayer:
    layer = GraphLayer(storage_dir=storage / "graph", codebase_path=code, supported_extensions=[".py"])
    layer.rebuild(force=True)
    return layer


def _sigs(layer):
    nodes = {(nid, d.get("kind"), d.get("label")) for nid, d in layer.graph.nodes(data=True)}
    edges = {(u, v, d.get("relation"), d.get("confidence")) for u, v, d in layer.graph.edges(data=True)}
    return nodes, edges


def _same_as_full(layer, code: Path, tmp: Path, tag: str) -> None:
    assert _sigs(layer) == _sigs(_layer(code, tmp / f"gt_{tag}")), tag


def _imports_of(layer, file_name: str) -> dict:
    """module written -> (kind, file name) of what the import points at."""
    G = layer.graph
    src = next(n for n, d in G.nodes(data=True)
               if d.get("kind") == "file" and Path(d.get("file") or "").name == file_name)
    return {d.get("module"): (G.nodes[v].get("kind"), Path(G.nodes[v].get("file") or "").name)
            for _u, v, d in G.out_edges(src, data=True) if d.get("relation") in ("imports", "imports_from")}


def _project(code: Path) -> None:
    (code / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
    (code / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    _write(code / "src" / "app" / "__init__.py", "")
    _write(code / "src" / "app" / "engine.py", "class Engine:\n    def start(self):\n        pass\n")
    _write(code / "src" / "app" / "motor.py", "class Motor:\n    def start(self):\n        pass\n")
    _write(code / "src" / "app" / "util" / "__init__.py", "")
    _write(code / "src" / "app" / "util" / "maths.py", "def clamp(x):\n    return x\n")
    _write(code / "src" / "app" / "runner.py", (
        "import os\n"
        "import numpy as np\n"
        "from .engine import Engine\n"
        "from . import motor\n"
        "from .util.maths import clamp\n"
        "from app.config import DEBUG\n"
        "from app.tools import sweep\n"
        "\n"
        "def run(e: Engine):\n"
        "    e.start()\n"))
    _write(code / "src" / "app" / "tools" / "__init__.py", "")
    _write(code / "src" / "app" / "tools" / "sweep.py", "def sweep():\n    pass\n")
    _write(code / "src" / "app" / "config.py", "DEBUG = False\n")
    _write(code / "src" / "app" / "solo.py", (
        "from .engine import Engine\n"
        "\n"
        "def go(e: Engine):\n"
        "    e.start()\n"))


def test_project_imports_point_at_files_and_the_rest_stays_external(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    imports = _imports_of(_layer(code, tmp_path / "s"), "runner.py")
    assert imports[".engine"] == ("file", "engine.py")
    assert imports[".motor"] == ("file", "motor.py")
    assert imports[".util.maths"] == ("file", "maths.py")
    assert imports["app.config"] == ("file", "config.py")
    assert imports["app.tools.sweep"] == ("file", "sweep.py")
    assert imports["os"][0] == "external"
    assert imports["numpy"][0] == "external"


def test_a_member_call_picks_the_definition_in_the_imported_file(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    G = _layer(code, tmp_path / "s").graph
    def targets(caller):
        return {G.nodes[v].get("label") for u, v, d in G.edges(data=True)
                if d.get("relation") == "calls" and G.nodes[u].get("label") == caller}
    # Engine.start and Motor.start share the name: solo imports only Engine;
    # runner imports both modules, so nothing tells them apart.
    assert targets("go") == {"Engine.start"}
    assert targets("run") == set()


def test_re_indexing_an_imported_file_keeps_who_imports_it(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    layer = _layer(code, tmp_path / "s")
    _write(code / "src" / "app" / "engine.py", "class Engine:\n    def start(self):\n        return 2\n")
    layer.update_file(str(code / "src" / "app" / "engine.py"))
    assert _imports_of(layer, "runner.py")[".engine"] == ("file", "engine.py")
    _same_as_full(layer, code, tmp_path, "update")


def test_deleting_an_imported_file_leaves_the_import_unresolved_as_a_rebuild_would(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    layer = _layer(code, tmp_path / "s")
    for name in ("engine.py", "config.py"):
        (code / "src" / "app" / name).unlink()
        layer.remove_file(str(code / "src" / "app" / name))
    imports = _imports_of(layer, "runner.py")
    assert imports[".engine"][0] == "external"
    assert imports["app.config"][0] == "external"
    _same_as_full(layer, code, tmp_path, "delete")


def test_a_file_created_later_takes_the_imports_waiting_for_it(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    _write(code / "src" / "app" / "runner.py", "from .late import thing\nfrom app.later import other\n")
    layer = _layer(code, tmp_path / "s")
    _write(code / "src" / "app" / "late.py", "def thing():\n    pass\n")
    layer.update_file(str(code / "src" / "app" / "late.py"))
    _write(code / "src" / "app" / "later.py", "def other():\n    pass\n")
    layer.update_file(str(code / "src" / "app" / "later.py"))
    imports = _imports_of(layer, "runner.py")
    assert imports[".late"] == ("file", "late.py")
    assert imports["app.later"] == ("file", "later.py")
    _same_as_full(layer, code, tmp_path, "create")


def test_an_import_inside_a_function_is_the_files_import(tmp_path: Path):
    code = tmp_path / "code"
    _project(code)
    _write(code / "src" / "app" / "lazy.py", (
        "def boot():\n"
        "    from .engine import Engine\n"
        "    Engine().start()\n"))
    assert _imports_of(_layer(code, tmp_path / "s"), "lazy.py")[".engine"] == ("file", "engine.py")
