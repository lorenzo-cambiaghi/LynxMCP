"""Known gaps, pinned as strict expected failures (TASKS_2026-11-24.md).

Each test describes how Lynx should behave and fails today. `strict=True`
turns a fix into a failure of the suite (XPASS), so whoever fixes a gap has
to drop its marker here, and from then on the test guards the fix.

  T1  a map of a module's conventions: the base types implemented there and
      where implementations are registered
  T2  the startup instructions send "add a type" to the graph
  T3  a Python base class is resolved through the file's imports, not linked
      to every class with that name
  T6  every script suite (the files conftest keeps out of pytest) runs in CI
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _graph(tmp_path: Path, files: dict):
    code = tmp_path / "code"
    for rel, text in files.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code, supported_extensions=[".py"])
    layer.rebuild(force=True)
    return layer._graph


# Two classes named Base. mid.py imports the one in util.py, so Mid's base
# is util.Base and nothing else (E35: on Django 2,085 of 3,300 inheritance
# edges were linked to every namesake).
NAMESAKES = {
    "util.py": "class Base:\n    pass\n",
    "other.py": "class Base:\n    pass\n\nclass OtherChild(Base):\n    pass\n",
    "mid.py": "from util import Base\n\nclass Mid(Base):\n    pass\n",
}


def test_t3_base_resolved_through_the_import(tmp_path):
    G = _graph(tmp_path, NAMESAKES)
    bases = [(G.nodes[v].get("file") or "", d.get("confidence")) for u, v, d in G.edges(data=True)
             if d.get("relation") == "inherits" and (G.nodes[u].get("label") or "").endswith("Mid")]
    assert len(bases) == 1, bases
    assert bases[0][0].replace("\\", "/").endswith("/util.py") and bases[0][1] != "ambiguous", bases


def test_t3_subclasses_keep_namesake_hierarchies_apart(tmp_path):
    from lynx.graph.query import get_subclasses
    G = _graph(tmp_path, NAMESAKES)
    for edge in get_subclasses(G, "Base", depth=3):
        child = (edge["source"].get("label") or "").split(".")[-1]
        base_file = (edge["target"].get("file") or "").replace("\\", "/")
        if child == "Mid":
            assert base_file.endswith("/util.py"), "Mid hangs under other.Base too"


# A module with an extension point: three systems implement GameSystem and
# are registered in one place.
CONVENTIONS = {
    "engine/system.py": "class GameSystem:\n    def update(self):\n        pass\n",
    "engine/registry.py": (
        "from engine.systems.rain import Rain\nfrom engine.systems.wind import Wind\n"
        "from engine.systems.fog import Fog\n\n"
        "def make_systems(world):\n    world.add(Rain())\n    world.add(Wind())\n    world.add(Fog())\n"
    ),
    "engine/systems/rain.py": "from engine.system import GameSystem\n\nclass Rain(GameSystem):\n    pass\n",
    "engine/systems/wind.py": "from engine.system import GameSystem\n\nclass Wind(GameSystem):\n    pass\n",
    "engine/systems/fog.py": "from engine.system import GameSystem\n\nclass Fog(GameSystem):\n    pass\n",
}


@pytest.mark.xfail(strict=True, reason="T1: no map of a module's conventions yet")
def test_t1_conventions_name_the_extension_point_and_where_it_is_registered(tmp_path):
    from lynx.graph.query import module_conventions  # does not exist yet
    G = _graph(tmp_path, CONVENTIONS)
    conv = module_conventions(G, "engine/systems")
    bases = {b["base"] for b in conv["bases"]}
    assert "GameSystem" in bases
    sites = {s["file"].replace("\\", "/").split("/code/")[-1] for s in conv["registered_in"]}
    assert "engine/registry.py" in sites


class _Backend:
    type_name = "codebase"

    def __init__(self):
        self.graph = object()
        self.source_config = {"path": "/x", "git_integration": {"enabled": False}}


class _Manager:
    def __init__(self):
        self.backends = {"lib": _Backend()}


def test_t2_instructions_send_a_new_type_to_the_graph():
    from lynx.server import _build_instructions
    text = _build_instructions(_Manager())
    assert re.search(r"subclasses[^.]*depth", text), "no route from 'add a type' to graph_query subclasses"


def test_t6_every_script_suite_runs_in_ci():
    """One runner finds the script suites with the probe conftest.py uses and
    runs every one but the three that open the real config; CI calls it. A
    new suite is picked up without touching the workflow."""
    workflow = (REPO / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    runner = REPO / "tests" / "run_script_suites.py"
    assert runner.exists(), "no runner for the script suites"
    assert "run_script_suites.py" in workflow, "CI does not call the runner"
    text = runner.read_text(encoding="utf-8")
    for real_config in ("test_watch.py", "test_deep_search.py", "test_search_batch.py"):
        assert real_config in text, f"the runner must leave out {real_config}"


# The same gap in C#: two classes named Base in two namespaces; Mid's file
# uses one of them. A 1.4-million-line C# library is where T1 has to work.
CS_NAMESAKES = {
    "Util/Base.cs": "namespace Util { public class Base { } }\n",
    "Other/Base.cs": "namespace Other { public class Base { } }\n",
    "Mid.cs": "using Util;\nnamespace App { public class Mid : Base { } }\n",
}


def test_t3_csharp_base_resolved_through_using(tmp_path):
    code = tmp_path / "code"
    for rel, text in CS_NAMESAKES.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    G = layer._graph
    bases = [(G.nodes[v].get("file") or "", d.get("confidence")) for u, v, d in G.edges(data=True)
             if d.get("relation") == "inherits" and (G.nodes[u].get("label") or "").endswith("Mid")]
    assert len(bases) == 1, bases
    assert bases[0][0].replace("\\", "/").endswith("/Util/Base.cs") and bases[0][1] != "ambiguous", bases
