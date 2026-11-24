"""C# base types resolved by C#'s own lookup rules (TASKS_2026-11-24.md, R1).

On a 1.4-million-line C# library 77% of the `inherits` edges were ambiguous:
a base was linked to every type, method and namespace sharing its name
(7,400 edges pointed at methods). These pin the rules that took it to 0.2%.
"""
from __future__ import annotations

from pathlib import Path


def _graph(tmp_path: Path, files: dict):
    code = tmp_path / "code"
    for rel, text in files.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    return layer._graph


def _bases(G, child: str) -> list:
    return sorted((G.nodes[v].get("label"), d.get("confidence")) for u, v, d in G.edges(data=True)
                  if d.get("relation") == "inherits" and G.nodes[u].get("label") == child)


def test_a_base_is_a_type_not_a_method_or_a_namespace_of_that_name(tmp_path):
    G = _graph(tmp_path, {
        "Shapes/Shape.cs": "namespace Shapes;\npublic class Shape { }\n",
        "Tools/Painter.cs": "namespace Tools;\npublic class Painter { public void Shape() { } }\n",
        "Circle.cs": "using Shapes;\nnamespace App;\npublic class Circle : Shape { }\n",
    })
    assert _bases(G, "Circle") == [("Shape", "resolved")]


def test_a_base_outside_the_codebase_links_to_no_namesake(tmp_path):
    # `Window` is Avalonia's: the codebase's own Window lives in a namespace
    # the file does not import, so no edge at all.
    G = _graph(tmp_path, {
        "Tests/Window.cs": "namespace Graphics.Tests;\npublic class Window { }\n",
        "About.cs": "using Avalonia.Controls;\nnamespace Editor;\npublic class AboutWindow : Window { }\n",
    })
    assert _bases(G, "AboutWindow") == []


def test_project_usings_and_aliases_make_a_base_visible(tmp_path):
    # A csproj `<Using Include="Docs" />` holds for every file of the
    # project; `using Step = Steps.Step;` names Step's namespace.
    G = _graph(tmp_path, {
        "Docs/IUndo.cs": "namespace Docs;\npublic interface IUndo { }\n",
        "Steps/Step.cs": "namespace Steps;\npublic class Step { }\n",
        "App/App.csproj": '<Project Sdk="Microsoft.NET.Sdk">\n  <ItemGroup>\n    <Using Include="Docs" />\n  </ItemGroup>\n</Project>\n',
        "App/Edit.cs": "namespace App;\npublic class Edit : IUndo { }\n",
        "App/Move.cs": "using Step = Steps.Step;\nnamespace App;\npublic class Move : Step { }\n",
    })
    assert _bases(G, "Edit") == [("IUndo", "resolved")]
    assert _bases(G, "Move") == [("Step", "resolved")]


def test_partial_declarations_are_one_type(tmp_path):
    G = _graph(tmp_path, {
        "Loop.A.cs": "namespace Engine;\npublic partial class Loop { }\n",
        "Loop.B.cs": "namespace Engine;\npublic partial class Loop { }\n",
        "Game.cs": "namespace Engine.Games;\npublic class Game : Loop { }\n",
    })
    bases = _bases(G, "Game")
    assert bases and {c for _, c in bases} == {"resolved"}, bases


def _calls(G, caller: str) -> list:
    return sorted(G.nodes[v].get("label") for u, v, d in G.edges(data=True)
                  if d.get("relation") == "calls" and G.nodes[u].get("label") == caller)


CALLS = {
    "Net/Server.cs": ("namespace Net;\npublic class Server {\n  public void Poll() { }\n"
                      "  public void Run() { Poll(); }\n}\n"),
    "Other/Pump.cs": "namespace Other;\npublic class Pump { public void Poll() { } public void Empty() { } }\n",
    "Engine/Loop.cs": ("namespace Engine;\npublic class Systems { public void Add<T>() { } }\n"
                       "public static class LoopExtensions { public static void AddBullets(this Systems s) { } }\n"),
    "Game.cs": ("using Net;\nusing Engine;\nnamespace Game;\npublic class Match {\n"
                "  private Server _server;\n"
                "  public void Tick(Systems systems) {\n"
                "    _server.Poll();\n    var other = new Server(); other.Run();\n"
                "    systems.Add<Match>();\n    systems.AddBullets();\n    Assert.Empty(systems);\n  }\n}\n"),
}


def test_a_csharp_call_binds_by_the_receivers_type_not_by_name(tmp_path):
    # `Poll()` in Server is Server's own, not Pump's namesake; `_server.Poll()`
    # and `other.Run()` go to the declared types; Assert is xUnit's, so its
    # `Empty` is not the codebase's Pump.Empty.
    G = _graph(tmp_path, CALLS)
    assert _calls(G, "Server.Run") == ["Server.Poll"]
    assert _calls(G, "Match.Tick") == ["LoopExtensions.AddBullets", "Server.Poll", "Server.Run", "Systems.Add"]


def test_a_type_named_by_new_typeof_generics_and_signatures_is_used(tmp_path):
    # A system registered as `Add<Rain>()` or `typeof(Fog)` is used even if
    # nothing calls its methods: those references are `uses_type` edges.
    G = _graph(tmp_path, {
        "Weather/Kinds.cs": ("namespace Weather;\npublic class Rain { }\npublic class Fog { }\npublic class Wind { }\n"
                             "public class Hail { }\npublic class Snow { }\n"),
        "Other/Rain.cs": "namespace Other;\npublic class Rain { }\n",
        "Setup.cs": ("using Weather;\nnamespace Game;\npublic class Setup {\n  private Snow _snow;\n"
                     "  public Hail Make(Wind w) {\n    systems.Add<Rain>();\n    var t = typeof(Fog);\n"
                     "    return new Hail();\n  }\n}\n"),
    })
    used = sorted((G.nodes[u].get("label"), G.nodes[v].get("label"), (G.nodes[v].get("file") or "").replace("\\", "/").split("/")[-2])
                  for u, v, d in G.edges(data=True) if d.get("relation") == "uses_type")
    assert used == [("Setup", "Snow", "Weather"), ("Setup.Make", "Fog", "Weather"), ("Setup.Make", "Hail", "Weather"),
                    ("Setup.Make", "Rain", "Weather"), ("Setup.Make", "Wind", "Weather")], used


def test_a_field_of_the_base_type_does_not_hide_the_inheritance(tmp_path):
    G = _graph(tmp_path, {
        "Node.cs": "namespace Tree;\npublic class Node { }\npublic class Group : Node { private Node _first; }\n",
    })
    assert _bases(G, "Group") == [("Node", "resolved")]


def test_an_extension_block_holds_methods_not_one_long_function(tmp_path):
    G = _graph(tmp_path, {
        "Ext.cs": ("namespace Util;\npublic static class SizeExtensions {\n"
                   "  extension(int value) {\n    public int Twice() => value * 2;\n    public int Half() => value / 2;\n  }\n}\n"),
    })
    labels = {d.get("label") for _, d in G.nodes(data=True) if d.get("kind") == "function"}
    assert "SizeExtensions.Twice" in labels and "SizeExtensions.Half" in labels, labels
    assert "SizeExtensions.extension" not in labels
