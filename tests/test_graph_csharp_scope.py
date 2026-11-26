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


def test_a_pattern_variable_has_the_type_it_was_tested_from(tmp_path):
    # `Panels is { } panels`, `host is var h` and a `switch` take the tested
    # property's or parameter's type, written `Docking.IHost` from a namespace
    # that reaches it as Docking: `panels.Restore()` is the host's, not the
    # one visible Restore a receiver of no known type would fall to.
    # `o is Docking.Dock { } d` names Dock and types d.
    G = _graph(tmp_path, {
        "Docking/Host.cs": ("namespace App.Docking;\npublic interface IHost { void Restore(int l); }\n"
                            "public class Dock { public void Restore(int l) { } }\n"),
        "Docs/Doc.cs": "namespace App.Docs;\npublic abstract class Doc { public abstract void Restore(byte[] s); }\n",
        "Shell/Shell.cs": ("using App.Docs;\nnamespace App.Shell;\npublic class Shell {\n"
                           "  public Docking.IHost? Panels { get; set; }\n"
                           "  public void Switch(object o, Docking.IHost host) {\n"
                           "    if (Panels is { } panels) panels.Restore(1);\n"
                           "    if (host is var h) h.Restore(2);\n"
                           "    switch (Panels) { case { } s: s.Restore(3); break; }\n"
                           "    if (o is Docking.Dock { } d) d.Restore(4);\n  }\n}\n"),
    })
    assert _calls(G, "Shell.Switch") == ["Dock.Restore", "IHost.Restore"]
    assert ("Shell.Switch", "Dock") in _uses(G)
    assert ("Shell", "IHost") in _uses(G), "a property typed Docking.IHost uses IHost"


def test_an_inherited_method_is_found_in_the_base(tmp_path):
    # Bases are resolved before calls: `base.Start()`, `BaseOnly()` and
    # `_child.BaseOnly()` all live in BaseServer.
    G = _graph(tmp_path, {
        "Net/BaseServer.cs": "namespace Net;\npublic class BaseServer { public virtual void Start() { } public void BaseOnly() { } }\n",
        "Net/ChildServer.cs": ("namespace Net;\npublic class ChildServer : BaseServer {\n"
                               "  public override void Start() { }\n  public void Boot() { base.Start(); BaseOnly(); }\n}\n"),
        "Game/Runner.cs": "using Net;\nnamespace Game;\npublic class Runner {\n  private ChildServer _srv;\n  public void Go() { _srv.BaseOnly(); }\n}\n",
    })
    assert _calls(G, "ChildServer.Boot") == ["BaseServer.BaseOnly", "BaseServer.Start"]
    assert _calls(G, "Runner.Go") == ["BaseServer.BaseOnly"]


def test_an_update_to_a_base_list_rebinds_the_inherited_calls_elsewhere(tmp_path):
    code = tmp_path / "code"
    files = {
        "Net/BaseServer.cs": "namespace Net;\npublic class BaseServer { public void BaseOnly() { } }\n",
        "Net/ChildServer.cs": "namespace Net;\npublic class ChildServer : BaseServer { }\n",
        "Game/Runner.cs": "using Net;\nnamespace Game;\npublic class Runner {\n  private ChildServer _srv;\n  public void Go() { _srv.BaseOnly(); }\n}\n",
    }
    for rel, text in files.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    assert _calls(layer.graph, "Runner.Go") == ["BaseServer.BaseOnly"]
    (code / "Net" / "ChildServer.cs").write_text("namespace Net;\npublic class ChildServer { }\n", encoding="utf-8")
    layer.update_file(str(code / "Net" / "ChildServer.cs"))
    assert _calls(layer.graph, "Runner.Go") == [], "ChildServer no longer inherits BaseOnly"


def test_an_update_drops_a_type_reference_that_became_ambiguous(tmp_path):
    code = tmp_path / "code"
    files = {
        "W1/Rain.cs": "namespace W1;\npublic class Rain { }\n",
        "Setup.cs": "using W1;\nusing W2;\nnamespace Game;\npublic class Setup { public object Make() => new Rain(); }\n",
    }
    for rel, text in files.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=code, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    uses = lambda: [(layer.graph.nodes[u].get("label"), layer.graph.nodes[v].get("label"))  # noqa: E731
                    for u, v, d in layer.graph.edges(data=True) if d.get("relation") == "uses_type"]
    assert uses() == [("Setup.Make", "Rain")]
    (code / "W2").mkdir()
    (code / "W2" / "Rain.cs").write_text("namespace W2;\npublic class Rain { }\n", encoding="utf-8")
    layer.update_file(str(code / "W2" / "Rain.cs"))
    assert uses() == [], "two visible Rain types: the reference is ambiguous and gets no edge"


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


def _uses(G) -> list:
    return sorted((G.nodes[u].get("label"), G.nodes[v].get("label"))
                  for u, v, d in G.edges(data=True) if d.get("relation") == "uses_type")


def test_every_place_a_type_is_written_is_a_use(tmp_path):
    # Attributes (`[Marker]` is MarkerAttribute), a field's generic arguments
    # and initializer, a base's generic argument, a constraint, locals,
    # casts, `as`, `catch`, arrays, `default`, tuples.
    kinds = ["MarkerAttribute", "Key", "Val", "Info", "Arg", "IRule", "Local", "Cast", "AsT", "MyEx", "Elem",
             "Def", "Left"]
    G = _graph(tmp_path, {
        "Kinds.cs": "namespace K;\n" + "".join(f"public class {k} {{ }}\n" for k in kinds) + "public class Base<T> { }\n",
        "Setup.cs": ("using K;\nnamespace Game;\n[Marker]\npublic class Setup<T> : Base<Arg> where T : IRule {\n"
                     "  private Dictionary<Key, Val> _m = new() { [1] = new Info() };\n"
                     "  public void Run(object o) {\n"
                     "    Local l = null; var c = (Cast)o; var z = o as AsT; var a = new Elem[3];\n"
                     "    var d = default(Def); (Left x, int y) t = default;\n"
                     "    try { } catch (MyEx e) { }\n  }\n}\n"),
    })
    used = _uses(G)
    assert {v for u, v in used if u == "Setup"} == {"MarkerAttribute", "Key", "Val", "Info", "Arg", "IRule"}, used
    assert {v for u, v in used if u == "Setup.Run"} == {"Local", "Cast", "AsT", "Elem", "Def", "Left", "MyEx"}, used


def test_two_nested_types_of_one_name_are_each_their_outers(tmp_path):
    G = _graph(tmp_path, {
        "Runners.cs": ("namespace Work;\npublic class SharedRunner {\n  public object Add() => new Registration();\n"
                       "  private sealed class Registration { }\n}\n"
                       "public class StepRunner {\n  public object Add() => new Registration();\n"
                       "  private sealed class Registration { }\n}\n"),
    })
    assert _uses(G) == [("SharedRunner.Add", "SharedRunner.Registration"), ("StepRunner.Add", "StepRunner.Registration")]


def test_a_nested_type_named_in_a_field_of_its_outer_type_is_used(tmp_path):
    # `Dictionary<int, MsgInfo>` named only Dictionary, which is not the
    # codebase's: MsgInfo looked unused.
    G = _graph(tmp_path, {
        "Codec.cs": ("namespace Net;\npublic static class Codec {\n  private record struct MsgInfo(int Length);\n"
                     "  private static readonly Dictionary<int, MsgInfo> Info = new();\n}\n"),
    })
    assert _uses(G) == [("Codec", "Codec.MsgInfo")]


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


# A file with a statement left half written, as it is while someone types it:
# the grammar puts the whole class into an ERROR node, as loose tokens
# (`public`, `class`, `Box`, the base list, `{`, the members, `}`). A newer
# grammar did the same to a whole 2,400-line class of a real library over a
# valid line, and the class and everything it owned left the graph.
HALF_WRITTEN = (
    "namespace A;\n\n"
    "public class Base { }\n\n"
    "public sealed class Box : Base\n{\n"
    "    private int _n;\n\n"
    "    public Box(int n) { _n = n; }\n\n"
    "    public void Bad() { if (x) else }\n\n"
    "    public int Count() => _n;\n}\n"
)


def test_a_class_the_grammar_left_in_pieces_is_put_back_together(tmp_path):
    # (`Bad` swallows `Count` and the closing brace: that much is the grammar's.)
    G = _graph(tmp_path, {"Box.cs": HALF_WRITTEN})
    box = [n for n, d in G.nodes(data=True) if d.get("label") == "Box" and d.get("kind") == "class"]
    assert box, "the class is in the graph"
    owned = sorted(G.nodes[v]["label"] for _, v, d in G.out_edges(box[0], data=True) if d.get("relation") == "contains")
    assert owned == ["Box.Bad", "Box.Box"]
    assert _bases(G, "Box") == [("Base", "resolved")]
