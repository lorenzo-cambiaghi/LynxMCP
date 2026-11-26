"""The SOLID audit (lynx.audit.solid): what each check catches and the fence
it puts around a finding. Thresholds are relative to the codebase and are
not pinned here; the mechanisms are."""
from __future__ import annotations

from pathlib import Path

CODE = {
    "src/Pkg/Pkg.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
    "src/Pkg/ITransport.cs": "namespace Pkg;\npublic interface ITransport { void ConnectAsync(); }\n",
    "src/Pkg/Server.cs": (
        "namespace Pkg;\npublic class Server : ITransport {\n"
        "  private static int _count;\n"
        "  [ThreadStatic] private static int t_scratch;\n"
        "  private static readonly int _limit = 4;\n"
        "  private static readonly List<int> _seen = new();\n"
        "  private static readonly Dictionary<int, string> _names = new() { [1] = \"one\" };\n"
        "  public void Send() { _seen.Add(_names.Count); }\n"
        "  public void ConnectAsync() => throw new NotSupportedException();\n"
        "  private static void NotHere() => throw new NotSupportedException();\n"
        "  void Route(Kind kind) { switch (kind) { case Kind.A: break; case Kind.B: break; case Kind.C: break; case Kind.D: break; } }\n"
        "}\n"),
    "src/Pkg/Router.cs": (
        "namespace Pkg;\npublic class Router {\n"
        "  void One(Kind kind) { switch (kind) { case Kind.A: break; case Kind.B: break; case Kind.C: break; case Kind.D: break; } }\n"
        "  void Two(Kind kind) { switch (kind) { case Kind.A: break; case Kind.B: break; case Kind.C: break; case Kind.D: break; } }\n"
        "}\n"),
    "src/Pkg/Parts.cs": ("namespace Pkg;\ninternal class Helper { }\ninternal class Dead { }\n"
                         "public class Maker { public object Make() => new Helper(); }\n"),
    "src/App/App.csproj": ('<Project Sdk="Microsoft.NET.Sdk"><ItemGroup>'
                           '<ProjectReference Include="../Pkg/Pkg.csproj" /></ItemGroup></Project>\n'),
    "src/App/Game.cs": "using Pkg;\nnamespace App;\npublic class Game {\n  private Server _server;\n  public void Tick() { _server.Send(); }\n}\n",
    "tests/Pkg.Tests/Pkg.Tests.csproj": ('<Project Sdk="Microsoft.NET.Sdk"><ItemGroup>'
                                         '<ProjectReference Include="../../src/Pkg/Pkg.csproj" /></ItemGroup></Project>\n'),
    "tests/Pkg.Tests/ServerTests.cs": "using Pkg;\nnamespace Pkg.Tests;\npublic class ServerTests {\n  public void Sends() { var s = new Server(); s.Send(); }\n}\n",
}


def _audit(tmp_path: Path):
    root = tmp_path / "code"
    for rel, text in CODE.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    from lynx.audit.solid import audit
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    findings, rel = audit(layer.graph, root, "Pkg")
    return findings, rel


def test_the_checks_name_what_breaks_and_leave_the_harmless_alone(tmp_path):
    findings, rel = _audit(tmp_path)
    assert rel == "src/Pkg"
    titles = [f.title for f in findings]
    assert "Server._count is mutable global state" in titles
    assert "Server._seen is a shared mutable collection" in titles      # Send adds to it
    assert not any("t_scratch" in t or "_limit" in t for t in titles), titles
    assert not any("_names" in t for t in titles), "a private table nothing writes after its initializer"
    assert "Server.ConnectAsync only throws NotImplemented/NotSupported" in titles
    assert not any("NotHere" in t for t in titles), "a private helper that throws is no contract"
    assert "Dead is used by nothing" in titles                # never named anywhere
    assert "Helper is used by nothing" not in titles          # `new Helper()` uses it
    switch = [f for f in findings if f.principle == "O"]
    assert len(switch) == 1 and "3 places" in switch[0].title, titles


def test_a_finding_carries_the_fence_of_its_refactor(tmp_path):
    findings, _ = _audit(tmp_path)
    fence = next(f for f in findings if f.title.startswith("Server._count")).fence
    assert fence["files"] == ["src/Pkg/Server.cs"]
    assert dict(fence["contract"]).get("Send") == 1          # Game.Tick calls it from App
    assert fence["outside_projects"] == ["App"]
    assert fence["build"] == ["App", "Pkg.Tests"]
    assert fence["tests"] == ["tests/Pkg.Tests/ServerTests.cs"]


LONG = "\n".join(f"    int x{i} = {i};" for i in range(120))
EDGES = {
    "src/Pkg/Pkg.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
    # A partial class: the long method lives in the second file.
    "src/Pkg/NetServer.A.cs": "namespace Pkg;\npublic partial class NetServer {\n" + "\n" * 120 + "}\n",
    "src/Pkg/NetServer.B.cs": "namespace Pkg;\npublic partial class NetServer {\n  void BigWork() {\n" + LONG + "\n  }\n}\n",
    # Two switches inside a nested type, counted once each.
    "src/Pkg/Outer.cs": ("namespace Pkg;\npublic class Outer {\n  public class Inner {\n"
                         "    void A(int kind) { switch (kind) { case 1: break; case 2: break; case 3: break; case 4: break; } }\n"
                         "    void B(int kind) { switch (kind) { case 1: break; case 2: break; case 3: break; case 4: break; } }\n"
                         "  }\n}\n"),
    # Many short methods, so the codebase's own norm stays small.
    "src/Pkg/Small.cs": "namespace Pkg;\npublic class Small {\n" + "".join(f"  public void M{i}() {{ }}\n" for i in range(200)) + "}\n",
    # Public types nothing in the codebase uses are API, not dead code.
    "src/Pkg/Api.cs": ("namespace Pkg;\n[System.Serializable]\npublic class AttributedPublic { }\n"
                       "partial class SplitPublic { }\npublic partial class SplitPublic { }\n"),
}


def test_partial_and_nested_types_are_read_as_one_type_each(tmp_path):
    root = tmp_path / "code"
    for rel, text in EDGES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    from lynx.audit.solid import audit
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    findings, _ = audit(layer.graph, root, "Pkg")
    long = [f for f in findings if f.title == "NetServer.BigWork is long"]
    assert long and long[0].where == "src/Pkg/NetServer.B.cs:3", [(f.title, f.where) for f in findings]
    assert not [f for f in findings if f.principle == "O"], "a nested type's switches counted twice"
    unused = {f.title for f in findings if f.title.endswith("is used by nothing")}
    assert not unused & {"AttributedPublic is used by nothing", "SplitPublic is used by nothing"}, unused


USED = {
    "src/Pkg/Pkg.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
    # Named only by an attribute.
    "src/Pkg/Tagged.cs": ("namespace Pkg;\n[Marker]\npublic class Tagged { }\n"
                          "internal sealed class MarkerAttribute : System.Attribute { }\n"),
    # Named only in a static field's initializer.
    "src/Pkg/Json.cs": ("namespace Pkg;\npublic class Json {\n"
                        "  private static readonly Options O = new Options { Converters = { new Conv() } };\n"
                        "  private sealed class Conv { }\n}\npublic class Options { public object Converters; }\n"),
    # Each runner makes its own Registration.
    "src/Pkg/Runners.cs": ("namespace Pkg;\npublic class SharedRunner {\n  public object Add() => new Registration();\n"
                           "  private sealed class Registration { }\n}\n"
                           "public class StepRunner {\n  public object Add() => new Registration();\n"
                           "  private sealed class Registration { }\n}\n"),
    # Named only by its outer type, as a generic argument of a field.
    "src/Pkg/Codec.cs": ("namespace Pkg;\npublic static class Codec {\n  private readonly record struct MsgInfo(int Length);\n"
                         "  private static readonly System.Collections.Generic.Dictionary<int, MsgInfo> Info = new();\n"
                         "  public static int Length(int id) => Info[id].Length;\n}\n"),
}


def test_a_type_named_by_an_attribute_an_initializer_or_its_outer_type_is_used(tmp_path):
    root = tmp_path / "code"
    for rel, text in USED.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    from lynx.audit.solid import audit
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    findings, _ = audit(layer.graph, root, "Pkg")
    unused = sorted(f.title for f in findings if f.title.endswith("is used by nothing"))
    assert unused == [], unused


def test_method_names_point_to_the_jobs_of_a_type():
    from lynx.audit.solid import name_areas
    names = ["HandleFreshHandshake", "HandleResumeHandshake", "FailHandshake",
             "BroadcastSnapshot", "SendPeerSetSnapshot", "SnapshotOf", "Poll"]
    areas = dict(name_areas(names))
    assert areas["handshake"] == ["FailHandshake", "HandleFreshHandshake", "HandleResumeHandshake"]
    assert areas["snapshot"] == ["BroadcastSnapshot", "SendPeerSetSnapshot", "SnapshotOf"]
    assert "poll" not in areas


def test_a_class_the_grammar_left_in_pieces_is_read(tmp_path):
    """A statement left half written puts the whole class into an ERROR node:
    the type, its fields and its methods are read all the same."""
    from test_graph_csharp_scope import HALF_WRITTEN
    from lynx.audit.facts import read_file
    p = tmp_path / "Box.cs"
    p.write_text(HALF_WRITTEN, encoding="utf-8")
    ns, _, types = read_file(p, "Box.cs")
    box = next((t for t in types if t.name == "Box"), None)
    assert box is not None and ns == "A"
    assert box.bases == ["Base"] and [f.name for f in box.fields] == ["_n"]
    assert {m.name for m in box.methods} == {"Box", "Bad"}


def test_a_clean_package_says_so_in_a_sentence():
    from lynx.audit.solid import render
    head = render("src/Clean", [])
    assert "0 findings. Ranked" not in head and "findings: ." not in head
    assert "No findings in src/Clean." in head


def test_a_static_property_with_a_setter_is_global_state_too(tmp_path, monkeypatch):
    # Seen on a real library: a `public static bool { get; private set; }` set
    # by one call and read by the next carried one caller's state into
    # another's, while the audit flagged only the locked set beside it.
    monkeypatch.setitem(CODE, "src/Pkg/Flags.cs", (
        "namespace Pkg;\npublic class Flags {\n"
        "  public static bool Unwritable { get; private set; }\n"
        "  public static int Limit { get; } = 3;\n"
        "  public static int Twice => Limit * 2;\n"
        "  public bool Instance { get; set; }\n"
        "  public static void Fail() { Unwritable = true; }\n"
        "}\n"))
    findings, _ = _audit(tmp_path)
    by_title = {f.title: f for f in findings}
    assert "Flags.Unwritable is mutable global state" in by_title, list(by_title)
    assert "setter" in by_title["Flags.Unwritable is mutable global state"].evidence
    assert not any(n in t for t in by_title for n in ("Limit", "Twice", "Instance")), list(by_title)


def test_a_static_built_on_first_read_is_a_constant_not_shared_state(tmp_path, monkeypatch):
    # Seen on a real library: four `static T? _x; ... _x ??= Build()` caches
    # flagged as mutable global state. Written only by `??=`, each holds the same
    # value for the whole process; one also assigned plainly stays a finding.
    monkeypatch.setitem(CODE, "src/Pkg/Lazy.cs", (
        "namespace Pkg;\npublic class Lazy {\n"
        "  private static byte[]? _picture;\n"
        "  private static string? _label;\n"
        "  public static byte[] Picture => _picture ??= new byte[4];\n"
        "  public static string Label => _label ??= \"x\";\n"
        "  public static void Reset() { _label = null; }\n"
        "}\n"))
    titles = [f.title for f in _audit(tmp_path)[0]]
    assert not any("_picture" in t for t in titles), titles
    assert "Lazy._label is mutable global state" in titles
