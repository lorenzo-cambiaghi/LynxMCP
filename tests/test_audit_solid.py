"""The SOLID audit (lynx.audit.solid): what each check catches and the fence
it puts around a finding. Thresholds are relative to the codebase and are
not pinned here; the mechanisms are."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_switches_on_one_name_over_two_enums_are_two_subjects(tmp_path):
    # Three `switch (op)` in two types, but over two enums: two over MathOp,
    # one over Cmp. A new MathOp value edits two places, not three, and
    # nothing groups them. An `or` pattern names its enum too.
    root = tmp_path / "code"
    arms = "MathOp.Add or MathOp.Sub => 1, MathOp.Mul => 2, MathOp.Div => 3, MathOp.Min => 4, _ => 0"
    files = {
        "src/Pkg/Pkg.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
        "src/Pkg/Ops.cs": "namespace Pkg;\npublic enum MathOp { Add, Sub, Mul, Div, Min }\npublic enum Cmp { Eq, Ne, Lt, Gt }\n",
        "src/Pkg/MathExpr.cs": ("namespace Pkg;\npublic class MathExpr {\n"
                                f"  public int Eval(MathOp op) => op switch {{ {arms} }};\n"
                                f"  public int Symbol(MathOp op) => op switch {{ {arms} }};\n}}\n"),
        "src/Pkg/Numbers.cs": ("namespace Pkg;\npublic static class Numbers {\n"
                               "  public static bool Compare(Cmp op) { switch (op) { case Cmp.Eq: return true; case Cmp.Ne: return false;"
                               " case Cmp.Lt: return true; case Cmp.Gt: return false; default: return false; } }\n}\n"),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    from lynx.audit.solid import audit
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    findings, _ = audit(layer.graph, root, "Pkg")
    assert not [f.title for f in findings if f.principle == "O"]


def test_a_capability_a_type_says_it_lacks_is_no_substitution_finding(tmp_path):
    # A read-only Stream throws NotSupported from Write and SetLength and does
    # nothing in Flush: Stream's own contract once CanWrite says false. One
    # that says it can write and throws still is a finding, and so is a Read
    # that throws when only seeking was declined.
    root = tmp_path / "code"
    rest = ("  public override long Seek(long o, System.IO.SeekOrigin s) => 0;\n"
            "  public override long Length => 0;\n  public override long Position { get; set; }\n")
    files = {
        "src/Pkg/Pkg.csproj": '<Project Sdk="Microsoft.NET.Sdk"></Project>\n',
        "src/Pkg/Window.cs": ("namespace Pkg;\npublic sealed class Window : System.IO.Stream {\n"
                              "  public override bool CanRead => true;\n  public override bool CanSeek => true;\n"
                              "  public override bool CanWrite { get { return false; } }\n"
                              "  public override int Read(byte[] b, int o, int c) => 0;\n" + rest +
                              "  public override void Flush() { }\n"
                              "  public override void SetLength(long v) => throw new NotSupportedException();\n"
                              "  public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();\n}\n"),
        "src/Pkg/Liar.cs": ("namespace Pkg;\npublic sealed class Liar : System.IO.Stream {\n"
                            "  public override bool CanRead => true;\n  public override bool CanSeek => false;\n"
                            "  public override bool CanWrite => true;\n"
                            "  public override int Read(byte[] b, int o, int c) => throw new NotSupportedException();\n" + rest +
                            "  public override void Flush() { }\n  public override void SetLength(long v) { }\n"
                            "  public override void Write(byte[] b, int o, int c) => throw new NotSupportedException();\n}\n"),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    from lynx.audit.solid import audit
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root, supported_extensions=[".cs"])
    layer.rebuild(force=True)
    findings, _ = audit(layer.graph, root, "Pkg")
    l = sorted(f.title for f in findings if f.principle == "L")
    assert not [t for t in l if t.startswith("Window.")], l
    assert "Liar.Write only throws NotImplemented/NotSupported" in l, l
    assert "Liar.Read only throws NotImplemented/NotSupported" in l, l


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


def test_an_empty_override_of_an_abstract_member_is_nothing_to_do(tmp_path, monkeypatch):
    # Seen on a real library: a selection gesture's Cancel was empty because the
    # selection is only made at End, and the abstract Gesture.Cancel asks every
    # gesture for one. An empty override that drops the base's work stays.
    monkeypatch.setitem(CODE, "src/Pkg/Gestures.cs", (
        "namespace Pkg;\npublic class Tools {\n"
        "  public abstract class Gesture { public abstract void Cancel(); }\n"
        "  public sealed class Select : Gesture { public override void Cancel() { } }\n"
        "  public class Hooked { public virtual void Hook() { System.Console.WriteLine(1); } }\n"
        "  public sealed class Muted : Hooked { public override void Hook() { } }\n"
        "}\n"))
    titles = [f.title for f in _audit(tmp_path)[0]]
    assert not any("Select.Cancel" in t for t in titles), titles
    assert "Tools.Muted.Hook is an empty override" in titles, titles


DEAD_CODE_FALSE_ALARMS = {
    # `HeldKind<T>.Instance`: a generic type named in a member access
    "src/Pkg/Clips.cs": ("namespace Pkg;\npublic static class Clips {\n"
                         "  public static object Held<T>() => HeldKind<T>.Instance;\n"
                         "  private sealed class HeldKind<T> { public static readonly HeldKind<T> Instance = new(); }\n}\n"),
    # `using static Pkg.Look;` and its members called bare
    "src/Pkg/Look.cs": "namespace Pkg;\ninternal static class Look { public static int Size(int x) => x * 2; }\n",
    "src/Pkg/UsesLook.cs": "using static Pkg.Look;\nnamespace Pkg;\npublic class UsesLook { public int Big() => Size(3); }\n",
    # `[Menu(..)]` in its own namespace, with a public namesake that a `using` brings in
    "src/Pkg/MenuAttribute.cs": ("namespace Pkg;\ninternal sealed class MenuAttribute(string path) : System.Attribute "
                                 "{ public string Path { get; } = path; }\n"),
    "src/Pkg/Shared.cs": ("namespace Pkg.Shared;\npublic sealed class MenuAttribute : System.Attribute "
                          "{ public MenuAttribute(string path) { } }\n"),
    "src/Pkg/Spline.cs": "using Pkg.Shared;\nnamespace Pkg;\n[Menu(\"Splines/Spline\")]\npublic class Spline { }\n",
    # a module initializer: the runtime calls it
    "src/Pkg/Kernels.cs": ("using System.Runtime.CompilerServices;\nnamespace Pkg;\n"
                           "internal static class Kernels { [ModuleInitializer] internal static void Register() { } }\n"),
    # a nested partial type over two files, made from a third, with a partial namesake nested in another type
    "src/Pkg/Doc.cs": "namespace Pkg;\npublic static partial class Doc {\n  public static void Save() { new Writing(1).Go(); }\n}\n",
    "src/Pkg/Doc.Writing.cs": ("namespace Pkg;\npublic static partial class Doc {\n"
                               "  private sealed partial class Writing { private readonly int _n; public Writing(int n) { _n = n; } }\n}\n"),
    "src/Pkg/Doc.Writing.More.cs": ("namespace Pkg;\npublic static partial class Doc {\n"
                                    "  private sealed partial class Writing { public void Go() { } }\n}\n"),
    "src/Pkg/Compiler.cs": ("namespace Pkg;\npublic static partial class Compiler {\n"
                            "  public static void Run() => new Writing().Go();\n"
                            "  internal sealed partial class Writing { public void Go() { } }\n}\n"),
    "src/Pkg/Compiler.More.cs": ("namespace Pkg;\npublic static partial class Compiler {\n"
                                 "  internal sealed partial class Writing { public void Stop() { } }\n}\n"),
}


@pytest.mark.parametrize("name", ["HeldKind", "Look", "MenuAttribute", "Kernels", "Doc.Writing", "Compiler.Writing"])
def test_a_type_used_in_these_ways_is_not_dead_code(tmp_path, monkeypatch, name):
    # Seen on a real library: six types flagged «used by nothing», each used.
    for rel, text in DEAD_CODE_FALSE_ALARMS.items():
        monkeypatch.setitem(CODE, rel, text)
    dead = [f.title for f in _audit(tmp_path)[0] if f.title.endswith("is used by nothing")]
    assert not any(name in t for t in dead), dead
