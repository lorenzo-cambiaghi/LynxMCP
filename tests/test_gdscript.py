"""GDScript: chunks, the script as a class, and its graph.

A GDScript file is a class. `class_name Player` names it (a script without
one is known by its path, so it takes the file stem), `extends` at the top
is its base, and its functions are its methods. Chunk names and graph labels
both say `Player.take_damage`, so search and the graph point at the same
symbol.

The grammar has wheels for three platforms only; elsewhere it is not
installed and `.gd` files must still index, as text.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

from lynx.chunking import chunk_file, grammar_extensions
from lynx.graph.extractor import extract_file

grammar = pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_gdscript") is None,
    reason="tree-sitter-gdscript has no wheel for this platform")

PLAYER = textwrap.dedent('''\
    @tool
    class_name Player
    extends CharacterBody2D

    ## Emitted when health reaches zero.
    signal died

    const Bullet = preload("res://scenes/bullet.tscn")

    @export var speed := 200.0
    var health: int = 100

    enum State { IDLE, RUN }

    func _init():
    \thealth = 100

    ## Applies damage and dies at zero.
    func take_damage(amount: int) -> void:
    \thealth -= amount
    \tif health <= 0:
    \t\tdie()

    func heal():
    \tself.take_damage(-10)

    func die():
    \tdied.emit()

    static func clamp_speed(v: float) -> float:
    \treturn clampf(v, 0.0, 400.0)

    class Inventory extends RefCounted:
    \tvar items := []

    \tfunc add(item):
    \t\titems.append(item)
    \t\t_changed()

    \tfunc _changed():
    \t\tpass
    ''')

ENEMY = textwrap.dedent('''\
    extends Node2D

    func attack():
    \tpass
    ''')

BOSS = textwrap.dedent('''\
    extends "res://enemies/enemy.gd"

    func attack():
    \troar()

    func roar():
    \tpass
    ''')


def test_gd_files_index_as_text_without_the_grammar(tmp_path):
    # sys.modules[name] = None makes the import raise ImportError, as on a
    # platform the wheel does not exist for.
    script = textwrap.dedent(f'''
        import sys
        sys.modules["tree_sitter_gdscript"] = None
        from lynx import chunking
        assert ".gd" not in chunking.grammar_extensions()
        chunks = chunking.chunk_file({str(tmp_path / "player.gd")!r}, {PLAYER!r})
        assert chunks and all(c["chunker"] == "sentence_splitter" for c in chunks)
        from lynx.graph.extractor import extract_file
        assert extract_file({str(tmp_path / "player.gd")!r}, {PLAYER!r}) is None
        print("ok")
    ''')
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")




def _by_name(chunks: list) -> dict:
    return {c["symbol_name"]: c for c in chunks}


@grammar
def test_gd_is_parsed_with_its_grammar():
    assert ".gd" in grammar_extensions()
    chunks = chunk_file("player.gd", PLAYER)
    assert {c["language"] for c in chunks} == {"gdscript"}
    assert {c["chunker"] for c in chunks} == {"tree_sitter"}


@grammar
def test_top_level_functions_are_methods_of_the_script_class():
    names = set(_by_name(chunk_file("player.gd", PLAYER)))
    assert {"Player.take_damage", "Player.heal", "Player.die",
            "Player.clamp_speed", "Player._init", "Player.State",
            "Player.Inventory.add", "Player.Inventory._changed"} <= names
    assert not any("<anonymous>" in n for n in names)


@grammar
def test_a_script_without_class_name_takes_the_file_stem():
    names = set(_by_name(chunk_file("enemies/enemy.gd", ENEMY)))
    assert "enemy.attack" in names


@grammar
def test_a_doc_comment_opens_the_function_it_documents():
    chunks = _by_name(chunk_file("player.gd", PLAYER))
    take = chunks["Player.take_damage"]["text"]
    assert take.startswith("## Applies damage")
    # ...and did not stay behind in the function above it.
    assert "## Applies damage" not in chunks["Player._init"]["text"]


@grammar
def test_the_header_holds_what_the_script_declares():
    header = _by_name(chunk_file("player.gd", PLAYER))["<header>"]["text"]
    for line in ("class_name Player", "extends CharacterBody2D",
                 "signal died", "@export var speed", "const Bullet"):
        assert line in header
    assert "func " not in header


@grammar
def test_an_inner_class_gets_a_skeleton_with_its_fields():
    inner = _by_name(chunk_file("player.gd", PLAYER))["Player.Inventory"]["text"]
    assert inner.startswith("class Inventory extends RefCounted")
    assert "var items" in inner
    assert "items.append" not in inner           # bodies belong to the methods


def _graph(path: str, text: str):
    r = extract_file(path, text)
    assert r is not None
    labels = {n["label"]: n for n in r["nodes"]}
    ids = {n["id"]: n["label"] for n in r["nodes"]}
    edges = {(ids.get(e["source"]), ids.get(e["target"]), e["relation"])
             for e in r["edges"]}
    return r, labels, edges


@grammar
def test_the_script_is_a_class_node_that_contains_its_functions():
    _r, labels, edges = _graph("player.gd", PLAYER)
    assert labels["Player"]["kind"] == "class"
    assert labels["Player._init"]["kind"] == "function"
    assert ("Player", "Player.take_damage", "contains") in edges
    assert ("Player.Inventory", "Player.Inventory.add", "contains") in edges


@grammar
def test_calls_inside_the_file_resolve():
    _r, _labels, edges = _graph("player.gd", PLAYER)
    assert ("Player.take_damage", "Player.die", "calls") in edges      # die()
    assert ("Player.heal", "Player.take_damage", "calls") in edges     # self.x()
    assert ("Player.Inventory.add", "Player.Inventory._changed", "calls") in edges


@grammar
def test_a_method_call_on_another_object_is_left_to_the_resolver():
    r, _labels, _edges = _graph("player.gd", PLAYER)
    emits = [c for c in r["raw_calls"] if c["callee_name"] == "emit"]
    assert emits and all(c["is_member"] for c in emits)


@grammar
def test_extends_gives_the_base_by_name_or_by_script_path():
    r, labels, _edges = _graph("player.gd", PLAYER)
    bases = {(i["child"], i["base_name"]) for i in r["raw_inherits"]}
    assert (labels["Player"]["id"], "CharacterBody2D") in bases
    assert (labels["Player.Inventory"]["id"], "RefCounted") in bases
    boss, _l, _e = _graph("enemies/boss.gd", BOSS)
    assert [i["base_name"] for i in boss["raw_inherits"]] == ["enemy"]


@grammar
def test_preload_and_a_script_path_extends_are_imports():
    r, _labels, _edges = _graph("player.gd", PLAYER)
    assert "res://scenes/bullet.tscn" in {e.get("module") for e in r["edges"]
                                          if e["relation"] == "imports"}
    boss, _l, _e = _graph("enemies/boss.gd", BOSS)
    assert "res://enemies/enemy.gd" in {e.get("module") for e in boss["edges"]
                                        if e["relation"] == "imports"}


@grammar
def test_a_subclass_by_path_resolves_across_files(tmp_path):
    from lynx.graph import GraphLayer
    root = tmp_path / "game"
    (root / "enemies").mkdir(parents=True)
    (root / "enemies" / "enemy.gd").write_text(ENEMY, encoding="utf-8")
    (root / "enemies" / "boss.gd").write_text(BOSS, encoding="utf-8")
    layer = GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root,
                       supported_extensions=[".gd"])
    layer.rebuild(force=True)
    g = layer.graph
    label = {nid: d.get("label") for nid, d in g.nodes(data=True)}
    inherits = {(label[u], label[v]) for u, v, d in g.edges(data=True)
                if d.get("relation") == "inherits"}
    assert ("boss", "enemy") in inherits


SPAWNER = textwrap.dedent('''\
    class_name Spawner
    extends Node

    ## Spawns one enemy.
    @rpc("any_peer")
    func spawn():
    \tvar later = func(): return build()
    \tlater.call()

    func build():
    \tpass
    ''')


@grammar
def test_an_annotation_on_its_own_line_belongs_to_the_function_below():
    chunks = _by_name(chunk_file("spawner.gd", SPAWNER))
    spawn = chunks["Spawner.spawn"]["text"]
    assert spawn.startswith("## Spawns one enemy.\n@rpc(")
    assert "@rpc" not in chunks["<header>"]["text"]


@grammar
def test_a_call_inside_a_lambda_counts_for_the_function_around_it():
    _r, _labels, edges = _graph("spawner.gd", SPAWNER)
    assert ("Spawner.spawn", "Spawner.build", "calls") in edges


BASE_ENEMY = textwrap.dedent('''\
    class_name BaseEnemy
    extends Node2D

    func attack():
    \tpass
    ''')


def _inherits(layer) -> set:
    g = layer.graph
    label = {nid: d.get("label") for nid, d in g.nodes(data=True)}
    return {(label[u], label[v]) for u, v, d in g.edges(data=True)
            if d.get("relation") == "inherits"}


@grammar
def test_extends_by_path_finds_a_script_with_another_class_name(tmp_path):
    # `extends "res://enemies/enemy_base.gd"` names the file; the script in
    # it declared `class_name BaseEnemy`. The edge must survive an edit of
    # either file and a reload from disk.
    from lynx.graph import GraphLayer
    root = tmp_path / "game"
    (root / "enemies").mkdir(parents=True)
    base = root / "enemies" / "enemy_base.gd"
    boss = root / "enemies" / "boss.gd"
    base.write_text(BASE_ENEMY, encoding="utf-8")
    boss.write_text(BOSS.replace("enemy.gd", "enemy_base.gd"), encoding="utf-8")

    def layer():
        return GraphLayer(storage_dir=tmp_path / "graph", codebase_path=root,
                          supported_extensions=[".gd"])

    first = layer()
    first.rebuild(force=True)
    assert ("boss", "BaseEnemy") in _inherits(first)
    base.write_text(BASE_ENEMY + "\nfunc retreat():\n\tpass\n", encoding="utf-8")
    assert first.update_file(str(base))
    assert ("boss", "BaseEnemy") in _inherits(first)
    boss.write_text(boss.read_text(encoding="utf-8") + "\nfunc taunt():\n\tpass\n",
                    encoding="utf-8")
    assert first.update_file(str(boss))
    assert ("boss", "BaseEnemy") in _inherits(first)
    assert ("boss", "BaseEnemy") in _inherits(layer())


def test_a_godot_project_is_recognized(tmp_path):
    from lynx.overview import build_overview
    (tmp_path / "project.godot").write_text("[application]\n", encoding="utf-8")
    (tmp_path / "player.gd").write_text(ENEMY, encoding="utf-8")
    ov = build_overview(tmp_path)
    assert "project.godot" in ov["manifests"]
    assert "Godot" in ov["frameworks"]


def test_test_files_named_test_underscore_are_tests():
    import re
    from lynx.sources.codebase import CodebaseBackend
    rx = re.compile(CodebaseBackend._DEFAULT_TEST_PATH_PATTERN, re.IGNORECASE)
    for path in ("addons/gut_tests/test_player.gd", "test_player.gd",
                 "app/test_views.py", "tests/unit/x.py"):
        assert rx.search(path), path
    for path in ("src/latest.py", "src/contest_entry.gd", "src/testing.py"):
        assert not rx.search(path), path
