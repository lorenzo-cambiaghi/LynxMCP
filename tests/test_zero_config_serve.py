"""Zero-config `lynx serve <PATH>` tests.

Nothing here loads an embedding model or starts stdio transport: `run_server`
is monkeypatched where the CLI dispatch is exercised, so the whole module runs
in milliseconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lynx.cli import main as cli_main
from lynx.config import (
    SOURCE_NAME_RE,
    build_zero_config,
    default_auto_storage_dir,
    sanitize_source_name,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("my-cool-repo", "my_cool_repo"),
        ("2026-project", "repo_2026_project"),
        ("___", "project"),
        ("", "project"),
        ("already_valid_1", "already_valid_1"),
        ("a" * 60, "a" * 40),
    ],
)
def test_sanitize_source_name_matches_schema_regex(raw, expected):
    out = sanitize_source_name(raw)
    assert out == expected
    assert SOURCE_NAME_RE.match(out), f"{out!r} does not match SOURCE_NAME_RE"


def test_default_auto_storage_dir_lives_outside_repo_and_is_deterministic(tmp_path):
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()

    s1 = default_auto_storage_dir(repo_a)
    s2 = default_auto_storage_dir(repo_a)
    s3 = default_auto_storage_dir(repo_b)

    assert s1 == s2
    assert s1 != s3
    # Never pollutes the repository itself.
    assert repo_a.resolve() not in s1.parents
    assert s1 != repo_a.resolve()


def test_build_zero_config_detects_extensions_git_and_enables_graph(tmp_path):
    repo = tmp_path / "my-game"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "player.gd").write_text("extends Node\nfunc move():\n    pass\n", encoding="utf-8")
    (repo / "README.md").write_text("# game\n", encoding="utf-8")

    cfg = build_zero_config(repo)

    assert list(cfg.sources) == ["my_game"]
    src = cfg.sources["my_game"]
    assert src["type"] == "codebase"
    assert src["path"] == repo.resolve()
    assert {".gd", ".md"}.issubset(src["supported_extensions"])
    assert src["graph"] == {"enabled": True}
    assert src["git_integration"] == {"enabled": True}
    assert src["watcher"]["enabled"] is True
    joined_ignores = " ".join(src["ignored_path_fragments"]).replace("\\", "/")
    assert "/node_modules/" in joined_ignores
    # Not a Godot or Unity project: their folder names stay indexable.
    assert "/Library/" not in joined_ignores
    # Building the config must not touch the user's repo.
    assert sorted(p.name for p in repo.iterdir()) == [".git", "README.md", "player.gd"]


def _ignores(cfg):
    src = next(iter(cfg.sources.values()))
    return " ".join(src["ignored_path_fragments"]).replace("\\", "/")


def test_build_zero_config_keeps_binary_and_engine_files_out(tmp_path):
    # A Unity project: assets and their .meta files outnumber the scripts.
    repo = tmp_path / "shooter"
    (repo / "Assets" / "Scripts").mkdir(parents=True)
    (repo / "ProjectSettings").mkdir()
    (repo / "Library" / "PackageCache").mkdir(parents=True)
    (repo / "Assets" / "Scripts" / "Player.cs").write_text("class Player {}\n", encoding="utf-8")
    for i in range(20):
        (repo / "Assets" / f"tex{i}.png").write_bytes(b"\x89PNG")
        (repo / "Assets" / f"tex{i}.png.meta").write_text("guid: x\n", encoding="utf-8")
    (repo / "Library" / "PackageCache" / "Gen.cs").write_text("class Gen {}\n", encoding="utf-8")

    cfg = build_zero_config(repo)

    src = cfg.sources["shooter"]
    assert sorted(src["supported_extensions"]) == [".cs"]
    assert "/Library/" in _ignores(cfg)


def test_build_zero_config_godot_project_ignores_its_cache(tmp_path):
    repo = tmp_path / "platformer"
    repo.mkdir()
    (repo / "project.godot").write_text("[application]\n", encoding="utf-8")
    (repo / "main.gd").write_text("extends Node\n", encoding="utf-8")

    assert "/.godot/" in _ignores(build_zero_config(repo))


def test_zero_config_extensions_cover_every_grammar():
    from lynx.chunking import grammar_extensions
    from lynx.config import ZERO_CONFIG_EXTENSIONS

    missing = set(grammar_extensions()) - ZERO_CONFIG_EXTENSIONS
    assert not missing, f"zero-config would skip parsed languages: {sorted(missing)}"


@pytest.mark.parametrize("which", ["home", "parent_of_home", "root"])
def test_build_zero_config_refuses_home_and_drive_root(tmp_path, monkeypatch, which):
    home = tmp_path / "users" / "me"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    target = {"home": home, "parent_of_home": home.parent, "root": Path(tmp_path.anchor)}[which]

    with pytest.raises(ValueError, match="refusing to index"):
        build_zero_config(target)


def test_build_zero_config_accepts_a_project_inside_home(tmp_path, monkeypatch):
    home = tmp_path / "me"
    project = home / "code" / "app"
    project.mkdir(parents=True)
    (project / "app.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert "app" in build_zero_config(project, storage_path=tmp_path / "store").sources


def test_build_zero_config_empty_dir_falls_back_to_default_extensions(tmp_path):
    empty = tmp_path / "fresh"
    empty.mkdir()

    cfg = build_zero_config(empty, graph_enabled=False)

    src = cfg.sources["fresh"]
    assert ".py" in src["supported_extensions"]
    assert ".gd" in src["supported_extensions"]
    assert src["git_integration"] == {"enabled": False}
    assert src["graph"] == {"enabled": False}


def test_build_zero_config_rejects_missing_or_file_path(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        build_zero_config(tmp_path / "nope")

    f = tmp_path / "file.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        build_zero_config(f)


def test_cli_serve_zero_config_passes_in_memory_config(tmp_path, monkeypatch):
    repo = tmp_path / "demo-app"
    repo.mkdir()
    (repo / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")

    captured = {}

    def _fake_run_server(config_path=None, profile=None, sources=None, *, config=None, http_port=None):
        captured["config_path"] = config_path
        captured["profile"] = profile
        captured["sources"] = sources
        captured["config"] = config

    import lynx.server as server_mod
    monkeypatch.setattr(server_mod, "run_server", _fake_run_server)

    rc = cli_main(["serve", str(repo), "--profile", "core"])
    assert rc == 0
    assert captured["config_path"] is None
    assert captured["profile"] == "core"
    assert captured["config"] is not None
    assert "demo_app" in captured["config"].sources
    assert captured["config"].sources["demo_app"]["graph"] == {"enabled": True}


def test_cli_serve_zero_config_no_graph_flag(tmp_path, monkeypatch):
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

    captured = {}

    def _fake_run_server(config_path=None, profile=None, sources=None, *, config=None, http_port=None):
        captured["config"] = config

    import lynx.server as server_mod
    monkeypatch.setattr(server_mod, "run_server", _fake_run_server)

    rc = cli_main(["serve", str(repo), "--no-graph"])
    assert rc == 0
    assert captured["config"].sources["demo"]["graph"] == {"enabled": False}


def test_cli_serve_rejects_path_combined_with_config(tmp_path, capsys):
    repo = tmp_path / "demo"
    repo.mkdir()
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")

    rc = cli_main(["serve", str(repo), "--config", str(cfg)])
    assert rc == 2
    assert "--config cannot be combined" in capsys.readouterr().err


def test_cli_serve_rejects_missing_path(tmp_path, capsys):
    rc = cli_main(["serve", str(tmp_path / "missing")])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
