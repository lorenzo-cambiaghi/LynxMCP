"""Unit tests for `lynx.client_connector` and `lynx connect` CLI.

All filesystem operations run inside `tmp_path` with injected `home`/`env`,
so the user's real editor configs are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lynx.cli import main as cli_main
from lynx.client_connector import (
    connect_client,
    disconnect_client,
    get_client_spec,
    inspect_client,
    render_snippet,
)


def test_connect_and_disconnect_preserves_existing_servers_and_writes_backup(tmp_path):
    cursor_cfg = tmp_path / ".cursor" / "mcp.json"
    cursor_cfg.parent.mkdir(parents=True)
    original = {
        "mcpServers": {
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
        },
        "customSetting": True,
    }
    cursor_cfg.write_text(json.dumps(original, indent=2), encoding="utf-8")

    cfg_file = tmp_path / "lynx-config.json"
    cfg_file.write_text("{}", encoding="utf-8")

    res = connect_client("cursor", config_path=cfg_file, home=tmp_path, python_exe="/usr/bin/python3")
    assert res.ok is True
    assert res.config_path == cursor_cfg
    assert res.backup_path.name.startswith("mcp.json.lynx-")
    assert res.backup_path.is_file()

    written = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    # Existing MCP servers and unrelated keys stay completely untouched.
    assert written["customSetting"] is True
    assert "github" in written["mcpServers"]
    assert written["mcpServers"]["lynx"]["command"] == "/usr/bin/python3"
    assert written["mcpServers"]["lynx"]["args"] == [
        "-m", "lynx", "serve", "--config", str(cfg_file.resolve()),
    ]

    # Inspect now reports connected=True
    st = inspect_client(get_client_spec("cursor"), home=tmp_path)
    assert st.detected is True
    assert st.connected is True

    # Disconnect removes ONLY lynx and keeps github
    disc = disconnect_client("cursor", home=tmp_path)
    assert disc.ok is True
    after = json.loads(cursor_cfg.read_text(encoding="utf-8"))
    assert "lynx" not in after["mcpServers"]
    assert "github" in after["mcpServers"]


def test_connect_vscode_uses_servers_key_and_stdio_type(tmp_path):
    repo = tmp_path / "my-repo"
    repo.mkdir()
    env = {"APPDATA": str(tmp_path / "AppData")}

    res = connect_client(
        "vscode",
        repo_path=repo,
        profile="core",
        home=tmp_path,
        env=env,
        platform="win32",
        python_exe="C:/py/python.exe",
    )
    assert res.ok is True
    dest = tmp_path / "AppData" / "Code" / "User" / "mcp.json"
    assert res.config_path == dest
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert "servers" in data and "mcpServers" not in data
    entry = data["servers"]["lynx"]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "lynx", "serve", str(repo.resolve()), "--profile", "core"]


def test_connect_refuses_corrupt_json_without_overwriting(tmp_path):
    claude_cfg = tmp_path / ".claude.json"
    claude_cfg.write_text("{broken json", encoding="utf-8")

    res = connect_client("claude", repo_path=tmp_path, home=tmp_path)
    assert res.ok is False
    assert "invalid JSON" in res.message
    assert claude_cfg.read_text(encoding="utf-8") == "{broken json"


def test_connect_dry_run_does_not_create_file(tmp_path):
    res = connect_client("antigravity", repo_path=tmp_path, dry_run=True, home=tmp_path)
    assert res.ok is True
    assert res.dry_run is True
    assert res.preview_json and '"lynx"' in res.preview_json
    assert not (tmp_path / ".gemini" / "antigravity" / "mcp_config.json").exists()


def test_aliases_resolve_correctly():
    assert get_client_spec("claude-code").key == "claude"
    assert get_client_spec("claude-desktop").key == "claude_desktop"
    assert get_client_spec("vs-code").key == "vscode"
    assert get_client_spec("copilot").key == "vscode"
    assert get_client_spec("unknown") is None


def test_render_snippet_contains_expected_structure(tmp_path):
    spec = get_client_spec("cursor")
    out = render_snippet(spec, config_path=tmp_path / "config.json", python_exe="/bin/py")
    parsed = json.loads(out)
    assert parsed["mcpServers"]["lynx"]["command"] == "/bin/py"


def test_cli_connect_list_and_dry_run(tmp_path, capsys):
    rc = cli_main(["connect", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    keys = {c["key"] for c in payload}
    assert {"claude", "cursor", "claude_desktop", "windsurf", "antigravity", "vscode"}.issubset(keys)

    rc = cli_main(["connect", "cursor", "--path", str(tmp_path), "--dry-run", "--json"])
    assert rc == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] is True
    assert res["dry_run"] is True


def _fake_claude_cli():
    calls = []

    def run(cmd):
        calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return calls, (lambda name: "/bin/claude" if name == "claude" else None), run


def test_claude_project_scope_entries_count_as_connected(tmp_path):
    (tmp_path / ".claude.json").write_text(json.dumps({
        "projects": {
            "C:/code/game": {"mcpServers": {"lynxmcp": {
                "command": "C:/uv/tools/lynx-mcp/Scripts/python.exe",
                "args": ["-m", "lynx", "serve", "--config", "C:/lynx/config.json"]}}},
            "C:/code/web": {"mcpServers": {"github": {"command": "npx", "args": ["gh"]}}},
        },
    }), encoding="utf-8")

    st = inspect_client(get_client_spec("claude"), home=tmp_path)

    assert st.connected is True
    assert st.locations == ("project C:/code/game",)


def test_connect_updates_an_existing_lynx_entry_instead_of_adding_a_second(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"code-search": {
        "command": "uvx", "args": ["--from", "lynx-mcp", "lynx", "serve", "."]}}}), encoding="utf-8")

    res = connect_client("cursor", repo_path=tmp_path, home=tmp_path, python_exe="/py")

    assert res.ok
    servers = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]
    assert list(servers) == ["code-search"]
    assert servers["code-search"]["command"] == "/py"


def test_connect_claude_goes_through_the_cli_when_present(tmp_path):
    calls, which, run = _fake_claude_cli()

    res = connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py",
                         which=which, runner=run)

    assert res.ok, res.message
    assert calls[0] == ["/bin/claude", "mcp", "remove", "--scope", "user", "lynx"]
    assert calls[1][:7] == ["/bin/claude", "mcp", "add", "--scope", "user", "lynx", "--"]
    assert calls[1][7:] == ["/py", "-m", "lynx", "serve", str(tmp_path.resolve())]
    # The CLI is the writer: the JSON file is left to Claude Code.
    assert not (tmp_path / ".claude.json").exists()


def test_repeated_connects_keep_the_original_file_in_a_backup(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir(parents=True)
    original = '{"mcpServers": {"github": {"command": "npx", "args": ["gh"]}}}'
    cfg.write_text(original, encoding="utf-8")

    first = connect_client("cursor", repo_path=tmp_path, home=tmp_path, python_exe="/py")
    second = connect_client("cursor", repo_path=tmp_path, home=tmp_path, python_exe="/py2")

    assert first.backup_path != second.backup_path
    assert first.backup_path.read_text(encoding="utf-8") == original


def test_entry_names_uvx_when_lynx_runs_from_the_uvx_cache(tmp_path, monkeypatch):
    import lynx.client_connector as cc
    from lynx.client_connector import build_server_entry

    monkeypatch.setattr(cc, "_running_under_uvx", lambda: True)
    entry = build_server_entry(repo_path=tmp_path, profile="core")

    assert entry["command"] == "uvx"
    assert entry["args"] == ["--from", "lynx-mcp", "lynx", "serve", str(tmp_path.resolve()), "--profile", "core"]


@pytest.mark.parametrize("client,expected", [("claude", "."), ("cursor", "abs")])
def test_cli_connect_without_config_uses_dot_only_for_claude_code(tmp_path, monkeypatch, client, expected):
    import lynx.client_connector as cc

    seen = {}

    def _fake_connect(key, **kw):
        seen.update(kw)
        return cc.ConnectResult(ok=True, client_key=key, message="ok")

    monkeypatch.setattr(cc, "connect_client", _fake_connect)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)

    assert cli_main(["connect", client]) == 0
    if expected == ".":
        assert str(seen["repo_path"]) == "."
    else:
        assert seen["repo_path"] == tmp_path.resolve()


def test_dot_repo_path_stays_relative_in_the_entry():
    from lynx.client_connector import build_server_entry

    assert build_server_entry(repo_path=".", python_exe="/py")["args"] == ["-m", "lynx", "serve", "."]


# ---------------------------------------------------------------------------
# Claude Code prompt hook (~/.claude/settings.json)
# ---------------------------------------------------------------------------

def _settings(home):
    return json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))


def _lynx_hooks(settings):
    return [h for g in settings.get("hooks", {}).get("UserPromptSubmit", [])
            for h in g["hooks"] if h.get("tool") == "prompt_hook"]


def test_connect_claude_adds_the_prompt_hook_and_keeps_the_rest(tmp_path):
    other = {"hooks": [{"type": "command", "command": "echo hi"}]}
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps({
        "model": "opus", "hooks": {"UserPromptSubmit": [other], "Stop": [other]}}), encoding="utf-8")
    calls, which, run = _fake_claude_cli()

    res = connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py",
                         which=which, runner=run)

    assert res.ok, res.message
    s = _settings(tmp_path)
    assert s["model"] == "opus" and s["hooks"]["Stop"] == [other]
    assert s["hooks"]["UserPromptSubmit"][0] == other
    # The shape Claude Code reads: the tool's hook JSON becomes additionalContext.
    assert _lynx_hooks(s) == [{"type": "mcp_tool", "server": "lynx", "tool": "prompt_hook",
                               "input": {"prompt": "${prompt}"}}]


def test_a_second_connect_does_not_add_a_second_hook(tmp_path):
    _, which, run = _fake_claude_cli()
    connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py", which=which, runner=run)
    before = (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")

    res = connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py", which=which, runner=run)

    assert res.ok and "already" in res.message
    assert (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8") == before
    assert len(_lynx_hooks(_settings(tmp_path))) == 1


def test_disconnect_claude_removes_only_the_lynx_hook(tmp_path):
    other = {"hooks": [{"type": "command", "command": "echo hi"}]}
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps({
        "hooks": {"UserPromptSubmit": [other]}}), encoding="utf-8")
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"lynx": {
        "command": "/py", "args": ["-m", "lynx", "serve", "."]}}}), encoding="utf-8")
    _, which, run = _fake_claude_cli()
    connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py", which=which, runner=run)
    assert len(_lynx_hooks(_settings(tmp_path))) == 1

    res = disconnect_client("claude", home=tmp_path, which=which, runner=run)

    assert res.ok, res.message
    assert _settings(tmp_path) == {"hooks": {"UserPromptSubmit": [other]}}


def test_no_hook_and_other_clients_leave_claude_settings_alone(tmp_path):
    _, which, run = _fake_claude_cli()
    connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py",
                   which=which, runner=run, hook=False)
    connect_client("cursor", repo_path=tmp_path, home=tmp_path, python_exe="/py")
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_corrupt_claude_settings_are_not_overwritten(tmp_path):
    (tmp_path / ".claude").mkdir()
    bad = (tmp_path / ".claude" / "settings.json")
    bad.write_text("{ not json", encoding="utf-8")
    _, which, run = _fake_claude_cli()

    res = connect_client("claude", repo_path=tmp_path, home=tmp_path, python_exe="/py", which=which, runner=run)

    assert not res.ok and "invalid JSON" in res.message
    assert bad.read_text(encoding="utf-8") == "{ not json"
