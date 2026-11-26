"""1-click / 1-command AI client MCP connector — shared by CLI and Manager UI.

Single source of truth for supported MCP clients, their OS-specific config
paths, snippet generation, connection status detection, and safe idempotent
JSON merging (with automatic `.bak` backup and atomic write).

Pure stdlib — safe to import from both CLI and FastAPI without heavy deps.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ClientSpec:
    """Static definition of a supported AI client."""

    key: str
    label: str
    short_label: str
    config_file_hint: str
    rules_file: Optional[str]
    notes: str
    root_key: str = "mcpServers"
    include_stdio_type: bool = False
    supports_auto_connect: bool = True
    path_resolver: Optional[Callable[[Path, Mapping[str, str], str], Path]] = None


@dataclass(frozen=True)
class ClientStatus:
    """Live detection state for one AI client on this machine."""

    spec: ClientSpec
    config_path: Optional[Path]
    detected: bool
    connected: bool
    current_args: Tuple[str, ...] = ()
    error: Optional[str] = None
    # Where Lynx is registered, e.g. ("user", "project C:/code/app"). An
    # entry counts by what it runs, not by its name.
    locations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectResult:
    """Outcome of connecting or disconnecting Lynx in a client config."""

    ok: bool
    client_key: str
    message: str
    config_path: Optional[Path] = None
    backup_path: Optional[Path] = None
    dry_run: bool = False
    preview_json: Optional[str] = None


# ---------------------------------------------------------------------------
# OS-aware config path resolvers
# ---------------------------------------------------------------------------


def _claude_code_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    return home / ".claude.json"


def _claude_desktop_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    if platform == "win32":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    xdg = env.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else home / ".config"
    return base / "Claude" / "claude_desktop_config.json"


def _cursor_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    return home / ".cursor" / "mcp.json"


def _windsurf_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    return home / ".codeium" / "windsurf" / "mcp_config.json"


def _antigravity_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    return home / ".gemini" / "antigravity" / "mcp_config.json"


def _vscode_path(home: Path, env: Mapping[str, str], platform: str) -> Path:
    if platform == "win32":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Code" / "User" / "mcp.json"
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    xdg = env.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else home / ".config"
    return base / "Code" / "User" / "mcp.json"


CLIENTS: Tuple[ClientSpec, ...] = (
    ClientSpec(
        key="claude",
        label="Claude Code (CLI + VS Code extension)",
        short_label="Claude Code",
        config_file_hint="~/.claude.json",
        rules_file="CLAUDE.md",
        notes="Restart Claude Code (or run /mcp) to pick up the server. "
              "Prefer CLI? Run `claude mcp add lynx -- uvx --from lynx-mcp lynx serve .`.",
        path_resolver=_claude_code_path,
    ),
    ClientSpec(
        key="cursor",
        label="Cursor",
        short_label="Cursor",
        config_file_hint="~/.cursor/mcp.json",
        rules_file=".cursor/rules/lynx.md",
        notes="Applies globally across projects. For project-only config, copy the snippet into .cursor/mcp.json.",
        path_resolver=_cursor_path,
    ),
    ClientSpec(
        key="claude_desktop",
        label="Claude Desktop",
        short_label="Claude Desktop",
        config_file_hint="claude_desktop_config.json",
        rules_file="CLAUDE.md",
        notes="Quit and reopen Claude Desktop after connecting.",
        path_resolver=_claude_desktop_path,
    ),
    ClientSpec(
        key="windsurf",
        label="Windsurf",
        short_label="Windsurf",
        config_file_hint="~/.codeium/windsurf/mcp_config.json",
        rules_file="AGENTS.md",
        notes="Reload Windsurf (or refresh MCP servers in Cascade settings) after connecting.",
        path_resolver=_windsurf_path,
    ),
    ClientSpec(
        key="antigravity",
        label="Google Antigravity",
        short_label="Antigravity",
        config_file_hint="~/.gemini/antigravity/mcp_config.json",
        rules_file="AGENTS.md",
        notes="Connects globally in Antigravity. Project-local .agents/mcp.json uses the same JSON shape.",
        path_resolver=_antigravity_path,
    ),
    ClientSpec(
        key="vscode",
        label="VS Code (GitHub Copilot)",
        short_label="VS Code",
        config_file_hint="Code/User/mcp.json",
        rules_file="AGENTS.md",
        notes="VS Code reads MCP servers under the `servers` key. Reload the window after connecting.",
        root_key="servers",
        include_stdio_type=True,
        path_resolver=_vscode_path,
    ),
    ClientSpec(
        key="aider",
        label="Aider / Continue.dev / generic stdio",
        short_label="Generic stdio",
        config_file_hint="(client-specific)",
        rules_file="AGENTS.md",
        notes="Use this command as a stdio MCP server in any MCP-compatible client.",
        supports_auto_connect=False,
        path_resolver=None,
    ),
)

_CLIENTS_BY_KEY: Dict[str, ClientSpec] = {c.key: c for c in CLIENTS}


def get_client_spec(key: str) -> Optional[ClientSpec]:
    """Look up a supported client by key (accepts common aliases)."""
    aliases = {
        "claude-code": "claude",
        "claude-desktop": "claude_desktop",
        "vs-code": "vscode",
        "copilot": "vscode",
    }
    norm = aliases.get((key or "").strip().lower(), (key or "").strip().lower())
    return _CLIENTS_BY_KEY.get(norm)


def supported_client_keys(*, auto_only: bool = False) -> Tuple[str, ...]:
    return tuple(
        c.key for c in CLIENTS if (not auto_only or c.supports_auto_connect)
    )


def resolve_client_config_path(
    spec: ClientSpec,
    *,
    home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> Optional[Path]:
    if spec.path_resolver is None:
        return None
    h = home if home is not None else Path.home()
    e = env if env is not None else os.environ
    p = platform if platform is not None else sys.platform
    return spec.path_resolver(h, e, p)


# ---------------------------------------------------------------------------
# Entry & snippet builders
# ---------------------------------------------------------------------------


def _running_under_uvx() -> bool:
    from .updater import detect_install_method

    return detect_install_method().method == "uvx"


def build_server_entry(
    *,
    config_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
    profile: Optional[str] = None,
    python_exe: Optional[str] = None,
    include_stdio_type: bool = False,
) -> dict:
    """Build the `mcpServers.lynx` (or `servers.lynx`) JSON dict."""
    args: List[str] = ["-m", "lynx", "serve"]
    if repo_path is not None and str(repo_path) == ".":
        args.append(".")  # the folder the editor starts the server in
    elif repo_path is not None:
        args.append(str(Path(repo_path).expanduser().resolve()))
    elif config_path is not None:
        args.extend(["--config", str(Path(config_path).expanduser().resolve())])
    else:
        args.extend(["--config", "<path to config.json>"])

    if profile:
        args.extend(["--profile", profile])

    command = python_exe or sys.executable
    if python_exe is None and _running_under_uvx():
        # uvx's environment lives in a cache that uv may prune: name the
        # launcher instead, so the entry survives a cache clean or an update.
        command, args = "uvx", ["--from", "lynx-mcp", "lynx", *args[2:]]

    entry: dict = {
        "command": command,
        "args": args,
    }
    if include_stdio_type:
        entry = {"type": "stdio", **entry}
    return entry


def render_snippet(
    spec: ClientSpec,
    *,
    config_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
    profile: Optional[str] = None,
    python_exe: Optional[str] = None,
) -> str:
    """Render the copy-paste JSON snippet for a specific client."""
    entry = build_server_entry(
        config_path=config_path,
        repo_path=repo_path,
        profile=profile,
        python_exe=python_exe,
        include_stdio_type=spec.include_stdio_type,
    )
    return json.dumps({spec.root_key: {"lynx": entry}}, indent=2)


# ---------------------------------------------------------------------------
# Status inspection & safe JSON read-modify-write
# ---------------------------------------------------------------------------

# `python -m lynx serve`, the `lynx` launcher, or `uvx --from lynx-mcp`.
_LYNX_COMMAND = re.compile(
    r"(?:-m\s+lynx\b|(?:^|[\\/\s])lynx(?:\.exe)?\s+serve\b|lynx-mcp)", re.IGNORECASE,
)

_BACKUPS_KEPT = 3


def is_lynx_entry(name: str, entry) -> bool:
    """True when an MCP server entry runs Lynx, whatever it is called."""
    if not isinstance(entry, dict):
        return False
    parts = [str(entry.get("command") or "")] + [str(a) for a in (entry.get("args") or [])]
    return bool(_LYNX_COMMAND.search(" ".join(parts)))


def _lynx_names(servers) -> List[str]:
    if not isinstance(servers, dict):
        return []
    return [n for n, e in servers.items() if n == "lynx" or is_lynx_entry(n, e)]


def _read_json_dict(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Read a JSON object from `path`. Missing/empty file -> `{}`."""
    try:
        if not path.exists():
            return {}, None
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        return None, f"Could not read {path}: {e}"
    if not raw_text.strip():
        return {}, None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"{path} contains invalid JSON (line {e.lineno}, col {e.colno}): {e.msg}"
    if not isinstance(parsed, dict):
        return None, f"{path} must contain a JSON object at top level."
    return parsed, None


def _backup(path: Path) -> Path:
    """Copy `path` to a timestamped `.lynx-<time>.bak` next to it and keep
    the newest few, so a second connect never overwrites the copy of the
    file as it was before the first one."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.lynx-{stamp}.bak")
    n = 1
    while backup.exists():
        n += 1
        backup = path.with_name(f"{path.name}.lynx-{stamp}-{n}.bak")
    shutil.copyfile(path, backup)
    old = sorted(path.parent.glob(f"{path.name}.lynx-*.bak"), key=lambda q: q.stat().st_mtime)
    for stale in old[:-_BACKUPS_KEPT]:
        try:
            stale.unlink()
        except OSError:
            pass
    return backup


def _write_json_atomic(path: Path, data: dict) -> Tuple[Optional[Path], Optional[str]]:
    """Create parent dirs, back up `path` if it existed, atomically write."""
    backup: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = _backup(path)
        tmp = path.with_name(path.name + ".lynx.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        return backup, f"Could not write {path}: {e}"
    return backup, None


def inspect_client(
    spec: ClientSpec,
    *,
    home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> ClientStatus:
    """Inspect whether a client appears installed and whether Lynx is wired in."""
    cfg_path = resolve_client_config_path(spec, home=home, env=env, platform=platform)
    if cfg_path is None:
        return ClientStatus(spec=spec, config_path=None, detected=False, connected=False)

    # `Path.exists()` raises on a denied folder (only "not found" errors are
    # swallowed): one client's sandboxed or locked folder must show as that
    # client's error, not stop `lynx connect` and the Integrations page from
    # listing the others.
    try:
        present = cfg_path.exists()
        detected = present or cfg_path.parent.exists()
    except OSError as e:
        return ClientStatus(spec=spec, config_path=cfg_path, detected=False, connected=False,
                            error=f"Could not read {cfg_path}: {e}")
    if not present:
        return ClientStatus(spec=spec, config_path=cfg_path, detected=detected, connected=False)

    data, err = _read_json_dict(cfg_path)
    if err is not None or data is None:
        return ClientStatus(
            spec=spec, config_path=cfg_path, detected=True, connected=False, error=err,
        )

    servers = data.get(spec.root_key)
    locations: List[str] = []
    args: Tuple[str, ...] = ()
    names = _lynx_names(servers)
    if names:
        locations.append("user" if spec.key == "claude" else "global")
        args = tuple(str(a) for a in (servers[names[0]].get("args") or ()))
    if spec.key == "claude":
        # Claude Code keeps per-project ("local" scope) servers in the same file.
        for project, block in (data.get("projects") or {}).items():
            if isinstance(block, dict) and _lynx_names(block.get("mcpServers")):
                locations.append(f"project {project}")
    return ClientStatus(
        spec=spec,
        config_path=cfg_path,
        detected=True,
        connected=bool(locations),
        current_args=args,
        locations=tuple(locations),
    )


def inspect_all_clients(
    *,
    home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> List[ClientStatus]:
    return [
        inspect_client(spec, home=home, env=env, platform=platform)
        for spec in CLIENTS
    ]


def connect_client(client_key: str, *, hook: bool = True, **kw) -> ConnectResult:
    """Register Lynx in `client_key`'s MCP configuration (see `_connect_servers`)
    and, for Claude Code at user scope, add the prompt hook unless `hook` is
    False. Keyword arguments as `_connect_servers`."""
    res = _connect_servers(client_key, **kw)
    if not (res.ok and hook and _claude_user_scope(res.client_key, kw)):
        return res
    name = _lynx_names((_read_json_dict(res.config_path)[0] or {}).get("mcpServers"))         if res.config_path is not None and res.config_path.exists() else []
    added = install_prompt_hook(name[0] if name else "lynx", home=kw.get("home"),
                                dry_run=kw.get("dry_run", False))
    return _joined(res, added)


def disconnect_client(client_key: str, **kw) -> ConnectResult:
    """Remove Lynx from `client_key`'s configuration (see `_disconnect_servers`),
    and for Claude Code at user scope its prompt hook too."""
    res = _disconnect_servers(client_key, **kw)
    if not (res.ok and _claude_user_scope(res.client_key, kw)):
        return res
    return _joined(res, remove_prompt_hook(home=kw.get("home")))


def _claude_user_scope(client_key: str, kw) -> bool:
    return client_key == "claude" and kw.get("target_file") is None


def _joined(first: ConnectResult, second: ConnectResult) -> ConnectResult:
    """`first` with `second`'s message appended, failing if either failed."""
    return ConnectResult(
        ok=first.ok and second.ok, client_key=first.client_key,
        message=f"{first.message} {second.message}",
        config_path=first.config_path, backup_path=first.backup_path,
        dry_run=first.dry_run,
        preview_json="".join(x for x in (first.preview_json, second.preview_json) if x) or None,
    )


def _connect_servers(
    client_key: str,
    *,
    config_path: Optional[Path] = None,
    repo_path: Optional[Path] = None,
    profile: Optional[str] = None,
    target_file: Optional[Path] = None,
    dry_run: bool = False,
    python_exe: Optional[str] = None,
    home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
    url: Optional[str] = None,
) -> ConnectResult:
    """Safely merge Lynx's MCP entry into `client_key`'s configuration file.

    `url` points the client at a running `lynx serve --http` instead of a
    server started per session (Claude Code only). `which` and `runner`
    locate and run the `claude` CLI (Claude Code only); tests pass fakes.
    Without the CLI the JSON file is edited directly."""
    spec = get_client_spec(client_key)
    if spec is None:
        return ConnectResult(
            ok=False,
            client_key=client_key,
            message=f"Unknown client {client_key!r}. Supported: {', '.join(supported_client_keys(auto_only=True))}.",
        )
    if url and spec.key != "claude":
        return ConnectResult(
            ok=False, client_key=spec.key,
            message=f"--url connects Claude Code only; {spec.short_label} starts its own server.",
        )
    if not spec.supports_auto_connect and target_file is None:
        return ConnectResult(
            ok=False,
            client_key=spec.key,
            message=f"{spec.short_label} does not have a single standard config file; copy the manual snippet instead.",
        )

    dest = (
        Path(target_file).expanduser().resolve()
        if target_file is not None
        else resolve_client_config_path(spec, home=home, env=env, platform=platform)
    )
    if dest is None:
        return ConnectResult(
            ok=False, client_key=spec.key, message="Could not resolve client config path.",
        )

    data, err = _read_json_dict(dest)
    if err is not None or data is None:
        return ConnectResult(ok=False, client_key=spec.key, message=err or "Invalid config", config_path=dest)

    servers = data.get(spec.root_key)
    if servers is None:
        servers = {}
    elif not isinstance(servers, dict):
        return ConnectResult(
            ok=False,
            client_key=spec.key,
            message=f"`{spec.root_key}` in {dest} is not a JSON object — refusing to overwrite.",
            config_path=dest,
        )

    entry = {"type": "http", "url": url} if url else build_server_entry(
        config_path=config_path,
        repo_path=repo_path,
        profile=profile,
        python_exe=python_exe,
        include_stdio_type=spec.include_stdio_type,
    )
    # Update an existing Lynx entry in place, whatever its name, rather than
    # adding a second server that would run next to it.
    existing = _lynx_names(servers)
    name = existing[0] if existing else "lynx"
    note = _project_scope_note(spec, data)

    # The CLI writes the real ~/.claude.json: only when the caller did not
    # point us at another home (tests always do).
    if spec.key == "claude" and target_file is None and (home is None or which is not None):
        cli = (which or shutil.which)("claude")
        if cli:
            return _connect_with_claude_cli(cli, name, entry, dest, dry_run, note, runner)

    updated = dict(data)
    updated[spec.root_key] = {**servers, name: entry}
    preview = json.dumps(updated, indent=2) + "\n"

    if dry_run:
        return ConnectResult(
            ok=True,
            client_key=spec.key,
            message=f"[dry-run] Would write Lynx MCP entry to {dest}",
            config_path=dest,
            dry_run=True,
            preview_json=preview,
        )

    backup, write_err = _write_json_atomic(dest, updated)
    if write_err is not None:
        return ConnectResult(ok=False, client_key=spec.key, message=write_err, config_path=dest)

    return ConnectResult(
        ok=True,
        client_key=spec.key,
        message=f"Connected {spec.short_label} ({dest}). Restart {spec.short_label} to activate Lynx.{note}",
        config_path=dest,
        backup_path=backup,
    )


def _project_scope_note(spec: ClientSpec, data: dict) -> str:
    if spec.key != "claude":
        return ""
    projects = [
        p for p, block in (data.get("projects") or {}).items()
        if isinstance(block, dict) and _lynx_names(block.get("mcpServers"))
    ]
    if not projects:
        return ""
    shown = ", ".join(projects[:3]) + (" ..." if len(projects) > 3 else "")
    return (f" These projects keep their own Lynx entry, which Claude Code uses there "
            f"instead: {shown}.")


def _connect_with_claude_cli(cli, name, entry, dest, dry_run, note, runner) -> ConnectResult:
    """Register through `claude mcp add`: Claude Code rewrites ~/.claude.json
    all the time, so its own CLI is the safe writer while it runs."""
    if entry.get("type") == "http":
        add = [cli, "mcp", "add", "--transport", "http", "--scope", "user", name, entry["url"]]
    else:
        add = [cli, "mcp", "add", "--scope", "user", name, "--", entry["command"], *entry["args"]]
    if dry_run:
        return ConnectResult(
            ok=True, client_key="claude", dry_run=True, config_path=dest,
            message="[dry-run] Would run: " + " ".join(add),
        )
    run = runner or (lambda c: subprocess.run(c, capture_output=True, text=True, check=False))
    try:
        run([cli, "mcp", "remove", "--scope", "user", name])  # absent is fine
        proc = run(add)
    except OSError as e:
        return ConnectResult(ok=False, client_key="claude", config_path=dest,
                             message=f"Could not run the claude CLI: {e}")
    if proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip()
        return ConnectResult(ok=False, client_key="claude", config_path=dest,
                             message=f"`claude mcp add` failed: {detail[:300]}")
    return ConnectResult(
        ok=True, client_key="claude", config_path=dest,
        message=f"Connected Claude Code for every project (user scope). Restart Claude Code "
                f"or run /mcp to load Lynx.{note}",
    )


def _disconnect_servers(
    client_key: str,
    *,
    target_file: Optional[Path] = None,
    home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> ConnectResult:
    """Remove the Lynx entries (by what they run, whatever their name) from
    `client_key`'s global config. Claude Code project entries stay."""
    spec = get_client_spec(client_key)
    if spec is None:
        return ConnectResult(ok=False, client_key=client_key, message=f"Unknown client {client_key!r}.")

    dest = (
        Path(target_file).expanduser().resolve()
        if target_file is not None
        else resolve_client_config_path(spec, home=home, env=env, platform=platform)
    )
    if dest is None or not dest.exists():
        return ConnectResult(
            ok=True,
            client_key=spec.key,
            message=f"{spec.short_label} is already disconnected (config file does not exist).",
            config_path=dest,
        )

    data, err = _read_json_dict(dest)
    if err is not None or data is None:
        return ConnectResult(ok=False, client_key=spec.key, message=err or "Invalid config", config_path=dest)

    servers = data.get(spec.root_key)
    names = _lynx_names(servers)
    if not names:
        return ConnectResult(
            ok=True,
            client_key=spec.key,
            message=f"Lynx was not present in {dest}.",
            config_path=dest,
        )

    # The CLI writes the real ~/.claude.json: only when the caller did not
    # point us at another home (tests always do).
    if spec.key == "claude" and target_file is None and (home is None or which is not None):
        cli = (which or shutil.which)("claude")
        if cli:
            run = runner or (lambda c: subprocess.run(c, capture_output=True, text=True, check=False))
            failed = []
            for name in names:
                try:
                    if run([cli, "mcp", "remove", "--scope", "user", name]).returncode != 0:
                        failed.append(name)
                except OSError:
                    failed.append(name)
            if failed:
                return ConnectResult(ok=False, client_key=spec.key, config_path=dest,
                                     message=f"`claude mcp remove` failed for: {', '.join(failed)}")
            return ConnectResult(ok=True, client_key=spec.key, config_path=dest,
                                 message="Disconnected Claude Code (user scope).")

    new_servers = {k: v for k, v in servers.items() if k not in names}
    updated = {**data, spec.root_key: new_servers}
    backup, write_err = _write_json_atomic(dest, updated)
    if write_err is not None:
        return ConnectResult(ok=False, client_key=spec.key, message=write_err, config_path=dest)

    return ConnectResult(
        ok=True,
        client_key=spec.key,
        message=f"Disconnected {spec.short_label} ({dest}).",
        config_path=dest,
        backup_path=backup,
    )


# ---------------------------------------------------------------------------
# Claude Code prompt hook
# ---------------------------------------------------------------------------
#
# A UserPromptSubmit hook of type `mcp_tool` that calls the `prompt_hook` tool
# of the Lynx server Claude Code already runs: every request reaches the model
# with the Lynx map of the code it is about (see prompt_hook.py for the
# measured effect). It lives in the user's ~/.claude/settings.json, next to
# the user-scope server entry `connect` registers.

PROMPT_HOOK_TOOL = "prompt_hook"


def claude_settings_path(home: Optional[Path] = None) -> Path:
    return (home or Path.home()) / ".claude" / "settings.json"


def prompt_hook_entry(server_name: str = "lynx") -> dict:
    return {"type": "mcp_tool", "server": server_name, "tool": PROMPT_HOOK_TOOL,
            "input": {"prompt": "${prompt}"}}


def _is_lynx_hook(hook) -> bool:
    return (isinstance(hook, dict) and hook.get("type") == "mcp_tool"
            and hook.get("tool") == PROMPT_HOOK_TOOL)


def _without_lynx_hook(groups: list) -> list:
    """`groups` with Lynx's hook taken out; a group left with no hook goes."""
    out = []
    for group in groups:
        hooks = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(hooks, list) or not any(_is_lynx_hook(h) for h in hooks):
            out.append(group)
            continue
        kept = [h for h in hooks if not _is_lynx_hook(h)]
        if kept:
            out.append({**group, "hooks": kept})
    return out


def _settings_groups(data: dict, dest: Path) -> Tuple[Optional[dict], Optional[list], Optional[str]]:
    hooks = data.get("hooks", {})
    if not isinstance(hooks, dict):
        return None, None, f"`hooks` in {dest} is not a JSON object; refusing to overwrite."
    groups = hooks.get("UserPromptSubmit", [])
    if not isinstance(groups, list):
        return None, None, f"`hooks.UserPromptSubmit` in {dest} is not a list; refusing to overwrite."
    return hooks, groups, None


def install_prompt_hook(server_name: str = "lynx", *, home: Optional[Path] = None,
                        dry_run: bool = False) -> ConnectResult:
    """Add (or update) Lynx's UserPromptSubmit hook in ~/.claude/settings.json,
    keeping every other setting and hook; idempotent."""
    dest = claude_settings_path(home)
    data, err = _read_json_dict(dest)
    if err is not None or data is None:
        return ConnectResult(ok=False, client_key="claude", message=err or "Invalid settings",
                             config_path=dest)
    hooks, groups, err = _settings_groups(data, dest)
    if err is not None:
        return ConnectResult(ok=False, client_key="claude", message=err, config_path=dest)
    new_groups = _without_lynx_hook(groups) + [{"hooks": [prompt_hook_entry(server_name)]}]
    if new_groups == groups:
        return ConnectResult(ok=True, client_key="claude", config_path=dest,
                             message=f"The Lynx prompt hook is already in {dest}.")
    updated = {**data, "hooks": {**hooks, "UserPromptSubmit": new_groups}}
    if dry_run:
        return ConnectResult(ok=True, client_key="claude", config_path=dest, dry_run=True,
                             message=f"[dry-run] Would add the Lynx prompt hook to {dest}",
                             preview_json=json.dumps(updated, indent=2) + "\n")
    backup, write_err = _write_json_atomic(dest, updated)
    if write_err is not None:
        return ConnectResult(ok=False, client_key="claude", message=write_err, config_path=dest)
    return ConnectResult(ok=True, client_key="claude", config_path=dest, backup_path=backup,
                         message=f"Added the Lynx prompt hook to {dest}.")


def remove_prompt_hook(*, home: Optional[Path] = None) -> ConnectResult:
    """Take Lynx's hook out of ~/.claude/settings.json, and nothing else."""
    dest = claude_settings_path(home)
    if not dest.exists():
        return ConnectResult(ok=True, client_key="claude", config_path=dest,
                             message="No Lynx prompt hook to remove.")
    data, err = _read_json_dict(dest)
    if err is not None or data is None:
        return ConnectResult(ok=False, client_key="claude", message=err or "Invalid settings",
                             config_path=dest)
    hooks, groups, err = _settings_groups(data, dest)
    if err is not None:
        return ConnectResult(ok=False, client_key="claude", message=err, config_path=dest)
    new_groups = _without_lynx_hook(groups)
    if new_groups == groups:
        return ConnectResult(ok=True, client_key="claude", config_path=dest,
                             message="No Lynx prompt hook to remove.")
    new_hooks = {**hooks, "UserPromptSubmit": new_groups}
    if not new_groups:
        del new_hooks["UserPromptSubmit"]
    updated = {**data, "hooks": new_hooks}
    if not new_hooks:
        del updated["hooks"]
    backup, write_err = _write_json_atomic(dest, updated)
    if write_err is not None:
        return ConnectResult(ok=False, client_key="claude", message=write_err, config_path=dest)
    return ConnectResult(ok=True, client_key="claude", config_path=dest, backup_path=backup,
                         message=f"Removed the Lynx prompt hook from {dest}.")
