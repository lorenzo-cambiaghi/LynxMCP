"""Self-update detection and safe upgrade helper for Lynx.

Stdlib-only. Never makes network calls automatically on import or page load:
network access happens ONLY when the user explicitly runs `lynx update` or
clicks "Check for updates" in LynxManager (and can be disabled entirely via
`LYNX_NO_UPDATE_CHECK=1`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import __version__

PYPI_JSON_URL = "https://pypi.org/pypi/lynx-mcp/json"
RELEASES_URL = "https://github.com/lorenzo-cambiaghi/LynxMCP/releases/latest"
_CACHE_TTL_SECONDS = 6 * 3600

# Module-level cache of the last PyPI lookup: (monotonic_ts, UpdateStatus).
_cached_check: Optional[Tuple[float, "UpdateStatus"]] = None


@dataclass(frozen=True)
class InstallInfo:
    """How this Lynx installation was installed and how to upgrade it."""

    method: str  # "uv" | "pipx" | "pip" | "installer" | "editable"
    command: Optional[Tuple[str, ...]]
    command_display: str
    can_auto_upgrade: bool
    notes: str


@dataclass(frozen=True)
class UpdateStatus:
    """Outcome of a version check against PyPI."""

    current_version: str
    latest_version: Optional[str]
    update_available: bool
    release_url: str
    install: InstallInfo
    checked: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "release_url": self.release_url,
            "checked": self.checked,
            "error": self.error,
            "install": {
                "method": self.install.method,
                "command": list(self.install.command) if self.install.command else None,
                "command_display": self.install.command_display,
                "can_auto_upgrade": self.install.can_auto_upgrade,
                "notes": self.install.notes,
            },
        }


def parse_version(version: str) -> Tuple[int, ...]:
    """Extract numeric version tuple for conservative semver comparison.

    Handles standard releases (`1.11.0`), v-prefixes (`v1.11.0`), and
    pre-release suffixes (`1.11.0rc1` -> `(1, 11, 0)`).
    """
    cleaned = (version or "").strip().lstrip("vV")
    nums = [int(n) for n in re.findall(r"\d+", cleaned.split("+")[0].split("-")[0])]
    return tuple(nums) if nums else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    """True when `latest` is strictly newer than `current`."""
    a = list(parse_version(latest))
    b = list(parse_version(current))
    width = max(len(a), len(b))
    a.extend([0] * (width - len(a)))
    b.extend([0] * (width - len(b)))
    return tuple(a) > tuple(b)


def detect_install_method(
    *,
    executable: Optional[str] = None,
    package_file: Optional[str] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> InstallInfo:
    """Infer how Lynx was installed from interpreter and package paths."""
    exe_str = (executable or sys.executable or "").replace("\\", "/")
    exe_low = exe_str.lower()
    pkg_path = Path(package_file or __file__).resolve()

    # 1. Editable / source checkout: repo has pyproject.toml two levels above src/lynx.
    repo_candidate = pkg_path.parent.parent.parent
    if (repo_candidate / "pyproject.toml").is_file() and (repo_candidate / ".git").exists():
        return InstallInfo(
            method="editable",
            command=None,
            command_display="git pull && pip install -e .",
            can_auto_upgrade=False,
            notes="Running from a local git checkout; pull the latest commits to update.",
        )

    # 2. Standalone double-click installer bundle (Windows %LOCALAPPDATA%\Lynx or macOS .app).
    if (
        "/lynxmcp.app/" in exe_low
        or "/lynx/venv/" in exe_low
        or "/lynx-app/" in exe_low
        or "/programs/lynx/" in exe_low
    ):
        return InstallInfo(
            method="installer",
            command=None,
            command_display=f"Download the latest installer from {RELEASES_URL}",
            can_auto_upgrade=False,
            notes="Installed via the desktop installer. Download and run the latest installer to upgrade; your config and indexes are preserved.",
        )

    # 3. uvx --from lynx-mcp: an environment in uv's cache. uvx keeps using
    # the version it resolved first; a refresh makes it fetch the latest.
    if "/uv/cache/" in exe_low or "/cache/uv/" in exe_low or "/archive-v0/" in exe_low:
        uvx_bin = which("uvx")
        cmd = (uvx_bin or "uvx", "--refresh", "--from", "lynx-mcp", "lynx", "--version")
        return InstallInfo(
            method="uvx",
            command=cmd if uvx_bin else None,
            command_display="uvx --refresh --from lynx-mcp lynx --version",
            can_auto_upgrade=bool(uvx_bin),
            notes="Running through uvx. The refresh downloads the latest release; "
                  "your editor picks it up the next time it starts Lynx.",
        )

    # 4. uv tool install lynx-mcp
    if "/uv/tools/" in exe_low or "/.local/share/uv/tools/" in exe_low:
        uv_bin = which("uv")
        cmd = (uv_bin or "uv", "tool", "upgrade", "lynx-mcp")
        return InstallInfo(
            method="uv",
            command=cmd if uv_bin else None,
            command_display="uv tool upgrade lynx-mcp",
            can_auto_upgrade=bool(uv_bin),
            notes="Installed via uv tool.",
        )

    # 5. pipx install lynx-mcp
    if "/pipx/venvs/" in exe_low or "/.local/pipx/" in exe_low:
        pipx_bin = which("pipx")
        cmd = (pipx_bin or "pipx", "upgrade", "lynx-mcp")
        return InstallInfo(
            method="pipx",
            command=cmd if pipx_bin else None,
            command_display="pipx upgrade lynx-mcp",
            can_auto_upgrade=bool(pipx_bin),
            notes="Installed via pipx.",
        )

    # 6. Standard pip / venv installation
    py_exe = executable or sys.executable
    return InstallInfo(
        method="pip",
        command=(py_exe, "-m", "pip", "install", "--upgrade", "lynx-mcp"),
        command_display="pip install --upgrade lynx-mcp",
        can_auto_upgrade=True,
        notes="Installed in a Python environment via pip.",
    )


def _fetch_pypi_latest(timeout: float = 3.0) -> str:
    req = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"lynx-mcp/{__version__}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    version = (data.get("info") or {}).get("version")
    if not version or not isinstance(version, str):
        raise ValueError("PyPI response did not include info.version")
    return version


def check_latest_version(
    *,
    force: bool = False,
    timeout: float = 3.0,
    current_version: Optional[str] = None,
    install_info: Optional[InstallInfo] = None,
    fetcher: Optional[Callable[[float], str]] = None,
) -> UpdateStatus:
    """Check PyPI for a newer release of `lynx-mcp`.

    Never raises: network errors or disabled checks return an `UpdateStatus`
    with `checked=False` and an informative `error`.
    """
    global _cached_check
    curr = current_version or __version__
    info = install_info or detect_install_method()

    if not force and os.environ.get("LYNX_NO_UPDATE_CHECK") in ("1", "true", "yes"):
        return UpdateStatus(
            current_version=curr,
            latest_version=None,
            update_available=False,
            release_url=RELEASES_URL,
            install=info,
            checked=False,
            error="Update check disabled by LYNX_NO_UPDATE_CHECK.",
        )

    now = time.monotonic()
    if not force and fetcher is None and _cached_check is not None:
        ts, cached = _cached_check
        if now - ts < _CACHE_TTL_SECONDS:
            return cached

    fetch_fn = fetcher or _fetch_pypi_latest
    try:
        latest = fetch_fn(timeout)
        status = UpdateStatus(
            current_version=curr,
            latest_version=latest,
            update_available=is_newer_version(latest, curr),
            release_url=RELEASES_URL,
            install=info,
            checked=True,
            error=None,
        )
        if fetcher is None:
            _cached_check = (now, status)
        return status
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
        return UpdateStatus(
            current_version=curr,
            latest_version=None,
            update_available=False,
            release_url=RELEASES_URL,
            install=info,
            checked=False,
            error=f"Could not reach PyPI: {e}",
        )


@dataclass(frozen=True)
class RunningProcess:
    pid: int
    command_line: str


# `python -m lynx ...`, `lynx serve` / `lynx manager` through the launcher,
# or `uvx --from lynx-mcp ...`.
_LYNX_CMDLINE = re.compile(
    r"(?:-m\s+lynx\b|[\\/]lynx(?:\.exe)?\"?\s+(?:serve|manager)\b|--from\s+lynx-mcp)",
    re.IGNORECASE,
)

# Exit code of `lynx update` when running Lynx processes block the upgrade.
EXIT_BLOCKED = 3


def _list_windows_processes() -> List[Tuple[int, int, str]]:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return []
    script = (
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*lynx*' } | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
        rows = json.loads(out) if out.strip() else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    if isinstance(rows, dict):
        rows = [rows]
    return [
        (int(r.get("ProcessId") or 0), int(r.get("ParentProcessId") or 0), r.get("CommandLine") or "")
        for r in rows
    ]


def blocking_processes(
    *,
    platform: Optional[str] = None,
    lister: Optional[Callable[[], List[Tuple[int, int, str]]]] = None,
    own_pids: Optional[set] = None,
) -> List[RunningProcess]:
    """Lynx processes that would make an upgrade fail half-way.

    Only Windows refuses to replace files a running process has loaded (the
    ONNX and Chroma native modules), and a failed `uv tool upgrade` leaves
    the install broken. Elsewhere the upgrade replaces files under running
    processes safely, so nothing blocks it.
    """
    if (platform or sys.platform) != "win32":
        return []
    mine = own_pids if own_pids is not None else {os.getpid(), os.getppid()}
    rows = (lister or _list_windows_processes)()
    return [
        RunningProcess(pid, cmd.strip())
        for pid, _ppid, cmd in rows
        if pid not in mine and _runs_lynx_code(cmd)
    ]


def _runs_lynx_code(cmd: str) -> bool:
    # Only interpreters and launchers load the install's files. A shell or
    # an editor whose command line merely mentions lynx holds nothing open.
    cmd = cmd.strip()
    exe = cmd[1:].split('"', 1)[0] if cmd.startswith('"') else cmd.split(" ", 1)[0]
    name = exe.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.endswith(".exe"):
        name = name[:-4]
    if not (name.startswith("python") or name in ("lynx", "uvx", "uv")):
        return False
    return bool(_LYNX_CMDLINE.search(cmd))


def describe_blockers(procs: List[RunningProcess]) -> str:
    lines = [
        f"Lynx is still running in {len(procs)} other process(es). Windows cannot "
        "replace files they hold open, so the update would stop half-way.",
        "Close your AI editors (Claude Code, Cursor, ...) and LynxManager, then run "
        "`lynx update` again.",
    ]
    for p in procs[:8]:
        lines.append(f"  pid {p.pid}: {p.command_line[:110]}")
    return "\n".join(lines)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_exit(pid: int, timeout: float = 60.0, poll: float = 0.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(poll)
    return not pid_alive(pid)


def run_upgrade(
    *,
    dry_run: bool = False,
    install_info: Optional[InstallInfo] = None,
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
    blockers: Optional[Callable[[], List[RunningProcess]]] = None,
) -> Tuple[int, str]:
    """Execute the detected upgrade command for this Lynx installation.

    Returns `(exit_code, output_text)`. Refuses with `EXIT_BLOCKED` while
    other Lynx processes hold the install open (Windows only).
    """
    info = install_info or detect_install_method()
    if not info.can_auto_upgrade or not info.command:
        return 1, f"{info.notes}\nUpgrade manually: {info.command_display}"

    cmd = list(info.command)
    if dry_run:
        return 0, f"[dry-run] would run: {' '.join(cmd)}"

    procs = (blockers or blocking_processes)()
    if procs:
        return EXIT_BLOCKED, describe_blockers(procs)

    exec_fn = runner or (
        lambda c: subprocess.run(c, capture_output=True, text=True, check=False)
    )
    try:
        proc = exec_fn(cmd)
    except OSError as e:
        return 1, f"Failed to start upgrade command ({info.command_display}): {e}"

    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode == 0:
        msg = "Lynx upgraded successfully. Restart your AI editor and LynxManager to use the new version."
        return 0, f"{msg}\n{combined}".strip()
    return proc.returncode or 1, combined or f"Upgrade command failed with exit code {proc.returncode}."
