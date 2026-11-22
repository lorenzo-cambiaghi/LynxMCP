"""Unit tests for `lynx.updater` and `lynx update` CLI (100% offline)."""
from __future__ import annotations

import json
import subprocess

import pytest

from lynx.cli import main as cli_main
from lynx.updater import (
    InstallInfo,
    check_latest_version,
    detect_install_method,
    is_newer_version,
    EXIT_BLOCKED,
    RunningProcess,
    blocking_processes,
    parse_version,
    run_upgrade,
)

_UV_INFO = InstallInfo(
    method="uv",
    command=("uv", "tool", "upgrade", "lynx-mcp"),
    command_display="uv tool upgrade lynx-mcp",
    can_auto_upgrade=True,
    notes="Installed via uv tool.",
)


def test_run_upgrade_refuses_while_lynx_processes_hold_the_install():
    ran = []
    blockers = [RunningProcess(4242, r"C:\uv\tools\lynx-mcp\Scripts\python.exe -m lynx serve .")]

    rc, msg = run_upgrade(install_info=_UV_INFO, runner=ran.append, blockers=lambda: blockers)

    assert rc == EXIT_BLOCKED
    assert ran == [], "the upgrade must not start while files are held open"
    assert "4242" in msg and "Close your AI editors" in msg


def test_blocking_processes_only_on_windows_and_not_self():
    rows = [
        (10, 1, r"C:\uv\tools\lynx-mcp\Scripts\python.exe -m lynx serve --config C:\c.json"),
        (11, 1, r"C:\Users\me\.local\bin\lynx.exe manager ui"),
        (12, 1, "uvx --from lynx-mcp lynx serve ."),
        (13, 1, r"notepad.exe C:\notes\lynx.txt"),
        (15, 1, r'"C:\Program Files\Git\bin\bash.exe" -c "python -m lynx serve ."'),
        (16, 1, r'"C:\Python312\python.exe" -m lynx.integrity C:\store'),
        (14, 1, "powershell -Command Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*lynx*' }"),
        (99, 1, r"C:\uv\tools\lynx-mcp\Scripts\python.exe -m lynx update"),
    ]
    found = blocking_processes(platform="win32", lister=lambda: rows, own_pids={99})
    assert [p.pid for p in found] == [10, 11, 12, 16]
    assert blocking_processes(platform="linux", lister=lambda: rows, own_pids=set()) == []


def test_detect_install_method_recognises_uvx_cache():
    info = detect_install_method(
        executable=r"C:\Users\me\AppData\Local\uv\cache\archive-v0\abc\Scripts\python.exe",
        package_file=r"C:\Users\me\AppData\Local\uv\cache\archive-v0\abc\Lib\site-packages\lynx\updater.py",
        which=lambda name: f"C:/bin/{name}.exe",
    )
    assert info.method == "uvx"
    assert info.command[:3] == ("C:/bin/uvx.exe", "--refresh", "--from")


@pytest.mark.parametrize(
    "latest,current,expected",
    [
        ("1.12.0", "1.11.0", True),
        ("2.0.0", "1.11.0", True),
        ("1.11.1", "1.11.0", True),
        ("1.11.0", "1.11.0", False),
        ("1.10.9", "1.11.0", False),
        ("v1.12.0", "1.11.0", True),
    ],
)
def test_version_comparison(latest, current, expected):
    assert is_newer_version(latest, current) is expected
    assert isinstance(parse_version(latest), tuple)


def test_detect_install_method_variants(tmp_path):
    fake_pkg = tmp_path / "site-packages" / "lynx" / "updater.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("", encoding="utf-8")

    uv = detect_install_method(
        executable="/home/u/.local/share/uv/tools/lynx-mcp/bin/python",
        package_file=str(fake_pkg),
        which=lambda name: "/usr/bin/uv" if name == "uv" else None,
    )
    assert uv.method == "uv"
    assert uv.can_auto_upgrade is True
    assert uv.command == ("/usr/bin/uv", "tool", "upgrade", "lynx-mcp")

    pipx = detect_install_method(
        executable="/home/u/.local/pipx/venvs/lynx-mcp/bin/python",
        package_file=str(fake_pkg),
        which=lambda name: "/usr/bin/pipx" if name == "pipx" else None,
    )
    assert pipx.method == "pipx"
    assert pipx.can_auto_upgrade is True
    assert pipx.command == ("/usr/bin/pipx", "upgrade", "lynx-mcp")

    inst = detect_install_method(
        executable="/Applications/LynxMCP.app/Contents/Resources/venv/bin/python",
        package_file=str(fake_pkg),
    )
    assert inst.method == "installer"
    assert inst.can_auto_upgrade is False

    pip_env = detect_install_method(
        executable="/opt/venv/bin/python",
        package_file=str(fake_pkg),
    )
    assert pip_env.method == "pip"
    assert pip_env.can_auto_upgrade is True
    assert pip_env.command == ("/opt/venv/bin/python", "-m", "pip", "install", "--upgrade", "lynx-mcp")


def test_check_latest_version_handles_update_uptodate_and_offline(monkeypatch):
    monkeypatch.delenv("LYNX_NO_UPDATE_CHECK", raising=False)

    st_new = check_latest_version(
        force=True, current_version="1.11.0", fetcher=lambda _: "1.12.0",
    )
    assert st_new.checked is True
    assert st_new.update_available is True
    assert st_new.latest_version == "1.12.0"

    st_same = check_latest_version(
        force=True, current_version="1.11.0", fetcher=lambda _: "1.11.0",
    )
    assert st_same.checked is True
    assert st_same.update_available is False

    def _boom(_timeout):
        raise OSError("network unreachable")

    st_off = check_latest_version(force=True, current_version="1.11.0", fetcher=_boom)
    assert st_off.checked is False
    assert st_off.update_available is False
    assert "Could not reach PyPI" in (st_off.error or "")


def test_check_latest_version_respects_opt_out_env(monkeypatch):
    monkeypatch.setenv("LYNX_NO_UPDATE_CHECK", "1")
    called = []
    st = check_latest_version(force=False, fetcher=lambda _: called.append(1) or "9.9.9")
    assert st.checked is False
    assert called == []


def test_run_upgrade_dry_run_and_execution():
    info = InstallInfo(
        method="uv",
        command=("uv", "tool", "upgrade", "lynx-mcp"),
        command_display="uv tool upgrade lynx-mcp",
        can_auto_upgrade=True,
        notes="Installed via uv tool.",
    )
    rc, msg = run_upgrade(dry_run=True, install_info=info)
    assert rc == 0
    assert "uv tool upgrade lynx-mcp" in msg

    ran = []

    def _fake_runner(cmd):
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Upgraded lynx-mcp\n", stderr="")

    rc2, msg2 = run_upgrade(dry_run=False, install_info=info, runner=_fake_runner,
                            blockers=lambda: [])
    assert rc2 == 0
    assert ran == [["uv", "tool", "upgrade", "lynx-mcp"]]
    assert "upgraded successfully" in msg2.lower()


def test_cli_update_check_json(monkeypatch, capsys):
    import lynx.updater as upd

    monkeypatch.setattr(
        upd,
        "check_latest_version",
        lambda **kw: upd.UpdateStatus(
            current_version="1.11.0",
            latest_version="1.12.0",
            update_available=True,
            release_url=upd.RELEASES_URL,
            install=upd.InstallInfo("uv", ("uv", "tool", "upgrade", "lynx-mcp"), "uv tool upgrade lynx-mcp", True, ""),
            checked=True,
        ),
    )
    rc = cli_main(["update", "--check", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["update_available"] is True
    assert data["latest_version"] == "1.12.0"
