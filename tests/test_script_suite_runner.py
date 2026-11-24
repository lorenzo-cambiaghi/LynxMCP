"""The script-suite runner (tests/run_script_suites.py) never lets a suite
fall back on the machine's own configuration: three suites read
./config.json and opened the real indexes until CI, which has no such file,
said so (2026-11-24)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_a_suite_asking_for_the_default_configuration_gets_none(tmp_path):
    import run_script_suites as runner
    env = runner.suite_env()
    probe = "from lynx.config import resolve_config_path; print(resolve_config_path(None))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(runner.HERE.parent), env=env,
                         capture_output=True, text=True, timeout=120)
    resolved = Path(out.stdout.strip())
    assert resolved != runner.HERE.parent / "config.json", out.stderr
    assert not resolved.exists()
    assert env["PYTHONIOENCODING"] == "utf-8"
