"""Dispatcher for `lynx manager <cmd>`.

Imported lazily by `lynx.cli._cmd_manager`. Each sub-command is itself
lazy-imported here so e.g. `lynx manager doctor` doesn't drag in FastAPI
just because the `ui` sub-module is part of the same package.
"""
from __future__ import annotations

import sys


def dispatch(sub_command: str, args) -> int:
    """Route `lynx manager <sub_command> ...` to the right module.

    Returns the exit code the CLI should use.
    """
    if sub_command == "init":
        from . import init
        return init.run_init(args)
    if sub_command == "doctor":
        from . import doctor
        return doctor.run_doctor(args)
    if sub_command == "install":
        from . import install
        return install.run_install(args)
    if sub_command == "ui":
        from .ui import app as ui_app
        return ui_app.run_ui(args)
    print(
        f"error: unknown `lynx manager` sub-command: {sub_command!r}",
        file=sys.stderr,
    )
    return 2
