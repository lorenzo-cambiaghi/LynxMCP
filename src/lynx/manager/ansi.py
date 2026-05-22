"""Tiny ANSI helpers — colored output without depending on `rich`.

We deliberately don't pull in `rich` (or any third-party lib) for two
reasons:

1. **Zero new deps.** LynxManager runs in CLI mode on the user's first
   contact with the tool; failing to colorize because some transitive
   dep is missing would be a bad first impression.

2. **TTY detection done right.** `rich` does this too, but we want
   explicit control: when stdout is a pipe / file (e.g. `lynx manager
   doctor > report.txt`), strip the escape codes entirely so the
   output stays readable.

Pattern:
    from .ansi import success, warn, error, bold, dim, heading, bullet

    print(heading("Diagnostic results"))
    print(success("HF cache present"))
    print(warn("Config drift: chunker_version 3 -> 4"))
"""
from __future__ import annotations

import os
import sys


# ANSI escape codes. We keep them as module-level constants rather than
# returning closures so the cost per call is just a string concat.
_ESC = "\033["
_RESET = f"{_ESC}0m"

_COLORS = {
    "black":   "30",  "red":     "31",  "green":   "32",  "yellow":  "33",
    "blue":    "34",  "magenta": "35",  "cyan":    "36",  "white":   "37",
    "bright_red":    "91",
    "bright_green":  "92",
    "bright_yellow": "93",
    "bright_blue":   "94",
}

_STYLES = {
    "bold": "1",
    "dim":  "2",
}


def _isatty() -> bool:
    """Return True only if stdout is connected to a real terminal.

    Honors `NO_COLOR` environment variable (https://no-color.org) so
    users can force-disable colors. Honors `FORCE_COLOR` for the
    opposite case (e.g. CI logs that render ANSI but report non-tty).
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def c(text: str, color: str) -> str:
    """Wrap `text` in the given ANSI color (or noop if non-tty)."""
    if not _isatty():
        return text
    code = _COLORS.get(color)
    if code is None:
        return text
    return f"{_ESC}{code}m{text}{_RESET}"


def style(text: str, name: str) -> str:
    """Apply a style (bold / dim)."""
    if not _isatty():
        return text
    code = _STYLES.get(name)
    if code is None:
        return text
    return f"{_ESC}{code}m{text}{_RESET}"


def bold(text: str) -> str:
    return style(text, "bold")


def dim(text: str) -> str:
    return style(text, "dim")


def heading(text: str) -> str:
    """A bold + cyan section header. Use sparingly — too many headings
    make the output noisy."""
    return bold(c(text, "cyan"))


def success(text: str) -> str:
    """Green check + text."""
    return f"{c('✓', 'green')} {text}"


def warn(text: str) -> str:
    """Yellow ⚠ + text."""
    return f"{c('⚠', 'yellow')} {text}"


def error(text: str) -> str:
    """Red ✗ + text."""
    return f"{c('✗', 'red')} {text}"


def bullet(text: str) -> str:
    """A dim grey bullet — useful for nested detail lines."""
    return f"  {dim('•')} {text}"
