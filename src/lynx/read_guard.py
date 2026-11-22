"""A read of hundreds of lines meets the outline Lynx already has.

Claude Code's PreToolUse hook (type mcp_tool) calls the `read_guard` tool of
the running Lynx server before every `Read`. When the read would return more
than READ_LINES lines of a file Lynx indexes (the whole file, or a wide
range), the hook refuses it once and gives the reason Claude sees: the file's
outline, one line per symbol with its lines, and how to read less. The same
read asked again goes through: a wide read is sometimes what is needed.

Why (lynx-zombie-bench, trace analysis of 35 runs): reading code already
found to understand it was 66% of the characters read before the first edit
on an 8.6k-line game and 79% on a 214k-line one; of those characters 2.9% were
lines the fix then changed. The costliest reads were whole files (462 lines,
24,865 characters, to see how a test rig is built) whose outline is 2,184.

Probes of 2026-11-22: the PreToolUse `mcp_tool` hook receives
`${tool_input.offset}` as an empty string when Claude gives none; a
`permissionDecision: deny` reason reaches the model as the tool's error, and
Claude read the narrower range next.

Standard library only; the server supplies the outline.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from typing import Callable, Optional

READ_LINES = 200

# On The Lost Storm Claude read code with Bash (`sed -n 225,260p`, `cat
# file`) far more than with Read (E25: 32-47k characters through Bash, under
# 1.3k through Read in the guarded runs), so a guard on Read alone never
# fired. The same rule applies to the plain readers below.
_READERS = {"cat", "nl", "less", "more", "sed", "head", "tail"}
_SED_RANGE = re.compile(r"^(\d+)(?:,(\d+|\$))?p$")
_GIT_BASH_DRIVE = re.compile(r"^/([a-zA-Z])/")


def _native(path: str) -> str:
    if os.name == "nt":
        m = _GIT_BASH_DRIVE.match(path)
        if m:
            path = f"{m.group(1).upper()}:/{path[3:]}"
    return path


def _commands(command: str) -> list:
    """The command split into simple commands, each a list of pipelines'
    argv lists: `cd x && sed -n 1,9p f | head; cat g` gives
    [[["cd", "x"]], [["sed", "-n", "1,9p", "f"], ["head"]], [["cat", "g"]]]."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    out, current, pipeline = [], [], []
    for tok in tokens + [";"]:
        if tok in (";", "&&", "||", "&"):
            if current:
                pipeline.append(current)
            if pipeline:
                out.append(pipeline)
            current, pipeline = [], []
        elif tok == "|":
            if current:
                pipeline.append(current)
            current = []
        else:
            current.append(tok)
    return out


def bash_reads(command: str, base_dir: str) -> list:
    """(absolute path, first line, last line or None for the end, tail count
    or None) for each plain read of a file the command prints in full: cat,
    nl, less, more, `sed -n 'A,Bp'`, head, tail. A reader piped into another
    command is filtered and not counted; `cd` is followed."""
    reads, cwd = [], base_dir
    for pipeline in _commands(command):
        argv = pipeline[0]
        if not argv:
            continue
        if argv[0] == "cd":
            if len(argv) > 1:
                cwd = os.path.normpath(os.path.join(cwd, _native(argv[1])))
            continue
        if len(pipeline) > 1 or argv[0] not in _READERS:
            continue
        reads.extend(_reads_of(argv, cwd))
    return reads


def _reads_of(argv: list, cwd: str) -> list:
    name, args = argv[0], argv[1:]

    def path_of(a):
        return os.path.normpath(os.path.join(cwd, _native(a)))

    if name in ("cat", "nl", "less", "more"):
        return [(path_of(a), 1, None, None) for a in args if not a.startswith("-")]
    if name == "sed":
        if "-n" not in args:
            return []
        rest = [a for a in args if a != "-n"]
        if not rest:
            return []
        script, files = rest[0], [a for a in rest[1:] if not a.startswith("-")]
        out = []
        for chunk in script.split(";"):
            m = _SED_RANGE.match(chunk.strip())
            if not m:
                continue
            first = int(m.group(1))
            last = None if m.group(2) == "$" else int(m.group(2) or first)
            out += [(path_of(f), first, last, None) for f in files]
        return out
    count, files, i = 10, [], 0
    while i < len(args):
        a = args[i]
        if a in ("-n", "--lines") and i + 1 < len(args):
            count = _number(args[i + 1].lstrip("+-")) or count
            i += 2
            continue
        if a.startswith("--lines="):
            count = _number(a.split("=", 1)[1]) or count
        elif re.match(r"^-\d+$", a):
            count = int(a[1:])
        elif not a.startswith("-"):
            files.append(a)
        i += 1
    if name == "head":
        return [(path_of(f), 1, count, None) for f in files]
    return [(path_of(f), None, None, count) for f in files]


def span_of(read: tuple, total_lines: int) -> tuple:
    """(first, last) lines a read from `bash_reads` prints of a file of
    `total_lines` lines."""
    _path, first, last, tail = read
    if tail is not None:
        return max(1, total_lines - tail + 1), total_lines
    first = max(1, first or 1)
    return first, total_lines if last is None else min(last, total_lines)


def _number(value) -> Optional[int]:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def requested_span(total_lines: int, offset, limit) -> tuple:
    """(first, last) 1-based lines a Read with this offset and limit returns."""
    first = max(1, _number(offset) or 1)
    count = _number(limit)
    last = total_lines if not count else min(total_lines, first + count - 1)
    return first, last


def decide(rel_path: str, total_lines: int, offset, limit, outline: Callable[[], str],
           already_refused: Callable[[tuple], bool]) -> str:
    """The PreToolUse hook answer for one Read: `{}` to let it run, or a
    deny with the outline as its reason."""
    first, last = requested_span(total_lines, offset, limit)
    return _deny_if_wide(rel_path, first, last, (rel_path, first, last), outline, already_refused,
                         "Read only the lines you need with offset and limit",
                         "Asking for the same read again lets it through.")


def decide_span(rel_path: str, first: int, last: int, command: str, outline: Callable[[], str],
                already_refused: Callable[[tuple], bool]) -> str:
    """The same for a Bash command that prints lines `first`-`last`."""
    return _deny_if_wide(rel_path, first, last, ("bash", command), outline, already_refused,
                         "Print only the lines you need (sed -n 'A,Bp')",
                         "Running the same command again lets it through.")


def _deny_if_wide(rel_path, first, last, key, outline, already_refused, narrower, again) -> str:
    if last - first + 1 <= READ_LINES:
        return "{}"
    if already_refused(key):
        return "{}"
    text = outline().strip()
    reason = (
        f"Lynx: this would return lines {first}-{last} of {rel_path} ({last - first + 1} lines). "
        f"Here is its outline, one line per symbol with its lines:\n{text}\n"
        f"{narrower}, or call describe_symbol with file={rel_path!r} for a symbol's body, "
        f"callers and callees. {again}"
    )
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                              "permissionDecision": "deny",
                                              "permissionDecisionReason": reason}})
