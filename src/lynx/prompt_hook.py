"""The Lynx map of a request, for Claude Code's `UserPromptSubmit` hook.

Claude Code with Lynx connected still explores with grep and cat unless it
chooses Lynx, and it chooses late or not at all. A hook of type `mcp_tool`
calls the `prompt_hook` tool of the Lynx server that is already running, with
the user's prompt, and puts the answer in front of the model before its first
turn: the places in the code the request is about, one line each.

Measured on the Sketchbook bench (lynx-zombie-bench, E10): mean cost per task
-22% against Claude Code without Lynx, -11% against Lynx connected with no
hook, from fewer API calls and less read into the context.

On The Lost Storm (214k lines, 889 test files, E14) the map was 7 test files,
2 data or doc files and no file of the fix, and made the search dearer: the
tests and data tell the symptom in the user's own words. The map now holds
code only; tests, data and docs stay one search away (E17: a fix's file in
the map's top 5 for one task of two, against none).

Two findings shaped the format (probes of 2026-11-21):
- Claude Code adds an `mcp_tool` hook's text to the context only when it is
  the hook JSON (`hookSpecificOutput.additionalContext`); plain text is
  logged as "hook success" and dropped.
- One search per sentence of the request, fused, finds the code; the whole
  request as one query ranks README and rules files first (scores near
  0.016).

Standard library only, so it can be tested without a server.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Iterable, List, Sequence

from .file_kinds import is_secondary

MAX_SENTENCES = 6
PER_SENTENCE = 4
# Asked per sentence, so that PER_SENTENCE code hits are left once tests,
# data and docs are set aside.
FETCH_PER_SENTENCE = PER_SENTENCE * 3
MAX_HITS = 12
MIN_WORDS = 4
MAX_QUERY_CHARS = 300

# Agent rules files: they mention every tool and every subsystem, so they rank
# for any request, and the model has them in context already.
SKIP_FILES = frozenset({
    "claude.md", "agents.md", "gemini.md", ".clauderules", ".cursorrules",
    ".windsurfrules", "copilot-instructions.md",
})

HEADER = ("Lynx (local code index) found these places for this request, one search "
          "per sentence. Signatures only: read just the lines you need, and use the "
          "Lynx search / find_usages tools for more.")


def sentences(prompt: str) -> List[str]:
    """The sentences of `prompt` worth one search each: at least MIN_WORDS
    words, at most MAX_SENTENCES of them, each cut at MAX_QUERY_CHARS."""
    parts = re.split(r"(?<=[.!?:])\s+|\n+", prompt or "")
    out = [p.strip(" -*`>#\t") for p in parts]
    return [p[:MAX_QUERY_CHARS] for p in out if len(p.split()) >= MIN_WORDS][:MAX_SENTENCES]


def _path(hit: dict) -> str:
    return hit.get("rel_path") or hit.get("file_path") or "?"


def _skipped(path: str) -> bool:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower() in SKIP_FILES


def fuse(ranked: Sequence[Sequence[dict]], *, limit: int = MAX_HITS) -> List[dict]:
    """Round-robin over the per-sentence result lists, so every sentence gets
    its best hits in before any gets its second; duplicates and rules files
    are dropped."""
    seen, hits = set(), []
    depth = max((len(r) for r in ranked), default=0)
    for rank in range(depth):
        for results in ranked:
            if rank >= len(results):
                continue
            hit = results[rank]
            path = _path(hit)
            key = (hit.get("source"), path, hit.get("start_line"))
            if key in seen or _skipped(path):
                continue
            seen.add(key)
            hits.append(hit)
            if len(hits) >= limit:
                return hits
    return hits


def render(hits: Iterable[dict], signature: Callable[[dict], str], *,
           show_source: bool = False) -> str:
    """One line per hit: path, line span, signature."""
    lines = []
    for hit in hits:
        span = f"L{hit.get('start_line')}-{hit.get('end_line')}" if hit.get("start_line") else ""
        where = f" [{hit.get('source')}]" if show_source and hit.get("source") else ""
        lines.append(f"- {_path(hit)}{where} {span}  {signature(hit)}".rstrip())
    return HEADER + "\n" + "\n".join(lines) if lines else ""


def hook_json(text: str) -> str:
    """The `UserPromptSubmit` hook answer. `{}` adds nothing to the context,
    which is what a prompt with nothing to map must do."""
    if not text:
        return "{}"
    return json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                              "additionalContext": text}})


def build(prompt: str, search: Callable[[str], Sequence[dict]],
          signature: Callable[[dict], str], *, show_source: bool = False) -> str:
    """Hook JSON for `prompt`, given a `search(query) -> hits` function."""
    queries = sentences(prompt)
    if not queries:
        return "{}"
    ranked = [[h for h in search(q) if not is_secondary(_path(h))][:PER_SENTENCE] for q in queries]
    hits = fuse(ranked)
    return hook_json(render(hits, signature, show_source=show_source))
