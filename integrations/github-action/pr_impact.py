#!/usr/bin/env python3
"""PR impact analysis via a local Lynx API.

For the files changed in a pull request, query Lynx for:
  1. cross-file callers of the symbols the change defines (the blast radius —
     what elsewhere might break), via `GET /api/v1/graph?operation=callers`;
  2. code elsewhere that is semantically related to the change, via
     `GET /api/v1/search`.

Emits a Markdown report on stdout (or `--out FILE`) that a workflow step posts
as a sticky PR comment. Pure stdlib (urllib) so it runs on a bare CI runner
with no extra dependencies.

Heuristic by design: symbol names are extracted from the changed files with
simple per-language patterns, then resolved fuzzily by Lynx. It surfaces
candidates for a human reviewer, it is not a sound static analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

MARKER = "<!-- lynx-impact-analysis -->"

CODE_EXT = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cs", ".cpp", ".cc",
    ".cxx", ".hpp", ".hxx", ".c", ".h", ".go", ".rs", ".java", ".rb",
    ".php", ".kt", ".kts", ".swift", ".sh", ".bash", ".sql", ".scala",
    ".sc", ".lua", ".m", ".mm",
}

# Best-effort declaration patterns across the languages Lynx chunks; we capture
# the declared *name*, which Lynx then resolves fuzzily.
_DECL_PATTERNS = [
    re.compile(r"\bclass\s+([A-Za-z_]\w*)"),
    re.compile(r"\b(?:struct|interface|trait|enum|object|protocol)\s+([A-Za-z_]\w*)"),
    re.compile(r"\bdef\s+([A-Za-z_]\w*)"),
    re.compile(r"\bfunc(?:tion)?\s+([A-Za-z_]\w*)"),
    re.compile(r"\bfn\s+([A-Za-z_]\w*)"),
    # C#/Java/TS-style `<modifiers> <type> Name(` method headers.
    re.compile(r"\b(?:public|private|protected|internal|static|final|override|async)\s+[\w<>\[\].]+\s+([A-Za-z_]\w*)\s*\("),
]
_NAME_STOP = {"if", "for", "while", "switch", "return", "get", "set", "new",
              "in", "of", "is", "as", "do", "to"}


def _http_ndjson(url: str, timeout: int = 30) -> list:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    return [json.loads(ln) for ln in body.splitlines() if ln.strip()]


def lynx_search(api: str, q: str, source: str | None, top_k: int = 6) -> list:
    qs = {"q": q, "top_k": str(top_k), "format": "ndjson"}
    if source:
        qs["source"] = source
    try:
        return _http_ndjson(f"{api}/api/v1/search?" + urllib.parse.urlencode(qs))
    except Exception:
        return []


def lynx_callers(api: str, symbol: str, source: str | None, limit: int = 10) -> list:
    qs = {"operation": "callers", "symbol": symbol, "limit": str(limit), "format": "ndjson"}
    if source:
        qs["source"] = source
    try:
        return _http_ndjson(f"{api}/api/v1/graph?" + urllib.parse.urlencode(qs))
    except urllib.error.HTTPError:
        return []  # no graph-enabled source, or symbol not found
    except Exception:
        return []


def changed_files(args) -> list:
    if args.changed_files:
        files = re.split(r"[,\n ]+", args.changed_files.strip())
    else:
        rng = f"{args.base}...{args.head}" if args.base and args.head else "HEAD~1...HEAD"
        out = subprocess.run(
            ["git", "diff", "--name-only", rng],
            capture_output=True, text=True, cwd=args.repo_root or ".",
        )
        files = out.stdout.split()
    return [f for f in files if os.path.splitext(f)[1].lower() in CODE_EXT]


def extract_symbols(path: str, repo_root: str, cap: int) -> list:
    try:
        with open(os.path.join(repo_root, path), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    seen: list = []
    for pat in _DECL_PATTERNS:
        for m in pat.finditer(text):
            name = m.group(1)
            if (name and name not in seen and not name[0].isdigit()
                    and name.lower() not in _NAME_STOP and len(name) > 2):
                seen.append(name)
    return seen[:cap]


def _same_file(a: str, b: str) -> bool:
    return os.path.basename(a or "") == os.path.basename(b or "")


def build_report(args) -> str:
    files = changed_files(args)
    if not files:
        return f"{MARKER}\n### 🐈 Lynx impact analysis\n\nNo indexed-language files changed."

    sections: list = []
    n_callers = 0
    for path in files[: args.max_files]:
        syms = extract_symbols(path, args.repo_root or ".", args.max_symbols)

        impacted: list = []
        for sym in syms:
            for e in lynx_callers(args.api, sym, args.source):
                cf = e.get("from_file") or ""
                if cf and not _same_file(cf, path):
                    impacted.append((sym, e.get("from_symbol", "?"), cf, e.get("call_site_line", 0)))
        impacted = impacted[:8]
        n_callers += len(impacted)

        q = " ".join(syms[:5]) or os.path.splitext(os.path.basename(path))[0]
        related = [
            h for h in lynx_search(args.api, q, args.source, top_k=8)
            if not _same_file(h.get("file", ""), path)
        ][:4]

        if not impacted and not related:
            continue
        body = [f"<details>\n<summary><code>{path}</code></summary>\n"]
        if impacted:
            body.append("\n**Downstream callers** (may be affected by this change):\n")
            for sym, frm, cf, ln in impacted:
                body.append(f"- `{frm}` — `{cf}:{ln}` → calls `{sym}`")
        if related:
            body.append("\n**Semantically related code** (consider reviewing together):\n")
            for h in related:
                body.append(
                    f"- `{h.get('symbol') or h.get('file')}` — "
                    f"`{h.get('file', '?')}:{h.get('start_line', '?')}` (~{float(h.get('score', 0)):.3f})"
                )
        body.append("\n</details>")
        sections.append("\n".join(body))

    if not sections:
        return f"{MARKER}\n### 🐈 Lynx impact analysis\n\nNo related code or downstream callers found for the changed files."

    head = (
        f"{MARKER}\n### 🐈 Lynx impact analysis\n\n"
        f"{len(files)} changed file(s) · {n_callers} downstream caller(s) found. "
        f"All analysis ran locally against an indexed copy of this repo.\n"
    )
    return head + "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser(description="PR impact analysis via a local Lynx API")
    ap.add_argument("--api", default=os.environ.get("LYNX_API", "http://127.0.0.1:8765"))
    ap.add_argument("--source", default=os.environ.get("LYNX_SOURCE"))
    ap.add_argument("--base", default=os.environ.get("BASE_SHA"))
    ap.add_argument("--head", default=os.environ.get("HEAD_SHA"))
    ap.add_argument("--changed-files", default=os.environ.get("CHANGED_FILES"),
                    help="Explicit list (comma/space/newline separated); else uses git diff base...head")
    ap.add_argument("--repo-root", default=os.environ.get("GITHUB_WORKSPACE", "."))
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--max-symbols", type=int, default=8)
    ap.add_argument("--out", default=None, help="Write the Markdown here (default: stdout)")
    args = ap.parse_args()

    report = build_report(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
