"""Render benchmarks/chart.svg — the headline benchmark numbers as a
hand-crafted SVG (no matplotlib dependency, deterministic output).

Reads results.json (written by run_benchmark.py) and structural.json
(written by structural_demo.py); falls back to the published 2026-06-12
Django 5.2 numbers when the JSON artifacts aren't present, so the chart
is regenerable from a fresh checkout without re-running the long parts.
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# Published run (Django 5.2, 2026-06-12) — overridden by artifacts if present.
DATA = {
    "tokens_grep": 4150,
    "tokens_lynx": 1725,
    "calls_grep": 101,
    "calls_lynx": 4,
}

results = BENCH_DIR / "results.json"
if results.exists():
    r = json.loads(results.read_text(encoding="utf-8"))
    DATA["tokens_grep"] = r["grep"]["median_answer_tokens"]
    DATA["tokens_lynx"] = r["lynx"]["median_answer_tokens"]
structural = BENCH_DIR / "structural.json"
if structural.exists():
    s = json.loads(structural.read_text(encoding="utf-8"))
    DATA["calls_grep"] = s["grep_calls"]
    DATA["calls_lynx"] = s["graph_calls"]


GREP = "#8b949e"   # neutral gray
LYNX = "#e8742c"   # lynx amber
INK = "#24292f"
SUB = "#57606a"


def bar_panel(x, title, subtitle, grep_val, lynx_val, fmt, delta):
    """One panel: two horizontal bars with big value labels."""
    maxv = max(grep_val, lynx_val)
    full_w = 320
    g_w = max(8, full_w * grep_val / maxv)
    l_w = max(8, full_w * lynx_val / maxv)
    return f"""
  <g transform="translate({x},0)">
    <text x="0" y="24" font-size="17" font-weight="700" fill="{INK}">{title}</text>
    <text x="0" y="44" font-size="12.5" fill="{SUB}">{subtitle}</text>

    <text x="0" y="84" font-size="13" fill="{SUB}">agentic grep</text>
    <rect x="0" y="92" width="{g_w:.0f}" height="30" rx="5" fill="{GREP}"/>
    <text x="{g_w + 10:.0f}" y="113" font-size="16" font-weight="700" fill="{INK}">{fmt.format(grep_val)}</text>

    <text x="0" y="152" font-size="13" fill="{SUB}">Lynx</text>
    <rect x="0" y="160" width="{l_w:.0f}" height="30" rx="5" fill="{LYNX}"/>
    <text x="{l_w + 10:.0f}" y="181" font-size="16" font-weight="800" fill="{LYNX}">{fmt.format(lynx_val)}</text>

    <text x="0" y="218" font-size="14" font-weight="700" fill="{LYNX}">{delta}</text>
  </g>"""


tokens_delta = f"−{(1 - DATA['tokens_lynx'] / DATA['tokens_grep']):.0%} tokens, code chunks included"
calls_delta = f"{DATA['calls_grep'] // DATA['calls_lynx']}× fewer calls, same 100/100 recall"

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="330" viewBox="0 0 980 330" font-family="-apple-system,'Segoe UI',Helvetica,Arial,sans-serif">
  <rect x="0" y="0" width="980" height="330" rx="12" fill="#ffffff" stroke="#d0d7de"/>
  <g transform="translate(36,28)">
    <text x="0" y="0" font-size="20" font-weight="800" fill="{INK}">Lynx vs agentic grep — Django 5.2 (883 files, 158k lines)</text>
    <text x="0" y="22" font-size="13" fill="{SUB}">Reproducible: benchmarks/run_benchmark.py · grep baseline deliberately strong · full methodology in RESULTS.md</text>
  </g>
  <g transform="translate(36,80)">
{bar_panel(0, "Tokens to get the code into context",
           "median per question, incl. the file read grep still needs",
           DATA["tokens_grep"], DATA["tokens_lynx"], "{:,} tok", tokens_delta)}
{bar_panel(490, "Tool calls to map a class hierarchy",
           "all 100 descendants of django Field, with file:line",
           DATA["calls_grep"], DATA["calls_lynx"], "{} calls", calls_delta)}
  </g>
</svg>
"""

out = BENCH_DIR / "chart.svg"
out.write_text(svg, encoding="utf-8", newline="\n")
print(f"wrote {out}")
