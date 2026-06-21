"""Render benchmarks/chart.svg — the headline benchmark numbers as a
hand-crafted SVG (no matplotlib dependency, deterministic output).

Reads measured.json (per-codebase token deltas) and structural.json (the
Django class-hierarchy graph demo). Both have embedded fallbacks so the chart
regenerates from a clean checkout without re-running the long benchmark.
Horizontal rows, so it scales to any number of benchmarked codebases.
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# Published per-codebase deltas — overridden by measured.json when present.
CODEBASES = [
    {"label": "Django 5.2", "language": "Python", "grep": 4150, "lynx": 1725},
    {"label": "Json.NET",   "language": "C#",     "grep": 6590, "lynx": 1540},
]
measured = BENCH_DIR / "measured.json"
if measured.exists():
    spec = json.loads(measured.read_text(encoding="utf-8"))
    CODEBASES = [
        {"label": c["label"], "language": c["language"],
         "grep": c["grep_tokens"], "lynx": c["lynx_tokens"]}
        for c in spec["codebases"]
    ]

CALLS = {"grep": 101, "lynx": 4}
structural = BENCH_DIR / "structural.json"
if structural.exists():
    s = json.loads(structural.read_text(encoding="utf-8"))
    CALLS["grep"], CALLS["lynx"] = s["grep_calls"], s["graph_calls"]


GREP, LYNX, INK, SUB, GRID = "#8b949e", "#e8742c", "#24292f", "#57606a", "#e1e4e8"
W = 1000
BX, BMAX = 232, 520   # bar start x, max bar width


def hrow(y, label, sub, grep_val, lynx_val, scale, fmt, delta, lcol=LYNX):
    """One codebase: two horizontal bars (grep, Lynx) on a shared scale, with
    value labels and a big delta on the right."""
    gw = max(4, BMAX * grep_val / scale)
    lw = max(4, BMAX * lynx_val / scale)
    return f"""
    <text x="36" y="{y+15}" font-size="14.5" font-weight="700" fill="{INK}">{label}</text>
    <text x="36" y="{y+33}" font-size="11.5" fill="{SUB}">{sub}</text>
    <text x="{BX-10}" y="{y+11}" font-size="10.5" fill="{SUB}" text-anchor="end">grep</text>
    <rect x="{BX}" y="{y}" width="{gw:.0f}" height="17" rx="3" fill="{GREP}"/>
    <text x="{BX+gw+8:.0f}" y="{y+13}" font-size="12.5" font-weight="700" fill="{INK}">{fmt.format(grep_val)}</text>
    <text x="{BX-10}" y="{y+35}" font-size="10.5" fill="{SUB}" text-anchor="end">Lynx</text>
    <rect x="{BX}" y="{y+24}" width="{lw:.0f}" height="17" rx="3" fill="{lcol}"/>
    <text x="{BX+lw+8:.0f}" y="{y+37}" font-size="12.5" font-weight="800" fill="{lcol}">{fmt.format(lynx_val)}</text>
    <text x="{W-30}" y="{y+27}" font-size="17" font-weight="800" fill="{lcol}" text-anchor="end">{delta}</text>"""


parts = []
y = 92
parts.append(
    f'<text x="36" y="{y}" font-size="14" font-weight="800" fill="{INK}">'
    f'Tokens to get the code into context '
    f'<tspan fill="{SUB}" font-weight="400">— median per question, incl. the file read grep still needs</tspan></text>')
y += 24
scale = max(c["grep"] for c in CODEBASES) or 1
for c in CODEBASES:
    delta = f"−{(1 - c['lynx'] / c['grep']):.0%}" if c["grep"] else "—"
    parts.append(hrow(y, f"{c['label']}  ({c['language']})", "tokens to answer",
                      c["grep"], c["lynx"], scale, "{:,}", delta))
    y += 58

y += 8
parts.append(f'<line x1="36" y1="{y}" x2="{W-30}" y2="{y}" stroke="{GRID}"/>')
y += 26
parts.append(
    f'<text x="36" y="{y}" font-size="14" font-weight="800" fill="{INK}">'
    f'Map a 100-class hierarchy '
    f'<tspan fill="{SUB}" font-weight="400">— Django: every descendant of Field with file:line; grep cannot do this in one shot</tspan></text>')
y += 24
parts.append(hrow(y, "Tool calls", "one grep round per class vs resolved graph edges",
                  CALLS["grep"], CALLS["lynx"], CALLS["grep"], "{}",
                  f"{CALLS['grep'] // CALLS['lynx']}× fewer"))
y += 58
H = y + 16

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,'Segoe UI',Helvetica,Arial,sans-serif">
  <rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#d0d7de"/>
  <text x="36" y="38" font-size="20" font-weight="800" fill="{INK}">Lynx vs agentic grep — measured across Python, C# and Java</text>
  <text x="36" y="60" font-size="13" fill="{SUB}">Reproducible: benchmarks/run_benchmark.py · grep baseline deliberately strong · RESULTS.md / RESULTS_csharp.md / RESULTS_java.md</text>
  {''.join(parts)}
</svg>
'''

out = BENCH_DIR / "chart.svg"
out.write_text(svg, encoding="utf-8", newline="\n")
print(f"wrote {out}")
