#!/usr/bin/env python3
"""Measure the token saving of `view=outline` on /api/v1/search — reproducible
field data on any indexed source.

Needs a running Lynx >= 1.7 API (which has `view=outline`) and `tiktoken`
(`pip install tiktoken`). `matplotlib` is optional, only for `--chart`.

Reproduce the docs/OUTLINE.md numbers on a public repo:

    git clone --depth 1 https://github.com/psf/requests
    # index ./requests as a Lynx codebase source named "requests", then:
    lynx manager ui --port 8765 --no-browser &
    python benchmarks/outline_tokens.py --source requests --chart docs/img/outline_tokens.png

For each query it asks the API for `view=full` and `view=outline`, counts the
tokens of each result set, and reports the saving (plus the realistic
"outline + read the one chosen body" net).
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

DEFAULT_QUERIES = [
    "send an http request and follow redirects",
    "extract cookies from a response",
    "prepare the request body and headers",
    "build the basic authorization header",
    "merge environment settings with the session",
]


def fetch(api: str, source: str | None, q: str, view: str, top_k: int) -> list:
    qs = {"q": q, "top_k": str(top_k), "view": view}
    if source:
        qs["source"] = source
    url = f"{api}/api/v1/search?" + urllib.parse.urlencode(qs)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["results"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="http://127.0.0.1:8765")
    ap.add_argument("--source", default=None, help="source name (optional if only one)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding")
    ap.add_argument("--chart", default=None, help="write a PNG bar chart here")
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding(args.encoding)
    ntok = lambda o: len(enc.encode(json.dumps(o, ensure_ascii=False)))

    labels, full, outl, net = [], [], [], []
    tf = to = tn = 0
    print(f"{'query':42s} {'full':>7} {'outline':>8} {'save':>6} {'+1 body':>8}")
    for q in args.queries:
        f_rows = fetch(args.api, args.source, q, "full", args.top_k)
        o_rows = fetch(args.api, args.source, q, "outline", args.top_k)
        f, o = ntok(f_rows), ntok(o_rows)
        b1 = len(enc.encode(f_rows[0].get("content", ""))) if f_rows else 0
        labels.append(q[:18]); full.append(f); outl.append(o); net.append(o + b1)
        tf += f; to += o; tn += o + b1
        print(f"{q[:42]:42s} {f:>7} {o:>8} {100*(f-o)/f:>5.1f}% {o+b1:>8}")
    print("-" * 76)
    print(f"TOTAL  full={tf}  outline={to}  ({100*(tf-to)/tf:.1f}% / {tf/to:.2f}x)   "
          f"net(+1 body)={tn}  ({100*(tf-tn)/tf:.1f}% / {tf/tn:.2f}x)")

    if args.chart:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(labels)); w = 0.27
        fig, ax = plt.subplots(figsize=(10, 5.2))
        b = ax.bar(x - w, full, w, label="Full bodies (default)", color="#d9534f")
        ax.bar(x, outl, w, label="Outline (signatures)", color="#5cb85c")
        ax.bar(x + w, net, w, label="Outline + read 1 body", color="#f0ad4e")
        ax.set_ylabel(f"tokens ({args.encoding})")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"Lynx search-result tokens — full vs outline (top_k={args.top_k})\n"
                     f"Total: {tf} → {to} tokens  ·  {100*(tf-to)/tf:.0f}% less for triage  ·  {tf/to:.1f}x cheaper")
        ax.bar_label(b, fmt="%d", padding=2, fontsize=8)
        ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
        fig.savefig(args.chart, dpi=130)
        print("chart:", args.chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
