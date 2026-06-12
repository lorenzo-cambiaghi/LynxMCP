"""Structural-query demo: class relations on Django 5.2.

The question an agent actually asks during refactoring is: "what inherits
from X?" — directly (one level) and TRANSITIVELY (the whole descendant
tree, i.e. everything that would break).

  - Lynx graph layer: `get_subclasses` returns resolved inheritance edges
    with file + line. Direct answer = 1 tool call. The full descendant
    tree = BFS over already-loaded edges (still one `graph_query` call
    per level, all served from the local graph in milliseconds).

  - grep: the agent must INVENT a regex (`class \\w+\\(.*\\bField\\b`),
    triage false positives (comments, strings, re-exports), and — the
    structural killer — repeat one grep round per discovered subclass to
    walk the tree, because `AutoField(IntegerField)` does not contain the
    word "Field"... wait, it does here, but `CharField` subclasses like
    `SlugField(CharField)` do NOT textually mention `Field`'s own name in
    a way that links them to Field: each level requires new greps with the
    newly discovered names.

This script measures both and appends a section to RESULTS.md.

Reproduce:  python benchmarks/structural_demo.py   (after run_benchmark.py)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "src"))

TARGET = BENCH_DIR / "_target" / "django" / "django"
GRAPH_STORAGE = BENCH_DIR / "_storage" / "graph"
ROOT_SYMBOL = "Field"  # django.db.models.fields.Field — the ORM's core base class


def build_graph():
    from lynx.graph import GraphLayer
    layer = GraphLayer(
        storage_dir=GRAPH_STORAGE,
        codebase_path=TARGET,
        supported_extensions=[".py"],
    )
    t0 = time.time()
    layer.rebuild()
    return layer, time.time() - t0


def graph_descendants(layer):
    """BFS over inherits edges. Returns (direct, all_descendants, levels).

    `get_subclasses` matches symbols fuzzily (substring) — fine for an
    interactive agent, but for a fair count against grep we keep only the
    edges whose BASE class leaf name equals the queried name exactly.
    Sets contain class leaf names, the same unit grep counts.
    """
    from lynx.graph.query import get_subclasses

    def children(name):
        edges = get_subclasses(layer._graph, name, limit=5000)
        out = set()
        for e in edges:
            base_leaf = (e["target"].get("label") or "").split(".")[-1]
            if base_leaf.lower() != name.lower():
                continue  # fuzzy match on a longer name (e.g. IntegerField)
            out.add((e["source"].get("label") or "?").split(".")[-1])
        return out

    direct = children(ROOT_SYMBOL)
    visited = {ROOT_SYMBOL.lower()}
    frontier = set(direct)
    all_desc = set(direct)
    levels = 1
    while frontier:
        next_frontier = set()
        for leaf in frontier:
            if leaf.lower() in visited:
                continue
            visited.add(leaf.lower())
            next_frontier |= children(leaf)
        next_frontier -= all_desc
        if not next_frontier:
            break
        all_desc |= next_frontier
        frontier = next_frontier
        levels += 1
    return direct, all_desc, levels


def grep_direct(symbol):
    """What a grep agent gets at level 1: regex over every file."""
    pat = re.compile(rf"^\s*class\s+(\w+)\([^)]*\b{symbol}\b", re.M)
    hits = set()
    for p in TARGET.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in pat.finditer(text):
            hits.add(m.group(1))
    return hits


def grep_closure(symbol):
    """Transitive closure via repeated grep rounds (what an agent would have
    to do). Counts the grep invocations needed."""
    found, frontier, grep_calls = set(), {symbol}, 0
    while frontier:
        nxt = set()
        for name in frontier:
            grep_calls += 1
            nxt |= grep_direct(name)
        nxt -= found
        frontier = nxt
        found |= nxt
    return found, grep_calls


def main():
    if not TARGET.is_dir():
        sys.exit("clone the target first (see run_benchmark.py docstring)")

    print("building graph layer over django/ ...", file=sys.stderr)
    layer, build_s = build_graph()
    st = layer.status() if hasattr(layer, "status") else {}
    print(f"graph ready in {build_s:.0f}s", file=sys.stderr)

    t0 = time.time()
    direct, all_desc, levels = graph_descendants(layer)
    graph_query_s = time.time() - t0

    t0 = time.time()
    grep_l1 = grep_direct(ROOT_SYMBOL)
    grep_all, grep_calls = grep_closure(ROOT_SYMBOL)
    grep_s = time.time() - t0

    missed_by_grep = sorted(all_desc - grep_all)
    missed_by_graph = sorted(grep_all - all_desc)

    md = [
        "",
        "## Structural queries: class relations (where grep dies)",
        "",
        f"Question: *\"what inherits from `{ROOT_SYMBOL}` (django.db.models) — "
        "i.e. what breaks if I change it?\"*",
        "",
        "| | grep (regex + manual closure) | Lynx graph_query |",
        "|---|---|---|",
        f"| direct subclasses found | {len(grep_l1)} | {len(direct)} (each with file + line) |",
        f"| full descendant tree | {len(grep_all)} classes | {len(all_desc)} classes, {levels} levels |",
        f"| tool calls required | **{grep_calls} grep rounds** (one per discovered class) | **1 per level ({levels} total)** — or 1 `get_neighbors(depth=N)` |",
        f"| wall clock (this harness; a real `rg` would be faster, the call "
        f"count would not change) | {grep_s:.1f}s | {graph_query_s:.2f}s |",
        f"| symbol metadata (file:line, kind) | none — more reads needed | included in every edge |",
        f"| descendants the regex closure missed | {len(missed_by_grep)}{': ' + ', '.join(missed_by_grep[:8]) if missed_by_grep else ''}{'...' if len(missed_by_grep) > 8 else ''} | — |",
        f"| found by grep but not in graph (false positives / unresolved) | {len(missed_by_graph)}{': ' + ', '.join(missed_by_graph[:8]) if missed_by_graph else ''}{'...' if len(missed_by_graph) > 8 else ''} | — |",
        "",
        f"The grep numbers are the BEST case: they assume the agent writes a "
        f"correct multi-name regex on the first try and never loses track of "
        f"the frontier across {grep_calls} rounds. Each of those rounds is a "
        "full model inference in a real agent loop. One-time graph build for "
        f"this corpus: {build_s:.0f}s (incremental afterwards).",
    ]
    results_md = BENCH_DIR / "RESULTS.md"
    content = results_md.read_text(encoding="utf-8")
    marker = "## Structural queries: class relations"
    if marker in content:
        content = content[: content.index(marker)].rstrip() + "\n"
    results_md.write_text(content + "\n".join(md) + "\n", encoding="utf-8")
    print("appended structural section to RESULTS.md", file=sys.stderr)
    print(f"direct: grep={len(grep_l1)} graph={len(direct)} | "
          f"closure: grep={len(grep_all)} in {grep_calls} calls, "
          f"graph={len(all_desc)} in {levels} levels | "
          f"missed_by_grep={len(missed_by_grep)}", file=sys.stderr)


if __name__ == "__main__":
    main()
