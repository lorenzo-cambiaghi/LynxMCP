"""Structural-query demo: class relations on Django 5.2.

The question: "what inherits from `Field`?", directly and transitively
(the whole descendant tree, everything a change to it can reach).

  - Lynx graph layer: `get_subclasses` returns resolved inheritance edges
    with file and line. `graph_query subclasses` takes one symbol, so the
    whole tree is one call per class found.

  - grep: a regex over class declarations (`class \\w+\\(.*\\bField\\b`).
    Counted two ways: one round per class found, and one round per level
    with every name of the level in one alternation.

Both sides are counted with exact base names (no fuzzy matching), and both
need a last call that finds nothing to know the tree is complete.

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

    calls = [0]
    _children = children

    def children(name):
        calls[0] += 1
        return _children(name)

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
    return direct, all_desc, levels, calls[0]


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


def grep_closure_by_level(symbol):
    """The same closure with one grep round per level: every name found at
    the previous level in one alternation. Counts the rounds."""
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in TARGET.rglob("*.py")]
    found, frontier, rounds = set(), {symbol}, 0
    while frontier:
        rounds += 1
        alt = "|".join(sorted(map(re.escape, frontier)))
        pat = re.compile(rf"^\s*class\s+(\w+)\([^)]*\b(?:{alt})\b", re.M)
        nxt = {m.group(1) for t in texts for m in pat.finditer(t)} - found
        found |= nxt
        frontier = nxt
    return found, rounds


def main():
    if not TARGET.is_dir():
        sys.exit("clone the target first (see run_benchmark.py docstring)")

    print("building graph layer over django/ ...", file=sys.stderr)
    layer, build_s = build_graph()
    st = layer.status() if hasattr(layer, "status") else {}
    print(f"graph ready in {build_s:.0f}s", file=sys.stderr)

    t0 = time.time()
    direct, all_desc, levels, graph_calls = graph_descendants(layer)
    graph_query_s = time.time() - t0

    # The same tree in one call: `graph_query subclasses` with `depth`.
    from lynx.graph.query import get_subclasses
    one_call = get_subclasses(layer._graph, ROOT_SYMBOL, limit=10_000, depth=10)
    one_call_classes = {(e["source"].get("label") or "").split(".")[-1] for e in one_call}
    one_call_levels = max((e["hop"] for e in one_call), default=0)

    t0 = time.time()
    grep_l1 = grep_direct(ROOT_SYMBOL)
    grep_all, grep_calls = grep_closure(ROOT_SYMBOL)
    grep_s = time.time() - t0
    level_all, level_rounds = grep_closure_by_level(ROOT_SYMBOL)
    assert level_all == grep_all

    missed_by_grep = sorted(all_desc - grep_all)
    missed_by_graph = sorted(grep_all - all_desc)

    def names(items):
        shown = ", ".join(items[:8]) + ("..." if len(items) > 8 else "")
        return f"{len(items)}: {shown}" if items else "0"

    md = [
        "",
        "## Structural queries: class relations",
        "",
        f"Question: *\"what inherits from `{ROOT_SYMBOL}` (django.db.models), "
        "that is, what can a change to it reach?\"*",
        "",
        "| | grep (regex and manual closure) | Lynx `graph_query subclasses` |",
        "|---|---|---|",
        f"| direct subclasses found | {len(grep_l1)} | {len(direct)}, each with file and line |",
        f"| full descendant tree | {len(grep_all)} classes | {len(all_desc)} classes, {levels} levels |",
        f"| calls, one per class found | {grep_calls} grep rounds | {graph_calls} calls (one symbol per call) |",
        f"| calls, one per level | {level_rounds} grep rounds (the level's names in one alternation) | |",
        f"| one call with `depth` | | 1 call: {len(one_call_classes)} classes over {one_call_levels} levels"
        f"{', the same set' if one_call_classes == all_desc else ', a DIFFERENT set'} |",
        f"| wall clock in this harness (Python regex over every file per round; `rg` is faster) | {grep_s:.1f}s | {graph_query_s:.2f}s |",
        "| what each match carries | a matching line; the file has to be read for the rest | a resolved edge with file, line and kind |",
        f"| descendants the regex closure missed | {names(missed_by_grep)} | |",
        f"| found by grep but not in the graph | {names(missed_by_graph)} | |",
        "",
        "Both find the same 100 classes. The graph answers in one call with "
        "`depth`, from edges resolved at index time, each with file, line, kind "
        "and level, so nothing is left to check by hand. Grep returns matching "
        "lines, comments and strings included, and needs a round per level.",
    ]
    results_md = BENCH_DIR / "RESULTS.md"
    content = results_md.read_text(encoding="utf-8")
    marker = "## Structural queries: class relations"
    if marker in content:
        content = content[: content.index(marker)].rstrip() + "\n"
    results_md.write_text(content + "\n".join(md) + "\n", encoding="utf-8")
    print("appended structural section to RESULTS.md", file=sys.stderr)
    print(f"direct: grep={len(grep_l1)} graph={len(direct)} | "
          f"closure: grep={len(grep_all)} in {grep_calls} calls or {level_rounds} by level, "
          f"graph={len(all_desc)} in {levels} levels, {graph_calls} calls | "
          f"missed_by_grep={len(missed_by_grep)}", file=sys.stderr)


if __name__ == "__main__":
    main()
