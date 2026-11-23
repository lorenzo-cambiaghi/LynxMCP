"""`subclasses` / `superclasses` with `depth`: a whole hierarchy in one call.

On Django 5.2 the tree under `Field` took one `graph_query subclasses` call
per class (101); with `depth` it is one call, the same 100 classes. These
tests pin that on a real graph built from a small codebase:

  1. depth 1 is what it always was: direct subtypes only
  2. depth N walks down level by level, each type tagged with its level
  3. superclasses walk up the same way
  4. a type linked to several namesake bases (`ambiguous`) comes once
  5. the dispatch layer passes `depth` and says it in the header
"""
from __future__ import annotations

from pathlib import Path

import pytest

FILES = {
    # Two classes named Base: `Mid(Base)` cannot be told apart by name, so
    # the resolver links it to both, as `ambiguous`.
    "util.py": "class Base:\n    pass\n",
    "other.py": "class Base:\n    pass\n",
    "mid.py": "from util import Base\n\nclass Mid(Base):\n    pass\n",
    "leaf.py": "from mid import Mid\n\nclass Leaf(Mid):\n    pass\n\nclass Tip(Leaf):\n    pass\n",
}


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("inherit-depth")
    code = root / "code"
    for rel, text in FILES.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    from lynx.graph import GraphLayer
    layer = GraphLayer(storage_dir=root / "graph", codebase_path=code, supported_extensions=[".py"])
    layer.rebuild(force=True)
    return layer._graph


def _names(edges, end):
    return [(e[end].get("label") or "").split(".")[-1] for e in edges]


def test_depth_one_is_direct_subtypes_only(graph):
    from lynx.graph.query import get_subclasses
    names = _names(get_subclasses(graph, "Mid"), "source")
    assert names == ["Leaf"]


def test_depth_walks_down_with_levels(graph):
    from lynx.graph.query import get_subclasses
    edges = get_subclasses(graph, "Mid", depth=5)
    levels = {(e["source"].get("label") or "").split(".")[-1]: e["hop"] for e in edges}
    assert levels == {"Leaf": 1, "Tip": 2}


def test_superclasses_walk_up(graph):
    from lynx.graph.query import get_superclasses
    edges = get_superclasses(graph, "Tip", depth=5)
    levels = {(e["target"].get("label") or "").split(".")[-1]: e["hop"] for e in edges}
    assert levels.get("Leaf") == 1 and levels.get("Mid") == 2 and levels.get("Base") == 3


def test_a_type_linked_to_namesakes_comes_once(graph):
    """`Mid` inherits from a `Base` the resolver cannot pick among two, so
    both Base nodes have it as a subclass. A tree is a set of types: Mid
    once, then Leaf and Tip below it."""
    from lynx.graph.query import get_subclasses
    edges = get_subclasses(graph, "Base", depth=5)
    names = _names(edges, "source")
    assert sorted(names) == ["Leaf", "Mid", "Tip"], names


def test_dispatch_passes_depth_and_says_so(graph):
    from lynx.graph import query as q
    from lynx.graph.dispatch import query_graph

    class _Manager:
        def get_subclasses(self, source, symbol, limit=50, depth=1):
            return q.get_subclasses(graph, symbol, limit=limit, depth=depth)

    res = query_graph(_Manager(), "demo", "subclasses", symbol="Mid", depth=3)
    assert res.ok
    assert "depth=3" in res.text.splitlines()[0]
    assert "Tip" in res.text and "level 2" in res.text
    assert [e["hop"] for e in res.data["edges"]] == [1, 2]
