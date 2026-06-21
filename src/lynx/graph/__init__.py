"""Opt-in knowledge graph layer for code sources.

A complement to the vector store: the graph models *structural* facts (who
calls whom, what imports what, which symbols live in which container) that
similarity search cannot answer directly. Activated per-source via
`graph: { enabled: true }` in config.json.

Two public surfaces:

  - `extractor.extract_file(abs_path, content)` — pure, stateless. Reuses the
    tree-sitter pass from `lynx.chunking.parse_file`, walks the AST, and
    returns `{nodes, edges, raw_calls}` for one file. Cross-file calls live
    in `raw_calls` for the builder to resolve later.
  - `GraphLayer` (added in the next phase) — orchestrates extraction across a
    codebase, persists the graph as JSON, exposes query / analysis methods.

Kept independent from `rag_manager.py` so the graph layer remains optional:
disabling it removes a layer of behavior without touching the search path.
"""
from __future__ import annotations

from .extractor import extract_file, LangGraphRules, GRAPH_LANGUAGES
from .builder import GraphLayer, GRAPH_SCHEMA_VERSION
from .analyzer import god_nodes, communities, surprising_connections
from .query import (
    get_callers, get_callees, get_imports, get_neighbors, shortest_path,
    find_symbols, get_subclasses, get_superclasses,
    transitive_callers, nodes_in_file,
)
from .render import build_symbol_view, build_module_view, render_html

__all__ = [
    # extractor
    "extract_file", "LangGraphRules", "GRAPH_LANGUAGES",
    # builder
    "GraphLayer", "GRAPH_SCHEMA_VERSION",
    # analyzer
    "god_nodes", "communities", "surprising_connections",
    # query
    "get_callers", "get_callees", "get_imports", "get_neighbors",
    "shortest_path", "find_symbols",
    "get_subclasses", "get_superclasses",
    "transitive_callers", "nodes_in_file",
    # render
    "build_symbol_view", "build_module_view", "render_html",
]
