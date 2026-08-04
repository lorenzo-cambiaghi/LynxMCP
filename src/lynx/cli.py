"""Command-line entry point for lynx (multi-source).

Subcommands:
  serve           - run the MCP server (default if no command is given)
  build           - force a full rebuild of a source's index
  reset           - wipe a source's index and rebuild it from scratch
  search          - run an ad-hoc search query against a source
  status          - show git state, last update time, config drift per source
  list-sources    - enumerate configured sources
  source          - add / remove a source in config.json (no web UI needed)
  graph           - build, query, inspect, or export the knowledge graph
  manager         - setup wizard, doctor, extras installer, web UI
  migrate-config  - convert a v1 config.json to the v2 schema

The package version is available via `--version` at the top level.

All subcommands (except migrate-config) accept `--config PATH` to override
the default config.json resolution chain (CLI flag > RAG_CONFIG_PATH env >
./config.json).

Subcommands that operate on a single source accept `--source NAME`. When
the config has exactly one source, the flag is optional and defaults to
that source; when there are multiple, it's required.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__

# Graph operations `lynx graph query --op` accepts. Duplicated from
# `lynx.graph.dispatch.OPERATIONS` on purpose: the parser is built on EVERY
# invocation (including `lynx serve`), and importing the graph package to
# read one tuple costs 340ms warm (1.6s cold) measured with -X importtime,
# against 84ms for all of lynx.cli. The bulk is networkx, pulled in by
# graph/builder.py; tree-sitter, via graph/extractor.py, is the rest. Drift
# is caught by test_cli_graph_query.py, which asserts the two lists match.
_GRAPH_OPERATIONS = (
    "callers", "callees", "subclasses", "superclasses", "imports",
    "neighbors", "shortest_path", "overview", "surprising_connections",
    "status",
)

# Source types `lynx source add --type` accepts, mirroring the v2 config
# schema's per-type validators in config.py.
_SOURCE_TYPES = ("codebase", "webdoc", "pdf")


def _add_query_parsers(sub) -> None:
    """Register the retrieval / navigation subcommands.

    Names map 1:1 onto the MCP tools (underscores become hyphens) so the
    tool table in the docs doubles as the CLI reference — nobody should
    have to translate `find_definition` into some other spelling. The
    implementations live in `query_cli.py`; only the argument surface is
    here, next to every other subcommand.
    """
    def _common(p, *, source_help: str, with_json: bool = True):
        p.add_argument("--config", "-c", metavar="PATH")
        p.add_argument("--source", "-s", metavar="NAME", help=source_help)
        if with_json:
            p.add_argument(
                "--json", action="store_true", dest="as_json",
                help="Emit the raw result as JSON instead of text.",
            )
        return p

    _CODE_SRC = ("Codebase source to use. Optional when only one is "
                 "configured.")

    sp_deep = sub.add_parser(
        "deep-search",
        help="Multi-query escalation search: tries each phrasing until one "
             "returns strong results. Use when `search` came back weak.",
    )
    sp_deep.add_argument(
        "queries", nargs="+", metavar="QUERY",
        help="2-4 genuinely different phrasings (different angles, not "
             "paraphrases), tried in order.",
    )
    sp_deep.add_argument("--config", "-c", metavar="PATH")
    sp_deep.add_argument(
        "--source", "-s", metavar="NAME", action="append",
        help="Source to query. Repeat to fuse a subset. Omitted: every source.",
    )
    sp_deep.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw result as JSON instead of text.",
    )
    sp_deep.add_argument("--top-k", "-k", type=int, default=None)
    sp_deep.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default=None,
        help="Retrieval mode override (single-source only).",
    )
    sp_deep.add_argument("--ext", action="append", metavar="EXT")
    sp_deep.add_argument("--glob", metavar="PATTERN")
    sp_deep.add_argument("--path-contains", metavar="SUBSTRING")
    sp_deep.add_argument(
        "--min-score", type=float, default=None,
        help="Override the score a variant must beat to count as strong.",
    )
    sp_deep.add_argument(
        "--min-results", type=int, default=None,
        help="Override how many results a variant needs to count as strong.",
    )
    sp_deep.add_argument(
        "--return-all-variants", action="store_true",
        help="Include per-variant diagnostics (single-source only).",
    )

    sp_def = sub.add_parser(
        "find-definition",
        help="Where is a symbol defined? AST-precise with the graph layer, "
             "BM25 fallback otherwise.",
    )
    sp_def.add_argument("symbol", help="Identifier to locate.")
    _common(sp_def, source_help=_CODE_SRC)
    sp_def.add_argument("--limit", type=int, default=10)

    sp_use = sub.add_parser(
        "find-usages", help="Where is a symbol referenced?",
    )
    sp_use.add_argument("symbol", help="Identifier to look for.")
    _common(sp_use, source_help=_CODE_SRC)
    sp_use.add_argument("--limit", type=int, default=50)

    sp_tests = sub.add_parser(
        "find-tests-for", help="Which tests exercise a symbol?",
    )
    sp_tests.add_argument("symbol", help="Identifier under test.")
    _common(sp_tests, source_help=_CODE_SRC)
    sp_tests.add_argument("--limit", type=int, default=20)
    sp_tests.add_argument(
        "--test-path-pattern", metavar="SUBSTRING",
        help="Restrict to test files whose path contains this (e.g. `/spec/`).",
    )

    sp_sim = sub.add_parser(
        "find-similar",
        help="Find code that resembles a snippet — duplicates, parallel "
             "implementations, the other place that needs the same fix.",
    )
    sp_sim.add_argument(
        "snippet", nargs="?",
        help="Code to match. Omit and use --file for anything multi-line.",
    )
    _common(sp_sim, source_help=_CODE_SRC)
    sp_sim.add_argument(
        "--file", "-f", metavar="PATH",
        help="Read the snippet from a file instead of the command line.",
    )
    sp_sim.add_argument("--top-k", "-k", type=int, default=10)

    sp_desc = sub.add_parser(
        "describe-symbol",
        help="One-shot context: definition + called by + calls + tests.",
    )
    sp_desc.add_argument("symbol", help="Identifier to describe.")
    _common(sp_desc, source_help=_CODE_SRC)
    sp_desc.add_argument("--callers-limit", type=int, default=10)
    sp_desc.add_argument("--callees-limit", type=int, default=10)
    sp_desc.add_argument("--tests-limit", type=int, default=5)

    sp_imp = sub.add_parser(
        "impact",
        help="Blast radius: everything that transitively reaches a symbol, "
             "plus the tests to re-run.",
    )
    sp_imp.add_argument("symbol", help="Identifier whose impact to compute.")
    _common(sp_imp, source_help=_CODE_SRC)
    sp_imp.add_argument("--max-depth", type=int, default=3,
                        help="Call-graph hops to walk outward (1-6).")
    sp_imp.add_argument("--tests-limit", type=int, default=10)

    sp_ovw = sub.add_parser(
        "repo-overview",
        help="Orientation map: languages, frameworks, entry points.",
    )
    _common(sp_ovw, source_help=_CODE_SRC)

    sp_mod = sub.add_parser(
        "module-summary",
        help="What a file contains and how it connects: symbols, imports, "
             "dependents.",
    )
    sp_mod.add_argument("file", metavar="FILE",
                        help="File path (or a symbol inside it).")
    _common(sp_mod, source_help=_CODE_SRC)
    sp_mod.add_argument("--limit", type=int, default=40)

    sp_diff = sub.add_parser(
        "search-diff",
        help="Search only the files you changed vs a base branch.",
    )
    sp_diff.add_argument("query", help="Natural-language search query.")
    _common(sp_diff, source_help="Git-enabled codebase source.")
    sp_diff.add_argument(
        "--base", metavar="REF",
        help="Base branch. Default: auto-detected main / master / develop.",
    )
    sp_diff.add_argument("--top-k", "-k", type=int, default=8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lynx",
        description="Lynx — self-hosted MCP server with semantic + lexical "
                    "search over your local code and documentation. Multi-source, "
                    "100 percent local, no data egress.",
    )
    parser.add_argument(
        "--version", action="version", version=f"lynx {__version__}"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sp_serve = sub.add_parser("serve", help="Run the MCP server (default)")
    sp_serve.add_argument("--config", "-c", metavar="PATH")

    sp_build = sub.add_parser("build", help="Force a full rebuild of a source's index")
    sp_build.add_argument("--config", "-c", metavar="PATH")
    sp_build.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to rebuild. Optional when only one source is configured.",
    )

    sp_reset = sub.add_parser(
        "reset",
        help="Wipe a source's index and rebuild from scratch (fixes a corrupt index)",
    )
    sp_reset.add_argument("--config", "-c", metavar="PATH")
    sp_reset.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to reset. Optional when only one source is configured.",
    )
    sp_reset.add_argument(
        "--all", action="store_true", help="Reset every configured source.",
    )
    sp_reset.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt.",
    )
    sp_reset.add_argument(
        "--no-rebuild", action="store_true",
        help="Only wipe the index; don't rebuild (a later build/launch will).",
    )

    sp_search = sub.add_parser("search", help="Run an ad-hoc search query")
    sp_search.add_argument("query", help="Natural-language search query")
    sp_search.add_argument("--config", "-c", metavar="PATH")
    sp_search.add_argument(
        "--source", "-s", metavar="NAME", action="append",
        help="Source to query. Optional when only one source is configured. "
             "Repeat to fuse a subset (-s api -s docs); use 'ALL' for every "
             "source.",
    )
    sp_search.add_argument("--top-k", "-k", type=int, default=None)
    sp_search.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default=None,
        help="Per-call retrieval mode override (only used for direct search, "
             "not cross-source).",
    )
    sp_search.add_argument("--ext", action="append", metavar="EXT")
    sp_search.add_argument("--glob", metavar="PATTERN")
    sp_search.add_argument("--path-contains", metavar="SUBSTRING")
    sp_search.add_argument(
        "--outline", action="store_true",
        help="Return signatures + first doc line instead of full bodies — "
             "cheap triage for a broad query or a large --top-k.",
    )
    sp_search.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw hits as JSON instead of text (for scripts).",
    )

    _add_query_parsers(sub)

    sp_status = sub.add_parser("status", help="Show RAG status per source")
    sp_status.add_argument("--config", "-c", metavar="PATH")
    sp_status.add_argument(
        "--source", "-s", metavar="NAME",
        help="Inspect only this source. Default: all sources.",
    )

    sp_list = sub.add_parser("list-sources", help="Enumerate configured sources")
    sp_list.add_argument("--config", "-c", metavar="PATH")

    sp_graph = sub.add_parser(
        "graph",
        help="Build, query, inspect, or export the per-source knowledge "
             "graph layer (opt-in via `graph: { enabled: true }` in the "
             "source config).",
    )
    graph_sub = sp_graph.add_subparsers(dest="graph_command", metavar="GRAPH_COMMAND")

    sp_graph_build = graph_sub.add_parser(
        "build", help="Rebuild the graph layer for a source"
    )
    sp_graph_build.add_argument("--config", "-c", metavar="PATH")
    sp_graph_build.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source whose graph to rebuild. Optional when only one source "
             "has the graph layer enabled.",
    )
    sp_graph_build.add_argument(
        "--force", action="store_true",
        help="Wipe state and rebuild from scratch (default: SHA-incremental).",
    )

    sp_graph_status = graph_sub.add_parser(
        "status", help="Show graph layer status (nodes/edges/by-language/...)"
    )
    sp_graph_status.add_argument("--config", "-c", metavar="PATH")
    sp_graph_status.add_argument(
        "--source", "-s", metavar="NAME",
        help="Show status for a single source. Default: all sources with "
             "the graph layer enabled.",
    )

    sp_graph_query = graph_sub.add_parser(
        "query",
        help="Ask the graph a question (callers, callees, subclasses, "
             "shortest_path, overview, ...) — the CLI face of the "
             "`graph_query` MCP tool.",
    )
    sp_graph_query.add_argument("--config", "-c", metavar="PATH")
    sp_graph_query.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to query. Optional when only one has the graph layer.",
    )
    sp_graph_query.add_argument(
        "--op", "--operation", dest="op", metavar="OP", required=True,
        choices=list(_GRAPH_OPERATIONS),
        help="Operation to run: " + " | ".join(_GRAPH_OPERATIONS) + ".",
    )
    sp_graph_query.add_argument(
        "--symbol", metavar="NAME",
        help="Symbol the operation acts on (required for callers, callees, "
             "subclasses, superclasses, imports, neighbors, shortest_path). "
             "Matching is fuzzy: case-insensitive substring.",
    )
    sp_graph_query.add_argument(
        "--target", metavar="NAME",
        help="Destination symbol for --op shortest_path.",
    )
    sp_graph_query.add_argument(
        "--relation", metavar="REL", dest="relation_filter",
        help="For --op neighbors: keep only this edge relation "
             "(calls | imports | imports_from | contains | inherits).",
    )
    sp_graph_query.add_argument(
        "--depth", type=int, default=1,
        help="For --op neighbors: hops to traverse (default 1).",
    )
    sp_graph_query.add_argument(
        "--limit", type=int, default=50,
        help="Maximum edges/results to return (default 50).",
    )
    sp_graph_query.add_argument(
        "--max-hops", type=int, default=8,
        help="For --op shortest_path: longest path to search (default 8).",
    )
    sp_graph_query.add_argument(
        "--top-n", type=int, default=10,
        help="For --op overview / surprising_connections (default 10).",
    )
    sp_graph_query.add_argument(
        "--min-community-size", type=int, default=3,
        help="For --op overview: smallest cluster to report (default 3).",
    )
    sp_graph_query.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Emit the raw result (edges, path, counts) as JSON instead of "
             "the text an AI client would receive.",
    )

    sp_graph_export = graph_sub.add_parser(
        "export",
        help="Write a self-contained graph view (single offline .html) for a "
             "symbol's blast radius or a file's dependencies.",
    )
    sp_graph_export.add_argument("--config", "-c", metavar="PATH")
    sp_graph_export.add_argument(
        "--source", "-s", metavar="NAME",
        help="Source to read. Optional when only one has the graph layer.",
    )
    grp = sp_graph_export.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbol", metavar="NAME", help="Render a symbol's blast radius.")
    grp.add_argument("--module", metavar="FILE", help="Render a file's import/dependent hub.")
    sp_graph_export.add_argument(
        "--depth", type=int, default=2, help="Call-graph hops for --symbol (1-6, default 2).",
    )
    sp_graph_export.add_argument(
        "--out", "-o", metavar="PATH",
        help="Output file. Default: <reports_path or storage/reports>/<name>.html",
    )

    # --------------------------------------------------------------
    # `lynx source <cmd>` — add/remove sources without the web UI.
    # Writes config.json through the same validator the UI uses, so a
    # headless or scripted install never needs a browser.
    # --------------------------------------------------------------
    sp_source = sub.add_parser(
        "source",
        help="Add or remove a source in config.json (the CLI equivalent of "
             "the web UI's guided form).",
    )
    source_sub = sp_source.add_subparsers(dest="source_command", metavar="SOURCE_COMMAND")

    sp_src_add = source_sub.add_parser(
        "add",
        help="Add a source. The config is validated before it is written; "
             "on failure the existing file is left untouched.",
    )
    sp_src_add.add_argument("name", help="Name for the new source (letters, digits, underscore).")
    sp_src_add.add_argument("--config", "-c", metavar="PATH")
    sp_src_add.add_argument(
        "--type", "-t", choices=list(_SOURCE_TYPES), dest="source_type",
        help="Source type. Required unless --block is used.",
    )
    sp_src_add.add_argument(
        "--block", metavar="JSON", dest="block_json",
        help="Raw JSON source block, bypassing the per-type flags below. "
             "Escape hatch for fields the flags don't cover. Codebase blocks "
             "still receive the default ignore list unless they set "
             "`ignored_path_fragments` themselves.",
    )
    sp_src_add.add_argument(
        "--build", action="store_true",
        help="Index the source immediately after adding it "
             "(equivalent to a following `lynx build --source NAME`).",
    )
    sp_src_add.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Report the outcome as JSON instead of text (for scripts).",
    )

    g_code = sp_src_add.add_argument_group("codebase options")
    g_code.add_argument("--path", metavar="DIR", help="Directory to index (codebase, pdf).")
    g_code.add_argument(
        "--ext", action="append", metavar="EXT",
        help="File extension to index, repeatable (e.g. --ext .cs --ext .py). "
             "Omitted: the folder is scanned and its top extensions are used.",
    )
    g_code.add_argument(
        "--ignore", action="append", metavar="FRAGMENT",
        help="Path fragment to skip, repeatable (e.g. --ignore /Library/). "
             "Omitted: the standard VCS/deps/build ignore list is applied.",
    )
    g_code.add_argument(
        "--graph", action="store_true", help="Enable the knowledge graph layer.",
    )
    g_code.add_argument(
        "--no-watcher", action="store_true",
        help="Don't watch the folder for changes (watcher is on by default "
             "for codebase sources, off for pdf).",
    )
    g_code.add_argument(
        "--watcher-debounce", type=float, metavar="SECONDS",
        help="Seconds to coalesce rapid edits before re-indexing.",
    )
    g_code.add_argument(
        "--no-git", action="store_true",
        help="Disable git integration (commit-based staleness detection).",
    )

    g_web = sp_src_add.add_argument_group("webdoc options")
    g_web.add_argument("--url", metavar="URL", help="Crawl starting point (webdoc).")
    g_web.add_argument("--max-depth", type=int, metavar="N", help="Crawl depth (default 3).")
    g_web.add_argument("--max-pages", type=int, metavar="N", help="Page cap (default 500).")
    g_web.add_argument(
        "--request-delay", type=float, metavar="SECONDS",
        help="Politeness delay between requests (default 0.5).",
    )
    g_web.add_argument(
        "--include-url", action="append", metavar="PATTERN",
        help="Only crawl URLs matching this pattern, repeatable.",
    )
    g_web.add_argument(
        "--exclude-url", action="append", metavar="PATTERN",
        help="Skip URLs matching this pattern, repeatable.",
    )
    g_web.add_argument(
        "--render-js", action="store_true",
        help="Render pages with headless Chromium (needs the `webdoc-js` extra).",
    )
    g_web.add_argument(
        "--allow-cross-origin", action="store_true",
        help="Follow links off the starting host (default: same origin only).",
    )

    g_pdf = sp_src_add.add_argument_group("pdf options")
    g_pdf.add_argument(
        "--no-recursive", action="store_true",
        help="Only read PDFs directly in --path, not in sub-folders.",
    )
    g_pdf.add_argument(
        "--file-glob", metavar="PATTERN", help="PDF match pattern (default `**/*.pdf`).",
    )
    g_pdf.add_argument(
        "--extractor", choices=["auto", "pypdf", "pymupdf"],
        help="PDF text extraction backend (default auto).",
    )
    g_pdf.add_argument(
        "--watcher", action="store_true",
        help="Watch the folder for new/changed PDFs (off by default: "
             "re-extraction costs 10-30s per file).",
    )

    sp_src_rm = source_sub.add_parser(
        "remove",
        help="Remove a source from config.json. The index on disk is kept "
             "unless --purge is given.",
    )
    sp_src_rm.add_argument("name", help="Source to remove.")
    sp_src_rm.add_argument("--config", "-c", metavar="PATH")
    sp_src_rm.add_argument(
        "--purge", action="store_true",
        help="Also delete the source's index directory under storage_path.",
    )
    sp_src_rm.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt.",
    )
    sp_src_rm.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Report the outcome as JSON instead of text (for scripts).",
    )

    # --------------------------------------------------------------
    # `lynx manager <cmd>` — LynxManager (setup wizard, doctor, install,
    # web UI). All four sub-commands lazy-import the manager package so
    # `lynx serve` doesn't pay the FastAPI / huggingface_hub import cost
    # when only the MCP server is needed.
    # --------------------------------------------------------------
    sp_manager = sub.add_parser(
        "manager",
        help="LynxManager: interactive setup wizard, diagnostic, "
             "extras/model installer, and local web UI.",
    )
    manager_sub = sp_manager.add_subparsers(
        dest="manager_command", metavar="MANAGER_COMMAND",
    )

    sp_mgr_init = manager_sub.add_parser(
        "init",
        help="Bootstrap a fresh Lynx install: write a default config.json "
             "and pre-download the embedding model. Sources are added "
             "afterwards via `lynx manager ui` (guided form).",
    )
    sp_mgr_init.add_argument(
        "--output", "-o", metavar="PATH", default="config.json",
        help="Where to write the generated config (default ./config.json).",
    )
    sp_mgr_init.add_argument(
        "--non-interactive", action="store_true",
        help="Skip prompts; overwrite any existing config; do not offer to "
             "launch the UI. Useful in scripts / CI.",
    )
    sp_mgr_init.add_argument(
        "--skip-model-download", action="store_true",
        help="Don't pre-download the embedding model. It will be fetched "
             "lazily on the first `lynx serve` query instead.",
    )

    sp_mgr_doctor = manager_sub.add_parser(
        "doctor",
        help="Run diagnostic checks: HF cache, drift, paths, deps, "
             "watcher health. Exit code = 0 ok, 1 warn, 2 error.",
    )
    sp_mgr_doctor.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_doctor.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of colored text.",
    )
    sp_mgr_doctor.add_argument(
        "--heal-wal", metavar="SOURCE",
        help="Surgically heal SOURCE's wedged index WAL (writes left stuck "
             "by a killed process make ChromaDB hang on every open). Purges "
             "the stuck writes and queues the affected files for re-indexing "
             "— no full rebuild. Stop all Lynx processes first.",
    )
    sp_mgr_doctor.add_argument(
        "--heal-coverage", metavar="SOURCE",
        help="Re-queue SOURCE's files that the SHA cache lists as indexed but "
             "that are absent from the index (a failed insert used to be "
             "recorded as success, so those files were skipped forever and "
             "were invisible to search). Drops just those cache entries — no "
             "full rebuild. Stop all Lynx processes first.",
    )

    sp_mgr_install = manager_sub.add_parser(
        "install",
        help="Manage optional extras (pip) and HuggingFace model downloads.",
    )
    install_group = sp_mgr_install.add_mutually_exclusive_group()
    install_group.add_argument(
        "--list", action="store_true",
        help="List available optional extras and which are installed.",
    )
    install_group.add_argument(
        "--model", metavar="MODEL_NAME", nargs="?", const="__default__",
        help="Download a HuggingFace model into the local cache. With no "
             "value, downloads the embedding model from the active config.",
    )
    install_group.add_argument(
        "--from-archive", metavar="PATH_OR_URL",
        help="Import a model archive into the local HF cache (offline / "
             "air-gapped). Accepts a local path or a direct http(s) URL that "
             "serves the file with no auth/interstitial (e.g. a public GitHub "
             "Release asset). The archive must contain the `models--ORG--NAME/` "
             "layout produced by --export-archive.",
    )
    install_group.add_argument(
        "--export-archive", metavar="PATH",
        help="Zip the cached model's HF directory to PATH so it can be shared "
             "(copy it to the offline machine, or host it as a public download) "
             "and imported elsewhere with --from-archive.",
    )
    install_group.add_argument(
        "extra", nargs="?", metavar="EXTRA",
        help="Optional extra to install via pip (e.g. `pdf-fast`). "
             "Equivalent to `pip install lynx[<extra>]`.",
    )
    sp_mgr_install.add_argument(
        "--with-reranker", action="store_true",
        help="When used with --model, also download the reranker model.",
    )
    sp_mgr_install.add_argument(
        "--model-name", metavar="NAME",
        help="Model the archive is for (used with --from-archive / "
             "--export-archive). Defaults to the config's embedding model.",
    )
    sp_mgr_install.add_argument(
        "--config", "-c", metavar="PATH",
        help="Config file to read for model name detection (used by --model).",
    )

    sp_mgr_feedback = manager_sub.add_parser(
        "feedback",
        help="Summarize the local feedback log (reports agents filed when the "
             "index couldn't answer). Read-only; nothing leaves your machine.",
    )
    sp_mgr_feedback.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_feedback.add_argument(
        "--limit", type=int, default=10,
        help="How many recent reports to show (default 10).",
    )
    sp_mgr_feedback.add_argument(
        "--json", action="store_true",
        help="Output the summary as JSON instead of colored text.",
    )

    sp_mgr_ui = manager_sub.add_parser(
        "ui",
        help="Launch the local web UI (FastAPI + HTMX). Listens only on "
             "127.0.0.1; opens your browser automatically.",
    )
    sp_mgr_ui.add_argument("--config", "-c", metavar="PATH")
    sp_mgr_ui.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default 8765, falls back to next free).",
    )
    sp_mgr_ui.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default 127.0.0.1 — localhost-only by design).",
    )
    sp_mgr_ui.add_argument(
        "--no-browser", action="store_true",
        help="Don't open the browser automatically.",
    )

    sp_mig = sub.add_parser(
        "migrate-config",
        help="Convert a v1 config.json to the v2 schema",
    )
    sp_mig.add_argument(
        "--input", "-i", metavar="PATH", required=True,
        help="v1 config.json to read",
    )
    sp_mig.add_argument(
        "--output", "-o", metavar="PATH",
        help="Where to write the v2 config. Default: alongside input as "
             "<input>.v2.json",
    )
    sp_mig.add_argument(
        "--source-name", default="codebase",
        help="Name to give the migrated source in the v2 schema (default: 'codebase')",
    )

    return parser


# ----------------------------------------------------------------------
# Subcommand: migrate-config (v1 → v2)
# ----------------------------------------------------------------------


def _cmd_migrate_config(args) -> int:
    """Read a v1 config.json and produce a v2 equivalent.

    v1 had a flat schema with `codebase_path` + per-codebase fields at the
    top level. v2 nests those under `sources.<name>`. Anything that wasn't
    in v1 falls back to the v2 defaults.
    """
    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[migrate] input file not found: {in_path}", file=sys.stderr)
        return 1

    try:
        # utf-8-sig: a v1 config hand-edited on Windows may carry a BOM, and
        # failing the migration on it would be a dead end — v1 is exactly the
        # config the user can no longer load any other way.
        raw = json.loads(in_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        print(f"[migrate] invalid JSON in {in_path}: {e}", file=sys.stderr)
        return 1

    if "sources" in raw and raw.get("config_version") == 2:
        print(
            f"[migrate] {in_path} already looks like a v2 config "
            "(has 'sources' and 'config_version': 2). Nothing to do.",
            file=sys.stderr,
        )
        return 0

    if "codebase_path" not in raw:
        print(
            f"[migrate] {in_path} does not look like a v1 config "
            "(no top-level 'codebase_path'). Aborting to avoid corrupting it.",
            file=sys.stderr,
        )
        return 1

    # Validate source name early — same regex as the loader.
    import re
    SOURCE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$")
    if not SOURCE_NAME_RE.match(args.source_name):
        print(
            f"[migrate] --source-name {args.source_name!r} is invalid. "
            "Allowed: letter followed by letters / digits / underscore (max 40).",
            file=sys.stderr,
        )
        return 1

    # Build the v2 structure.
    v2 = {
        "config_version": 2,
        "storage_path": raw.get("storage_path", "./rag_storage"),
        "loading_timeout_seconds": raw.get("loading_timeout_seconds", 600),
        "embedding": raw.get("embedding", {"model_name": "BAAI/bge-small-en-v1.5"}),
        "search": raw.get("search", {}),
        "sources": {
            args.source_name: {
                "type": "codebase",
                "path": raw["codebase_path"],
                "supported_extensions": raw.get("supported_extensions", [".py", ".md", ".txt"]),
                "ignored_path_fragments": raw.get("ignored_path_fragments", []),
                "watcher": raw.get("watcher", {"enabled": True, "debounce_seconds": 2.0}),
                "git_integration": raw.get("git_integration", {"enabled": True}),
            },
        },
    }

    # Default output: alongside the input as <stem>.v2.json
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + ".v2.json")
    out_path.write_text(json.dumps(v2, indent=2) + "\n", encoding="utf-8")

    print(f"[migrate] wrote v2 config to: {out_path}", file=sys.stderr)
    print(
        f"[migrate] review it, then either replace {in_path.name} or point "
        f"your launcher at the new file via --config or RAG_CONFIG_PATH.",
        file=sys.stderr,
    )
    print(
        "[migrate] NOTE: your existing 'rag_storage/' (built under v1) "
        f"is now at the wrong layout for v2. After replacing the config, "
        "delete rag_storage/ and run 'lynx build' to "
        "rebuild under the per-source storage layout.",
        file=sys.stderr,
    )
    return 0


# ----------------------------------------------------------------------
# Helpers shared by serve / build / search / status / list-sources
# ----------------------------------------------------------------------


def _load_config_or_exit(config_path):
    from .config import load_config
    return load_config(config_path=config_path)


def _resolve_source(args, config, allow_all: bool = False) -> str:
    """Pick a source name from --source or the config's first/only source.

    If `allow_all` is True, the literal string 'ALL' is returned untouched
    (used by `search --source ALL` for cross-source RRF).
    """
    requested = getattr(args, "source", None)
    available = list(config.sources.keys())

    if allow_all and requested == "ALL":
        return "ALL"

    if requested is not None:
        if requested not in config.sources:
            print(
                f"[cli] unknown source {requested!r}. Available: "
                f"{available}",
                file=sys.stderr,
            )
            sys.exit(1)
        return requested

    if len(available) == 1:
        return available[0]

    print(
        f"[cli] this command needs --source NAME (config has {len(available)} "
        f"sources: {available})",
        file=sys.stderr,
    )
    sys.exit(1)


def _build_manager(config_path):
    """Construct SourceManager synchronously. Used by build / search / status /
    list-sources. The MCP `serve` subcommand uses its own threaded loader."""
    config = _load_config_or_exit(config_path)
    # Decide HF offline mode BEFORE the heavy imports freeze the env flags.
    from .config import configure_hf_offline
    configure_hf_offline(config)
    from .source_manager import SourceManager
    return config, SourceManager(config)


# ----------------------------------------------------------------------
# Subcommand dispatch
# ----------------------------------------------------------------------


def _cmd_serve(args) -> int:
    from .server import run_server
    run_server(config_path=getattr(args, "config", None))
    return 0


def _cmd_build(args) -> int:
    """Build or refresh a specific source.

    Always calls update(force=True). The pre-M2 behavior probed for an
    existing metadata.json and skipped update() on first install to avoid
    double-indexing, but that's now redundant: the SHA-256 cache (M2)
    makes the second update a fast no-op when nothing on disk has changed.

    Keeping it unconditional ALSO fixes a subtle bug for webdoc sources:
    their __init__ builds an empty index (no auto-fetch), so the "skip on
    first build" path would never trigger the actual crawl. Calling
    update(force=True) is the only entry point that fetches new doc pages.
    """
    config_path = getattr(args, "config", None)
    config = _load_config_or_exit(config_path)
    source_name = _resolve_source(args, config)
    _, manager = _build_manager(config_path)
    if source_name in getattr(manager, "broken", {}):
        info = manager.broken[source_name]
        if info.get("health") == "in_use":
            print(
                f"[cli] source {source_name!r} is busy: {info['error']}. "
                f"Stop the other Lynx process (usually `lynx serve`) and retry.",
                file=sys.stderr,
            )
        elif info.get("health") == "unverified":
            print(
                f"[cli] source {source_name!r} could not be verified "
                f"({info['error']}). Close any other running Lynx process and "
                f"retry; use `lynx reset --source {source_name}` only if it "
                f"persists.",
                file=sys.stderr,
            )
        else:
            print(
                f"[cli] source {source_name!r} has a corrupt index and can't be "
                f"incrementally built. Run `lynx reset --source {source_name}` to "
                f"wipe and rebuild it from scratch.",
                file=sys.stderr,
            )
        return 1
    manager.update(source_name, force=True)
    print(f"Source {source_name!r} ready.")
    return 0


def _cmd_reset(args) -> int:
    """Wipe a source's index and rebuild it. The remedy for a corrupt /
    version-incompatible index — the data is disposable derived embeddings."""
    config = _load_config_or_exit(getattr(args, "config", None))
    if getattr(args, "all", False):
        targets = list(config.sources.keys())
    else:
        targets = [_resolve_source(args, config)]
    if not targets:
        print("[cli] no sources configured to reset", file=sys.stderr)
        return 1

    rebuild = not getattr(args, "no_rebuild", False)
    if not getattr(args, "yes", False) and sys.stdin.isatty():
        what = ", ".join(targets)
        verb = "wipe and rebuild" if rebuild else "wipe"
        answer = input(f"This will {verb} the index for [{what}]. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    _, manager = _build_manager(getattr(args, "config", None))
    for name in targets:
        print(f"Resetting {name!r}: wiping index...", flush=True)
        try:
            status = manager.reset_source(name, rebuild=rebuild)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if rebuild:
            print(f"  rebuilt -> {status.get('chunk_count', 'n/a')} chunks. "
                  f"Source {name!r} ready.")
        else:
            print(f"  wiped. Run `lynx build --source {name}` to rebuild.")
    return 0


def _cmd_search(args) -> int:
    as_json = getattr(args, "as_json", False)
    with _muted_stdout(as_json):
        config, manager = _build_manager(getattr(args, "config", None))
    top_k = args.top_k if args.top_k is not None else config.search.default_top_k
    filters = dict(file_glob=args.glob, extensions=args.ext,
                   path_contains=args.path_contains)

    # A repeated --source fuses just those sources, the request-time way to
    # scope a query to a subset. One name, 'ALL', and omitting it keep
    # behaving exactly as before.
    requested = list(args.source or [])
    if len(requested) > 1:
        unknown = [n for n in requested if n not in config.sources]
        if unknown:
            msg = (f"unknown source(s) {unknown}. "
                   f"Available: {list(config.sources)}")
            if as_json:
                _emit_json({"ok": False, "operation": "search",
                            "source": None, "error": msg})
            else:
                print(f"[cli] {msg}", file=sys.stderr)
            return 2
        with _muted_stdout(as_json):
            results = manager.search_all(args.query, top_k=top_k,
                                         only=requested, **filters)
        return _render_search(args, results, f"sources {requested}",
                              requested, as_json)

    args.source = requested[0] if requested else None
    source_name = _resolve_source(args, config, allow_all=True)

    if source_name == "ALL":
        with _muted_stdout(as_json):
            results = manager.search_all(args.query, top_k=top_k, **filters)
        return _render_search(args, results, "all sources", None, as_json)
    # --mode applies only to single-source search (cross-source uses RRF
    # over per-source default modes).
    with _muted_stdout(as_json):
        if args.mode is not None:
            # Temporarily override the backend's mode for this one call.
            backend = manager.get(source_name)
            saved = backend.rag.search_mode if hasattr(backend, "rag") else None
            try:
                if hasattr(backend, "rag"):
                    backend.rag.search_mode = args.mode
                results = manager.search(source_name, args.query,
                                         top_k=top_k, **filters)
            finally:
                if hasattr(backend, "rag") and saved is not None:
                    backend.rag.search_mode = saved
        else:
            results = manager.search(source_name, args.query,
                                     top_k=top_k, **filters)
    return _render_search(args, results, f"source {source_name!r}",
                          source_name, as_json)


def _render_search(args, results, label, source, as_json) -> int:
    """Render search hits.

    Text goes through the same `_format_*` renderers the `search` MCP tool
    uses: the CLI used to carry its own thinner rendering (file, score, six
    raw lines), which meant the terminal showed strictly less than the
    model got — no symbol names, no line ranges to cite. One renderer, one
    thing to fix.
    """
    if as_json:
        _emit_json({"ok": True, "operation": "search", "source": source,
                    "query": args.query, "count": len(results),
                    "results": results})
        return 0
    from ._format import (
        _build_filter_suffix, _format_outline_results, _format_search_results,
    )
    suffix = _build_filter_suffix(args.glob, args.ext, args.path_contains)
    fmt = (_format_outline_results if getattr(args, "outline", False)
           else _format_search_results)
    print(fmt(args.query, results, label, suffix))
    return 0


def _cmd_status(args) -> int:
    config, manager = _build_manager(getattr(args, "config", None))
    requested = getattr(args, "source", None)
    if requested is not None and requested not in config.sources:
        print(f"[cli] unknown source {requested!r}", file=sys.stderr)
        return 1

    names = [requested] if requested else list(config.sources.keys())
    for name in names:
        if name in getattr(manager, "broken", {}):
            info = manager.broken[name]
            health = info.get("health", "corrupt")
            label = {
                "unverified": "UNVERIFIED (probe timed out)",
                "in_use": "IN USE (held by another Lynx process)",
            }.get(health, "CORRUPT INDEX")
            print(f"=== Source: {name} (type: {info['type']}) ===")
            print(f"Status:       {label}")
            if info.get("path"):
                print(f"Path:         {info['path']}")
            print(f"Error:        {info['error']}")
            if health == "in_use":
                print(f"Fix:          nothing to fix — the index is healthy and "
                      f"serving that process; stop `lynx serve` if you need it here")
            elif health == "unverified":
                print(f"Fix:          close other Lynx processes and retry; "
                      f"`lynx reset --source {name}` only if it persists")
            else:
                print(f"Fix:          lynx reset --source {name}")
            print()
            continue
        backend = manager.get(name)
        s = backend.status()
        needs = backend.needs_update() if hasattr(backend, "needs_update") else False
        print(f"=== Source: {name} (type: {s['type']}) ===")
        print(f"Status:       {'Needs update' if needs else 'Up to date'}")
        if s.get("path"):
            print(f"Path:         {s['path']}")
        print(f"Chunks:       {s.get('chunk_count', 'n/a')}")
        if s.get("last_commit"):
            print(f"Last commit:  {s['last_commit']}")
        print(f"Last update:  {s.get('last_update', 'Never')}")
        print()
        print(backend.drift_status_text())
        print()
    return 0


def _cmd_list_sources(args) -> int:
    config, manager = _build_manager(getattr(args, "config", None))
    print(f"Sources ({len(manager.backends)}):")
    for status in manager.list_sources():
        line = f"  - {status['name']} (type: {status['type']}, chunks: {status.get('chunk_count', 'n/a')})"
        if status.get("path"):
            line += f"\n      path: {status['path']}"
        if status.get("drift_severity"):
            line += f"\n      drift: {status['drift_severity'].upper()}"
        print(line)
    return 0


# ----------------------------------------------------------------------
# Subcommand: source add / source remove
# ----------------------------------------------------------------------


def _normalize_extensions(raw) -> List[str]:
    """Accept `.cs`, `cs`, or `.CS` and store the lower-case dotted form the
    loader compares against. Order is preserved, duplicates dropped."""
    out: List[str] = []
    for ext in raw or []:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in out:
            out.append(ext)
    return out


def _resolve_source_dir(raw: str):
    """Resolve and check a --path argument. Returns `(path, error_message)`.

    Checking here rather than leaving it to the loader is worth the
    duplication: the loader's failure arrives via the tempfile validation
    pass, so the user would see a generic "schema validation failed" where
    the real problem is a typo in a directory name.
    """
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError as e:
        return None, f"couldn't resolve --path {raw!r}: {e}"
    if not path.is_dir():
        return None, f"--path is not an existing directory: {path}"
    return path, None


def _source_block_from_args(args, quiet: bool = False):
    """Turn `lynx source add` flags into a config.json source block.

    Returns `(block, error_message)`. Only keys the user actually set are
    emitted — everything else is left to the loader's defaults. That keeps
    config.json readable and means a future change of default still
    reaches sources added today.

    Errors are returned rather than printed so the caller can render them
    as text or as JSON; `quiet` suppresses the purely informational
    extension-detection line, which --json callers get in the block.
    """
    if args.block_json is not None:
        try:
            block = json.loads(args.block_json)
        except json.JSONDecodeError as e:
            return None, f"--block is not valid JSON: {e}"
        if not isinstance(block, dict):
            return None, (f"--block must be a JSON object describing one "
                          f"source, got {type(block).__name__}")
        return block, None

    stype = args.source_type
    if stype is None:
        return None, ("`lynx source add` needs --type "
                      f"({'|'.join(_SOURCE_TYPES)}) or --block JSON")

    path = None
    if stype in ("codebase", "pdf"):
        if not args.path:
            return None, f"--path DIR is required for --type {stype}"
        path, err = _resolve_source_dir(args.path)
        if err:
            return None, err

    if stype == "codebase":
        extensions = _normalize_extensions(args.ext)
        if not extensions:
            # Same folder scan the web UI runs when you pick a directory.
            from .manager.ui.detect import detect_extensions
            extensions = detect_extensions(path, top_n=10)
            if not extensions:
                return None, (f"found no indexable files under {path}; pass "
                              f"--ext explicitly (e.g. --ext .py --ext .md).")
            if not quiet:
                print(f"[cli] detected extensions: {' '.join(extensions)}")
        block = {
            "type": "codebase",
            "path": str(path),
            "supported_extensions": extensions,
        }
        if args.ignore:
            block["ignored_path_fragments"] = list(args.ignore)
        watcher = {}
        if args.no_watcher:
            watcher["enabled"] = False
        if args.watcher_debounce is not None:
            watcher["debounce_seconds"] = args.watcher_debounce
        if watcher:
            block["watcher"] = watcher
        if args.no_git:
            block["git_integration"] = {"enabled": False}
        if args.graph:
            block["graph"] = {"enabled": True}
        return block, None

    if stype == "webdoc":
        if not args.url:
            return None, "--url is required for --type webdoc"
        block = {"type": "webdoc", "url": args.url}
        if args.max_depth is not None:
            block["max_depth"] = args.max_depth
        if args.max_pages is not None:
            block["max_pages"] = args.max_pages
        if args.request_delay is not None:
            block["request_delay_seconds"] = args.request_delay
        if args.include_url:
            block["include_url_patterns"] = list(args.include_url)
        if args.exclude_url:
            block["exclude_url_patterns"] = list(args.exclude_url)
        if args.render_js:
            block["render_js"] = True
        if args.allow_cross_origin:
            block["same_origin_only"] = False
        return block, None

    # pdf
    block = {"type": "pdf", "path": str(path)}
    if args.no_recursive:
        block["recursive"] = False
    if args.file_glob:
        block["file_glob"] = args.file_glob
    if args.extractor:
        block["extractor"] = {"backend": args.extractor}
    if args.watcher:
        block["watcher"] = {"enabled": True}
    if args.watcher_debounce is not None:
        block.setdefault("watcher", {})["debounce_seconds"] = args.watcher_debounce
    return block, None


def _cmd_source(args) -> int:
    """Dispatch `lynx source add|remove`.

    Mutates config.json only — no index is touched (except `remove --purge`,
    and `add --build`). The write goes through the same validate-then-write
    helper as the web UI, so a rejected config leaves the file untouched.
    """
    sub = getattr(args, "source_command", None)
    if sub not in ("add", "remove"):
        print(
            "error: `lynx source` requires a sub-command (add|remove). "
            "Run `lynx source --help` for details.",
            file=sys.stderr,
        )
        return 2

    from .config import resolve_config_path
    config_path = resolve_config_path(getattr(args, "config", None))
    if not config_path.is_file():
        print(
            f"[cli] config file not found at {config_path}. Run "
            f"`lynx manager init` to create one, or pass --config PATH.",
            file=sys.stderr,
        )
        return 1

    if sub == "add":
        return _cmd_source_add(args, config_path)
    return _cmd_source_remove(args, config_path)


def _emit_json(payload) -> None:
    """Write one JSON object to stdout. Human-facing lines go to stderr in
    --json mode so the stream stays parseable by `jq` and friends."""
    print(json.dumps(payload, indent=2, default=str))


@contextlib.contextmanager
def _muted_stdout(enabled: bool = True):
    """Send everything written to fd 1 to fd 2 for the duration.

    Building a SourceManager imports llama_index, which prints "LLM is
    explicitly disabled. Using MockLLM." straight to stdout, and an index
    build logs its progress there too. Harmless in text mode, fatal for
    `--json | jq`: the JSON object stops being the only thing on stdout.
    `server.py` performs the same dance around the MCP stdio channel, for
    the same reason — the library writes below Python's logging, so
    redirecting the file descriptor is the only thing that catches it.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        # Flush before restoring, or output buffered while muted lands on
        # the real stdout afterwards — right in the middle of the JSON.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(saved, 1)
        os.close(saved)


def _cmd_source_add(args, config_path: Path) -> int:
    from .manager.sources import (
        add_source, load_config_dict, SOURCE_NAME_RE, SOURCE_NAME_HELP,
    )
    as_json = getattr(args, "as_json", False)

    def _fail(msg: str, code: int = 1) -> int:
        if as_json:
            _emit_json({"ok": False, "error": msg})
        else:
            print(f"[cli] {msg}", file=sys.stderr)
        return code

    # Check the name up front. `add_source` checks it again — it is the
    # contract both front-ends rely on — but building the block first would
    # mean a rejected name is reported only after a full folder walk.
    name = (args.name or "").strip()
    if not SOURCE_NAME_RE.match(name):
        return _fail(SOURCE_NAME_HELP)
    try:
        existing = load_config_dict(config_path).get("sources") or {}
    except Exception as e:
        return _fail(f"couldn't read {config_path}: {e}")
    if name in existing:
        return _fail(f"a source named {name!r} already exists in {config_path}.")

    block, err = _source_block_from_args(args, quiet=as_json)
    if err:
        return _fail(err, code=2)

    res = add_source(config_path, name, block)
    if not res.ok:
        return _fail(res.message)

    if not as_json:
        print(res.message)
        print(f"  config: {config_path}")
        if res.defaults_applied:
            print("  ignores: applied the default VCS/deps/build ignore list "
                  "(set ignored_path_fragments to override)")

    build_rc = 0
    built = None
    if args.build:
        # `name`, not `args.name`: add_source stored the stripped form, and
        # a build for a name that isn't in the config fails with "unknown
        # source" on the line right after reporting success.
        build_args = argparse.Namespace(
            config=getattr(args, "config", None), source=name,
        )
        # The build logs progress to stdout; in --json mode that would sit
        # in front of the object we're about to print.
        with _muted_stdout(as_json):
            build_rc = _cmd_build(build_args)
        built = build_rc == 0

    if as_json:
        payload = {
            "ok": build_rc == 0, "added": True, "name": name,
            "config": str(config_path),
            # What landed in the file, defaults included — reporting the
            # input instead would tell a script the source has no ignores
            # while config.json says otherwise.
            "block": res.written_block if res.written_block is not None else block,
            "defaults_applied": res.defaults_applied,
        }
        if built is not None:
            payload["built"] = built
            if not built:
                # An `ok: false` with no `error` was the one mute failure
                # left in the family.
                payload["error"] = (
                    f"source {name!r} was added, but indexing it failed "
                    f"(exit {build_rc}); see stderr, then retry with "
                    f"`lynx build --source {name}`"
                )
        _emit_json(payload)
    elif not args.build:
        print(f"  next:   lynx build --source {name}")
    return build_rc


def _cmd_source_remove(args, config_path: Path) -> int:
    as_json = getattr(args, "as_json", False)
    name = (args.name or "").strip()

    if not args.yes and sys.stdin.isatty():
        what = f"remove source {name!r} from {config_path}"
        if args.purge:
            what += " AND delete its index from disk"
        answer = input(f"This will {what}. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    from .manager.sources import remove_source
    res = remove_source(config_path, name, purge=args.purge)

    if as_json:
        _emit_json({
            "ok": res.ok,
            "name": name,
            "config": str(config_path),
            "purged_path": res.purged_path,
            **({} if res.ok else {"error": res.message}),
        })
        return 0 if res.ok else 1

    if not res.ok:
        # Includes the failed-purge case, where nothing was changed at all.
        print(f"[cli] {res.message}", file=sys.stderr)
        return 1

    print(res.message)
    if res.purged_path:
        print(f"  index wiped: {res.purged_path}")
    elif args.purge:
        print("  (no index directory on disk to remove)")
    else:
        print("  its index is still on disk — re-add the source to reuse it, "
              "or re-run with --purge to reclaim the space.")
    return 0


def _resolve_graph_source(manager, args):
    """Pick the source to operate on for `lynx graph ...`.

    Returns `(name, error_message)`. If --source is provided, validate it
    has the graph layer enabled; otherwise default to the single source
    with graph enabled, and report zero or more than one as an error.

    The error is returned rather than printed-and-exited so `graph query
    --json` can render it as an object like every other failure of that
    command. A helper that calls `sys.exit` leaves its caller no say in
    how the failure is reported — which is how the JSON mode came to emit
    an empty stdout with a non-zero exit and nothing to parse.
    """
    candidates = [
        n for n, b in manager.backends.items()
        if getattr(b, "graph", None) is not None
    ]
    if not candidates:
        return None, ("no source has the graph layer enabled. Add "
                      "`graph: { enabled: true }` to a codebase source's config.")
    if args.source:
        if args.source not in manager.backends:
            return None, (f"unknown source {args.source!r}. "
                          f"Available: {list(manager.backends)}")
        if args.source not in candidates:
            return None, f"source {args.source!r} has no graph layer enabled."
        return args.source, None
    if len(candidates) == 1:
        return candidates[0], None
    return None, (f"multiple sources have the graph layer enabled "
                  f"({candidates}); specify --source NAME")


def _cmd_graph(args) -> int:
    sub = getattr(args, "graph_command", None)
    if sub not in ("build", "status", "export", "query"):
        print("error: `lynx graph` requires a sub-command "
              "(build|status|export|query). "
              "Run `lynx graph --help` for details.", file=sys.stderr)
        return 2

    # Loading the manager pulls in llama_index, which greets stdout on
    # import. Mute it while there's a JSON object to keep clean.
    as_json = getattr(args, "as_json", False)
    with _muted_stdout(as_json):
        config, manager = _build_manager(getattr(args, "config", None))

    if sub == "query":
        # Rendering is shared with the `graph_query` MCP tool, so without
        # --json the terminal shows exactly what an agent would receive.
        from .graph.dispatch import query_graph
        source, err = _resolve_graph_source(manager, args)
        if err:
            # Same shape and keys as every other --json failure of this
            # command: a script must never have to tell "no object" apart
            # from "an object saying no". `source` is null here because
            # resolving it IS the failure.
            if as_json:
                _emit_json({"ok": False, "operation": args.op, "source": None,
                            "error": err})
            else:
                print(f"error: {err}", file=sys.stderr)
            return 2
        try:
            with _muted_stdout(as_json):
                res = query_graph(
                    manager, source, args.op,
                    symbol=args.symbol,
                    target=args.target,
                    relation_filter=args.relation_filter,
                    depth=args.depth,
                    limit=args.limit,
                    max_hops=args.max_hops,
                    top_n=args.top_n,
                    min_community_size=args.min_community_size,
                )
        except Exception as e:
            # The MCP tool wraps this same dispatch in a try/except; without
            # this one, the CLI's --json contract (exactly one object on
            # stdout, always) died on the first unexpected error — empty
            # stdout, a traceback, nothing to parse.
            msg = f"{type(e).__name__}: {e}"
            if as_json:
                _emit_json({"ok": False, "operation": args.op,
                            "source": source, "error": msg})
            else:
                print(f"error: {msg}", file=sys.stderr)
            return 1
        if as_json:
            _emit_json({"ok": res.ok, **res.data})
        else:
            print(res.text)
        # ok=False means a usage problem (unknown op, missing --symbol).
        # An operation that ran and found nothing is a real answer: exit 0.
        return 0 if res.ok else 1

    if sub == "export":
        from pathlib import Path
        source, err = _resolve_graph_source(manager, args)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        if getattr(args, "symbol", None):
            mode, target = "symbol", args.symbol
        else:
            mode, target = "module", args.module
        res = manager.export_graph(source, mode, target, depth=getattr(args, "depth", 2))
        if res.get("empty"):
            print(f"Nothing to export: {res.get('reason')}", file=sys.stderr)
            return 1
        if getattr(args, "out", None):
            out_path = Path(args.out)
        else:
            rp = getattr(config, "reports_path", None)
            out_path = (Path(rp) if rp else Path(config.storage_path) / "reports") / res["suggested_name"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(res["content"], encoding="utf-8")
        print(f"Wrote self-contained graph view to {out_path}")
        return 0

    if sub == "build":
        source, err = _resolve_graph_source(manager, args)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        force = bool(getattr(args, "force", False))
        print(f"Rebuilding graph for source {source!r} (force={force})...")
        summary = manager.get(source).graph.rebuild(force=force)
        print(f"  candidates:       {summary['candidates']}")
        print(f"  added:            {summary['added']}")
        print(f"  changed:          {summary['changed']}")
        print(f"  removed:          {summary['removed']}")
        print(f"  unchanged:        {summary['unchanged']}")
        print(f"  extracted_files:  {summary['extracted_files']}")
        print(f"  nodes_total:      {summary['nodes_total']}")
        print(f"  edges_total:      {summary['edges_total']}")
        print(f"  resolved_x-file:  {summary['resolved_cross_file']}")
        return 0

    # status
    if args.source:
        source, err = _resolve_graph_source(manager, args)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 2
        sources = [source]
    else:
        sources = [
            n for n, b in manager.backends.items()
            if getattr(b, "graph", None) is not None
        ]
        if not sources:
            print("No source has the graph layer enabled.", file=sys.stderr)
            return 0
    for name in sources:
        st = manager.graph_status(name)
        print(f"=== Graph status: {name} ===")
        print(f"  schema_version:    {st['schema_version']}")
        print(f"  nodes:             {st['nodes']}")
        print(f"  edges:             {st['edges']}")
        print(f"  files_indexed:     {st['files_indexed']}")
        print(f"  raw_calls_pending: {st['raw_calls_pending']}")
        print(f"  last_update:       {st['last_update']}")
        print(f"  last_full_rebuild: {st['last_full_rebuild']}")
        print(f"  by_language:       {st['by_language']}")
        print(f"  by_kind:           {st['by_kind']}")
        print(f"  by_relation:       {st['by_relation']}")
        print()
    return 0


def _cmd_manager(args) -> int:
    """Dispatch the `lynx manager <cmd>` sub-namespace.

    Lazy-import the manager package so users who never run a `lynx
    manager *` command don't pay its import cost (FastAPI alone is
    ~300ms cold).
    """
    sub = getattr(args, "manager_command", None)
    if sub is None:
        print(
            "error: `lynx manager` requires a sub-command "
            "(init | doctor | install | feedback | ui).\n"
            "Run `lynx manager --help` for details.",
            file=sys.stderr,
        )
        return 2
    from .manager import cli as manager_cli
    return manager_cli.dispatch(sub, args)


def _cmd_query(args) -> int:
    """Route the retrieval / navigation subcommands.

    Lazy-imported so `lynx serve` and the config-only commands never pay
    for the formatting layer, and injected with the two CLI helpers rather
    than letting `query_cli` import this module back.
    """
    from . import query_cli
    return query_cli.dispatch(args.command, args, _build_manager,
                              _muted_stdout)


_DISPATCH = {
    "serve": _cmd_serve,
    "build": _cmd_build,
    "search": _cmd_search,
    "status": _cmd_status,
    "reset": _cmd_reset,
    "list-sources": _cmd_list_sources,
    "source": _cmd_source,
    "graph": _cmd_graph,
    "manager": _cmd_manager,
    "migrate-config": _cmd_migrate_config,
}

# The retrieval / navigation commands all route through one handler; their
# names are listed in `query_cli.COMMANDS`, but importing that module here
# would cost every invocation the formatting layer. Kept in sync by
# test_query_cli.py, which asserts the two lists match.
_QUERY_COMMANDS = (
    "deep-search", "find-definition", "find-usages", "find-tests-for",
    "find-similar", "describe-symbol", "impact", "repo-overview",
    "module-summary", "search-diff",
)
_DISPATCH.update({name: _cmd_query for name in _QUERY_COMMANDS})


def main(argv: Optional[List[str]] = None) -> int:
    # Must run before anything imports `ssl` (transitively via requests /
    # huggingface_hub / chromadb): strips an antivirus-injected SSLKEYLOGFILE
    # that otherwise aborts the bundled interpreter on first TLS use.
    from .config import sanitize_tls_keylog_env
    sanitize_tls_keylog_env()

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    if command not in _DISPATCH:
        parser.error(f"unknown command: {command!r}")
    return _DISPATCH[command](args)


if __name__ == "__main__":
    sys.exit(main())
