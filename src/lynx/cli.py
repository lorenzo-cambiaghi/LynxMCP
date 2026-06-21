"""Command-line entry point for lynx (multi-source).

Subcommands:
  serve           - run the MCP server (default if no command is given)
  build           - force a full rebuild of a source's index
  search          - run an ad-hoc search query against a source
  status          - show git state, last update time, config drift per source
  list-sources    - enumerate configured sources
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
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


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
        "--source", "-s", metavar="NAME",
        help="Source to query. Optional when only one source is configured. "
             "Use 'ALL' to fuse results across every source via RRF.",
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
        help="Manage the per-source knowledge graph layer (opt-in via "
             "`graph: { enabled: true }` in the source config).",
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
        "extra", nargs="?", metavar="EXTRA",
        help="Optional extra to install via pip (e.g. `pdf-fast`). "
             "Equivalent to `pip install lynx[<extra>]`.",
    )
    sp_mgr_install.add_argument(
        "--with-reranker", action="store_true",
        help="When used with --model, also download the reranker model.",
    )
    sp_mgr_install.add_argument(
        "--config", "-c", metavar="PATH",
        help="Config file to read for model name detection (used by --model).",
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
        raw = json.loads(in_path.read_text(encoding="utf-8"))
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
    config, manager = _build_manager(getattr(args, "config", None))
    source_name = _resolve_source(args, config, allow_all=True)
    top_k = args.top_k if args.top_k is not None else config.search.default_top_k

    if source_name == "ALL":
        results = manager.search_all(
            args.query,
            top_k=top_k,
            file_glob=args.glob,
            extensions=args.ext,
            path_contains=args.path_contains,
        )
        label = "all sources"
    else:
        # --mode applies only to single-source search (cross-source uses RRF
        # over per-source default modes).
        if args.mode is not None:
            # Temporarily override the backend's mode for this one call.
            backend = manager.get(source_name)
            saved = backend.rag.search_mode if hasattr(backend, "rag") else None
            try:
                if hasattr(backend, "rag"):
                    backend.rag.search_mode = args.mode
                results = manager.search(
                    source_name, args.query, top_k=top_k,
                    file_glob=args.glob, extensions=args.ext, path_contains=args.path_contains,
                )
            finally:
                if hasattr(backend, "rag") and saved is not None:
                    backend.rag.search_mode = saved
        else:
            results = manager.search(
                source_name, args.query, top_k=top_k,
                file_glob=args.glob, extensions=args.ext, path_contains=args.path_contains,
            )
        label = f"source {source_name!r}"

    if not results:
        print(f"No results for {args.query!r} in {label}.")
        return 0
    print(f"Found {len(results)} result(s) for {args.query!r} in {label}:\n")
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        src_tag = f" [{r['source']}]" if "source" in r else ""
        print(f"--- {i}. {r.get('file', 'unknown')}{src_tag}  (score {score:.3f}) ---")
        if r.get("file_path"):
            print(f"    path: {r['file_path']}")
        snippet = (r.get("content") or "").strip().splitlines()[:6]
        for line in snippet:
            print(f"    {line}")
        print()
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
            print(f"=== Source: {name} (type: {info['type']}) ===")
            print(f"Status:       CORRUPT INDEX")
            if info.get("path"):
                print(f"Path:         {info['path']}")
            print(f"Error:        {info['error']}")
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


def _resolve_graph_source(manager, args) -> str:
    """Return the source name to operate on for `lynx graph ...`.

    If --source is provided, validate it has the graph layer enabled.
    Otherwise default to the single source with graph enabled; error out
    when there are zero or more than one such sources.
    """
    candidates = [
        n for n, b in manager.backends.items()
        if getattr(b, "graph", None) is not None
    ]
    if not candidates:
        print(
            "error: no source has the graph layer enabled. "
            "Add `graph: { enabled: true }` to a codebase source's config.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.source:
        if args.source not in manager.backends:
            print(f"error: unknown source {args.source!r}. Available: {list(manager.backends)}",
                  file=sys.stderr)
            sys.exit(2)
        if args.source not in candidates:
            print(f"error: source {args.source!r} has no graph layer enabled.",
                  file=sys.stderr)
            sys.exit(2)
        return args.source
    if len(candidates) == 1:
        return candidates[0]
    print(
        f"error: multiple sources have the graph layer enabled ({candidates}); "
        f"specify --source NAME",
        file=sys.stderr,
    )
    sys.exit(2)


def _cmd_graph(args) -> int:
    sub = getattr(args, "graph_command", None)
    if sub not in ("build", "status", "export"):
        print("error: `lynx graph` requires a sub-command (build|status|export). "
              "Run `lynx graph --help` for details.", file=sys.stderr)
        return 2
    config, manager = _build_manager(getattr(args, "config", None))

    if sub == "export":
        from pathlib import Path
        source = _resolve_graph_source(manager, args)
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
        source = _resolve_graph_source(manager, args)
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
        sources = [_resolve_graph_source(manager, args)]
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
            "(init | doctor | install | ui).\n"
            "Run `lynx manager --help` for details.",
            file=sys.stderr,
        )
        return 2
    from .manager import cli as manager_cli
    return manager_cli.dispatch(sub, args)


_DISPATCH = {
    "serve": _cmd_serve,
    "build": _cmd_build,
    "search": _cmd_search,
    "status": _cmd_status,
    "reset": _cmd_reset,
    "list-sources": _cmd_list_sources,
    "graph": _cmd_graph,
    "manager": _cmd_manager,
    "migrate-config": _cmd_migrate_config,
}


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
