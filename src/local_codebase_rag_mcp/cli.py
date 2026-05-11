"""Command-line entry point for local-codebase-rag-mcp (multi-source).

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
        prog="local-codebase-rag-mcp",
        description="Self-hosted MCP server with semantic + lexical search "
                    "over your local code and documentation. Runs 100 percent "
                    "local, no data egress.",
    )
    parser.add_argument(
        "--version", action="version", version=f"local-codebase-rag-mcp {__version__}"
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
        "delete rag_storage/ and run 'local-codebase-rag-mcp build' to "
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
    from .source_manager import SourceManager
    config = _load_config_or_exit(config_path)
    return config, SourceManager(config)


# ----------------------------------------------------------------------
# Subcommand dispatch
# ----------------------------------------------------------------------


def _cmd_serve(args) -> int:
    from .server import run_server
    run_server(config_path=getattr(args, "config", None))
    return 0


def _cmd_build(args) -> int:
    """Rebuild a specific source's index.

    On a fresh install the backend's __init__ already builds the index, so
    calling update(force=True) afterwards would re-do the same work. We
    probe metadata.json BEFORE construction and skip the explicit rebuild
    in the first-build case.
    """
    config_path = getattr(args, "config", None)
    config = _load_config_or_exit(config_path)
    source_name = _resolve_source(args, config)

    storage_dir = Path(config.storage_path) / source_name
    metadata_existed = (storage_dir / "metadata.json").is_file()

    _, manager = _build_manager(config_path)
    if metadata_existed:
        manager.update(source_name, force=True)
    print(f"Source {source_name!r} ready.")
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


_DISPATCH = {
    "serve": _cmd_serve,
    "build": _cmd_build,
    "search": _cmd_search,
    "status": _cmd_status,
    "list-sources": _cmd_list_sources,
    "migrate-config": _cmd_migrate_config,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    if command not in _DISPATCH:
        parser.error(f"unknown command: {command!r}")
    return _DISPATCH[command](args)


if __name__ == "__main__":
    sys.exit(main())
