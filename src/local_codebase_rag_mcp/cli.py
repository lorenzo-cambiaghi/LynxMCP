"""Command-line entry point for local-codebase-rag-mcp.

Subcommands:
  serve   - run the MCP server (default if no command is given)
  build   - force a full rebuild of the index
  search  - run an ad-hoc search query (no MCP client needed)
  status  - print git state, last update time, and config drift

The package version is available via `--version` at the top level.

All subcommands accept `--config PATH` to override the default
config.json resolution chain (CLI flag > RAG_CONFIG_PATH env > ./config.json).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-codebase-rag-mcp",
        description="Self-hosted MCP server with semantic + lexical search "
                    "over your local codebase. Runs 100 percent local, "
                    "no data egress.",
    )
    parser.add_argument(
        "--version", action="version", version=f"local-codebase-rag-mcp {__version__}"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sp_serve = sub.add_parser("serve", help="Run the MCP server (default)")
    sp_serve.add_argument(
        "--config", "-c", metavar="PATH",
        help="Path to config.json. Default: $RAG_CONFIG_PATH or ./config.json",
    )

    sp_build = sub.add_parser("build", help="Force a full rebuild of the index")
    sp_build.add_argument("--config", "-c", metavar="PATH")

    sp_search = sub.add_parser("search", help="Run an ad-hoc search query")
    sp_search.add_argument("query", help="Natural-language search query")
    sp_search.add_argument("--config", "-c", metavar="PATH")
    sp_search.add_argument(
        "--top-k", "-k", type=int, default=None,
        help="Number of results to return (default: search.default_top_k from config)",
    )
    sp_search.add_argument(
        "--mode", choices=["hybrid", "dense", "sparse"], default=None,
        help="Override the search mode for this query only",
    )
    sp_search.add_argument(
        "--ext", action="append", metavar="EXT",
        help="Restrict to files with this extension (repeat for multiple, "
             "e.g. --ext .py --ext .md)",
    )
    sp_search.add_argument(
        "--glob", metavar="PATTERN",
        help="Restrict to files matching this glob pattern",
    )
    sp_search.add_argument(
        "--path-contains", metavar="SUBSTRING",
        help="Restrict to files whose path contains this substring",
    )

    sp_status = sub.add_parser("status", help="Show RAG status (git, drift)")
    sp_status.add_argument("--config", "-c", metavar="PATH")

    return parser


def _build_rag(config_path: Optional[str], search_mode_override: Optional[str] = None):
    """Construct CodebaseRAG synchronously from the given config path.

    Used by `build`, `search`, and `status`. (The MCP `serve` subcommand
    does its own threaded loading inside server.run_server.)
    """
    from .config import load_config
    from .rag_manager import CodebaseRAG

    config = load_config(config_path=config_path)
    return config, CodebaseRAG(
        codebase_path=str(config.codebase_path),
        rag_storage_path=str(config.storage_path),
        supported_extensions=config.supported_extensions,
        embedding_model_name=config.embedding.model_name,
        collection_name=config.collection_name,
        search_mode=search_mode_override or config.search.mode,
        rrf_k=config.search.rrf_k,
        candidate_pool_size=config.search.candidate_pool_size,
    )


def _cmd_serve(args) -> int:
    from .server import run_server
    run_server(config_path=getattr(args, "config", None))
    return 0


def _cmd_build(args) -> int:
    """Build (or rebuild) the index.

    On a fresh install the constructor already builds the index implicitly,
    so calling update(force=True) afterwards would re-do the same work.
    Detect that case by probing for metadata.json BEFORE construction:
      - metadata exists  → existing index, user wants an explicit rebuild
      - metadata missing → first build, the constructor handles it
    """
    from .config import load_config
    config_path = getattr(args, "config", None)
    config = load_config(config_path=config_path)
    metadata_existed = (Path(config.storage_path) / "metadata.json").is_file()

    _, rag = _build_rag(config_path)
    if metadata_existed:
        rag.update(force=True)
    return 0


def _cmd_search(args) -> int:
    config, rag = _build_rag(getattr(args, "config", None), search_mode_override=args.mode)
    top_k = args.top_k if args.top_k is not None else config.search.default_top_k
    results = rag.search(
        args.query,
        top_k=top_k,
        file_glob=args.glob,
        extensions=args.ext,
        path_contains=args.path_contains,
    )
    if not results:
        print(f"No results for {args.query!r}.")
        return 0
    print(f"Found {len(results)} result(s) for {args.query!r}:\n")
    for i, r in enumerate(results, 1):
        print(f"--- {i}. {r['file']}  (score {r['score']:.3f}) ---")
        print(f"    path: {r.get('file_path', '')}")
        # Show the first ~6 lines of content to keep terminal output tight.
        snippet = (r.get("content") or "").strip().splitlines()[:6]
        for line in snippet:
            print(f"    {line}")
        print()
    return 0


def _cmd_status(args) -> int:
    _, rag = _build_rag(getattr(args, "config", None))
    metadata = rag.metadata
    needs = rag.needs_update()
    print(f"Status:       {'Needs update' if needs else 'Up to date'}")
    print(f"Last commit:  {metadata.get('last_commit', 'None')}")
    print(f"Last update:  {metadata.get('last_update', 'Never')}")
    print()
    print(rag.drift_status_text())
    return 0


_DISPATCH = {
    "serve": _cmd_serve,
    "build": _cmd_build,
    "search": _cmd_search,
    "status": _cmd_status,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Default to `serve` if no subcommand was given.
    command = args.command or "serve"
    if command not in _DISPATCH:
        parser.error(f"unknown command: {command!r}")
    return _DISPATCH[command](args)


if __name__ == "__main__":
    sys.exit(main())
