"""Enable `python -m local_codebase_rag_mcp [subcommand] [args...]`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
