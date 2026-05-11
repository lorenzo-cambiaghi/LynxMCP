"""Enable `python -m lynx [subcommand] [args...]`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
