"""Lynx — a self-hosted MCP server with semantic + lexical search over your
local code and library documentation.

Named after Lynceus, the Argonaut whose sharp eyes could see through walls
and trees to find anything hidden — the same job an AI assistant needs done
when reaching into a large codebase or unfamiliar docs.

100% local, no data egress. See README.md for full documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lynx")
except PackageNotFoundError:
    # Source checkout without an editable install (e.g. running tests
    # directly from src/). Fall back to a sentinel.
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
