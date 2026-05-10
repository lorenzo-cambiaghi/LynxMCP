"""local-codebase-rag-mcp: a self-hosted MCP server with semantic + lexical
search over your local codebase.

100% local, no data egress. See README.md for full documentation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("local-codebase-rag-mcp")
except PackageNotFoundError:
    # Source checkout without an editable install (e.g. running tests
    # directly from src/). Fall back to a sentinel.
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
