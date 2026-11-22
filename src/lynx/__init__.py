"""Lynx — a self-hosted MCP server with semantic + lexical search over your
local code and library documentation.

Named after Lynceus, the Argonaut whose sharp eyes could see through walls
and trees to find anything hidden — the same job an AI assistant needs done
when reaching into a large codebase or unfamiliar docs.

100% local, no data egress. See README.md for full documentation.
"""

from importlib.metadata import PackageNotFoundError, version

from .config import sanitize_tls_keylog_env as _sanitize_tls_keylog_env

# Here and not only in the `lynx` entry point: a script, a hook or a notebook
# that imports Lynx skipped `cli.main`, and on a machine whose antivirus sets
# SSLKEYLOGFILE the bundled interpreter then aborted on first TLS use with
# "OPENSSL_Uplink: no OPENSSL_Applink" (found 2026-11-21 by a Claude Code hook
# that imports lynx.cli). config is stdlib-only, so this costs nothing.
_sanitize_tls_keylog_env()

try:
    # Distribution name (PyPI: lynx-mcp) differs from the import package
    # name (`lynx` was already taken on PyPI by an unrelated project).
    __version__ = version("lynx-mcp")
except PackageNotFoundError:
    try:
        # Editable installs made before the rename registered as `lynx`.
        __version__ = version("lynx")
    except PackageNotFoundError:
        # Source checkout without an editable install (e.g. running tests
        # directly from src/). Fall back to a sentinel.
        __version__ = "0.0.0+dev"

__all__ = ["__version__"]
