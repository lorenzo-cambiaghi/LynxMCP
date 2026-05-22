"""LynxManager — setup wizard, diagnostic, installer, web UI.

Sub-package for the `lynx manager <cmd>` sub-namespace. All submodules
are lazy-loaded: `lynx serve` and `lynx search` paths never pay the
import cost of FastAPI / Jinja / huggingface_hub when the user only
needs the MCP server or the CLI search.

Public entry points are exposed via `lynx.manager.cli.dispatch()` from
the main `lynx.cli` argparse layer.
"""
