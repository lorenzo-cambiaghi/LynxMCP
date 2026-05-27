"""Minimal-bootstrap setup — `lynx manager init`.

This command exists to do the one thing the web UI cannot do for itself:
write the first `config.json` and pull the embedding model into the HF
cache. Everything else (adding sources, choosing a reranker, picking an
AI-client rules file) lives in `lynx manager ui`, where each option can
be explained in context with multi-line help text and form widgets
instead of one-line terminal prompts.

Flow
----
1. Resolve `--output` (default `./config.json`).
2. If it already exists, ask once whether to overwrite. Otherwise write
   a default config with **no sources** (the UI handles add-source).
3. Download the embedding model into the HF cache (unless
   `--skip-model-download`).
4. Print the MCP-server snippet so the user can paste it into their AI
   client.
5. Offer to launch `lynx manager ui` immediately as the next step.

`--non-interactive` skips both the overwrite prompt and the "open UI"
prompt — used by CI / scripts. Model download still runs (toggle off
with `--skip-model-download` for fully offline CI).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .ansi import success, warn, error, bold, dim, heading, bullet


# Default values for the generated config. Kept in sync with the v2
# schema in `lynx.config` and the example file (`config.example.json`).
DEFAULT_STORAGE = "./rag_storage"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 8
DEFAULT_SEARCH_MODE = "hybrid"


# ---------------------------------------------------------------------------
# Prompt helpers (kept tiny — we only ever ask y/n confirmations now)
# ---------------------------------------------------------------------------


def _read_line() -> str:
    """input() that maps EOF to Ctrl+C so the wizard exits cleanly when
    stdin is piped (e.g. `echo "" | lynx manager init`)."""
    try:
        return input()
    except EOFError as e:
        raise KeyboardInterrupt() from e


def _confirm(label: str, default: bool = True) -> bool:
    """Yes/no prompt. Returns True for yes."""
    suffix = "Y/n" if default else "y/N"
    while True:
        sys.stdout.write(f"{bold('?')} {label} {dim('[' + suffix + ']')}: ")
        sys.stdout.flush()
        raw = _read_line().strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(error("Please answer y or n."))


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def _default_config() -> dict:
    """The skeleton config we write on a fresh `init`. No sources — the
    UI's `/sources/new` flow is where the user adds them."""
    return {
        "config_version": 2,
        "storage_path": DEFAULT_STORAGE,
        "loading_timeout_seconds": 600,
        "embedding": {"model_name": DEFAULT_EMBED_MODEL},
        "search": {
            "default_top_k": DEFAULT_TOP_K,
            "mode": DEFAULT_SEARCH_MODE,
            "rrf_k": 60,
            "candidate_pool_size": 30,
            "deep": {
                "min_results": 2,
                "score_thresholds": {"dense": 0.45, "hybrid": 0.012, "sparse": 3.0},
            },
            "reranker": {
                "enabled": False,
                "model_name": DEFAULT_RERANKER_MODEL,
                "top_n_before_rerank": 30,
            },
        },
        "sources": {},
    }


def _write_config(cfg: dict, output_path: Path) -> None:
    """Write `cfg` to `output_path` as pretty JSON.

    Caller is responsible for prompting before overwrite — we don't
    re-check here so the non-interactive path stays branch-free.
    """
    output_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MCP client snippet
# ---------------------------------------------------------------------------


def _print_mcp_snippet(config_path: Path) -> None:
    """Show the user what to paste into their AI client's MCP config."""
    cfg = config_path.resolve()
    print()
    print(heading("To wire Lynx into your AI client, add this MCP server:"))
    print()
    snippet = {
        "mcpServers": {
            "lynx": {
                "command": sys.executable,
                "args": ["-m", "lynx", "serve", "--config", str(cfg)],
            }
        }
    }
    print(json.dumps(snippet, indent=2))
    print()
    print(dim("  - Claude Code: paste into ~/.claude/mcp_settings.json"))
    print(dim("  - Cursor: .cursor/mcp.json"))
    print(dim("  - Antigravity: .agents/mcp.json"))
    print(dim("  - (or open LynxManager → Integrations for one-click snippets)"))


# ---------------------------------------------------------------------------
# UI launch
# ---------------------------------------------------------------------------


def _launch_ui(config_path: Path) -> None:
    """Spawn `lynx manager ui --config <path>` and detach.

    We use Popen (not `subprocess.run`) so `init` exits cleanly instead of
    blocking on uvicorn. The UI process inherits the terminal for log
    output — closing the terminal stops it, which matches user expectation
    for a one-shot launch.
    """
    cmd = [
        sys.executable, "-m", "lynx", "manager", "ui",
        "--config", str(config_path.resolve()),
    ]
    print(dim(f"  launching: {' '.join(cmd)}"))
    try:
        subprocess.Popen(cmd)
    except OSError as e:
        print(error(f"Couldn't launch the UI: {e}"))
        print(dim(f"  Try manually: lynx manager ui --config {config_path}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_init(args) -> int:
    """CLI entry point. Returns exit code."""
    output_path = Path(args.output)
    non_interactive = bool(getattr(args, "non_interactive", False))
    skip_model_download = bool(getattr(args, "skip_model_download", False))

    try:
        return _run(output_path, non_interactive, skip_model_download)
    except KeyboardInterrupt:
        print()
        print(warn("Cancelled."))
        return 130  # standard SIGINT exit code


def _run(output_path: Path, non_interactive: bool, skip_model_download: bool) -> int:
    print()
    print(heading("Lynx setup"))

    # 1. Write config (overwrite-protected unless non-interactive)
    if output_path.exists() and not non_interactive:
        if not _confirm(f"{output_path} exists. Overwrite?", default=False):
            print(warn("Aborted — config not written."))
            return 1
    _write_config(_default_config(), output_path)
    print(success(f"Wrote {output_path}"))
    print(dim("  (no sources yet — add them via the web UI in step 3)"))

    # 2. Download the embedding model into the HF cache
    if skip_model_download:
        print(dim("  --skip-model-download set; model will be fetched on "
                  "first `lynx serve` query instead."))
    else:
        print()
        print(heading("Downloading embedding model"))
        from . import install  # lazy import — huggingface_hub is heavy
        rc = install.download_model(DEFAULT_EMBED_MODEL)
        if rc != 0:
            print(warn("Model download failed — you can retry later with "
                       "`lynx manager install --model`."))
            # Non-fatal: the config is still valid, so we continue.

    # 3. MCP snippet
    _print_mcp_snippet(output_path)

    # 4. Next-step menu
    print()
    print(heading("Next steps"))
    print(bullet(f"`lynx manager ui --config {output_path}` — add your "
                 f"first source via the web UI"))
    print(bullet(f"`lynx manager doctor --config {output_path}` — verify "
                 f"the install"))
    print()
    print(success(bold("Setup complete!")))

    # 5. Optional UI launch
    if non_interactive:
        return 0
    print()
    if _confirm("Open LynxManager now?", default=True):
        _launch_ui(output_path)
    return 0
