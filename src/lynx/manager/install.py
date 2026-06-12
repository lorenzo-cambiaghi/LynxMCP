"""`lynx manager install` — manage optional extras and HuggingFace models.

Three modes:
  - `lynx manager install --list` — enumerate available extras + which
    are installed
  - `lynx manager install <extra>` — pip-install the packages behind an extra
  - `lynx manager install --model [NAME]` — download a model into the
    HF cache. Without an argument, reads the embedding model from the
    active config. With `--with-reranker`, also downloads the reranker.

Why this exists
---------------
First-run friction. Today a new user has to discover that:
  - `pdf-fast` is an opt-in extra for better PDF extraction
  - The 130MB embedding model auto-downloads on first search, blocking
    that search until done — pre-fetching it here keeps the first
    `lynx serve` snappy

This command centralizes all of that into something documentable as
"the first thing to run after install".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .ansi import success, warn, error, bold, dim, heading, bullet


# Known optional extras. Keep in sync with pyproject.toml `[project.
# optional-dependencies]`. We hard-code rather than parse pyproject so
# we have a place to put the human-readable description.
KNOWN_EXTRAS = {
    "pdf-fast": {
        "pip_package": "pymupdf",
        # Concrete requirement installed for this extra. We install the
        # requirement directly rather than `pip install lynx-mcp[pdf-fast]`
        # so the command also works on editable/git installs (and never
        # resolves against the unrelated `lynx` package on PyPI).
        "pip_requirement": "pymupdf>=1.24",
        "extra_name": "pdf-fast",
        "what_for": (
            "Faster PDF extraction with better reading order on "
            "multi-column layouts (academic papers, IEEE/ACM specs). "
            "AGPL — opt-in."
        ),
    },
    # Future extras land here.
}


# ---------------------------------------------------------------------------
# Extras (pip install <requirement>)
# ---------------------------------------------------------------------------


def list_extras() -> int:
    """Print the known extras with install status."""
    print(heading("Optional extras"))
    print()
    for name, info in KNOWN_EXTRAS.items():
        installed = _is_installed(info["pip_package"])
        marker = success("installed") if installed else dim("not installed")
        print(f"  {bold(name)} — {marker}")
        print(bullet(info["what_for"]))
        if not installed:
            print(bullet(dim(f"install: lynx manager install {name}")))
        print()
    return 0


def install_extra(extra_name: str) -> int:
    """Install the packages behind an extra via subprocess.

    We invoke `python -m pip` rather than `pip` so the install lands in
    the SAME venv that's running Lynx (the bare `pip` command on a
    user's PATH might be a different interpreter).
    """
    if extra_name not in KNOWN_EXTRAS:
        print(error(f"Unknown extra: {extra_name!r}"))
        print(dim(f"Available: {', '.join(KNOWN_EXTRAS)}"))
        return 2
    info = KNOWN_EXTRAS[extra_name]
    if _is_installed(info["pip_package"]):
        print(success(f"{extra_name} already installed."))
        return 0
    cmd = [sys.executable, "-m", "pip", "install", info["pip_requirement"]]
    print(dim(f"  running: {' '.join(cmd)}"))
    try:
        rc = subprocess.run(cmd, check=False).returncode
    except FileNotFoundError as e:
        # python -m pip should always work in a sane venv; if it doesn't,
        # the user has a broken environment.
        print(error(f"Couldn't run pip via {sys.executable}: {e}"))
        return 2
    if rc != 0:
        print(error(f"pip install failed (exit {rc})"))
        return 2
    # Re-check installation
    if not _is_installed(info["pip_package"]):
        print(warn(f"pip reported success but `import {info['pip_package']}` "
                   f"still fails — check the install log above."))
        return 1
    print(success(f"{extra_name} installed."))
    return 0


def _is_installed(package: str) -> bool:
    """Cheap import probe. Catches ImportError + any package-level
    init failure (a half-broken install would still show as 'installed'
    via pip metadata but fail to import — we care about the latter)."""
    try:
        __import__(package)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# HuggingFace model download
# ---------------------------------------------------------------------------


def download_model(model_name: str) -> int:
    """Explicitly fetch a HuggingFace model into the local cache.

    Search-time code runs with HF offline mode on whenever the models are
    already cached (see config.configure_hf_offline). We clear the flags
    here ONLY for the duration of the download (restoring them on exit so
    any follow-up code keeps the offline guarantee).
    """
    print(dim(f"  Downloading {model_name} (this can take a minute)..."))

    # Save + clear offline flags
    saved = {}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                "HF_HUB_DISABLE_PROGRESS_BARS"):
        saved[key] = os.environ.pop(key, None)
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            print(error(f"huggingface_hub not available: {e}"))
            return 2
        try:
            snapshot_download(repo_id=model_name)
        except Exception as e:
            print(error(f"download failed: {type(e).__name__}: {e}"))
            return 2
    finally:
        # Restore the env exactly as we found it
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val
    print(success(f"{model_name} downloaded."))
    return 0


def download_models_for_config(config_path: Optional[Path],
                               with_reranker: bool) -> int:
    """Read the embedding model name from the active config and download
    it. With `--with-reranker`, also download the reranker model from
    the same config (or the default if reranker is disabled — useful
    for "I plan to enable it later")."""
    # Default values used when no config is available.
    embed_model = "BAAI/bge-small-en-v1.5"
    rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    if config_path is None or not config_path.exists():
        print(warn(f"No config provided — using defaults: {embed_model}"))
    else:
        try:
            from ..config import load_config
            cfg = load_config(config_path)
            embed_model = cfg.embedding.model_name
            rerank_model = cfg.search.reranker.model_name
        except SystemExit:
            print(warn(f"Config at {config_path} failed validation; "
                       f"using built-in defaults"))
        except Exception as e:
            print(warn(f"Couldn't read config ({e}); using defaults"))

    rc = download_model(embed_model)
    if rc != 0:
        return rc
    if with_reranker:
        rc = download_model(rerank_model)
        if rc != 0:
            return rc
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_install(args) -> int:
    """Dispatch the `lynx manager install ...` flavors."""
    if getattr(args, "list", False):
        return list_extras()

    model_arg = getattr(args, "model", None)
    if model_arg is not None:
        # Resolve via the shared chain so --model without --config picks
        # up ./config.json or $RAG_CONFIG_PATH like `lynx serve` does.
        from ..config import resolve_config_path
        resolved = resolve_config_path(getattr(args, "config", None))
        config_path = resolved if resolved.is_file() else None
        with_reranker = bool(getattr(args, "with_reranker", False))
        if model_arg == "__default__":
            # --model with no value → use config-driven model name
            return download_models_for_config(config_path, with_reranker)
        # --model with explicit name
        rc = download_model(model_arg)
        if rc == 0 and with_reranker:
            # Also pull reranker, either from config or default
            return download_models_for_config(config_path, with_reranker=True)
        return rc

    if getattr(args, "extra", None):
        return install_extra(args.extra)

    # No flag → show help-ish hint
    print(warn("`lynx manager install` requires an action."))
    print(bullet("`lynx manager install --list` — show available extras"))
    print(bullet("`lynx manager install <extra>` — install one (e.g. pdf-fast)"))
    print(bullet("`lynx manager install --model` — download embedding model"))
    print(bullet("`lynx manager install --model NAME` — download a specific model"))
    return 2
