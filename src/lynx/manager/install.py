"""`lynx manager install` — manage optional extras and HuggingFace models.

Modes:
  - `lynx manager install --list` — enumerate available extras + which
    are installed
  - `lynx manager install <extra>` — pip-install the packages behind an extra
  - `lynx manager install --model [NAME]` — download a model into the
    HF cache. Without an argument, reads the embedding model from the
    active config. With `--with-reranker`, also downloads the reranker.
  - `lynx manager install --export-archive PATH` — zip a cached model so it
    can be shared (e.g. via Google Drive).
  - `lynx manager install --from-archive PATH_OR_URL` — import such an
    archive into the HF cache on a machine that can't reach huggingface.co
    (offline / air-gapped / firewalled).

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
    "webdoc-js": {
        "pip_package": "playwright",
        "pip_requirement": "playwright>=1.40",
        "extra_name": "webdoc-js",
        # Extra shell steps after the pip install (each is an argv list run
        # with the current interpreter context). Playwright ships the driver
        # via pip but the browser binary is a separate ~150MB download.
        "post_install": [
            [sys.executable, "-m", "playwright", "install", "chromium"],
        ],
        "what_for": (
            "JS rendering for webdoc sources (`render_js: true` in config): "
            "crawls SPA / client-side-rendered docs sites through headless "
            "Chromium. Downloads the Chromium binary (~150MB) on install."
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
    if _is_installed(info["pip_package"]) and not info.get("post_install"):
        print(success(f"{extra_name} already installed."))
        return 0
    if _is_installed(info["pip_package"]):
        # Package present, but the extra has post-install steps (e.g. the
        # Chromium download) that may not have run — they are idempotent,
        # so re-run them instead of guessing.
        print(dim(f"  {info['pip_package']} already installed; "
                  f"running post-install steps."))
        for step in info.get("post_install", []):
            print(dim(f"  running: {' '.join(step)}"))
            rc = subprocess.run(step, check=False).returncode
            if rc != 0:
                print(error(f"post-install step failed (exit {rc})"))
                return 2
        print(success(f"{extra_name} installed."))
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
    # Post-install steps (e.g. playwright's browser binary download).
    for step in info.get("post_install", []):
        print(dim(f"  running: {' '.join(step)}"))
        try:
            rc = subprocess.run(step, check=False).returncode
        except FileNotFoundError as e:
            print(error(f"post-install step failed to start: {e}"))
            return 2
        if rc != 0:
            print(error(f"post-install step failed (exit {rc})"))
            return 2
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
# Model archive import / export (offline / air-gapped transfer)
# ---------------------------------------------------------------------------


def _config_embed_model(config_path: Optional[Path]) -> str:
    """Embedding model name from the active config, or the built-in default.

    Mirrors the resolution in `download_models_for_config` so archive
    import/export default to the SAME model the runtime will load."""
    default = "BAAI/bge-small-en-v1.5"
    if config_path is None or not config_path.exists():
        return default
    try:
        from ..config import load_config
        return load_config(config_path).embedding.model_name
    except Exception:
        return default


def _normalize_archive_source(src: str):
    """Classify an archive location. Returns ('url', src) for http(s) URLs
    (including already-direct Google Drive `uc?export=download&id=...` links),
    else ('path', src) for a local filesystem path. Pure — no I/O."""
    lowered = src.strip().lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return ("url", src.strip())
    return ("path", src)


def _is_within(base: Path, target: Path) -> bool:
    """True if `target` resolves inside `base` (path-traversal guard)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _extract_archive(archive: Path, dest: Path) -> int:
    """Extract a .zip or .tar(.gz|.tgz) into `dest`, refusing any member that
    would escape `dest`. Returns 0 on success, 2 on failure."""
    import tarfile
    import zipfile

    name = archive.name.lower()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    if not _is_within(dest, dest / member):
                        print(error(f"refusing unsafe path in archive: {member!r}"))
                        return 2
                zf.extractall(dest)
        elif name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if not _is_within(dest, dest / member.name):
                        print(error(f"refusing unsafe path in archive: {member.name!r}"))
                        return 2
                tf.extractall(dest)
        else:
            print(error(f"unsupported archive type: {archive.name} "
                        f"(use .zip, .tar.gz, .tgz, or .tar)"))
            return 2
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        print(error(f"extraction failed: {type(e).__name__}: {e}"))
        return 2
    return 0


def import_model_archive(archive: str, model_name: Optional[str],
                         config_path: Optional[Path]) -> int:
    """Import a model archive into the local HF hub cache (offline path).

    `archive` is a local path or an http(s) URL (a direct Google Drive
    download link works as a plain URL). The archive must contain the HF hub
    layout `models--ORG--NAME/...` (what `export_model_archive` produces). On
    success the model is usable fully offline — no huggingface.co needed.
    """
    import shutil
    import tempfile
    import urllib.request

    from ..config import _hf_cache_dir, _hf_model_cached

    model = model_name or _config_embed_model(config_path)
    cache_dir = _hf_cache_dir()
    kind, value = _normalize_archive_source(archive)

    tmp_download: Optional[Path] = None
    try:
        if kind == "url":
            print(dim(f"  Downloading archive from {value} ..."))
            try:
                fd, tmp_path = tempfile.mkstemp(prefix="lynx-model-", suffix=".archive")
                os.close(fd)
                tmp_download = Path(tmp_path)
                with urllib.request.urlopen(value) as resp, open(tmp_download, "wb") as out:
                    shutil.copyfileobj(resp, out)
            except Exception as e:
                print(error(f"download failed: {type(e).__name__}: {e}"))
                return 2
            # Name the temp file after the URL so _extract_archive can sniff
            # the suffix (.zip / .tar.gz). Default to .zip when ambiguous.
            url_name = value.split("?", 1)[0].rsplit("/", 1)[-1].lower()
            suffix = next((s for s in (".tar.gz", ".tgz", ".tar", ".zip")
                           if url_name.endswith(s)), ".zip")
            archive_path = tmp_download.with_name(tmp_download.name + suffix)
            tmp_download.rename(archive_path)
            tmp_download = archive_path
        else:
            archive_path = Path(value).expanduser()
            if not archive_path.is_file():
                print(error(f"archive not found: {archive_path}"))
                return 2

        print(dim(f"  Extracting into {cache_dir} ..."))
        rc = _extract_archive(archive_path, cache_dir)
        if rc != 0:
            return rc
    finally:
        if tmp_download is not None and tmp_download.exists():
            try:
                tmp_download.unlink()
            except OSError:
                pass

    if not _hf_model_cached(model):
        safe = model.replace("/", "--")
        print(error(
            f"archive extracted but {model} is still not in the cache. The "
            f"archive must contain the HF hub layout `models--{safe}/snapshots/"
            f"<rev>/...` (use `lynx manager install --export-archive` on a "
            f"machine that has the model to produce a compatible archive)."
        ))
        return 2
    print(success(f"{model} imported into {cache_dir}."))
    return 0


def export_model_archive(model_name: Optional[str], dest: str,
                         config_path: Optional[Path]) -> int:
    """Zip a cached model's hub directory so it can be shared (e.g. uploaded
    to Google Drive) and imported on an offline machine via
    `import_model_archive`. Returns 0 on success, 2 on failure."""
    import zipfile

    from ..config import _hf_cache_dir

    model = model_name or _config_embed_model(config_path)
    safe = model.replace("/", "--")
    cache_dir = _hf_cache_dir()
    model_dir = cache_dir / f"models--{safe}"
    if not model_dir.is_dir():
        print(error(
            f"{model} is not in the local cache ({model_dir}). Download it "
            f"first with `lynx manager install --model {model}`."
        ))
        return 2

    dest_path = Path(dest).expanduser()
    if dest_path.is_dir():
        dest_path = dest_path / f"{safe}.zip"
    elif not dest_path.name.lower().endswith(".zip"):
        dest_path = dest_path.with_name(dest_path.name + ".zip")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(dim(f"  Zipping {model_dir} → {dest_path} ..."))
    try:
        with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in model_dir.rglob("*"):
                if path.is_file():  # follows symlinks → materializes real files
                    zf.write(path, arcname=path.relative_to(cache_dir))
    except (OSError, zipfile.BadZipFile) as e:
        print(error(f"export failed: {type(e).__name__}: {e}"))
        return 2

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(success(f"{model} exported to {dest_path} ({size_mb:.0f} MB)."))
    print(bullet(dim("Share this file, then import it on the offline machine "
                     "with `lynx manager install --from-archive <path|url>`.")))
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

    from_archive = getattr(args, "from_archive", None)
    export_archive = getattr(args, "export_archive", None)
    if from_archive is not None or export_archive is not None:
        from ..config import resolve_config_path
        resolved = resolve_config_path(getattr(args, "config", None))
        config_path = resolved if resolved.is_file() else None
        model_name = getattr(args, "model_name", None)
        if from_archive is not None:
            return import_model_archive(from_archive, model_name, config_path)
        return export_model_archive(model_name, export_archive, config_path)

    if getattr(args, "extra", None):
        return install_extra(args.extra)

    # No flag → show help-ish hint
    print(warn("`lynx manager install` requires an action."))
    print(bullet("`lynx manager install --list` — show available extras"))
    print(bullet("`lynx manager install <extra>` — install one (e.g. pdf-fast)"))
    print(bullet("`lynx manager install --model` — download embedding model"))
    print(bullet("`lynx manager install --model NAME` — download a specific model"))
    print(bullet("`lynx manager install --from-archive <path|url>` — import a "
                 "shared model archive (offline / air-gapped)"))
    print(bullet("`lynx manager install --export-archive <path>` — zip the "
                 "cached model to share it"))
    return 2
