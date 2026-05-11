"""
Configuration loader for the local-codebase-rag-mcp server (schema v2).

Reads `config.json` and exposes a typed `Config` dataclass used by
`server.py`, `cli.py`, and `source_manager.py`.

Resolution order (first match wins):
  1. Explicit `config_path` argument to `load_config()`.
  2. `RAG_CONFIG_PATH` environment variable.
  3. `./config.json` in the current working directory.

The CLI's `--config` flag plumbs straight to (1). The env var is the
recommended escape hatch when an MCP client launches the server from an
unpredictable working directory.

Schema v2 — top-level shape:

    {
      "config_version": 2,
      "storage_path": "./rag_storage",
      "embedding": {...},
      "search": {...},
      "sources": {
        "<source-name>": {
          "type": "codebase" | "webdoc" | "pdf" | ...,
          ... type-specific fields ...
        },
        ...
      }
    }

A v1 config (recognized by the presence of a top-level `codebase_path` field
and the absence of `sources`) is rejected with a pointer to
`local-codebase-rag-mcp migrate-config`.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


CURRENT_CONFIG_VERSION = 2
SUPPORTED_SOURCE_TYPES = ("codebase",)  # extended in M2/M3

# MCP tool names get the source name verbatim as a suffix. Restrict to a
# conservative subset so the generated names like `search_<name>` are always
# valid identifiers and easy to type. See README for the full rule.
SOURCE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$")


# ---------------------------------------------------------------------------
# Shared (server-wide) sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class DeepSearchConfig:
    """Tunables for the `deep_search_*` fallback tools.

    See README "Hybrid retrieval" and "deep_search_codebase" sections for
    rationale on mode-specific thresholds.
    """
    min_results: int = 2
    score_thresholds: dict = field(default_factory=lambda: {
        "dense": 0.45,
        "hybrid": 0.012,
        "sparse": 3.0,
    })


@dataclass(frozen=True)
class SearchConfig:
    default_top_k: int = 5
    mode: str = "hybrid"            # "hybrid" | "dense" | "sparse"
    rrf_k: int = 60                 # standard RRF constant
    candidate_pool_size: int = 30   # per-retriever candidate pool size
    deep: DeepSearchConfig = field(default_factory=DeepSearchConfig)


# ---------------------------------------------------------------------------
# Top-level Config object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    config_version: int
    storage_path: Path
    loading_timeout_seconds: int
    embedding: EmbeddingConfig
    search: SearchConfig
    # `sources` maps source-name -> raw dict from JSON (already validated /
    # path-resolved). Per-source typed dataclasses are intentionally NOT used
    # here because each source type has its own shape; the SourceManager and
    # backends know how to read their own fields.
    sources: Dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path / utility helpers
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    return Path.cwd() / "config.json"


def _normalize_ignored_fragments(fragments: List[str]) -> tuple:
    """Convert forward-slash fragments to OS-native separators so they match
    actual filesystem paths on Windows."""
    sep = os.sep
    return tuple(frag.replace("/", sep) for frag in fragments)


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a path from the config file. Relative paths are anchored to
    the directory containing the config file, not the current working
    directory (which is unpredictable when launched by an MCP client)."""
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


# ---------------------------------------------------------------------------
# Per-type source validation
# ---------------------------------------------------------------------------


def _validate_codebase_source(name: str, raw: dict, base_dir: Path) -> dict:
    """Validate and normalize a `type=codebase` source entry."""
    path_raw = raw.get("path")
    if not path_raw:
        _config_error(f"source {name!r}: 'path' is required for type=codebase")
    path = _resolve_path(path_raw, base_dir)
    if not path.is_dir():
        _config_error(
            f"source {name!r}: path does not exist or is not a directory: {path}"
        )

    extensions = raw.get("supported_extensions") or [".py", ".md", ".txt"]
    extensions = frozenset(ext.lower() for ext in extensions)

    ignored = _normalize_ignored_fragments(raw.get("ignored_path_fragments") or [])

    watcher_raw = raw.get("watcher") or {}
    watcher = {
        "enabled": bool(watcher_raw.get("enabled", True)),
        "debounce_seconds": float(watcher_raw.get("debounce_seconds", 2.0)),
    }

    git_raw = raw.get("git_integration") or {}
    git_integration = {"enabled": bool(git_raw.get("enabled", True))}

    return {
        "type": "codebase",
        "path": path,
        "supported_extensions": extensions,
        "ignored_path_fragments": ignored,
        "watcher": watcher,
        "git_integration": git_integration,
    }


_TYPE_VALIDATORS = {
    "codebase": _validate_codebase_source,
    # "webdoc": _validate_webdoc_source,   # M2
    # "pdf":    _validate_pdf_source,       # M3
}


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def _config_error(msg: str) -> None:
    """Print a friendly error to stderr and exit. Used by the loader."""
    print(f"\n[config] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def _looks_like_v1(raw: dict) -> bool:
    """Heuristic: v1 had `codebase_path` at the top level and no `sources`."""
    return "codebase_path" in raw and "sources" not in raw


def load_config(config_path: Path | None = None) -> Config:
    """Load and validate the JSON config file.

    Exits with a clear error message if the config is missing, malformed,
    or in an unsupported schema version.
    """
    if config_path is None:
        env_override = os.environ.get("RAG_CONFIG_PATH")
        if env_override:
            config_path = Path(env_override)
        else:
            config_path = _default_config_path()
    else:
        # Accept str / os.PathLike from callers (CLI, tests).
        config_path = Path(config_path)

    if not config_path.is_file():
        _config_error(
            f"config file not found at {config_path}\n"
            f"[config] Copy 'config.example.json' to '{config_path.name}' and edit "
            f"'sources' to point at your code / docs."
        )

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _config_error(f"invalid JSON in {config_path}: {e}")

    base_dir = config_path.resolve().parent

    # --- version handling --------------------------------------------------
    version = raw.get("config_version")
    if version is None:
        if _looks_like_v1(raw):
            _config_error(
                f"{config_path} looks like a v1 config (has 'codebase_path' at the "
                f"top level). Schema v2 reorganizes the file under a 'sources' "
                f"block.\n[config] Run:  local-codebase-rag-mcp migrate-config "
                f"--input \"{config_path}\"\n[config] (or rewrite manually — see "
                f"config.example.json for the v2 template.)"
            )
        _config_error(
            f"'config_version' field is missing from {config_path}. "
            f"Expected: 'config_version': {CURRENT_CONFIG_VERSION}."
        )
    if not isinstance(version, int):
        _config_error(f"'config_version' must be an integer, got {version!r}")
    if version < CURRENT_CONFIG_VERSION:
        _config_error(
            f"'config_version': {version} is older than this package's schema "
            f"(v{CURRENT_CONFIG_VERSION}). Run "
            f"'local-codebase-rag-mcp migrate-config --input \"{config_path}\"' "
            f"to upgrade."
        )
    if version > CURRENT_CONFIG_VERSION:
        _config_error(
            f"'config_version': {version} is newer than this package's schema "
            f"(v{CURRENT_CONFIG_VERSION}). Upgrade the package, or downgrade the "
            f"config to match."
        )

    # --- shared / top-level fields ----------------------------------------
    storage_path = _resolve_path(raw.get("storage_path", "./rag_storage"), base_dir)
    loading_timeout_seconds = int(raw.get("loading_timeout_seconds", 600))

    embedding_raw = raw.get("embedding") or {}
    embedding = EmbeddingConfig(
        model_name=str(embedding_raw.get("model_name", "BAAI/bge-small-en-v1.5")),
    )

    search_raw = raw.get("search") or {}
    mode = str(search_raw.get("mode", "hybrid")).lower()
    if mode not in ("hybrid", "dense", "sparse"):
        _config_error(
            f"search.mode must be one of 'hybrid' | 'dense' | 'sparse', "
            f"got {mode!r}"
        )

    deep_raw = search_raw.get("deep") or {}
    deep_thresholds_raw = deep_raw.get("score_thresholds") or {}
    deep_thresholds = {
        "dense": float(deep_thresholds_raw.get("dense", 0.45)),
        "hybrid": float(deep_thresholds_raw.get("hybrid", 0.012)),
        "sparse": float(deep_thresholds_raw.get("sparse", 3.0)),
    }
    deep = DeepSearchConfig(
        min_results=int(deep_raw.get("min_results", 2)),
        score_thresholds=deep_thresholds,
    )
    search = SearchConfig(
        default_top_k=int(search_raw.get("default_top_k", 5)),
        mode=mode,
        rrf_k=int(search_raw.get("rrf_k", 60)),
        candidate_pool_size=int(search_raw.get("candidate_pool_size", 30)),
        deep=deep,
    )

    # --- sources -----------------------------------------------------------
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, dict) or not sources_raw:
        _config_error(
            "'sources' must be a non-empty object mapping source name -> "
            "source config. See config.example.json for the shape."
        )

    sources: Dict[str, dict] = {}
    for name, entry in sources_raw.items():
        if not SOURCE_NAME_RE.match(name):
            _config_error(
                f"source name {name!r} is invalid. Allowed pattern: "
                f"letter followed by letters / digits / underscore "
                f"(max 40 chars). Example valid names: 'myproject', "
                f"'unityDoc', 'avalonia_docs'."
            )
        if not isinstance(entry, dict):
            _config_error(f"source {name!r} must be a JSON object")
        type_name = entry.get("type")
        if type_name not in SUPPORTED_SOURCE_TYPES:
            _config_error(
                f"source {name!r}: type {type_name!r} is not supported. "
                f"Supported: {list(SUPPORTED_SOURCE_TYPES)}"
            )
        validator = _TYPE_VALIDATORS[type_name]
        sources[name] = validator(name, entry, base_dir)

    return Config(
        config_version=version,
        storage_path=storage_path,
        loading_timeout_seconds=loading_timeout_seconds,
        embedding=embedding,
        search=search,
        sources=sources,
    )
