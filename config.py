"""
Configuration loader for the local-codebase-rag-mcp server.

Reads `config.json` (next to this file by default) and exposes a typed
`Config` dataclass used by `mcp_server.py` and `rag_manager.py`.

The path to the config file can be overridden via the `RAG_CONFIG_PATH`
environment variable. This is useful when the server is launched by an MCP
client whose working directory is unpredictable.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class WatcherConfig:
    enabled: bool = True
    debounce_seconds: float = 2.0


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class GitIntegrationConfig:
    enabled: bool = True


@dataclass(frozen=True)
class SearchConfig:
    default_top_k: int = 5
    mode: str = "hybrid"            # "hybrid" | "dense" | "sparse"
    rrf_k: int = 60                 # standard RRF constant
    candidate_pool_size: int = 30   # how many candidates to fetch from each retriever before fusion


@dataclass(frozen=True)
class Config:
    codebase_path: Path
    storage_path: Path
    collection_name: str
    loading_timeout_seconds: int
    supported_extensions: frozenset
    ignored_path_fragments: tuple
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    git_integration: GitIntegrationConfig = field(default_factory=GitIntegrationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.json"


def _normalize_ignored_fragments(fragments: List[str]) -> tuple:
    """Convert forward-slash fragments to OS-native separators.

    Users write `/.git/` in JSON for portability; on Windows we need
    `\\.git\\` for substring matches against actual filesystem paths.
    """
    sep = os.sep
    normalized = []
    for frag in fragments:
        normalized.append(frag.replace("/", sep))
    return tuple(normalized)


def _resolve_path(value: str, base: Path) -> Path:
    """Resolve a path from the config file. Relative paths are anchored to the
    directory containing the config file, not the current working directory
    (which is unpredictable when the server is launched by an MCP client)."""
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def load_config(config_path: Path | None = None) -> Config:
    """Load and validate the JSON config file.

    Resolution order:
      1. explicit `config_path` argument
      2. `RAG_CONFIG_PATH` environment variable
      3. `config.json` next to this script

    Exits with a clear error message if the config is missing or malformed.
    """
    if config_path is None:
        env_override = os.environ.get("RAG_CONFIG_PATH")
        if env_override:
            config_path = Path(env_override)
        else:
            config_path = _default_config_path()

    if not config_path.is_file():
        print(
            f"\n[config] ERROR: config file not found at {config_path}\n"
            f"[config] Copy 'config.example.json' to '{config_path.name}' and "
            f"edit 'codebase_path' to point at your codebase.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"\n[config] ERROR: invalid JSON in {config_path}: {e}\n", file=sys.stderr)
        sys.exit(1)

    base_dir = config_path.resolve().parent

    codebase_path_raw = raw.get("codebase_path")
    if not codebase_path_raw:
        print("\n[config] ERROR: 'codebase_path' is required in config.json\n", file=sys.stderr)
        sys.exit(1)

    codebase_path = _resolve_path(codebase_path_raw, base_dir)
    if not codebase_path.is_dir():
        print(
            f"\n[config] ERROR: codebase_path does not exist or is not a directory: "
            f"{codebase_path}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    storage_path = _resolve_path(raw.get("storage_path", "./rag_storage"), base_dir)

    collection_name = str(raw.get("collection_name", "codebase"))
    loading_timeout_seconds = int(raw.get("loading_timeout_seconds", 600))

    extensions = raw.get("supported_extensions") or [".py", ".md", ".txt"]
    extensions = frozenset(ext.lower() for ext in extensions)

    ignored = _normalize_ignored_fragments(raw.get("ignored_path_fragments") or [])

    watcher_raw = raw.get("watcher") or {}
    watcher = WatcherConfig(
        enabled=bool(watcher_raw.get("enabled", True)),
        debounce_seconds=float(watcher_raw.get("debounce_seconds", 2.0)),
    )

    embedding_raw = raw.get("embedding") or {}
    embedding = EmbeddingConfig(
        model_name=str(embedding_raw.get("model_name", "BAAI/bge-small-en-v1.5")),
    )

    git_raw = raw.get("git_integration") or {}
    git = GitIntegrationConfig(enabled=bool(git_raw.get("enabled", True)))

    search_raw = raw.get("search") or {}
    mode = str(search_raw.get("mode", "hybrid")).lower()
    if mode not in ("hybrid", "dense", "sparse"):
        print(
            f"\n[config] ERROR: search.mode must be one of "
            f"'hybrid' | 'dense' | 'sparse', got '{mode}'\n",
            file=sys.stderr,
        )
        sys.exit(1)
    search = SearchConfig(
        default_top_k=int(search_raw.get("default_top_k", 5)),
        mode=mode,
        rrf_k=int(search_raw.get("rrf_k", 60)),
        candidate_pool_size=int(search_raw.get("candidate_pool_size", 30)),
    )

    return Config(
        codebase_path=codebase_path,
        storage_path=storage_path,
        collection_name=collection_name,
        loading_timeout_seconds=loading_timeout_seconds,
        supported_extensions=extensions,
        ignored_path_fragments=ignored,
        watcher=watcher,
        embedding=embedding,
        git_integration=git,
        search=search,
    )
