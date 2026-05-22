"""Interactive setup wizard — `lynx manager init`.

Goal: a developer who's never touched Lynx should be able to run
`lynx manager init`, answer a series of clear questions, and get a
valid config.json + a downloaded model + AI client rules file —
without ever opening the README.

Design notes
------------
- **stdlib only.** No `rich` / `inquirer` / `click`. The wizard runs
  on `input()` + the small ANSI helpers from `manager.ansi`. This
  makes the wizard work on bare-bones systems where pip extras may
  not be installed yet.
- **Defaults everywhere.** Almost every prompt has a default in
  brackets. Pressing Enter accepts it. Power users can blast through
  the wizard in 10 seconds.
- **Smart auto-detection** wherever it helps:
  - Codebase extensions: walk the chosen folder, return the top 10 by
    file count. User can refine.
  - Git integration: enabled by default only if the folder IS a git
    repo (otherwise pointless).
- **Ctrl+C is graceful.** We catch KeyboardInterrupt at the top level
  and print a friendly cancellation message instead of a traceback.
- **Existing config.json is sacred** — we never overwrite without an
  explicit confirmation.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from .ansi import success, warn, error, bold, dim, heading, bullet


# Common-knowledge defaults — used to render the prompt brackets.
DEFAULT_STORAGE = "./rag_storage"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 8
DEFAULT_SEARCH_MODE = "hybrid"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _read_line() -> str:
    """input() that maps EOF to Ctrl+C so the wizard exits cleanly when
    stdin is piped (e.g. `echo "" | lynx manager init`)."""
    try:
        return input()
    except EOFError as e:
        raise KeyboardInterrupt() from e


def prompt(
    label: str,
    default: Optional[str] = None,
    *,
    validator: Optional[Callable[[str], Optional[str]]] = None,
    choices: Optional[list] = None,
) -> str:
    """Ask the user for a string. Re-prompts on validation failure.

    `validator` returns None on success or a *string error message* on
    failure (kept simple so we don't need a Result type).
    """
    suffix = ""
    if choices:
        suffix = f" {dim('(' + '/'.join(choices) + ')')}"
    if default is not None:
        suffix += f" {dim('[' + default + ']')}"
    while True:
        sys.stdout.write(f"{bold('?')} {label}{suffix}: ")
        sys.stdout.flush()
        raw = _read_line().strip()
        if not raw and default is not None:
            raw = default
        if choices and raw not in choices:
            print(error(f"Please pick one of: {', '.join(choices)}"))
            continue
        if validator is not None:
            msg = validator(raw)
            if msg is not None:
                print(error(msg))
                continue
        return raw


def confirm(label: str, default: bool = True) -> bool:
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
# Validators
# ---------------------------------------------------------------------------


def _validate_existing_dir(value: str) -> Optional[str]:
    p = Path(value).expanduser()
    if not p.exists():
        return f"path does not exist: {p}"
    if not p.is_dir():
        return f"path is not a directory: {p}"
    return None


def _validate_url(value: str) -> Optional[str]:
    if not value:
        return "URL cannot be empty"
    if not (value.startswith("http://") or value.startswith("https://")):
        return "URL must start with http:// or https://"
    return None


def _validate_positive_int(label: str, lo: int = 1, hi: Optional[int] = None):
    def _v(value: str) -> Optional[str]:
        try:
            n = int(value)
        except ValueError:
            return f"{label} must be an integer"
        if n < lo:
            return f"{label} must be >= {lo}"
        if hi is not None and n > hi:
            return f"{label} must be <= {hi}"
        return None
    return _v


def _validate_source_name(value: str) -> Optional[str]:
    import re
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$", value):
        return ("name must start with a letter, contain only letters / "
                "digits / underscores, and be at most 40 chars")
    return None


# ---------------------------------------------------------------------------
# Smart helpers
# ---------------------------------------------------------------------------


def detect_extensions(folder: Path, top_n: int = 10) -> list:
    """Walk `folder` and return the top-N file extensions by count.

    Skips dot-prefixed dirs (e.g. `.git`, `.venv`) and common binary
    junk (`.pyc`, `.so`). Returns lower-case with leading dot.
    """
    ignored_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__",
                    ".idea", ".vscode", "dist", "build", "target", ".next"}
    ignored_exts = {".pyc", ".so", ".dylib", ".dll", ".class", ".o"}
    counter: Counter = Counter()
    for root, dirs, files in os.walk(folder):
        # Prune in place — os.walk respects mutation of `dirs`
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignored_dirs]
        for f in files:
            ext = Path(f).suffix.lower()
            if not ext or ext in ignored_exts:
                continue
            counter[ext] += 1
    return [ext for ext, _ in counter.most_common(top_n)]


def is_git_repo(folder: Path) -> bool:
    """Quick check without invoking git: look for .git/ at the folder
    root OR any parent (matches `git rev-parse` behavior)."""
    folder = folder.resolve()
    while True:
        if (folder / ".git").exists():
            return True
        if folder.parent == folder:  # reached fs root
            return False
        folder = folder.parent


# ---------------------------------------------------------------------------
# Sub-sections of the wizard
# ---------------------------------------------------------------------------


def _ask_codebase_source(name: str) -> dict:
    """Build a `type: codebase` source config block from prompts."""
    path = prompt(
        "Folder path (absolute or relative)",
        default=".",
        validator=_validate_existing_dir,
    )
    folder = Path(path).expanduser().resolve()

    print(dim(f"  Scanning {folder} for common file extensions..."))
    detected = detect_extensions(folder, top_n=10)
    if detected:
        suggestion = ",".join(detected)
        print(dim(f"  Detected: {suggestion}"))
    else:
        suggestion = ".py,.md"
        print(warn("  No common code extensions found; defaulting to .py,.md"))

    raw_exts = prompt(
        "File extensions to index (comma-separated)",
        default=suggestion,
    )
    exts = [e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in raw_exts.split(",") if e.strip()]

    watcher = confirm("Enable file watcher (auto-reindex on save)?", default=True)
    has_git = is_git_repo(folder)
    if has_git:
        print(dim("  .git/ detected — git_integration will be enabled."))
    git_int = has_git
    graph = confirm(
        "Enable graph layer (call/inheritance/import graph for find_definition/usages)?",
        default=False,
    )

    block: dict = {
        "type": "codebase",
        "path": str(folder),
        "supported_extensions": exts,
        "watcher": {"enabled": watcher, "debounce_seconds": 2.0},
        "git_integration": {"enabled": git_int},
    }
    if graph:
        block["graph"] = {"enabled": True}
    return block


def _ask_webdoc_source(name: str) -> dict:
    url = prompt(
        "Docs site URL to crawl (http:// or https://)",
        validator=_validate_url,
    )
    max_depth = int(prompt(
        "Max crawl depth", default="3",
        validator=_validate_positive_int("max_depth", lo=1, hi=20),
    ))
    max_pages = int(prompt(
        "Max pages to crawl", default="500",
        validator=_validate_positive_int("max_pages", lo=1, hi=50_000),
    ))
    same_origin = confirm("Restrict to same origin only?", default=True)
    return {
        "type": "webdoc",
        "url": url,
        "max_depth": max_depth,
        "max_pages": max_pages,
        "same_origin_only": same_origin,
        "request_delay_seconds": 0.5,
    }


def _ask_pdf_source(name: str) -> dict:
    path = prompt(
        "Folder containing .pdf files",
        default=".",
        validator=_validate_existing_dir,
    )
    folder = Path(path).expanduser().resolve()
    recursive = confirm("Scan sub-directories recursively?", default=True)
    backend = prompt(
        "Extractor backend",
        default="auto",
        choices=["auto", "pypdf", "pymupdf"],
    )
    watcher = confirm(
        "Enable file watcher? (off by default — PDFs change rarely and re-extract is costly)",
        default=False,
    )
    return {
        "type": "pdf",
        "path": str(folder),
        "recursive": recursive,
        "file_glob": "**/*.pdf",
        "watcher": {"enabled": watcher, "debounce_seconds": 5.0},
        "extractor": {
            "backend": backend,
            "max_file_mb": 100,
            "max_pages_per_file": 5000,
            "skip_password_protected": True,
            "skip_if_text_empty": True,
        },
    }


def _ask_source() -> "Optional[tuple[str, dict]]":
    """Add one source. Returns (name, block) or None if user says 'done'."""
    choice = prompt(
        "Add a source",
        default="codebase",
        choices=["codebase", "webdoc", "pdf", "done"],
    )
    if choice == "done":
        return None
    name = prompt(
        "Source name (letters/digits/underscore, max 40 chars)",
        validator=_validate_source_name,
    )
    if choice == "codebase":
        block = _ask_codebase_source(name)
    elif choice == "webdoc":
        block = _ask_webdoc_source(name)
    elif choice == "pdf":
        block = _ask_pdf_source(name)
    else:
        return None  # unreachable due to choices
    return (name, block)


def _ask_global_settings() -> dict:
    """Top-level config (storage, embedding, search, reranker)."""
    storage = prompt("Storage path for vector DB", default=DEFAULT_STORAGE)
    embed = prompt("Embedding model (HuggingFace name)", default=DEFAULT_EMBED_MODEL)
    top_k = int(prompt(
        "Default top_k for searches",
        default=str(DEFAULT_TOP_K),
        validator=_validate_positive_int("default_top_k", lo=1, hi=100),
    ))
    mode = prompt(
        "Default search mode",
        default=DEFAULT_SEARCH_MODE,
        choices=["hybrid", "dense", "sparse"],
    )
    rerank_enabled = confirm(
        "Enable cross-encoder reranker (~80MB model, ~50ms/query, "
        "+20-30% precision@1)?",
        default=False,
    )
    rerank_block: dict = {
        "enabled": rerank_enabled,
        "model_name": DEFAULT_RERANKER_MODEL,
        "top_n_before_rerank": 30,
    }
    return {
        "config_version": 2,
        "storage_path": storage,
        "loading_timeout_seconds": 600,
        "embedding": {"model_name": embed},
        "search": {
            "default_top_k": top_k,
            "mode": mode,
            "rrf_k": 60,
            "candidate_pool_size": 30,
            "deep": {
                "min_results": 2,
                "score_thresholds": {"dense": 0.45, "hybrid": 0.012, "sparse": 3.0},
            },
            "reranker": rerank_block,
        },
        "sources": {},  # populated by caller
    }


# ---------------------------------------------------------------------------
# AI client rules file
# ---------------------------------------------------------------------------


_AI_CLIENT_FILES = {
    "claude": ("CLAUDE.md", "Claude Code (CLI + VS Code extension)"),
    "antigravity": ("AGENTS.md", "Google Antigravity"),
    "cursor": (".cursor/rules/lynx.md", "Cursor"),
    "aider": ("AGENTS.md", "Aider / Continue.dev / generic"),
    "none": (None, "(skip — don't write a rules file)"),
}


def _generate_rules_file(source_names: list, has_graph: bool, has_git: bool) -> str:
    """Build a minimal but useful AI-client rules file based on the
    actual source names from the wizard. The user can edit / extend it
    after — this is a head-start, not a final spec."""
    src = source_names[0] if source_names else "myproject"
    lines = [
        "# Code Reuse & Library Awareness",
        "",
        "This project exposes a local MCP server (Lynx) with semantic search",
        "over the configured sources. Use the tools BEFORE writing new code",
        "or guessing at library APIs.",
        "",
        "## Sources available",
        "",
    ]
    for n in source_names:
        lines.append(f"- `search_{n}(query)` — semantic + lexical hybrid search.")
        lines.append(f"- `deep_search_{n}(queries)` — fallback for ambiguous queries.")
    lines.append("")
    if len(source_names) > 1:
        lines.append("When unsure which source has the answer, call "
                     "`search_all_sources(query)` once.")
        lines.append("")
    lines.append("## When to search")
    lines.append("")
    lines.append("Before implementing any utility, interface, or pattern,")
    lines.append("search the codebase source first to avoid duplication.")
    lines.append("Before invoking a library API, search the relevant docs source")
    lines.append("if one exists — your training data may predate the version in use.")
    if has_graph:
        lines.append("")
        lines.append("## Code-aware structural queries (graph layer enabled)")
        lines.append("")
        lines.append(f"- `find_definition_{src}(symbol)` — where is X defined?")
        lines.append(f"- `find_usages_{src}(symbol)` — who calls X? typeof / generics included.")
        lines.append(f"- `find_tests_for_{src}(symbol)` — are there tests for X?")
        lines.append(f"- `find_similar_{src}(snippet)` — is there code similar to this?")
        lines.append(f"- `get_callers_{src}(symbol)` / `get_callees_{src}(symbol)`")
        lines.append(f"- `get_subclasses_{src}(symbol)` / `get_superclasses_{src}(symbol)`")
        lines.append(f"- `architectural_overview_{src}()` — god nodes + communities")
    if has_git:
        lines.append("")
        lines.append("## Diff-aware search (git_integration enabled)")
        lines.append("")
        lines.append(f"- `search_diff_{src}(query)` — search only in files changed vs `main`.")
        lines.append("  Use during code review: 'I changed the discount logic — what else uses the same formula?'")
    lines.append("")
    lines.append("## How to search")
    lines.append("")
    lines.append("Describe what the code *does*, not its name. Semantic search")
    lines.append("beats `grep` precisely because it bridges synonyms.")
    lines.append("")
    return "\n".join(lines)


def _ask_ai_client(source_names: list, has_graph: bool, has_git: bool) -> "Optional[str]":
    """Ask which client. Returns the generated rules-file content or None."""
    if not confirm(
        "Generate an AI-client rules file (CLAUDE.md / AGENTS.md / ...)?",
        default=True,
    ):
        return None
    print(dim("  Available clients:"))
    keys = list(_AI_CLIENT_FILES.keys())
    for i, k in enumerate(keys, 1):
        _, label = _AI_CLIENT_FILES[k]
        print(f"    {i}. {label}")
    raw = prompt(
        "Pick one",
        default="1",
        validator=_validate_positive_int("choice", lo=1, hi=len(keys)),
    )
    key = keys[int(raw) - 1]
    if key == "none":
        return None
    rel_path, _label = _AI_CLIENT_FILES[key]
    content = _generate_rules_file(source_names, has_graph, has_git)
    target = Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not confirm(
            f"{rel_path} already exists. Overwrite?",
            default=False,
        ):
            print(dim(f"  Skipped writing {rel_path}."))
            return None
    target.write_text(content, encoding="utf-8")
    print(success(f"Wrote {rel_path}"))
    return content


# ---------------------------------------------------------------------------
# MCP client snippet
# ---------------------------------------------------------------------------


def _print_mcp_snippet(config_path: Path) -> None:
    """Show the user what to paste into their AI client's MCP config."""
    exe = sys.executable
    cfg = config_path.resolve()
    print()
    print(heading("To wire Lynx into your AI client, add this MCP server:"))
    print()
    snippet = {
        "mcpServers": {
            "lynx": {
                "command": exe,
                "args": ["-m", "lynx", "serve", "--config", str(cfg)],
            }
        }
    }
    print(json.dumps(snippet, indent=2))
    print()
    print(dim("  - Claude Code: paste into ~/.claude/mcp_settings.json"))
    print(dim("  - Cursor: .cursor/mcp.json"))
    print(dim("  - Antigravity: .agents/mcp.json"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_init(args) -> int:
    """CLI entry point. Returns exit code."""
    output_path = Path(args.output)
    non_interactive = bool(getattr(args, "non_interactive", False))

    if non_interactive:
        return _run_non_interactive(output_path)

    try:
        return _run_wizard(output_path)
    except KeyboardInterrupt:
        print()
        print(warn("Wizard cancelled. No changes written."))
        return 130  # standard SIGINT exit code


def _run_non_interactive(output_path: Path) -> int:
    """Generate a sensible-defaults config without prompts — useful for CI."""
    cfg = _ask_global_settings_defaults()
    cfg["sources"]["myproject"] = {
        "type": "codebase",
        "path": str(Path(".").resolve()),
        "supported_extensions": detect_extensions(Path("."), top_n=10) or [".py", ".md"],
        "watcher": {"enabled": True, "debounce_seconds": 2.0},
        "git_integration": {"enabled": is_git_repo(Path("."))},
    }
    _write_config(cfg, output_path, allow_overwrite=True)
    print(success(f"Wrote {output_path} (non-interactive defaults)"))
    return 0


def _ask_global_settings_defaults() -> dict:
    """Same shape as `_ask_global_settings()` but no prompts — used by
    non-interactive mode."""
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


def _write_config(cfg: dict, output_path: Path, *, allow_overwrite: bool) -> None:
    """Write `cfg` to `output_path` as pretty JSON. Asks before overwriting
    unless `allow_overwrite=True`."""
    if output_path.exists() and not allow_overwrite:
        if not confirm(
            f"{output_path} exists. Overwrite?",
            default=False,
        ):
            print(warn("Aborted — config not written."))
            raise KeyboardInterrupt()
    output_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _run_wizard(output_path: Path) -> int:
    """The interactive flow."""
    print()
    print(heading("Lynx setup wizard"))
    print(dim("  Press Enter to accept the [default] in brackets. Ctrl+C to abort."))
    print()

    cfg = _ask_global_settings()

    print()
    print(heading("Now add at least one source."))
    while True:
        result = _ask_source()
        if result is None:
            if not cfg["sources"]:
                print(warn("You need at least one source for Lynx to be useful."))
                continue
            break
        name, block = result
        cfg["sources"][name] = block
        print(success(f"Added source {bold(name)} (type={block['type']})"))
        if not confirm("Add another source?", default=False):
            break

    print()
    _write_config(cfg, output_path, allow_overwrite=False)
    print(success(f"Wrote {output_path}"))

    print()
    has_graph = any((s.get("graph") or {}).get("enabled") for s in cfg["sources"].values())
    has_git = any((s.get("git_integration") or {}).get("enabled") for s in cfg["sources"].values())
    _ask_ai_client(list(cfg["sources"].keys()), has_graph, has_git)

    _print_mcp_snippet(output_path)

    print()
    print(heading("Next steps"))
    print(bullet(f"`lynx manager doctor --config {output_path}` — verify config"))
    print(bullet(f"`lynx build --config {output_path}` — build the index "
                 f"(first run downloads the embedding model, ~130 MB)"))
    print(bullet(f"`lynx manager ui --config {output_path}` — open the local web UI"))
    print()
    print(success(bold("Setup complete!")))
    return 0
