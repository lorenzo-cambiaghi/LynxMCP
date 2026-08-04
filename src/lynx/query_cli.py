"""Retrieval and navigation subcommands — the CLI face of the MCP tools
that answer questions about indexed code.

Until these existed the CLI could search and manage, but the ten tools an
agent uses to actually navigate a codebase — find a definition, list
usages, describe a symbol, compute a blast radius — had no terminal
equivalent. Anyone without an MCP client, and any script, was locked out
of most of what Lynx does.

Two rules hold the two surfaces together:

  - **Names map 1:1.** `find_definition` is `lynx find-definition`. Nobody
    should have to translate between the tool table and `--help`.
  - **Rendering is shared, never copied.** Every command calls the same
    `manager.*` method and the same `_format_*` renderer the MCP tool
    calls, so the terminal shows byte-for-byte what the model receives. A
    formatting fix lands in both at once; there is no second copy to
    forget.

`--json` on every command emits the raw structure behind that text —
exactly one object on stdout, always carrying `ok`, `operation` and
`source`, so a script never has to tell "no output" apart from "output
saying no". Implementations live here rather than in `cli.py` to keep the
argparse wiring readable; `cli.py` lazy-imports this module so `lynx
serve` never pays for it.
"""
from __future__ import annotations

import sys
from typing import Optional

from ._format import (
    _format_definition_results,
    _format_deep_response,
    _format_describe_symbol,
    _format_impact,
    _format_module_summary,
    _format_repo_overview,
    _format_search_diff,
    _format_similar_results,
    _format_test_results,
    _format_usage_results,
)


# ----------------------------------------------------------------------
# Source resolution
# ----------------------------------------------------------------------


def _is_codebase(backend) -> bool:
    return getattr(backend, "type_name", None) == "codebase"


def _has_git(backend) -> bool:
    return (
        _is_codebase(backend)
        and bool(getattr(backend, "source_config", {})
                 .get("git_integration", {}).get("enabled"))
    )


def resolve_backend(manager, requested: Optional[str], predicate, kind: str):
    """Pick the source a command should run against. Returns `(name, error)`.

    Mirrors `server._resolve_source`: an explicit name is validated against
    the predicate, otherwise the single qualifying source is used and both
    zero and many are errors. Errors are returned, never printed-and-exited
    — the caller decides whether that becomes a line on stderr or a JSON
    object, which is the whole reason `--json` can promise an object in
    every case.
    """
    candidates = [n for n, b in manager.backends.items() if predicate(b)]
    if not candidates:
        return None, f"no {kind} is configured."
    if requested:
        if requested not in manager.backends:
            return None, (f"unknown source {requested!r}. "
                          f"Available: {list(manager.backends)}")
        if requested not in candidates:
            return None, f"source {requested!r} is not a {kind}."
        return requested, None
    if len(candidates) == 1:
        return candidates[0], None
    return None, (f"multiple sources qualify ({candidates}); "
                  f"specify --source NAME")


# ----------------------------------------------------------------------
# Shared plumbing
# ----------------------------------------------------------------------


class _Ctx:
    """Everything a command needs after the boilerplate has run."""

    def __init__(self, manager, source, as_json):
        self.manager = manager
        self.source = source
        self.as_json = as_json


def _prepare(args, build_manager, muted, predicate, kind, operation):
    """Build the manager and resolve the source, or report why not.

    Returns `(ctx, exit_code)`; exactly one of them is None.
    """
    from .cli import _emit_json

    as_json = bool(getattr(args, "as_json", False))
    with muted(as_json):
        _config, manager = build_manager(getattr(args, "config", None))
    source, err = resolve_backend(
        manager, getattr(args, "source", None), predicate, kind,
    )
    if err:
        if as_json:
            _emit_json({"ok": False, "operation": operation,
                        "source": None, "error": err})
        else:
            print(f"error: {err}", file=sys.stderr)
        return None, 2
    return _Ctx(manager, source, as_json), None


def _deliver(ctx, operation, text, data) -> int:
    from .cli import _emit_json

    if ctx.as_json:
        _emit_json({"ok": True, "operation": operation,
                    "source": ctx.source, **data})
    else:
        print(text)
    return 0


def _failed(ctx, operation, exc) -> int:
    """An unexpected error from the manager. The MCP tools turn these into
    an `Error: ...` string for the model; the CLI owes a script the same
    guarantee it makes everywhere else — one object, non-zero exit."""
    from .cli import _emit_json

    msg = f"{type(exc).__name__}: {exc}"
    if ctx.as_json:
        _emit_json({"ok": False, "operation": operation,
                    "source": ctx.source, "error": msg})
    else:
        print(f"error: {msg}", file=sys.stderr)
    return 1


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def _cmd_deep_search(args, build_manager, muted) -> int:
    """Multi-query escalation search. Mirrors the `deep_search` tool,
    including the single-source-only restriction on --mode."""
    from .cli import _emit_json

    operation = "deep_search"
    as_json = bool(getattr(args, "as_json", False))
    with muted(as_json):
        config, manager = build_manager(getattr(args, "config", None))

    names = [n for n in (args.source or []) if n]
    unknown = [n for n in names if n not in manager.backends]
    if unknown:
        err = (f"unknown source(s) {unknown}. "
               f"Available: {list(manager.backends)}")
        if as_json:
            _emit_json({"ok": False, "operation": operation,
                        "source": None, "error": err})
        else:
            print(f"error: {err}", file=sys.stderr)
        return 2

    top_k = args.top_k if args.top_k is not None else config.search.default_top_k
    filters = dict(file_glob=args.glob, extensions=args.ext,
                   path_contains=args.path_contains)
    single = names[0] if len(names) == 1 else None
    try:
        with muted(as_json):
            if single is None:
                response = manager.deep_search_all(
                    queries=args.queries, top_k=top_k,
                    min_score=args.min_score, min_results=args.min_results,
                    only=names or None, **filters,
                )
                label = "all sources" if not names else f"sources {names}"
            else:
                response = manager.deep_search(
                    single, queries=args.queries, top_k=top_k, mode=args.mode,
                    min_score=args.min_score, min_results=args.min_results,
                    return_all_variants=args.return_all_variants, **filters,
                )
                label = f"source {single!r}"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if as_json:
            _emit_json({"ok": False, "operation": operation,
                        "source": single, "error": msg})
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    meta_parts = []
    if args.mode and single is not None:
        meta_parts.append(f"mode={args.mode!r}")
    if args.glob:
        meta_parts.append(f"file_glob={args.glob!r}")
    if args.ext:
        meta_parts.append(f"extensions={list(args.ext)!r}")
    if args.path_contains:
        meta_parts.append(f"path_contains={args.path_contains!r}")
    meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""

    if as_json:
        _emit_json({"ok": True, "operation": operation,
                    "source": single, "sources": names or None,
                    "queries": args.queries, **response})
        return 0
    print(_format_deep_response(response, args.queries, label, meta_suffix))
    return 0


def _cmd_find_definition(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "find_definition")
    if ctx is None:
        return rc
    try:
        results = ctx.manager.find_definition(ctx.source, args.symbol,
                                              limit=args.limit)
    except Exception as e:
        return _failed(ctx, "find_definition", e)
    return _deliver(ctx, "find_definition",
                    _format_definition_results(args.symbol, results),
                    {"symbol": args.symbol, "count": len(results),
                     "results": results})


def _cmd_find_usages(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "find_usages")
    if ctx is None:
        return rc
    try:
        results = ctx.manager.find_usages(ctx.source, args.symbol,
                                          limit=args.limit)
    except Exception as e:
        return _failed(ctx, "find_usages", e)
    return _deliver(ctx, "find_usages",
                    _format_usage_results(args.symbol, results),
                    {"symbol": args.symbol, "count": len(results),
                     "results": results})


def _cmd_find_tests_for(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "find_tests_for")
    if ctx is None:
        return rc
    try:
        results = ctx.manager.find_tests_for(
            ctx.source, args.symbol, limit=args.limit,
            test_path_pattern=args.test_path_pattern,
        )
    except Exception as e:
        return _failed(ctx, "find_tests_for", e)
    return _deliver(ctx, "find_tests_for",
                    _format_test_results(args.symbol, results),
                    {"symbol": args.symbol, "count": len(results),
                     "results": results})


def _cmd_find_similar(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "find_similar")
    if ctx is None:
        return rc

    snippet = args.snippet
    if args.file:
        # A snippet on a shell command line is painful to quote and loses
        # its indentation; pointing at a file is the terminal-native way.
        from pathlib import Path
        try:
            snippet = Path(args.file).expanduser().read_text(
                encoding="utf-8-sig", errors="replace")
        except OSError as e:
            return _failed(ctx, "find_similar", e)
    if not snippet or not snippet.strip():
        return _failed(ctx, "find_similar",
                       ValueError("empty snippet: pass code as the argument "
                                  "or point --file at a readable file"))
    try:
        results = ctx.manager.find_similar(ctx.source, snippet,
                                           top_k=args.top_k)
    except Exception as e:
        return _failed(ctx, "find_similar", e)
    return _deliver(ctx, "find_similar", _format_similar_results(results),
                    {"count": len(results), "results": results})


def _cmd_describe_symbol(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "describe_symbol")
    if ctx is None:
        return rc
    try:
        data = ctx.manager.describe_symbol(
            ctx.source, args.symbol,
            callers_limit=args.callers_limit,
            callees_limit=args.callees_limit,
            tests_limit=args.tests_limit,
        )
    except Exception as e:
        return _failed(ctx, "describe_symbol", e)
    return _deliver(ctx, "describe_symbol",
                    _format_describe_symbol(args.symbol, data),
                    {"symbol": args.symbol, **data})


def _cmd_impact(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "impact")
    if ctx is None:
        return rc
    try:
        data = ctx.manager.impact_of(ctx.source, args.symbol,
                                     max_depth=args.max_depth,
                                     tests_limit=args.tests_limit)
    except Exception as e:
        return _failed(ctx, "impact", e)
    return _deliver(ctx, "impact", _format_impact(args.symbol, data),
                    {"symbol": args.symbol, **data})


def _cmd_repo_overview(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "repo_overview")
    if ctx is None:
        return rc
    try:
        data = ctx.manager.repo_overview(ctx.source)
    except Exception as e:
        return _failed(ctx, "repo_overview", e)
    return _deliver(ctx, "repo_overview", _format_repo_overview(data), data)


def _cmd_module_summary(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _is_codebase,
                       "codebase source", "module_summary")
    if ctx is None:
        return rc
    try:
        data = ctx.manager.module_summary(ctx.source, args.file,
                                          limit=args.limit)
    except Exception as e:
        return _failed(ctx, "module_summary", e)
    return _deliver(ctx, "module_summary",
                    _format_module_summary(args.file, data),
                    {"file": args.file, **data})


def _cmd_search_diff(args, build_manager, muted) -> int:
    ctx, rc = _prepare(args, build_manager, muted, _has_git,
                       "git-enabled codebase source", "search_diff")
    if ctx is None:
        return rc
    try:
        data = ctx.manager.search_diff(ctx.source, args.query,
                                       base=args.base, top_k=args.top_k)
    except Exception as e:
        return _failed(ctx, "search_diff", e)
    return _deliver(ctx, "search_diff", _format_search_diff(ctx.source, data),
                    {"query": args.query, **data})


_DISPATCH = {
    "deep-search": _cmd_deep_search,
    "find-definition": _cmd_find_definition,
    "find-usages": _cmd_find_usages,
    "find-tests-for": _cmd_find_tests_for,
    "find-similar": _cmd_find_similar,
    "describe-symbol": _cmd_describe_symbol,
    "impact": _cmd_impact,
    "repo-overview": _cmd_repo_overview,
    "module-summary": _cmd_module_summary,
    "search-diff": _cmd_search_diff,
}

# The commands this module serves, for `cli.py` to route on without
# importing it (which would cost every `lynx serve` the import).
COMMANDS = tuple(_DISPATCH)


def dispatch(command: str, args, build_manager, muted) -> int:
    """Route one of `COMMANDS` to its implementation.

    `build_manager` and `muted` are injected rather than imported so this
    module doesn't import `cli` at module level — the dependency runs one
    way, and the commands stay unit-testable with a stub manager.
    """
    return _DISPATCH[command](args, build_manager, muted)
