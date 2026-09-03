"""Source CRUD over config.json — one implementation, two front-ends.

The web UI (`POST /api/sources`, `DELETE /api/sources/{name}`) and the CLI
(`lynx source add` / `lynx source remove`) both mutate the same file with
the same rules, so the rules live here rather than inside the FastAPI
route handlers. A validation fix lands in both at once.

Deliberately FastAPI-free: `lynx source add` must not pay the ~300ms
FastAPI import just to append a JSON block, and `lynx manager ui` is not
a prerequisite for configuring a headless install.

Every mutation is read-modify-write on the *raw* JSON dict — a config
with one slightly off-schema field elsewhere shouldn't block adding a
source — followed by a round-trip through `load_config` for schema
validation before anything touches disk. The previous content is kept
as `<config>.bak`.
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import NamedTuple, Optional


# Source-name shape mirrors the validator used by the v2 config loader.
# Letter followed by letters / digits / underscore, max 40 chars.
SOURCE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,39}$")

SOURCE_NAME_HELP = (
    "Source name must start with a letter and contain only letters, "
    "digits, and underscores (max 40 chars)."
)

# Path fragments a new codebase source ignores by default: VCS, virtualenvs,
# dependency dirs, and build outputs. Build/dist/vendored dirs in particular
# mirror source files, so indexing them produces duplicate-content hits. Stored
# in forward-slash form; config load normalizes "/" to the OS separator.
DEFAULT_CODEBASE_IGNORES = [
    "/.git/", "/.venv/", "/venv/", "/node_modules/", "/__pycache__/",
    "/.idea/", "/.vscode/", "/dist/", "/build/", "/target/", "/.next/",
]


class CrudResult(NamedTuple):
    """Outcome of a config mutation.

    `message` is plain text (no markup) so the CLI can print it as-is and
    the UI can wrap it in a toast. `status` carries the HTTP-ish code the
    UI reports; the CLI only looks at `ok`. `purged_path` / `purge_error`
    let the UI keep rendering the storage path in a <code> span without
    this module knowing anything about HTML.
    """
    ok: bool
    message: str
    status: int = 200
    purged_path: Optional[str] = None
    purge_error: Optional[str] = None
    defaults_applied: bool = False
    # The block as actually written, defaults included. `--json` callers get
    # this rather than what they passed in: reporting the input would tell a
    # script the source has no ignores when the file on disk says otherwise.
    written_block: Optional[dict] = None


# How long to wait for a competing writer before giving up, and how old a
# lock has to be before we assume its owner died. The critical section is a
# read, a dict edit, a `load_config` on a tempfile and an `os.replace` —
# well under a second even on a cold disk, so a lock older than this is
# not a slow writer, it's a corpse.
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 60.0


@contextlib.contextmanager
def config_lock(config_path, timeout: float = _LOCK_TIMEOUT_SECONDS):
    """Serialize read-modify-write of config.json across processes.

    Both front-ends mutate the same file, and neither reads it under a
    lock: `lynx source add` from a terminal while the web UI is open on the
    sources page is enough for one write to silently drop the other, since
    each rewrites the whole file from the snapshot it read.

    An exclusive `O_CREAT|O_EXCL` sidecar file is the portable primitive —
    `fcntl.flock` doesn't exist on Windows and `msvcrt.locking` doesn't
    exist elsewhere. A lock left behind by a killed process is broken once
    it is older than `_LOCK_STALE_SECONDS`; the alternative (waiting
    forever on a dead owner) fails a working install for good.

    Failing to acquire is NOT fatal: the caller proceeds unlocked rather
    than refusing to work, because a config edit blocked by a stuck lock
    would be worse than the race it prevents.
    """
    lock_path = Path(config_path).with_name(Path(config_path).name + ".lock")
    fd = None
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                age = 0.0  # vanished between the two calls: retry immediately
            if age > _LOCK_STALE_SECONDS:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                break  # proceed unlocked — see the docstring
            time.sleep(0.05)
        except OSError as e:
            if e.errno == errno.EACCES:
                break  # read-only dir or a permissions quirk: don't block work
            raise
    try:
        if fd is not None:
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            except OSError:
                pass
        yield fd is not None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                lock_path.unlink()
            except OSError:
                pass


def apply_codebase_defaults(block: dict) -> dict:
    """Fill in sane defaults a new codebase source should have but the client
    may omit. Currently: ignore VCS/deps/build dirs unless the caller set its
    own list. Non-codebase blocks and explicit lists are left untouched."""
    if block.get("type") == "codebase" and not block.get("ignored_path_fragments"):
        block["ignored_path_fragments"] = list(DEFAULT_CODEBASE_IGNORES)
    return block


def load_config_dict(config_path) -> dict:
    """Read config.json as a raw dict (no schema validation).

    Used for read-modify-write of the source CRUD operations — we don't
    want the source-add flow to fail because some unrelated config field
    is slightly off-schema. Schema validation runs at write time via
    `validate_and_write_config`.

    Decoded as utf-8-sig: PowerShell's `Out-File -Encoding utf8` and older
    Windows editors prepend a BOM, and plain utf-8 leaves it in the string
    where it becomes an unreadable "Unexpected UTF-8 BOM" from the JSON
    parser. utf-8-sig strips a BOM when present and is identical to utf-8
    when it isn't.
    """
    raw = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected a JSON object at the top level, found "
            f"{type(raw).__name__}"
        )
    return raw


def validate_config_dict(config_dict: dict) -> Optional[str]:
    """Dry-run the dict through `load_config`. Returns None if it would
    load, an error message otherwise. Nothing on disk is touched beyond a
    throwaway tempfile.

    `load_config` is the only authority on the schema, so calling it is the
    only honest way to know the result will still load. Note that it
    validates the WHOLE file — including sources this mutation didn't touch
    — so a failure here may be about a *sibling* source (e.g. a codebase
    whose folder has since been moved).
    """
    content = json.dumps(config_dict, indent=2) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as tf:
        tf.write(content)
        tmp_path = Path(tf.name)
    try:
        try:
            from ..config import load_config
            load_config(tmp_path)
        except SystemExit as e:
            # load_config reports the specific problem on stderr and exits;
            # that text is the useful part, and it has already been printed.
            return (f"Schema validation failed (exit {e.code}). See the "
                    f"[config] error printed above for the specific field.")
        except Exception as e:
            return f"Validation error: {type(e).__name__}: {e}"
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return None


def write_config(config_dict: dict, config_path) -> Optional[str]:
    """Write the dict to disk with a `.bak` backup of the previous content.
    Assumes the dict was already validated. Returns None on success, or an
    error message string.

    The write is atomic — content to a sibling tempfile, then `os.replace`
    — because config.json is the file every entry point reads at startup.
    A crash or a full disk halfway through a plain write would leave a
    truncated config that nothing can load, and the `.bak` written moments
    earlier would be the only copy left.
    """
    content = json.dumps(config_dict, indent=2) + "\n"
    target = Path(config_path)
    staged = target.with_name(target.name + ".tmp")
    try:
        if target.exists():
            # copyfile, not read+write: a config saved with a BOM or in an
            # unusual encoding must come back byte-identical if restored.
            shutil.copyfile(target, target.with_suffix(target.suffix + ".bak"))
        staged.write_text(content, encoding="utf-8")
        os.replace(staged, target)
    except OSError as e:
        try:
            staged.unlink()
        except OSError:
            pass
        return f"Couldn't write config: {e}"
    return None


def validate_and_write_config(config_dict: dict, config_path) -> Optional[str]:
    """Validate, then write. The one-call form used by `add_source` and
    re-exported for the UI's routes."""
    err = validate_config_dict(config_dict)
    if err is not None:
        return err
    return write_config(config_dict, config_path)


def add_source(config_path, name: str, block: dict) -> CrudResult:
    """Add a `sources.<name>` block to config.json.

    `block` must carry at least a `type`; per-type required fields are
    enforced by the loader during validation, so a bad block is rejected
    with the loader's own message instead of a second, drifting copy of
    the schema here.
    """
    name = (name or "").strip()
    if not name or not SOURCE_NAME_RE.match(name):
        return CrudResult(False, SOURCE_NAME_HELP, 400)
    if not isinstance(block, dict) or not block.get("type"):
        return CrudResult(False, "`block` must be an object with a `type` field.", 400)

    with config_lock(config_path):
        try:
            cfg = load_config_dict(config_path)
        except Exception as e:
            return CrudResult(False, f"Couldn't read existing config: {e}", 500)

        sources = cfg.get("sources")
        if sources is None:
            sources = {}
        elif not isinstance(sources, dict):
            return CrudResult(
                False,
                f"'sources' in {config_path} is a {type(sources).__name__}, not an "
                f"object — fix it by hand before adding sources.",
                422,
            )
        if name in sources:
            return CrudResult(False, f"A source named {name!r} already exists.", 409)

        # Note the injection so the caller can report it rather than perform
        # it silently: someone who handed over an explicit block (the CLI's
        # --block, the UI's form) should be told the source will skip paths
        # it never mentioned. `apply_codebase_defaults` mutates in place, so
        # the "before" state has to be captured first.
        block = dict(block)
        had_ignores = bool(block.get("ignored_path_fragments"))
        filled = apply_codebase_defaults(block)
        defaults_applied = not had_ignores and bool(filled.get("ignored_path_fragments"))

        sources[name] = filled
        cfg["sources"] = sources

        err = validate_and_write_config(cfg, config_path)
        if err is not None:
            # Same guidance `remove_source` gives: the loader validates the
            # whole file, so the rejection may well be about a source this
            # command never touched.
            return CrudResult(
                False,
                err + " Nothing was changed. If the [config] error names a "
                      "different source, fix or remove that one first — the "
                      "loader validates the whole file.",
                422,
            )
    return CrudResult(True, f"Source {name!r} added.", 200,
                      defaults_applied=defaults_applied, written_block=filled)


def remove_source(config_path, name: str, purge: bool = False) -> CrudResult:
    """Remove `sources.<name>` from config.json.

    With `purge`, also delete the source's index directory.

    Order: validate the would-be config first (dry run, nothing written),
    THEN delete the index, THEN write. Each step runs only once the
    previous one is known to succeed, because both alternative orderings
    failed in the field:

      - write-then-purge left an orphaned index nothing could reach when
        the purge failed — and on Windows a running `lynx serve` holds the
        ChromaDB files open, so the failing purge was the common case, not
        the edge;
      - purge-then-validate deleted the index and then refused the config
        write when validation failed for an unrelated reason — the loader
        validates the whole file, so a broken *sibling* source (a moved
        folder) was enough to strand the operation halfway.

    The remaining window — validated, purged, then the write itself fails
    — is a disk-level error; its message says the index is gone and how to
    rebuild it.
    """
    with config_lock(config_path):
        try:
            cfg = load_config_dict(config_path)
        except Exception as e:
            return CrudResult(False, f"Couldn't read existing config: {e}", 500)

        sources = cfg.get("sources")
        if not isinstance(sources, dict) or name not in sources:
            return CrudResult(False, f"Source {name!r} not found in config.", 404)

        del sources[name]
        cfg["sources"] = sources

        err = validate_config_dict(cfg)
        if err is not None:
            return CrudResult(
                False,
                err + " Nothing was changed. If the [config] error names a "
                      "different source, fix or remove that one first — the "
                      "loader validates the whole file.",
                422,
            )

        purged_path = None
        if purge:
            # `storage_path` is resolved relative to the config file, same as
            # the loader does — not the CWD, which differs between the UI
            # (launch dir) and the CLI (wherever the user happens to be).
            root = Path(cfg.get("storage_path", "./rag_storage"))
            if not root.is_absolute():
                root = Path(config_path).resolve().parent / root
            src_storage = root / name
            if src_storage.exists():
                # Ask first, rather than relying on the delete to fail. Windows
                # refuses to unlink a file a live process holds open, which is
                # what used to stop this; everywhere else the unlink succeeds
                # and the running session is left reading an index that no
                # longer exists on disk.
                from ..integrity import store_in_use_by_other_process
                from .. import ownership
                if store_in_use_by_other_process(src_storage):
                    return CrudResult(
                        False,
                        f"Couldn't delete the index at {src_storage}: "
                        f"{ownership.describe(src_storage)}. Source {name!r} was "
                        f"left in the config — stop that process and run the "
                        f"same command again.",
                        409, purge_error="index in use",
                    )
                try:
                    shutil.rmtree(src_storage)
                except OSError as e:
                    return CrudResult(
                        False,
                        f"Couldn't delete the index at {src_storage}: {e}. "
                        f"Source {name!r} was left in the config — stop whatever "
                        f"is holding the index open (a running `lynx serve`, the "
                        f"web UI, an open file browser) and run the same command "
                        f"again.",
                        409, purge_error=str(e),
                    )
                purged_path = str(src_storage)

        err = write_config(cfg, config_path)
        if err is not None:
            if purged_path:
                err += (f" NOTE: the index at {purged_path} was already deleted; "
                        f"`lynx build --source {name}` rebuilds it.")
            return CrudResult(False, err, 422, purged_path=purged_path)

    return CrudResult(True, f"Source {name!r} removed.", 200,
                      purged_path=purged_path)
