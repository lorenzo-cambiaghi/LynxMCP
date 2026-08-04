"""`lynx source add` / `lynx source remove` — configuring a Lynx install
without the web UI.

Before this existed, `lynx manager init` wrote a config with `sources: {}`
and told you to open a browser: a headless box or a CI job had no way to
declare a source other than hand-writing JSON. These commands go through
`manager/sources.py`, the same validate-then-write path the UI's
POST/DELETE `/api/sources` uses, so both front-ends accept and reject the
same configs.

Nothing here loads an embedding model: `lynx source` only touches
config.json, so the whole file runs in well under a second.

Covered:
  1.  add codebase with explicit --ext → block written, path absolutised
  2.  add codebase without --ext → extensions detected from the folder
  3.  --graph / --no-watcher / --no-git / --ignore land in the block
  4.  add webdoc (and: --url is required)
  5.  add pdf with --no-recursive / --extractor
  6.  --block escape hatch, and the default-ignore injection it announces
  7.  duplicate name → exit 1, config untouched
  8.  invalid name → exit 1
  9.  missing directory → exit 2
  10. a block the loader rejects leaves the original config intact
  11. a .bak of the previous config is kept, byte-for-byte
  12. remove keeps the index; remove --purge deletes it
  13. remove of an unknown source → exit 1
  14. a purge that CAN'T happen aborts everything and stays retryable
  15. BOM-prefixed configs load (PowerShell writes them by default)
  16. a malformed config is reported, not raised at the user
  17. --json output on both subcommands, success and failure
  18. --build receives the stored (stripped) source name
  19. UI and CLI produce identical configs and reject identical names
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lynx.cli import main as cli_main


def _write_config(tmp_path: Path, sources: dict | None = None) -> Path:
    cfg = {
        "config_version": 2,
        "storage_path": "./storage",
        "sources": sources if sources is not None else {},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return p


def _sources(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))["sources"]


@pytest.fixture
def code_dir(tmp_path: Path) -> Path:
    d = tmp_path / "code"
    d.mkdir()
    (d / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (d / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    (d / "notes.md").write_text("# notes\n", encoding="utf-8")
    return d


# ----------------------------------------------------------------------
# add — codebase
# ----------------------------------------------------------------------


def test_add_codebase_explicit_extensions(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    rc = cli_main([
        "source", "add", "demo", "--config", str(cfg),
        "--type", "codebase", "--path", str(code_dir),
        "--ext", ".py", "--ext", "MD",  # dot optional, case-insensitive
    ])
    assert rc == 0

    block = _sources(cfg)["demo"]
    assert block["type"] == "codebase"
    assert block["supported_extensions"] == [".py", ".md"]
    # Stored absolute: the CWD of the shell that ran the command is not the
    # config's directory, so a relative path would resolve differently later.
    assert Path(block["path"]).is_absolute()
    assert Path(block["path"]) == code_dir.resolve()
    # Defaults the UI applies must apply here too.
    assert "/node_modules/" in block["ignored_path_fragments"]


def test_add_codebase_detects_extensions(tmp_path, code_dir, capsys):
    cfg = _write_config(tmp_path)
    rc = cli_main([
        "source", "add", "demo", "--config", str(cfg),
        "--type", "codebase", "--path", str(code_dir),
    ])
    assert rc == 0
    assert set(_sources(cfg)["demo"]["supported_extensions"]) == {".py", ".md"}
    # The scan is not silent — the user needs to see what was picked.
    assert "detected extensions" in capsys.readouterr().out


def test_add_codebase_optional_flags(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    rc = cli_main([
        "source", "add", "demo", "--config", str(cfg),
        "--type", "codebase", "--path", str(code_dir),
        "--ext", ".py", "--graph", "--no-watcher", "--no-git",
        "--ignore", "/Library/", "--watcher-debounce", "5",
    ])
    assert rc == 0

    block = _sources(cfg)["demo"]
    assert block["graph"] == {"enabled": True}
    assert block["watcher"] == {"enabled": False, "debounce_seconds": 5.0}
    assert block["git_integration"] == {"enabled": False}
    # An explicit --ignore replaces the defaults rather than extending them.
    assert block["ignored_path_fragments"] == ["/Library/"]


def test_add_codebase_omits_untouched_keys(tmp_path, code_dir):
    """Flags the user didn't pass must not be written out: the loader's
    defaults should keep applying (and keep evolving) for this source."""
    cfg = _write_config(tmp_path)
    cli_main([
        "source", "add", "demo", "--config", str(cfg),
        "--type", "codebase", "--path", str(code_dir), "--ext", ".py",
    ])
    block = _sources(cfg)["demo"]
    assert "watcher" not in block
    assert "git_integration" not in block
    assert "graph" not in block


# ----------------------------------------------------------------------
# add — webdoc / pdf
# ----------------------------------------------------------------------


def test_add_webdoc(tmp_path):
    cfg = _write_config(tmp_path)
    rc = cli_main([
        "source", "add", "docs", "--config", str(cfg),
        "--type", "webdoc", "--url", "https://example.com/docs",
        "--max-pages", "50", "--render-js", "--exclude-url", "/blog/",
        "--allow-cross-origin",
    ])
    assert rc == 0

    block = _sources(cfg)["docs"]
    assert block["url"] == "https://example.com/docs"
    assert block["max_pages"] == 50
    assert block["render_js"] is True
    assert block["exclude_url_patterns"] == ["/blog/"]
    assert block["same_origin_only"] is False


def test_add_webdoc_requires_url(tmp_path):
    cfg = _write_config(tmp_path)
    assert cli_main(["source", "add", "docs", "--config", str(cfg),
                     "--type", "webdoc"]) == 2
    assert _sources(cfg) == {}


def test_add_pdf(tmp_path):
    cfg = _write_config(tmp_path)
    pdfs = tmp_path / "books"
    pdfs.mkdir()
    rc = cli_main([
        "source", "add", "books", "--config", str(cfg),
        "--type", "pdf", "--path", str(pdfs),
        "--no-recursive", "--extractor", "pypdf",
    ])
    assert rc == 0

    block = _sources(cfg)["books"]
    assert block["type"] == "pdf"
    assert block["recursive"] is False
    assert block["extractor"] == {"backend": "pypdf"}


def test_add_block_escape_hatch(tmp_path, code_dir):
    """--block covers fields the typed flags don't, validated identically."""
    cfg = _write_config(tmp_path)
    block = {"type": "codebase", "path": str(code_dir),
             "supported_extensions": [".py"]}
    rc = cli_main(["source", "add", "raw", "--config", str(cfg),
                   "--block", json.dumps(block)])
    assert rc == 0
    assert _sources(cfg)["raw"]["supported_extensions"] == [".py"]


def test_add_block_reports_injected_defaults(tmp_path, code_dir, capsys):
    """A "raw" block still gets the default ignore list — indexing
    node_modules is a worse surprise than an extra line of output — but the
    injection is announced rather than silent."""
    cfg = _write_config(tmp_path)
    block = {"type": "codebase", "path": str(code_dir),
             "supported_extensions": [".py"]}
    cli_main(["source", "add", "raw", "--config", str(cfg),
              "--block", json.dumps(block)])

    assert "/node_modules/" in _sources(cfg)["raw"]["ignored_path_fragments"]
    assert "ignore" in capsys.readouterr().out.lower()


def test_add_block_with_explicit_ignores_is_left_alone(tmp_path, code_dir, capsys):
    cfg = _write_config(tmp_path)
    block = {"type": "codebase", "path": str(code_dir),
             "supported_extensions": [".py"],
             "ignored_path_fragments": ["/vendor/"]}
    cli_main(["source", "add", "raw", "--config", str(cfg),
              "--block", json.dumps(block)])

    assert _sources(cfg)["raw"]["ignored_path_fragments"] == ["/vendor/"]
    assert "ignore" not in capsys.readouterr().out.lower()


# ----------------------------------------------------------------------
# add — rejection paths. Every one of these must leave config.json alone.
# ----------------------------------------------------------------------


def test_add_duplicate_name_rejected(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
              "codebase", "--path", str(code_dir), "--ext", ".py"])
    before = cfg.read_text(encoding="utf-8")

    rc = cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
                   "codebase", "--path", str(code_dir), "--ext", ".md"])
    assert rc == 1
    assert cfg.read_text(encoding="utf-8") == before


def test_add_invalid_name_rejected(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    rc = cli_main(["source", "add", "9lives", "--config", str(cfg), "--type",
                   "codebase", "--path", str(code_dir), "--ext", ".py"])
    assert rc == 1
    assert _sources(cfg) == {}


def test_add_missing_directory_rejected(tmp_path):
    cfg = _write_config(tmp_path)
    rc = cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
                   "codebase", "--path", str(tmp_path / "nope")])
    assert rc == 2
    assert _sources(cfg) == {}


def test_add_block_rejected_by_loader_leaves_config_intact(tmp_path):
    """A block that passes the CLI's own checks but fails schema validation
    (here: a codebase source with no `path`) must not corrupt the file."""
    cfg = _write_config(tmp_path, {"keep": {"type": "webdoc",
                                            "url": "https://example.com"}})
    before = cfg.read_text(encoding="utf-8")

    rc = cli_main(["source", "add", "broken", "--config", str(cfg),
                   "--block", '{"type": "codebase"}'])
    assert rc == 1
    assert cfg.read_text(encoding="utf-8") == before


def test_add_missing_config_file(tmp_path):
    rc = cli_main(["source", "add", "demo", "--config",
                   str(tmp_path / "absent.json"), "--type", "webdoc",
                   "--url", "https://example.com"])
    assert rc == 1


def test_add_keeps_backup_of_previous_config(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    original = cfg.read_text(encoding="utf-8")
    cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
              "codebase", "--path", str(code_dir), "--ext", ".py"])

    backup = cfg.with_suffix(".json.bak")
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == original


# ----------------------------------------------------------------------
# remove
# ----------------------------------------------------------------------


def _config_with_source(tmp_path, code_dir):
    return _write_config(tmp_path, {
        "demo": {"type": "codebase", "path": str(code_dir),
                 "supported_extensions": [".py"]},
    })


def test_remove_keeps_index_by_default(tmp_path, code_dir):
    cfg = _config_with_source(tmp_path, code_dir)
    storage = tmp_path / "storage" / "demo"
    storage.mkdir(parents=True)
    (storage / "chroma.sqlite3").write_text("x", encoding="utf-8")

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg), "--yes"])
    assert rc == 0
    assert _sources(cfg) == {}
    # Dropping a source from the config is cheap to undo; deleting a built
    # index is not, so it survives until --purge asks for it explicitly.
    assert storage.is_dir()


def test_remove_purge_deletes_index(tmp_path, code_dir):
    cfg = _config_with_source(tmp_path, code_dir)
    storage = tmp_path / "storage" / "demo"
    storage.mkdir(parents=True)
    (storage / "chroma.sqlite3").write_text("x", encoding="utf-8")

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg),
                   "--purge", "--yes"])
    assert rc == 0
    assert _sources(cfg) == {}
    assert not storage.exists()


def test_remove_purge_resolves_storage_against_config_dir(tmp_path, code_dir,
                                                          monkeypatch):
    """`storage_path` is relative to the config file, exactly as the loader
    reads it — not to whatever directory the command was typed in."""
    cfg = _config_with_source(tmp_path, code_dir)
    storage = tmp_path / "storage" / "demo"
    storage.mkdir(parents=True)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert cli_main(["source", "remove", "demo", "--config", str(cfg),
                     "--purge", "--yes"]) == 0
    assert not storage.exists()


def test_remove_unknown_source(tmp_path, code_dir):
    cfg = _config_with_source(tmp_path, code_dir)
    rc = cli_main(["source", "remove", "ghost", "--config", str(cfg), "--yes"])
    assert rc == 1
    assert "demo" in _sources(cfg)


def test_source_without_subcommand_is_an_error(tmp_path):
    assert cli_main(["source"]) == 2


# ----------------------------------------------------------------------
# A purge that can't happen must not half-happen.
#
# The first cut deleted the config entry and *then* the folder. On Windows a
# running `lynx serve` holds the ChromaDB file open, so rmtree raised
# WinError 32 with the entry already gone: exit 0, "removed", index still on
# disk, and no way to retry --purge because the source was no longer in the
# config. rmtree is monkeypatched here rather than locking a real file so
# the test means the same thing on POSIX.
# ----------------------------------------------------------------------


@pytest.fixture
def unremovable(monkeypatch):
    import shutil

    def _boom(*a, **kw):
        raise OSError(32, "The process cannot access the file because it is "
                          "being used by another process")
    monkeypatch.setattr(shutil, "rmtree", _boom)


def test_failed_purge_changes_nothing(tmp_path, code_dir, unremovable):
    cfg = _config_with_source(tmp_path, code_dir)
    before = cfg.read_text(encoding="utf-8")
    storage = tmp_path / "storage" / "demo"
    storage.mkdir(parents=True)

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg),
                   "--purge", "--yes"])

    assert rc == 1                                    # not a silent success
    assert cfg.read_text(encoding="utf-8") == before  # still removable later
    assert storage.is_dir()
    assert "demo" in _sources(cfg)


def test_failed_purge_says_what_to_do(tmp_path, code_dir, unremovable, capsys):
    cfg = _config_with_source(tmp_path, code_dir)
    (tmp_path / "storage" / "demo").mkdir(parents=True)

    cli_main(["source", "remove", "demo", "--config", str(cfg),
              "--purge", "--yes"])

    err = capsys.readouterr().err
    assert "lynx serve" in err          # names the likely culprit
    assert "left in the config" in err  # states what did NOT happen


def test_failed_purge_is_retryable(tmp_path, code_dir, monkeypatch):
    """The whole point of aborting: once the lock is gone, the same command
    works. The first cut left an orphan no command could reach."""
    import shutil
    cfg = _config_with_source(tmp_path, code_dir)
    storage = tmp_path / "storage" / "demo"
    storage.mkdir(parents=True)

    real_rmtree = shutil.rmtree

    def _locked(*a, **kw):
        raise OSError(32, "locked")

    monkeypatch.setattr(shutil, "rmtree", _locked)
    assert cli_main(["source", "remove", "demo", "--config", str(cfg),
                     "--purge", "--yes"]) == 1

    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    assert cli_main(["source", "remove", "demo", "--config", str(cfg),
                     "--purge", "--yes"]) == 0
    assert not storage.exists()
    assert _sources(cfg) == {}


# ----------------------------------------------------------------------
# Validation runs BEFORE the purge.
#
# The loader validates the WHOLE file, so removing healthy alpha fails
# while sibling beta's folder is missing. With the previous order (purge,
# then validate) that deleted alpha's index and then refused the config
# write — the exact half-state the purge fix existed to prevent, reachable
# through a source the command never touched. Reproduced live before the
# reorder.
# ----------------------------------------------------------------------


def _config_with_broken_sibling(tmp_path, code_dir):
    return _write_config(tmp_path, {
        "alpha": {"type": "codebase", "path": str(code_dir),
                  "supported_extensions": [".py"]},
        "beta": {"type": "codebase", "path": str(tmp_path / "gone"),
                 "supported_extensions": [".py"]},
    })


def test_remove_with_broken_sibling_costs_nothing(tmp_path, code_dir, capsys):
    cfg = _config_with_broken_sibling(tmp_path, code_dir)
    storage = tmp_path / "storage" / "alpha"
    storage.mkdir(parents=True)
    (storage / "index.bin").write_text("x", encoding="utf-8")

    rc = cli_main(["source", "remove", "alpha", "--config", str(cfg),
                   "--purge", "--yes"])

    assert rc == 1
    assert "alpha" in _sources(cfg)   # config untouched
    assert storage.is_dir()           # index untouched — the new guarantee
    err = capsys.readouterr().err
    assert "Nothing was changed" in err
    # The failure is beta's, not alpha's — the message must say where to look.
    assert "different source" in err


def test_add_with_broken_sibling_gets_the_same_guidance(tmp_path, code_dir, capsys):
    """`remove` explained that the rejection may be about another source;
    `add` failed the same way and said only "Schema validation failed"."""
    cfg = _config_with_broken_sibling(tmp_path, code_dir)

    rc = cli_main(["source", "add", "gamma", "--config", str(cfg), "--type",
                   "codebase", "--path", str(code_dir), "--ext", ".py"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Nothing was changed" in err
    assert "different source" in err
    assert "gamma" not in _sources(cfg)


def test_removing_the_broken_sibling_itself_works(tmp_path, code_dir):
    """The escape hatch: config-minus-beta validates, so beta is removable —
    and afterwards alpha is too."""
    cfg = _config_with_broken_sibling(tmp_path, code_dir)

    assert cli_main(["source", "remove", "beta", "--config", str(cfg),
                     "--yes"]) == 0
    assert cli_main(["source", "remove", "alpha", "--config", str(cfg),
                     "--yes"]) == 0
    assert _sources(cfg) == {}


# ----------------------------------------------------------------------
# The UI must release its own ChromaDB handles before purging.
#
# Measured on Windows: a process that has built a SourceManager holds the
# source's index files open (WinError 32 on data_level0.bin) even after
# dropping the reference — chromadb caches the client system per path. So
# a UI purge that doesn't dispose the manager first 409s against the UI's
# own lock, telling the user to stop the very process they're clicking in.
# ----------------------------------------------------------------------


def test_ui_purge_disposes_its_manager_first(tmp_path, code_dir, monkeypatch):
    from chromadb.api.shared_system_client import SharedSystemClient
    from fastapi.testclient import TestClient

    import lynx.manager.ui.routes as routes_mod
    from lynx.manager.sources import CrudResult
    from lynx.manager.ui.app import create_app

    cfg = _write_config(tmp_path, {
        "demo": {"type": "codebase", "path": str(code_dir),
                 "supported_extensions": [".py"]},
    })
    app = create_app(cfg)
    app.state.manager = object()  # a "loaded" manager holding handles

    # Nulling the field is NOT what releases the ChromaDB files — chromadb
    # caches the client system per storage path, so the handles survive
    # every dropped Python reference. Asserting only `manager is None`
    # would keep passing against the exact behaviour that caused the bug.
    clears = []
    monkeypatch.setattr(SharedSystemClient, "clear_system_cache",
                        classmethod(lambda cls: clears.append(1)))

    seen = {}

    def _spy_remove(config_path, name, purge=False):
        seen["manager_when_purging"] = app.state.manager
        seen["clears_when_purging"] = len(clears)
        return CrudResult(True, f"Source {name!r} removed.", 200,
                          purged_path="whatever")

    monkeypatch.setattr(routes_mod, "_remove_source", _spy_remove)

    r = TestClient(app).delete("/api/sources/demo?purge=true")

    assert r.status_code == 200
    # By the time the purge ran, this process had let go of the index —
    # both the reference and the cached chromadb client behind it.
    assert seen["manager_when_purging"] is None
    assert seen["clears_when_purging"] == 1


def test_ui_plain_delete_keeps_the_manager_until_success(tmp_path, code_dir,
                                                         monkeypatch):
    """Without a purge there is no rmtree to unblock: the manager is only
    invalidated after the config write succeeds, as before."""
    from fastapi.testclient import TestClient

    import lynx.manager.ui.routes as routes_mod
    from lynx.manager.sources import CrudResult
    from lynx.manager.ui.app import create_app

    cfg = _write_config(tmp_path, {
        "demo": {"type": "codebase", "path": str(code_dir),
                 "supported_extensions": [".py"]},
    })
    app = create_app(cfg)
    sentinel = object()
    app.state.manager = sentinel

    seen = {}

    def _spy_remove(config_path, name, purge=False):
        seen["manager_when_removing"] = app.state.manager
        return CrudResult(True, f"Source {name!r} removed.", 200)

    monkeypatch.setattr(routes_mod, "_remove_source", _spy_remove)

    r = TestClient(app).delete("/api/sources/demo")

    assert r.status_code == 200
    assert seen["manager_when_removing"] is sentinel  # still loaded during
    assert app.state.manager is None                  # invalidated after


# ----------------------------------------------------------------------
# Encoding, atomicity, malformed input
# ----------------------------------------------------------------------


def test_config_with_utf8_bom_is_readable(tmp_path, code_dir):
    """PowerShell's `Out-File -Encoding utf8` writes a BOM. Read as plain
    utf-8 it survives into the string and json.loads rejects it, so a config
    that looks fine in an editor failed with an error pointing at column 1."""
    cfg = _write_config(tmp_path)
    cfg.write_text("﻿" + cfg.read_text(encoding="utf-8"), encoding="utf-8")

    rc = cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
                   "codebase", "--path", str(code_dir), "--ext", ".py"])
    assert rc == 0
    assert "demo" in _sources(cfg)

    # The loader every other entry point uses must agree.
    from lynx.config import load_config
    assert "demo" in load_config(cfg).sources


def test_backup_preserves_bytes(tmp_path, code_dir):
    """The .bak is the rollback copy: it has to come back byte-identical,
    BOM and all, not normalised by a decode/encode round-trip."""
    cfg = _write_config(tmp_path)
    original = "﻿" + cfg.read_text(encoding="utf-8")
    cfg.write_text(original, encoding="utf-8")
    raw_before = cfg.read_bytes()

    cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
              "codebase", "--path", str(code_dir), "--ext", ".py"])

    assert cfg.with_suffix(".json.bak").read_bytes() == raw_before


def test_no_tempfile_left_behind(tmp_path, code_dir):
    cfg = _write_config(tmp_path)
    cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
              "codebase", "--path", str(code_dir), "--ext", ".py"])
    assert not (tmp_path / "config.json.tmp").exists()


def test_malformed_config_reports_instead_of_crashing(tmp_path, code_dir):
    """A config that is valid JSON but not an object used to reach `.get`
    on a list and raise AttributeError at the user."""
    cfg = tmp_path / "config.json"
    cfg.write_text('["not", "a", "config"]', encoding="utf-8")

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg), "--yes"])
    assert rc == 1


# ----------------------------------------------------------------------
# --json output (the reason these commands exist: scripting)
# ----------------------------------------------------------------------


def test_add_json_output(tmp_path, code_dir, capsys):
    cfg = _write_config(tmp_path)
    rc = cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
                   "codebase", "--path", str(code_dir), "--json"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)  # stdout must be pure JSON
    assert out["ok"] is True
    assert out["name"] == "demo"
    assert out["defaults_applied"] is True
    # The detected extensions are in the payload, so scripts don't have to
    # scrape the informational line (which --json suppresses).
    assert set(out["block"]["supported_extensions"]) == {".py", ".md"}


def test_add_json_reports_the_block_that_was_written(tmp_path, code_dir, capsys):
    """The payload used to echo the input block, so a script reading
    `.block.ignored_path_fragments` was told there were none while the file
    on disk had the full default list."""
    cfg = _write_config(tmp_path)
    cli_main(["source", "add", "demo", "--config", str(cfg), "--type",
              "codebase", "--path", str(code_dir), "--ext", ".py", "--json"])

    out = json.loads(capsys.readouterr().out)
    assert out["block"] == _sources(cfg)["demo"]
    assert "/node_modules/" in out["block"]["ignored_path_fragments"]


def test_add_json_output_on_failure(tmp_path, capsys):
    cfg = _write_config(tmp_path)
    rc = cli_main(["source", "add", "demo", "--config", str(cfg),
                   "--type", "webdoc", "--json"])
    assert rc == 2

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "--url" in out["error"]


def test_remove_json_output(tmp_path, code_dir, capsys):
    cfg = _config_with_source(tmp_path, code_dir)
    (tmp_path / "storage" / "demo").mkdir(parents=True)

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg),
                   "--purge", "--yes", "--json"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["purged_path"].endswith("demo")


def test_remove_json_output_on_failed_purge(tmp_path, code_dir, unremovable, capsys):
    cfg = _config_with_source(tmp_path, code_dir)
    (tmp_path / "storage" / "demo").mkdir(parents=True)

    rc = cli_main(["source", "remove", "demo", "--config", str(cfg),
                   "--purge", "--yes", "--json"])
    assert rc == 1

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["purged_path"] is None


# ----------------------------------------------------------------------
# --build
# ----------------------------------------------------------------------


def test_build_uses_the_stored_name(tmp_path, code_dir, monkeypatch):
    """`add_source` stores the stripped name; passing the raw argv value to
    the build meant `lynx source add " demo " --build` reported success and
    then failed with "unknown source" on the next line."""
    import lynx.cli as cli
    seen = {}
    monkeypatch.setattr(cli, "_cmd_build",
                        lambda a: seen.update(source=a.source) or 0)

    rc = cli_main(["source", "add", "  demo  ", "--config",
                   str(_write_config(tmp_path)), "--type", "codebase",
                   "--path", str(code_dir), "--ext", ".py", "--build"])

    assert rc == 0
    assert seen["source"] == "demo"


def test_build_json_output_is_not_polluted_by_the_build(tmp_path, code_dir,
                                                        monkeypatch, capfd):
    """`lynx build` logs indexing progress to stdout. With --json that used
    to land in front of the object — and the build's own log goes through
    file descriptor 1, so only capfd sees it."""
    import os

    import lynx.cli as cli

    def _noisy_build(a):
        os.write(1, b"[rag] Indexed 3 file(s) -> 4 chunk(s).\n")
        return 0

    monkeypatch.setattr(cli, "_cmd_build", _noisy_build)

    rc = cli_main(["source", "add", "demo", "--config",
                   str(_write_config(tmp_path)), "--type", "codebase",
                   "--path", str(code_dir), "--ext", ".py", "--build",
                   "--json"])
    out, err = capfd.readouterr()

    assert rc == 0
    payload = json.loads(out)
    assert payload["built"] is True and payload["ok"] is True
    assert "Indexed" in err


def test_failed_build_is_reported_in_the_exit_code(tmp_path, code_dir, monkeypatch):
    """The source was added, but the command as a whole did not succeed."""
    import lynx.cli as cli
    monkeypatch.setattr(cli, "_cmd_build", lambda a: 1)

    rc = cli_main(["source", "add", "demo", "--config",
                   str(_write_config(tmp_path)), "--type", "codebase",
                   "--path", str(code_dir), "--ext", ".py", "--build"])
    assert rc == 1


def test_failed_build_json_says_why(tmp_path, code_dir, monkeypatch, capsys):
    """`{"ok": false}` with no `error` was the last mute failure in the
    family — and the one where the distinction matters most, since the
    source WAS added and only the indexing failed."""
    import lynx.cli as cli
    monkeypatch.setattr(cli, "_cmd_build", lambda a: 1)

    rc = cli_main(["source", "add", "demo", "--config",
                   str(_write_config(tmp_path)), "--type", "codebase",
                   "--path", str(code_dir), "--ext", ".py", "--build",
                   "--json"])

    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and out["added"] is True and out["built"] is False
    assert "lynx build --source demo" in out["error"]


# ----------------------------------------------------------------------
# Two writers, one config.
#
# Both front-ends read-modify-write the whole file. `lynx source add` in a
# terminal while the web UI is open on the sources page was enough for one
# write to drop the other, since each rewrites from the snapshot it read.
# ----------------------------------------------------------------------


def test_concurrent_adds_do_not_lose_each_other(tmp_path, code_dir):
    """Ten threads adding ten sources to one config: all ten must survive."""
    import threading

    cfg = _write_config(tmp_path)
    names = [f"src{i}" for i in range(10)]
    errors = []

    def _add(name):
        try:
            from lynx.manager.sources import add_source
            res = add_source(cfg, name, {
                "type": "codebase", "path": str(code_dir),
                "supported_extensions": [".py"],
            })
            if not res.ok:
                errors.append(res.message)
        except Exception as e:  # pragma: no cover - only on a real failure
            errors.append(repr(e))

    threads = [threading.Thread(target=_add, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(_sources(cfg)) == sorted(names)


def test_lock_is_released_even_when_the_write_fails(tmp_path, code_dir):
    """A failed mutation must not leave a lockfile behind — the next
    command would then wait out the full stale timeout for nothing."""
    from lynx.manager.sources import add_source

    cfg = _write_config(tmp_path)
    lock = cfg.with_name(cfg.name + ".lock")

    assert add_source(cfg, "9bad", {"type": "codebase"}).ok is False
    assert not lock.exists()

    assert add_source(cfg, "good", {
        "type": "codebase", "path": str(code_dir),
        "supported_extensions": [".py"],
    }).ok is True
    assert not lock.exists()


def test_a_stale_lock_is_broken_rather_than_waited_out(tmp_path, code_dir,
                                                       monkeypatch):
    """A lock left by a killed process must not brick config edits."""
    import lynx.manager.sources as sources_mod

    cfg = _write_config(tmp_path)
    lock = cfg.with_name(cfg.name + ".lock")
    lock.write_text("99999", encoding="utf-8")  # owner long gone
    monkeypatch.setattr(sources_mod, "_LOCK_STALE_SECONDS", 0.0)

    res = sources_mod.add_source(cfg, "demo", {
        "type": "codebase", "path": str(code_dir),
        "supported_extensions": [".py"],
    })

    assert res.ok is True
    assert not lock.exists()


def test_a_held_lock_does_not_block_forever(tmp_path, code_dir, monkeypatch):
    """Proceeding unlocked beats refusing to work: a stuck lock that is not
    yet stale must not make the command hang until the user kills it."""
    import lynx.manager.sources as sources_mod

    cfg = _write_config(tmp_path)
    lock = cfg.with_name(cfg.name + ".lock")
    lock.write_text("12345", encoding="utf-8")
    monkeypatch.setattr(sources_mod, "_LOCK_STALE_SECONDS", 10_000.0)
    monkeypatch.setattr(sources_mod, "_LOCK_TIMEOUT_SECONDS", 0.2)

    res = sources_mod.add_source(cfg, "demo", {
        "type": "codebase", "path": str(code_dir),
        "supported_extensions": [".py"],
    })

    assert res.ok is True
    assert "demo" in _sources(cfg)
    lock.unlink()  # still someone else's; we never took it


def test_build_is_not_run_when_not_asked(tmp_path, code_dir, monkeypatch):
    import lynx.cli as cli
    called = []
    monkeypatch.setattr(cli, "_cmd_build", lambda a: called.append(a) or 0)

    cli_main(["source", "add", "demo", "--config", str(_write_config(tmp_path)),
              "--type", "codebase", "--path", str(code_dir), "--ext", ".py"])
    assert called == []


# ----------------------------------------------------------------------
# The premise of extracting manager/sources.py: UI and CLI agree.
# Asserted end-to-end, because "they call the same function" is exactly the
# kind of claim that stays true in the docstring and stops being true in
# the code.
# ----------------------------------------------------------------------


def test_ui_and_cli_write_the_same_config(tmp_path, code_dir):
    from fastapi.testclient import TestClient
    from lynx.manager.ui.app import create_app

    via_cli = _write_config(tmp_path / "a")
    via_ui = _write_config(tmp_path / "b")
    block = {"type": "codebase", "path": str(code_dir),
             "supported_extensions": [".py"]}

    assert cli_main(["source", "add", "demo", "--config", str(via_cli),
                     "--block", json.dumps(block)]) == 0

    client = TestClient(create_app(via_ui))
    assert client.post("/api/sources",
                       json={"name": "demo", "block": block}).status_code == 200

    assert _sources(via_cli) == _sources(via_ui)


def test_ui_and_cli_reject_the_same_names(tmp_path, code_dir):
    from fastapi.testclient import TestClient
    from lynx.manager.ui.app import create_app

    via_cli = _write_config(tmp_path / "a")
    via_ui = _write_config(tmp_path / "b")
    block = {"type": "codebase", "path": str(code_dir),
             "supported_extensions": [".py"]}

    client = TestClient(create_app(via_ui))
    for bad in ("9lives", "has-dash", "", "x" * 41):
        cli_rc = cli_main(["source", "add", bad, "--config", str(via_cli),
                           "--block", json.dumps(block)])
        ui_status = client.post(
            "/api/sources", json={"name": bad, "block": block}).status_code
        assert cli_rc != 0 and ui_status >= 400, f"disagreed on {bad!r}"
    assert _sources(via_cli) == {} and _sources(via_ui) == {}
