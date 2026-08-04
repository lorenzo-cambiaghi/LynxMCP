"""Manager lifecycle in the web UI: `_dispose_manager` and the two routes
that depend on it.

Why this file exists. The UI process holds a source's ChromaDB files open
through the cached SourceManager, so `DELETE /api/sources/{name}?purge=true`
used to fail against the UI's *own* lock (WinError 32 on `data_level0.bin`)
and advise the user to stop whatever was holding the index — the page they
were clicking in. Measured: dropping every Python reference is NOT enough,
because chromadb caches the client system per storage path; clearing that
cache is what actually closes the handles.

So `_dispose_manager` does two things, and only the second one fixes the
bug:

    app.state.manager = None            # necessary, not sufficient
    SharedSystemClient.clear_system_cache()   # this is the fix

The first version of these tests asserted only that the manager field went
to None — which the pre-fix code already did. They would have passed
against the broken behaviour. Everything here pins the cache clear.

Covered:
  1. _dispose_manager clears the chromadb system cache, not just the field
  2. ... and clears manager_error, so a previously-failed manager retries
  3. ... and survives a chromadb API change instead of 500-ing the route
  4. ... and works when chromadb exposes no SharedSystemClient at all
  5. DELETE ?purge=true releases the handles before deleting
  6. DELETE without purge does NOT pay the cache-clear cost
  7. /api/manager/reload releases them too (a reload that keeps the old
     locks alive defeats its own purpose) and redirects
  8. /api/manager/reload lets a manager that failed to build be retried
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_config(tmp_path: Path, sources: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "config_version": 2,
        "storage_path": "./storage",
        "sources": sources or {},
    }, indent=2) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def app(tmp_path):
    """A UI app with a config it can read. The manager is never really
    built — these tests are about disposing it, not constructing it."""
    from lynx.manager.ui.app import create_app
    return create_app(_write_config(tmp_path))


@pytest.fixture
def cache_clears(monkeypatch):
    """Count calls to chromadb's system-cache clear — the operation that
    actually releases the file handles."""
    from chromadb.api.shared_system_client import SharedSystemClient

    calls = []
    monkeypatch.setattr(SharedSystemClient, "clear_system_cache",
                        classmethod(lambda cls: calls.append(1)))
    return calls


# ----------------------------------------------------------------------
# _dispose_manager
# ----------------------------------------------------------------------


def test_dispose_clears_the_chromadb_system_cache(app, cache_clears):
    """The assertion the previous test was missing: nulling the field alone
    leaves the index locked, so the cache clear is the behaviour to pin."""
    from lynx.manager.ui.app import _dispose_manager

    app.state.manager = object()
    _dispose_manager(app)

    assert app.state.manager is None
    assert cache_clears == [1], "handles were never released"


def test_dispose_clears_a_previous_manager_error(app, cache_clears):
    """`_get_manager` refuses to retry while manager_error is set, so a
    dispose that left it behind would freeze the UI on a stale failure."""
    from lynx.manager.ui.app import _dispose_manager

    app.state.manager = None
    app.state.manager_error = "ImportError: something transient"
    _dispose_manager(app)

    assert app.state.manager_error is None


def test_dispose_survives_a_chromadb_api_change(app, monkeypatch):
    """A failed cache clear must degrade to the pre-fix behaviour (a 409 the
    user can act on), never to a 500 on a delete."""
    from chromadb.api.shared_system_client import SharedSystemClient
    from lynx.manager.ui.app import _dispose_manager

    def _boom(cls):
        raise RuntimeError("clear_system_cache removed in a future chromadb")

    monkeypatch.setattr(SharedSystemClient, "clear_system_cache",
                        classmethod(_boom))
    app.state.manager = object()

    _dispose_manager(app)  # must not raise

    assert app.state.manager is None


def test_dispose_survives_a_missing_shared_system_client(app, monkeypatch):
    """Both module paths gone (a chromadb major bump): still no exception."""
    import builtins

    from lynx.manager.ui.app import _dispose_manager

    real_import = builtins.__import__

    def _no_chromadb(name, *a, **kw):
        if name.startswith("chromadb"):
            raise ImportError("no chromadb here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_chromadb)
    app.state.manager = object()

    _dispose_manager(app)

    assert app.state.manager is None


# ----------------------------------------------------------------------
# The routes that depend on it
# ----------------------------------------------------------------------


def _client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_purge_route_releases_the_handles(tmp_path, monkeypatch, cache_clears):
    """The regression test for the original bug: the UI must let go of the
    index before trying to delete it, or it 409s against itself."""
    import lynx.manager.ui.routes as routes_mod
    from lynx.manager.sources import CrudResult
    from lynx.manager.ui.app import create_app

    cfg = _write_config(tmp_path, {
        "demo": {"type": "codebase", "path": str(tmp_path),
                 "supported_extensions": [".py"]},
    })
    app = create_app(cfg)
    app.state.manager = object()

    seen = {}

    def _spy_remove(config_path, name, purge=False):
        # Snapshot the world as the rmtree would find it.
        seen["clears_before_delete"] = len(cache_clears)
        return CrudResult(True, f"Source {name!r} removed.", 200,
                          purged_path=str(tmp_path / "storage" / name))

    monkeypatch.setattr(routes_mod, "_remove_source", _spy_remove)

    r = _client(app).delete("/api/sources/demo?purge=true")

    assert r.status_code == 200
    assert seen["clears_before_delete"] == 1, (
        "the delete ran while this process still held the ChromaDB handles"
    )


def test_plain_delete_does_not_clear_the_cache(tmp_path, monkeypatch,
                                               cache_clears):
    """No rmtree, nothing to unblock: don't throw away a working manager
    (and pay a full GC) for a config-only edit."""
    import lynx.manager.ui.routes as routes_mod
    from lynx.manager.sources import CrudResult
    from lynx.manager.ui.app import create_app

    cfg = _write_config(tmp_path, {
        "demo": {"type": "codebase", "path": str(tmp_path),
                 "supported_extensions": [".py"]},
    })
    app = create_app(cfg)
    app.state.manager = object()

    monkeypatch.setattr(
        routes_mod, "_remove_source",
        lambda config_path, name, purge=False:
            CrudResult(True, f"Source {name!r} removed.", 200),
    )

    r = _client(app).delete("/api/sources/demo")

    assert r.status_code == 200
    assert cache_clears == []


def test_reload_releases_the_handles_and_redirects(app, cache_clears):
    """`/api/manager/reload` is the button offered when the index looks
    held. Nulling the reference while keeping chromadb's cached client —
    and its open files — would make the button a no-op for the one thing
    it is for."""
    app.state.manager = object()

    r = _client(app).post("/api/manager/reload")

    assert r.status_code == 200
    assert r.headers["HX-Redirect"] == "/"
    assert app.state.manager is None
    assert cache_clears == [1]


def test_reload_lets_a_failed_manager_be_retried(app, cache_clears):
    """_get_manager short-circuits on a stored error, so reload has to
    clear it — otherwise the UI stays broken until a restart."""
    app.state.manager = None
    app.state.manager_error = "RuntimeError: index was locked a minute ago"

    r = _client(app).post("/api/manager/reload")

    assert r.status_code == 200
    assert app.state.manager_error is None
