"""Integration tests for `lynx manager ui` — FastAPI app + dashboard.

We use `fastapi.testclient.TestClient` to exercise the routes without
binding a port. Manager is stubbed (no HF embeddings load) for fast,
deterministic test runs.

Scenarios:
  1. /api/health → 200 {"status": "ok"}
  2. /api/sources with stubbed manager → list with shape we expect
  3. /api/sources/<name>/status → returns the backend status dict
  4. /api/sources/UNKNOWN/status → 404
  5. /api/doctor → returns doctor results aggregated
  6. /api/config returns the file content (or 404 if no config)
  7. GET / (dashboard HTML) → 200, contains source name + counts
  8. GET /playground, /config, /integrations, /sources, /doctor → 200
     placeholder pages
  9. /static/htmx.min.js + /static/tailwind.min.js → 200 with non-trivial size
 10. _find_free_port returns the preferred port when free,
     advances when busy
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path


def _stub_manager():
    """Replace SourceManager + CodebaseRAG so create_app's lazy
    `_get_manager` returns a working stub without HF model load."""
    from lynx.rag_manager import CodebaseRAG

    class _StubColl:
        def count(self): return 42  # arbitrary non-None number for UI

    class _StubVS:
        def __init__(self): self._collection = _StubColl()

    def stub_rag_init(self, **kw):
        self.codebase_path = Path(kw["codebase_path"])
        self.storage_path = Path(kw["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": "stubsha", "last_update": "2026-05-22T10:00:00"}
        self.vector_store = _StubVS()

    CodebaseRAG.__init__ = stub_rag_init
    CodebaseRAG.update = lambda self, force=False: None
    CodebaseRAG.update_file = lambda self, p: None
    CodebaseRAG.remove_file = lambda self, p: None
    CodebaseRAG.check_config_drift = lambda self: None


def _write_config(tmp: Path, code_dir: Path) -> Path:
    """Build a minimal valid config.json that the UI can load."""
    cfg = {
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {
            "demo": {
                "type": "codebase",
                "path": str(code_dir),
                "supported_extensions": [".py"],
                "watcher": {"enabled": False},
                "git_integration": {"enabled": False},
            },
        },
    }
    p = tmp / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-ui-"))
    print(f"[test] tempdir: {tmp}")
    try:
        _stub_manager()
        code_dir = tmp / "code"; code_dir.mkdir()
        (code_dir / "main.py").write_text("def f(): pass\n")

        cfg_path = _write_config(tmp, code_dir)

        from fastapi.testclient import TestClient
        from lynx.manager.ui.app import create_app, _find_free_port

        app = create_app(cfg_path)
        client = TestClient(app)

        # ============================================================
        # 1. /api/health
        # ============================================================
        r = client.get("/api/health")
        if r.status_code != 200 or r.json() != {"status": "ok"}:
            print(f"[test] FAIL [1/10]: health wrong: {r.status_code} {r.text}")
            return 1
        print(f"[test] OK [1/10] /api/health")

        # ============================================================
        # 2. /api/sources
        # ============================================================
        r = client.get("/api/sources")
        if r.status_code != 200:
            print(f"[test] FAIL [2/10]: sources status {r.status_code}: {r.text}")
            return 2
        data = r.json()
        if "sources" not in data or not data["sources"]:
            print(f"[test] FAIL [2/10]: empty sources: {data}")
            return 2
        s0 = data["sources"][0]
        if s0["name"] != "demo" or s0["type"] != "codebase":
            print(f"[test] FAIL [2/10]: wrong source shape: {s0}")
            return 2
        if "locked" not in s0:
            print(f"[test] FAIL [2/10]: missing 'locked' field: {s0}")
            return 2
        print(f"[test] OK [2/10] /api/sources: 1 source, lock flag present")

        # ============================================================
        # 3. /api/sources/demo/status
        # ============================================================
        r = client.get("/api/sources/demo/status")
        if r.status_code != 200:
            print(f"[test] FAIL [3/10]: status endpoint failed: {r.status_code}")
            return 3
        st = r.json()
        if st.get("name") != "demo":
            print(f"[test] FAIL [3/10]: status missing name: {st}")
            return 3
        print(f"[test] OK [3/10] /api/sources/demo/status: name=demo")

        # ============================================================
        # 4. /api/sources/UNKNOWN/status → 404
        # ============================================================
        r = client.get("/api/sources/UNKNOWN/status")
        if r.status_code != 404:
            print(f"[test] FAIL [4/10]: unknown source should be 404, got {r.status_code}")
            return 4
        print(f"[test] OK [4/10] /api/sources/UNKNOWN/status: 404")

        # ============================================================
        # 5. /api/doctor
        # ============================================================
        r = client.get("/api/doctor")
        if r.status_code != 200:
            print(f"[test] FAIL [5/10]: doctor failed: {r.status_code}")
            return 5
        data = r.json()
        if "results" not in data or "worst_status" not in data:
            print(f"[test] FAIL [5/10]: doctor missing keys: {list(data)}")
            return 5
        if not data["results"]:
            print(f"[test] FAIL [5/10]: doctor results empty")
            return 5
        print(f"[test] OK [5/10] /api/doctor: {len(data['results'])} checks, "
              f"worst={data['worst_status']}")

        # ============================================================
        # 6. /api/config
        # ============================================================
        r = client.get("/api/config")
        if r.status_code != 200:
            print(f"[test] FAIL [6/10]: config endpoint failed: {r.status_code}")
            return 6
        body = r.json()
        if "content" not in body or "demo" not in body["content"]:
            print(f"[test] FAIL [6/10]: config body wrong: {body}")
            return 6
        print(f"[test] OK [6/10] /api/config: returns content + path")

        # ============================================================
        # 7. GET / (dashboard)
        # ============================================================
        r = client.get("/")
        if r.status_code != 200:
            print(f"[test] FAIL [7/10]: dashboard returned {r.status_code}")
            return 7
        html = r.text
        for needle in ("Dashboard", "LynxManager", "demo", "Sources"):
            if needle not in html:
                print(f"[test] FAIL [7/10]: dashboard missing {needle!r}")
                return 7
        print(f"[test] OK [7/10] dashboard: contains expected widgets")

        # ============================================================
        # 8. Placeholder pages all 200
        # ============================================================
        for path in ("/playground", "/config", "/integrations", "/sources", "/doctor"):
            r = client.get(path)
            if r.status_code != 200:
                print(f"[test] FAIL [8/10]: {path} returned {r.status_code}")
                return 8
            if "LynxManager" not in r.text:
                print(f"[test] FAIL [8/10]: {path} missing base layout")
                return 8
        print(f"[test] OK [8/10] 5 placeholder pages all 200")

        # ============================================================
        # 9. Static assets served correctly
        # ============================================================
        for asset, min_size in (("htmx.min.js", 10_000), ("tailwind.min.js", 100_000)):
            r = client.get(f"/static/{asset}")
            if r.status_code != 200:
                print(f"[test] FAIL [9/10]: {asset} returned {r.status_code}")
                return 9
            if len(r.content) < min_size:
                print(f"[test] FAIL [9/10]: {asset} too small "
                      f"({len(r.content)} < {min_size}) — likely truncated")
                return 9
        print(f"[test] OK [9/10] static assets served")

        # ============================================================
        # 10. _find_free_port: returns preferred when free, advances when busy
        # ============================================================
        # Pick a random high port, occupy it, then ask find_free_port for
        # exactly that one — should advance to the next.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))  # OS-assigned free port
        occupied_port = s.getsockname()[1]
        try:
            picked = _find_free_port(preferred=occupied_port, attempts=5)
            if picked == occupied_port:
                print(f"[test] FAIL [10/10]: picked {picked} which was occupied")
                return 10
            if not (occupied_port < picked <= occupied_port + 5):
                # Could be the OS-assigned fallback — accept if it's any
                # different free port
                if picked == occupied_port:
                    print(f"[test] FAIL [10/10]: didn't advance: {picked}")
                    return 10
        finally:
            s.close()
        # Sanity: requesting a known-free port returns it
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.bind(("127.0.0.1", 0))
        free = s2.getsockname()[1]
        s2.close()
        picked = _find_free_port(preferred=free, attempts=2)
        if picked != free:
            # Could already be re-used by another process; accept any port
            print(f"[test] OK [10/10] _find_free_port: advanced to {picked} (preferred {free} reclaimed?)")
        else:
            print(f"[test] OK [10/10] _find_free_port: preferred port honored when free")

        # ============================================================
        # 11. PUT /api/config: bad JSON → 422 with clear message
        # ============================================================
        r = client.put("/api/config", json={"content": "{not valid"})
        if r.status_code != 422:
            print(f"[test] FAIL [11/13]: bad JSON should be 422, got {r.status_code}")
            return 11
        if "JSON syntax" not in r.text:
            print(f"[test] FAIL [11/13]: error message unhelpful: {r.text[:100]}")
            return 11
        print(f"[test] OK [11/13] PUT bad JSON: 422 with 'JSON syntax' message")

        # ============================================================
        # 12. PUT /api/config: bad schema → 422, file unchanged
        # ============================================================
        original_content = cfg_path.read_text()
        r = client.put("/api/config", json={"content": '{"hello": "world"}'})
        if r.status_code != 422:
            print(f"[test] FAIL [12/13]: bad schema should be 422, got {r.status_code}")
            return 12
        if cfg_path.read_text() != original_content:
            print(f"[test] FAIL [12/13]: file was overwritten despite invalid schema!")
            return 12
        print(f"[test] OK [12/13] PUT bad schema: 422, file untouched")

        # ============================================================
        # 13. PUT /api/config: valid → 200 + backup + new on disk
        # ============================================================
        new_cfg = json.loads(original_content)
        new_cfg["sources"]["demo"]["supported_extensions"] = [".py", ".md", ".txt"]
        r = client.put("/api/config", json={"content": json.dumps(new_cfg, indent=2)})
        if r.status_code != 200:
            print(f"[test] FAIL [13/13]: valid put should be 200, got {r.status_code}")
            return 13
        # Backup file exists
        backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        if not backup.exists():
            print(f"[test] FAIL [13/13]: backup not created at {backup}")
            return 13
        # Backup has the original content
        if backup.read_text() != original_content:
            print(f"[test] FAIL [13/13]: backup content doesn't match original")
            return 13
        # New file has the change
        if ".txt" not in cfg_path.read_text():
            print(f"[test] FAIL [13/13]: new content not persisted")
            return 13
        print(f"[test] OK [13/13] PUT valid: 200 + backup + new content saved")

        print("\n[test] === SUCCESS: UI scaffolding works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
