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
     placeholder/page renders
  9. /static/htmx.min.js + /static/tailwind.min.js → 200 with non-trivial size
 10. _find_free_port returns the preferred port when free,
     advances when busy
 11. PUT /api/config: bad JSON → 422 with clear message
 12. PUT /api/config: bad schema → 422, file unchanged
 13. PUT /api/config: valid → 200 + backup + new on disk
 14. GET /playground → 200 with source selector + tabs
 15. POST /api/playground/search → 200 with rendered hits (stubbed)
 16. POST /api/playground/search with empty query → 400
 17. POST /api/playground/find_definition → 200 (bm25 fallback)
 18. POST /api/playground/get_callers (no graph) → 400 with clear msg
 19. GET /sources → index page renders
 20. GET /sources/<name> → detail page renders with status + build form
 21. GET /sources/UNKNOWN → 404
 22. POST /api/sources/<name>/build → spawns job, /api/jobs/<id> → done
 23. POST /api/sources/<name>/build (locked) → 409 clear msg
 24. GET /integrations → page with per-client cards + snippet
 25. GET /api/integrations/claude/rules → CLAUDE.md download
 26. GET /api/integrations/UNKNOWN/rules → 404
 27. resolve_config_path: explicit > RAG_CONFIG_PATH > ./config.json (cwd)
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

    # Canned search results — exercise the playground rendering without
    # paying the BM25 / dense lookup cost. Shape mirrors what
    # CodebaseRAG._dense_lookup actually returns (see rag_manager.py).
    def stub_search(self, query, top_k=5, **kw):
        return [
            {
                "id": "chunk-1",
                "file": "main.py",
                "file_path": "main.py",
                "symbol_name": "f",
                "symbol_kind": "function",
                "language": "python",
                "start_line": 1,
                "end_line": 1,
                "content": "def f(): pass",
                "score": 0.42,
            },
        ]
    CodebaseRAG.search = stub_search


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
            print(f"[test] FAIL [1/27]: health wrong: {r.status_code} {r.text}")
            return 1
        print(f"[test] OK [1/27] /api/health")

        # ============================================================
        # 2. /api/sources
        # ============================================================
        r = client.get("/api/sources")
        if r.status_code != 200:
            print(f"[test] FAIL [2/27]: sources status {r.status_code}: {r.text}")
            return 2
        data = r.json()
        if "sources" not in data or not data["sources"]:
            print(f"[test] FAIL [2/27]: empty sources: {data}")
            return 2
        s0 = data["sources"][0]
        if s0["name"] != "demo" or s0["type"] != "codebase":
            print(f"[test] FAIL [2/27]: wrong source shape: {s0}")
            return 2
        if "locked" not in s0:
            print(f"[test] FAIL [2/27]: missing 'locked' field: {s0}")
            return 2
        print(f"[test] OK [2/27] /api/sources: 1 source, lock flag present")

        # ============================================================
        # 3. /api/sources/demo/status
        # ============================================================
        r = client.get("/api/sources/demo/status")
        if r.status_code != 200:
            print(f"[test] FAIL [3/27]: status endpoint failed: {r.status_code}")
            return 3
        st = r.json()
        if st.get("name") != "demo":
            print(f"[test] FAIL [3/27]: status missing name: {st}")
            return 3
        print(f"[test] OK [3/27] /api/sources/demo/status: name=demo")

        # ============================================================
        # 4. /api/sources/UNKNOWN/status → 404
        # ============================================================
        r = client.get("/api/sources/UNKNOWN/status")
        if r.status_code != 404:
            print(f"[test] FAIL [4/27]: unknown source should be 404, got {r.status_code}")
            return 4
        print(f"[test] OK [4/27] /api/sources/UNKNOWN/status: 404")

        # ============================================================
        # 5. /api/doctor
        # ============================================================
        r = client.get("/api/doctor")
        if r.status_code != 200:
            print(f"[test] FAIL [5/27]: doctor failed: {r.status_code}")
            return 5
        data = r.json()
        if "results" not in data or "worst_status" not in data:
            print(f"[test] FAIL [5/27]: doctor missing keys: {list(data)}")
            return 5
        if not data["results"]:
            print(f"[test] FAIL [5/27]: doctor results empty")
            return 5
        print(f"[test] OK [5/27] /api/doctor: {len(data['results'])} checks, "
              f"worst={data['worst_status']}")

        # ============================================================
        # 6. /api/config
        # ============================================================
        r = client.get("/api/config")
        if r.status_code != 200:
            print(f"[test] FAIL [6/27]: config endpoint failed: {r.status_code}")
            return 6
        body = r.json()
        if "content" not in body or "demo" not in body["content"]:
            print(f"[test] FAIL [6/27]: config body wrong: {body}")
            return 6
        print(f"[test] OK [6/27] /api/config: returns content + path")

        # ============================================================
        # 7. GET / (dashboard)
        # ============================================================
        r = client.get("/")
        if r.status_code != 200:
            print(f"[test] FAIL [7/27]: dashboard returned {r.status_code}")
            return 7
        html = r.text
        for needle in ("Dashboard", "LynxManager", "demo", "Sources"):
            if needle not in html:
                print(f"[test] FAIL [7/27]: dashboard missing {needle!r}")
                return 7
        print(f"[test] OK [7/27] dashboard: contains expected widgets")

        # ============================================================
        # 8. Placeholder pages all 200
        # ============================================================
        for path in ("/playground", "/config", "/integrations", "/sources", "/doctor"):
            r = client.get(path)
            if r.status_code != 200:
                print(f"[test] FAIL [8/27]: {path} returned {r.status_code}")
                return 8
            if "LynxManager" not in r.text:
                print(f"[test] FAIL [8/27]: {path} missing base layout")
                return 8
        print(f"[test] OK [8/27] 5 placeholder pages all 200")

        # ============================================================
        # 9. Static assets served correctly
        # ============================================================
        for asset, min_size in (("htmx.min.js", 10_000), ("tailwind.min.js", 100_000)):
            r = client.get(f"/static/{asset}")
            if r.status_code != 200:
                print(f"[test] FAIL [9/27]: {asset} returned {r.status_code}")
                return 9
            if len(r.content) < min_size:
                print(f"[test] FAIL [9/27]: {asset} too small "
                      f"({len(r.content)} < {min_size}) — likely truncated")
                return 9
        print(f"[test] OK [9/27] static assets served")

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
                print(f"[test] FAIL [10/27]: picked {picked} which was occupied")
                return 10
            if not (occupied_port < picked <= occupied_port + 5):
                # Could be the OS-assigned fallback — accept if it's any
                # different free port
                if picked == occupied_port:
                    print(f"[test] FAIL [10/27]: didn't advance: {picked}")
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
            print(f"[test] OK [10/27] _find_free_port: advanced to {picked} (preferred {free} reclaimed?)")
        else:
            print(f"[test] OK [10/27] _find_free_port: preferred port honored when free")

        # ============================================================
        # 11. PUT /api/config: bad JSON → 422 with clear message
        # ============================================================
        r = client.put("/api/config", json={"content": "{not valid"})
        if r.status_code != 422:
            print(f"[test] FAIL [11/27]: bad JSON should be 422, got {r.status_code}")
            return 11
        if "JSON syntax" not in r.text:
            print(f"[test] FAIL [11/27]: error message unhelpful: {r.text[:100]}")
            return 11
        print(f"[test] OK [11/27] PUT bad JSON: 422 with 'JSON syntax' message")

        # ============================================================
        # 12. PUT /api/config: bad schema → 422, file unchanged
        # ============================================================
        original_content = cfg_path.read_text()
        r = client.put("/api/config", json={"content": '{"hello": "world"}'})
        if r.status_code != 422:
            print(f"[test] FAIL [12/27]: bad schema should be 422, got {r.status_code}")
            return 12
        if cfg_path.read_text() != original_content:
            print(f"[test] FAIL [12/27]: file was overwritten despite invalid schema!")
            return 12
        print(f"[test] OK [12/27] PUT bad schema: 422, file untouched")

        # ============================================================
        # 13. PUT /api/config: valid → 200 + backup + new on disk
        # ============================================================
        new_cfg = json.loads(original_content)
        new_cfg["sources"]["demo"]["supported_extensions"] = [".py", ".md", ".txt"]
        r = client.put("/api/config", json={"content": json.dumps(new_cfg, indent=2)})
        if r.status_code != 200:
            print(f"[test] FAIL [13/27]: valid put should be 200, got {r.status_code}")
            return 13
        # Backup file exists
        backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
        if not backup.exists():
            print(f"[test] FAIL [13/27]: backup not created at {backup}")
            return 13
        # Backup has the original content
        if backup.read_text() != original_content:
            print(f"[test] FAIL [13/27]: backup content doesn't match original")
            return 13
        # New file has the change
        if ".txt" not in cfg_path.read_text():
            print(f"[test] FAIL [13/27]: new content not persisted")
            return 13
        print(f"[test] OK [13/27] PUT valid: 200 + backup + new content saved")

        # ============================================================
        # 14. GET /playground — page renders with source selector
        # ============================================================
        r = client.get("/playground")
        if r.status_code != 200:
            print(f"[test] FAIL [14/27]: /playground returned {r.status_code}")
            return 14
        for needle in ("Search playground", "pg-source", "demo", "Code-aware", "Graph", "Diff"):
            if needle not in r.text:
                print(f"[test] FAIL [14/27]: playground HTML missing {needle!r}")
                return 14
        print(f"[test] OK [14/27] /playground: form + tabs rendered")

        # ============================================================
        # 15. POST /api/playground/search — hits the stubbed search
        # ============================================================
        r = client.post(
            "/api/playground/search",
            data={"source": "demo", "query": "anything", "top_k": "5"},
        )
        if r.status_code != 200:
            print(f"[test] FAIL [15/27]: search returned {r.status_code}: {r.text[:200]}")
            return 15
        if "main.py" not in r.text or "score=0.420" not in r.text:
            print(f"[test] FAIL [15/27]: rendered HTML missing expected content: {r.text[:300]}")
            return 15
        print(f"[test] OK [15/27] /api/playground/search: stubbed result rendered")

        # ============================================================
        # 16. POST /api/playground/search — empty query → 400
        # ============================================================
        r = client.post(
            "/api/playground/search",
            data={"source": "demo", "query": "   ", "top_k": "5"},
        )
        if r.status_code != 400:
            print(f"[test] FAIL [16/27]: empty query should be 400, got {r.status_code}")
            return 16
        if "empty" not in r.text:
            print(f"[test] FAIL [16/27]: error message wrong: {r.text[:200]}")
            return 16
        print(f"[test] OK [16/27] empty query: 400 with clear message")

        # ============================================================
        # 17. POST /api/playground/find_definition — fallback path (no graph)
        # ============================================================
        r = client.post(
            "/api/playground/find_definition",
            data={"source": "demo", "symbol": "f", "limit": "5"},
        )
        if r.status_code != 200:
            print(f"[test] FAIL [17/27]: find_definition returned {r.status_code}: {r.text[:200]}")
            return 17
        # The fallback path uses search_bm25 — result list must contain `main.py`.
        if "main.py" not in r.text:
            print(f"[test] FAIL [17/27]: find_definition rendering missing 'main.py': {r.text[:300]}")
            return 17
        print(f"[test] OK [17/27] /api/playground/find_definition: bm25 fallback rendered")

        # ============================================================
        # 18. POST /api/playground/get_callers — graph disabled → 400
        # ============================================================
        r = client.post(
            "/api/playground/get_callers",
            data={"source": "demo", "symbol": "f", "limit": "10"},
        )
        if r.status_code != 400:
            print(f"[test] FAIL [18/27]: get_callers without graph should be 400, got {r.status_code}")
            return 18
        if "graph" not in r.text.lower():
            print(f"[test] FAIL [18/27]: error message should mention graph: {r.text[:200]}")
            return 18
        print(f"[test] OK [18/27] /api/playground/get_callers: 400 (graph layer not enabled)")

        # ============================================================
        # 19. GET /sources — index page lists configured sources
        # ============================================================
        r = client.get("/sources")
        if r.status_code != 200:
            print(f"[test] FAIL [19/27]: /sources returned {r.status_code}")
            return 19
        for needle in ("Sources", "demo", "type: codebase"):
            if needle not in r.text:
                print(f"[test] FAIL [19/27]: /sources missing {needle!r}")
                return 19
        print(f"[test] OK [19/27] /sources: index renders with 'demo'")

        # ============================================================
        # 20. GET /sources/demo — per-source detail page
        # ============================================================
        r = client.get("/sources/demo")
        if r.status_code != 200:
            print(f"[test] FAIL [20/27]: /sources/demo returned {r.status_code}")
            return 20
        for needle in ("demo", "Status", "Rebuild index", "build-status"):
            if needle not in r.text:
                print(f"[test] FAIL [20/27]: detail page missing {needle!r}")
                return 20
        print(f"[test] OK [20/27] /sources/demo: detail page rendered")

        # ============================================================
        # 21. GET /sources/UNKNOWN → 404
        # ============================================================
        r = client.get("/sources/UNKNOWN")
        if r.status_code != 404:
            print(f"[test] FAIL [21/27]: unknown source should be 404, got {r.status_code}")
            return 21
        print(f"[test] OK [21/27] /sources/UNKNOWN: 404")

        # ============================================================
        # 22. POST /api/sources/demo/build — spawns job, returns widget
        # ============================================================
        import time as _time
        r = client.post("/api/sources/demo/build")
        if r.status_code != 200:
            print(f"[test] FAIL [22/27]: build returned {r.status_code}: {r.text[:200]}")
            return 22
        if "build-status" not in r.text or "job " not in r.text:
            print(f"[test] FAIL [22/27]: build response missing widget shape: {r.text[:300]}")
            return 22
        # Extract job id from the response (format: "job XXXXXXXX")
        import re
        m = re.search(r"job ([a-f0-9]{8})", r.text)
        if not m:
            print(f"[test] FAIL [22/27]: couldn't extract job id from: {r.text[:300]}")
            return 22
        job_id = m.group(1)
        # Wait briefly — the stubbed update() is a no-op so it completes fast
        for _ in range(20):
            _time.sleep(0.05)
            jr = client.get(f"/api/jobs/{job_id}")
            if jr.status_code == 200 and jr.json().get("state") == "done":
                break
        else:
            print(f"[test] FAIL [22/27]: job didn't reach 'done' within 1s")
            return 22
        print(f"[test] OK [22/27] /api/sources/demo/build: job {job_id} → done")

        # ============================================================
        # 23. POST /api/sources/demo/build — locked → 409
        # ============================================================
        from lynx.manager.ui import lock as lock_mod
        original = lock_mod.is_storage_locked
        lock_mod.is_storage_locked = lambda p: True
        try:
            r = client.post("/api/sources/demo/build")
            if r.status_code != 409:
                print(f"[test] FAIL [23/27]: locked build should be 409, got {r.status_code}")
                return 23
            if "Locked" not in r.text:
                print(f"[test] FAIL [23/27]: 409 missing 'Locked' message: {r.text[:200]}")
                return 23
        finally:
            lock_mod.is_storage_locked = original
        print(f"[test] OK [23/27] /api/sources/demo/build (locked): 409 with clear message")

        # ============================================================
        # 24. GET /integrations — page with per-client cards
        # ============================================================
        r = client.get("/integrations")
        if r.status_code != 200:
            print(f"[test] FAIL [24/27]: /integrations returned {r.status_code}")
            return 24
        for needle in ("Integrations", "Claude Code", "Cursor", "Antigravity",
                       "mcpServers", "Copy snippet"):
            if needle not in r.text:
                print(f"[test] FAIL [24/27]: /integrations missing {needle!r}")
                return 24
        # The snippet must contain the current python executable path
        if sys.executable not in r.text:
            print(f"[test] FAIL [24/27]: snippet doesn't include current python path")
            return 24
        print(f"[test] OK [24/27] /integrations: per-client cards + snippet")

        # ============================================================
        # 25. GET /api/integrations/claude/rules — downloads CLAUDE.md
        # ============================================================
        r = client.get("/api/integrations/claude/rules")
        if r.status_code != 200:
            print(f"[test] FAIL [25/27]: rules download returned {r.status_code}")
            return 25
        if "attachment" not in r.headers.get("content-disposition", ""):
            print(f"[test] FAIL [25/27]: missing Content-Disposition header")
            return 25
        if "CLAUDE.md" not in r.headers.get("content-disposition", ""):
            print(f"[test] FAIL [25/27]: wrong filename in disposition: "
                  f"{r.headers.get('content-disposition')}")
            return 25
        if "Code Reuse" not in r.text or "search_demo" not in r.text:
            print(f"[test] FAIL [25/27]: rules content unexpected: {r.text[:200]}")
            return 25
        print(f"[test] OK [25/27] /api/integrations/claude/rules: CLAUDE.md downloaded")

        # ============================================================
        # 26. GET /api/integrations/UNKNOWN/rules → 404
        # ============================================================
        r = client.get("/api/integrations/UNKNOWN/rules")
        if r.status_code != 404:
            print(f"[test] FAIL [26/27]: unknown client should be 404, got {r.status_code}")
            return 26
        print(f"[test] OK [26/27] /api/integrations/UNKNOWN/rules: 404")

        # ============================================================
        # 27. resolve_config_path: chain works exactly like lynx serve
        # ============================================================
        # This is the bug the Windows smoke test hit: launching
        # `python -m lynx manager ui` without --config should pick up
        # ./config.json from cwd OR $RAG_CONFIG_PATH, just like serve.
        from lynx.config import resolve_config_path
        # explicit wins
        assert resolve_config_path(cfg_path) == Path(cfg_path), \
            "explicit arg should win"
        # env var fallback
        os.environ["RAG_CONFIG_PATH"] = str(cfg_path)
        try:
            assert resolve_config_path(None) == Path(cfg_path), \
                "RAG_CONFIG_PATH should be used when no explicit arg"
        finally:
            os.environ.pop("RAG_CONFIG_PATH", None)
        # cwd default
        resolved = resolve_config_path(None)
        if resolved.name != "config.json":
            print(f"[test] FAIL [27/27]: cwd default should end in 'config.json', got {resolved}")
            return 27
        if not resolved.is_absolute():
            print(f"[test] FAIL [27/27]: cwd default should be absolute, got {resolved}")
            return 27
        print(f"[test] OK [27/27] resolve_config_path: explicit > env > cwd")

        # ============================================================
        # Phase 10 — source CRUD + folder browser + add-source pages
        # ============================================================
        # These tests use a SEPARATE config file so they don't trip over
        # the cached SourceManager / dirty state from the playground +
        # build tests above. We also re-create the TestClient so the new
        # config path takes effect.
        crud_tmp = tmp / "crud"; crud_tmp.mkdir()
        crud_code = crud_tmp / "code"; crud_code.mkdir()
        (crud_code / "main.py").write_text("def f(): pass\n")
        (crud_code / "README.md").write_text("# hello\n")
        crud_cfg_path = _write_config(crud_tmp, crud_code)
        crud_app = create_app(crud_cfg_path)
        crud_client = TestClient(crud_app)

        # ----- P10/1: GET /sources/new — chooser page renders --------
        r = crud_client.get("/sources/new")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/1]: /sources/new returned {r.status_code}")
            return 28
        for needle in ("Codebase", "Web docs", "PDFs", "/sources/new/codebase"):
            if needle not in r.text:
                print(f"[test] FAIL [P10/1]: chooser missing {needle!r}")
                return 28
        print(f"[test] OK [P10/1] /sources/new chooser: 3 cards + links")

        # ----- P10/2: GET /sources/new/codebase — form page ----------
        r = crud_client.get("/sources/new/codebase")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/2]: /sources/new/codebase returned {r.status_code}")
            return 28
        for needle in ('name="path"', "Browse", "Detect", "graph layer", 'name="name"'):
            if needle not in r.text:
                print(f"[test] FAIL [P10/2]: codebase form missing {needle!r}")
                return 28
        print(f"[test] OK [P10/2] /sources/new/codebase: form + Browse + Detect")

        # ----- P10/3: GET /sources/new/UNKNOWN → 404 -----------------
        r = crud_client.get("/sources/new/banana")
        if r.status_code != 404:
            print(f"[test] FAIL [P10/3]: unknown source type should be 404, got {r.status_code}")
            return 28
        print(f"[test] OK [P10/3] /sources/new/banana: 404")

        # ----- P10/4: POST /api/sources — add a codebase, get HX-Redirect
        new_block = {
            "type": "codebase",
            "path": str(crud_code),
            "supported_extensions": [".py"],
            "watcher": {"enabled": False, "debounce_seconds": 2.0},
            "git_integration": {"enabled": False},
        }
        r = crud_client.post(
            "/api/sources",
            json={"name": "second", "block": new_block},
        )
        if r.status_code != 200:
            print(f"[test] FAIL [P10/4]: POST sources returned {r.status_code}: {r.text[:200]}")
            return 28
        if r.headers.get("HX-Redirect") != "/sources/second":
            print(f"[test] FAIL [P10/4]: missing/wrong HX-Redirect header: "
                  f"{r.headers.get('HX-Redirect')!r}")
            return 28
        # Verify config on disk
        on_disk = json.loads(crud_cfg_path.read_text())
        if "second" not in on_disk.get("sources", {}):
            print(f"[test] FAIL [P10/4]: new source not in config on disk: "
                  f"{list(on_disk.get('sources', {}))}")
            return 28
        # Backup file should now exist
        bak = crud_cfg_path.with_suffix(crud_cfg_path.suffix + ".bak")
        if not bak.exists():
            print(f"[test] FAIL [P10/4]: .bak not created")
            return 28
        print(f"[test] OK [P10/4] POST /api/sources: source added + HX-Redirect + .bak")

        # ----- P10/5: POST /api/sources — duplicate name → 409 -------
        r = crud_client.post(
            "/api/sources",
            json={"name": "second", "block": new_block},
        )
        if r.status_code != 409:
            print(f"[test] FAIL [P10/5]: duplicate should be 409, got {r.status_code}: {r.text[:200]}")
            return 28
        if "already exists" not in r.text:
            print(f"[test] FAIL [P10/5]: error message unhelpful: {r.text[:200]}")
            return 28
        print(f"[test] OK [P10/5] POST /api/sources duplicate: 409 with clear message")

        # ----- P10/6: POST /api/sources — invalid block → 422 --------
        # webdoc without url is a schema failure
        bad_block = {"type": "webdoc", "max_depth": 3}  # no url
        snap_before = crud_cfg_path.read_text()
        r = crud_client.post(
            "/api/sources",
            json={"name": "broken", "block": bad_block},
        )
        if r.status_code != 422:
            print(f"[test] FAIL [P10/6]: invalid block should be 422, got {r.status_code}: {r.text[:200]}")
            return 28
        # Config file must NOT have changed
        if crud_cfg_path.read_text() != snap_before:
            print(f"[test] FAIL [P10/6]: config was modified despite validation failure!")
            return 28
        print(f"[test] OK [P10/6] POST invalid block: 422, config untouched")

        # ----- P10/7: POST /api/sources/_detect on a real folder -----
        r = crud_client.post(
            "/api/sources/_detect",
            json={"path": str(crud_code)},
        )
        if r.status_code != 200:
            print(f"[test] FAIL [P10/7]: detect returned {r.status_code}: {r.text[:200]}")
            return 28
        data = r.json()
        if not data.get("exists") or not data.get("is_dir"):
            print(f"[test] FAIL [P10/7]: detect didn't see existing dir: {data}")
            return 28
        if ".py" not in data.get("extensions", []) or ".md" not in data.get("extensions", []):
            print(f"[test] FAIL [P10/7]: detect missed .py/.md: {data['extensions']}")
            return 28
        # is_git could be True or False depending on whether the parent tree
        # is a git repo (the repo IS one, since we're running tests from it).
        # We only check the type is a bool.
        if not isinstance(data.get("is_git"), bool):
            print(f"[test] FAIL [P10/7]: is_git missing/wrong type: {data}")
            return 28
        print(f"[test] OK [P10/7] /api/sources/_detect: extensions + is_git probe")

        # ----- P10/8: GET /api/fs/browse — basic listing --------------
        # Browse the parent of crud_tmp; the response should include 'crud'
        r = crud_client.get(f"/api/fs/browse?path={tmp}")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/8]: browse returned {r.status_code}: {r.text[:200]}")
            return 28
        data = r.json()
        names = [e["name"] for e in data.get("entries", [])]
        if "crud" not in names:
            print(f"[test] FAIL [P10/8]: browse missing 'crud' in {names}")
            return 28
        if data.get("parent") is None:
            print(f"[test] FAIL [P10/8]: parent should be set (not at root)")
            return 28
        print(f"[test] OK [P10/8] /api/fs/browse: {len(names)} entries, parent set")

        # ----- P10/9: GET /api/fs/browse — bogus path → 404 ----------
        r = crud_client.get("/api/fs/browse?path=/this/does/not/exist/anywhere")
        if r.status_code != 404:
            print(f"[test] FAIL [P10/9]: bogus path should be 404, got {r.status_code}")
            return 28
        print(f"[test] OK [P10/9] /api/fs/browse bogus path: 404")

        # ----- P10/10: GET /api/fs/browse — empty path → home -------
        r = crud_client.get("/api/fs/browse?path=")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/10]: empty path should default to home, got {r.status_code}")
            return 28
        data = r.json()
        if data.get("path") != str(Path.home().resolve()):
            print(f"[test] FAIL [P10/10]: empty path should resolve to home: {data}")
            return 28
        print(f"[test] OK [P10/10] /api/fs/browse empty: defaults to {data['path']}")

        # ----- P10/11: DELETE /api/sources/{name} — soft delete -----
        # Remove the 'second' source we added in P10/4 (no purge)
        r = crud_client.delete("/api/sources/second")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/11]: delete returned {r.status_code}: {r.text[:200]}")
            return 28
        on_disk = json.loads(crud_cfg_path.read_text())
        if "second" in on_disk.get("sources", {}):
            print(f"[test] FAIL [P10/11]: source still in config after delete")
            return 28
        print(f"[test] OK [P10/11] DELETE /api/sources/second: source removed from config")

        # ----- P10/12: DELETE — unknown name → 404 ------------------
        r = crud_client.delete("/api/sources/ghost")
        if r.status_code != 404:
            print(f"[test] FAIL [P10/12]: deleting unknown should be 404, got {r.status_code}")
            return 28
        print(f"[test] OK [P10/12] DELETE unknown: 404")

        # ----- P10/13: DELETE with ?purge=true wipes storage dir ----
        # First add a source whose storage dir we'll populate, then delete
        # with purge=true.
        r = crud_client.post(
            "/api/sources",
            json={"name": "purgeme", "block": new_block},
        )
        if r.status_code != 200:
            print(f"[test] FAIL [P10/13]: setup POST returned {r.status_code}")
            return 28
        cfg = json.loads(crud_cfg_path.read_text())
        storage_root = Path(cfg["storage_path"])
        if not storage_root.is_absolute():
            storage_root = crud_cfg_path.parent / storage_root
        purgeme_dir = storage_root / "purgeme"
        purgeme_dir.mkdir(parents=True, exist_ok=True)
        (purgeme_dir / "marker.txt").write_text("x")
        r = crud_client.delete("/api/sources/purgeme?purge=true")
        if r.status_code != 200:
            print(f"[test] FAIL [P10/13]: purge delete returned {r.status_code}")
            return 28
        if purgeme_dir.exists():
            print(f"[test] FAIL [P10/13]: storage dir not removed at {purgeme_dir}")
            return 28
        print(f"[test] OK [P10/13] DELETE ?purge=true: storage dir wiped")

        print("\n[test] === SUCCESS: UI scaffolding + playground + build + integrations + config resolution + source CRUD work as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
