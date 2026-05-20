"""Integration test for the graph layer plumbing:
config -> CodebaseBackend -> SourceManager pass-through.

We monkey-patch `CodebaseRAG.__init__` to a no-op so we don't have to
load the embedding model, but everything else (config validation, the
GraphLayer construction, the SourceManager dispatch) runs for real.

Scenarios:
  1. Config validation accepts `graph: {enabled: true/false}` and defaults
     to false when absent. Backward-compat: existing configs still load.
  2. When graph.enabled=true, CodebaseBackend.graph is a real GraphLayer.
  3. When graph.enabled=false (or absent), CodebaseBackend.graph is None.
  4. SourceManager.get_callers / etc. work when graph is enabled.
  5. SourceManager.get_callers raises ValueError when graph is disabled.
  6. status() includes the `graph` sub-dict only when graph enabled.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


def _make_codebase(root: Path, files: dict) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _stub_rag(monkeypatch_target_module):
    """Replace CodebaseRAG.__init__ with a no-op + minimal attributes that
    CodebaseBackend.status() needs. Avoids the HuggingFace embedding load.
    """
    from lynx.rag_manager import CodebaseRAG

    original_init = CodebaseRAG.__init__

    class _StubVectorStoreCollection:
        def count(self):
            return 0

    class _StubVectorStore:
        def __init__(self):
            self._collection = _StubVectorStoreCollection()

    def stub_init(self, **kwargs):
        self.codebase_path = Path(kwargs["codebase_path"])
        self.storage_path = Path(kwargs["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVectorStore()

    def stub_check_drift(self):
        return None

    def stub_drift_status_text(self):
        return "No config drift detected."

    def stub_needs_update(self):
        return False

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.check_config_drift = stub_check_drift
    CodebaseRAG.drift_status_text = stub_drift_status_text
    CodebaseRAG.needs_update = stub_needs_update
    return original_init


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-graph-int-"))
    print(f"[test] tempdir: {tmp}")
    try:
        code = tmp / "code"
        code.mkdir()
        _make_codebase(code, {
            "util.py": "def helper(x):\n    return x + 1\n",
            "main.py": "from util import helper\ndef go():\n    return helper(1)\n",
        })

        # Stub RAG so we don't load HF embeddings.
        _stub_rag(None)

        from lynx.config import load_config

        # =========================================================
        # 1. Config validation: graph.enabled defaults to False
        # =========================================================
        cfg_path = tmp / "cfg_no_graph.json"
        cfg_path.write_text(json.dumps({
            "config_version": 2,
            "storage_path": str(tmp / "storage_no_graph"),
            "sources": {
                "myproj": {
                    "type": "codebase",
                    "path": str(code),
                    "supported_extensions": [".py"],
                }
            }
        }), encoding="utf-8")
        cfg = load_config(cfg_path)
        src_cfg = cfg.sources["myproj"]
        if src_cfg.get("graph", {}).get("enabled") is not False:
            print(f"[test] FAIL [1/6]: default graph.enabled should be False, got {src_cfg!r}")
            return 1
        print("[test] OK [1/6] config: missing graph block -> enabled=False (backward compat)")

        # =========================================================
        # 2. graph.enabled=true -> CodebaseBackend.graph is real
        # =========================================================
        cfg_path = tmp / "cfg_graph.json"
        cfg_path.write_text(json.dumps({
            "config_version": 2,
            "storage_path": str(tmp / "storage_graph"),
            "sources": {
                "withgraph": {
                    "type": "codebase",
                    "path": str(code),
                    "supported_extensions": [".py"],
                    "graph": {"enabled": True},
                    "watcher": {"enabled": False},
                },
                "withoutgraph": {
                    "type": "codebase",
                    "path": str(code),
                    "supported_extensions": [".py"],
                    "watcher": {"enabled": False},
                },
            }
        }), encoding="utf-8")
        cfg = load_config(cfg_path)
        from lynx.source_manager import SourceManager
        mgr = SourceManager(cfg)

        if mgr.get("withgraph").graph is None:
            print("[test] FAIL [2/6]: 'withgraph' backend.graph is None despite graph.enabled=true")
            return 2
        if mgr.get("withoutgraph").graph is not None:
            print("[test] FAIL [2/6]: 'withoutgraph' backend.graph is NOT None (should be)")
            return 2
        print("[test] OK [2/6] backend.graph: present only when graph.enabled=true")

        # =========================================================
        # 3. (covered by #2) — implicit
        # =========================================================
        print("[test] OK [3/6] backend.graph: None when graph disabled (covered above)")

        # =========================================================
        # 4. SourceManager.get_callers works on enabled source
        # =========================================================
        # Build the graph for the enabled source.
        mgr.get("withgraph").graph.rebuild(force=True)
        callers = mgr.get_callers("withgraph", "helper")
        if not callers:
            print(f"[test] FAIL [4/6]: get_callers('withgraph', 'helper') returned []")
            return 4
        if not any(c["target"]["label"] == "helper" for c in callers):
            print(f"[test] FAIL [4/6]: callers payload missing target=helper: {callers}")
            return 4
        # architectural_overview should produce god_nodes / communities
        overview = mgr.architectural_overview("withgraph")
        if "god_nodes" not in overview or "communities" not in overview:
            print(f"[test] FAIL [4/6]: architectural_overview missing keys: {overview.keys()}")
            return 4
        print(f"[test] OK [4/6] manager.get_callers / architectural_overview work end-to-end")

        # =========================================================
        # 5. Disabled source raises ValueError
        # =========================================================
        try:
            mgr.get_callers("withoutgraph", "helper")
            print("[test] FAIL [5/6]: expected ValueError for disabled source")
            return 5
        except ValueError as e:
            if "graph layer enabled" not in str(e):
                print(f"[test] FAIL [5/6]: ValueError message unhelpful: {e}")
                return 5
        print("[test] OK [5/6] disabled source raises ValueError with clear message")

        # =========================================================
        # 6. status() shape varies by graph enablement
        # =========================================================
        s_with = mgr.get("withgraph").status()
        s_without = mgr.get("withoutgraph").status()
        if "graph" not in s_with:
            print(f"[test] FAIL [6/6]: status of enabled source missing 'graph' key: {s_with}")
            return 6
        if "graph" in s_without:
            print(f"[test] FAIL [6/6]: status of disabled source has spurious 'graph' key: {s_without}")
            return 6
        # The graph sub-status must report counts after rebuild
        if s_with["graph"]["nodes"] <= 0:
            print(f"[test] FAIL [6/6]: status.graph.nodes <= 0 after rebuild: {s_with}")
            return 6
        print(f"[test] OK [6/6] status(): includes graph subdict only when enabled "
              f"(nodes={s_with['graph']['nodes']})")

        print("\n[test] === SUCCESS: graph integration plumbing works ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
