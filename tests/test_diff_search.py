"""Unit tests for `CodebaseBackend.search_diff()`.

Creates a real git repo in a tempdir, makes commits on `main` and on a
feature branch, then verifies search_diff returns only results from the
files changed on the feature branch.

CodebaseRAG is stubbed (no HF embeddings load) — we provide canned
search results and verify the `paths=` filter correctly narrows them
down.

Scenarios:
  1. Auto-detect `main`: feature branch modifies 2 files; search_diff
     restricts results to those 2 files.
  2. Explicit base override: `base="develop"` works when develop exists.
  3. No `base` and no main/master/develop → ValueError.
  4. Specified base doesn't exist → ValueError.
  5. No changes vs base → returns {"hits": [], "modified_files": []}.
  6. Added file (status A) included; deleted file (status D) excluded.
  7. Non-git directory → ValueError.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _git(repo: Path, *args: str) -> str:
    """Run a git command in `repo`. Returns stdout."""
    result = subprocess.run(
        ["git", *args], cwd=str(repo),
        check=True, capture_output=True, text=True,
        env={**os.environ,
             "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@x",
             "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@x"},
    )
    return result.stdout


def _stub_rag():
    """Stub CodebaseRAG; tests inject `search_returns` per scenario."""
    from lynx.rag_manager import CodebaseRAG

    class _StubColl:
        def count(self): return 0

    class _StubVS:
        def __init__(self): self._collection = _StubColl()

    state = {"search_returns": [], "last_kw": None}

    def stub_init(self, **kw):
        self.codebase_path = Path(kw["codebase_path"])
        self.storage_path = Path(kw["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVS()
        self.search_mode = kw.get("search_mode", "hybrid")

    def stub_search(self, query, top_k=5, **kw):
        # Record what paths the backend passed in — tests assert on this
        state["last_kw"] = kw
        # Apply the `paths` filter the real impl would do, so the test
        # exercises the end-to-end restriction.
        results = list(state["search_returns"])
        paths = kw.get("paths") or []
        if paths:
            filtered = []
            for r in results:
                fp = (r.get("file_path") or "").replace("\\", "/")
                if any(fp.endswith(p) or p in fp for p in paths):
                    filtered.append(r)
            results = filtered
        return results[:top_k]

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.search = stub_search
    CodebaseRAG.update = lambda self, force=False: None
    CodebaseRAG.update_file = lambda self, p: None
    CodebaseRAG.remove_file = lambda self, p: None
    CodebaseRAG.check_config_drift = lambda self: None
    return state


def _shared():
    return SimpleNamespace(
        embedding=SimpleNamespace(model_name="stub"),
        search=SimpleNamespace(
            mode="hybrid", rrf_k=60, candidate_pool_size=30,
            deep=SimpleNamespace(score_thresholds={"hybrid": 0.012}),
            reranker=SimpleNamespace(enabled=False, model_name="", top_n_before_rerank=30),
        ),
    )


def _make_backend(repo_path: Path, storage_dir: Path):
    from lynx.sources.codebase import CodebaseBackend
    cfg = {
        "type": "codebase",
        "path": repo_path,
        "supported_extensions": frozenset({".py"}),
        "ignored_path_fragments": [],
        "watcher": {"enabled": False},
        "git_integration": {"enabled": True},
        "graph": {"enabled": False},
    }
    return CodebaseBackend(
        name="repo", source_config=cfg, shared_config=_shared(),
        storage_dir=storage_dir,
    )


def _seed_repo(repo: Path, default_branch: str = "main") -> None:
    """Init repo, commit two files on `default_branch`."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", default_branch)
    (repo / "a.py").write_text("def a():\n    return 1\n")
    (repo / "b.py").write_text("def b():\n    return 2\n")
    _git(repo, "add", "a.py", "b.py")
    _git(repo, "commit", "-q", "-m", "initial")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-diff-"))
    print(f"[test] tempdir: {tmp}")
    try:
        state = _stub_rag()

        # ============================================================
        # 1. Auto-detect main + feature-branch diff
        # ============================================================
        repo1 = tmp / "r1"
        _seed_repo(repo1, "main")
        _git(repo1, "checkout", "-q", "-b", "feature/x")
        (repo1 / "a.py").write_text("def a():\n    return 999  # CHANGED\n")
        _git(repo1, "add", "a.py")
        _git(repo1, "commit", "-q", "-m", "modify a")
        (repo1 / "c.py").write_text("def c():\n    return 3  # NEW\n")
        _git(repo1, "add", "c.py")
        _git(repo1, "commit", "-q", "-m", "add c")

        b1 = _make_backend(repo1, tmp / "s1")
        # Pretend search returns hits for a.py, b.py, c.py — only a + c
        # should survive the path filter (b wasn't touched).
        state["search_returns"] = [
            {"file": "a.py", "file_path": str(repo1 / "a.py"), "score": 0.9, "content": "..."},
            {"file": "b.py", "file_path": str(repo1 / "b.py"), "score": 0.8, "content": "..."},
            {"file": "c.py", "file_path": str(repo1 / "c.py"), "score": 0.7, "content": "..."},
        ]
        out = b1.search_diff("any query", top_k=10)
        if out["base"] != "main":
            print(f"[test] FAIL [1/7]: expected base=main, got {out['base']!r}")
            return 1
        modified_set = set(out["modified_files"])
        if not {"a.py", "c.py"}.issubset(modified_set):
            print(f"[test] FAIL [1/7]: missing expected modified files: {modified_set}")
            return 1
        if "b.py" in modified_set:
            print(f"[test] FAIL [1/7]: b.py incorrectly in modified list: {modified_set}")
            return 1
        hit_files = {h["file"] for h in out["hits"]}
        if "b.py" in hit_files:
            print(f"[test] FAIL [1/7]: b.py leaked into hits despite not modified")
            return 1
        if not {"a.py", "c.py"}.issubset(hit_files):
            print(f"[test] FAIL [1/7]: hits should include a.py and c.py: {hit_files}")
            return 1
        print(f"[test] OK [1/7] auto-detect main: 2 modified files, hits restricted")

        # ============================================================
        # 2. Explicit base override
        # ============================================================
        repo2 = tmp / "r2"
        _seed_repo(repo2, "main")
        _git(repo2, "checkout", "-q", "-b", "develop")
        (repo2 / "x.py").write_text("def x():\n    return 1\n")
        _git(repo2, "add", "x.py"); _git(repo2, "commit", "-q", "-m", "add x on develop")
        _git(repo2, "checkout", "-q", "-b", "feature/y")
        (repo2 / "x.py").write_text("def x():\n    return 42  # changed\n")
        _git(repo2, "add", "x.py"); _git(repo2, "commit", "-q", "-m", "modify x")

        b2 = _make_backend(repo2, tmp / "s2")
        state["search_returns"] = [
            {"file": "x.py", "file_path": str(repo2 / "x.py"), "score": 0.9, "content": "..."},
        ]
        out = b2.search_diff("any", base="develop")
        if out["base"] != "develop":
            print(f"[test] FAIL [2/7]: explicit base=develop ignored: {out['base']}")
            return 2
        if "x.py" not in out["modified_files"]:
            print(f"[test] FAIL [2/7]: x.py not in modified vs develop: {out['modified_files']}")
            return 2
        print(f"[test] OK [2/7] explicit base=develop honored")

        # ============================================================
        # 3. No main/master/develop + no base → ValueError
        # ============================================================
        repo3 = tmp / "r3"
        _seed_repo(repo3, "trunk")  # unusual name
        b3 = _make_backend(repo3, tmp / "s3")
        try:
            b3.search_diff("any")
            print(f"[test] FAIL [3/7]: should raise when no default branch found")
            return 3
        except ValueError as e:
            if "no default base branch" not in str(e):
                print(f"[test] FAIL [3/7]: unhelpful error: {e}")
                return 3
        print(f"[test] OK [3/7] no default branch → ValueError with hint")

        # ============================================================
        # 4. Specified base doesn't exist → ValueError
        # ============================================================
        try:
            b3.search_diff("any", base="nonexistent_branch")
            print(f"[test] FAIL [4/7]: bogus base should raise")
            return 4
        except ValueError as e:
            if "not found in repo" not in str(e):
                print(f"[test] FAIL [4/7]: unhelpful error: {e}")
                return 4
        print(f"[test] OK [4/7] bogus base → ValueError")

        # ============================================================
        # 5. No changes vs base → empty hits + note
        # ============================================================
        repo5 = tmp / "r5"
        _seed_repo(repo5, "main")
        # Stay on main — no changes between HEAD and main (it IS main)
        b5 = _make_backend(repo5, tmp / "s5")
        out = b5.search_diff("any")
        if out["hits"] != [] or out["modified_files"] != []:
            print(f"[test] FAIL [5/7]: expected empty result, got {out}")
            return 5
        if "No files" not in out.get("note", ""):
            print(f"[test] FAIL [5/7]: expected explanatory note: {out}")
            return 5
        print(f"[test] OK [5/7] no changes vs base: empty hits + note")

        # ============================================================
        # 6. Added (A) included; deleted (D) excluded
        # ============================================================
        repo6 = tmp / "r6"
        _seed_repo(repo6, "main")
        _git(repo6, "checkout", "-q", "-b", "feature/z")
        (repo6 / "added.py").write_text("def added(): pass\n")
        (repo6 / "a.py").unlink()  # delete a.py
        _git(repo6, "add", "-A")
        _git(repo6, "commit", "-q", "-m", "add and delete")
        b6 = _make_backend(repo6, tmp / "s6")
        state["search_returns"] = [
            {"file": "added.py", "file_path": str(repo6 / "added.py"), "score": 0.9, "content": "..."},
        ]
        out = b6.search_diff("any")
        if "added.py" not in out["modified_files"]:
            print(f"[test] FAIL [6/7]: added.py missing from diff list: {out['modified_files']}")
            return 6
        if "a.py" in out["modified_files"]:
            print(f"[test] FAIL [6/7]: deleted a.py incorrectly listed: {out['modified_files']}")
            return 6
        print(f"[test] OK [6/7] Added file included, Deleted file excluded")

        # ============================================================
        # 7. Non-git directory → ValueError
        # ============================================================
        plain = tmp / "plain"; plain.mkdir()
        (plain / "x.py").write_text("def x(): pass\n")
        b7 = _make_backend(plain, tmp / "s7")
        try:
            b7.search_diff("any")
            print(f"[test] FAIL [7/7]: non-git dir should raise")
            return 7
        except ValueError as e:
            if "not a git repository" not in str(e):
                print(f"[test] FAIL [7/7]: unhelpful error: {e}")
                return 7
        print(f"[test] OK [7/7] non-git directory → ValueError")

        print("\n[test] === SUCCESS: search_diff works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
