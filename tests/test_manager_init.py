"""Unit tests for `lynx manager init` wizard.

We feed scripted input into the wizard via monkey-patching `_read_line`
in `lynx.manager.init`. This is more reliable than redirecting stdin
because it bypasses any TTY detection / buffering issues.

Scenarios:
  1. Non-interactive mode writes a sensible default config
  2. Existing config without overwrite confirmation → wizard aborts cleanly
  3. Interactive: single codebase source, all defaults → config valid
  4. Interactive: codebase + pdf + webdoc → all three sources captured
  5. Interactive: declines AI rules file → no file written
  6. Interactive: accepts AI rules file → CLAUDE.md written
  7. Validators: invalid path retried until valid one given
  8. Validators: invalid source name retried
  9. detect_extensions: returns top N file extensions in a folder
 10. is_git_repo: True for repo, False for plain folder
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _scripted_input(answers: list):
    """Build a closure that returns the next answer on each call."""
    iter_answers = iter(answers)

    def _next() -> str:
        try:
            return next(iter_answers)
        except StopIteration:
            raise EOFError("test ran out of scripted answers")
    return _next


def _patch_input(monkey_answers: list):
    """Install the scripted input + return a teardown to undo it."""
    from lynx.manager import init as init_mod
    original = init_mod._read_line
    init_mod._read_line = _scripted_input(monkey_answers)
    return lambda: setattr(init_mod, "_read_line", original)


def _make_args(output_path: Path, non_interactive: bool = False):
    return argparse.Namespace(
        output=str(output_path),
        non_interactive=non_interactive,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-init-"))
    print(f"[test] tempdir: {tmp}")
    original_cwd = os.getcwd()
    try:
        # Each test changes cwd to its own sub-dir so AI rules files
        # (CLAUDE.md / AGENTS.md) don't collide.

        # ============================================================
        # 1. Non-interactive mode writes defaults
        # ============================================================
        d1 = tmp / "t1"; d1.mkdir()
        os.chdir(d1)
        from lynx.manager.init import run_init
        rc = run_init(_make_args(d1 / "cfg.json", non_interactive=True))
        if rc != 0:
            print(f"[test] FAIL [1/10]: non-interactive returned {rc}")
            return 1
        cfg = json.loads((d1 / "cfg.json").read_text())
        if cfg["config_version"] != 2:
            print(f"[test] FAIL [1/10]: config_version wrong: {cfg['config_version']}")
            return 1
        if "myproject" not in cfg["sources"]:
            print(f"[test] FAIL [1/10]: default source 'myproject' missing")
            return 1
        print(f"[test] OK [1/10] non-interactive: defaults written cleanly")

        # ============================================================
        # 2. Interactive: existing config + decline overwrite → abort
        # ============================================================
        d2 = tmp / "t2"; d2.mkdir(); os.chdir(d2)
        existing = d2 / "cfg.json"
        existing.write_text('{"placeholder": true}')
        teardown = _patch_input([
            "./rag_storage",          # storage_path (use default actually)
            "",                       # embed default
            "",                       # top_k default
            "",                       # mode default
            "n",                      # reranker no
            "codebase",               # add source: codebase
            "src",                    # source name
            str(d2),                  # path
            "",                       # extensions default
            "n",                      # watcher no (avoid background)
            "n",                      # graph no
            "n",                      # add another?
            "n",                      # overwrite existing? NO → abort
        ])
        try:
            rc = run_init(_make_args(existing, non_interactive=False))
        finally:
            teardown()
        # KeyboardInterrupt from _write_config → run_init returns 130
        if rc != 130:
            print(f"[test] FAIL [2/10]: decline overwrite should exit 130, got {rc}")
            return 2
        # Original file untouched
        kept = json.loads(existing.read_text())
        if not kept.get("placeholder"):
            print(f"[test] FAIL [2/10]: existing config was overwritten despite decline")
            return 2
        print(f"[test] OK [2/10] decline overwrite: aborts cleanly, existing config preserved")

        # ============================================================
        # 3. Interactive: single codebase source, defaults → valid
        # ============================================================
        d3 = tmp / "t3"; d3.mkdir(); os.chdir(d3)
        (d3 / "main.py").write_text("def f(): pass\n")
        out3 = d3 / "cfg.json"
        teardown = _patch_input([
            "",                       # storage_path default
            "",                       # embed default
            "",                       # top_k default
            "",                       # mode default
            "n",                      # reranker no
            "codebase",               # add source: codebase
            "demo",                   # name
            str(d3),                  # path
            "",                       # extensions auto-detected default
            "n",                      # watcher
            "n",                      # graph
            "n",                      # add another?
            "n",                      # don't write rules file
        ])
        try:
            rc = run_init(_make_args(out3, non_interactive=False))
        finally:
            teardown()
        if rc != 0:
            print(f"[test] FAIL [3/10]: wizard returned {rc}")
            return 3
        cfg = json.loads(out3.read_text())
        if cfg["sources"]["demo"]["type"] != "codebase":
            print(f"[test] FAIL [3/10]: source type wrong: {cfg['sources']}")
            return 3
        # Validate the generated config actually parses through load_config
        from lynx.config import load_config
        try:
            load_config(out3)
        except SystemExit as e:
            print(f"[test] FAIL [3/10]: generated config rejected by load_config: {e}")
            return 3
        print(f"[test] OK [3/10] interactive minimal: valid config generated")

        # ============================================================
        # 4. Three sources (codebase + pdf + webdoc)
        # ============================================================
        d4 = tmp / "t4"; d4.mkdir(); os.chdir(d4)
        (d4 / "code").mkdir(); (d4 / "code" / "x.py").write_text("def x(): pass\n")
        (d4 / "pdfs").mkdir()
        out4 = d4 / "cfg.json"
        teardown = _patch_input([
            "",                       # storage
            "",                       # embed
            "",                       # top_k
            "",                       # mode
            "n",                      # reranker
            # source 1: codebase
            "codebase", "src1", str(d4 / "code"), "", "n", "n",
            "y",                      # add another?
            # source 2: pdf
            "pdf", "manuals", str(d4 / "pdfs"), "y", "auto", "n",
            "y",                      # add another?
            # source 3: webdoc
            "webdoc", "docs", "https://example.com/docs/", "3", "500", "y",
            "n",                      # add another?
            "n",                      # rules file
        ])
        try:
            rc = run_init(_make_args(out4, non_interactive=False))
        finally:
            teardown()
        if rc != 0:
            print(f"[test] FAIL [4/10]: 3-source wizard returned {rc}")
            return 4
        cfg = json.loads(out4.read_text())
        expected_names = {"src1", "manuals", "docs"}
        if set(cfg["sources"]) != expected_names:
            print(f"[test] FAIL [4/10]: source names wrong: {set(cfg['sources'])}")
            return 4
        if cfg["sources"]["manuals"]["type"] != "pdf":
            print(f"[test] FAIL [4/10]: pdf type wrong")
            return 4
        if cfg["sources"]["docs"]["type"] != "webdoc":
            print(f"[test] FAIL [4/10]: webdoc type wrong")
            return 4
        print(f"[test] OK [4/10] 3 sources captured: src1/manuals/docs")

        # ============================================================
        # 5. Decline AI rules file → no file written
        # ============================================================
        d5 = tmp / "t5"; d5.mkdir(); os.chdir(d5)
        (d5 / "x.py").write_text("def x(): pass\n")
        out5 = d5 / "cfg.json"
        teardown = _patch_input([
            "", "", "", "", "n",      # globals
            "codebase", "p", str(d5), "", "n", "n", "n",
            "n",                      # rules file: NO
        ])
        try:
            run_init(_make_args(out5, non_interactive=False))
        finally:
            teardown()
        if (d5 / "CLAUDE.md").exists() or (d5 / "AGENTS.md").exists():
            print(f"[test] FAIL [5/10]: rules file written despite decline")
            return 5
        print(f"[test] OK [5/10] decline rules file: nothing written")

        # ============================================================
        # 6. Accept AI rules file → CLAUDE.md written
        # ============================================================
        d6 = tmp / "t6"; d6.mkdir(); os.chdir(d6)
        (d6 / "x.py").write_text("def x(): pass\n")
        out6 = d6 / "cfg.json"
        teardown = _patch_input([
            "", "", "", "", "n",      # globals
            "codebase", "myproj", str(d6), "", "n", "n", "n",
            "y",                      # rules file: yes
            "1",                      # claude
        ])
        try:
            run_init(_make_args(out6, non_interactive=False))
        finally:
            teardown()
        rules = d6 / "CLAUDE.md"
        if not rules.exists():
            print(f"[test] FAIL [6/10]: CLAUDE.md not written")
            return 6
        content = rules.read_text()
        if "search_myproj" not in content:
            print(f"[test] FAIL [6/10]: rules file doesn't use real source name 'myproj'")
            return 6
        print(f"[test] OK [6/10] CLAUDE.md written with real source name")

        # ============================================================
        # 7. Validator retries on bad path
        # ============================================================
        d7 = tmp / "t7"; d7.mkdir(); os.chdir(d7)
        (d7 / "x.py").write_text("def x(): pass\n")
        out7 = d7 / "cfg.json"
        teardown = _patch_input([
            "", "", "", "", "n",
            "codebase", "demo",
            "/this/does/not/exist",   # bad path #1
            "/also/bad",              # bad path #2
            str(d7),                  # finally valid
            "", "n", "n", "n", "n",
        ])
        try:
            rc = run_init(_make_args(out7, non_interactive=False))
        finally:
            teardown()
        if rc != 0:
            print(f"[test] FAIL [7/10]: retry-on-bad-path wizard returned {rc}")
            return 7
        print(f"[test] OK [7/10] validator retries on bad path until valid")

        # ============================================================
        # 8. Validator retries on bad source name
        # ============================================================
        d8 = tmp / "t8"; d8.mkdir(); os.chdir(d8)
        (d8 / "x.py").write_text("def x(): pass\n")
        out8 = d8 / "cfg.json"
        teardown = _patch_input([
            "", "", "", "", "n",
            "codebase",
            "9bad-name",              # invalid (starts with digit, has dash)
            "also.bad",               # invalid (dot)
            "good_name",              # finally valid
            str(d8), "", "n", "n", "n", "n",
        ])
        try:
            rc = run_init(_make_args(out8, non_interactive=False))
        finally:
            teardown()
        if rc != 0:
            print(f"[test] FAIL [8/10]: retry-on-bad-name wizard returned {rc}")
            return 8
        print(f"[test] OK [8/10] validator retries on bad source name")

        # ============================================================
        # 9. detect_extensions returns top file types
        # ============================================================
        d9 = tmp / "t9"; d9.mkdir(); os.chdir(d9)
        for i in range(5):
            (d9 / f"a{i}.py").write_text("")
        for i in range(2):
            (d9 / f"b{i}.md").write_text("")
        (d9 / "junk.pyc").write_text("")  # should be filtered
        (d9 / ".git").mkdir()
        (d9 / ".git" / "irrelevant.py").write_text("")  # under dot dir, ignored
        from lynx.manager.init import detect_extensions
        exts = detect_extensions(d9)
        if ".py" not in exts:
            print(f"[test] FAIL [9/10]: .py missing from {exts}")
            return 9
        if ".md" not in exts:
            print(f"[test] FAIL [9/10]: .md missing from {exts}")
            return 9
        if ".pyc" in exts:
            print(f"[test] FAIL [9/10]: .pyc not filtered: {exts}")
            return 9
        # .py count should be 5 (not 6) — dot dir excluded
        print(f"[test] OK [9/10] detect_extensions: {exts}")

        # ============================================================
        # 10. is_git_repo: True for actual repo
        # ============================================================
        d10 = tmp / "t10"; d10.mkdir(); os.chdir(d10)
        from lynx.manager.init import is_git_repo
        plain = d10 / "plain"; plain.mkdir()
        if is_git_repo(plain):
            print(f"[test] FAIL [10/10]: plain dir incorrectly reported as git")
            return 10
        repo = d10 / "repo"; (repo / ".git").mkdir(parents=True)
        if not is_git_repo(repo):
            print(f"[test] FAIL [10/10]: real git repo not detected")
            return 10
        # Sub-directory inside repo should also be detected
        sub = repo / "sub"; sub.mkdir()
        if not is_git_repo(sub):
            print(f"[test] FAIL [10/10]: subdir of git repo not detected")
            return 10
        print(f"[test] OK [10/10] is_git_repo: detects .git in dir + parents")

        print("\n[test] === SUCCESS: init wizard works as expected ===")
        return 0
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
