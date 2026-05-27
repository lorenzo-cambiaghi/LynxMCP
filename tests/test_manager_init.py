"""Unit tests for the minimal `lynx manager init`.

The v0.9 redesign stripped init to a tiny bootstrap: write a default
config (no sources) + pre-download the embedding model + optionally
launch the UI. All source/AI-client configuration now lives in the web
UI (`/sources/new`), so the wizard's old per-source prompt loop is gone.

We monkey-patch:
  - `init._read_line` to script y/n answers to the 2 prompts
    (overwrite-confirm + open-UI-confirm)
  - `install.download_model` to avoid hitting HuggingFace
  - `subprocess.Popen` to verify whether init tried to launch the UI

Scenarios:
  1. --non-interactive writes default config with sources={}, no prompts
  2. Interactive on a fresh path: no overwrite prompt, model download
     called, "open UI? n" → returns 0, no Popen call
  3. Interactive on an existing path + decline overwrite → returns 1,
     original file untouched
  4. Interactive on an existing path + accept overwrite → returns 0,
     file replaced with default config
  5. --skip-model-download → install.download_model NOT called
  6. Generated config validates through `lynx.config.load_config`
  7. "Open UI? y" answer triggers a subprocess.Popen call
  8. detect_extensions: top-N file extensions, dotted dirs / compiled
     artifacts filtered
  9. is_git_repo: True for .git dir, True for subdir of git repo,
     False for plain folder
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess as _subprocess
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


def _patch_download_model(record: list, return_code: int = 0):
    """Stub out install.download_model. Records calls into `record`.
    Returns a teardown."""
    from lynx.manager import install as install_mod
    original = install_mod.download_model

    def _fake(model_name: str) -> int:
        record.append(model_name)
        return return_code

    install_mod.download_model = _fake
    return lambda: setattr(install_mod, "download_model", original)


def _patch_popen(record: list):
    """Stub subprocess.Popen at the init module's import site. Records
    every call. Returns a teardown."""
    from lynx.manager import init as init_mod
    original = init_mod.subprocess.Popen

    class _FakePopen:
        def __init__(self, args, *a, **kw):
            record.append(list(args))
    init_mod.subprocess.Popen = _FakePopen
    return lambda: setattr(init_mod.subprocess, "Popen", original)


def _make_args(output_path: Path, *, non_interactive: bool = False,
               skip_model_download: bool = False):
    return argparse.Namespace(
        output=str(output_path),
        non_interactive=non_interactive,
        skip_model_download=skip_model_download,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-init-"))
    print(f"[test] tempdir: {tmp}")
    original_cwd = os.getcwd()
    try:
        from lynx.manager.init import run_init

        # ============================================================
        # 1. --non-interactive writes a sensible default config
        # ============================================================
        d1 = tmp / "t1"; d1.mkdir(); os.chdir(d1)
        download_calls: list = []
        teardown_dl = _patch_download_model(download_calls)
        try:
            rc = run_init(_make_args(d1 / "cfg.json", non_interactive=True))
        finally:
            teardown_dl()
        if rc != 0:
            print(f"[test] FAIL [1/9]: non-interactive returned {rc}")
            return 1
        cfg = json.loads((d1 / "cfg.json").read_text())
        if cfg["config_version"] != 2:
            print(f"[test] FAIL [1/9]: config_version wrong: {cfg['config_version']}")
            return 1
        if cfg["sources"] != {}:
            print(f"[test] FAIL [1/9]: sources should be empty dict, got {cfg['sources']}")
            return 1
        if download_calls != ["BAAI/bge-small-en-v1.5"]:
            print(f"[test] FAIL [1/9]: model download not invoked, got {download_calls}")
            return 1
        print(f"[test] OK [1/9] non-interactive: defaults + model download")

        # ============================================================
        # 2. Interactive on fresh path: no overwrite prompt, "open UI? n"
        # ============================================================
        d2 = tmp / "t2"; d2.mkdir(); os.chdir(d2)
        download_calls = []
        popen_calls: list = []
        teardown_in = _patch_input(["n"])  # answer "n" to "Open UI now?"
        teardown_dl = _patch_download_model(download_calls)
        teardown_pop = _patch_popen(popen_calls)
        try:
            rc = run_init(_make_args(d2 / "cfg.json"))
        finally:
            teardown_in(); teardown_dl(); teardown_pop()
        if rc != 0:
            print(f"[test] FAIL [2/9]: interactive fresh returned {rc}")
            return 2
        if download_calls != ["BAAI/bge-small-en-v1.5"]:
            print(f"[test] FAIL [2/9]: model download not invoked, got {download_calls}")
            return 2
        if popen_calls:
            print(f"[test] FAIL [2/9]: UI was launched despite 'n' answer: {popen_calls}")
            return 2
        print(f"[test] OK [2/9] interactive fresh: no Popen on 'n'")

        # ============================================================
        # 3. Existing file + decline overwrite → exit 1, file preserved
        # ============================================================
        d3 = tmp / "t3"; d3.mkdir(); os.chdir(d3)
        existing = d3 / "cfg.json"
        existing.write_text('{"placeholder": true}')
        teardown_in = _patch_input(["n"])  # decline overwrite
        teardown_dl = _patch_download_model([])
        try:
            rc = run_init(_make_args(existing))
        finally:
            teardown_in(); teardown_dl()
        if rc != 1:
            print(f"[test] FAIL [3/9]: decline overwrite should exit 1, got {rc}")
            return 3
        kept = json.loads(existing.read_text())
        if not kept.get("placeholder"):
            print(f"[test] FAIL [3/9]: existing config was overwritten despite decline")
            return 3
        print(f"[test] OK [3/9] decline overwrite: exit 1, file preserved")

        # ============================================================
        # 4. Existing file + accept overwrite → file replaced
        # ============================================================
        d4 = tmp / "t4"; d4.mkdir(); os.chdir(d4)
        existing4 = d4 / "cfg.json"
        existing4.write_text('{"placeholder": true}')
        download_calls = []
        teardown_in = _patch_input(["y", "n"])  # accept overwrite + "open UI? n"
        teardown_dl = _patch_download_model(download_calls)
        teardown_pop = _patch_popen([])
        try:
            rc = run_init(_make_args(existing4))
        finally:
            teardown_in(); teardown_dl(); teardown_pop()
        if rc != 0:
            print(f"[test] FAIL [4/9]: accept overwrite should exit 0, got {rc}")
            return 4
        cfg4 = json.loads(existing4.read_text())
        if cfg4.get("config_version") != 2:
            print(f"[test] FAIL [4/9]: file not actually replaced: {cfg4}")
            return 4
        if cfg4["sources"] != {}:
            print(f"[test] FAIL [4/9]: replaced file has unexpected sources: {cfg4['sources']}")
            return 4
        print(f"[test] OK [4/9] accept overwrite: file replaced with default config")

        # ============================================================
        # 5. --skip-model-download bypasses install.download_model
        # ============================================================
        d5 = tmp / "t5"; d5.mkdir(); os.chdir(d5)
        download_calls = []
        teardown_dl = _patch_download_model(download_calls)
        try:
            rc = run_init(_make_args(d5 / "cfg.json",
                                     non_interactive=True,
                                     skip_model_download=True))
        finally:
            teardown_dl()
        if rc != 0:
            print(f"[test] FAIL [5/9]: skip-model returned {rc}")
            return 5
        if download_calls:
            print(f"[test] FAIL [5/9]: download_model invoked despite --skip: {download_calls}")
            return 5
        print(f"[test] OK [5/9] --skip-model-download: download bypassed")

        # ============================================================
        # 6. Generated config validates via load_config
        # ============================================================
        d6 = tmp / "t6"; d6.mkdir(); os.chdir(d6)
        teardown_dl = _patch_download_model([])
        try:
            run_init(_make_args(d6 / "cfg.json", non_interactive=True))
        finally:
            teardown_dl()
        from lynx.config import load_config
        try:
            cfg6 = load_config(d6 / "cfg.json")
        except SystemExit as e:
            print(f"[test] FAIL [6/9]: generated config rejected by load_config (exit {e.code})")
            return 6
        if cfg6.embedding.model_name != "BAAI/bge-small-en-v1.5":
            print(f"[test] FAIL [6/9]: embedding model unexpected: {cfg6.embedding.model_name}")
            return 6
        print(f"[test] OK [6/9] generated config validates via load_config")

        # ============================================================
        # 7. "Open UI? y" triggers a subprocess.Popen call
        # ============================================================
        d7 = tmp / "t7"; d7.mkdir(); os.chdir(d7)
        popen_calls = []
        teardown_in = _patch_input(["y"])  # answer "y" to "Open UI?"
        teardown_dl = _patch_download_model([])
        teardown_pop = _patch_popen(popen_calls)
        try:
            rc = run_init(_make_args(d7 / "cfg.json"))
        finally:
            teardown_in(); teardown_dl(); teardown_pop()
        if rc != 0:
            print(f"[test] FAIL [7/9]: open-ui-y returned {rc}")
            return 7
        if len(popen_calls) != 1:
            print(f"[test] FAIL [7/9]: expected 1 Popen call, got {len(popen_calls)}: {popen_calls}")
            return 7
        cmd = popen_calls[0]
        # cmd[0] is sys.executable; cmd[1:] should be ['-m','lynx','manager','ui','--config',<path>]
        if cmd[1:5] != ["-m", "lynx", "manager", "ui"]:
            print(f"[test] FAIL [7/9]: Popen cmd unexpected: {cmd}")
            return 7
        if "--config" not in cmd or str((d7 / "cfg.json").resolve()) not in cmd:
            print(f"[test] FAIL [7/9]: Popen cmd missing --config path: {cmd}")
            return 7
        print(f"[test] OK [7/9] 'open UI? y' spawns subprocess.Popen with right argv")

        # ============================================================
        # 8. detect_extensions
        # ============================================================
        d8 = tmp / "t8"; d8.mkdir(); os.chdir(d8)
        for i in range(5):
            (d8 / f"a{i}.py").write_text("")
        for i in range(2):
            (d8 / f"b{i}.md").write_text("")
        (d8 / "junk.pyc").write_text("")
        (d8 / ".git").mkdir()
        (d8 / ".git" / "irrelevant.py").write_text("")  # under dot dir, ignored
        from lynx.manager.ui.detect import detect_extensions, is_git_repo
        exts = detect_extensions(d8)
        if ".py" not in exts or ".md" not in exts:
            print(f"[test] FAIL [8/9]: expected .py + .md in {exts}")
            return 8
        if ".pyc" in exts:
            print(f"[test] FAIL [8/9]: .pyc not filtered: {exts}")
            return 8
        # Ordering: .py (5) should come before .md (2)
        if exts.index(".py") > exts.index(".md"):
            print(f"[test] FAIL [8/9]: .py should rank above .md: {exts}")
            return 8
        print(f"[test] OK [8/9] detect_extensions: {exts}")

        # ============================================================
        # 9. is_git_repo: plain dir vs .git dir vs subdir of repo
        # ============================================================
        d9 = tmp / "t9"; d9.mkdir(); os.chdir(d9)
        plain = d9 / "plain"; plain.mkdir()
        if is_git_repo(plain):
            print(f"[test] FAIL [9/9]: plain dir incorrectly reported as git")
            return 9
        repo = d9 / "repo"; (repo / ".git").mkdir(parents=True)
        if not is_git_repo(repo):
            print(f"[test] FAIL [9/9]: real git repo not detected")
            return 9
        sub = repo / "sub"; sub.mkdir()
        if not is_git_repo(sub):
            print(f"[test] FAIL [9/9]: subdir of git repo not detected")
            return 9
        print(f"[test] OK [9/9] is_git_repo: detects .git in dir + parents")

        print("\n[test] === SUCCESS: minimal init works as expected ===")
        return 0
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
