"""Unit tests for the type=pdf source validator.

Scenarios:
  1. Minimal valid config — defaults applied for omitted extractor fields
  2. Full explicit config round-trips
  3. Missing `path` → SystemExit with clear message
  4. Non-existent `path` → SystemExit
  5. Invalid extractor.backend → SystemExit
  6. extractor.max_file_mb out-of-range → SystemExit
  7. extractor as non-dict (string/list) → SystemExit
  8. Watcher off by default; opt-in respected
  9. graph: {enabled: true} on a pdf source is accepted but ignored
     with a stderr warning (no SystemExit)
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path


def _run_load(tmp: Path, source_block: dict, source_name: str = "docs"):
    """Write a config + return (loaded_or_None, stderr_text, exit_code|None)."""
    from lynx.config import load_config

    code_dir = tmp / "pdfs"
    code_dir.mkdir(exist_ok=True)
    # Inject path automatically when not provided
    if "path" not in source_block and source_block.get("type") == "pdf":
        source_block = {**source_block, "path": str(code_dir)}
    cfg = {
        "config_version": 2,
        "storage_path": str(tmp / "storage"),
        "sources": {source_name: source_block},
    }
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            loaded = load_config(cfg_path)
        return loaded, stderr.getvalue(), None
    except SystemExit as e:
        return None, stderr.getvalue(), e.code


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-pdf-config-"))
    print(f"[test] tempdir: {tmp}")
    try:
        # ============================================================
        # 1. Minimal valid config — defaults applied
        # ============================================================
        loaded, _err, _ec = _run_load(tmp, {"type": "pdf"})
        if loaded is None:
            print(f"[test] FAIL [1/9]: minimal valid config rejected")
            return 1
        s = loaded.sources["docs"]
        if s["type"] != "pdf":
            print(f"[test] FAIL [1/9]: type wrong: {s['type']}")
            return 1
        # Defaults
        if s["recursive"] is not True:
            print(f"[test] FAIL [1/9]: recursive default should be True, got {s['recursive']}")
            return 1
        if s["file_glob"] != "**/*.pdf":
            print(f"[test] FAIL [1/9]: file_glob default wrong: {s['file_glob']}")
            return 1
        ext = s["extractor"]
        if ext["backend"] != "auto":
            print(f"[test] FAIL [1/9]: extractor.backend default wrong: {ext['backend']}")
            return 1
        if ext["max_file_mb"] != 100:
            print(f"[test] FAIL [1/9]: extractor.max_file_mb default wrong: {ext['max_file_mb']}")
            return 1
        if ext["max_pages_per_file"] != 5000:
            print(f"[test] FAIL [1/9]: extractor.max_pages default wrong: {ext['max_pages_per_file']}")
            return 1
        if ext["skip_password_protected"] is not True:
            print(f"[test] FAIL [1/9]: skip_password_protected default wrong")
            return 1
        if s["watcher"]["enabled"] is not False:
            print(f"[test] FAIL [1/9]: watcher.enabled default should be False for pdf, got True")
            return 1
        print(f"[test] OK [1/9] minimal config: all defaults applied (watcher OFF, backend auto)")

        # ============================================================
        # 2. Full explicit config round-trips
        # ============================================================
        loaded, _, _ = _run_load(tmp, {
            "type": "pdf",
            "recursive": False,
            "file_glob": "manuals/*.pdf",
            "extractor": {
                "backend": "pypdf",
                "max_file_mb": 250,
                "max_pages_per_file": 2000,
                "skip_password_protected": False,
                "skip_if_text_empty": False,
            },
            "watcher": {"enabled": True, "debounce_seconds": 10.0},
        })
        if loaded is None:
            print(f"[test] FAIL [2/9]: full config rejected")
            return 2
        s = loaded.sources["docs"]
        if s["recursive"] is not False or s["file_glob"] != "manuals/*.pdf":
            print(f"[test] FAIL [2/9]: top-level overrides lost: {s}")
            return 2
        ext = s["extractor"]
        if ext["backend"] != "pypdf" or ext["max_file_mb"] != 250:
            print(f"[test] FAIL [2/9]: extractor overrides lost: {ext}")
            return 2
        if s["watcher"]["enabled"] is not True or s["watcher"]["debounce_seconds"] != 10.0:
            print(f"[test] FAIL [2/9]: watcher overrides lost: {s['watcher']}")
            return 2
        print(f"[test] OK [2/9] full config: all overrides preserved")

        # ============================================================
        # 3. Missing `path` → SystemExit
        # ============================================================
        # We pass an explicit empty cfg block, NOT going through the
        # path-inject helper, so 'path' really is missing.
        from lynx.config import load_config
        cfg_path = tmp / "missing_path.json"
        cfg_path.write_text(json.dumps({
            "config_version": 2, "storage_path": str(tmp / "s_missing"),
            "sources": {"docs": {"type": "pdf"}},  # no path
        }), encoding="utf-8")
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                load_config(cfg_path)
            print(f"[test] FAIL [3/9]: missing path should raise SystemExit")
            return 3
        except SystemExit:
            if "'path' is required" not in stderr.getvalue():
                print(f"[test] FAIL [3/9]: error message unclear: {stderr.getvalue()}")
                return 3
        print(f"[test] OK [3/9] missing path: clean SystemExit with helpful message")

        # ============================================================
        # 4. Non-existent path → SystemExit
        # ============================================================
        cfg_path = tmp / "bad_path.json"
        cfg_path.write_text(json.dumps({
            "config_version": 2, "storage_path": str(tmp / "s_bad"),
            "sources": {"docs": {"type": "pdf", "path": str(tmp / "does_not_exist")}},
        }), encoding="utf-8")
        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                load_config(cfg_path)
            print(f"[test] FAIL [4/9]: non-existent path should raise")
            return 4
        except SystemExit:
            err = stderr.getvalue()
            if "does not exist" not in err and "not a directory" not in err:
                print(f"[test] FAIL [4/9]: error message unclear: {err}")
                return 4
        print(f"[test] OK [4/9] non-existent path: SystemExit")

        # ============================================================
        # 5. Invalid extractor.backend → SystemExit
        # ============================================================
        loaded, err, ec = _run_load(tmp, {
            "type": "pdf",
            "extractor": {"backend": "wrong_backend"},
        })
        if loaded is not None:
            print(f"[test] FAIL [5/9]: invalid backend accepted")
            return 5
        if "must be one of" not in err.lower():
            print(f"[test] FAIL [5/9]: error message unclear: {err}")
            return 5
        print(f"[test] OK [5/9] invalid backend: SystemExit with enum list")

        # ============================================================
        # 6. max_file_mb out of range
        # ============================================================
        loaded, err, ec = _run_load(tmp, {
            "type": "pdf",
            "extractor": {"max_file_mb": 99999},  # > 2000 cap
        })
        if loaded is not None:
            print(f"[test] FAIL [6/9]: out-of-range max_file_mb accepted")
            return 6
        if "must be <=" not in err:
            print(f"[test] FAIL [6/9]: error message unclear: {err}")
            return 6
        # And below min
        loaded, err, ec = _run_load(tmp, {
            "type": "pdf",
            "extractor": {"max_file_mb": 0},
        })
        if loaded is not None:
            print(f"[test] FAIL [6/9]: max_file_mb=0 accepted (must be >=1)")
            return 6
        print(f"[test] OK [6/9] max_file_mb range check: both ends enforced")

        # ============================================================
        # 7. extractor as non-dict → SystemExit
        # ============================================================
        loaded, err, ec = _run_load(tmp, {
            "type": "pdf",
            "extractor": "auto",  # string instead of dict
        })
        if loaded is not None:
            print(f"[test] FAIL [7/9]: non-dict extractor accepted")
            return 7
        if "must be an object" not in err:
            print(f"[test] FAIL [7/9]: error message unclear: {err}")
            return 7
        print(f"[test] OK [7/9] non-dict extractor: SystemExit")

        # ============================================================
        # 8. Watcher off by default; opt-in respected (covered in 1+2)
        # ============================================================
        print(f"[test] OK [8/9] watcher default: OFF (covered by tests 1+2)")

        # ============================================================
        # 9. graph: {enabled: true} on pdf → ignored + warning, NOT error
        # ============================================================
        loaded, err, ec = _run_load(tmp, {
            "type": "pdf",
            "graph": {"enabled": True},
        })
        if loaded is None:
            print(f"[test] FAIL [9/9]: pdf+graph.enabled=true should be accepted (with warning), got rejection")
            return 9
        if "graph layer is only supported for type=codebase" not in err:
            print(f"[test] FAIL [9/9]: warning missing from stderr: {err!r}")
            return 9
        s = loaded.sources["docs"]
        if "graph" in s:
            # We don't propagate the graph block into the normalized pdf
            # source (it's purely codebase). Backend won't see it.
            print(f"[test] FAIL [9/9]: graph block leaked into normalized pdf source: {s}")
            return 9
        print(f"[test] OK [9/9] graph flag on pdf: accepted with warning, not propagated")

        print("\n[test] === SUCCESS: PDF config validation works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
