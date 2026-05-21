"""Unit tests for the PDF dump writer + extract state cache.

Scenarios:
  1. write_page_dump produces a .md with YAML frontmatter + body, named
     page_NNNN.md with 4-digit zero padding.
  2. Re-writing the same path is idempotent (overwrites cleanly).
  3. Sanitization of rel_path / pdf_stem: hostile chars get replaced.
  4. wipe_pdf_dump removes only the target subdirectory, prunes empty parents.
  5. save_state_atomic / load_state round-trip preserves data.
  6. Corrupt JSON in state file → load_state returns {} (forces rebuild).
  7. Schema mismatch in state file → load_state returns {}.
  8. Empty body still writes a valid file (frontmatter + blank body).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def main() -> int:
    from lynx.sources.pdf_dump import (
        write_page_dump,
        wipe_pdf_dump,
        load_state,
        save_state_atomic,
        page_dump_dir,
        page_dump_path,
        PdfStateEntry,
        PDF_EXTRACT_SCHEMA_VERSION,
    )

    tmp = Path(tempfile.mkdtemp(prefix="lynx-pdf-dump-"))
    print(f"[test] tempdir: {tmp}")
    try:
        dump_root = tmp / "_dump"

        # ========================================================
        # 1. write_page_dump: file shape + frontmatter
        # ========================================================
        out_path = write_page_dump(
            dump_root=dump_root,
            rel_path="manuals/v2",
            pdf_stem="user_guide",
            page_num=42,
            total_pages=300,
            pdf_abs_path="/data/pdfs/manuals/v2/user_guide.pdf",
            text="Section 4.2: introduction to widgets.\nLine two.",
            extracted_at="2026-05-21T10:00:00",
            pdf_title="User Guide v2",
        )
        if out_path.name != "page_0042.md":
            print(f"[test] FAIL [1/8]: wrong filename: {out_path.name!r}")
            return 1
        if not str(out_path).endswith("manuals/v2/user_guide/page_0042.md"):
            print(f"[test] FAIL [1/8]: wrong nested path: {out_path}")
            return 1
        content = out_path.read_text("utf-8")
        if not content.startswith("---\n"):
            print(f"[test] FAIL [1/8]: missing frontmatter delimiter at top: {content[:80]!r}")
            return 1
        for needle in ("page: 42", "total_pages: 300", "pdf_name: user_guide.pdf",
                       "Section 4.2"):
            if needle not in content:
                print(f"[test] FAIL [1/8]: expected substring missing: {needle!r}")
                return 1
        print(f"[test] OK [1/8] write_page_dump: {out_path.name}, frontmatter + body OK")

        # ========================================================
        # 2. Idempotent overwrite
        # ========================================================
        out_path_2 = write_page_dump(
            dump_root=dump_root,
            rel_path="manuals/v2",
            pdf_stem="user_guide",
            page_num=42,
            total_pages=300,
            pdf_abs_path="/data/pdfs/manuals/v2/user_guide.pdf",
            text="REPLACEMENT BODY",
            extracted_at="2026-05-22T11:00:00",
        )
        if out_path_2 != out_path:
            print(f"[test] FAIL [2/8]: idempotent path mismatch")
            return 2
        new_content = out_path.read_text("utf-8")
        if "Section 4.2" in new_content:
            print(f"[test] FAIL [2/8]: old body still present after overwrite")
            return 2
        if "REPLACEMENT BODY" not in new_content:
            print(f"[test] FAIL [2/8]: new body missing after overwrite")
            return 2
        print(f"[test] OK [2/8] re-write: overwrites cleanly")

        # ========================================================
        # 3. Sanitization
        # ========================================================
        # Filesystem-hostile chars: < > : " | ? * \ /
        hostile_path = write_page_dump(
            dump_root=dump_root,
            rel_path='weird:dir<sub>',
            pdf_stem='file"with*chars',
            page_num=1,
            total_pages=1,
            pdf_abs_path="/x/y.pdf",
            text="Sanitization payload that easily clears the empty threshold for downstream RAG indexing.",
            extracted_at="2026-05-21T10:00:00",
        )
        # No raw hostile char in the final path
        for bad in ('<', '>', ':', '"', '|', '?', '*'):
            if bad in hostile_path.name or bad in str(hostile_path.parent.relative_to(dump_root)):
                print(f"[test] FAIL [3/8]: hostile char {bad!r} survived in: {hostile_path}")
                return 3
        if not hostile_path.exists():
            print(f"[test] FAIL [3/8]: sanitized path not written: {hostile_path}")
            return 3
        print(f"[test] OK [3/8] sanitization: hostile chars replaced ({hostile_path.relative_to(dump_root)})")

        # ========================================================
        # 4. wipe_pdf_dump removes only target subdir, prunes empty parents
        # ========================================================
        # Create a second PDF dump alongside the first one in manuals/v2/
        sibling = write_page_dump(
            dump_root=dump_root,
            rel_path="manuals/v2",
            pdf_stem="release_notes",
            page_num=1,
            total_pages=1,
            pdf_abs_path="/data/release_notes.pdf",
            text="Release notes content that is long enough to be useful and not skipped by anything downstream.",
            extracted_at="2026-05-21T10:00:00",
        )
        # Also create another isolated PDF deeper in its own subtree
        isolated = write_page_dump(
            dump_root=dump_root,
            rel_path="archive/old",
            pdf_stem="legacy",
            page_num=1,
            total_pages=1,
            pdf_abs_path="/data/legacy.pdf",
            text="Legacy content for the isolated-prune test (over 100 chars to clear the empty-text threshold).",
            extracted_at="2026-05-21T10:00:00",
        )
        # Wipe user_guide: user_guide dir must go, sibling must survive,
        # parent manuals/v2 must remain (still has release_notes inside).
        n_removed = wipe_pdf_dump(dump_root, "manuals/v2", "user_guide")
        if n_removed < 1:
            print(f"[test] FAIL [4/8]: wipe reported {n_removed} files removed, expected >=1")
            return 4
        if (dump_root / "manuals" / "v2" / "user_guide").exists():
            print(f"[test] FAIL [4/8]: user_guide dir still exists after wipe")
            return 4
        if not sibling.exists():
            print(f"[test] FAIL [4/8]: sibling release_notes was wrongly removed")
            return 4
        # Wipe legacy: isolated parent tree archive/old/legacy should fully collapse
        wipe_pdf_dump(dump_root, "archive/old", "legacy")
        if (dump_root / "archive").exists():
            print(f"[test] FAIL [4/8]: empty parent 'archive' not pruned after wipe")
            return 4
        # Wiping a non-existent PDF dump is a no-op
        n_zero = wipe_pdf_dump(dump_root, "does/not/exist", "nada")
        if n_zero != 0:
            print(f"[test] FAIL [4/8]: wipe of missing dir reported {n_zero}, expected 0")
            return 4
        print(f"[test] OK [4/8] wipe_pdf_dump: targets only the requested PDF, prunes empty parents")

        # ========================================================
        # 5. save_state_atomic / load_state round-trip
        # ========================================================
        state_path = tmp / "_extract_state.json"
        state = {
            "/abs/a.pdf": PdfStateEntry(
                sha256="aaaa", n_pages=10, n_chars=5000, status="ok",
                extracted_at="2026-05-21T10:00:00",
                dump_rel="manuals", pdf_stem="a", error=None,
            ),
            "/abs/locked.pdf": PdfStateEntry(
                sha256="bbbb", n_pages=0, n_chars=0, status="skipped_password",
                extracted_at="2026-05-21T10:05:00",
                dump_rel="manuals", pdf_stem="locked",
                error="encrypted",
            ),
        }
        save_state_atomic(state_path, state)
        loaded = load_state(state_path)
        if set(loaded.keys()) != set(state.keys()):
            print(f"[test] FAIL [5/8]: keys mismatch: {set(loaded)} vs {set(state)}")
            return 5
        a = loaded["/abs/a.pdf"]
        if a.sha256 != "aaaa" or a.n_pages != 10 or a.status != "ok":
            print(f"[test] FAIL [5/8]: row a corrupted on round-trip: {a}")
            return 5
        b = loaded["/abs/locked.pdf"]
        if b.status != "skipped_password" or b.error != "encrypted":
            print(f"[test] FAIL [5/8]: row locked corrupted on round-trip: {b}")
            return 5
        # On-disk format: schema_version present + files dict
        raw = json.loads(state_path.read_text("utf-8"))
        if raw["schema_version"] != PDF_EXTRACT_SCHEMA_VERSION:
            print(f"[test] FAIL [5/8]: persisted schema_version wrong: {raw['schema_version']}")
            return 5
        print(f"[test] OK [5/8] state round-trip: 2 entries preserved, schema_version={raw['schema_version']}")

        # ========================================================
        # 6. Corrupt JSON in state file → empty dict
        # ========================================================
        state_path.write_text("{not valid json}", encoding="utf-8")
        loaded_bad = load_state(state_path)
        if loaded_bad != {}:
            print(f"[test] FAIL [6/8]: corrupt state should return {{}}, got {loaded_bad}")
            return 6
        # Missing file → also empty
        if load_state(tmp / "missing.json") != {}:
            print(f"[test] FAIL [6/8]: missing state file should return {{}}")
            return 6
        print(f"[test] OK [6/8] corrupt / missing state file: returns empty dict (forces rebuild)")

        # ========================================================
        # 7. Schema mismatch
        # ========================================================
        state_path.write_text(json.dumps({
            "schema_version": 99999,
            "files": {"/x": {"sha256": "x", "n_pages": 1, "n_chars": 1,
                              "status": "ok", "extracted_at": "...",
                              "dump_rel": "", "pdf_stem": "x", "error": None}},
        }), encoding="utf-8")
        loaded_old = load_state(state_path)
        if loaded_old != {}:
            print(f"[test] FAIL [7/8]: schema mismatch should reset state, got {loaded_old}")
            return 7
        print(f"[test] OK [7/8] schema mismatch: state reset")

        # ========================================================
        # 8. Empty body
        # ========================================================
        empty_path = write_page_dump(
            dump_root=dump_root,
            rel_path="edge",
            pdf_stem="blank",
            page_num=7,
            total_pages=7,
            pdf_abs_path="/data/blank.pdf",
            text="",
            extracted_at="2026-05-21T10:00:00",
        )
        ec = empty_path.read_text("utf-8")
        if not ec.startswith("---\n"):
            print(f"[test] FAIL [8/8]: empty-body file missing frontmatter: {ec[:80]!r}")
            return 8
        print(f"[test] OK [8/8] empty body: still writes a valid frontmatter-only file")

        print("\n[test] === SUCCESS: PDF dump writer + state cache work as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
