"""Integration tests for PdfBackend with CodebaseRAG stubbed out.

Same trick as test_graph_integration: stub `CodebaseRAG.__init__` so we
don't have to load the HuggingFace embedding model. Everything else
(extractor, dump writer, state cache, file discovery, force/incremental
diff) runs for real on synthesised PDFs.

Scenarios:
  1. update(force=True) extracts 3 PDFs → _dump/ populated, state written
  2. update(force=False) skips unchanged → no re-extract
  3. Modifying 1 PDF triggers only its re-extract
  4. Removing 1 PDF wipes its dump dir + state row
  5. Adding 1 PDF triggers extract for only the new file
  6. PDF with password → state row status=skipped_password, no dump
  7. status() aggregates pdf_count, skipped_count, total_pages
  8. recursive=False restricts to top-level only
  9. file_glob filter restricts to a sub-path
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _make_pdf(path: Path, pages_text: list, title: str = "") -> None:
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    if title:
        c.setTitle(title)
    for text in pages_text:
        c.drawString(100, 750, text); c.showPage()
    c.save()


def _make_password_pdf(path: Path, page_text: str, password: str = "x") -> None:
    from io import BytesIO
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader, PdfWriter
    buf = BytesIO()
    c = canvas.Canvas(buf); c.drawString(100, 750, page_text); c.showPage(); c.save()
    buf.seek(0)
    w = PdfWriter(clone_from=PdfReader(buf))
    w.encrypt(password)
    with open(path, "wb") as f:
        w.write(f)


def _stub_rag():
    """Replace CodebaseRAG.__init__ + a few methods with no-op stubs."""
    from lynx.rag_manager import CodebaseRAG

    class _StubColl:
        def count(self): return 0

    class _StubVS:
        def __init__(self): self._collection = _StubColl()

    def stub_init(self, **kw):
        self.codebase_path = Path(kw["codebase_path"])
        self.storage_path = Path(kw["rag_storage_path"])
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {"last_commit": None, "last_update": None}
        self.vector_store = _StubVS()
        # Track calls so tests can assert the backend triggered rag.update().
        if not hasattr(CodebaseRAG, "_test_calls"):
            CodebaseRAG._test_calls = []
        CodebaseRAG._test_calls.append(("init", kw["codebase_path"]))

    def stub_update(self, force=False):
        CodebaseRAG._test_calls.append(("update", force))

    def stub_update_file(self, path):
        CodebaseRAG._test_calls.append(("update_file", str(path)))

    def stub_remove_file(self, path):
        CodebaseRAG._test_calls.append(("remove_file", str(path)))

    CodebaseRAG.__init__ = stub_init
    CodebaseRAG.update = stub_update
    CodebaseRAG.update_file = stub_update_file
    CodebaseRAG.remove_file = stub_remove_file
    CodebaseRAG.check_config_drift = lambda self: None
    CodebaseRAG.drift_status_text = lambda self: "No drift."
    CodebaseRAG.needs_update = lambda self: False


def _make_backend(name, source_path, storage_dir, extractor_overrides=None, **src_overrides):
    """Construct a PdfBackend with sensible defaults for testing."""
    from types import SimpleNamespace
    from lynx.sources.pdf import PdfBackend

    shared = SimpleNamespace(
        embedding=SimpleNamespace(model_name="stub"),
        search=SimpleNamespace(
            mode="hybrid", rrf_k=60, candidate_pool_size=30,
            deep=SimpleNamespace(score_thresholds={"hybrid": 0.012}),
            reranker=SimpleNamespace(enabled=False, model_name="", top_n_before_rerank=30),
        ),
    )
    extractor = {
        "backend": "pypdf",
        "max_file_mb": 100,
        "max_pages_per_file": 5000,
        "skip_password_protected": True,
        "skip_if_text_empty": True,
    }
    if extractor_overrides:
        extractor.update(extractor_overrides)
    cfg = {
        "type": "pdf",
        "path": source_path,
        "recursive": True,
        "file_glob": "**/*.pdf",
        "extractor": extractor,
        "watcher": {"enabled": False},
        **src_overrides,
    }
    return PdfBackend(name=name, source_config=cfg, shared_config=shared, storage_dir=storage_dir)


def _state_for(backend):
    """Reload state from disk and return the dict {abs_path: PdfStateEntry}."""
    from lynx.sources.pdf_dump import load_state
    return load_state(backend.state_file)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lynx-pdf-backend-"))
    print(f"[test] tempdir: {tmp}")
    try:
        _stub_rag()
        src = tmp / "src"
        src.mkdir()

        # Create 3 PDFs with distinct content (>100 chars each)
        _make_pdf(src / "a.pdf", [
            "This is the alpha document, page one, with plenty of text to clear the empty-text threshold easily.",
            "This is alpha page two, still long enough to clear the threshold without ambiguity.",
        ])
        _make_pdf(src / "b.pdf", [
            "The beta document has only this single page, but the page is long enough that nothing trips the empty heuristic.",
        ])
        sub = src / "sub"; sub.mkdir()
        _make_pdf(sub / "c.pdf", [
            "Sub-folder file c, page one, padded with extra content to clear the empty-text threshold without issue.",
            "Sub-folder file c, page two, also padded with enough text to clear the empty threshold easily.",
            "Sub-folder file c, page three, similarly padded to provide enough characters above the threshold.",
        ])

        # ============================================================
        # 1. update(force=True) on fresh state extracts all 3 PDFs
        # ============================================================
        b = _make_backend("docs", str(src), tmp / "storage_1")
        # Bootstrap runs in __init__ when state is empty AND PDFs exist
        # → no need to call update() explicitly here.
        state = _state_for(b)
        if len(state) != 3:
            print(f"[test] FAIL [1/9]: expected 3 state rows, got {len(state)}")
            return 1
        for p, e in state.items():
            if e.status != "ok":
                print(f"[test] FAIL [1/9]: {p} status={e.status}, error={e.error}")
                return 1
        # Dump files: a has 2 pages, b has 1, c has 3 → 6 .md total
        dump_files = list((b.dump_dir).rglob("page_*.md"))
        if len(dump_files) != 6:
            print(f"[test] FAIL [1/9]: expected 6 dump files, got {len(dump_files)}: {dump_files}")
            return 1
        print(f"[test] OK [1/9] bootstrap: 3 PDFs extracted, 6 page dumps, state OK")

        # ============================================================
        # 2. update(force=False) skips unchanged
        # ============================================================
        from lynx.rag_manager import CodebaseRAG
        before = len(CodebaseRAG._test_calls)
        b.update(force=False)
        # No new dump files should be written; mtimes unchanged
        dump_files_after = list((b.dump_dir).rglob("page_*.md"))
        if len(dump_files_after) != 6:
            print(f"[test] FAIL [2/9]: dump count changed: {len(dump_files_after)}")
            return 2
        # State entries unchanged
        state2 = _state_for(b)
        if len(state2) != 3:
            print(f"[test] FAIL [2/9]: state count changed: {len(state2)}")
            return 2
        # rag.update was still called once (cheap no-op when nothing changed)
        if len(CodebaseRAG._test_calls) <= before:
            print(f"[test] FAIL [2/9]: rag.update not called")
            return 2
        print(f"[test] OK [2/9] SHA cache: re-update skips unchanged PDFs")

        # ============================================================
        # 3. Modify 1 PDF → only that one re-extracts
        # ============================================================
        # Wait long enough for filesystem mtime granularity (1s on some FS)
        time.sleep(0.05)
        _make_pdf(src / "b.pdf", [
            "Beta document REWRITTEN content, with plenty of text once again to clear the empty-text threshold.",
            "And a brand new second page, also long enough to clear the threshold without any ambiguity.",
        ])
        b.update(force=False)
        state3 = _state_for(b)
        # b now has 2 pages, a still 2, c still 3 → 7 .md total
        dump3 = list((b.dump_dir).rglob("page_*.md"))
        if len(dump3) != 7:
            print(f"[test] FAIL [3/9]: expected 7 dump files after b edit, got {len(dump3)}")
            return 3
        b_entry = state3[os.path.realpath(str(src / "b.pdf"))]
        if b_entry.n_pages != 2:
            print(f"[test] FAIL [3/9]: b should now have 2 pages, got {b_entry.n_pages}")
            return 3
        print(f"[test] OK [3/9] update_file: b.pdf modified → re-extracted (2 pages)")

        # ============================================================
        # 4. Remove 1 PDF → its dump dir + state row are wiped
        # ============================================================
        (src / "a.pdf").unlink()
        b.update(force=False)
        state4 = _state_for(b)
        a_key = os.path.realpath(str(src / "a.pdf"))
        if a_key in state4:
            print(f"[test] FAIL [4/9]: a.pdf still in state after removal: {a_key}")
            return 4
        if (b.dump_dir / "a").exists():
            print(f"[test] FAIL [4/9]: a/ dump dir still present after removal")
            return 4
        dump4 = list((b.dump_dir).rglob("page_*.md"))
        if len(dump4) != 5:  # b=2 + c=3
            print(f"[test] FAIL [4/9]: expected 5 dump files after a removed, got {len(dump4)}")
            return 4
        print(f"[test] OK [4/9] remove_file: a.pdf wiped + state cleaned")

        # ============================================================
        # 5. Add 1 PDF → it gets extracted, others untouched
        # ============================================================
        _make_pdf(src / "d.pdf", [
            "Delta document, freshly added later in the test, with text long enough to comfortably clear the empty-text threshold check used by the extractor.",
        ])
        b.update(force=False)
        state5 = _state_for(b)
        d_key = os.path.realpath(str(src / "d.pdf"))
        if d_key not in state5:
            print(f"[test] FAIL [5/9]: d.pdf not in state after add")
            return 5
        if state5[d_key].n_pages != 1:
            print(f"[test] FAIL [5/9]: d.pdf wrong page count: {state5[d_key].n_pages}")
            return 5
        print(f"[test] OK [5/9] new PDF picked up incrementally")

        # ============================================================
        # 6. Password-protected PDF → status=skipped_password
        # ============================================================
        _make_password_pdf(src / "locked.pdf", "secret content")
        b.update(force=False)
        state6 = _state_for(b)
        locked_key = os.path.realpath(str(src / "locked.pdf"))
        if locked_key not in state6:
            print(f"[test] FAIL [6/9]: locked.pdf missing from state")
            return 6
        if state6[locked_key].status != "skipped_password":
            print(f"[test] FAIL [6/9]: expected skipped_password, got {state6[locked_key].status}")
            return 6
        # No dump dir for locked
        if any((b.dump_dir).rglob("locked/page_*.md")):
            print(f"[test] FAIL [6/9]: locked.pdf produced dump files (should not)")
            return 6
        print(f"[test] OK [6/9] password-protected: status=skipped_password, no dump")

        # ============================================================
        # 7. status() aggregates correctly
        # ============================================================
        st = b.status()
        if st["type"] != "pdf":
            print(f"[test] FAIL [7/9]: status.type wrong: {st['type']}")
            return 7
        # 4 ok PDFs (b, c, d, and ... wait, a was removed; we have b/c/d plus locked):
        # b(2) + c(3) + d(1) = 3 ok. locked is skipped.
        if st["pdf_count"] != 3:
            print(f"[test] FAIL [7/9]: pdf_count should be 3, got {st['pdf_count']}")
            return 7
        if st["skipped_count"] != 1:
            print(f"[test] FAIL [7/9]: skipped_count should be 1, got {st['skipped_count']}")
            return 7
        if st["total_pages_extracted"] != 6:  # b=2, c=3, d=1
            print(f"[test] FAIL [7/9]: total_pages_extracted should be 6, got {st['total_pages_extracted']}")
            return 7
        if st["extractor_backend"] != "pypdf":
            print(f"[test] FAIL [7/9]: extractor_backend wrong: {st['extractor_backend']}")
            return 7
        print(f"[test] OK [7/9] status: pdf_count=3, skipped=1, total_pages=6")

        # ============================================================
        # 8. recursive=False restricts to top-level only
        # ============================================================
        flat = _make_backend(
            "docs_flat", str(src), tmp / "storage_8",
            recursive=False,
        )
        # Bootstrap ran with recursive=False, so c.pdf in sub/ must NOT appear
        state_flat = _state_for(flat)
        keys_flat = list(state_flat.keys())
        if any("sub/c.pdf" in k.replace("\\", "/") for k in keys_flat):
            print(f"[test] FAIL [8/9]: recursive=False included sub/c.pdf: {keys_flat}")
            return 8
        # b.pdf at top-level must be present
        b_at_root = os.path.realpath(str(src / "b.pdf"))
        if b_at_root not in state_flat:
            print(f"[test] FAIL [8/9]: recursive=False missed top-level b.pdf")
            return 8
        print(f"[test] OK [8/9] recursive=False: top-level only ({len(state_flat)} PDFs)")

        # ============================================================
        # 9. file_glob filter
        # ============================================================
        only_b = _make_backend(
            "docs_b", str(src), tmp / "storage_9",
            file_glob="b.pdf",
        )
        state_b = _state_for(only_b)
        keys_b = list(state_b.keys())
        if len(keys_b) != 1 or not keys_b[0].endswith("b.pdf"):
            print(f"[test] FAIL [9/9]: file_glob='b.pdf' should match only b.pdf, got {keys_b}")
            return 9
        print(f"[test] OK [9/9] file_glob filter: matched only b.pdf")

        print("\n[test] === SUCCESS: PdfBackend works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
