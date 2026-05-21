"""Unit tests for the PDF extractor.

No filesystem state beyond the per-test tempdir. PDFs are synthesised on
the fly with reportlab so the tests are deterministic and don't depend
on any third-party PDF file. No HuggingFace, no ChromaDB.

Scenarios:
  1. Single-page PDF — status=ok, 1 page, expected text
  2. Multi-page PDF — status=ok, total_pages matches, page numbers in order
  3. Password-protected PDF — status=skipped_password
  4. PDF with too many pages (max_pages cap) — status=skipped_too_many_pages
  5. Empty / image-only PDF (no extractable text) — status=skipped_empty
  6. Corrupted file (truncated bytes) — status=error
  7. Oversize file (size_mb > max_file_mb) — status=skipped_too_large
  8. Metadata extraction — title + author from PDF info dict
  9. Backend selection — "auto" uses pypdf when pymupdf absent;
     explicit "pymupdf" yields a clear error when not installed
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path


# ---------------------------------------------------------------------------
# PDF fixture helpers (reportlab)
# ---------------------------------------------------------------------------


def _write_pdf(path: Path, pages_text: list, title: str = "", author: str = "") -> None:
    """Write a PDF to `path` with one page per item in `pages_text`."""
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    if title:
        c.setTitle(title)
    if author:
        c.setAuthor(author)
    for text in pages_text:
        # Single line per page is enough for the assertions.
        c.drawString(100, 750, text)
        c.showPage()
    c.save()


def _write_password_protected_pdf(path: Path, page_text: str, password: str = "sekret") -> None:
    """Write a real encrypted PDF (round-trip via pypdf to add encryption)."""
    from pypdf import PdfReader, PdfWriter
    buf = BytesIO()
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(buf)
    c.drawString(100, 750, page_text); c.showPage(); c.save()
    buf.seek(0)
    w = PdfWriter(clone_from=PdfReader(buf))
    w.encrypt(password)
    with open(path, "wb") as f:
        w.write(f)


def _write_empty_image_only_pdf(path: Path) -> None:
    """Write a PDF with no text content at all. Mimics a scanned page
    (no extractable text). We just emit a blank page."""
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path))
    c.showPage()  # blank
    c.save()


def main() -> int:
    from lynx.sources.pdf_extractor import extract_pdf, _select_backend

    tmp = Path(tempfile.mkdtemp(prefix="lynx-pdf-extract-"))
    print(f"[test] tempdir: {tmp}")
    try:
        # ========================================================
        # 1. Single-page PDF
        # ========================================================
        p1 = tmp / "single.pdf"
        _write_pdf(p1, ["Hello world from page one. " * 5])
        r = extract_pdf(str(p1))
        if r.status != "ok":
            print(f"[test] FAIL [1/9]: status={r.status}, error={r.error}")
            return 1
        if r.total_pages != 1 or len(r.pages) != 1:
            print(f"[test] FAIL [1/9]: total_pages={r.total_pages}, n_pages={len(r.pages)}")
            return 1
        page_num, page_text = r.pages[0]
        if page_num != 1 or "Hello world" not in page_text:
            print(f"[test] FAIL [1/9]: wrong page content: ({page_num}, {page_text!r})")
            return 1
        print(f"[test] OK [1/9] single-page: status=ok, n_chars={r.n_chars}")

        # ========================================================
        # 2. Multi-page PDF
        # ========================================================
        p2 = tmp / "multi.pdf"
        _write_pdf(p2, [f"Page number {i} content line " * 5 for i in range(1, 6)])
        r = extract_pdf(str(p2))
        if r.status != "ok":
            print(f"[test] FAIL [2/9]: status={r.status}, error={r.error}")
            return 2
        if r.total_pages != 5 or len(r.pages) != 5:
            print(f"[test] FAIL [2/9]: expected 5 pages, got total={r.total_pages} extracted={len(r.pages)}")
            return 2
        page_nums = [pn for pn, _ in r.pages]
        if page_nums != [1, 2, 3, 4, 5]:
            print(f"[test] FAIL [2/9]: page numbers out of order: {page_nums}")
            return 2
        # Each page should reference its own number
        for pn, ptext in r.pages:
            if f"Page number {pn}" not in ptext:
                print(f"[test] FAIL [2/9]: page {pn} content doesn't contain its marker: {ptext[:80]!r}")
                return 2
        print(f"[test] OK [2/9] multi-page: 5 pages in order, content cross-checks")

        # ========================================================
        # 3. Password-protected PDF
        # ========================================================
        p3 = tmp / "locked.pdf"
        _write_password_protected_pdf(p3, "secret content")
        r = extract_pdf(str(p3))
        if r.status != "skipped_password":
            print(f"[test] FAIL [3/9]: expected skipped_password, got status={r.status}, error={r.error}")
            return 3
        if r.pages:
            print(f"[test] FAIL [3/9]: pages should be empty when locked: {r.pages}")
            return 3
        print(f"[test] OK [3/9] password-protected: skipped_password")

        # ========================================================
        # 4. Too many pages
        # ========================================================
        p4 = tmp / "many.pdf"
        _write_pdf(p4, [f"P{i}" * 50 for i in range(1, 11)])  # 10 pages
        r = extract_pdf(str(p4), max_pages=5)
        if r.status != "skipped_too_many_pages":
            print(f"[test] FAIL [4/9]: expected skipped_too_many_pages, got status={r.status}")
            return 4
        if r.total_pages != 10:
            print(f"[test] FAIL [4/9]: total_pages should still be reported: {r.total_pages}")
            return 4
        print(f"[test] OK [4/9] max_pages cap: skipped_too_many_pages (total={r.total_pages})")

        # ========================================================
        # 5. Empty / image-only PDF (scanned simulation)
        # ========================================================
        p5 = tmp / "empty.pdf"
        _write_empty_image_only_pdf(p5)
        r = extract_pdf(str(p5))
        if r.status != "skipped_empty":
            print(f"[test] FAIL [5/9]: expected skipped_empty, got status={r.status}, n_chars={r.n_chars}")
            return 5
        # With skip_if_text_empty=False, same file should now return "ok" (just empty pages)
        r2 = extract_pdf(str(p5), skip_if_text_empty=False)
        if r2.status != "ok":
            print(f"[test] FAIL [5/9]: with skip_if_text_empty=False, got status={r2.status}")
            return 5
        print(f"[test] OK [5/9] empty pdf: skipped_empty (and ok when flag off)")

        # ========================================================
        # 6. Corrupted file (truncated bytes)
        # ========================================================
        p6 = tmp / "corrupt.pdf"
        p6.write_bytes(b"%PDF-1.4\n%garbage data that is not a real pdf\n")
        r = extract_pdf(str(p6))
        if r.status != "error":
            print(f"[test] FAIL [6/9]: expected error, got status={r.status}, error={r.error}")
            return 6
        if not r.error:
            print(f"[test] FAIL [6/9]: error message empty: {r}")
            return 6
        print(f"[test] OK [6/9] corrupt pdf: error reported ({r.error[:50]}...)")

        # ========================================================
        # 7. Oversize file (cap)
        # ========================================================
        p7 = tmp / "small.pdf"
        # Payload >100 char so the empty-text heuristic doesn't fire when we
        # check the "extracts fine" half of the test.
        _write_pdf(p7, ["This is plenty of text on a single page to clear the empty-text threshold easily, repeat repeat repeat."])
        # max_file_mb=0 means anything > 0 MB is skipped
        r = extract_pdf(str(p7), max_file_mb=0)
        if r.status != "skipped_too_large":
            print(f"[test] FAIL [7/9]: expected skipped_too_large with max_file_mb=0, got {r.status}")
            return 7
        # And with a sensible cap the same file extracts fine
        r2 = extract_pdf(str(p7), max_file_mb=100)
        if r2.status != "ok":
            print(f"[test] FAIL [7/9]: PDF should extract with max_file_mb=100, got {r2.status} (n_chars={r2.n_chars})")
            return 7
        print(f"[test] OK [7/9] max_file_mb cap: skipped when too big, ok when in budget")

        # ========================================================
        # 8. Metadata extraction
        # ========================================================
        p8 = tmp / "with_meta.pdf"
        _write_pdf(
            p8,
            ["This page is intentionally very long so it easily clears the 100 char empty-text threshold; the test focuses on title and author extraction from the PDF info dict."],
            title="The Meaning of Life",
            author="D. Adams",
        )
        r = extract_pdf(str(p8))
        if r.status != "ok":
            print(f"[test] FAIL [8/9]: status={r.status}")
            return 8
        if r.title != "The Meaning of Life":
            print(f"[test] FAIL [8/9]: title wrong: {r.title!r}")
            return 8
        if r.author != "D. Adams":
            print(f"[test] FAIL [8/9]: author wrong: {r.author!r}")
            return 8
        print(f"[test] OK [8/9] metadata: title={r.title!r}, author={r.author!r}")

        # ========================================================
        # 9. Backend selection
        # ========================================================
        # Without pymupdf installed: auto should resolve to pypdf
        backend_auto = _select_backend("auto")
        if backend_auto not in ("pypdf", "pymupdf"):
            print(f"[test] FAIL [9/9]: auto returned unexpected backend: {backend_auto}")
            return 9
        # Explicit pypdf works
        backend_pypdf = _select_backend("pypdf")
        if backend_pypdf != "pypdf":
            print(f"[test] FAIL [9/9]: explicit pypdf returned: {backend_pypdf}")
            return 9
        # Explicit pymupdf: only check it doesn't crash on a clear error path.
        # On systems where pymupdf IS installed it should return "pymupdf".
        try:
            backend_pymupdf = _select_backend("pymupdf")
            assert backend_pymupdf == "pymupdf"
        except RuntimeError as e:
            if "pymupdf" not in str(e).lower():
                print(f"[test] FAIL [9/9]: explicit pymupdf error message unclear: {e}")
                return 9
        # Bogus backend raises ValueError
        try:
            _select_backend("doesnotexist")
            print(f"[test] FAIL [9/9]: unknown backend should raise")
            return 9
        except ValueError:
            pass
        # extract_pdf with explicit unsupported backend returns status=error
        r = extract_pdf(str(p1), backend="doesnotexist")
        if r.status != "error":
            print(f"[test] FAIL [9/9]: unknown backend in extract_pdf should produce status=error, got {r.status}")
            return 9
        print(f"[test] OK [9/9] backend selection: auto={backend_auto}, fallbacks/errors handled")

        print("\n[test] === SUCCESS: PDF extractor works as expected ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
