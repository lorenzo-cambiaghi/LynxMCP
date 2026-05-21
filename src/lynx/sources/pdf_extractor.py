"""Stand-alone PDF text extractor.

Pure function, no I/O on the source path beyond opening the PDF, no
state, no global env. Easy to test in isolation.

Two backends:
  - **pypdf** (default, pure Python, MIT) — coherent with Lynx's
    "100% local, zero system binaries" promise.
  - **pymupdf** (opt-in via the `[pdf-fast]` extra, AGPL) — faster and
    more accurate on multi-column layouts. Loaded only if importable.

What is intentionally NOT supported:
  - **OCR** of scanned PDFs (no Tesseract dep). Scanned files are
    detected (very low extracted text length) and reported as
    `status="skipped_empty"` so the caller can warn the user.
  - **Password-protected PDFs**. Detected and reported as
    `status="skipped_password"`. Adding a password store / keychain
    integration is out of scope for v0.7.
  - **Structured table / equation extraction**. Tables come out as
    flattened text (still searchable), equations rasterized in the PDF
    are simply lost. For richer extraction users can run a dedicated
    tool upstream and feed the resulting text via the `codebase` source.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    """Outcome of attempting to extract text from one PDF.

    `status` discriminates the outcome:
      - "ok"                      — `pages` populated, ready to dump.
      - "skipped_password"        — encrypted PDF, no password handling.
      - "skipped_too_large"       — file size > max_file_mb.
      - "skipped_too_many_pages"  — page count > max_pages.
      - "skipped_empty"           — total extracted text < empty_threshold;
                                    most likely a scanned PDF.
      - "error"                   — any other extraction failure; see
                                    `error` for details.

    Even when status != "ok" the dataclass carries useful diagnostics
    (page count when known, error message) so the caller can persist a
    state entry for the skipped file and avoid re-trying it next run.
    """
    status: str
    pages: list = field(default_factory=list)  # list[tuple[int, str]] — (page_num_1based, text)
    total_pages: int = 0
    n_chars: int = 0
    title: Optional[str] = None
    author: Optional[str] = None
    backend_used: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _try_import_pymupdf():
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        return None


def _try_import_pypdf():
    try:
        import pypdf
        return pypdf
    except ImportError:
        return None


def _select_backend(name: str) -> str:
    """Return the concrete backend name actually available.

    "auto" prefers pymupdf when installed, otherwise pypdf. Explicit
    "pymupdf" or "pypdf" raise ImportError-equivalents with a clear hint
    on how to install the missing one.
    """
    if name == "auto":
        if _try_import_pymupdf() is not None:
            return "pymupdf"
        if _try_import_pypdf() is not None:
            return "pypdf"
        raise RuntimeError(
            "PDF extractor: neither pymupdf nor pypdf is installed. "
            "Install pypdf with `pip install pypdf` (or `pip install lynx[pdf-fast]` for pymupdf)."
        )
    if name == "pymupdf":
        if _try_import_pymupdf() is None:
            raise RuntimeError(
                "PDF extractor: backend='pymupdf' requested but the package is not installed. "
                "Install with `pip install pymupdf` (or `pip install lynx[pdf-fast]`)."
            )
        return "pymupdf"
    if name == "pypdf":
        if _try_import_pypdf() is None:
            raise RuntimeError(
                "PDF extractor: backend='pypdf' requested but the package is not installed. "
                "Install with `pip install pypdf`."
            )
        return "pypdf"
    raise ValueError(f"PDF extractor: unknown backend {name!r}")


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Minimum total extracted characters below which we treat the file as
# "probably scanned" and skip it. 100 chars is roughly the length of a
# title page header — anything below that is almost certainly image-only.
# Configurable per-source via `skip_if_text_empty` (this constant is the
# threshold, the flag toggles whether we apply it).
DEFAULT_EMPTY_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Extractor implementations
# ---------------------------------------------------------------------------


def _extract_with_pypdf(abs_path: str) -> tuple:
    """Return (pages, total_pages, title, author, n_chars, password_protected).

    `password_protected` is True if the PDF was encrypted; in that case
    pages is empty and the caller emits `status="skipped_password"`.
    Other exceptions propagate to the caller for "error" classification.
    """
    from pypdf import PdfReader
    from pypdf.errors import FileNotDecryptedError, WrongPasswordError

    reader = PdfReader(abs_path)
    if reader.is_encrypted:
        # Try empty password (some PDFs are "encrypted" with no password,
        # used only to set permission flags). If that fails, give up.
        try:
            reader.decrypt("")
        except Exception:
            return ([], 0, None, None, 0, True)
        if reader.is_encrypted:
            return ([], 0, None, None, 0, True)

    total_pages = len(reader.pages)
    title = None
    author = None
    if reader.metadata is not None:
        try:
            title = reader.metadata.title or None
            author = reader.metadata.author or None
        except Exception:
            pass  # malformed metadata — ignore

    pages: list = []
    n_chars = 0
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (FileNotDecryptedError, WrongPasswordError):
            # Per-page decryption guard (rare, but happens with some PDFs).
            return ([], total_pages, title, author, 0, True)
        except Exception:
            # Single-page extraction failure — skip the page but keep going.
            # Common cause: a font without a /ToUnicode CMap. We don't want
            # one bad page to fail the whole document.
            text = ""
        text = text.strip()
        if text:
            pages.append((idx, text))
            n_chars += len(text)
    return (pages, total_pages, title, author, n_chars, False)


def _extract_with_pymupdf(abs_path: str) -> tuple:
    """Same return shape as `_extract_with_pypdf`, using PyMuPDF.

    PyMuPDF gives noticeably better text reading order on multi-column
    layouts (academic papers, two-column manuals) but is AGPL — hence
    opt-in via the `[pdf-fast]` extra rather than a hard dep.
    """
    import pymupdf

    doc = pymupdf.open(abs_path)
    try:
        if doc.needs_pass:
            # Try empty password (same logic as pypdf path).
            if not doc.authenticate(""):
                return ([], 0, None, None, 0, True)

        total_pages = doc.page_count
        meta = doc.metadata or {}
        title = (meta.get("title") or None) or None
        author = (meta.get("author") or None) or None

        pages: list = []
        n_chars = 0
        for idx in range(total_pages):
            try:
                page = doc.load_page(idx)
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            text = text.strip()
            if text:
                pages.append((idx + 1, text))
                n_chars += len(text)
        return (pages, total_pages, title, author, n_chars, False)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_pdf(
    abs_path: str,
    *,
    max_file_mb: int = 100,
    max_pages: int = 5000,
    backend: str = "auto",
    skip_if_text_empty: bool = True,
    empty_threshold: int = DEFAULT_EMPTY_THRESHOLD,
) -> ExtractResult:
    """Extract text from one PDF. Always returns an ExtractResult; never raises
    on a single file (callers iterate over many).

    Args:
        abs_path: Path to the .pdf file on disk.
        max_file_mb: Skip files larger than this (RAM-safety; pypdf loads
            the whole file in memory at ~3-5× disk size).
        max_pages: Skip files with more pages than this (defensive against
            pathological cases like 50k-page legal documents).
        backend: "auto" (default), "pypdf", or "pymupdf".
        skip_if_text_empty: When True (default), files with less than
            `empty_threshold` total extracted characters get
            `status="skipped_empty"` (most likely scanned PDFs).
        empty_threshold: See above.
    """
    p = Path(abs_path)
    backend_used = ""

    # Pre-checks (cheap, no library import).
    if not p.is_file():
        return ExtractResult(status="error", error=f"not a file: {abs_path}", backend_used=backend_used)
    try:
        size_mb = p.stat().st_size / (1024 * 1024)
    except OSError as e:
        return ExtractResult(status="error", error=f"stat failed: {e}", backend_used=backend_used)
    if size_mb > max_file_mb:
        return ExtractResult(
            status="skipped_too_large",
            backend_used=backend_used,
            error=f"file size {size_mb:.1f} MB exceeds max_file_mb={max_file_mb}",
        )

    # Pick the backend now so the error message is clearer than the one
    # thrown by the extraction function itself. We catch both:
    #   - RuntimeError  → the requested backend isn't installed
    #   - ValueError    → an invalid backend name like "doesnotexist"
    try:
        backend_used = _select_backend(backend)
    except (RuntimeError, ValueError) as e:
        return ExtractResult(status="error", backend_used=backend, error=str(e))

    try:
        if backend_used == "pymupdf":
            pages, total_pages, title, author, n_chars, locked = _extract_with_pymupdf(str(p))
        else:
            pages, total_pages, title, author, n_chars, locked = _extract_with_pypdf(str(p))
    except Exception as e:
        # Catch ANY remaining exception (malformed PDF, library bug, etc.)
        # so a single bad file never aborts a multi-file build.
        return ExtractResult(
            status="error",
            backend_used=backend_used,
            error=f"{type(e).__name__}: {e}",
        )

    if locked:
        return ExtractResult(
            status="skipped_password",
            backend_used=backend_used,
            total_pages=total_pages,
        )
    if total_pages > max_pages:
        return ExtractResult(
            status="skipped_too_many_pages",
            backend_used=backend_used,
            total_pages=total_pages,
            error=f"page count {total_pages} exceeds max_pages={max_pages}",
        )
    if skip_if_text_empty and n_chars < empty_threshold:
        return ExtractResult(
            status="skipped_empty",
            backend_used=backend_used,
            total_pages=total_pages,
            n_chars=n_chars,
            title=title, author=author,
            error=f"only {n_chars} chars extracted (< {empty_threshold}); likely scanned",
        )

    return ExtractResult(
        status="ok",
        pages=pages,
        total_pages=total_pages,
        n_chars=n_chars,
        title=title, author=author,
        backend_used=backend_used,
    )
