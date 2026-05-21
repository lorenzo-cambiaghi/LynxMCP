"""On-disk layout for PDF sources: per-page Markdown dump + extract state.

The PdfBackend produces one `.md` file per extracted page in
`<storage>/<source>/_dump/<rel_path>/<pdf_stem>/page_NNNN.md` and
records what was extracted in `_extract_state.json`. CodebaseRAG
(pointed at `_dump/`) then chunks, embeds and indexes those Markdown
files the same way it would for any other text source — the chunker
fallback `SentenceSplitter` handles them without modification.

Naming rationale: one file per page means

  - The chunker doesn't have to know about page breaks; each chunk
    inherits the page number from the file path (`page_0042.md`) AND
    from the YAML frontmatter at the top of the file.
  - Citations look natural: "User Manual.pdf p.42" instead of "chunk 7
    of User Manual.pdf".
  - 4-digit zero padding keeps the file listing sorted lexicographically
    up to 9999 pages — well beyond `max_pages_per_file` (5000 default).

Atomic writes via `path.tmp + os.replace` keep state consistent if
the process dies mid-write (the cache simply reverts to its previous
value on next load).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# Schema version of `_extract_state.json`. Bump when the per-PDF state
# dict shape changes incompatibly. The loader discards the cache on
# mismatch and the next `update(force=False)` re-extracts everything
# (preserving SHAs would risk binding old dump layouts to new state).
PDF_EXTRACT_SCHEMA_VERSION = 1

# Pad-width for page-number filenames. 4 digits supports up to 9999
# pages, comfortably above the default `max_pages_per_file=5000`.
_PAGE_PAD_WIDTH = 4

# Filename characters we definitely don't want in dump paths. macOS and
# Windows tolerate different sets — we intersect both to be safe.
_BAD_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_segment(name: str) -> str:
    """Replace filesystem-hostile characters in one path segment.

    Used for both the rel_path components and the pdf_stem. Empty input
    is replaced with `_` so we never create a zero-length directory.
    """
    cleaned = _BAD_FS_CHARS.sub("_", name)
    # Collapse runs of replacement chars + trim leading/trailing dots & spaces
    # (Windows treats trailing dots as ambiguous and silently strips them).
    cleaned = cleaned.strip(" .")
    return cleaned or "_"


def _safe_rel_segments(rel_path: str) -> list:
    """Split `rel_path` into path segments and sanitize each one.

    Empty string or "." → empty list (file at the top of _dump/).
    """
    if not rel_path or rel_path in (".", "./"):
        return []
    parts = Path(rel_path).parts
    return [_sanitize_segment(p) for p in parts if p not in ("", ".")]


# ---------------------------------------------------------------------------
# Per-page dump writer
# ---------------------------------------------------------------------------


def page_dump_dir(dump_root: Path, rel_path: str, pdf_stem: str) -> Path:
    """Return the directory that holds the per-page dumps for one PDF."""
    parts = _safe_rel_segments(rel_path) + [_sanitize_segment(pdf_stem)]
    return Path(dump_root, *parts) if parts else dump_root


def page_dump_path(dump_root: Path, rel_path: str, pdf_stem: str, page_num: int) -> Path:
    """Return the full path of the per-page Markdown dump."""
    return page_dump_dir(dump_root, rel_path, pdf_stem) / f"page_{page_num:0{_PAGE_PAD_WIDTH}d}.md"


def _format_frontmatter(meta: dict) -> str:
    """Inline mini-YAML for the frontmatter block.

    Values are coerced to strings and quoted only if they contain a
    YAML-relevant character. We don't need full PyYAML round-trip here
    — these blocks are read by humans and embedded in chunks by the
    sentence splitter, never re-parsed by Lynx.
    """
    def fmt_value(v) -> str:
        if v is None:
            return "null"
        s = str(v)
        if any(c in s for c in (":", "#", "'", '"', "\n", "\r")):
            return json.dumps(s)  # safe quoting via JSON
        return s

    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {fmt_value(v)}")
    lines.append("---")
    lines.append("")  # blank line before body
    return "\n".join(lines)


def write_page_dump(
    dump_root: Path,
    rel_path: str,
    pdf_stem: str,
    page_num: int,
    total_pages: int,
    pdf_abs_path: str,
    text: str,
    extracted_at: str,
    pdf_title: Optional[str] = None,
) -> Path:
    """Write one page's text to `_dump/<rel>/<stem>/page_NNNN.md`.

    Atomic via `path.tmp + os.replace` so a process kill mid-write never
    leaves a torn .md file that the chunker would later read as garbage.

    The frontmatter is also part of the chunk body — once the sentence
    splitter takes over, every chunk inherits `pdf_path`, `page`,
    `total_pages`, etc. in its text. This is enough for the AI client
    to cite the source precisely without us extending TextNode metadata.
    """
    target_dir = page_dump_dir(dump_root, rel_path, pdf_stem)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"page_{page_num:0{_PAGE_PAD_WIDTH}d}.md"

    frontmatter = _format_frontmatter({
        "pdf_path": pdf_abs_path,
        "pdf_name": Path(pdf_abs_path).name,
        "pdf_title": pdf_title or "",
        "page": page_num,
        "total_pages": total_pages,
        "extracted_at": extracted_at,
    })
    body = (text or "").rstrip() + "\n"

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(frontmatter + body, encoding="utf-8")
    os.replace(tmp, target)
    return target


def wipe_pdf_dump(dump_root: Path, rel_path: str, pdf_stem: str) -> int:
    """Remove the per-PDF subdirectory under `_dump/`.

    Returns the number of files removed (useful for the caller's stats).
    No-op if the directory doesn't exist. Errors propagate — callers
    decide whether to warn or swallow.
    """
    target_dir = page_dump_dir(dump_root, rel_path, pdf_stem)
    if not target_dir.exists():
        return 0
    n = sum(1 for _ in target_dir.rglob("*") if _.is_file())
    shutil.rmtree(target_dir)
    # Prune empty parents up to (but not including) dump_root, so we don't
    # leave a tree of empty rel_path directories behind.
    parent = target_dir.parent
    while parent != dump_root and parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break  # not empty
        parent = parent.parent
    return n


# ---------------------------------------------------------------------------
# Extract state (per-PDF cache)
# ---------------------------------------------------------------------------


@dataclass
class PdfStateEntry:
    """One row of `_extract_state.json`.

    Keyed by `pdf_abs_path`. Stores enough to (a) skip unchanged PDFs on
    incremental rebuilds (via SHA-256), (b) surface diagnostics in
    `lynx status`, (c) clean up dump dirs when the PDF is removed.
    """
    sha256: str
    n_pages: int
    n_chars: int
    status: str               # mirrors ExtractResult.status
    extracted_at: str
    dump_rel: str             # the rel_path passed at write time, for wipe_pdf_dump
    pdf_stem: str
    error: Optional[str] = None  # populated for skipped_* / error rows

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "n_pages": self.n_pages,
            "n_chars": self.n_chars,
            "status": self.status,
            "extracted_at": self.extracted_at,
            "dump_rel": self.dump_rel,
            "pdf_stem": self.pdf_stem,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PdfStateEntry":
        return cls(
            sha256=d.get("sha256", ""),
            n_pages=int(d.get("n_pages", 0)),
            n_chars=int(d.get("n_chars", 0)),
            status=str(d.get("status", "")),
            extracted_at=str(d.get("extracted_at", "")),
            dump_rel=str(d.get("dump_rel", "")),
            pdf_stem=str(d.get("pdf_stem", "")),
            error=d.get("error"),
        )


def load_state(path: Path) -> dict:
    """Load the persisted state dict (`{pdf_abs_path: PdfStateEntry}`).

    Returns an empty dict (forcing a full re-extract) when:
      - the file doesn't exist (fresh source),
      - the JSON is malformed,
      - the schema_version disagrees with `PDF_EXTRACT_SCHEMA_VERSION`.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _log(f"[pdf] state file unreadable ({e}); resetting cache")
        return {}
    if raw.get("schema_version") != PDF_EXTRACT_SCHEMA_VERSION:
        _log(
            f"[pdf] state schema_version mismatch "
            f"(got {raw.get('schema_version')!r}, expected {PDF_EXTRACT_SCHEMA_VERSION}); "
            f"resetting cache"
        )
        return {}
    files = raw.get("files") or {}
    return {k: PdfStateEntry.from_dict(v) for k, v in files.items()}


def save_state_atomic(path: Path, state: dict) -> None:
    """Persist the state dict. Atomic via `path.tmp + os.replace`."""
    payload = {
        "schema_version": PDF_EXTRACT_SCHEMA_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "files": {k: v.to_dict() for k, v in state.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
