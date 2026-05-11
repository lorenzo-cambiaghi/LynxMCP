"""Webdoc source backend: fetch a public docs site on demand, dump to disk,
index via the existing CodebaseRAG pipeline.

Why two layers? The fetch step is webdoc-specific (HTTP crawl + content
extraction). The indexing step (chunking, embedding, BM25, drift detection,
SHA cache, hybrid search) is exactly the same code that already drives the
codebase source. We get all of it for free by writing the dump to a folder
and pointing a CodebaseRAG at that folder. The fetch is the only new logic.

Fetch / re-fetch is **always explicit**: it runs only when the user invokes
`lynx build --source <name>` (or programmatically via update(force=True)).
No background refresh, no automatic re-crawl on a schedule. Detecting "is
this URL changed without downloading it" is unreliable in practice (ETag /
Last-Modified support varies by server), so we just don't try.

Supported docs site shapes: static HTML, mkdocs-built, docusaurus, sphinx,
anything that renders content server-side. JS-rendered SPAs require a
headless browser — explicitly out of scope (would pull in ~200MB of
Chromium). 90% of real-world technical docs are server-rendered.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from .base import SourceBackend


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard cap on per-URL response size (10 MB) to avoid pathological downloads
# from misconfigured servers. Real doc pages are tens to hundreds of KB.
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024

# Timeout per request. Documentation servers should be snappy; this is a
# liveness bound, not a performance one.
_REQUEST_TIMEOUT_SECONDS = 30.0

# Hard cap on the BFS frontier so a misconfigured filter can't run away on
# huge mirror sites. The user can raise this with `max_pages` in config.
_DEFAULT_MAX_PAGES = 500


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Strip fragment + normalize trailing slash inconsistencies."""
    url, _ = urldefrag(url)
    # Remove trailing slash for consistency (except for the bare root "/")
    if url.endswith("/") and urlparse(url).path != "/":
        url = url.rstrip("/")
    return url


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe local filename.

    Strategy: take the path, replace anything that isn't [A-Za-z0-9_-] with
    underscore, fall back to a hash suffix when the result would be too long
    or empty. The hash suffix keeps the mapping injective.
    """
    parsed = urlparse(url)
    path = parsed.path.lstrip("/").rstrip("/")
    if not path:
        path = "index"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path)
    # Keep filenames reasonable (Windows MAX_PATH considerations).
    if len(safe) > 100:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        safe = safe[:80] + "_" + digest
    return f"{safe}.md"


def _matches_url_filters(url: str, includes: list, excludes: list) -> bool:
    """URL passes if it matches ALL includes (when set) and NO excludes."""
    if includes:
        if not any(p in url for p in includes):
            return False
    if excludes:
        if any(p in url for p in excludes):
            return False
    return True


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------


def _crawl(
    *,
    start_url: str,
    max_depth: int,
    max_pages: int,
    same_origin_only: bool,
    include_patterns: list,
    exclude_patterns: list,
    request_delay_seconds: float,
    user_agent: str,
    http_client: Optional[httpx.Client] = None,
) -> list:
    """BFS over a docs site. Returns a list of (url, html, status_code) tuples.

    Polite by default: single-threaded, request_delay between requests, custom
    User-Agent so site owners can identify the crawler. Same-origin filter
    prevents accidental drift onto external sites.

    `http_client` is an injection point for tests — production passes None
    and we build a default client here.
    """
    own_client = http_client is None
    if own_client:
        http_client = httpx.Client(
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    start_url = _normalize_url(start_url)
    origin = urlparse(start_url).netloc

    seen: set = set()
    queue: list = [(start_url, 0)]
    results: list = []
    error_count = 0

    try:
        while queue and len(results) < max_pages:
            url, depth = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            if depth > max_depth:
                continue
            if same_origin_only and urlparse(url).netloc != origin:
                continue

            is_seed = (url == start_url)
            passes_filters = _matches_url_filters(url, include_patterns, exclude_patterns)

            # The seed URL is special-cased: we *visit* it (to discover links
            # to filter-matching children) even when it doesn't itself pass
            # the include filter. Real example: user crawls `docs.unity.com/`
            # with include=`/Manual/` — without this, we'd never discover the
            # Manual links from the homepage. Non-seed URLs that fail the
            # filter are skipped entirely.
            if not passes_filters and not is_seed:
                continue
            # Excludes always apply, including to the seed.
            if exclude_patterns and any(p in url for p in exclude_patterns):
                continue

            try:
                response = http_client.get(url)
            except (httpx.HTTPError, OSError) as e:
                error_count += 1
                print(f"[webdoc] fetch failed for {url}: {e}", file=sys.stderr)
                continue

            if response.status_code >= 400:
                error_count += 1
                continue

            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if content_type not in ("text/html", "application/xhtml+xml"):
                # PDFs, images, JSON, etc. — skip; webdoc only handles HTML.
                continue

            if len(response.content) > _MAX_RESPONSE_BYTES:
                print(
                    f"[webdoc] skipping {url}: response too large "
                    f"({len(response.content)} bytes > {_MAX_RESPONSE_BYTES})",
                    file=sys.stderr,
                )
                continue

            html = response.text

            # Only add to results if filters pass (the seed gets *visited*
            # for link discovery even when it doesn't pass the include
            # filter, but it's not added to the dump unless it also matches).
            if passes_filters:
                results.append((url, html, response.status_code))

            # Discover links and enqueue them. We stop discovering once we
            # would exceed max_depth on the children, to keep the frontier
            # small.
            if depth < max_depth:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                except Exception as e:
                    print(f"[webdoc] HTML parse failed for {url}: {e}", file=sys.stderr)
                    soup = None
                if soup is not None:
                    for a in soup.find_all("a", href=True):
                        absolute = _normalize_url(urljoin(url, a["href"]))
                        if absolute not in seen:
                            queue.append((absolute, depth + 1))

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
    finally:
        if own_client:
            http_client.close()

    if error_count:
        print(f"[webdoc] crawl finished with {error_count} error(s) skipped", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def _extract_main_content(html: str) -> str:
    """Run trafilatura to pull the main article content out of a page,
    discarding nav / footer / ads. Output is markdown-ish text — good
    enough for embedding."""
    try:
        text = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=False,
            include_images=False,
            include_tables=True,
            include_formatting=True,
            favor_recall=True,
        )
    except Exception as e:
        print(f"[webdoc] extraction failed: {e}", file=sys.stderr)
        text = None
    return (text or "").strip()


def _build_markdown_dump(url: str, html: str, fetched_at: str) -> str:
    """Wrap extracted content in YAML frontmatter so the URL stays adjacent
    to the text in every chunk that includes the file header."""
    content = _extract_main_content(html)
    if not content:
        # Page had no extractable content (maybe a redirect landing, or pure
        # navigation page). Save a stub so the URL is still recorded.
        content = f"(no extractable content from {url})"
    return (
        f"---\n"
        f"url: {url}\n"
        f"fetched_at: {fetched_at}\n"
        f"---\n\n"
        f"{content}\n"
    )


# ---------------------------------------------------------------------------
# WebdocBackend
# ---------------------------------------------------------------------------


class WebdocBackend(SourceBackend):
    """Indexes a public documentation site, fetched on demand.

    Layout under storage_dir (one per source):
      _dump/                      Extracted .md files, one per crawled URL
      _fetch_state.json           {url: fetched_at, ...}
      chroma.sqlite3              Vector store (managed by CodebaseRAG)
      metadata.json               Drift snapshot (managed by CodebaseRAG)
      file_hashes.json            SHA cache (managed by CodebaseRAG)

    The dump folder doubles as the codebase_path for the inner CodebaseRAG.
    """

    type_name = "webdoc"

    def __init__(self, name, source_config, shared_config, storage_dir):
        super().__init__(name, source_config, shared_config, storage_dir)

        self.dump_dir = self.storage_dir / "_dump"
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_state_file = self.storage_dir / "_fetch_state.json"

        # Crawler config (with defaults; config.py validates ranges).
        self.start_url: str = source_config["url"]
        self.max_depth: int = int(source_config.get("max_depth", 3))
        self.max_pages: int = int(source_config.get("max_pages", _DEFAULT_MAX_PAGES))
        self.same_origin_only: bool = bool(source_config.get("same_origin_only", True))
        self.include_patterns: list = list(source_config.get("include_url_patterns") or [])
        self.exclude_patterns: list = list(source_config.get("exclude_url_patterns") or [])
        self.request_delay_seconds: float = float(source_config.get("request_delay_seconds", 0.5))
        self.user_agent: str = str(
            source_config.get("user_agent")
            or "Lynx-DocFetcher/0.4 (+https://github.com/loren/lynx)"
        )

        # Inner indexing engine — same CodebaseRAG that handles the codebase
        # source type, just pointed at our dump folder.
        from ..rag_manager import CodebaseRAG
        self.rag = CodebaseRAG(
            codebase_path=str(self.dump_dir),
            rag_storage_path=str(self.storage_dir),
            # The dump only contains .md files we produced, so the extension
            # set is fixed regardless of the source's downstream config.
            supported_extensions=[".md"],
            embedding_model_name=shared_config.embedding.model_name,
            collection_name=name,
            search_mode=shared_config.search.mode,
            rrf_k=shared_config.search.rrf_k,
            candidate_pool_size=shared_config.search.candidate_pool_size,
        )

    # ------------------------------------------------------------------
    # Fetch (the webdoc-specific work)
    # ------------------------------------------------------------------

    def _load_fetch_state(self) -> dict:
        if not self.fetch_state_file.exists():
            return {}
        try:
            return json.loads(self.fetch_state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_fetch_state(self, state: dict) -> None:
        self.fetch_state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _wipe_dump(self) -> None:
        """Remove every .md file in the dump folder (but keep the dir).

        Called before a re-fetch so URLs no longer present in the crawl don't
        leave stale chunks indexed.
        """
        for f in self.dump_dir.glob("*.md"):
            try:
                f.unlink()
            except OSError as e:
                print(f"[webdoc] could not delete {f}: {e}", file=sys.stderr)

    def fetch(self, *, http_client: Optional[httpx.Client] = None) -> dict:
        """Crawl, extract, write the dump. Returns the new fetch state.

        Always full-refresh: wipes the dump dir first so deleted pages get
        evicted from the index. ETag/Last-Modified-based partial refresh
        wasn't worth the complexity (servers vary too much).
        """
        print(
            f"[webdoc:{self.name}] crawling {self.start_url} "
            f"(max_depth={self.max_depth}, max_pages={self.max_pages})",
            file=sys.stderr,
        )
        pages = _crawl(
            start_url=self.start_url,
            max_depth=self.max_depth,
            max_pages=self.max_pages,
            same_origin_only=self.same_origin_only,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
            request_delay_seconds=self.request_delay_seconds,
            user_agent=self.user_agent,
            http_client=http_client,
        )
        print(f"[webdoc:{self.name}] fetched {len(pages)} page(s)", file=sys.stderr)

        self._wipe_dump()

        fetched_at = datetime.now().isoformat()
        new_state: dict = {}
        for url, html, _status in pages:
            filename = _url_to_filename(url)
            dump_path = self.dump_dir / filename
            content = _build_markdown_dump(url, html, fetched_at)
            try:
                dump_path.write_text(content, encoding="utf-8")
            except OSError as e:
                print(f"[webdoc] could not write dump for {url}: {e}", file=sys.stderr)
                continue
            new_state[url] = {
                "fetched_at": fetched_at,
                "dump_file": filename,
            }

        self._save_fetch_state(new_state)
        return new_state

    # ------------------------------------------------------------------
    # SourceBackend interface
    # ------------------------------------------------------------------

    def search(self, query, top_k=5, **kw):
        return self.rag.search(query, top_k=top_k, **kw)

    def deep_search(self, queries, top_k=5, **kw):
        return self.rag.deep_search(
            queries=queries,
            top_k=top_k,
            score_thresholds=self.shared.search.deep.score_thresholds,
            **kw,
        )

    def update(self, force: bool = False) -> None:
        """For a webdoc source, "update" means: refetch + reindex.

        `force=True` (the only meaningful call path) crawls the configured
        URL, replaces the dump, then triggers a full rebuild of the index.
        `force=False` is a no-op — webdoc sources have no automatic refresh
        cadence; they don't get stale "on their own", they get stale when
        the upstream site changes, which we can't detect without crawling.
        """
        if not force:
            return
        self.fetch()
        # The inner CodebaseRAG sees a wiped + repopulated dump dir; force
        # rebuild brings the index in sync with the new dump.
        self.rag.update(force=True)

    def start_watcher(self) -> None:
        # Webdoc has no filesystem to watch; the dump dir is our own write
        # space, not user content. Refresh is manual via `lynx build`.
        return None

    def status(self) -> dict:
        state = self._load_fetch_state()
        try:
            chunk_count = self.rag.vector_store._collection.count()
        except Exception:
            chunk_count = None
        drift = self.rag.check_config_drift()
        # Pick any fetched_at — they should all match (single fetch run).
        last_fetched = None
        if state:
            sample = next(iter(state.values()))
            last_fetched = sample.get("fetched_at") if isinstance(sample, dict) else None
        return {
            "name": self.name,
            "type": self.type_name,
            "url": self.start_url,
            "page_count": len(state),
            "chunk_count": chunk_count,
            "last_fetched_at": last_fetched,
            # Reuse last_update from drift metadata for cross-type consistency.
            "last_update": self.rag.metadata.get("last_update"),
            "drift_severity": drift["severity"] if drift else None,
        }

    def drift_status_text(self) -> str:
        return self.rag.drift_status_text()

    def needs_update(self) -> bool:
        """A webdoc never auto-needs-update; refresh is always explicit."""
        return False
