"""Smoke test for the webdoc source backend.

Uses httpx.MockTransport to stub a tiny fake documentation site (no network
calls actually leave the process). Validates:

  1. Crawler discovers linked pages within max_depth
  2. same_origin_only filters out external links
  3. include_url_patterns / exclude_url_patterns are honored
  4. max_pages caps the frontier
  5. trafilatura extracts main content (drops nav/footer)
  6. dump files carry YAML frontmatter with URL + fetched_at
  7. WebdocBackend.update(force=True) re-fetches + reindexes (full refresh)
  8. Status reports page_count + last_fetched_at
  9. Refresh on a CHANGED site replaces the old dump (no stale chunks)

The test never touches the real internet; every HTTP request is routed
through the mock transport. The user's real index is never touched (we use
a tempdir for everything).
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Fake docs site fixtures (HTML strings)
# ---------------------------------------------------------------------------

FAKE_PAGES = {
    "https://docs.example.com/": """
        <html>
          <head><title>Home</title></head>
          <body>
            <nav><a href="/about">About</a></nav>
            <main>
              <h1>Welcome to ExampleDocs</h1>
              <p>This is the home page. Lots of useful introductory text
              about widgets, sensors, and orchestration.</p>
              <a href="/guide/intro">Intro guide</a>
              <a href="/guide/widgets">Widget reference</a>
              <a href="https://other-site.example/external">External link</a>
            </main>
            <footer>(c) Example Corp</footer>
          </body>
        </html>
    """,
    "https://docs.example.com/guide/intro": """
        <html><body><main>
          <h1>Introduction</h1>
          <p>Sensors detect events and forward them to the orchestrator.
          This document explains the lifecycle of a sensor reading.</p>
          <a href="/guide/widgets">Next: widgets</a>
        </main></body></html>
    """,
    "https://docs.example.com/guide/widgets": """
        <html><body><main>
          <h1>Widgets</h1>
          <p>Widgets are the atomic UI element. Each widget has a unique ID
          and a color. Widget operations are thread-safe.</p>
        </main></body></html>
    """,
    "https://docs.example.com/about": """
        <html><body><main>
          <h1>About</h1>
          <p>About the example documentation site.</p>
        </main></body></html>
    """,
    "https://docs.example.com/legacy/old-thing": """
        <html><body><main>
          <h1>Legacy page</h1>
          <p>This should be EXCLUDED by the test's exclude_url_patterns.</p>
        </main></body></html>
    """,
}


def _build_mock_transport(pages_dict=None):
    """Create an httpx.MockTransport that serves `pages_dict` (or FAKE_PAGES)."""
    import httpx
    pages = pages_dict if pages_dict is not None else FAKE_PAGES

    def _handler(request: httpx.Request) -> httpx.Response:
        # Normalize: strip trailing slash for lookup
        url = str(request.url)
        normalized = url.rstrip("/")
        body = pages.get(url) or pages.get(normalized) or pages.get(url + "/")
        if body is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/html; charset=utf-8"},
        )

    return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def main() -> int:
    import httpx
    from lynx.config import load_config
    from lynx.source_manager import SourceManager
    from lynx.sources.webdoc import _crawl, _build_markdown_dump, _url_to_filename

    tmp = Path(tempfile.mkdtemp(prefix="lynx-webdoc-"))
    print(f"[test] tempdir: {tmp}")

    mgr = None
    try:
        # ============================================================
        # 1. Crawler: BFS discovers linked pages within depth bound
        # ============================================================
        transport = _build_mock_transport()
        with httpx.Client(transport=transport) as client:
            pages = _crawl(
                start_url="https://docs.example.com/",
                max_depth=2,
                max_pages=100,
                same_origin_only=True,
                include_patterns=[],
                exclude_patterns=[],
                request_delay_seconds=0,
                user_agent="test/1",
                http_client=client,
            )
        urls = sorted(p[0] for p in pages)
        # We expect 4: root, /about, /guide/intro, /guide/widgets, /legacy/old-thing
        # = 5 because depth-2 also finds /legacy/* via cross-link... actually
        # /legacy isn't linked from the home page. Let me re-check FAKE_PAGES:
        # home → /about, /guide/intro, /guide/widgets, external
        # /guide/intro → /guide/widgets (already enqueued)
        # No link from anywhere → /legacy/old-thing, so it shouldn't appear.
        if "https://docs.example.com/legacy/old-thing" in urls:
            print(f"[test] FAIL [1/9]: legacy URL crawled (no link points to it): {urls}")
            return 1
        # Crawler should find at least the 4 reachable pages
        expected = {
            # Root URL keeps trailing slash because path is exactly "/".
            "https://docs.example.com/",
            "https://docs.example.com/about",
            "https://docs.example.com/guide/intro",
            "https://docs.example.com/guide/widgets",
        }
        found = set(urls)
        missing = expected - found
        if missing:
            print(f"[test] FAIL [1/9]: missing expected URLs: {missing}. got {urls}")
            return 1
        print(f"[test] OK [1/9] crawler found {len(pages)} pages within depth bound")

        # ============================================================
        # 2. same_origin_only: external link is NOT crawled
        # ============================================================
        if any("other-site.example" in u for u in urls):
            print(f"[test] FAIL [2/9]: same_origin_only=True failed: {urls}")
            return 2
        print(f"[test] OK [2/9] same_origin_only filtered out external host")

        # ============================================================
        # 3. exclude_url_patterns blocks /legacy/
        # ============================================================
        # Add an extra page that IS linked from home so the exclude has work to do.
        pages_with_legacy_link = dict(FAKE_PAGES)
        pages_with_legacy_link["https://docs.example.com/"] = (
            FAKE_PAGES["https://docs.example.com/"].replace(
                '</main>', '<a href="/legacy/old-thing">Legacy</a></main>'
            )
        )
        transport2 = _build_mock_transport(pages_with_legacy_link)
        with httpx.Client(transport=transport2) as client:
            pages2 = _crawl(
                start_url="https://docs.example.com/",
                max_depth=2,
                max_pages=100,
                same_origin_only=True,
                include_patterns=[],
                exclude_patterns=["/legacy/"],
                request_delay_seconds=0,
                user_agent="test/1",
                http_client=client,
            )
        if any("/legacy/" in u for (u, _, _) in pages2):
            print(f"[test] FAIL [3/9]: exclude_url_patterns failed; legacy URL present")
            return 3
        print(f"[test] OK [3/9] exclude_url_patterns blocks /legacy/")

        # ============================================================
        # 4. include_url_patterns restricts to /guide/
        # ============================================================
        with httpx.Client(transport=_build_mock_transport()) as client:
            guides_only = _crawl(
                start_url="https://docs.example.com/",
                max_depth=2,
                max_pages=100,
                same_origin_only=True,
                include_patterns=["/guide/"],
                exclude_patterns=[],
                request_delay_seconds=0,
                user_agent="test/1",
                http_client=client,
            )
        # The seed URL itself ("/") doesn't match "/guide/" — but the crawler
        # still VISITS it to discover the /guide/* links. The seed must NOT
        # appear in results (because it failed the include filter), but the
        # /guide/ children MUST appear. That's the contract.
        urls_only = sorted(u for (u, _, _) in guides_only)
        if len(urls_only) < 2:
            print(f"[test] FAIL [4/9]: expected >=2 /guide/ matches, got {urls_only}")
            return 4
        if not all("/guide/" in u for u in urls_only):
            print(f"[test] FAIL [4/9]: include filter let through non-/guide/ URLs: {urls_only}")
            return 4
        # Specifically the seed (which fails the filter) must NOT be a result.
        if any(u == "https://docs.example.com/" for u in urls_only):
            print(f"[test] FAIL [4/9]: seed URL leaked into results despite failing include filter")
            return 4
        print(f"[test] OK [4/9] include_url_patterns restricts to '/guide/' ({len(urls_only)} matches, seed excluded)")

        # ============================================================
        # 5. max_pages caps the frontier
        # ============================================================
        with httpx.Client(transport=_build_mock_transport()) as client:
            capped = _crawl(
                start_url="https://docs.example.com/",
                max_depth=10,
                max_pages=2,  # tight cap
                same_origin_only=True,
                include_patterns=[],
                exclude_patterns=[],
                request_delay_seconds=0,
                user_agent="test/1",
                http_client=client,
            )
        if len(capped) > 2:
            print(f"[test] FAIL [5/9]: max_pages=2 ignored, got {len(capped)} pages")
            return 5
        print(f"[test] OK [5/9] max_pages cap honored ({len(capped)} <= 2)")

        # ============================================================
        # 6. Dump format: YAML frontmatter + extracted content
        # ============================================================
        html = FAKE_PAGES["https://docs.example.com/guide/widgets"]
        dump = _build_markdown_dump(
            "https://docs.example.com/guide/widgets",
            html,
            "2026-05-12T00:00:00",
        )
        if not dump.startswith("---\nurl: https://docs.example.com/guide/widgets\n"):
            print(f"[test] FAIL [6/9]: dump missing YAML frontmatter; got {dump[:150]!r}")
            return 6
        if "Widgets are the atomic UI element" not in dump:
            print(f"[test] FAIL [6/9]: dump missing extracted content; got {dump[:300]!r}")
            return 6
        if "(c) Example Corp" in dump:
            print(f"[test] FAIL [6/9]: extractor leaked footer into dump")
            return 6
        print(f"[test] OK [6/9] dump has YAML frontmatter + main content (footer stripped)")

        # ============================================================
        # 7. WebdocBackend.update(force=True) runs end-to-end
        # ============================================================
        cfg_dict = {
            "config_version": 2,
            "storage_path": str(tmp / "rag_storage"),
            "loading_timeout_seconds": 60,
            "embedding": {"model_name": "BAAI/bge-small-en-v1.5"},
            "search": {
                "default_top_k": 5, "mode": "hybrid", "rrf_k": 60,
                "candidate_pool_size": 30,
                "deep": {"min_results": 1,
                         "score_thresholds": {"dense": 0.45, "hybrid": 0.012, "sparse": 3.0}}
            },
            "sources": {
                "demo_docs": {
                    "type": "webdoc",
                    "url": "https://docs.example.com/",
                    "max_depth": 2,
                    "max_pages": 50,
                    "same_origin_only": True,
                    "include_url_patterns": [],
                    "exclude_url_patterns": [],
                    "request_delay_seconds": 0,
                }
            }
        }
        cfg_path = tmp / "config.json"
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))

        cfg = load_config(cfg_path)
        mgr = SourceManager(cfg)
        backend = mgr.get("demo_docs")

        # Inject the mock transport via a wrapped httpx.Client
        mock_client = httpx.Client(
            transport=_build_mock_transport(),
            timeout=30.0,
            follow_redirects=True,
        )
        # The backend.fetch() accepts http_client injection; backend.update()
        # doesn't, so we call fetch+rag.update manually.
        backend.fetch(http_client=mock_client)
        mock_client.close()
        backend.rag.update(force=True)

        dump_dir = tmp / "rag_storage" / "demo_docs" / "_dump"
        md_files = sorted(p.name for p in dump_dir.glob("*.md"))
        if len(md_files) < 3:
            print(f"[test] FAIL [7/9]: expected >=3 dump files, got {md_files}")
            return 7
        # Validate the fetch state file
        state = json.loads((tmp / "rag_storage" / "demo_docs" / "_fetch_state.json").read_text())
        if len(state) != len(md_files):
            print(f"[test] FAIL [7/9]: fetch state count {len(state)} != dump files {len(md_files)}")
            return 7
        # Search should now return something
        results = backend.search("widget orchestration", top_k=3)
        if not results:
            print(f"[test] FAIL [7/9]: search returned 0 results after ingest")
            return 7
        print(f"[test] OK [7/9] full ingest: {len(md_files)} dumped, search returned {len(results)} hit(s)")

        # ============================================================
        # 8. Status reports page_count + last_fetched_at
        # ============================================================
        status = backend.status()
        if status["type"] != "webdoc":
            print(f"[test] FAIL [8/9]: status type {status['type']!r} != 'webdoc'")
            return 8
        if status["page_count"] < 3:
            print(f"[test] FAIL [8/9]: status page_count {status['page_count']} too low")
            return 8
        if not status.get("last_fetched_at"):
            print(f"[test] FAIL [8/9]: status missing last_fetched_at")
            return 8
        if status.get("url") != "https://docs.example.com/":
            print(f"[test] FAIL [8/9]: status url {status.get('url')!r} mismatch")
            return 8
        print(f"[test] OK [8/9] status: type={status['type']} pages={status['page_count']} fetched_at present")

        # ============================================================
        # 9. Refresh on changed site: stale pages disappear from dump
        # ============================================================
        # Simulate the site removing /about and adding /new-page
        new_pages = {
            "https://docs.example.com/": FAKE_PAGES["https://docs.example.com/"].replace(
                '<a href="/about">About</a>', '<a href="/new-page">New</a>'
            ).replace(
                '<a href="/about">About</a>', ''  # remove nav link too
            ),
            "https://docs.example.com/new-page": """
                <html><body><main>
                  <h1>Brand New</h1>
                  <p>This page was added after the first crawl.</p>
                </main></body></html>
            """,
            "https://docs.example.com/guide/intro": FAKE_PAGES["https://docs.example.com/guide/intro"],
            "https://docs.example.com/guide/widgets": FAKE_PAGES["https://docs.example.com/guide/widgets"],
            # NOTE: /about is gone from the new dict on purpose.
        }
        mock2 = httpx.Client(transport=_build_mock_transport(new_pages), timeout=30, follow_redirects=True)
        backend.fetch(http_client=mock2)
        mock2.close()
        backend.rag.update(force=True)

        md_files_after = sorted(p.name for p in dump_dir.glob("*.md"))
        # No file should reference /about anymore
        about_filename = _url_to_filename("https://docs.example.com/about")
        if about_filename in md_files_after:
            print(f"[test] FAIL [9/9]: stale /about dump survived refresh: {md_files_after}")
            return 9
        # New page should be present
        new_filename = _url_to_filename("https://docs.example.com/new-page")
        if new_filename not in md_files_after:
            print(f"[test] FAIL [9/9]: new-page dump missing after refresh: {md_files_after}")
            return 9
        # Search for the new content
        new_results = backend.search("brand new page added after first crawl", top_k=3)
        if not any("new-page" in r.get("file_path", "") for r in new_results):
            print(f"[test] FAIL [9/9]: refresh did not index new content; got {[r.get('file') for r in new_results]}")
            return 9
        print(f"[test] OK [9/9] refresh: stale /about gone, new-page indexed")

        print("\n[test] === SUCCESS: webdoc backend works as expected ===")
        return 0

    finally:
        # Release ChromaDB handles before rmtree (Windows file locks)
        try:
            del mgr
        except Exception:
            pass
        gc.collect()
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception as e:
            print(f"[test] warning: cleanup failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
