"""Tests for the opt-in JS rendering path of the webdoc source.

No real browser involved: the crawler accepts an injected `renderer`
object, so we exercise routing, link discovery on the *rendered* DOM, and
the error messages a user gets when Playwright isn't installed.
"""
from __future__ import annotations

import json

import pytest

from lynx.config import load_config
from lynx.sources.webdoc import _crawl, _PlaywrightRenderer


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _write_config(tmp_path, webdoc_fields):
    cfg = {
        "config_version": 2,
        "storage_path": str(tmp_path / "storage"),
        "sources": {
            "docs": {
                "type": "webdoc",
                "url": "https://example.com/docs",
                **webdoc_fields,
            }
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_render_js_defaults_off(tmp_path):
    cfg = load_config(_write_config(tmp_path, {}))
    src = cfg.sources["docs"]
    assert src["render_js"] is False
    assert src["render_wait_until"] == "networkidle"
    assert src["render_timeout_seconds"] == 30.0


def test_render_js_fields_parsed(tmp_path):
    cfg = load_config(_write_config(tmp_path, {
        "render_js": True,
        "render_wait_until": "domcontentloaded",
        "render_timeout_seconds": 10,
    }))
    src = cfg.sources["docs"]
    assert src["render_js"] is True
    assert src["render_wait_until"] == "domcontentloaded"
    assert src["render_timeout_seconds"] == 10.0


def test_invalid_wait_until_rejected(tmp_path):
    with pytest.raises(SystemExit):
        load_config(_write_config(tmp_path, {"render_wait_until": "whenever"}))


def test_nonpositive_render_timeout_rejected(tmp_path):
    with pytest.raises(SystemExit):
        load_config(_write_config(tmp_path, {"render_timeout_seconds": 0}))


# ---------------------------------------------------------------------------
# Crawler with an injected renderer
# ---------------------------------------------------------------------------


class FakeRenderer:
    """Maps url -> rendered html; simulates the post-JS DOM."""

    def __init__(self, pages):
        self.pages = pages
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        if url not in self.pages:
            raise ValueError(f"no such page: {url}")
        return self.pages[url], 200, "text/html"

    def close(self):
        pass


def test_crawl_uses_renderer_and_discovers_rendered_links():
    # The seed's <a> tags only exist in the *rendered* DOM — exactly the SPA
    # scenario plain HTTP crawling cannot handle.
    pages = {
        "https://spa.example/docs": (
            "<html><body><div id='app'>"
            "<a href='/docs/intro'>intro</a>"
            "<a href='/docs/api'>api</a>"
            "</div></body></html>"
        ),
        "https://spa.example/docs/intro": "<html><body>Intro content</body></html>",
        "https://spa.example/docs/api": "<html><body>API content</body></html>",
    }
    renderer = FakeRenderer(pages)
    results = _crawl(
        start_url="https://spa.example/docs",
        max_depth=2,
        max_pages=10,
        same_origin_only=True,
        include_patterns=[],
        exclude_patterns=[],
        request_delay_seconds=0,
        user_agent="test",
        renderer=renderer,
    )
    urls = {u for u, _html, _s in results}
    assert urls == set(pages), urls
    # All fetches went through the renderer, none through HTTP.
    assert set(renderer.requested) == set(pages)


def test_crawl_renderer_error_is_skipped_not_fatal():
    pages = {
        "https://spa.example/docs": "<html><body><a href='/docs/broken'>x</a></body></html>",
        # /docs/broken intentionally missing -> FakeRenderer raises
    }
    renderer = FakeRenderer(pages)
    results = _crawl(
        start_url="https://spa.example/docs",
        max_depth=2,
        max_pages=10,
        same_origin_only=True,
        include_patterns=[],
        exclude_patterns=[],
        request_delay_seconds=0,
        user_agent="test",
        renderer=renderer,
    )
    assert [u for u, _h, _s in results] == ["https://spa.example/docs"]


# ---------------------------------------------------------------------------
# Missing-Playwright error message
# ---------------------------------------------------------------------------


def test_missing_playwright_message_names_the_fix(monkeypatch):
    import sys
    # Setting the module entry to None makes `import playwright.sync_api`
    # raise ImportError even if Playwright happens to be installed here.
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(RuntimeError, match="webdoc-js"):
        _PlaywrightRenderer(user_agent="test")
