"""The `language` setting of a codebase source, and the sentence it puts in
the instructions the agent reads at startup.

Why it exists: people write to the agent in their own language and expect
it to search in the code's. On a private C# library (lynx-zombie-bench D5),
requests in Italian found the right file in the first five for none of the
65 that named no identifier; the same requests in English, for 32.

Covered:
  1. no setting: English for code and comments
  2. ISO 639-1 codes are read, case and spaces forgiven
  3. a bad code, a bad shape or an unknown key stops the load with a message
  4. the instructions tell the agent which language to search in
  5. mixed sources name each one; comments in another language are said
  6. no codebase source, no sentence
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _load(tmp_path: Path, language=None):
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    source = {"type": "codebase", "path": str(code), "supported_extensions": [".py"]}
    if language is not None:
        source["language"] = language
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"config_version": 2, "storage_path": str(tmp_path / "s"),
                               "sources": {"lib": source}}), encoding="utf-8")
    from lynx.config import load_config
    return load_config(cfg).sources["lib"]


def _lang(source):
    return source["language"] if isinstance(source, dict) else source.language


def test_default_is_english(tmp_path):
    assert _lang(_load(tmp_path)) == {"code": "en", "comments": "en"}


def test_codes_are_read(tmp_path):
    assert _lang(_load(tmp_path, {"code": " EN ", "comments": "it"})) == {"code": "en", "comments": "it"}
    assert _lang(_load(tmp_path, {"comments": "de"})) == {"code": "en", "comments": "de"}


@pytest.mark.parametrize("bad, words", [
    ({"code": "english"}, "two-letter"),
    ({"code": "e1"}, "two-letter"),
    ("en", "must be an object"),
    ({"code": "en", "docs": "it"}, "unknown key"),
])
def test_bad_settings_stop_the_load(tmp_path, capsys, bad, words):
    with pytest.raises(SystemExit):
        _load(tmp_path, bad)
    assert words in capsys.readouterr().err


class _Backend:
    type_name = "codebase"
    graph = None

    def __init__(self, language=None, type_name="codebase"):
        self.type_name = type_name
        self.source_config = {"path": "/x", "git_integration": {"enabled": False}}
        if language:
            self.source_config["language"] = language


class _Manager:
    def __init__(self, backends):
        self.backends = backends


def _instructions(backends):
    from lynx.server import _build_instructions
    return _build_instructions(_Manager(backends))


def test_instructions_say_the_code_language():
    text = _instructions({"lib": _Backend()})
    assert "Write the query in English, the code's language, whatever language the user writes in." in text


def test_mixed_sources_and_comments_are_named():
    text = _instructions({
        "game": _Backend(),
        "lib": _Backend({"code": "it", "comments": "it"}),
        "tools": _Backend({"code": "en", "comments": "it"}),
    })
    assert "(game: English, lib: Italian, tools: English)" in text
    assert "the comments (Italian in tools) are found in their own language" in text


def test_no_codebase_no_sentence():
    text = _instructions({"docs": _Backend(type_name="webdoc")})
    assert "Write the query" not in text
