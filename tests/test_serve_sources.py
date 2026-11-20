"""`lynx serve --source NAME`: one config, a subset of its sources per session."""
from __future__ import annotations

import json

import pytest

from lynx.cli_parser import _build_parser as build_parser
from lynx.config import load_config
from lynx.server import only_sources


def _config(tmp_path):
    for name in ("game", "engine", "docs"):
        (tmp_path / name).mkdir()
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "config_version": 2,
        "storage_path": str(tmp_path / "storage"),
        "sources": {
            name: {"type": "codebase", "path": str(tmp_path / name),
                   "supported_extensions": [".py"]}
            for name in ("game", "engine", "docs")
        },
    }), encoding="utf-8")
    return load_config(path)


def test_only_the_named_sources_are_served_in_the_order_given(tmp_path):
    cfg = _config(tmp_path)
    served = only_sources(cfg, ["docs", "game", "docs"])
    assert list(served.sources) == ["docs", "game"]
    assert served.sources["game"] == cfg.sources["game"]
    assert list(cfg.sources) == ["game", "engine", "docs"]      # config untouched


def test_no_selection_serves_everything(tmp_path):
    cfg = _config(tmp_path)
    assert only_sources(cfg, None) is cfg
    assert only_sources(cfg, []) is cfg


def test_an_unknown_name_is_refused_with_the_known_ones(tmp_path):
    cfg = _config(tmp_path)
    with pytest.raises(ValueError) as e:
        only_sources(cfg, ["game", "gmae"])
    assert "gmae" in str(e.value) and "engine" in str(e.value)


def test_the_flag_repeats():
    args = build_parser().parse_args(["serve", "--source", "game", "-s", "docs"])
    assert args.sources == ["game", "docs"]
    assert build_parser().parse_args(["serve"]).sources is None
