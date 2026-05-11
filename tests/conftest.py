"""Make `src/` importable when tests are run without `pip install -e .`.

This is a convenience for development: contributors can `python tests/test_X.py`
straight from a fresh clone. CI/PyPI users always go through the installed
package, where this shim is a no-op.

Also provides `build_rag_from_first_source(config_path)` — shared helper for
the smoke tests that exercise the underlying `CodebaseRAG` directly under
the v2 schema's per-source storage layout.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def build_rag_from_first_source(config_path):
    """Construct a CodebaseRAG bound to the FIRST codebase source in a v2 config.

    Tests that exercise CodebaseRAG directly (drift, filters, deep_search) use
    this so they don't depend on which name the user picked for the source.
    Returns (config, source_name, rag).
    """
    from lynx.config import load_config
    from lynx.rag_manager import CodebaseRAG

    cfg = load_config(config_path)
    name = next(iter(cfg.sources))
    src = cfg.sources[name]
    storage_dir = Path(cfg.storage_path) / name
    rag = CodebaseRAG(
        codebase_path=str(src["path"]),
        rag_storage_path=str(storage_dir),
        supported_extensions=src["supported_extensions"],
        embedding_model_name=cfg.embedding.model_name,
        collection_name=name,
        search_mode=cfg.search.mode,
        rrf_k=cfg.search.rrf_k,
        candidate_pool_size=cfg.search.candidate_pool_size,
    )
    return cfg, name, rag
