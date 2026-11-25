"""LYNX_EMBED_THREADS caps the threads ONNX Runtime uses to embed.

By default it takes every core: a first index of a large repository then
holds the whole machine for hours. With a cap it can run beside other work
(a game measuring its frames, a dozen agents building), slower but out of
the way. Unset, the default stays ONNX Runtime's own.
"""
from __future__ import annotations


def _options(monkeypatch, value):
    from lynx import embeddings
    seen = {}

    class _Session:
        def __init__(self, path, opts, providers):
            seen["intra"], seen["inter"] = opts.intra_op_num_threads, opts.inter_op_num_threads

    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", _Session)
    if value is None:
        monkeypatch.delenv("LYNX_EMBED_THREADS", raising=False)
    else:
        monkeypatch.setenv("LYNX_EMBED_THREADS", value)
    embeddings.make_session("model.onnx")
    return seen


def test_the_threads_of_the_embedder_can_be_capped(monkeypatch):
    assert _options(monkeypatch, "3") == {"intra": 3, "inter": 1}


def test_without_a_cap_onnx_runtime_decides(monkeypatch):
    assert _options(monkeypatch, None) == {"intra": 0, "inter": 0}
    assert _options(monkeypatch, "zero") == {"intra": 0, "inter": 0}
