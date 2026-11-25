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


def test_a_build_waits_while_the_pause_file_exists_and_a_search_does_not(tmp_path, monkeypatch):
    """LYNX_PAUSE_FILE: while it exists the embedding of a build waits between
    two runs (a measurement wants the machine alone); a search in another
    thread embeds its query as usual."""
    import threading
    import time
    from lynx import embeddings
    flag = tmp_path / "measuring.txt"
    flag.write_text("x", encoding="utf-8")
    monkeypatch.setenv("LYNX_PAUSE_FILE", str(flag))
    monkeypatch.setattr(embeddings, "PAUSE_POLL_SECONDS", 0.02)

    t = time.monotonic()
    embeddings.wait_while_paused()                      # not in a build: no wait
    assert time.monotonic() - t < 0.5

    done = []

    def build():
        with embeddings.pausable():
            embeddings.wait_while_paused()
            done.append(time.monotonic())

    th = threading.Thread(target=build)
    th.start()
    time.sleep(0.3)
    assert not done, "the build went on while the file was there"
    flag.unlink()
    th.join(timeout=5)
    assert done, "the build did not resume once the file was gone"


def test_the_embedder_asks_between_two_runs(monkeypatch):
    from lynx import embeddings
    from lynx.embeddings import OnnxEmbedding
    emb = OnnxEmbedding("BAAI/bge-small-en-v1.5")
    emb._run_size = 2
    asked = []
    monkeypatch.setattr(embeddings, "wait_while_paused", lambda: asked.append(1))
    emb.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(asked) == 3
