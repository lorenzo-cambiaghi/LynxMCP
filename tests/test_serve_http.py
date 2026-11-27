"""`lynx serve --http PORT`: one server for every session of a machine.

A stdio server is one process per client, each with its own embedding model
and graph in memory: on a machine running a dozen agents on a 1.4-million-line
library that is a dozen copies of the same indexes. Over HTTP every session
connects to one process. Its tools must not block the event loop, or a long
search of one client would hold up the handshake of the next: sync tools run
on a worker thread, one at a time, as they do on stdio. The read-only tools
queue apart from the others, so one session's hour-long rebuild does not hold
up the searches and hooks of every other session.
"""
from __future__ import annotations

import threading

import anyio

from lynx.cli_parser import _build_parser as build_parser


def test_serve_takes_a_port_to_listen_on():
    assert build_parser().parse_args(["serve", "--http", "8765"]).http == 8765
    assert build_parser().parse_args(["serve"]).http is None


def test_over_http_a_sync_tool_runs_off_the_event_loop_one_at_a_time():
    from lynx.server import LynxServer
    mcp = LynxServer("t")
    mcp.offload_sync_tools()
    seen, running, overlap = [], [0], [False]
    lock = threading.Lock()

    @mcp.tool(name="slow", description="d")
    def slow(n: int) -> str:
        with lock:
            running[0] += 1
            overlap[0] |= running[0] > 1
        seen.append(threading.current_thread() is threading.main_thread())
        import time
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return str(n * 2)

    async def main():
        loop_free = []
        async with anyio.create_task_group() as tg:
            for i in range(3):
                tg.start_soon(mcp.call_tool, "slow", {"n": i})
            await anyio.sleep(0.01)
            loop_free.append(True)          # the loop ran while the tools did
        out = await mcp.call_tool("slow", {"n": 21})
        return loop_free, out

    loop_free, out = anyio.run(main)
    assert loop_free == [True]
    assert seen and not any(seen), "a tool ran on the event loop's thread"
    assert not overlap[0], "two tools ran at once"
    assert "42" in str(out)


def test_over_http_a_read_does_not_wait_for_a_rebuild_and_reads_queue_among_themselves():
    from mcp.types import ToolAnnotations
    from lynx.server import LynxServer
    mcp = LynxServer("t")
    mcp.offload_sync_tools()
    read_only = ToolAnnotations(readOnlyHint=True)
    rebuilding, done = threading.Event(), threading.Event()
    running, overlap = [0], [False]
    lock = threading.Lock()

    @mcp.tool(name="rebuild", description="d", annotations=ToolAnnotations(readOnlyHint=False))
    def rebuild() -> str:
        rebuilding.set()
        done.wait(5)                        # a rebuild that lasts until the reads are in
        return "rebuilt"

    @mcp.tool(name="search", description="d", annotations=read_only)
    def search(q: str) -> str:
        with lock:
            running[0] += 1
            overlap[0] |= running[0] > 1
        import time
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return "found " + q

    async def main():
        out = {}

        async def call(name, args, key):
            out[key] = str(await mcp.call_tool(name, args))

        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "rebuild", {}, "rebuild")
            await anyio.to_thread.run_sync(rebuilding.wait, 5)
            with anyio.fail_after(5):
                async with anyio.create_task_group() as reads:
                    for i in range(3):
                        reads.start_soon(call, "search", {"q": str(i)}, i)
            searched_while_rebuilding = not done.is_set() and "rebuild" not in out
            done.set()
        return out, searched_while_rebuilding

    out, searched_while_rebuilding = anyio.run(main)
    assert searched_while_rebuilding, "the searches waited for the rebuild"
    assert all(f"found {i}" in out[i] for i in range(3))
    assert "rebuilt" in out["rebuild"]
    assert not overlap[0], "two read-only tools ran at once"


def test_over_http_requests_from_another_host_or_origin_are_refused():
    # A web page can reach a loopback server through DNS rebinding: the
    # browser sends the attacker's host name. FastMCP on 127.0.0.1 checks the
    # Host and Origin headers from mcp 1.23, which pyproject requires.
    from lynx.server import LynxServer
    security = LynxServer("t").settings.transport_security
    assert security is not None and security.enable_dns_rebinding_protection
    assert set(security.allowed_hosts) == {"127.0.0.1:*", "localhost:*", "[::1]:*"}
