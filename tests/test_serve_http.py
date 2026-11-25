"""`lynx serve --http PORT`: one server for every session of a machine.

A stdio server is one process per client, each with its own embedding model
and graph in memory: on a machine running a dozen agents on a 1.4-million-line
library that is a dozen copies of the same indexes. Over HTTP every session
connects to one process. Its tools must not block the event loop, or a long
search of one client would hold up the handshake of the next: sync tools run
on a worker thread, one at a time, as they do on stdio.
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
