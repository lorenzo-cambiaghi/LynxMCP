"""Building blocks to combine Lynx (semantic code search) and Coral (SQL over
live tools) from Python.

This is NOT a framework and NOT a query engine — just two thin clients you
compose yourself. Coral resolves table-function arguments to constants at plan
time, so it can't drive `lynx.search` from another table's column (a per-row /
lateral correlation). You can, trivially, in a few lines of Python: query Coral
for the rows, then batch them into Lynx. That's the whole idea — here are the
bricks; build whatever ad-hoc logic you want.

Stdlib only (urllib + subprocess + json) so there's nothing to install.

    from toolkit import Lynx, Coral

    lynx  = Lynx(port=8765)
    coral = Coral(exe=r"C:\\coral\\coral.exe")  # the standalone Coral binary

    rows = coral.sql("SELECT number, title FROM github.pulls WHERE state='open'")
    hits = lynx.search_batch([r["title"] for r in rows], source="framework", top_k=3)
    for row, res in zip(rows, hits):
        print(row["number"], "->", [h["file"] for h in res["hits"]])
"""
from __future__ import annotations

import os

# Defensive (Windows): HTTPS-inspecting antivirus (Avast/AVG) injects
# SSLKEYLOGFILE into every process; on a bundled/standalone interpreter without
# OPENSSL_Applink, Python's `ssl` aborts the process on first TLS use. We never
# need a TLS key-log, so drop it before importing anything that touches `ssl`.
os.environ.pop("SSLKEYLOGFILE", None)

import json
import subprocess
import urllib.parse
import urllib.request


class Lynx:
    """Thin client for Lynx's local JSON API (`lynx manager ui` must be running).

    `search` / `search_batch` return the API's result rows verbatim (each hit
    carries file, file_path, symbol, kind, language, start_line, end_line,
    score, content). Everything is local; nothing leaves the machine.
    """

    def __init__(self, port: int = 8765, host: str = "127.0.0.1", timeout: float = 60.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.load(r)

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def sources(self) -> list:
        """List indexed sources: name, type, location, chunk_count, last_update."""
        return self._get("/api/v1/sources")["sources"]

    def search(self, q: str, source: str | None = None, top_k: int = 8) -> list:
        """One semantic search. Returns a list of hit rows (ranked)."""
        qs = {"q": q, "top_k": top_k}
        if source:
            qs["source"] = source
        return self._get("/api/v1/search?" + urllib.parse.urlencode(qs))["results"]

    def search_batch(self, queries: list, source: str | None = None, top_k: int = 8) -> list:
        """Search many queries in ONE call (Lynx embeds them as a batch).

        Returns a list aligned to `queries`: `[{"query": ..., "hits": [...]}, ...]`.
        This is the fast primitive for a row-driven fan-out.
        """
        return self._post(
            "/api/v1/search",
            {"queries": list(queries), "source": source, "top_k": top_k},
        )["results"]


class Coral:
    """Thin wrapper around the Coral CLI (`coral sql <query> --format json`).

    Returns each query as a list of row dicts. Pass `exe` if the binary isn't on
    PATH, and `inputs` to set source variables (e.g. LYNX_PORT) for this process.
    """

    def __init__(self, exe: str = "coral", inputs: dict | None = None, timeout: float = 120.0):
        self.exe = exe
        self.timeout = timeout
        self.env = {**os.environ, **{k: str(v) for k, v in (inputs or {}).items()}}

    def sql(self, query: str) -> list:
        """Run a SQL query through Coral, return rows as a list of dicts."""
        proc = subprocess.run(
            [self.exe, "sql", query, "--format", "json"],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"coral sql failed: {(proc.stderr or proc.stdout).strip()}"
            )
        out = proc.stdout.strip()
        if not out:
            return []
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"coral did not return JSON (is --format json supported by your "
                f"Coral version?): {out[:200]}"
            ) from e


def _demo():
    """Runnable example — the per-row pattern Coral can't express, done here in
    a few lines. Uses an inline VALUES list as the 'live data' so it runs with
    just Lynx + Coral (no credentials). Swap the VALUES for a real source, e.g.
    `SELECT id, title FROM linear.issues WHERE assignee = 'me'`.
    """
    import sys

    coral_exe = os.environ.get("CORAL_EXE", "coral")
    lynx = Lynx(port=int(os.environ.get("LYNX_PORT", "8765")))
    coral = Coral(exe=coral_exe, inputs={"LYNX_PORT": os.environ.get("LYNX_PORT", "8765")})

    # 1. "Live data" rows (here: fake tickets). In real use this is a Coral
    #    source like github.pulls / sentry.issues / linear.issues.
    tickets = coral.sql(
        """
        SELECT * FROM (VALUES
            ('TICK-1', 'camera zoom is clamped wrong'),
            ('TICK-2', 'object pool runs out of capacity'),
            ('TICK-3', 'quaternion slerp jitters')
        ) AS t(id, title)
        """
    )

    # 2. Drive a semantic code search from each row's title — one batched call.
    results = lynx.search_batch(
        [t["title"] for t in tickets], source="framework", top_k=2
    )

    # 3. Join: each ticket with its most likely code locations.
    print(f"{'ticket':8}  {'query':36}  -> top code hits")
    for t, res in zip(tickets, results):
        files = [h["file"] for h in res["hits"]]
        print(f"{t['id']:8}  {t['title'][:36]:36}  -> {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
