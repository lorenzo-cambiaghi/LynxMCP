"""The LynxManager guide and its pictures, made by driving the real UI.

    python scripts/manager_guide.py            # writes docs/MANAGER.md and docs/img/manager/
    python scripts/manager_guide.py --keep     # keeps the sandbox, for a look afterwards

What it does: copies Lynx's own source into a sandbox as the sample project,
starts `lynx manager ui` on an empty config there, and walks the interface
the way a new user would, in a headless browser: the first launch, adding
the project, the first search, connecting an editor, then the Expert pages.
Every step is a picture with numbered callouts, and the guide is written
from the same list of scenes, so text and pictures always match the UI they
were taken from. The Simple-mode pictures also make the tour GIF in the
README. Run it before a release, after any change to the UI.

Needs Playwright (`pip install playwright`, also in the `webdoc-js` extra).
It drives the Edge or Chrome already installed; without either, run
`python -m playwright install chromium` once.

Nothing personal reaches a picture. The manager runs with the home and
AppData folders pointed into the sandbox, so the editor cards and indexes it
sees are the sandbox's, never the user's. The sandbox, home and interpreter
paths are replaced with neutral ones before each shot, and the run stops if
a page still contains the user's name or home folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

os.environ.pop("SSLKEYLOGFILE", None)  # an antivirus injects it; OpenSSL then dies on Windows

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
IMG = DOCS / "img" / "manager"
GUIDE = DOCS / "MANAGER.md"
SOURCE = "myproject"
WIDTH, HEIGHT = 1280, 800

# What a reader sees instead of this machine's paths.
NEUTRAL_PROJECTS = r"C:\Users\you\projects"
NEUTRAL_LYNX = r"C:\Users\you\.lynx"
NEUTRAL_PYTHON = r"C:\Users\you\AppData\Roaming\uv\tools\lynx-mcp\Scripts\python.exe"
NEUTRAL_HOME = r"C:\Users\you"
NEUTRAL_INSTALL = r"C:\Users\you\AppData\Roaming\uv\tools\lynx-mcp\Lib\site-packages"
NEUTRAL_REPO = r"C:\Users\you\LynxMCP"


@dataclass
class Scene:
    """One step of the guide: where to go, what to do there, what to point at."""
    name: str
    mode: str                       # "simple" or "expert"
    title: str
    text: str
    path: str
    act: Optional[Callable] = None  # (page, run) -> None, before the picture
    # (selector, what it is[, side]): the number sits beside the element, on
    # the "left" (the default), "right", "above" or "below", at the "end" of a
    # full-width row (inside, on the right), or at the "top-left" of a tall
    # block, so it never covers the element's text.
    callouts: List[tuple] = field(default_factory=list)
    full_page: bool = False
    in_tour: bool = False           # a frame of the README GIF


# --------------------------------------------------------------------------
# What the scenes do on the page
# --------------------------------------------------------------------------

def add_project(page, run):
    page.fill("#path-input", str(run.project))
    page.click("#detect-btn")
    page.wait_for_function("document.querySelector('#detect-result').innerText.includes('Detected')"
                           " || document.querySelector('#extensions-input').value.length > 0",
                           timeout=30_000)
    page.fill("#name-input", SOURCE)


def submit_and_build(page, run):
    """Not a picture of its own: adds the project and waits for its index.
    The picture showed neutral paths, so the real folder goes back first."""
    page.evaluate("document.querySelectorAll('.lynx-guide-callout').forEach(e => e.remove())")
    page.fill("#path-input", str(run.project))
    page.fill("#name-input", SOURCE)
    page.click("#submit-btn")
    deadline = time.time() + 600
    last, steady = -1, 0
    while time.time() < deadline:
        status = run.api(f"/api/sources/{SOURCE}/status") or {}
        count = status.get("chunk_count") or 0
        steady = steady + 1 if count and count == last else 0
        if steady >= 3:
            return
        last = count
        time.sleep(2)
    raise RuntimeError("the sample project was not indexed in 10 minutes")


def search(page, run):
    page.fill("form[hx-post='/api/playground/search'] input[name='query']",
              "where the prompt hook builds its map of the code")
    page.click("form[hx-post='/api/playground/search'] button[type='submit']")
    page.wait_for_selector("#search-results *", timeout=60_000)


def describe(page, run):
    page.click("button[data-tab='code']")
    form = "form[hx-post='/api/playground/describe_symbol']"
    page.fill(f"{form} input[name='symbol']", "fuse")
    page.fill(f"{form} input[name='file']", "prompt_hook")
    page.click(f"{form} button[type='submit']")
    page.wait_for_selector("#describe-results *", timeout=60_000)


def callers(page, run):
    page.click("button[data-tab='graph']")
    page.select_option("#pg-graph-op", "callers")
    page.fill("form[hx-post='/api/playground/graph_query'] input[name='symbol']", "fuse")
    page.click("form[hx-post='/api/playground/graph_query'] button[type='submit']")
    page.wait_for_selector("#graph-results *", timeout=60_000)


SCENES = [
    Scene("01-first-launch", "simple", "First launch",
          "On a new install the dashboard lists the three steps to a working setup. "
          "The sidebar switches between Simple and Expert mode: Simple shows what "
          "everyone needs, Expert adds diagnostics and the raw config.",
          "/", callouts=[("text=Get running in 3 steps", "the three steps, ticked off as you go"),
                         ("text=+ Add project", "add a project folder", "below"),
                         ("button:has-text('Expert')", "Simple or Expert mode", "right")],
          in_tour=True),
    Scene("02-add-project", "simple", "Add a project",
          "Pick the folder of your repository with Browse, or paste its path, then "
          "Detect: Lynx scans the folder and fills in the file types it found. The "
          "graph and the file watcher are on by default. Add starts the first index "
          "build right away.",
          "/sources/new/codebase", act=add_project,
          callouts=[("text=Folder path", "the project folder"),
                    ("#detect-btn", "Detect fills in the file types", "below"),
                    ("text=Project / source name", "the name your AI editor sees"),
                    ("#submit-btn", "add it and build the index")],
          in_tour=True),
    Scene("03-dashboard", "simple", "The project is ready",
          "Once the index is built the project shows as Ready. The cards at the top "
          "count indexed projects and connected editors, and show the result of the "
          "health check.",
          "/", act=None,
          callouts=[("text=Indexed projects", "how many projects are indexed"),
                    ("text=Connected AI editors", "editors that already use Lynx"),
                    ("text=Ready", "the project, indexed and searchable")],
          in_tour=True),
    Scene("04-search", "simple", "Try a search",
          "The Playground runs the same search your AI editor gets over MCP. Ask in "
          "plain words, or paste a symbol name: each result is a whole function or "
          "class, with its file, lines and score.",
          "/playground", act=search,
          callouts=[("#pg-source", "the project to search", "below"),
                    ("form[hx-post='/api/playground/search'] input[name='query']", "a question in plain words, or a name"),
                    ("#search-results", "results with file, lines and code", "top-left")],
          in_tour=True),
    Scene("05-connect", "simple", "Connect your AI editor",
          "Connect AI writes the Lynx entry into your editor's MCP config with one "
          "click, keeps a backup, and leaves your other servers alone. Editors that "
          "are not installed are listed at the bottom; any MCP client can use the "
          "snippet in the last card.",
          "/integrations",
          callouts=[("button:has-text('Connect automatically')", "one click per editor"),
                    ("button:has-text('Copy snippet') >> visible=true", "the same entry, for any other MCP client")],
          in_tour=True),
    Scene("06-expert-dashboard", "expert", "Expert mode",
          "Expert mode adds the details: chunks and graph size per project, which "
          "process keeps each index current, and the Config, Doctor and OpenAPI pages.",
          "/", callouts=[("button:has-text('Expert')", "the switch", "right"),
                         ("text=Served by", "who keeps this index current"),
                         ("text=Advanced", "Config, Doctor, OpenAPI", "end")]),
    Scene("07-project", "expert", "A project's page",
          "Every project has a page with its index and graph status, a rebuild button "
          "and the way to remove it, with or without its index.",
          "/sources/" + SOURCE,
          callouts=[("text=Status", "index status"),
                    ("text=Code knowledge graph", "the code graph"),
                    ("text=Rebuild index", "rebuild"),
                    ("text=Remove this source", "remove it")]),
    Scene("08-code-aware", "expert", "Code-aware tools",
          "The Code-aware tab runs the tools that read the code graph: here "
          "`describe_symbol`, a function's code, callers and callees in one answer. "
          "The output is the text your AI editor receives.",
          "/playground", act=describe,
          callouts=[("button[data-tab='code']", "Code-aware tools", "above"),
                    ("#describe-results", "the answer the AI gets", "top-left")],
          full_page=True),
    Scene("09-graph", "expert", "The code graph",
          "The Graph tab asks the graph directly: callers, callees, subclasses, "
          "imports, paths between two symbols. Here, who calls `fuse`.",
          "/playground", act=callers,
          callouts=[("#pg-graph-op", "the question"),
                    ("#graph-results", "resolved edges, with file and line", "top-left")]),
    Scene("10-doctor", "expert", "Doctor",
          "Doctor runs the same checks as `lynx manager doctor`: the config, the "
          "embedding model, each index and the editors connected.",
          "/doctor", full_page=True),
    Scene("11-config", "expert", "Config",
          "The raw `config.json`, with the validation the command line uses and a "
          "backup of the previous version on every save.",
          "/config", full_page=True),
]


# --------------------------------------------------------------------------
# The sandbox and the manager
# --------------------------------------------------------------------------

class Run:
    def __init__(self, keep: bool):
        self.keep = keep
        self.root = Path(tempfile.mkdtemp(prefix="lynx-guide-"))
        self.project = self.root / "projects" / SOURCE
        self.config = self.root / "lynx" / "config.json"
        self.port = _free_port()
        self.ui = None

    def setup(self):
        shutil.copytree(REPO / "src" / "lynx", self.project / "lynx",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "static", "templates"))
        for d in ("home", "appdata", "localappdata"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text(json.dumps({
            "config_version": 2,
            "storage_path": str(self.root / "lynx" / "storage"),
            "sources": {},
        }, indent=2), encoding="utf-8")

    def env(self):
        env = dict(os.environ)
        for k in ("RAG_CONFIG_PATH", "SSLKEYLOGFILE", "LYNX_TOOL_PROFILE"):
            env.pop(k, None)
        env["HF_HOME"] = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        env.update(USERPROFILE=str(self.root / "home"), HOME=str(self.root / "home"),
                   APPDATA=str(self.root / "appdata"), LOCALAPPDATA=str(self.root / "localappdata"))
        return env

    def start(self):
        log = open(self.root / "manager.log", "w", encoding="utf-8")
        self.ui = subprocess.Popen(
            [sys.executable, "-m", "lynx", "manager", "ui", "--config", str(self.config),
             "--port", str(self.port), "--no-browser"],
            env=self.env(), stdout=log, stderr=subprocess.STDOUT)
        for _ in range(180):
            if self.api("/api/health") is not None:
                return
            if self.ui.poll() is not None:
                raise RuntimeError(f"the manager exited; see {self.root / 'manager.log'}")
            time.sleep(1)
        raise RuntimeError("the manager did not answer in 3 minutes")

    def api(self, path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
                return json.loads(r.read().decode("utf-8") or "null")
        except Exception:
            return None

    def stop(self):
        if self.ui and self.ui.poll() is None:
            self.ui.terminate()
            try:
                self.ui.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.ui.kill()
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)

    def replacements(self):
        """Longest first, each in its plain and its JSON-escaped spelling."""
        pairs = [
            (str(self.project.parent), NEUTRAL_PROJECTS),
            (str(self.config.parent), NEUTRAL_LYNX),
            (str(self.root / "home"), NEUTRAL_HOME),
            (sys.executable, NEUTRAL_PYTHON),
            (str(REPO / "src"), NEUTRAL_INSTALL),
            (str(REPO), NEUTRAL_REPO),
            (str(Path.home()), NEUTRAL_HOME),
        ]
        out = []
        for real, neutral in sorted(pairs, key=lambda p: -len(p[0])):
            out.append((real, neutral))
            out.append((real.replace("\\", "\\\\"), neutral.replace("\\", "\\\\")))
            out.append((real.replace("\\", "/"), neutral.replace("\\", "/")))
        return out

    def forbidden(self):
        """Strings that must not appear in a picture: this machine's home,
        repository and sandbox paths, in every spelling a page may use."""
        out = set()
        for path in (str(Path.home()), str(REPO), str(self.root)):
            out |= {path, path.replace("\\", "\\\\"), path.replace("\\", "/")}
        return sorted(out)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# On the page: neutral paths, callouts, the privacy check
# --------------------------------------------------------------------------

SCRUB_JS = """
(pairs) => {
  const swap = (s) => { for (const [a, b] of pairs) s = s.split(a).join(b); return s; };
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walk.nextNode(); n; n = walk.nextNode()) n.nodeValue = swap(n.nodeValue);
  for (const el of document.querySelectorAll('*')) {
    for (const at of Array.from(el.attributes)) {
      const v = swap(at.value); if (v !== at.value) el.setAttribute(at.name, v);
    }
    if ('value' in el && typeof el.value === 'string' && el.value) el.value = swap(el.value);
  }
}
"""

CALLOUT_JS = """
(items) => {
  document.querySelectorAll('.lynx-guide-callout').forEach(e => e.remove());
  items.forEach(([n, x, y]) => {
    const b = document.createElement('div');
    b.className = 'lynx-guide-callout';
    b.textContent = n;
    Object.assign(b.style, {position: 'absolute', left: (x - 13) + 'px', top: (y - 13) + 'px',
      width: '26px', height: '26px', borderRadius: '13px', background: '#e8742c', color: '#fff',
      font: '700 14px/26px system-ui, sans-serif', textAlign: 'center', zIndex: 99999,
      boxShadow: '0 0 0 3px #fff, 0 2px 6px rgba(0,0,0,.35)'});
    document.body.appendChild(b);
  });
}
"""

LEAK_JS = """
(words) => {
  const html = document.documentElement.outerHTML;
  const vals = Array.from(document.querySelectorAll('input, textarea')).map(e => e.value).join('\\n');
  const all = (html + '\\n' + vals).toLowerCase();
  for (const w of words) { const i = all.indexOf(w.toLowerCase()); if (i >= 0) return [w, all.slice(Math.max(0, i - 80), i + 80)]; }
  return null;
}
"""


def place_callouts(page, callouts):
    items = []
    for n, (selector, _label, *side) in enumerate(callouts, 1):
        box = page.locator(selector).first.bounding_box()
        if box is None:
            raise RuntimeError(f"callout {n}: nothing visible matches {selector!r}")
        sx, sy = page.evaluate("[window.scrollX, window.scrollY]")
        x, y, w, h = box["x"] + sx, box["y"] + sy, box["width"], box["height"]
        side = side[0] if side else "left"
        if side == "left" and x - 22 < 14:
            side = "right"
        cx, cy = {
            "left": (x - 22, y + h / 2),
            "right": (x + w + 22, y + h / 2),
            "above": (x + w / 2, y - 18),
            "below": (x + w / 2, y + h + 18),
            "end": (x + w - 20, y + h / 2),
            "top-left": (x - 22, y + 14),
        }[side]
        items.append([n, cx, cy])
    page.evaluate(CALLOUT_JS, items)


def shoot(page, run, scene):
    page.wait_for_load_state("networkidle")
    page.evaluate(SCRUB_JS, run.replacements())
    leak = page.evaluate(LEAK_JS, run.forbidden())
    if leak:
        raise RuntimeError(f"{scene.name}: the page still contains {leak[0]!r}: ...{leak[1]}...")
    place_callouts(page, scene.callouts)
    page.evaluate("window.scrollTo(0, 0)")
    IMG.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(IMG / f"{scene.name}.png"), full_page=scene.full_page)


def launch_browser(p):
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel, headless=True) if channel else p.chromium.launch(headless=True)
        except Exception:
            continue
    raise RuntimeError("no browser: install Edge or Chrome, or run `python -m playwright install chromium`")


# --------------------------------------------------------------------------
# The guide and the tour
# --------------------------------------------------------------------------

def write_guide():
    lines = [
        "<!-- Written by scripts/manager_guide.py from the running UI: change the scenes there, then run it. -->",
        "",
        "# LynxManager",
        "",
        "LynxManager is the local web page that sets Lynx up without a terminal: add a "
        "project, try a search, connect your AI editor. Start it with `lynx manager ui` "
        "(or the desktop installer); it listens on `127.0.0.1` only.",
        "",
        "The pictures below were taken from the current version, on a copy of Lynx's own "
        "source as the sample project.",
        "",
    ]
    for mode, heading in (("simple", "Simple mode"), ("expert", "Expert mode")):
        lines += [f"## {heading}", ""]
        for s in (s for s in SCENES if s.mode == mode):
            lines += [f"### {s.title}", "", s.text, "",
                      f'<img src="img/manager/{s.name}.png" alt="{s.title}" width="900">', ""]
            if s.callouts:
                lines += [f"{n}. {c[1]}" for n, c in enumerate(s.callouts, 1)]
                lines.append("")
    GUIDE.write_text("\n".join(lines), encoding="utf-8")


def write_tour():
    """The README GIF: the Simple-mode pictures, cropped to the window."""
    from PIL import Image
    frames = []
    for s in (s for s in SCENES if s.in_tour):
        img = Image.open(IMG / f"{s.name}.png").convert("RGB")
        img = img.crop((0, 0, WIDTH, min(img.height, HEIGHT)))
        img = img.resize((960, int(img.height * 960 / WIDTH)), Image.LANCZOS)
        frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(IMG / "tour.gif", save_all=True, append_images=frames[1:],
                   duration=2600, loop=0, optimize=True)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep", action="store_true", help="keep the sandbox and print where it is")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    run = Run(keep=args.keep)
    try:
        run.setup()
        run.start()
        with sync_playwright() as p:
            browser = launch_browser(p)
            for mode in ("simple", "expert"):
                ctx = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT})
                ctx.add_init_script(f"localStorage.setItem('lynx_ui_mode', '{mode}')")
                page = ctx.new_page()
                for scene in (s for s in SCENES if s.mode == mode):
                    page.goto(f"http://127.0.0.1:{run.port}{scene.path}")
                    page.wait_for_load_state("networkidle")
                    if scene.act:
                        scene.act(page, run)
                    shoot(page, run, scene)
                    print(f"  {scene.name}", file=sys.stderr)
                    if scene.name == "02-add-project":
                        submit_and_build(page, run)
                ctx.close()
            browser.close()
        write_guide()
        write_tour()
        print(f"wrote {GUIDE.relative_to(REPO)} and {len(SCENES)} pictures in {IMG.relative_to(REPO)}",
              file=sys.stderr)
    finally:
        run.stop()
        if args.keep:
            print(f"sandbox kept: {run.root}", file=sys.stderr)


if __name__ == "__main__":
    main()
