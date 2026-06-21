"""Repository orientation: a readable "what is this and where do I start" map.

`build_overview(root)` scans a codebase directory (no index, no model — pure
stdlib + filesystem) and returns languages, manifests, detected frameworks,
likely entry points, and suggested build/test/run commands. It powers the
`repo_overview` MCP tool: the first thing an AI wants when it lands in an
unfamiliar repository.

Everything here is heuristic and best-effort: a missing framework or command
is a non-event, never an error. The goal is a fast, useful first orientation,
not a guarantee.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


# Directories we never descend into (cost + noise: deps, build output, VCS).
_PRUNE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".idea", ".vscode", "dist", "build", "out", "target",
    ".next", ".nuxt", "bin", "obj", ".gradle", ".mvn", "vendor",
    ".pytest_cache", ".mypy_cache", ".tox", "coverage", ".terraform",
}

# Extension -> human language name. Kept aligned with the tree-sitter grammars
# Lynx ships, plus a few config/markup extensions worth surfacing.
_EXT_LANG = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
    ".cs": "C#", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".m": "Objective-C", ".c": "C", ".h": "C/C++ header",
    ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".scala": "Scala",
    ".sc": "Scala", ".lua": "Lua", ".sh": "Shell", ".bash": "Shell",
    ".sql": "SQL",
}

# Manifest filename -> the ecosystem it signals.
_MANIFESTS = {
    "package.json": "node", "pyproject.toml": "python",
    "setup.py": "python", "setup.cfg": "python", "requirements.txt": "python",
    "Pipfile": "python", "go.mod": "go", "Cargo.toml": "rust",
    "pom.xml": "java-maven", "build.gradle": "java-gradle",
    "build.gradle.kts": "java-gradle", "Gemfile": "ruby",
    "composer.json": "php", "Makefile": "make", "CMakeLists.txt": "cmake",
    "Dockerfile": "docker", "*.csproj": "dotnet", "*.sln": "dotnet",
}


def _walk(root: Path, max_files: int = 30000):
    """Yield files under `root`, pruning dependency/build/VCS dirs in place."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _PRUNE_DIRS and not d.startswith(".")
            or d in {".github"}  # keep CI config visible
        ]
        for f in filenames:
            yield Path(dirpath) / f
            count += 1
            if count >= max_files:
                return


def _detect_languages(files: list, top_n: int = 8) -> list:
    from collections import Counter
    counter: Counter = Counter()
    for p in files:
        lang = _EXT_LANG.get(p.suffix.lower())
        if lang:
            counter[lang] += 1
    return [
        {"language": lang, "files": n}
        for lang, n in counter.most_common(top_n)
    ]


def _detect_manifests(root: Path, files: list) -> list:
    """Manifest files present at the root (or, for project files like
    *.csproj, anywhere shallow). Returns sorted distinct manifest names."""
    found: set = set()
    root_names = {p.name for p in root.iterdir()} if root.is_dir() else set()
    for name, _eco in _MANIFESTS.items():
        if name.startswith("*."):
            ext = name[1:]
            if any(p.suffix == ext for p in files):
                found.add(name)
        elif name in root_names:
            found.add(name)
    return sorted(found)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _detect_frameworks(root: Path, files: list, manifests: list) -> list:
    """Best-effort framework detection from manifest contents + filenames."""
    frameworks: set = set()

    # Node: read dependencies from package.json.
    pkg = root / "package.json"
    if pkg.is_file():
        data = _read_json(pkg)
        deps = {}
        deps.update(data.get("dependencies") or {})
        deps.update(data.get("devDependencies") or {})
        signals = {
            "react": "React", "next": "Next.js", "vue": "Vue",
            "nuxt": "Nuxt", "@angular/core": "Angular", "svelte": "Svelte",
            "express": "Express", "@nestjs/core": "NestJS",
            "electron": "Electron", "vite": "Vite",
        }
        for dep, label in signals.items():
            if dep in deps:
                frameworks.add(label)

    # Python: cheap textual scan of dependency declarations.
    py_text = ""
    for fname in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile"):
        p = root / fname
        if p.is_file():
            try:
                py_text += "\n" + p.read_text(encoding="utf-8").lower()
            except Exception:
                pass
    py_signals = {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
        "pyramid": "Pyramid", "tornado": "Tornado", "scrapy": "Scrapy",
        "streamlit": "Streamlit", "torch": "PyTorch", "tensorflow": "TensorFlow",
    }
    for dep, label in py_signals.items():
        if dep in py_text:
            frameworks.add(label)
    if (root / "manage.py").is_file():
        frameworks.add("Django")

    # .NET: a web SDK csproj signals ASP.NET.
    for p in files:
        if p.suffix == ".csproj":
            try:
                if "Microsoft.NET.Sdk.Web" in p.read_text(encoding="utf-8"):
                    frameworks.add("ASP.NET Core")
                    break
            except Exception:
                pass
    # Unity: a csproj project with an Assets/ dir is a Unity project.
    if (root / "Assets").is_dir() and any(p.suffix == ".csproj" for p in files):
        frameworks.add("Unity")

    return sorted(frameworks)


# (filename, human description) heuristics for likely entry points.
_ENTRY_HINTS = [
    ("manage.py", "Django management entrypoint"),
    ("wsgi.py", "WSGI app entrypoint"),
    ("asgi.py", "ASGI app entrypoint"),
    ("__main__.py", "Python `python -m` entrypoint"),
    ("main.py", "Python main"),
    ("app.py", "Python app entrypoint"),
    ("main.go", "Go main"),
    ("Program.cs", ".NET program entrypoint"),
    ("index.js", "Node entrypoint"),
    ("index.ts", "Node/TS entrypoint"),
    ("server.js", "Node server entrypoint"),
    ("server.ts", "Node/TS server entrypoint"),
    ("Main.java", "Java main"),
]


def _detect_entry_points(root: Path, files: list, limit: int = 8) -> list:
    by_name: dict = {}
    for p in files:
        by_name.setdefault(p.name, p)
    out = []
    for name, hint in _ENTRY_HINTS:
        if name in by_name:
            try:
                rel = str(by_name[name].relative_to(root))
            except ValueError:
                rel = by_name[name].name
            out.append({"file": rel, "hint": hint})
            if len(out) >= limit:
                break
    return out


def _detect_commands(root: Path, manifests: list, files: list) -> dict:
    """Suggested build / test / run commands, derived from manifests."""
    build, test, run = [], [], []

    pkg = root / "package.json"
    if pkg.is_file():
        scripts = (_read_json(pkg).get("scripts") or {})
        for s in ("build",):
            if s in scripts:
                build.append(f"npm run {s}")
        for s in ("test",):
            if s in scripts:
                test.append(f"npm run {s}")
        for s in ("start", "dev", "serve"):
            if s in scripts:
                run.append(f"npm run {s}")

    if any(m in manifests for m in ("pyproject.toml", "setup.py", "requirements.txt")):
        build.append("pip install -e .  (or: uv pip install -e .)")
        test.append("pytest")
    if "go.mod" in manifests:
        build.append("go build ./...")
        test.append("go test ./...")
    if "Cargo.toml" in manifests:
        build.append("cargo build")
        test.append("cargo test")
    if any(m in manifests for m in ("*.csproj", "*.sln")):
        build.append("dotnet build")
        test.append("dotnet test")
    if "pom.xml" in manifests:
        build.append("mvn compile")
        test.append("mvn test")
    if any(m in manifests for m in ("build.gradle", "build.gradle.kts")):
        build.append("gradle build")
        test.append("gradle test")
    if "Makefile" in manifests:
        build.append("make")

    # De-dup while preserving order.
    def _uniq(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return {"build": _uniq(build), "test": _uniq(test), "run": _uniq(run)}


def build_overview(root) -> dict:
    """Produce the orientation map for a codebase rooted at `root`."""
    root = Path(root)
    if not root.is_dir():
        return {"root": str(root), "error": "path is not a directory"}

    files = list(_walk(root))
    manifests = _detect_manifests(root, files)
    return {
        "root": str(root),
        "file_count": len(files),
        "languages": _detect_languages(files),
        "manifests": manifests,
        "frameworks": _detect_frameworks(root, files, manifests),
        "entry_points": _detect_entry_points(root, files),
        "commands": _detect_commands(root, manifests, files),
    }
