"""SOLID findings for one package of a C# codebase, each with its fence.

    python -m lynx.audit.solid CONFIG_OR_ROOT PACKAGE [--out FILE]

A finding says which principle a type breaks, where, and why, with the
numbers. Thresholds are relative to the codebase itself (a high percentile
of the graph's own sizes, with a floor): "far beyond what this codebase
usually does", not a number from a book. Each finding carries the fence of
its refactor, from the graph: the files inside, the members other packages
use (the contract to keep, with how many callers), the projects that
reference the package (to build) and the test files that reach the type
(to run). An AI refactors inside the fence; the audit run again shows the
finding gone and nothing new inside it.

The graph gives sizes, callers, type references and bases; the package's
own files are parsed once more for what the graph does not keep (fields,
modifiers, branches, switches, bodies that only throw).
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .facts import LAZY, read_file

NAMESPACE_KINDS = {"namespace_declaration", "file_scoped_namespace_declaration"}
INBOUND = {"calls", "uses_type", "inherits"}

# Attributes whose method the runtime calls with no caller in the code.
RUNTIME_ENTRIES = ("ModuleInitializer", "UnmanagedCallersOnly")
MUTABLE_COLLECTIONS = ("List<", "Dictionary<", "HashSet<", "Queue<", "Stack<", "ConcurrentDictionary<",
                       "ConcurrentQueue<", "ConcurrentBag<", "SortedDictionary<", "SortedList<")
# Verbs and glue words left out when method names are read for the jobs of a type.
NAME_GLUE = {"get", "set", "try", "on", "is", "has", "can", "make", "build", "create", "add", "remove", "update",
             "handle", "send", "read", "write", "find", "to", "from", "of", "for", "with", "and", "or", "the",
             "a", "an", "in", "at", "by", "async", "do", "run", "start", "stop", "apply", "check", "new", "reset",
             "clear", "dispose", "init", "begin", "end", "next", "all", "if", "when", "as", "use", "load", "save"}
PRINCIPLES = {"S": "Single responsibility", "O": "Open/closed", "L": "Liskov substitution",
              "I": "Interface segregation", "D": "Dependency inversion", "X": "Next to SOLID"}


@dataclass
class Finding:
    principle: str
    title: str
    where: str
    evidence: str
    type_key: tuple
    severity: float
    fence: dict = field(default_factory=dict)
    score: float = 0.0


def _pct(values, q, floor):
    xs = sorted(values)
    return max(floor, xs[min(len(xs) - 1, int(len(xs) * q))]) if xs else floor


class _Codebase:
    """Paths, projects and type identities of a graph built over `root`."""

    def __init__(self, G, root: Path):
        self.G = G
        self.root = os.path.normcase(os.path.normpath(str(root)))
        self._proj: dict = {}

    def rel(self, f: str) -> str:
        f = os.path.normpath(f or "")
        if os.path.normcase(f).startswith(self.root + os.sep):
            return f[len(self.root) + 1:].replace("\\", "/")
        return f.replace("\\", "/")

    def project(self, f: str) -> str:
        d = os.path.dirname(os.path.normpath(f))
        if d in self._proj:
            return self._proj[d]
        found, probe = "", d
        while probe and os.path.normcase(probe).startswith(self.root):
            try:
                hit = next((e.name for e in os.scandir(probe) if e.name.endswith(".csproj")), None)
            except OSError:
                hit = None
            if hit:
                found = hit[:-len(".csproj")]
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        self._proj[d] = found or self.rel(d).split("/")[0]
        return self._proj[d]

    def is_test(self, f: str) -> bool:
        r = self.rel(f)
        return r.startswith("tests/") or "/tests/" in r or self.project(f).endswith(".Tests")

    def type_key(self, nid) -> tuple:
        n = self.G.nodes[nid]
        ns, label = n.get("namespace", ""), n.get("label") or ""
        return (ns, label[len(ns) + 1:] if ns and label.startswith(ns + ".") else label)


def _project_refs(root: Path) -> dict:
    refs = {}
    for csproj in root.rglob("*.csproj"):
        if any(p in ("bin", "obj") for p in csproj.parts):
            continue
        try:
            tree = ET.parse(csproj)
        except (ET.ParseError, OSError):
            continue
        refs[csproj.stem] = {Path(el.get("Include").replace("\\", "/")).stem for el in tree.iter()
                             if el.tag.endswith("ProjectReference") and el.get("Include")}
    return refs


def _churn(root: Path, package_rel: str) -> Counter:
    try:
        # --relative: paths relative to `root` even when it is a subfolder of
        # the repository, the way the fence names the files.
        out = subprocess.run(["git", "-C", str(root), "log", "--since=180 days ago", "--format=", "--name-only",
                              "--relative", "--", package_rel], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return Counter()
    return Counter(line.strip() for line in out.splitlines() if line.strip())


def _name_words(name: str) -> list:
    return [w.lower() for w in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", name)]


def name_areas(method_names, min_methods: int = 3) -> list:
    """The jobs a type's method names point to: the words (verbs and glue
    left out) shared by several methods, with those methods. Read as seams
    to look at, not as a verdict."""
    by_word = defaultdict(set)
    for m in set(method_names):
        for w in set(_name_words(m)) - NAME_GLUE:
            if len(w) > 2:
                by_word[w].add(m)
    areas, taken = [], set()
    for w, ms in sorted(by_word.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fresh = ms - taken
        if len(fresh) >= min_methods:
            areas.append((w, sorted(fresh)))
            taken |= fresh
    return areas


def audit(G, root: Path, package: str, top_contract: int = 6, repo: Path | None = None) -> tuple:
    """(findings, package_rel) for the package at `root/package` (or the
    folder under `root` named `package`)."""
    root = Path(root)
    cb = _Codebase(G, root)
    pkg_dir = root / package
    if not pkg_dir.is_dir():
        hits = [p for p in root.glob(f"*/{package}") if p.is_dir()]
        if not hits:
            raise FileNotFoundError(f"no folder {package!r} under {root}")
        pkg_dir = hits[0]
    package_rel = pkg_dir.relative_to(root).as_posix()
    if package_rel in ("", "."):
        inside = lambda f: True  # noqa: E731  the whole codebase is the package
    else:
        inside = lambda f: cb.rel(f).startswith(package_rel + "/")  # noqa: E731

    # The codebase's own sizes, from the graph.
    decls = defaultdict(list)
    members = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "contains" and G.nodes[v].get("kind") == "function":
            members[u] += 1
    for nid, n in G.nodes(data=True):
        if n.get("kind") == "class" and n.get("subkind") not in NAMESPACE_KINDS and n.get("file"):
            decls[cb.type_key(nid)].append(nid)
    src_types = {k: v for k, v in decls.items() if not cb.is_test(G.nodes[v[0]]["file"])}
    sizes = [sum(G.nodes[x]["end_line"] - G.nodes[x]["start_line"] + 1 for x in v) for v in src_types.values()]
    fn_lines = [n["end_line"] - n["start_line"] + 1 for _, n in G.nodes(data=True)
                if n.get("kind") == "function" and n.get("file") and not cb.is_test(n["file"])]
    iface_sizes = [members[v[0]] for v in src_types.values() if G.nodes[v[0]].get("subkind") == "interface_declaration"]
    big_type = _pct(sizes, 0.99, 800)
    long_method = _pct(fn_lines, 0.99, 80)
    big_iface = _pct(iface_sizes, 0.95, 10)
    depth_cap = 4

    owner = {}
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "contains" and G.nodes[u].get("kind") == "class" and G.nodes[v].get("kind") == "function":
            owner[v] = u

    refs = _project_refs(root)
    churn = _churn(Path(repo) if repo else root, package_rel)
    pkg_types = {k: v for k, v in decls.items() if inside(G.nodes[v[0]]["file"]) and not cb.is_test(G.nodes[v[0]]["file"])}
    pkg_projects = {cb.project(G.nodes[v[0]]["file"]) for v in pkg_types.values()}
    dependents = sorted(p for p, deps in refs.items() if deps & pkg_projects)

    def fence(key: tuple) -> dict:
        ids = set(decls.get(key, ()))
        member_ids = {f for f, o in owner.items() if o in ids}
        outside, contract, tests = set(), Counter(), set()
        for target in ids | member_ids:
            for src, _, d in G.in_edges(target, data=True):
                if d.get("relation") not in INBOUND:
                    continue
                sf = G.nodes[src].get("file") or ""
                if not sf or src in ids or src in member_ids:
                    continue
                if cb.is_test(sf):
                    tests.add(cb.rel(sf))
                elif not inside(sf):
                    outside.add(sf)
                    label = (G.nodes[target].get("label") or "").rsplit(".", 1)[-1]
                    contract[label if target in member_ids else "(the type)"] += 1
        files = sorted({cb.rel(G.nodes[x]["file"]) for x in ids})
        return {"files": files, "outside_files": len(outside),
                "outside_projects": sorted({cb.project(f) for f in outside}),
                "contract": contract.most_common(top_contract), "contract_total": len(contract),
                "build": dependents, "tests": sorted(tests), "churn": sum(churn.get(f, 0) for f in files)}

    # The package's files, parsed for the facts the graph does not keep.
    facts = {}
    for p in pkg_dir.rglob("*.cs"):
        if any(part in ("bin", "obj") for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        try:
            facts[rel] = read_file(p, rel)[2]
        except Exception:
            continue

    def key_of(t) -> tuple:
        return (t.namespace, t.name)

    by_key = defaultdict(list)
    for rel, types in facts.items():
        for t in types:
            by_key[key_of(t)].append(t)

    # The members each type of the package leaves abstract, by its simple name:
    # an empty override of one of them is "nothing to do here", not a member
    # that stopped doing its base's work.
    abstract_members = defaultdict(set)
    for ts in by_key.values():
        for t in ts:
            abstract_members[t.name.rsplit(".", 1)[-1]] |= {m.name for m in t.methods if "abstract" in m.modifiers}

    findings = []
    for key, ts in by_key.items():
        first = min(ts, key=lambda t: (t.file, t.line))
        at = f"{first.file}:{first.line}"
        lines = sum(t.end - t.line + 1 for t in ts)
        meths = [m for t in ts for m in t.methods]
        files = {t.file for t in ts}
        if first.kind in ("class", "struct", "record") and lines > big_type:
            areas = name_areas([m.name for m in meths])
            ev = f"{lines} lines over {len(files)} file(s), {len(meths)} methods, {sum(len(t.fields) for t in ts)} fields"
            if len(areas) >= 2:
                ev += "; the method names point to " + "; ".join(
                    f"{w} ({len(ms)}: {', '.join(ms[:3])}{', ...' if len(ms) > 3 else ''})" for w, ms in areas[:4])
            findings.append(Finding("S", f"{first.name} does several jobs", at, ev, key, 3.0 if len(areas) >= 2 else 2.0))
        for t in ts:                       # each partial declaration with its own file
            for m in t.methods:
                n = m.end - m.line + 1
                if n > long_method:
                    findings.append(Finding("S", f"{first.name}.{m.name} is long", f"{t.file}:{m.line}",
                                            f"{n} lines, {m.branches} branches (1% of this codebase's methods are over {long_method} lines)",
                                            key, 1.5))
        iface_members = sum(t.members for t in ts)
        if first.kind == "interface" and iface_members > big_iface:
            findings.append(Finding("I", f"{first.name} is a wide interface", at,
                                    f"{iface_members} members (95% of this codebase's interfaces have {big_iface} or fewer)", key, 2.0))
        writes = set().union(*(t.writes for t in ts))
        for t in ts:
            for m in t.methods:
                # A private helper that throws is how a type says "not here";
                # only a member a caller can reach through a contract breaks
                # substitution.
                reachable = (m.explicit or first.kind == "interface"
                             or m.modifiers & {"public", "protected", "internal", "override", "virtual", "abstract"})
                if not reachable:
                    continue
                if m.body == "empty" and any(m.name in abstract_members.get(base.split("<", 1)[0].rsplit(".", 1)[-1], ())
                                             for base in t.bases):
                    continue
                if m.body == "throws" or (m.body == "empty" and "override" in m.modifiers):
                    what = "only throws NotImplemented/NotSupported" if m.body == "throws" else "is an empty override"
                    findings.append(Finding("L", f"{first.name}.{m.name} {what}", f"{t.file}:{m.line}",
                                            f"a caller holding {', '.join(t.bases) or 'the base'} gets a member that does not work",
                                            key, 2.0 if m.body == "throws" else 1.0))
            for f in [*t.fields, *t.static_setters]:
                if "static" not in f.modifiers or "const" in f.modifiers or "ThreadStatic" in f.attributes:
                    continue
                if "readonly" not in f.modifiers:
                    if "private" in f.modifiers and f.name not in writes and LAZY + f.name in writes:
                        # `X ??= Build()` and nothing else: built on first read,
                        # the same value for the whole process.
                        continue
                    how = "static with a setter" if f.kind == "property" else "static and not readonly"
                    findings.append(Finding("D", f"{first.name}.{f.name} is mutable global state", f"{t.file}:{f.line}",
                                            f"{how} ({f.type}): every user shares it and can change it", key, 2.0))
                elif f.type.startswith(MUTABLE_COLLECTIONS) and (
                        f.name in writes or f.modifiers & {"public", "protected", "internal"}):
                    # A private table filled once by its initializer is a
                    # constant in practice; one a member adds to, or one
                    # anybody can reach, is global state.
                    findings.append(Finding("D", f"{first.name}.{f.name} is a shared mutable collection", f"{t.file}:{f.line}",
                                            f"static readonly {f.type}: the reference is fixed, the contents are global state", key, 1.5))
    # O: one switch subject in several places.
    subjects = defaultdict(list)
    for key, ts in by_key.items():
        for t in ts:
            for s in t.switches:
                if s.cases >= 4:
                    subjects[s.subject.rsplit(".", 1)[-1]].append((key, t.file, s.line))
    for subj, places in subjects.items():
        if len(places) >= 3 and len({p[0] for p in places}) >= 2:
            k, f, line = places[0]
            findings.append(Finding("O", f"`switch ({subj})` in {len(places)} places", f"{f}:{line}",
                                    "a new value means editing " + ", ".join(f"{p[0][1]} ({p[1].rsplit('/', 1)[-1]}:{p[2]})" for p in places[:5]),
                                    k, 2.0))
    # D: a package that many depend on, depending on one that is less stable.
    def instability(p):
        # Test projects are not dependents a design has to stay stable for.
        ca = sum(1 for q, deps in refs.items() if p in deps and not q.endswith(".Tests"))
        ce = len({q for q in refs.get(p, ()) if not q.endswith(".Tests")})
        return ce / (ca + ce) if ca + ce else 0.0
    for proj in sorted(pkg_projects):
        mine = instability(proj)
        for q in sorted(refs.get(proj, ())):
            if instability(q) > mine + 0.25:
                findings.append(Finding("D", f"{proj} depends on the less stable {q}", f"{package_rel}/{proj}.csproj",
                                        f"instability {mine:.2f} here, {instability(q):.2f} there", ("", proj), 2.5))
    # Next to SOLID: deep inheritance, types nothing in the codebase uses.
    up = defaultdict(set)
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "inherits" and d.get("confidence") != "ambiguous":
            up[cb.type_key(u)].add(cb.type_key(v))
    for key in pkg_types:
        depth, level, seen = 0, {key}, {key}
        while level and depth < 12:
            level = {b for k in level for b in up.get(k, ()) if b not in seen}
            seen |= level
            depth += bool(level)
        if depth > depth_cap:
            first = G.nodes[pkg_types[key][0]]
            findings.append(Finding("X", f"{key[1]} sits {depth} levels down a hierarchy",
                                    f"{cb.rel(first['file'])}:{first['start_line']}",
                                    "every level is a contract its subtypes inherit; composition keeps them apart", key, 1.5))
        ids = set(pkg_types[key])
        reached = any(d.get("relation") in INBOUND and src not in ids and owner.get(src) not in ids
                      for x in ids | {f for f, o in owner.items() if o in ids}
                      for src, _, d in G.in_edges(x, data=True))
        # A method the runtime calls by itself (`[ModuleInitializer]`, a native
        # entry point) uses its type, though no code names it.
        reached = reached or any(word in m.attributes for t in by_key.get(key, ()) for m in t.methods
                                 for word in RUNTIME_ENTRIES)
        first = G.nodes[pkg_types[key][0]]
        # The modifiers of every partial declaration, as parsed: a public type
        # is API for code outside this codebase. No facts, no verdict.
        mods = set().union(*(t.modifiers for t in by_key.get(key, ()))) if key in by_key else {"public"}
        if not reached and first.get("subkind") != "interface_declaration" and not mods & {"public", "protected"}:
            findings.append(Finding("X", f"{key[1]} is used by nothing", f"{cb.rel(first['file'])}:{first['start_line']}",
                                    "not public and nothing in the codebase calls, names or extends it", key, 1.0))

    for f in findings:
        f.fence = fence(f.type_key) if f.type_key in decls else {
            "files": [f.where.split(":")[0]], "outside_files": 0, "outside_projects": [], "contract": [],
            "contract_total": 0, "build": dependents, "tests": [], "churn": 0}
        f.score = f.severity * (1 + math.log1p(f.fence["outside_files"])) * (1 + 0.5 * math.log1p(f.fence["churn"]))
    findings.sort(key=lambda f: f.score, reverse=True)
    return findings, package_rel


def render(package: str, findings: list, commit: str = "", per_principle: int = 5) -> str:
    title = f"# Audit of {package}" + (f" at {commit}" if commit else "")
    if not findings:
        return "\n".join([title, "", f"No findings in {package}."])
    out = [title, "",
           f"{len(findings)} finding{'' if len(findings) == 1 else 's'}: " + ", ".join(f"{p} {n}" for p in PRINCIPLES if (n := sum(f.principle == p for f in findings))) +
           ". Ranked by severity, by how many files outside the package reach the type, and by how often its files changed in 180 days.",
           "Refactor inside the fence: keep the contract, build the dependents, run the tests; then run the audit again.", ""]
    for p, title in PRINCIPLES.items():
        fs = [f for f in findings if f.principle == p][:per_principle]
        if not fs:
            continue
        out += [f"## {p}: {title}", ""]
        for f in fs:
            fe = f.fence
            contract = ", ".join(f"{name} ({n})" for name, n in fe["contract"])
            more = fe["contract_total"] - len(fe["contract"])
            out.append(f"- **{f.title}** (`{f.where}`). {f.evidence}.")
            out.append(f"  Fence: {len(fe['files'])} file(s); used from {fe['outside_files']} files in "
                       f"{len(fe['outside_projects'])} other projects"
                       + (f"; contract: {contract}{f' and {more} more' if more > 0 else ''}" if contract else "; no outside use")
                       + f"; build {len(fe['build'])} dependent projects; {len(fe['tests'])} test files"
                       + (f" ({', '.join(t.rsplit('/', 1)[-1] for t in fe['tests'][:3])}{', ...' if len(fe['tests']) > 3 else ''})" if fe["tests"] else "")
                       + f"; {fe['churn']} changes in 180 days.")
        out.append("")
    return "\n".join(out)


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="a codebase folder; its graph is built (or read) in --graph")
    ap.add_argument("package")
    ap.add_argument("--graph", help="graph storage folder (default: ROOT/.lynx_audit_graph)")
    ap.add_argument("--out")
    ap.add_argument("--repo", help="the git repository the files come from, for how often they changed")
    args = ap.parse_args()
    from ..graph import GraphLayer
    root = Path(args.root)
    layer = GraphLayer(storage_dir=Path(args.graph or root / ".lynx_audit_graph"), codebase_path=root,
                       supported_extensions=[".cs"])
    layer.rebuild(force=False)
    findings, rel = audit(layer.graph, root, args.package, repo=Path(args.repo) if args.repo else None)
    commit = (root / ".commit").read_text().strip() if (root / ".commit").exists() else ""
    text = render(rel, findings, commit)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
