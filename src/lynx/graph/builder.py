"""Graph builder: orchestrates extraction, cross-file resolution, persistence.

`GraphLayer` is the entry point used by `CodebaseBackend`. It owns one
on-disk directory (`storage_dir/graph/`) and one in-memory `nx.DiGraph`.

Responsibilities:
  - walk the codebase (same filtering rules as CodebaseRAG)
  - extract per-file graph data via `extract_file`
  - resolve cross-file calls against a global symbol index
  - persist nodes/edges/raw_calls/file_hashes/metadata as JSON
  - support incremental updates from the watcher (update_file / remove_file)

Persistence format choices:
  - JSON, one file per concern. Debuggable, diff-friendly, no pickle.
  - SHA-256 per file in file_hashes.json (mirrors rag_manager's cache so
    the same incremental-rebuild rules apply to both layers).
  - Atomic writes via `path.tmp + rename` to survive crashes mid-write.

Cross-file resolution policy:
  - 1 candidate match → emit edge with confidence="resolved"
  - >1 candidates  → emit edges to all of them, confidence="ambiguous"
  - 0 candidates    → skip silently (external / stdlib / dynamic call)

Member calls (`obj.foo()`) are resolved with the same name lookup but only
when there is a unique match. Multiple member matches are too noisy to be
useful and would dominate the edge count.

Thread-safety: a single re-entrant lock guards every write path. Read-only
queries (executed via `graph` accessor) can run concurrently because
NetworkX's read API is thread-safe as long as the structure isn't mutated.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from .extractor import (extract_file, CSHARP_NAMESPACE_KINDS, GRAPH_LANGUAGES, IMPORT_SCOPED_EXTS,
                        _file_id_stem, _make_id, file_name_keys, import_could_name, import_key,
                        import_via_alias, resolution_context)


def _log(msg: str) -> None:
    """Write logs to stderr so we never corrupt the MCP stdout channel."""
    print(msg, file=sys.stderr)


# Schema version of the on-disk format. Bump (and add a "History:" line)
# whenever the JSON layout OR the extractor rules change in a way that
# makes old graphs incompatible with the live code. The loader detects a
# mismatch and resets state, triggering a full re-bootstrap on next boot.
#
# Bump when:
#   - the node / edge dict shape changes
#   - LangGraphRules adds/removes a field that changes extraction output
#   - the symbol index format changes
#   - cross-file resolution policy changes in a way that produces a
#     different set of resolved edges from the same raw_calls
#
# History:
#   1 — initial release (Lynx 0.5.0)
#   2 — adds `inherits` edges + raw_inherits.json persistence (Lynx 0.6.0)
#   3 — node IDs carry a digest of the file path, so files sharing a
#       directory name and a file name no longer merge (after Lynx 1.9.0)
GRAPH_SCHEMA_VERSION = 3


# What a graph depends on besides the files it was built from: the code that
# extracts and resolves it, relative to the package root.
_EXTRACTOR_SOURCES = ("graph/extractor.py", "graph/builder.py", "graph/cs_recovery.py", "chunking.py")


@functools.lru_cache(maxsize=1)
def extractor_fingerprint() -> str:
    """Changes whenever an unchanged file could now extract to another graph.

    A delta rebuild re-extracts only the files whose bytes changed, so it is
    only as good as a full one while this stays put. Upgrading Lynx or a
    tree-sitter grammar moves it (so does editing an editable install), and
    the next rebuild is then a full one. The schema version above covers the
    file format; this covers the rules, without anyone remembering to bump.
    """
    from importlib import metadata

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for rel in _EXTRACTOR_SOURCES:
        digest.update(rel.encode())
        try:
            digest.update((root / rel).read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            digest.update(b"unreadable")
    grammars = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name.startswith("tree-sitter"):
            grammars.add(f"{name}=={dist.version}")
    digest.update("\n".join(sorted(grammars)).encode())
    return digest.hexdigest()[:16]


def _index_keys(node: "dict | None") -> set:
    """The names a symbol is found by across files: the leaf of its label
    ("Greeter.hello" -> "hello") and any alias the extractor gave it. A
    GDScript script declared `class_name BaseEnemy` in `enemy_base.gd` is
    also `enemy_base`, the name an `extends "res://.../enemy_base.gd"` uses."""
    if not node or node.get("kind") not in ("function", "class"):
        return set()
    keys = {a.lower() for a in node.get("aliases") or () if a}
    leaf = node.get("label", "").split(".")[-1].lower()
    if leaf:
        keys.add(leaf)
    return keys


# SHA-256 read buffer (1 MiB — same value as rag_manager).
_HASH_READ_CHUNK_BYTES = 1 << 20


def _compute_sha256(abs_path: str) -> str:
    """SHA-256 of file bytes. Returns empty string on read error."""
    h = hashlib.sha256()
    try:
        with open(abs_path, "rb") as f:
            while True:
                block = f.read(_HASH_READ_CHUNK_BYTES)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError as e:
        _log(f"[graph] hash failed for {abs_path}: {e}")
        return ""


def _atomic_write_json(path: Path, data) -> None:
    # Compact on purpose: `indent` makes json fall back from its C encoder to
    # the pure-Python one. On a 45k-node graph that is 3.5 s instead of 1.1 s
    # per persist, and the watcher persists after every saved file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# GraphLayer
# ---------------------------------------------------------------------------


class GraphLayer:
    """One graph index per `codebase` source. Lives next to the ChromaDB
    collection in `storage_dir/graph/` and survives process restarts via
    plain JSON files."""

    def __init__(
        self,
        storage_dir: Path,
        codebase_path,
        supported_extensions: Iterable[str],
        ignored_path_fragments: Optional[Iterable[str]] = None,
        ignored_globs: Optional[Iterable[str]] = None,
        follower: bool = False,
    ):
        # A follower reads a graph another Lynx process owns and maintains:
        # it never writes, never bootstraps, and re-reads the JSON when the
        # owner has rewritten it. See lynx/ownership.py.
        self.follower = bool(follower)
        self._disk_stamp = None
        self._last_staleness_check = 0.0

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.codebase_path = Path(codebase_path)
        # Normalize: lowercased with leading dot.
        self.supported_extensions = frozenset(
            (e if e.startswith(".") else f".{e}").lower()
            for e in supported_extensions
        )
        self.ignored_path_fragments = list(ignored_path_fragments or [])
        self.ignored_globs = list(ignored_globs or [])

        # Paths.
        self.nodes_file = self.storage_dir / "nodes.json"
        self.edges_file = self.storage_dir / "edges.json"
        self.raw_calls_file = self.storage_dir / "raw_calls.json"
        self.raw_inherits_file = self.storage_dir / "raw_inherits.json"
        self.hashes_file = self.storage_dir / "file_hashes.json"
        self.metadata_file = self.storage_dir / "metadata.json"

        # In-memory state.
        self._lock = threading.RLock()
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes_by_id: dict = {}
        self._raw_calls_by_file: dict = {}  # abs_path -> list[raw_call dict]
        self._raw_inherits_by_file: dict = {}  # abs_path -> list[raw_inherit dict]
        self._symbol_index: dict = {}       # symbol_name.lower() -> [node_id, ...]
        self._file_hashes: dict = {}        # abs_path -> {sha256, last_indexed_at}
        self._metadata: dict = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "last_update": None,
            "last_full_rebuild": None,
        }

        # A follower that opened mid-write keeps no stamp, so its first
        # staleness check loads the finished graph.
        loaded = self._load_from_disk()
        self._disk_stamp = self._read_disk_stamp() if loaded else None

        # First-boot bootstrap. If the persisted state is empty (either we
        # never ran or the cache was wiped by a schema bump) AND there's
        # source code on disk, build the graph synchronously now. Without
        # this, every graph MCP tool returns [] until the user discovers
        # they have to run `lynx graph build` manually — terrible UX.
        # Mirrors the behavior of `CodebaseRAG._load_or_build_index`.
        # A follower never bootstraps: building is the owner's job, and two
        # processes writing this graph would race on the same JSON files.
        if not self._file_hashes and not self.follower:
            try:
                if self.codebase_path.exists() and any(self._list_candidate_files()):
                    _log(f"[graph] first-boot bootstrap of {self.codebase_path} "
                         f"(use `lynx graph build --force` to rebuild from scratch later)")
                    self.rebuild(force=False)
            except Exception as e:
                # Never let the graph layer's bootstrap break server startup.
                _log(f"[graph] bootstrap failed (graph tools will return empty results): {e}")

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    # Followers re-check the graph on disk at most this often.
    _STALENESS_INTERVAL_SEC = 2.0

    def _read_disk_stamp(self):
        """Cheap fingerprint of the persisted graph: metadata's size + mtime.

        The builder rewrites metadata.json on every persist, so a change here
        means the owner has written the graph since we loaded it.
        """
        try:
            st = self.metadata_file.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def refresh_if_stale(self) -> bool:
        """Reload the graph when the process that owns it has rewritten it.

        No-op for an owner (its in-memory graph is the newest thing there is)
        and throttled for a follower. Reloading the whole graph costs 0.66s on
        a 28k-node / 80k-edge graph, and only happens after a real change.
        """
        if not self.follower:
            return False
        now = time.monotonic()
        if now - self._last_staleness_check < self._STALENESS_INTERVAL_SEC:
            return False
        self._last_staleness_check = now
        stamp = self._read_disk_stamp()
        if stamp is None or stamp == self._disk_stamp:
            return False
        with self._lock:
            if not self._load_from_disk():
                return False              # the owner is mid-write: next check
            self._disk_stamp = stamp
        _log(f"[graph] reloaded {self.storage_dir.parent.name}: the owning "
             f"process rewrote the graph")
        return True

    def _load_from_disk(self) -> bool:
        """Restore in-memory state from JSON files. Missing or invalid
        files reset to empty (and force a rebuild on the next call).

        False when nothing was loaded because a write is in progress or was
        cut short (see `_persist`). A follower keeps what it has and looks
        again later; the owner discards the graph, since files from two
        different writes would disagree, and the bootstrap rebuilds it."""
        # Metadata first — controls schema validation for the rest.
        if self.metadata_file.exists():
            try:
                metadata = json.loads(self.metadata_file.read_text("utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] metadata.json unreadable ({e}); resetting.")
                metadata = {"schema_version": GRAPH_SCHEMA_VERSION,
                            "last_update": None, "last_full_rebuild": None}
            if metadata.get("write_complete") is False:
                if self.follower:
                    return False
                _log("[graph] the last write of this graph did not finish; "
                     "discarding it so it is rebuilt in full.")
                self._reset_state()
                return False
            self._metadata = metadata
        if self._metadata.get("schema_version") != GRAPH_SCHEMA_VERSION:
            _log(f"[graph] schema_version mismatch (got "
                 f"{self._metadata.get('schema_version')!r}, expected "
                 f"{GRAPH_SCHEMA_VERSION}); discarding cache.")
            self._reset_state()
            return True

        # Nodes
        nodes: list = []
        if self.nodes_file.exists():
            try:
                nodes = json.loads(self.nodes_file.read_text("utf-8")) or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] nodes.json unreadable ({e}); resetting.")
                self._reset_state()
                return True
        # Edges
        edges: list = []
        if self.edges_file.exists():
            try:
                edges = json.loads(self.edges_file.read_text("utf-8")) or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] edges.json unreadable ({e}); resetting.")
                self._reset_state()
                return True
        # Raw calls (keyed by file)
        raw_calls: dict = {}
        if self.raw_calls_file.exists():
            try:
                raw_calls = json.loads(self.raw_calls_file.read_text("utf-8")) or {}
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] raw_calls.json unreadable ({e}); resetting raw_calls only.")
                raw_calls = {}
        # Raw inherits (keyed by file)
        raw_inherits: dict = {}
        if self.raw_inherits_file.exists():
            try:
                raw_inherits = json.loads(self.raw_inherits_file.read_text("utf-8")) or {}
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] raw_inherits.json unreadable ({e}); resetting raw_inherits only.")
                raw_inherits = {}
        # File hashes
        hashes: dict = {}
        if self.hashes_file.exists():
            try:
                hashes = json.loads(self.hashes_file.read_text("utf-8")) or {}
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] file_hashes.json unreadable ({e}); SHA cache reset.")
                hashes = {}

        # Rehydrate graph + indices.
        self._graph = nx.DiGraph()
        self._nodes_by_id = {}
        self._symbol_index = {}
        for n in nodes:
            self._add_node_inplace(n)
        for e in edges:
            # Skip edges whose endpoints disappeared from nodes (defensive).
            if e["source"] in self._nodes_by_id and e["target"] in self._nodes_by_id:
                self._add_edge_inplace(e)
            else:
                # External imports keep their target node implicit — we
                # still want them in the graph for "imports" queries.
                self._add_edge_inplace(e, register_external=True)
        self._raw_calls_by_file = raw_calls
        self._raw_inherits_by_file = raw_inherits
        self._file_hashes = hashes
        return True

    def _reset_state(self) -> None:
        self._graph = nx.DiGraph()
        self._nodes_by_id = {}
        self._raw_calls_by_file = {}
        self._raw_inherits_by_file = {}
        self._symbol_index = {}
        self._file_hashes = {}
        self._metadata = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "last_update": None,
            "last_full_rebuild": None,
        }

    def _persist(self) -> None:
        """Write the graph, bracketed by metadata saying the write is under way.

        Each file is replaced atomically, the six together are not: a process
        killed between two of them leaves nodes from one write and edges from
        another, and nothing later would notice, because a delta rebuild trusts
        what is on disk. The marker is what `_load_from_disk` checks."""
        nodes = list(self._nodes_by_id.values())
        edges = [
            {**data, "source": u, "target": v}
            for u, v, data in self._graph.edges(data=True)
        ]
        self._metadata["write_complete"] = False
        _atomic_write_json(self.metadata_file, self._metadata)
        _atomic_write_json(self.nodes_file, nodes)
        _atomic_write_json(self.edges_file, edges)
        _atomic_write_json(self.raw_calls_file, self._raw_calls_by_file)
        _atomic_write_json(self.raw_inherits_file, self._raw_inherits_by_file)
        _atomic_write_json(self.hashes_file, self._file_hashes)
        self._metadata["write_complete"] = True
        _atomic_write_json(self.metadata_file, self._metadata)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _add_node_inplace(self, node: dict) -> None:
        nid = node["id"]
        self._nodes_by_id[nid] = node
        # Add to nx graph with all attributes
        attrs = {k: v for k, v in node.items() if k != "id"}
        self._graph.add_node(nid, **attrs)
        # Index symbol for cross-file lookup. Use the leaf (last
        # qualified segment) so "Greeter.hello" is findable as "hello".
        for key in _index_keys(node):
            self._symbol_index.setdefault(key, [])
            if nid not in self._symbol_index[key]:
                self._symbol_index[key].append(nid)

    def _add_edge_inplace(self, edge: dict, register_external: bool = False) -> None:
        u = edge["source"]
        v = edge["target"]
        attrs = {k: val for k, val in edge.items() if k not in ("source", "target")}
        if register_external and v not in self._nodes_by_id:
            # Synthesize a stub "external" node so the import edge has a
            # target. We don't register it in symbol_index — external nodes
            # are not call targets. The graph gets the same attributes the
            # node will have when this graph is loaded back from disk.
            stub = {
                "id": v, "label": attrs.get("module", v),
                "kind": "external", "subkind": "external",
                "file": "", "start_line": 0, "end_line": 0, "lang_key": "",
            }
            self._nodes_by_id[v] = stub
            self._graph.add_node(v, **{k: val for k, val in stub.items() if k != "id"})
        self._graph.add_edge(u, v, **attrs)

    def _in_index(self, path: str) -> bool:
        """A file of this graph's index: what an import may resolve to."""
        from ..fs_scan import in_scope
        return in_scope(path, self.codebase_path, self.supported_extensions,
                        self.ignored_path_fragments, self.ignored_globs)

    def _importers_of(self, paths, among: "set | None" = None) -> set:
        """Files whose imports may name one of `paths`: an import edge to its
        file node, or an import they try (`import_specs`) that the cheap
        test `import_could_name` does not rule out. A superset: each is
        extracted again and the resolver decides. `among` limits the answer
        to those files (a delta rebuild asks only about the unchanged ones:
        the others are extracted anyway).

        One pass over the file nodes, the paths found by name: a first build
        through `rebuild(force=False)` sees every file as added, and asking
        file by file made it 11 times slower (12 s to 145 s on 1,918 files)."""
        paths = [p for p in paths]
        if not paths or (among is not None and not among):
            return set()
        targets = set(paths)
        by_key: dict = {}
        for p in paths:
            for k in file_name_keys(p):
                by_key.setdefault(k, []).append(p)
        out = set()
        for n in self._nodes_by_id.values():
            if n.get("kind") != "file":
                continue
            importer = n.get("file") or ""
            if not importer or importer in targets or (among is not None and importer not in among):
                continue
            for spec in n.get("import_specs") or ():
                module = spec[1]
                key = import_key(module, importer)
                if key is None:
                    continue
                hits = paths if key in ("", "..") else by_key.get(key, ())
                if any(import_could_name(module, importer, p) for p in hits) or (
                        key not in by_key and import_via_alias(module, importer)):
                    out.add(importer)
                    break
        for p in paths:
            nid = _make_id(_file_id_stem(p))
            if nid not in self._graph:
                continue
            for u, _v, d in self._graph.in_edges(nid, data=True):
                if d.get("relation") in ("imports", "imports_from"):
                    importer = (self._nodes_by_id.get(u) or {}).get("file")
                    if importer and importer not in targets and (among is None or importer in among):
                        out.add(importer)
        return out

    def _reextract(self, abs_path: str) -> set:
        """Extract an unchanged file again, because what its imports name
        changed. Returns the names whose resolution may move, as
        update_file does."""
        affected = self._drop_file(abs_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            _log(f"[graph] read failed for {abs_path}: {e}")
            return affected
        sha = _compute_sha256(abs_path)
        if sha:
            self._file_hashes[abs_path] = {"sha256": sha,
                                           "last_indexed_at": datetime.now().isoformat()}
        res = extract_file(abs_path, content)
        if res is None:
            return affected
        for n in res["nodes"]:
            self._add_node_inplace(n)
            affected |= _index_keys(n)
        for e in res["edges"]:
            self._add_edge_inplace(e, register_external=True)
        if res["raw_calls"]:
            self._raw_calls_by_file[abs_path] = res["raw_calls"]
            affected |= {rc["callee_name"].lower() for rc in res["raw_calls"]}
        if res.get("raw_inherits"):
            self._raw_inherits_by_file[abs_path] = res["raw_inherits"]
            affected |= {ri["base_name"].lower() for ri in res["raw_inherits"]}
        return affected

    def _calls_of_moved_reach(self, reach_before: dict) -> set:
        """Callee names of the files whose reach (the files they import,
        through barrels too) is not what it was: their member and ambiguous
        calls resolve by it. A barrel that changes what it re-exports moves
        the reach of every file importing it, and defines no name itself."""
        after = self._imported_files()
        names = set()
        for f in set(reach_before) | set(after):
            if reach_before.get(f, set()) != after.get(f, set()):
                names |= {rc["callee_name"].lower() for rc in self._raw_calls_by_file.get(f, ())}
        return names

    def _drop_file(self, abs_path: str, gone: bool = False) -> set:
        """Remove every node/edge attached to `abs_path` from the in-memory
        graph and indices. Raw calls from that file are also cleared.
        Caller is responsible for persistence after a batch of drops.

        Returns the set of leaf symbol names (lowercased) whose symbol-index
        entry was touched by the removal — i.e. the names whose cross-file
        resolution may now have changed. Used by the incremental update path to
        re-resolve only the affected names instead of the whole graph.
        """
        touched_names: set = set()
        # Find affected node IDs.
        affected = [nid for nid, n in self._nodes_by_id.items() if n.get("file") == abs_path]
        # External stubs exist only as targets of this file's imports might be.
        externals = {
            succ for nid in affected for succ in self._graph.successors(nid)
            if (self._nodes_by_id.get(succ) or {}).get("kind") == "external"
        }
        # Other files' imports of this file point at its file node now that
        # relative imports resolve. They belong to the importing files and
        # must outlive this one being re-indexed: they are put back on a stub
        # below, which the file node replaces when the file comes back.
        affected_set = set(affected)
        kept_imports = [
            {"source": u, "target": v, **data}
            for v in affected if (self._nodes_by_id.get(v) or {}).get("kind") == "file"
            for u, _v, data in self._graph.in_edges(v, data=True)
            if u not in affected_set and data.get("relation") in ("imports", "imports_from")
        ]
        # Drop edges touching those nodes (DiGraph drops them with the node).
        for nid in affected:
            for key in _index_keys(self._nodes_by_id[nid]):
                touched_names.add(key)
                if key in self._symbol_index:
                    self._symbol_index[key] = [x for x in self._symbol_index[key] if x != nid]
                    if not self._symbol_index[key]:
                        del self._symbol_index[key]
            self._graph.remove_node(nid)
            del self._nodes_by_id[nid]
        # A stub nobody imports any more would not exist in a graph built from
        # scratch; leaving it behind is how an incremental graph drifts.
        for ext in externals:
            if ext in self._graph and self._graph.degree(ext) == 0:
                self._graph.remove_node(ext)
                self._nodes_by_id.pop(ext, None)
        # A deleted file's importers are extracted again by the caller
        # (`_importers_of`), which gives their imports what a full rebuild
        # gives: another file, a stub, or nothing (`from . import x` of a
        # submodule that went).
        if not gone:
            for edge in kept_imports:
                self._add_edge_inplace(edge, register_external=True)
        # Clear raw_calls + raw_inherits bucketed under this file.
        self._raw_calls_by_file.pop(abs_path, None)
        self._raw_inherits_by_file.pop(abs_path, None)
        # Drop the SHA so the next walk re-processes it.
        self._file_hashes.pop(abs_path, None)
        return touched_names

    # ------------------------------------------------------------------
    # File discovery (mirrors CodebaseRAG._list_candidate_files)
    # ------------------------------------------------------------------

    def _list_candidate_files(self) -> list:
        # Shared with the vector index (rag_manager) via fs_scan so the two
        # layers never disagree on the file set.
        from ..fs_scan import list_candidate_files
        return list_candidate_files(
            self.codebase_path,
            self.supported_extensions,
            self.ignored_path_fragments,
            self.ignored_globs,
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve_raw_calls(self, only_names: "set | None" = None) -> int:
        """Walk every raw_call and emit edges where the callee resolves.

        When `only_names` is given, only raw_calls whose `callee_name` (lower)
        is in that set are (re-)resolved — the incremental path uses this so a
        single-file edit doesn't re-walk the entire call set. `only_names=None`
        resolves everything (the full rebuild path).

        Returns the number of edges added. Strategy:
          - 1 candidate                       → confidence="resolved"
          - >1 candidates + non-member call    → all, confidence="ambiguous"
          - >1 candidates +     member call    → SKIP (too noisy without
                                                  type info to disambiguate)
          - 0 candidates                       → skip

        Performance note: this is O(total raw_calls) per call. update_file
        and remove_file invoke it after every file change to keep cross-
        file edges consistent with the symbol index. On very large repos
        (~50k+ files) the watcher should be the bottleneck — if it ever
        becomes a problem, the right fix is incremental resolution
        (only re-resolve raw_calls whose callee_name appears in the
        symbols added/removed by the recent change). Deferred until
        someone reports the latency.
        """
        added = 0
        imported = self._imported_files()
        cs = None
        for file_path, calls in self._raw_calls_by_file.items():
            for rc in calls:
                key = rc["callee_name"].lower()
                if only_names is not None and key not in only_names:
                    continue
                if "recv" in rc:
                    if cs is None:
                        cs = self._csharp_call_context()
                    added += self._resolve_csharp_call(rc, file_path, cs)
                    continue
                candidates = self._symbol_index.get(key) or []
                if not candidates:
                    continue
                reach = imported.get(file_path, set()) | {rc.get("file") or file_path}
                if rc.get("is_member") and os.path.splitext(file_path)[1].lower() in IMPORT_SCOPED_EXTS:
                    # `obj.f()` in an ES module calls a method of a class the
                    # caller can see: defined in its own file or in a file it
                    # imports. A free function, or a method of a class the
                    # caller never imports, is not it: `Object.keys()`,
                    # `map.has()` and `array.push()` used to resolve to the
                    # one project function or method named `keys`, `has` or
                    # `push`. (A namespace import, `math.f()` after `import *
                    # as math`, is lost this way; The Lost Storm has none.)
                    candidates = [c for c in candidates
                                  if "." in ((self._nodes_by_id.get(c) or {}).get("label") or "")
                                  and (self._nodes_by_id.get(c) or {}).get("file") in reach]
                    if not candidates:
                        continue
                elif rc.get("is_member"):
                    if os.path.splitext(file_path)[1].lower() in (".py", ".pyi"):
                        # In Python `x.f()` reaches a module's free function
                        # only as `mod.f()` with `mod` imported: `d.keys()`
                        # is not the project's `def keys()` of a module the
                        # caller never imports. Methods keep the name rule.
                        candidates = [c for c in candidates
                                      if "." in ((self._nodes_by_id.get(c) or {}).get("label") or "")
                                      or (self._nodes_by_id.get(c) or {}).get("file") in reach]
                        if not candidates:
                            continue
                    if len(candidates) > 1:
                        # Other languages keep the name rule (one definition
                        # of the name resolves, several are skipped), except
                        # that one definition among several in a file the
                        # caller imports, or in its own, is the one it calls.
                        seen = [c for c in candidates
                                if (self._nodes_by_id.get(c) or {}).get("file") in reach]
                        if len(seen) == 1:
                            candidates = seen
                if len(candidates) > 1 and not rc.get("is_member"):
                    # Several definitions of the name: the ones in files the
                    # caller imports are the ones it can call. On a 214k-line
                    # TypeScript game this resolved 86% of such calls to one
                    # definition (6,992 ambiguous edges to 3,117). None
                    # imported (a global, an import Lynx cannot resolve):
                    # every candidate, as before.
                    scoped = [c for c in candidates
                              if (self._nodes_by_id.get(c) or {}).get("file") in imported.get(file_path, ())]
                    if scoped:
                        candidates = scoped
                if len(candidates) == 1:
                    target = candidates[0]
                    # A self-match (overload collapse / recursion via name) is
                    # noise as a self-loop edge — skip it. See _walk_calls.
                    if target == rc["caller"]:
                        continue
                    if self._graph.has_edge(rc["caller"], target):
                        # Already have this edge from a different path; keep best confidence.
                        continue
                    self._add_edge_inplace({
                        "source": rc["caller"],
                        "target": target,
                        "relation": "calls",
                        "confidence": "resolved",
                        "from_file": file_path,
                        "from_line": rc.get("line"),
                    })
                    added += 1
                elif not rc.get("is_member"):
                    for target in candidates:
                        if target == rc["caller"]:
                            continue  # don't fuse a self-loop into the graph
                        if self._graph.has_edge(rc["caller"], target):
                            continue
                        self._add_edge_inplace({
                            "source": rc["caller"],
                            "target": target,
                            "relation": "calls",
                            "confidence": "ambiguous",
                            "from_file": file_path,
                            "from_line": rc.get("line"),
                        })
                        added += 1
                # member call with multiple candidates: deliberately skipped.
        return added

    def _csharp_call_context(self) -> dict:
        """What resolving C# calls by type needs, built once per pass: the
        type owning each method, each type's methods and bases, the
        extension methods, and what each file can see."""
        owner, methods, ext = {}, {}, {}
        for u, v, d in self._graph.edges(data=True):
            if d.get("relation") != "contains":
                continue
            un, vn = self._nodes_by_id.get(u) or {}, self._nodes_by_id.get(v) or {}
            if un.get("kind") == "class" and un.get("subkind") not in CSHARP_NAMESPACE_KINDS and vn.get("kind") == "function":
                owner[v] = u
                tk = self._type_key(u)
                leaf = (vn.get("label") or "").rsplit(".", 1)[-1].lower()
                methods.setdefault(tk, {}).setdefault(leaf, []).append(v)
                if vn.get("extends"):
                    ext.setdefault(leaf, []).append(v)
        bases = {}
        for u, v, d in self._graph.edges(data=True):
            if d.get("relation") == "inherits":
                bases.setdefault(self._type_key(u), set()).add(self._type_key(v))
        return {"owner": owner, "methods": methods, "ext": ext, "bases": bases,
                "usings": self._csharp_usings(), "chains": {}}

    def _type_chain(self, start: list, cs: dict) -> list:
        """The types a member lookup walks: the given ones, then their bases,
        nearest first (ten levels at most)."""
        key = tuple(start)
        if key in cs["chains"]:
            return cs["chains"][key]
        out, seen, level = [], set(), list(start)
        for _ in range(10):
            nxt = []
            for tk in level:
                if tk in seen:
                    continue
                seen.add(tk)
                out.append(tk)
                nxt.extend(cs["bases"].get(tk, ()))
            if not nxt:
                break
            level = nxt
        cs["chains"][key] = out
        return out

    def _resolve_csharp_call(self, rc: dict, file_path: str, cs: dict) -> int:
        """A C# call resolved the way the compiler binds it, as far as the
        source shows: `Foo()` and `this.Foo()` in the caller's type and its
        bases, `base.Foo()` in the bases, `x.Foo()` in the declared type of
        `x` (a field, property, parameter or local), `Type.Foo()` in a type
        the file can see; then extension methods on those types. A call on a
        receiver whose type is unknown resolves only when one visible type
        declares the name. Nothing found: a member of a type outside the
        codebase, and no edge."""
        name = rc["callee_name"].lower()
        caller = rc["caller"]
        owner = cs["owner"].get(caller)
        recv = rc.get("recv")
        if recv == "typeref":
            return self._resolve_csharp_type_ref(rc, file_path, cs, owner)
        start = []
        if recv in ("self", "this", "base") and owner is not None:
            mine = self._type_key(owner)
            start = sorted(cs["bases"].get(mine, ())) if recv == "base" else [mine]
        elif recv in ("type", "static", "chain"):
            tname = (rc.get("recv_type") or "").lower()
            types = [c for c in (self._symbol_index.get(tname) or [])
                     if (self._nodes_by_id.get(c) or {}).get("kind") == "class"
                     and (self._nodes_by_id.get(c) or {}).get("subkind") not in CSHARP_NAMESPACE_KINDS]
            if types and owner is not None:
                types = self._csharp_base_candidates({"child": owner, "base_name": rc.get("recv_type"),
                                                      "qualifier": rc.get("recv_qualifier")},
                                                     file_path, types, cs["usings"])
            start = sorted({self._type_key(t) for t in types})
        targets = []
        if start:
            chain = self._type_chain(start, cs)
            for tk in chain:
                found = cs["methods"].get(tk, {}).get(name)
                if found:
                    targets = found
                    break
            if not targets and recv != "base":
                simple = {tk[-1].rsplit(".", 1)[-1].lower() for tk in chain}
                targets = [f for f in cs["ext"].get(name, ())
                           if ((self._nodes_by_id.get(f) or {}).get("extends") or "").lower() in simple]
            if not targets and recv in ("self", "this") and owner is not None:
                # A nested type calling its enclosing type's member.
                label = (self._nodes_by_id.get(owner) or {}).get("label") or ""
                while "." in label and not targets:
                    label = label.rsplit(".", 1)[0]
                    outer = [c for c in (self._symbol_index.get(label.rsplit(".", 1)[-1].lower()) or [])
                             if (self._nodes_by_id.get(c) or {}).get("file") == file_path]
                    for o in outer:
                        targets = cs["methods"].get(self._type_key(o), {}).get(name) or []
                        if targets:
                            break
        if not start and (recv in ("unknown", "chain") or (recv in ("self", "this") and owner is None)):
            # (A `Type.Foo()` on a type the codebase does not declare, such
            # as xUnit's `Assert` or `string`, is that type's: no edge. A
            # chain that names no type is a static property: unknown.)
            ns = (self._nodes_by_id.get(owner) or {}).get("namespace", "") if owner else ""
            parts = ns.split(".") if ns else []
            visible = {".".join(parts[:i]) for i in range(len(parts) + 1)} | cs["usings"].get(file_path, set())
            cands = [f for f in (self._symbol_index.get(name) or [])
                     if f in cs["owner"] and (self._nodes_by_id.get(cs["owner"][f]) or {}).get("namespace", "") in visible]
            if len({self._type_key(cs["owner"][f]) for f in cands}) == 1:
                targets = cands
        added = 0
        for target in targets:
            if target == caller or self._graph.has_edge(caller, target):
                continue
            self._add_edge_inplace({
                "source": caller,
                "target": target,
                "relation": "calls",
                "confidence": "resolved",
                "from_file": file_path,
                "from_line": rc.get("line"),
            })
            added += 1
        return added

    def _resolve_csharp_type_ref(self, rc: dict, file_path: str, cs: dict, owner) -> int:
        """`new T()`, `typeof(T)`, a generic argument, a parameter, return,
        field or property type: a `uses_type` edge to the one type the file
        can see by that name (to its first declaration when it is partial)."""
        caller = rc["caller"]
        scope = owner if owner is not None else (
            caller if (self._nodes_by_id.get(caller) or {}).get("kind") == "class" else None)
        if scope is None:
            if rc.get("ref") != "using":
                return 0
            scope = caller                   # `using static A.B;`, from the file, with A written
        types = [c for c in (self._symbol_index.get(rc["callee_name"].lower()) or [])
                 if (self._nodes_by_id.get(c) or {}).get("kind") == "class"
                 and (self._nodes_by_id.get(c) or {}).get("subkind") not in CSHARP_NAMESPACE_KINDS]
        if not types:
            return 0
        types = self._csharp_base_candidates({"child": scope, "base_name": rc["callee_name"],
                                              "qualifier": rc.get("qualifier")},
                                             file_path, types, cs["usings"])
        if not types or len({self._type_key(t) for t in types}) != 1:
            return 0
        target = min(types, key=lambda t: ((self._nodes_by_id.get(t) or {}).get("file") or "", t))
        if target in (caller, scope) or self._graph.has_edge(caller, target):
            return 0
        self._add_edge_inplace({
            "source": caller,
            "target": target,
            "relation": "uses_type",
            "confidence": "resolved",
            "ref": rc.get("ref"),
            "from_file": file_path,
            "from_line": rc.get("line"),
        })
        return 1

    def _imported_files(self) -> dict:
        """file path -> the project files it imports (resolved imports), and
        through a barrel (`export * from "./y"`) the files it re-exports."""
        out: dict = {}
        reexports: dict = {}
        for u, v, d in self._graph.edges(data=True):
            if d.get("relation") not in ("imports", "imports_from"):
                continue
            target = self._nodes_by_id.get(v) or {}
            source = self._nodes_by_id.get(u) or {}
            if target.get("kind") == "file" and source.get("file"):
                out.setdefault(source["file"], set()).add(target.get("file"))
                if d.get("reexport"):
                    reexports.setdefault(source["file"], set()).add(target.get("file"))
        if reexports:
            for files in out.values():
                stack = [f for f in files if f in reexports]
                while stack:
                    for g in reexports.get(stack.pop(), ()):
                        if g not in files:
                            files.add(g)
                            stack.append(g)
        return out

    def _derived_edge_counts(self) -> dict:
        """Cross-file edges currently in the graph, for a rebuild summary
        that had nothing to resolve."""
        counts = {"calls": 0, "inherits": 0}
        for _u, _v, d in self._graph.edges(data=True):
            if (d.get("relation") in counts
                    and d.get("confidence") in ("resolved", "ambiguous")):
                counts[d["relation"]] += 1
        return counts

    def _clear_resolved_and_ambiguous(self, only_target_names: "set | None" = None) -> None:
        """Remove derived edges (calls, inherits, uses_type) whose confidence
        is 'resolved' or 'ambiguous'. These come from raw_calls / raw_inherits
        and must be re-emitted whenever the symbol index changes.

        When `only_target_names` is given, only edges whose TARGET node's leaf
        name is in that set are cleared. The target leaf name of a derived edge
        is exactly the callee/base name it was resolved from, so this clears
        precisely the edges whose resolution could have changed — letting the
        incremental path re-resolve only the affected names without disturbing
        the rest of the graph (and without leaving stale edges that the
        add-only resolver would never remove)."""
        to_drop = []
        for u, v, d in self._graph.edges(data=True):
            if d.get("relation") not in ("calls", "inherits", "uses_type"):
                continue
            if d.get("confidence") not in ("resolved", "ambiguous"):
                continue
            if only_target_names is not None:
                if not _index_keys(self._nodes_by_id.get(v)) & only_target_names:
                    continue
            to_drop.append((u, v))
        for u, v in to_drop:
            self._graph.remove_edge(u, v)

    def _reresolve_for_names(self, affected_names: set) -> None:
        """Incremental cross-file re-resolution limited to `affected_names`.

        Clears the derived calls/inherits edges that resolved to those names,
        then re-resolves only the raw_calls / raw_inherits referencing them.
        Result is identical to a full clear + full resolve, because a
        single-file change can only alter the symbol-index entries (and thus
        the resolution) of the names defined/removed in that file."""
        if not affected_names:
            return
        self._clear_resolved_and_ambiguous(only_target_names=affected_names)
        # Bases first: a C# call to an inherited member walks them.
        self._resolve_raw_inherits(only_names=affected_names)
        self._resolve_raw_calls(only_names=affected_names)

    def _type_key(self, node_id: str) -> tuple:
        """One type, whatever the number of its declarations: the partial
        declarations of a C# class are one type."""
        n = self._nodes_by_id.get(node_id) or {}
        if "namespace" not in n:
            return (node_id,)
        ns, label = n["namespace"], n.get("label") or ""
        return (ns, label[len(ns) + 1:] if ns and label.startswith(ns + ".") else label)

    def _scoped_base_candidates(self, file_path: str, candidates: list, imported: dict) -> list:
        """Several classes with the base's name: the one in the child's own
        file, else the ones in files the file imports, else all of them."""
        same = [c for c in candidates if (self._nodes_by_id.get(c) or {}).get("file") == file_path]
        if same:
            return same
        reach = imported.get(file_path, set())
        seen = [c for c in candidates if (self._nodes_by_id.get(c) or {}).get("file") in reach]
        if seen:
            return seen
        # A package re-exports what its modules define: `from django.db import
        # models` then `models.Model`, defined in models/base.py and imported
        # by models/__init__.py. One step further through the imports.
        further = set().union(*(imported.get(f, set()) for f in reach)) if reach else set()
        seen = [c for c in candidates if (self._nodes_by_id.get(c) or {}).get("file") in further]
        return seen or candidates

    def _csproj_dir(self, file_path: str) -> str:
        """The folder of the nearest .csproj above a file ('' if none)."""
        cache = self.__dict__.setdefault("_csproj_cache", {})
        d = os.path.dirname(file_path)
        walked = []
        while d and d not in cache:
            walked.append(d)
            try:
                if any(e.name.endswith(".csproj") for e in os.scandir(d)):
                    cache[d] = d
                    break
            except OSError:
                pass
            parent = os.path.dirname(d)
            if parent == d:
                cache[d] = ""
                break
            d = parent
        found = cache.get(d, "")
        for w in walked:
            cache.setdefault(w, found)
        return found

    _MSBUILD_USING = re.compile(r'<Using\s+Include="([^"]+)"([^>]*)>')

    def _msbuild_usings(self, project_dir: str) -> set:
        """The `<Using Include="..."/>` items of a project's csproj and of the
        Directory.Build.props/.targets above it: usings for every file."""
        cache = self.__dict__.setdefault("_msbuild_using_cache", {})
        if project_dir in cache:
            return cache[project_dir]
        found: set = set()
        d, root = project_dir, os.path.normpath(str(self.codebase_path))
        while d:
            try:
                names = [e.path for e in os.scandir(d) if e.name.endswith(".csproj") and d == project_dir
                         or e.name in ("Directory.Build.props", "Directory.Build.targets")]
            except OSError:
                names = []
            for path in names:
                try:
                    text = Path(path).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for include, rest in self._MSBUILD_USING.findall(text):
                    found.add(include)
                    if "Alias=" in rest and "." in include:
                        found.add(include.rsplit(".", 1)[0])
            if os.path.normpath(d) == root or os.path.dirname(d) == d:
                break
            d = os.path.dirname(d)
        cache[project_dir] = found
        return found

    def _csharp_usings(self) -> dict:
        """file -> the namespaces it can name without qualifying: its own
        `using` lines (an alias names its target's namespace), the `global
        using` lines of its project and the project's MSBuild `<Using>` items."""
        own: dict = {}
        project_wide: dict = {}
        for u, _v, d in self._graph.edges(data=True):
            if d.get("relation") != "imports" or not d.get("module"):
                continue
            f = (self._nodes_by_id.get(u) or {}).get("file") or ""
            if not f.lower().endswith(".cs"):
                continue
            names = {d["module"]}
            if d.get("alias") and "." in d["module"]:
                names.add(d["module"].rsplit(".", 1)[0])
            own.setdefault(f, set()).update(names)
            if d.get("global"):
                project_wide.setdefault(self._csproj_dir(f), set()).update(names)
        files = {n.get("file") for n in self._nodes_by_id.values()
                 if n.get("kind") == "file" and (n.get("file") or "").lower().endswith(".cs")}
        for f in files:
            project = self._csproj_dir(f)
            extra = project_wide.get(project, set()) | (self._msbuild_usings(project) if project else set())
            if extra:
                own.setdefault(f, set()).update(extra)
        return own

    def _csharp_base_candidates(self, ri: dict, file_path: str, candidates: list, usings: dict) -> list:
        """C# name lookup for a base type: a type declared in the same file,
        else one in the child's namespace or an enclosing one, in a namespace
        the file imports with `using`, or the one a qualified base names
        (`Util.Base`). None visible: the base comes from outside the
        codebase, and no edge is drawn."""
        child = self._nodes_by_id.get(ri["child"]) or {}
        ns = child.get("namespace", "")
        q = ri.get("qualifier")

        def ns_of(c):
            return (self._nodes_by_id.get(c) or {}).get("namespace", "")

        def outer(c):
            label = (self._nodes_by_id.get(c) or {}).get("label") or ""
            return label.rsplit(".", 1)[0] if "." in label else ""

        if not q and child.get("kind") == "class":
            # The types nested in the referring type come first, then those
            # nested in each enclosing type, in any file of a partial type:
            # `new Writing()` in a third file of `Doc` is Doc's Writing, not
            # the Writing of another type of the namespace.
            scope = child.get("label") or ""
            while scope:
                inner = [c for c in candidates if outer(c) == scope and ns_of(c) == ns]
                if inner:
                    return inner
                scope = scope.rsplit(".", 1)[0] if "." in scope else ""
        same = [c for c in candidates if (self._nodes_by_id.get(c) or {}).get("file") == file_path]
        if same and not q:
            return same
        parts = ns.split(".") if ns else []
        visible = {".".join(parts[:i]) for i in range(len(parts) + 1)} | usings.get(file_path, set())

        if q:
            name = ri["base_name"]
            return [c for c in candidates
                    if ns_of(c) == q or ns_of(c).endswith("." + q)
                    or (self._nodes_by_id.get(c) or {}).get("label", "").endswith(f"{q}.{name}")
                    or any(ns_of(c) == f"{v}.{q}" for v in visible if v)]
        # The namespace and the ones around it, innermost first, win over
        # what a `using` brings in: `[AddComponentMenu]` in a namespace with
        # its own attribute is that one, whatever `using VooDodo.Data;` adds.
        for i in range(len(parts), -1, -1):
            here = [c for c in candidates if ns_of(c) == ".".join(parts[:i])]
            if here:
                return here
        return [c for c in candidates if ns_of(c) in usings.get(file_path, set())]

    def _resolve_raw_inherits(self, only_names: "set | None" = None) -> int:
        """Walk every raw_inherit and emit `inherits` edges using the same
        cross-file resolution policy as raw_calls.

        `only_names` restricts (re-)resolution to raw_inherits whose `base_name`
        (lower) is in the set — used by the incremental path. None = all.

        Returns the number of edges added.
        Policy mirrors _resolve_raw_calls:
          - 1 candidate                       → confidence="resolved"
          - >1 candidates                     → all, confidence="ambiguous"
          - 0 candidates                      → skip silently

        We never apply the "member call" carve-out here because inheritance
        bases are always direct identifiers (no `obj.foo()` ambiguity).
        """
        added = 0
        imported = usings = None
        for file_path, items in self._raw_inherits_by_file.items():
            for ri in items:
                key = ri["base_name"].lower()
                if only_names is not None and key not in only_names:
                    continue
                # A base is a type: not a method or a namespace that shares
                # its name, and not the class itself (`class Field(models.Field)`).
                candidates = [c for c in (self._symbol_index.get(key) or [])
                              if c != ri["child"]
                              and (self._nodes_by_id.get(c) or {}).get("kind") == "class"
                              and (self._nodes_by_id.get(c) or {}).get("subkind") not in CSHARP_NAMESPACE_KINDS]
                if not candidates:
                    continue
                if file_path.lower().endswith(".cs"):
                    # C# sees a type only through its namespaces and usings,
                    # even when the codebase has one type of that name: an
                    # Avalonia `Window` is not the tests' `Window`.
                    if usings is None:
                        usings = self._csharp_usings()
                    candidates = self._csharp_base_candidates(ri, file_path, candidates, usings)
                elif len(candidates) > 1:
                    if imported is None:
                        imported = self._imported_files()
                    candidates = self._scoped_base_candidates(file_path, candidates, imported)
                if not candidates:
                    continue  # a type from outside the codebase (System.Exception)
                confidence = "resolved" if len({self._type_key(c) for c in candidates}) == 1 else "ambiguous"
                for target in candidates:
                    if self._graph.has_edge(ri["child"], target):
                        # One edge per pair: a field of the base's type does
                        # not hide the inheritance.
                        if self._graph.edges[ri["child"], target].get("relation") != "uses_type":
                            continue
                        self._graph.remove_edge(ri["child"], target)
                    edge = {
                        "source": ri["child"],
                        "target": target,
                        "relation": "inherits",
                        "confidence": confidence,
                        "base_kind": ri.get("base_kind", "extends_or_implements"),
                        "from_file": file_path,
                        "from_line": ri.get("line"),
                    }
                    self._add_edge_inplace(edge)
                    added += 1
        return added

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebuild(self, force: bool = False) -> dict:
        """Full or incremental rebuild based on SHA cache + force flag.

        Returns a summary dict with counts of nodes/edges/changed/added/
        removed for observability.

        Without `force` the result is the graph a full rebuild would give,
        at the cost of hashing the files: only changed files are extracted
        again, cross-file resolution is redone over the whole graph, and
        when nothing changed the graph on disk is already that result and is
        left alone. A graph built by another extractor (see
        `extractor_fingerprint`) is rebuilt in full.
        """
        if self.follower:
            from ..errors import StoreNotOwnedError
            raise StoreNotOwnedError(self.storage_dir.parent.name,
                                     str(self.storage_dir.parent))
        with self._lock, resolution_context(self._in_index):
            fingerprint = extractor_fingerprint()
            if (not force and self._file_hashes
                    and self._metadata.get("extractor") != fingerprint):
                _log("[graph] built by a different extractor or grammar "
                     "version; rebuilding it in full")
                force = True
            if force:
                _log("[graph] full rebuild (force=True): wiping state")
                self._reset_state()

            candidates = self._list_candidate_files()
            current_set = set(candidates)
            cached_set = set(self._file_hashes.keys())

            removed = cached_set - current_set
            added = current_set - cached_set

            # Files present in both: SHA compare to decide if changed.
            changed: list = []
            unchanged: list = []
            for p in current_set & cached_set:
                live_sha = _compute_sha256(p)
                if live_sha and live_sha != self._file_hashes[p].get("sha256"):
                    changed.append(p)
                else:
                    unchanged.append(p)

            if not force and not (added or changed or removed):
                # Nothing moved and this extractor built the graph, so the
                # graph on disk is already what a full rebuild would give: the
                # watcher's per-file updates keep it equal to one (pinned by
                # tests/test_graph_incremental.py). Resolving again and
                # rewriting 100+ MB would reproduce the same bytes.
                derived = self._derived_edge_counts()
                return {
                    "candidates": len(current_set),
                    "added": 0,
                    "changed": 0,
                    "removed": 0,
                    "unchanged": len(unchanged),
                    "extracted_files": 0,
                    "nodes_total": self._graph.number_of_nodes(),
                    "edges_total": self._graph.number_of_edges(),
                    "extracted_nodes": 0,
                    "extracted_edges": 0,
                    "resolved_cross_file": derived["calls"],
                    "resolved_inherits": derived["inherits"],
                }

            to_process = sorted(list(added) + list(changed))

            # The files whose imports name a file that goes or comes: found
            # before the drop (it needs the import edges), extracted again
            # below, as a full rebuild would read them.
            importers = set()
            unchanged_set = set(unchanged)
            if not force and unchanged_set:
                importers = self._importers_of(removed, among=unchanged_set)

            # Drop everything we're replacing.
            for p in removed:
                self._drop_file(p, gone=True)
            for p in changed:
                self._drop_file(p)

            # Re-extract changed/added.
            extracted_files = 0
            extracted_nodes = 0
            extracted_edges = 0
            for p in to_process:
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError as e:
                    _log(f"[graph] read failed for {p}: {e}")
                    continue
                res = extract_file(p, content)
                sha = _compute_sha256(p)
                if sha:
                    self._file_hashes[p] = {"sha256": sha,
                                            "last_indexed_at": datetime.now().isoformat()}
                if res is None:
                    # File has no graph rules — still record the SHA so we
                    # don't re-attempt extraction on every run.
                    continue
                extracted_files += 1
                for n in res["nodes"]:
                    self._add_node_inplace(n)
                    extracted_nodes += 1
                for e in res["edges"]:
                    self._add_edge_inplace(e, register_external=True)
                    extracted_edges += 1
                if res["raw_calls"]:
                    self._raw_calls_by_file[p] = res["raw_calls"]
                if res.get("raw_inherits"):
                    self._raw_inherits_by_file[p] = res["raw_inherits"]

            if not force and unchanged_set:
                importers |= self._importers_of(added, among=unchanged_set)
                for p in sorted(importers):
                    self._reextract(p)

            # Cross-file resolution: always re-do from scratch on the
            # current symbol index. The intra-file "extracted" calls are
            # left alone; we only clear resolved/ambiguous.
            self._clear_resolved_and_ambiguous()
            # Bases first: a C# call to an inherited member walks them.
            resolved_inherits = self._resolve_raw_inherits()
            resolved_calls = self._resolve_raw_calls()

            now = datetime.now().isoformat()
            self._metadata["last_update"] = now
            if force or not self._metadata.get("last_full_rebuild"):
                self._metadata["last_full_rebuild"] = now
            self._metadata["extractor"] = fingerprint
            self._persist()

            return {
                "candidates": len(current_set),
                "added": len(added),
                "changed": len(changed),
                "removed": len(removed),
                "unchanged": len(unchanged),
                "extracted_files": extracted_files,
                "nodes_total": self._graph.number_of_nodes(),
                "edges_total": self._graph.number_of_edges(),
                "extracted_nodes": extracted_nodes,
                "extracted_edges": extracted_edges,
                "resolved_cross_file": resolved_calls,
                "resolved_inherits": resolved_inherits,
            }

    def update_file(self, abs_path: str) -> bool:
        """Re-process a single file (called by the watcher on FS events).

        Returns True if the graph changed, False if SHA unchanged.
        A follower never writes: the owning process indexes this file and the
        change arrives through `refresh_if_stale`."""
        if self.follower:
            return False
        with self._lock, resolution_context(self._in_index):
            abs_norm = os.path.normpath(os.path.abspath(abs_path))
            if os.path.splitext(abs_norm)[1].lower() not in self.supported_extensions:
                return False
            if not os.path.isfile(abs_norm):
                return self.remove_file(abs_norm)
            live_sha = _compute_sha256(abs_norm)
            cached_sha = (self._file_hashes.get(abs_norm) or {}).get("sha256")
            if live_sha and live_sha == cached_sha:
                return False  # nothing to do
            is_new = abs_norm not in self._file_hashes
            reach_before = self._imported_files()
            bases_before = {(ri["child"], ri["base_name"]) for ri in self._raw_inherits_by_file.get(abs_norm, ())}
            # Drop old state for this file; capture the names whose symbol-index
            # entry it touched (their cross-file resolution may now change).
            affected_names = self._drop_file(abs_norm)
            try:
                with open(abs_norm, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as e:
                _log(f"[graph] read failed for {abs_norm}: {e}")
                return False
            res = extract_file(abs_norm, content)
            if live_sha:
                self._file_hashes[abs_norm] = {"sha256": live_sha,
                                               "last_indexed_at": datetime.now().isoformat()}
            if res is not None:
                for n in res["nodes"]:
                    self._add_node_inplace(n)
                    # New/changed symbols defined here are also affected names.
                    affected_names |= _index_keys(n)
                for e in res["edges"]:
                    self._add_edge_inplace(e, register_external=True)
                if res["raw_calls"]:
                    self._raw_calls_by_file[abs_norm] = res["raw_calls"]
                    # The file's OWN outgoing calls reference names defined
                    # elsewhere; those must be (re-)resolved too, even though
                    # their definitions didn't change in this edit.
                    for rc in res["raw_calls"]:
                        affected_names.add(rc["callee_name"].lower())
                if res.get("raw_inherits"):
                    self._raw_inherits_by_file[abs_norm] = res["raw_inherits"]
                    for ri in res["raw_inherits"]:
                        affected_names.add(ri["base_name"].lower())
            if is_new:
                # Imports waiting for this file, by any name that resolves to
                # it (relative, alias, `from . import x`), are redone.
                for importer in sorted(self._importers_of([abs_norm])):
                    affected_names |= self._reextract(importer)
            affected_names |= self._calls_of_moved_reach(reach_before)
            bases_after = {(ri["child"], ri["base_name"]) for ri in self._raw_inherits_by_file.get(abs_norm, ())}
            if abs_norm.lower().endswith(".cs") and bases_after != bases_before:
                # A C# type gained or lost a base: the members it inherits
                # change for every caller, in any file, under any name.
                # Rare (a base list edit), so everything is resolved again.
                self._clear_resolved_and_ambiguous()
                self._resolve_raw_inherits()
                self._resolve_raw_calls()
            else:
                # Re-resolve ONLY the affected names cross-file (the symbol
                # index only moved for symbols defined/removed in this file).
                self._reresolve_for_names(affected_names)
            self._metadata["last_update"] = datetime.now().isoformat()
            self._persist()
            return True

    def remove_file(self, abs_path: str) -> bool:
        if self.follower:
            return False
        with self._lock, resolution_context(self._in_index):
            abs_norm = os.path.normpath(os.path.abspath(abs_path))
            if abs_norm not in self._file_hashes and not any(
                n.get("file") == abs_norm for n in self._nodes_by_id.values()
            ):
                return False
            importers = self._importers_of([abs_norm])
            reach_before = self._imported_files()
            had_bases = abs_norm.lower().endswith(".cs") and bool(self._raw_inherits_by_file.get(abs_norm))
            affected_names = self._drop_file(abs_norm, gone=True)
            for importer in sorted(importers):
                affected_names |= self._reextract(importer)
            affected_names |= self._calls_of_moved_reach(reach_before)
            if had_bases:
                # C# types with bases are gone: calls to what they inherited,
                # from any file, are resolved again (as in update_file).
                self._clear_resolved_and_ambiguous()
                self._resolve_raw_inherits()
                self._resolve_raw_calls()
            else:
                # Re-resolve only the names this file defined (other files'
                # raw_calls to them may now resolve differently, or not at all).
                self._reresolve_for_names(affected_names)
            self._metadata["last_update"] = datetime.now().isoformat()
            self._persist()
            return True

    def status(self) -> dict:
        """Snapshot for `lynx graph status` / MCP tool."""
        self.refresh_if_stale()
        with self._lock:
            lang_counts: dict = {}
            for n in self._nodes_by_id.values():
                k = n.get("lang_key") or "external"
                lang_counts[k] = lang_counts.get(k, 0) + 1
            kind_counts: dict = {}
            for n in self._nodes_by_id.values():
                k = n.get("kind") or "?"
                kind_counts[k] = kind_counts.get(k, 0) + 1
            relation_counts: dict = {}
            for _u, _v, d in self._graph.edges(data=True):
                r = d.get("relation") or "?"
                relation_counts[r] = relation_counts.get(r, 0) + 1
            return {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "files_indexed": len(self._file_hashes),
                "raw_calls_pending": sum(len(v) for v in self._raw_calls_by_file.values()),
                "raw_inherits_pending": sum(len(v) for v in self._raw_inherits_by_file.values()),
                "by_language": lang_counts,
                "by_kind": kind_counts,
                "by_relation": relation_counts,
                "last_update": self._metadata.get("last_update"),
                "last_full_rebuild": self._metadata.get("last_full_rebuild"),
            }

    @property
    def graph(self) -> nx.DiGraph:
        """The in-memory `nx.DiGraph`. Read-only queries are safe to run
        concurrently; mutate ONLY through update/remove/rebuild methods.

        Every graph query goes through here, which is why the follower's
        staleness check lives here too.
        """
        self.refresh_if_stale()
        return self._graph

    @property
    def nodes_by_id(self) -> dict:
        return self._nodes_by_id
