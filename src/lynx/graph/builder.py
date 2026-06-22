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

import hashlib
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from .extractor import extract_file, GRAPH_LANGUAGES


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
GRAPH_SCHEMA_VERSION = 2

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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.codebase_path = Path(codebase_path)
        # Normalize: lowercased with leading dot.
        self.supported_extensions = frozenset(
            (e if e.startswith(".") else f".{e}").lower()
            for e in supported_extensions
        )
        self.ignored_path_fragments = list(ignored_path_fragments or [])

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

        self._load_from_disk()

        # First-boot bootstrap. If the persisted state is empty (either we
        # never ran or the cache was wiped by a schema bump) AND there's
        # source code on disk, build the graph synchronously now. Without
        # this, every graph MCP tool returns [] until the user discovers
        # they have to run `lynx graph build` manually — terrible UX.
        # Mirrors the behavior of `CodebaseRAG._load_or_build_index`.
        if not self._file_hashes:
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

    def _load_from_disk(self) -> None:
        """Restore in-memory state from JSON files. Missing or invalid
        files reset to empty (and force a rebuild on the next call)."""
        # Metadata first — controls schema validation for the rest.
        if self.metadata_file.exists():
            try:
                self._metadata = json.loads(self.metadata_file.read_text("utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] metadata.json unreadable ({e}); resetting.")
                self._metadata = {"schema_version": GRAPH_SCHEMA_VERSION,
                                  "last_update": None, "last_full_rebuild": None}
        if self._metadata.get("schema_version") != GRAPH_SCHEMA_VERSION:
            _log(f"[graph] schema_version mismatch (got "
                 f"{self._metadata.get('schema_version')!r}, expected "
                 f"{GRAPH_SCHEMA_VERSION}); discarding cache.")
            self._reset_state()
            return

        # Nodes
        nodes: list = []
        if self.nodes_file.exists():
            try:
                nodes = json.loads(self.nodes_file.read_text("utf-8")) or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] nodes.json unreadable ({e}); resetting.")
                self._reset_state()
                return
        # Edges
        edges: list = []
        if self.edges_file.exists():
            try:
                edges = json.loads(self.edges_file.read_text("utf-8")) or []
            except (OSError, json.JSONDecodeError) as e:
                _log(f"[graph] edges.json unreadable ({e}); resetting.")
                self._reset_state()
                return
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
        nodes = list(self._nodes_by_id.values())
        edges = [
            {**data, "source": u, "target": v}
            for u, v, data in self._graph.edges(data=True)
        ]
        _atomic_write_json(self.nodes_file, nodes)
        _atomic_write_json(self.edges_file, edges)
        _atomic_write_json(self.raw_calls_file, self._raw_calls_by_file)
        _atomic_write_json(self.raw_inherits_file, self._raw_inherits_by_file)
        _atomic_write_json(self.hashes_file, self._file_hashes)
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
        if node["kind"] in ("function", "class"):
            label = node.get("label", "")
            leaf = label.split(".")[-1].lower()
            if leaf:
                self._symbol_index.setdefault(leaf, [])
                if nid not in self._symbol_index[leaf]:
                    self._symbol_index[leaf].append(nid)

    def _add_edge_inplace(self, edge: dict, register_external: bool = False) -> None:
        u = edge["source"]
        v = edge["target"]
        attrs = {k: val for k, val in edge.items() if k not in ("source", "target")}
        if register_external and v not in self._nodes_by_id:
            # Synthesize a stub "external" node so the import edge has a
            # target. We don't register it in symbol_index — external nodes
            # are not call targets.
            self._graph.add_node(v, kind="external", label=attrs.get("module", v))
            self._nodes_by_id[v] = {
                "id": v, "label": attrs.get("module", v),
                "kind": "external", "subkind": "external",
                "file": "", "start_line": 0, "end_line": 0, "lang_key": "",
            }
        self._graph.add_edge(u, v, **attrs)

    def _drop_file(self, abs_path: str) -> set:
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
        # Drop edges touching those nodes (DiGraph drops them with the node).
        for nid in affected:
            label = self._nodes_by_id[nid].get("label", "")
            leaf = label.split(".")[-1].lower()
            if leaf:
                touched_names.add(leaf)
            if leaf in self._symbol_index:
                self._symbol_index[leaf] = [x for x in self._symbol_index[leaf] if x != nid]
                if not self._symbol_index[leaf]:
                    del self._symbol_index[leaf]
            self._graph.remove_node(nid)
            del self._nodes_by_id[nid]
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
        for file_path, calls in self._raw_calls_by_file.items():
            for rc in calls:
                key = rc["callee_name"].lower()
                if only_names is not None and key not in only_names:
                    continue
                candidates = self._symbol_index.get(key) or []
                if not candidates:
                    continue
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

    def _clear_resolved_and_ambiguous(self, only_target_names: "set | None" = None) -> None:
        """Remove derived edges (calls + inherits) whose confidence is
        'resolved' or 'ambiguous'. These come from raw_calls / raw_inherits
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
            if d.get("relation") not in ("calls", "inherits"):
                continue
            if d.get("confidence") not in ("resolved", "ambiguous"):
                continue
            if only_target_names is not None:
                tgt = self._nodes_by_id.get(v)
                leaf = (tgt.get("label", "").split(".")[-1].lower()) if tgt else ""
                if leaf not in only_target_names:
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
        self._resolve_raw_calls(only_names=affected_names)
        self._resolve_raw_inherits(only_names=affected_names)

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
        for file_path, items in self._raw_inherits_by_file.items():
            for ri in items:
                key = ri["base_name"].lower()
                if only_names is not None and key not in only_names:
                    continue
                candidates = self._symbol_index.get(key) or []
                if not candidates:
                    continue
                confidence = "resolved" if len(candidates) == 1 else "ambiguous"
                for target in candidates:
                    if self._graph.has_edge(ri["child"], target):
                        continue
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
        """
        with self._lock:
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

            to_process = sorted(list(added) + list(changed))

            # Drop everything we're replacing.
            for p in list(removed) + changed:
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

            # Cross-file resolution: always re-do from scratch on the
            # current symbol index. The intra-file "extracted" calls are
            # left alone; we only clear resolved/ambiguous.
            self._clear_resolved_and_ambiguous()
            resolved_calls = self._resolve_raw_calls()
            resolved_inherits = self._resolve_raw_inherits()

            now = datetime.now().isoformat()
            self._metadata["last_update"] = now
            if force or not self._metadata.get("last_full_rebuild"):
                self._metadata["last_full_rebuild"] = now
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

        Returns True if the graph changed, False if SHA unchanged."""
        with self._lock:
            abs_norm = os.path.normpath(os.path.abspath(abs_path))
            if os.path.splitext(abs_norm)[1].lower() not in self.supported_extensions:
                return False
            if not os.path.isfile(abs_norm):
                return self.remove_file(abs_norm)
            live_sha = _compute_sha256(abs_norm)
            cached_sha = (self._file_hashes.get(abs_norm) or {}).get("sha256")
            if live_sha and live_sha == cached_sha:
                return False  # nothing to do
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
                    if n.get("kind") in ("function", "class"):
                        leaf = (n.get("label", "").split(".")[-1].lower())
                        if leaf:
                            affected_names.add(leaf)
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
            # Re-resolve ONLY the affected names cross-file (the symbol index
            # only moved for symbols defined/removed in this one file).
            self._reresolve_for_names(affected_names)
            self._metadata["last_update"] = datetime.now().isoformat()
            self._persist()
            return True

    def remove_file(self, abs_path: str) -> bool:
        with self._lock:
            abs_norm = os.path.normpath(os.path.abspath(abs_path))
            if abs_norm not in self._file_hashes and not any(
                n.get("file") == abs_norm for n in self._nodes_by_id.values()
            ):
                return False
            affected_names = self._drop_file(abs_norm)
            # Re-resolve only the names this file defined (other files' raw_calls
            # to them may now resolve differently — or no longer resolve).
            self._reresolve_for_names(affected_names)
            self._metadata["last_update"] = datetime.now().isoformat()
            self._persist()
            return True

    def status(self) -> dict:
        """Snapshot for `lynx graph status` / MCP tool."""
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
        concurrently; mutate ONLY through update/remove/rebuild methods."""
        return self._graph

    @property
    def nodes_by_id(self) -> dict:
        return self._nodes_by_id
