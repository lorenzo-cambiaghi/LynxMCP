"""Single-file graph extraction over a tree-sitter AST.

For one file, produce:
  - nodes: classes and functions/methods defined in the file
  - edges: containment (class → method), imports (file → module/symbol),
           and *intra-file* calls (caller_id → callee_id)
  - raw_calls: unresolved callees (the callee name was not found in the
           current file's symbol table). The builder resolves these later
           against the global symbol index built across all files.

Design notes
------------
- We REUSE the parser cache from `lynx.chunking` via `parse_file()`. That
  way every file is parsed exactly once even when both the chunker and the
  graph layer process the same source.
- Rules are declarative (`LangGraphRules`) so adding a language is a matter
  of listing the AST node types we care about; the walker logic is shared.
- Cross-file resolution is intentionally NOT here. Keeping it out means
  this module is pure (no I/O, no global state) and trivially testable.
- We do NOT extract every node-type some other tool might want (concept
  nodes, hyper-edges, semantic_similar_to). The graph is built to answer
  three concrete questions: "who calls X", "what does X call", "what does
  this file import". Anything else is out of scope.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..chunking import parse_file


# ---------------------------------------------------------------------------
# ID generation (stable, deterministic, cross-file)
# ---------------------------------------------------------------------------
#
# Mirrors the pattern from graphify/extract.py:_make_id — we keep it
# byte-for-byte compatible so anyone who wants to diff our graph against a
# graphify graph can do so without an ID normalization step.

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_REPEAT_UNDERSCORE = re.compile(r"_+")


def _make_id(*parts: str) -> str:
    """Build a stable node ID from one or more name parts.

    NFKC-normalize so composed and decomposed Unicode collapse to the same
    ID. Lowercase so case-only renames don't bifurcate the node. Strip
    everything that isn't `\\w`. Empty / whitespace-only parts are dropped
    before joining.
    """
    combined = "_".join(p.strip("_.") for p in parts if p)
    combined = unicodedata.normalize("NFKC", combined)
    cleaned = _NON_WORD.sub("_", combined)
    cleaned = _REPEAT_UNDERSCORE.sub("_", cleaned)
    return cleaned.strip("_").casefold()


def _file_stem(path: str) -> str:
    """Return `parent.name + "." + stem` so files with the same name in
    different directories don't collide on ID generation."""
    p = Path(path)
    parent = p.parent.name
    if parent and parent not in (".", ""):
        return f"{parent}.{p.stem}"
    return p.stem


# ---------------------------------------------------------------------------
# Language rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LangGraphRules:
    """Declarative rules for one language. The walker is shared.

    Each field is a *set of AST node types*; matching is done by exact
    string comparison against `node.type`.
    """

    # Container nodes (class, struct, namespace, module, ...) — we recurse
    # INTO them and prefix nested chunk names with the container's name.
    container_types: frozenset

    # Top-level function / method / constructor declarations — these become
    # graph nodes (kind="function").
    function_types: frozenset

    # File-level import statements (handled via `import_handler`).
    import_types: frozenset

    # Call expressions — the walker scans function bodies looking for these.
    call_types: frozenset

    # Field name to read the callee subtree from a call node. Default
    # matches Python / JS / Go / Rust / Swift / Kotlin / Java.
    call_function_field: str = "function"

    # Member-style accessors (`obj.foo`, `obj->foo`, `obj::foo`). When the
    # callee subtree's type is in this set, treat it as a method call and
    # read the method name from `call_accessor_field`.
    call_accessor_types: frozenset = frozenset()
    call_accessor_field: str = "name"

    # Some grammars (PHP `member_call_expression`, `scoped_call_expression`)
    # don't wrap the callee in an accessor node — instead the method name is
    # a direct field on the call node itself. List those call types here.
    direct_member_call_types: frozenset = frozenset()
    direct_member_call_field: str = "name"

    # When walking a function body, do NOT recurse into nested chunks of
    # these types — nested function definitions are independent callers.
    function_boundary_types: frozenset = frozenset()

    # Custom import extractor. Signature:
    #   handler(node, source: bytes, file_id: str, edges: list, file_path: str)
    # Each handler appends edges to the list. The default handler skips
    # nodes it does not recognize so unsupported languages still build a
    # partial graph (functions + intra-file calls).
    import_handler: Optional[Callable] = None

    # Custom inheritance extractor. Signature:
    #   handler(container_node, source: bytes, container_id: str,
    #           file_path: str, raw_inherits: list)
    # Each handler appends entries to raw_inherits with shape:
    #   {"child": container_id, "base_name": "<name>",
    #    "base_kind": "extends"|"implements"|"extends_or_implements",
    #    "file": file_path, "line": int}
    # The builder later resolves base_name against the global symbol index
    # and emits `inherits` edges with the same confidence policy as calls.
    # Languages without a class-extension model (C, Go) leave this None.
    inheritance_handler: Optional[Callable] = None

    # Custom container-scan hook for languages where containers can show up
    # under unusual wrappers (e.g. C# file-scoped namespace). Optional.
    extra_container_hook: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Helpers (shared across handlers)
# ---------------------------------------------------------------------------


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name_of(node, source: bytes) -> Optional[str]:
    """Best-effort symbol name extraction.

    1. Prefer `node.child_by_field_name("name")` — this is what every
       tree-sitter grammar uses when there's exactly one canonical name
       field on the declaration. Critical for C# method_declaration which
       has two `identifier` children (return type + name).
    2. Fall back to the first non-keyword child for grammars that don't
       expose a `name` field (Ruby `class` has `constant`, etc.).
    """
    nm = node.child_by_field_name("name")
    if nm is not None:
        return _text(nm, source)
    # Generic fallback: first child that looks like an identifier.
    for c in node.children:
        if c.type in ("identifier", "constant", "type_identifier",
                      "simple_identifier", "name", "qualified_name",
                      "property_identifier", "field_identifier"):
            return _text(c, source)
    return None


def _first_keyword_child(node) -> Optional[str]:
    """Return the first leaf-keyword type of a node (e.g. `class`, `struct`,
    `interface`, `extension`). Used to disambiguate Kotlin/Swift grammars
    where a single node type covers several declaration keywords."""
    for c in node.children:
        if not c.is_named:
            return c.type
    return None


# ---------------------------------------------------------------------------
# Import handlers (one per language family)
# ---------------------------------------------------------------------------
#
# Each handler is intentionally minimal — we extract enough to build a
# qualitative import graph ("file X depends on module Y") without trying
# to resolve every path alias / extension rewrite the way graphify does.


def _handle_python_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    # import_statement: `import a.b.c` or `import a as alias`
    # import_from_statement: `from a.b import c, d`
    if node.type == "import_statement":
        for c in node.children:
            if c.type in ("dotted_name", "aliased_import"):
                # For aliased_import, the imported module is the first child.
                target = c
                if c.type == "aliased_import":
                    target = c.child_by_field_name("name") or c.children[0]
                mod = _text(target, source).strip()
                if mod:
                    edges.append({
                        "source": file_id,
                        "target": _make_id(mod.split(".")[0]),
                        "relation": "imports",
                        "module": mod,
                    })
    elif node.type == "import_from_statement":
        mod_node = node.child_by_field_name("module_name")
        if mod_node is None:
            return
        mod = _text(mod_node, source).strip().lstrip(".")
        if not mod:
            return
        edges.append({
            "source": file_id,
            "target": _make_id(mod.split(".")[0]),
            "relation": "imports_from",
            "module": mod,
        })


def _handle_js_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    # import_statement covers ES module imports. The module specifier is the
    # single `string` child of the import.
    if node.type != "import_statement":
        return
    for c in node.children:
        if c.type == "string":
            raw = _text(c, source).strip().strip('"').strip("'")
            if not raw:
                continue
            # Naive resolution: relative imports keep the full path as the
            # module label; bare specifiers use the package basename.
            if raw.startswith("."):
                target = _make_id(file_path, raw)
            else:
                target = _make_id(raw.split("/")[0])
            edges.append({
                "source": file_id,
                "target": target,
                "relation": "imports_from",
                "module": raw,
            })
            return


def _handle_csharp_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "using_directive":
        return
    for c in node.children:
        if c.type in ("qualified_name", "identifier"):
            mod = _text(c, source).strip()
            if mod:
                edges.append({
                    "source": file_id,
                    "target": _make_id(mod),
                    "relation": "imports",
                    "module": mod,
                })
                return


def _handle_java_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "import_declaration":
        return
    # Walk children until we find a (possibly nested) scoped_identifier or identifier.
    def collect(n) -> str:
        if n.type in ("identifier",):
            return _text(n, source)
        if n.type == "scoped_identifier":
            scope = n.child_by_field_name("scope")
            name = n.child_by_field_name("name")
            left = collect(scope) if scope else ""
            right = collect(name) if name else ""
            return f"{left}.{right}" if left and right else (left or right)
        # Fallback
        return _text(n, source).strip(";").strip()
    for c in node.children:
        if c.type in ("scoped_identifier", "identifier"):
            mod = collect(c).strip(".")
            if mod:
                edges.append({
                    "source": file_id,
                    "target": _make_id(mod),
                    "relation": "imports",
                    "module": mod,
                })
                return


def _handle_go_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    # import_declaration wraps import_spec or import_spec_list.
    if node.type != "import_declaration":
        return
    def emit_spec(spec_node):
        path_node = spec_node.child_by_field_name("path")
        if path_node is None:
            for c in spec_node.children:
                if c.type == "interpreted_string_literal":
                    path_node = c
                    break
        if path_node is None:
            return
        raw = _text(path_node, source).strip().strip('"')
        if raw:
            edges.append({
                "source": file_id,
                "target": _make_id(raw),
                "relation": "imports",
                "module": raw,
            })
    for c in node.children:
        if c.type == "import_spec":
            emit_spec(c)
        elif c.type == "import_spec_list":
            for sub in c.children:
                if sub.type == "import_spec":
                    emit_spec(sub)


def _handle_rust_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "use_declaration":
        return
    arg = node.child_by_field_name("argument")
    if arg is None:
        for c in node.children:
            if c.type in ("scoped_use_list", "scoped_identifier", "identifier", "use_list"):
                arg = c
                break
    if arg is None:
        return
    raw = _text(arg, source).split("::")[0].strip()
    if raw:
        edges.append({
            "source": file_id,
            "target": _make_id(raw),
            "relation": "imports",
            "module": _text(arg, source).strip(";").strip(),
        })


def _handle_c_include(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "preproc_include":
        return
    path_node = node.child_by_field_name("path")
    if path_node is None:
        for c in node.children:
            if c.type in ("string_literal", "system_lib_string"):
                path_node = c
                break
    if path_node is None:
        return
    raw = _text(path_node, source).strip().strip('"').strip("<>")
    if raw:
        edges.append({
            "source": file_id,
            "target": _make_id(raw),
            "relation": "imports",
            "module": raw,
        })


def _handle_ruby_require(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    # Ruby has no import node. `require "foo"` is a regular call. We only
    # treat top-level calls whose method name matches a known loader.
    if node.type != "call":
        return
    method = node.child_by_field_name("method")
    if method is None:
        for c in node.children:
            if c.type == "identifier":
                method = c
                break
    if method is None or _text(method, source) not in ("require", "require_relative", "load", "autoload"):
        return
    args = node.child_by_field_name("arguments")
    if args is None:
        for c in node.children:
            if c.type == "argument_list":
                args = c
                break
    if args is None:
        return
    for c in args.children:
        if c.type == "string":
            for cc in c.children:
                if cc.type == "string_content":
                    raw = _text(cc, source).strip()
                    if raw:
                        edges.append({
                            "source": file_id,
                            "target": _make_id(raw),
                            "relation": "imports",
                            "module": raw,
                        })
            return


def _handle_php_use(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "namespace_use_declaration":
        return
    for c in node.children:
        if c.type == "namespace_use_clause":
            # The clause wraps a qualified_name (or name) — take its text.
            for cc in c.children:
                if cc.type in ("qualified_name", "name"):
                    mod = _text(cc, source).strip()
                    if mod:
                        edges.append({
                            "source": file_id,
                            "target": _make_id(mod),
                            "relation": "imports",
                            "module": mod,
                        })


def _handle_kotlin_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "import":
        return
    for c in node.children:
        if c.type in ("qualified_identifier", "identifier"):
            mod = _text(c, source).strip()
            if mod:
                edges.append({
                    "source": file_id,
                    "target": _make_id(mod),
                    "relation": "imports",
                    "module": mod,
                })
                return


def _handle_swift_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    if node.type != "import_declaration":
        return
    for c in node.children:
        if c.type in ("identifier", "simple_identifier"):
            mod = _text(c, source).strip()
            if mod:
                edges.append({
                    "source": file_id,
                    "target": _make_id(mod),
                    "relation": "imports",
                    "module": mod,
                })
                return


# ---------------------------------------------------------------------------
# Inheritance handlers (one per language family)
# ---------------------------------------------------------------------------
#
# Each handler is called by the walker once for every container node (class /
# interface / struct / trait / ...). It appends entries to `raw_inherits` for
# every base type listed on the declaration; the builder resolves the names
# cross-file with the same policy as calls (1 match → resolved, N → ambiguous,
# 0 → drop). The single relation emitted on resolved edges is "inherits"; the
# distinction between extends-a-class vs implements-an-interface (when the
# language exposes it) is preserved in the edge attribute `base_kind`.
#
# Conservative behavior across grammars: we extract the *innermost identifier*
# of each base expression (so generics like `List<T>` resolve as `List`, and
# qualified names like `System.Collections.IList` resolve as `IList`). Better
# than nothing; the global symbol index handles the rest.


def _innermost_identifier(node, source: bytes) -> Optional[str]:
    """Walk a base-type expression down to its innermost identifier.

    Handles `List<T>` (returns `List`), `System.Foo.Bar` (returns `Bar`),
    `Container[int]` (returns `Container`), `a::b::C` (returns `C`).
    Returns None when no usable name can be extracted.
    """
    # Direct leaf identifier types.
    if node.type in ("identifier", "type_identifier", "simple_identifier",
                     "constant", "name", "property_identifier"):
        text = _text(node, source).strip()
        return text or None

    # Try a `name` field first — common across grammars.
    name_field = node.child_by_field_name("name")
    if name_field is not None and name_field is not node:
        out = _innermost_identifier(name_field, source)
        if out:
            return out

    # Qualified/scoped/attribute access: the rightmost child is the leaf name.
    if node.type in (
        "qualified_name", "scoped_identifier", "scoped_type_identifier",
        "attribute", "member_expression", "navigation_expression",
        "qualified_identifier",
    ):
        named = [c for c in node.children if c.is_named]
        if named:
            out = _innermost_identifier(named[-1], source)
            if out:
                return out

    # Generic / subscript wrappers: prefer the first identifier-ish child.
    if node.type in ("generic_name", "generic_type", "subscript",
                     "parameterized_type"):
        named = [c for c in node.children if c.is_named]
        if named:
            out = _innermost_identifier(named[0], source)
            if out:
                return out

    # Last-resort: walk children once looking for any direct identifier.
    for c in node.children:
        if c.is_named and c.type in (
            "identifier", "type_identifier", "simple_identifier",
            "constant", "name", "property_identifier",
        ):
            text = _text(c, source).strip()
            if text:
                return text
    return None


def _handle_csharp_inheritance(container_node, source: bytes,
                               container_id: str, file_path: str,
                               raw_inherits: list) -> None:
    """C#: `class Foo : Bar, IBaz, Generic<T> { ... }`. The base list is a
    `base_list` child of the class/struct/interface declaration. The grammar
    does not distinguish a class extension from interface implementation
    syntactically, so we emit every entry with `base_kind="extends_or_implements"`.
    """
    if container_node.type not in (
        "class_declaration", "struct_declaration", "interface_declaration",
        "record_declaration",
    ):
        return
    base_list = None
    for c in container_node.children:
        if c.type == "base_list":
            base_list = c
            break
    if base_list is None:
        return
    for c in base_list.children:
        if not c.is_named:
            continue
        name = _innermost_identifier(c, source)
        if not name:
            continue
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": "extends_or_implements",
            "file": file_path,
            "line": c.start_point[0] + 1,
        })


def _handle_python_inheritance(container_node, source: bytes,
                               container_id: str, file_path: str,
                               raw_inherits: list) -> None:
    """Python: `class Foo(Base, Mixin, generic.Container[int]):`. The bases
    are inside an `argument_list` named child. Python has no formal
    distinction between extends and implements (duck typing); everything is
    `base_kind="extends"`.
    """
    if container_node.type != "class_definition":
        return
    args = None
    for c in container_node.children:
        if c.type == "argument_list":
            args = c
            break
    if args is None:
        return
    for c in args.children:
        if not c.is_named:
            continue
        if c.type == "keyword_argument":
            # `metaclass=X` or similar — not a base class.
            continue
        name = _innermost_identifier(c, source)
        if not name:
            continue
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": "extends",
            "file": file_path,
            "line": c.start_point[0] + 1,
        })


def _handle_java_inheritance(container_node, source: bytes,
                             container_id: str, file_path: str,
                             raw_inherits: list) -> None:
    """Java separates `extends` (one superclass) from `implements` (N
    interfaces) at the grammar level: fields `superclass` and `super_interfaces`.
    """
    if container_node.type not in ("class_declaration", "interface_declaration",
                                   "enum_declaration"):
        return
    # superclass: extends X
    for c in container_node.children:
        if c.type == "superclass":
            for sub in c.children:
                if sub.is_named and sub.type in (
                    "type_identifier", "scoped_type_identifier",
                    "generic_type",
                ):
                    name = _innermost_identifier(sub, source)
                    if name:
                        raw_inherits.append({
                            "child": container_id,
                            "base_name": name,
                            "base_kind": "extends",
                            "file": file_path,
                            "line": sub.start_point[0] + 1,
                        })
        elif c.type in ("super_interfaces", "extends_interfaces"):
            # Both wrap a `type_list` of interface names.
            for sub in c.children:
                if sub.type == "type_list":
                    for t in sub.children:
                        if not t.is_named:
                            continue
                        name = _innermost_identifier(t, source)
                        if name:
                            raw_inherits.append({
                                "child": container_id,
                                "base_name": name,
                                "base_kind": "implements",
                                "file": file_path,
                                "line": t.start_point[0] + 1,
                            })


def _handle_ts_inheritance(container_node, source: bytes,
                           container_id: str, file_path: str,
                           raw_inherits: list) -> None:
    """TypeScript / TSX / JavaScript: `class_heritage > {extends_clause,
    implements_clause}`. JavaScript only has `extends` (no implements).
    Also handles `abstract_class_declaration`."""
    if container_node.type not in (
        "class_declaration", "abstract_class_declaration",
        "interface_declaration",
    ):
        return
    heritage = None
    for c in container_node.children:
        if c.type == "class_heritage":
            heritage = c
            break
        if c.type == "extends_type_clause":
            # interface_declaration in TS uses a flat extends_type_clause sibling
            heritage = container_node  # treat container itself as heritage source
            break
    bases = []
    if heritage is None:
        return
    if heritage is container_node:
        # Interface case: iterate the container's own extends_type_clause(s)
        for c in container_node.children:
            if c.type == "extends_type_clause":
                for sub in c.children:
                    if sub.is_named and sub.type in (
                        "type_identifier", "nested_type_identifier",
                        "generic_type", "identifier",
                    ):
                        name = _innermost_identifier(sub, source)
                        if name:
                            bases.append((name, "extends", sub))
    else:
        for clause in heritage.children:
            if clause.type == "extends_clause":
                kind = "extends"
            elif clause.type == "implements_clause":
                kind = "implements"
            else:
                continue
            for sub in clause.children:
                if not sub.is_named:
                    continue
                # Skip the literal 'extends'/'implements' keywords.
                if sub.type in ("extends", "implements"):
                    continue
                if sub.type in (
                    "identifier", "type_identifier", "nested_type_identifier",
                    "generic_type", "member_expression",
                ):
                    name = _innermost_identifier(sub, source)
                    if name:
                        bases.append((name, kind, sub))
    for name, kind, node in bases:
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": kind,
            "file": file_path,
            "line": node.start_point[0] + 1,
        })


def _handle_cpp_inheritance(container_node, source: bytes,
                            container_id: str, file_path: str,
                            raw_inherits: list) -> None:
    """C++: `class Foo : public Bar, protected Baz { ... }`. The bases live
    in a `base_class_clause` child."""
    if container_node.type not in ("class_specifier", "struct_specifier"):
        return
    base_clause = None
    for c in container_node.children:
        if c.type == "base_class_clause":
            base_clause = c
            break
    if base_clause is None:
        return
    for c in base_clause.children:
        if not c.is_named:
            continue
        # Each base may be wrapped or be a direct identifier/qualified_identifier.
        name = _innermost_identifier(c, source)
        if not name:
            continue
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": "extends_or_implements",  # C++ doesn't formally split
            "file": file_path,
            "line": c.start_point[0] + 1,
        })


def _handle_kotlin_inheritance(container_node, source: bytes,
                               container_id: str, file_path: str,
                               raw_inherits: list) -> None:
    """Kotlin: `class Foo : Bar(), IBaz { ... }`. Bases live in a
    `delegation_specifier` list under the class_declaration."""
    if container_node.type not in ("class_declaration", "object_declaration"):
        return
    # tree-sitter-kotlin uses `delegation_specifiers` wrapper sometimes; iterate.
    for c in container_node.children:
        if c.type in ("delegation_specifier", "user_type", "constructor_invocation"):
            name = _innermost_identifier(c, source)
            if name:
                raw_inherits.append({
                    "child": container_id,
                    "base_name": name,
                    "base_kind": "extends_or_implements",
                    "file": file_path,
                    "line": c.start_point[0] + 1,
                })
        elif c.type == "delegation_specifiers":
            for sub in c.children:
                if sub.is_named:
                    name = _innermost_identifier(sub, source)
                    if name:
                        raw_inherits.append({
                            "child": container_id,
                            "base_name": name,
                            "base_kind": "extends_or_implements",
                            "file": file_path,
                            "line": sub.start_point[0] + 1,
                        })


def _handle_swift_inheritance(container_node, source: bytes,
                              container_id: str, file_path: str,
                              raw_inherits: list) -> None:
    """Swift: `class Foo: Bar, ProtocolP { ... }`. The list of base types
    is exposed as `inheritance_specifier` children."""
    if container_node.type not in ("class_declaration", "protocol_declaration"):
        return
    for c in container_node.children:
        if c.type in ("inheritance_specifier", "user_type", "type_inheritance_clause"):
            # type_inheritance_clause wraps inheritance_specifier list
            if c.type == "type_inheritance_clause":
                for sub in c.children:
                    if sub.is_named:
                        name = _innermost_identifier(sub, source)
                        if name:
                            raw_inherits.append({
                                "child": container_id,
                                "base_name": name,
                                "base_kind": "extends_or_implements",
                                "file": file_path,
                                "line": sub.start_point[0] + 1,
                            })
            else:
                name = _innermost_identifier(c, source)
                if name:
                    raw_inherits.append({
                        "child": container_id,
                        "base_name": name,
                        "base_kind": "extends_or_implements",
                        "file": file_path,
                        "line": c.start_point[0] + 1,
                    })


def _handle_php_inheritance(container_node, source: bytes,
                            container_id: str, file_path: str,
                            raw_inherits: list) -> None:
    """PHP: `class Foo extends Bar implements IBaz, IQux`. The base class
    is in `base_clause` and interfaces in `class_interface_clause`."""
    if container_node.type not in ("class_declaration", "interface_declaration",
                                   "trait_declaration"):
        return
    for c in container_node.children:
        if c.type == "base_clause":
            for sub in c.children:
                if sub.is_named and sub.type in ("name", "qualified_name"):
                    name = _innermost_identifier(sub, source)
                    if name:
                        raw_inherits.append({
                            "child": container_id,
                            "base_name": name,
                            "base_kind": "extends",
                            "file": file_path,
                            "line": sub.start_point[0] + 1,
                        })
        elif c.type == "class_interface_clause":
            for sub in c.children:
                if sub.is_named and sub.type in ("name", "qualified_name"):
                    name = _innermost_identifier(sub, source)
                    if name:
                        raw_inherits.append({
                            "child": container_id,
                            "base_name": name,
                            "base_kind": "implements",
                            "file": file_path,
                            "line": sub.start_point[0] + 1,
                        })


def _handle_ruby_inheritance(container_node, source: bytes,
                             container_id: str, file_path: str,
                             raw_inherits: list) -> None:
    """Ruby: `class Foo < Bar`. Single superclass via the `superclass` field."""
    if container_node.type != "class":
        return
    sup = container_node.child_by_field_name("superclass")
    if sup is None:
        return
    # superclass node wraps a constant / scope_resolution.
    for c in sup.children:
        if c.is_named:
            name = _innermost_identifier(c, source)
            if name:
                raw_inherits.append({
                    "child": container_id,
                    "base_name": name,
                    "base_kind": "extends",
                    "file": file_path,
                    "line": c.start_point[0] + 1,
                })
                return
    name = _innermost_identifier(sup, source)
    if name:
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": "extends",
            "file": file_path,
            "line": sup.start_point[0] + 1,
        })


def _handle_rust_inheritance(container_node, source: bytes,
                             container_id: str, file_path: str,
                             raw_inherits: list) -> None:
    """Rust: trait implementation `impl Trait for Type { ... }`. The
    container_types for Rust include `impl_item`, which has a `trait` field
    and a `type` field. We emit an `implements` edge from the *type* to the
    *trait* — i.e. Type-implements-Trait. That's the analog of "Type is a
    concrete realization of Trait" that callers want.
    """
    if container_node.type != "impl_item":
        return
    trait = container_node.child_by_field_name("trait")
    type_node = container_node.child_by_field_name("type")
    if trait is None or type_node is None:
        return
    trait_name = _innermost_identifier(trait, source)
    if not trait_name:
        return
    # Important: the `child` of the inherits relation should be the concrete
    # *type* (the implementer), not the impl block itself. The walker
    # registered the impl block as `container_id` though, and at the symbol
    # index level the type may or may not be defined in the same file.
    # Practical compromise: emit the edge with `child=container_id` (the impl
    # block, named `impl_<TraitName>_<TypeName>` via _name_of fallback).
    # Cross-file resolution via the type name as a separate `_inherits` base.
    type_name = _innermost_identifier(type_node, source)
    raw_inherits.append({
        "child": container_id,
        "base_name": trait_name,
        "base_kind": "implements",
        "file": file_path,
        "line": container_node.start_point[0] + 1,
        # Annotate with the impl's own type so queries can map back.
        "impl_target_type": type_name,
    })


def _handle_scala_import(node, source: bytes, file_id: str, edges: list, file_path: str) -> None:
    """Scala: `import a.b.C` / `import a.b.{C, D}`. The dotted prefix is a
    sequence of `path` fields; emit one `imports` edge labelled with the full
    dotted module path, targeting its first segment (best-effort, like the
    other languages)."""
    if node.type != "import_declaration":
        return
    parts = [_text(c, source).strip() for c in node.children_by_field_name("path")]
    parts = [p for p in parts if p]
    if not parts:
        return
    mod = ".".join(parts)
    edges.append({
        "source": file_id,
        "target": _make_id(parts[0]),
        "relation": "imports",
        "module": mod,
    })


def _handle_scala_inheritance(container_node, source: bytes, container_id: str,
                              file_path: str, raw_inherits: list) -> None:
    """Scala: `class A(...) extends Base with Trait1`. The bases live in an
    `extend` field (an `extends_clause`) whose `type` children are the
    supertypes. Scala doesn't syntactically separate class extension from
    trait mixin, so every entry is emitted as `extends`."""
    if container_node.type not in (
        "class_definition", "object_definition", "trait_definition",
    ):
        return
    ext = container_node.child_by_field_name("extend")
    if ext is None:
        return
    for t in ext.children_by_field_name("type"):
        name = _innermost_identifier(t, source)
        if not name:
            continue
        raw_inherits.append({
            "child": container_id,
            "base_name": name,
            "base_kind": "extends",
            "file": file_path,
            "line": t.start_point[0] + 1,
        })


# ---------------------------------------------------------------------------
# Rules table
# ---------------------------------------------------------------------------
#
def _handle_objc_inheritance(container_node, source: bytes,
                             container_id: str, file_path: str,
                             raw_inherits: list) -> None:
    """Objective-C inheritance / protocol adoption:

      @interface Foo : Bar <ProtoA, ProtoB>   → extends Bar, implements ProtoA/B
      @interface Foo (Category) <Proto>         → implements Proto (no superclass)
      @protocol P <NSObject>                     → implements NSObject

    The superclass sits under a `superclass` field; adopted protocols are
    identifiers inside a `parameterized_arguments` (class) or
    `protocol_reference_list` (protocol) child.
    """
    sup = container_node.child_by_field_name("superclass")
    if sup is not None:
        name = _innermost_identifier(sup, source)
        if name:
            raw_inherits.append({
                "child": container_id,
                "base_name": name,
                "base_kind": "extends",
                "file": file_path,
                "line": sup.start_point[0] + 1,
            })
    for c in container_node.children:
        if c.type not in ("parameterized_arguments", "protocol_reference_list"):
            continue
        for t in c.children:
            if not t.is_named:
                continue
            name = _innermost_identifier(t, source)
            if not name:
                continue
            raw_inherits.append({
                "child": container_id,
                "base_name": name,
                "base_kind": "implements",
                "file": file_path,
                "line": t.start_point[0] + 1,
            })


# Keep in sync with `chunking._LANG_RULES`: same language keys, same node
# types where possible. SQL is intentionally absent — its DDL has no
# call/inheritance/import graph to extract. Languages without a graph rule
# (markdown, shaders, JSON, SQL) are simply skipped by `extract_file`.


GRAPH_RULES: dict = {
    "python": LangGraphRules(
        container_types=frozenset({"class_definition"}),
        function_types=frozenset({"function_definition"}),
        import_types=frozenset({"import_statement", "import_from_statement"}),
        call_types=frozenset({"call"}),
        call_function_field="function",
        call_accessor_types=frozenset({"attribute"}),
        call_accessor_field="attribute",
        function_boundary_types=frozenset({"function_definition"}),
        import_handler=_handle_python_import,
        inheritance_handler=_handle_python_inheritance,
    ),
    "javascript": LangGraphRules(
        container_types=frozenset({"class_declaration"}),
        function_types=frozenset({"function_declaration", "method_definition", "generator_function_declaration"}),
        import_types=frozenset({"import_statement"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"member_expression"}),
        call_accessor_field="property",
        function_boundary_types=frozenset({"function_declaration", "method_definition", "arrow_function", "function_expression"}),
        import_handler=_handle_js_import,
        inheritance_handler=_handle_ts_inheritance,  # same class_heritage shape
    ),
    "typescript": LangGraphRules(
        container_types=frozenset({"class_declaration", "abstract_class_declaration",
                                   "interface_declaration", "namespace_declaration", "module"}),
        function_types=frozenset({"function_declaration", "method_definition", "abstract_method_signature"}),
        import_types=frozenset({"import_statement"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"member_expression"}),
        call_accessor_field="property",
        function_boundary_types=frozenset({"function_declaration", "method_definition", "arrow_function", "function_expression"}),
        import_handler=_handle_js_import,
        inheritance_handler=_handle_ts_inheritance,
    ),
    "tsx": LangGraphRules(
        container_types=frozenset({"class_declaration", "abstract_class_declaration",
                                   "interface_declaration", "namespace_declaration"}),
        function_types=frozenset({"function_declaration", "method_definition"}),
        import_types=frozenset({"import_statement"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"member_expression"}),
        call_accessor_field="property",
        function_boundary_types=frozenset({"function_declaration", "method_definition", "arrow_function", "function_expression"}),
        import_handler=_handle_js_import,
        inheritance_handler=_handle_ts_inheritance,
    ),
    "c_sharp": LangGraphRules(
        container_types=frozenset({
            "class_declaration", "interface_declaration", "struct_declaration",
            "record_declaration", "namespace_declaration",
            "file_scoped_namespace_declaration",
        }),
        function_types=frozenset({
            "method_declaration", "constructor_declaration", "destructor_declaration",
            "property_declaration", "indexer_declaration", "operator_declaration",
        }),
        import_types=frozenset({"using_directive"}),
        call_types=frozenset({"invocation_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"member_access_expression"}),
        call_accessor_field="name",
        function_boundary_types=frozenset({"method_declaration", "constructor_declaration", "local_function_statement"}),
        import_handler=_handle_csharp_import,
        inheritance_handler=_handle_csharp_inheritance,
    ),
    "java": LangGraphRules(
        container_types=frozenset({"class_declaration", "interface_declaration", "enum_declaration"}),
        function_types=frozenset({"method_declaration", "constructor_declaration"}),
        import_types=frozenset({"import_declaration"}),
        call_types=frozenset({"method_invocation"}),
        call_function_field="name",  # Java's method_invocation uses field "name" directly
        call_accessor_types=frozenset(),  # No accessor wrapping; "object" field is the receiver
        call_accessor_field="name",
        function_boundary_types=frozenset({"method_declaration", "constructor_declaration"}),
        import_handler=_handle_java_import,
        inheritance_handler=_handle_java_inheritance,
    ),
    "go": LangGraphRules(
        container_types=frozenset(),
        function_types=frozenset({"function_declaration", "method_declaration"}),
        import_types=frozenset({"import_declaration"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"selector_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_declaration", "method_declaration", "func_literal"}),
        import_handler=_handle_go_import,
    ),
    "rust": LangGraphRules(
        container_types=frozenset({"impl_item", "mod_item", "trait_item"}),
        function_types=frozenset({"function_item"}),
        import_types=frozenset({"use_declaration"}),
        # Rust splits regular calls (`f()`) from method calls (`x.m()`) into
        # different node types. We accept both as call edges.
        call_types=frozenset({"call_expression", "method_call_expression"}),
        call_function_field="function",  # call_expression
        call_accessor_types=frozenset({"field_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_item", "closure_expression"}),
        import_handler=_handle_rust_import,
        inheritance_handler=_handle_rust_inheritance,
    ),
    "c": LangGraphRules(
        container_types=frozenset(),
        function_types=frozenset({"function_definition"}),
        import_types=frozenset({"preproc_include"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"field_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_definition"}),
        import_handler=_handle_c_include,
    ),
    "cpp": LangGraphRules(
        container_types=frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
        function_types=frozenset({"function_definition"}),
        import_types=frozenset({"preproc_include"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"field_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_definition", "lambda_expression"}),
        import_handler=_handle_c_include,
        inheritance_handler=_handle_cpp_inheritance,
    ),
    "objc": LangGraphRules(
        container_types=frozenset({
            "class_interface", "class_implementation", "protocol_declaration",
        }),
        function_types=frozenset({"method_declaration", "method_definition"}),
        import_types=frozenset({"preproc_include"}),
        # `[obj msg:arg]` is a message_expression whose selector is the `method`
        # field (no accessor wrapper); plain C `f()` is a call_expression.
        call_types=frozenset({"message_expression", "call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"field_expression"}),
        call_accessor_field="field",
        direct_member_call_types=frozenset({"message_expression"}),
        direct_member_call_field="method",
        function_boundary_types=frozenset({
            "method_declaration", "method_definition", "function_definition",
        }),
        import_handler=_handle_c_include,
        inheritance_handler=_handle_objc_inheritance,
    ),
    "ruby": LangGraphRules(
        container_types=frozenset({"class", "module"}),
        function_types=frozenset({"method", "singleton_method"}),
        # Ruby `require` is a call, not a dedicated import node. We register
        # `call` so the walker emits the require edges via the handler, but
        # the handler is itself a no-op for non-require calls.
        import_types=frozenset({"call"}),
        call_types=frozenset({"call"}),
        call_function_field="method",
        call_accessor_types=frozenset(),
        call_accessor_field="method",
        function_boundary_types=frozenset({"method", "singleton_method"}),
        import_handler=_handle_ruby_require,
        inheritance_handler=_handle_ruby_inheritance,
    ),
    "php": LangGraphRules(
        container_types=frozenset({
            "class_declaration", "interface_declaration", "trait_declaration",
            "namespace_definition",
        }),
        function_types=frozenset({"method_declaration", "function_definition"}),
        import_types=frozenset({"namespace_use_declaration"}),
        # PHP splits regular `f()`, `$o->m()`, and `X::p()` into three node
        # types. function_call_expression has the regular `function` field;
        # the other two expose the method name directly via the `name`
        # field on the call node itself (no accessor wrapping).
        call_types=frozenset({
            "function_call_expression", "member_call_expression", "scoped_call_expression",
        }),
        call_function_field="function",
        direct_member_call_types=frozenset({"member_call_expression", "scoped_call_expression"}),
        direct_member_call_field="name",
        function_boundary_types=frozenset({"method_declaration", "function_definition"}),
        import_handler=_handle_php_use,
        inheritance_handler=_handle_php_inheritance,
    ),
    "kotlin": LangGraphRules(
        container_types=frozenset({"class_declaration", "object_declaration"}),
        function_types=frozenset({"function_declaration", "secondary_constructor"}),
        import_types=frozenset({"import"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"navigation_expression"}),
        call_accessor_field="name",
        function_boundary_types=frozenset({"function_declaration", "lambda_literal"}),
        import_handler=_handle_kotlin_import,
        inheritance_handler=_handle_kotlin_inheritance,
    ),
    "swift": LangGraphRules(
        container_types=frozenset({"class_declaration", "protocol_declaration"}),
        function_types=frozenset({
            "function_declaration", "init_declaration", "deinit_declaration",
            "protocol_function_declaration",
        }),
        import_types=frozenset({"import_declaration"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        call_accessor_types=frozenset({"navigation_expression"}),
        call_accessor_field="suffix",
        function_boundary_types=frozenset({"function_declaration", "init_declaration", "closure_expression"}),
        import_handler=_handle_swift_import,
        inheritance_handler=_handle_swift_inheritance,
    ),
    "scala": LangGraphRules(
        container_types=frozenset({"class_definition", "object_definition", "trait_definition"}),
        function_types=frozenset({"function_definition", "function_declaration"}),
        import_types=frozenset({"import_declaration"}),
        call_types=frozenset({"call_expression"}),
        call_function_field="function",
        # `a.speak()` → function: field_expression → field: identifier.
        call_accessor_types=frozenset({"field_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_definition", "function_declaration"}),
        import_handler=_handle_scala_import,
        inheritance_handler=_handle_scala_inheritance,
    ),
    "lua": LangGraphRules(
        # No classes — function nodes + a call graph. `M.f(...)` resolves via
        # dot_index_expression's `field`; `t:m()` via method_index_expression
        # (last-child fallback). The callee field is `name`, not `function`.
        container_types=frozenset(),
        function_types=frozenset({"function_declaration"}),
        import_types=frozenset(),
        call_types=frozenset({"function_call"}),
        call_function_field="name",
        call_accessor_types=frozenset({"dot_index_expression", "method_index_expression"}),
        call_accessor_field="field",
        function_boundary_types=frozenset({"function_declaration"}),
    ),
    "bash": LangGraphRules(
        # Shell: function nodes + a call graph between defined functions. Every
        # `command` is a candidate call; only those resolving to a defined
        # function become edges (echo/ls/source/... are dropped at resolution).
        # The callee name is the `command_name`'s inner word (last-child fallback).
        container_types=frozenset(),
        function_types=frozenset({"function_definition"}),
        import_types=frozenset(),
        call_types=frozenset({"command"}),
        call_function_field="name",
        call_accessor_types=frozenset({"command_name"}),
        call_accessor_field="name",
        function_boundary_types=frozenset({"function_definition"}),
    ),
}


# Convenience: languages the graph layer can extract structurally. Anything
# not in this set is fed to the chunker only and contributes nothing to the
# graph (no error, just silently skipped).
GRAPH_LANGUAGES = frozenset(GRAPH_RULES.keys())


# ---------------------------------------------------------------------------
# Call resolution
# ---------------------------------------------------------------------------


def _extract_callee(call_node, rules: LangGraphRules, source: bytes) -> "Optional[tuple[str, bool]]":
    """Return `(callee_name, is_member)` or None when we can't identify a callee.

    `is_member` is True for `obj.foo()` style calls — the resolver gives
    these lower priority to avoid false positives across the codebase.
    """
    # Grammars where the call node itself exposes the method name as a
    # direct field (PHP member_call_expression / scoped_call_expression).
    # We treat these as member calls without descending into an accessor.
    if call_node.type in rules.direct_member_call_types:
        name_node = call_node.child_by_field_name(rules.direct_member_call_field)
        if name_node is not None:
            name = _text(name_node, source).strip()
            return (name, True) if name else None
        return None

    callee = call_node.child_by_field_name(rules.call_function_field)

    # Java is a special case: `method_invocation` exposes the method name
    # directly under field `name`, not under a nested "function" field.
    if callee is None and rules.call_function_field == "name":
        callee = call_node.child_by_field_name("name")

    # Kotlin and a few other grammars don't expose a `function` field at
    # all — the callee is just the first named child of the call node
    # (an identifier for direct calls, a navigation_expression for method
    # calls). Fall back to the first named child that looks like one of
    # those forms.
    if callee is None:
        for c in call_node.children:
            if not c.is_named:
                continue
            if c.type in ("identifier", "simple_identifier", "name") \
                    or c.type in rules.call_accessor_types:
                callee = c
                break
    if callee is None:
        return None

    # If the callee is itself an accessor node (`a.b`, `a::b`), the method
    # name lives inside it under `call_accessor_field`.
    if callee.type in rules.call_accessor_types:
        method = callee.child_by_field_name(rules.call_accessor_field)
        if method is not None:
            name = _text(method, source).strip()
            return (name, True) if name else None
        # Some grammars don't expose a field — fall back to the LAST named child.
        named_children = [c for c in callee.children if c.is_named]
        if named_children:
            name = _text(named_children[-1], source).strip()
            return (name, True) if name else None
        return None

    # Plain identifier — direct call `foo()`.
    if callee.type in ("identifier", "simple_identifier", "name"):
        name = _text(callee, source).strip()
        return (name, False) if name else None

    # Fallback: try child_by_field_name("name") on the callee subtree.
    nm = callee.child_by_field_name("name")
    if nm is not None:
        name = _text(nm, source).strip()
        return (name, False) if name else None

    return None


def _walk_calls(node, caller_id: str, rules: LangGraphRules, source: bytes,
                local_index: dict, edges: list, raw_calls: list, file_path: str) -> None:
    """Walk a function body, emitting call edges (resolved or raw)."""
    for child in node.children:
        # Don't descend into nested function/method definitions — those are
        # independent callers and the walker handles them at the top level.
        if child.type in rules.function_boundary_types:
            continue
        if child.type in rules.call_types:
            extracted = _extract_callee(child, rules, source)
            if extracted is not None:
                callee_name, is_member = extracted
                callee_key = callee_name.lower()
                target_id = local_index.get(callee_key)
                if target_id is not None and target_id != caller_id:
                    # Resolved within this file, and not a self-reference.
                    edges.append({
                        "source": caller_id,
                        "target": target_id,
                        "relation": "calls",
                        "confidence": "extracted",
                    })
                elif target_id is None:
                    # Callee not defined in this file — defer to the cross-file
                    # resolver (it has the global symbol index + member caution).
                    raw_calls.append({
                        "caller": caller_id,
                        "callee_name": callee_name,
                        "is_member": is_member,
                        "file": file_path,
                        "line": child.start_point[0] + 1,
                    })
                # else: target_id == caller_id — a name-only self match. Drop it.
                #   Node ids are NOT signature-aware, so overloaded methods
                #   (Foo(a) and Foo(a, b)) collapse to a single id; a call from
                #   one overload to another then looks identical to recursion.
                #   A member self-match (obj.Foo() inside Foo) is also an outright
                #   wrong receiver. A self-loop `calls` edge is noise in every
                #   navigation view, so emit nothing rather than guess.
        # Recurse into the children — body / block / expression-statement etc.
        _walk_calls(child, caller_id, rules, source, local_index, edges, raw_calls, file_path)


# ---------------------------------------------------------------------------
# Top-level walker
# ---------------------------------------------------------------------------


def _walk_declarations(node, rules: LangGraphRules, source: bytes, file_path: str,
                       file_id: str, stem: str, container_path: list,
                       nodes: list, edges: list, local_index: dict,
                       function_bodies: list, raw_inherits: list) -> None:
    """First pass: collect class/function nodes, contains edges, imports,
    inheritance, and a list of (function_id, body_node) pairs for the call walker.
    """
    for child in node.children:
        if child.type in rules.import_types:
            if rules.import_handler is not None:
                rules.import_handler(child, source, file_id, edges, file_path)
            # Note: Ruby's `call` type is both an import_type AND a call_type.
            # We don't `continue` here — falling through lets the same node
            # also be considered as a call within a function body (it isn't
            # at the top level, but the recursive descent handles that).
        if child.type in rules.function_types:
            name = _name_of(child, source) or "<anonymous>"
            qualified = ".".join(container_path + [name]) if container_path else name
            node_id = _make_id(stem, qualified)
            nodes.append({
                "id": node_id,
                "label": qualified,
                "kind": "function",
                "subkind": child.type,
                "file": file_path,
                "start_line": child.start_point[0] + 1,
                "end_line": child.end_point[0] + 1,
                "lang_key": None,  # filled in by caller
            })
            local_index[name.lower()] = node_id
            # Also index the bare leaf (last segment) for member-call resolution.
            local_index.setdefault(name.split(".")[-1].lower(), node_id)
            # If we're inside a container, emit a contains edge.
            if container_path:
                parent_id = _make_id(stem, ".".join(container_path))
                edges.append({
                    "source": parent_id,
                    "target": node_id,
                    "relation": "contains",
                    "confidence": "extracted",
                })
            function_bodies.append((node_id, child))
            # Do NOT recurse into a function — its body is walked separately.
            continue
        if child.type in rules.container_types:
            cname = _name_of(child, source) or "<container>"
            qualified = ".".join(container_path + [cname]) if container_path else cname
            container_id = _make_id(stem, qualified)
            nodes.append({
                "id": container_id,
                "label": qualified,
                "kind": "class",
                "subkind": child.type,
                "file": file_path,
                "start_line": child.start_point[0] + 1,
                "end_line": child.end_point[0] + 1,
                "lang_key": None,
            })
            local_index[cname.lower()] = container_id
            # Inheritance: collect raw_inherits BEFORE recursing into the
            # container's body — the handler reads the same `child` node
            # for its base list / heritage clause.
            if rules.inheritance_handler is not None:
                rules.inheritance_handler(
                    child, source, container_id, file_path, raw_inherits,
                )
            # Recurse to find nested classes / methods.
            _walk_declarations(
                child, rules, source, file_path, file_id, stem,
                container_path + [cname], nodes, edges, local_index,
                function_bodies, raw_inherits,
            )
            continue
        # Generic recursion for wrapper nodes (decorated_definition,
        # modifiers, namespace bodies that don't appear in container_types,
        # etc.).
        _walk_declarations(
            child, rules, source, file_path, file_id, stem,
            container_path, nodes, edges, local_index, function_bodies,
            raw_inherits,
        )


def extract_file(abs_path: str, content: str) -> "Optional[dict]":
    """Extract graph data for a single file. Returns None if the file has
    no graph rules (markdown, shaders, JSON, etc.) — those files are still
    chunked for the RAG, they just don't contribute to the structural graph.

    Return shape:
        {
          "file_id":   "...",          # node ID for the file itself
          "file":      "/abs/path",
          "lang_key":  "python",
          "nodes":     [ {id, label, kind, subkind, file, start_line, end_line, lang_key}, ... ],
          "edges":     [ {source, target, relation, confidence, ...}, ... ],
          "raw_calls": [ {caller, callee_name, is_member, file, line}, ... ],
        }
    """
    parsed = parse_file(abs_path, content)
    if parsed is None:
        return None
    tree, lang_key, _chunk_rules = parsed
    if lang_key not in GRAPH_RULES:
        return None
    rules = GRAPH_RULES[lang_key]
    source = content.encode("utf-8")

    abs_path_norm = os.path.normpath(os.path.abspath(abs_path))
    stem = _file_stem(abs_path_norm)
    file_id = _make_id(stem)

    nodes: list = []
    edges: list = []
    local_index: dict = {}
    function_bodies: list = []
    raw_inherits: list = []

    # The file is itself a graph node — gives us something to attach imports to.
    nodes.append({
        "id": file_id,
        "label": stem,
        "kind": "file",
        "subkind": "file",
        "file": abs_path_norm,
        "start_line": 1,
        "end_line": tree.root_node.end_point[0] + 1,
        "lang_key": lang_key,
    })

    _walk_declarations(
        tree.root_node, rules, source, abs_path_norm, file_id, stem,
        container_path=[], nodes=nodes, edges=edges,
        local_index=local_index, function_bodies=function_bodies,
        raw_inherits=raw_inherits,
    )

    # Fill in lang_key on every node we created.
    for n in nodes:
        if n.get("lang_key") is None:
            n["lang_key"] = lang_key

    # Second pass: walk every function body collecting call edges.
    raw_calls: list = []
    for caller_id, body in function_bodies:
        _walk_calls(body, caller_id, rules, source, local_index, edges, raw_calls, abs_path_norm)

    return {
        "file_id": file_id,
        "file": abs_path_norm,
        "lang_key": lang_key,
        "nodes": nodes,
        "edges": edges,
        "raw_calls": raw_calls,
        "raw_inherits": raw_inherits,
    }
