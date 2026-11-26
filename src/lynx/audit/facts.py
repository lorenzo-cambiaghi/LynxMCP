"""Facts about C# types that the code graph does not record, for the audit.

The graph knows symbols, their lines, calls, imports and bases. A SOLID
audit also needs what a declaration says about itself: modifiers, fields
and whether they can change, how many branches a method has, a body that
only throws, the `switch`es and the type tests. They come from the same
tree-sitter grammar the chunker uses, read once per file.

A C# 14 `extension(T x) { ... }` block parses as a constructor named
`extension`; its members are read as methods of the enclosing type, not as
one long method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..graph import cs_recovery

TYPE_NODES = {
    "class_declaration": "class",
    "struct_declaration": "struct",
    "record_declaration": "record",
    "record_struct_declaration": "record",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
}
METHOD_NODES = {"method_declaration", "constructor_declaration", "operator_declaration",
                "conversion_operator_declaration", "destructor_declaration", "local_function_statement"}
BRANCH_NODES = {"if_statement", "for_statement", "foreach_statement", "while_statement", "do_statement",
                "switch_section", "switch_expression_arm", "catch_clause", "conditional_expression"}
TYPE_TEST_NODES = {"is_pattern_expression", "as_expression", "cast_expression", "declaration_pattern",
                   "type_pattern", "recursive_pattern"}
NOT_IMPLEMENTED = ("NotImplementedException", "NotSupportedException")
# Calls that change the collection they are made on.
MUTATORS = {"Add", "AddRange", "Remove", "RemoveAt", "RemoveAll", "RemoveRange", "RemoveWhere", "Clear",
            "Insert", "InsertRange", "TryAdd", "TryRemove", "TryUpdate", "AddOrUpdate", "GetOrAdd",
            "Enqueue", "Dequeue", "TryDequeue", "Push", "Pop", "TryPop", "Sort", "Reverse",
            "UnionWith", "ExceptWith", "IntersectWith", "SymmetricExceptWith", "AddFirst", "AddLast",
            "RemoveFirst", "RemoveLast"}


def _last_name(node, src: bytes) -> Optional[str]:
    """`X`, `this.X`, `Type.X`: X."""
    if node is None:
        return None
    if node.type == "identifier":
        return _text(node, src)
    if node.type == "member_access_expression":
        n = node.child_by_field_name("name")
        return _text(n, src) if n is not None else None
    return None


@dataclass
class Method:
    name: str
    line: int
    end: int
    params: int
    branches: int
    modifiers: set
    body: str              # "normal", "empty", "throws" (only throws NotImplemented/NotSupported), "none"
    names: set             # identifiers used in the body
    type_tests: int
    writes: set = field(default_factory=set)   # names the body changes: `X.Add(..)`, `X[k] = v`, `X = ..`
    explicit: bool = False                     # `void IFoo.Bar()`: an interface member, private by syntax
    attributes: str = ""                       # its attribute lists, as written: `[ModuleInitializer]`


@dataclass
class Field:
    name: str
    line: int
    modifiers: set
    type: str
    attributes: str = ""
    kind: str = "field"    # or "property": `static T X { get; set; }`, a static field with a name on it


@dataclass
class Switch:
    subject: str
    line: int
    cases: int
    over: Optional[str] = None   # the type its labels name values by: MathOperator of `case MathOperator.Add:`


@dataclass
class TypeDecl:
    name: str              # qualified within the file: Outer.Inner
    kind: str
    file: str
    namespace: str
    line: int
    end: int
    modifiers: set
    bases: list
    methods: list = field(default_factory=list)
    fields: list = field(default_factory=list)
    properties: list = field(default_factory=list)
    static_setters: list = field(default_factory=list)   # Field(kind="property") per static auto-property with a setter
    switches: list = field(default_factory=list)
    declines: set = field(default_factory=set)   # what it says it cannot do: `CanWrite => false`, `IsReadOnly => true`
    members: int = 0       # declared members (methods, properties, events, indexers)
    writes: set = field(default_factory=set)   # names its members change after construction (static constructors left out)


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _modifiers(node, src: bytes) -> set:
    return {_text(c, src) for c in node.children if c.type == "modifier"}


def _name(node, src: bytes) -> Optional[str]:
    n = node.child_by_field_name("name")
    if n is None:
        for c in node.children:
            if c.type == "identifier":
                n = c
                break
    return _text(n, src) if n is not None else None


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _walk_own(type_node):
    """The nodes of a type declaration, without the types nested in it: each
    of those is read on its own, and its switches belong to it alone."""
    stack = list(type_node.children)
    while stack:
        n = stack.pop()
        yield n
        if n.type not in TYPE_NODES:
            stack.extend(n.children)


def _body_of(node):
    for c in node.children:
        if c.type in ("block", "arrow_expression_clause"):
            return c
    return None


def _body_kind(body, src: bytes) -> str:
    if body is None:
        return "none"
    inner = [c for c in body.named_children if c.type != "comment"]
    if body.type == "block" and not inner:
        return "empty"
    if len(inner) == 1:
        one = inner[0]
        if one.type == "expression_statement" and one.named_children:
            one = one.named_children[0]
        if one.type in ("throw_statement", "throw_expression") and any(w in _text(one, src) for w in NOT_IMPLEMENTED):
            return "throws"
    return "normal"


def _writes(node, src: bytes) -> set:
    """The names a body changes: `X.Add(..)` and the other mutators,
    `X[k] = v`, `X = ..` (with `this.` or a type in front, too)."""
    out = set()
    for n in _walk(node):
        t = n.type
        if t == "invocation_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "member_access_expression":
                verb = fn.child_by_field_name("name")
                if verb is not None and _text(verb, src) in MUTATORS:
                    target = _last_name(fn.child_by_field_name("expression"), src)
                    if target:
                        out.add(target)
        elif t == "assignment_expression":
            left = n.child_by_field_name("left")
            if left is not None and left.type == "element_access_expression":
                left = left.child_by_field_name("expression")
            target = _last_name(left, src)
            if target:
                op = n.child_by_field_name("operator")
                # `X ??= Build()` fills X once, the first time it is read: kept
                # apart (LAZY + name), so a field written only so is a constant.
                out.add(LAZY + target if op is not None and op.type == "??=" else target)
    return out


# The prefix of a name `_writes` saw filled by `??=` only.
LAZY = "??"


def _method(node, src: bytes, name: Optional[str] = None) -> Method:
    body = _body_of(node)
    branches, tests, names = 0, 0, set()
    writes = _writes(body, src) if body is not None else set()
    if body is not None:
        for n in _walk(body):
            t = n.type
            if t in BRANCH_NODES:
                branches += 1
            elif t == "binary_expression":
                op = n.child_by_field_name("operator")
                if op is not None and _text(op, src) in ("&&", "||", "??"):
                    branches += 1
            elif t in TYPE_TEST_NODES:
                tests += 1
            elif t == "identifier":
                names.add(_text(n, src))
    plist = node.child_by_field_name("parameters")
    if plist is None:
        plist = next((c for c in node.children if c.type == "parameter_list"), None)
    params = sum(1 for c in plist.named_children if c.type == "parameter") if plist is not None else 0
    return Method(name=name or _name(node, src) or "?", line=node.start_point[0] + 1, end=node.end_point[0] + 1,
                  params=params, branches=branches, modifiers=_modifiers(node, src), body=_body_kind(body, src),
                  names=names, type_tests=tests, writes=writes,
                  explicit=any(c.type == "explicit_interface_specifier" for c in node.children),
                  attributes=" ".join(_text(c, src) for c in node.children if c.type == "attribute_list"))


def _bases(node, src: bytes) -> list:
    out = []
    for c in node.children:
        if c.type == "base_list":
            for b in c.named_children:
                text = _text(b, src).strip()
                if b.type == "primary_constructor_base_type":
                    text = text.split("(")[0].strip()
                out.append(text)
    return out


def _read_type(node, src: bytes, file: str, namespace: str, outer: str, out: list) -> None:
    name = _name(node, src) or "?"
    qual = f"{outer}.{name}" if outer else name
    decl = TypeDecl(name=qual, kind=TYPE_NODES[node.type], file=file, namespace=namespace,
                    line=node.start_point[0] + 1, end=node.end_point[0] + 1,
                    modifiers=_modifiers(node, src), bases=_bases(node, src))
    out.append(decl)
    body = next((c for c in node.children if c.type in ("declaration_list", "enum_member_declaration_list")), None)
    if body is None:
        return
    # Members the grammar could not parse cleanly sit in an ERROR node of the body.
    members = []
    for m in body.named_children:
        if m.type == "ERROR":
            members.extend(x for x in cs_recovery.children(m) if x.is_named)
        else:
            members.append(m)
    for m in members:
        t = m.type
        if t in TYPE_NODES:
            _read_type(m, src, file, namespace, qual, out)
        elif t == "constructor_declaration" and _name(m, src) == "extension":
            # C# 14 extension block: its members belong to the enclosing type.
            block = _body_of(m)
            for inner in (block.named_children if block is not None else ()):
                if inner.type in METHOD_NODES:
                    decl.methods.append(_method(inner, src))
                    decl.members += 1
                    decl.writes |= decl.methods[-1].writes
        elif t in METHOD_NODES:
            meth = _method(m, src)
            decl.methods.append(meth)
            decl.members += 1
            if not (t == "constructor_declaration" and "static" in meth.modifiers):
                decl.writes |= meth.writes
        elif t == "field_declaration":
            mods = _modifiers(m, src)
            var = next((c for c in m.children if c.type == "variable_declaration"), None)
            if var is None:
                continue
            typ = var.child_by_field_name("type")
            typ_text = _text(typ, src) if typ is not None else ""
            attrs = " ".join(_text(c, src) for c in m.children if c.type == "attribute_list")
            for v in var.named_children:
                if v.type == "variable_declarator":
                    decl.fields.append(Field(name=_name(v, src) or "?", line=v.start_point[0] + 1,
                                             modifiers=mods, type=typ_text, attributes=attrs))
        elif t in ("property_declaration", "event_declaration", "event_field_declaration", "indexer_declaration"):
            decl.members += 1
            for acc in m.children:
                if acc.type in ("accessor_list", "arrow_expression_clause"):
                    decl.writes |= _writes(acc, src)
            pname = _name(m, src)
            if pname:
                mods = _modifiers(m, src)
                decl.properties.append((pname, mods))
                said = _constant_bool(m, src) if t == "property_declaration" else None
                if (pname.startswith("Can") and said is False) or (pname in ("IsReadOnly", "IsFixedSize") and said is True):
                    decl.declines.add(pname)
                if t == "property_declaration" and "static" in mods and _auto_setter(m):
                    typ = m.child_by_field_name("type")
                    attrs = " ".join(_text(c, src) for c in m.children if c.type == "attribute_list")
                    decl.static_setters.append(Field(
                        name=pname, line=m.start_point[0] + 1, modifiers=mods,
                        type=_text(typ, src) if typ is not None else "", attributes=attrs, kind="property"))
    for n in _walk_own(node):
        if n.type in ("switch_statement", "switch_expression"):
            subj = n.child_by_field_name("value")
            if subj is None:
                subj = next((c for c in n.named_children if c.type not in ("switch_body", "switch_expression_arm")), None)
            cases = sum(1 for c in _walk(n) if c.type in ("switch_section", "switch_expression_arm"))
            if subj is not None:
                decl.switches.append(Switch(subject=_text(subj, src).strip("() "), line=n.start_point[0] + 1, cases=cases,
                                            over=_switch_over(n, src)))


def _switch_over(n, src: bytes) -> Optional[str]:
    """The type a switch's labels name their values by: MathOperator of
    `case MathOperator.Add:` and of `MathOperator.Add or MathOperator.Min =>`.
    Two switches on parameters both called `op` are one subject only when
    they are on one type. None when the labels name no type, or no one type
    names most of them."""
    if n.type == "switch_expression":
        labels = [a.named_children[0] for a in n.named_children if a.type == "switch_expression_arm" and a.named_children]
    else:
        body = n.child_by_field_name("body") or next((c for c in n.named_children if c.type == "switch_body"), None)
        labels = [c for sec in (body.named_children if body is not None else ()) if sec.type == "switch_section"
                  for c in sec.named_children if c.type.endswith("pattern")]
    seen: dict = {}
    for label in labels:
        for c in _walk(label):
            if c.type == "constant_pattern" and c.named_children and c.named_children[0].type == "member_access_expression":
                q = c.named_children[0].child_by_field_name("expression")
                if q is not None:
                    name = _text(q, src).rsplit(".", 1)[-1].strip()
                    seen[name] = seen.get(name, 0) + 1
    if not seen:
        return None
    name, count = max(seen.items(), key=lambda kv: kv[1])
    return name if count * 2 > sum(seen.values()) else None


def _constant_bool(prop, src: bytes) -> Optional[bool]:
    """The literal a property always gives: `=> false`, `{ get => true; }`,
    `{ get { return false; } }`. None for anything computed."""
    bodies = [c for c in prop.children if c.type == "arrow_expression_clause"]
    for acc_list in prop.children:
        if acc_list.type == "accessor_list":
            gets = [a for a in acc_list.named_children if a.type == "accessor_declaration"
                    and any(not c.is_named and _text(c, src) == "get" for c in a.children)]
            for a in gets:
                bodies.extend(c for c in a.children if c.type in ("arrow_expression_clause", "block"))
    if len(bodies) != 1:
        return None
    inner = [c for c in bodies[0].named_children if c.type != "comment"]
    if len(inner) == 1 and inner[0].type == "return_statement":
        inner = inner[0].named_children
    if len(inner) == 1 and inner[0].type == "boolean_literal":
        return _text(inner[0], src).strip() == "true"
    return None


def _auto_setter(prop) -> bool:
    """Whether a property has `set;` with no body: the compiler's backing field
    is then as writable as a field declared by hand."""
    for acc_list in prop.children:
        if acc_list.type != "accessor_list":
            continue
        for acc in acc_list.named_children:
            kinds = {c.type for c in acc.children}
            if acc.type == "accessor_declaration" and "set" in kinds and not kinds & {"block", "arrow_expression_clause"}:
                return True
    return False


def read_file(path: Path, rel: str) -> tuple[str, set, list]:
    """(namespace, using namespaces, type declarations) of one C# file."""
    from ..chunking import _get_parser
    src = path.read_bytes()
    tree = _get_parser("c_sharp").parse(src)
    namespace, usings, types = "", set(), []

    def visit(node, ns: str):
        # The type declarations an ERROR node holds in pieces count as declarations.
        for c in (x for x in cs_recovery.children(node) if x.is_named):
            t = c.type
            if t in ("file_scoped_namespace_declaration", "namespace_declaration"):
                n = c.child_by_field_name("name") or next((x for x in c.named_children if x.type in ("qualified_name", "identifier")), None)
                inner = _text(n, src) if n is not None else ns
                nonlocal namespace
                namespace = namespace or inner
                if t == "namespace_declaration":
                    body = next((x for x in c.children if x.type == "declaration_list"), None)
                    if body is not None:
                        visit(body, inner)
                else:
                    visit(c, inner)
            elif t == "using_directive":
                n = next((x for x in c.named_children if x.type in ("qualified_name", "identifier")), None)
                if n is not None and not any(x.type == "name_equals" for x in c.children):
                    usings.add(_text(n, src))
            elif t in TYPE_NODES:
                _read_type(c, src, rel, ns, "", types)
            elif t == "ERROR":
                visit(c, ns)

    visit(tree.root_node, "")
    # A file-scoped namespace is a sibling of the types that follow it.
    for t in types:
        t.namespace = t.namespace or namespace
    return namespace, usings, types
