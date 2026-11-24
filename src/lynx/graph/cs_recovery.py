"""C# type declarations a grammar left in pieces, put back together.

When tree-sitter cannot parse part of a C# file it keeps an ERROR node, and
that node sometimes holds a whole type as loose tokens: the modifiers,
`class`, the name, the base list, `{`, the members, `}`. It happens to a file
while a statement is half written, and a grammar release can do it to a valid
file (tree-sitter-c-sharp 0.24.0 did it to a 2,400-line class over a local
variable named `partial`). The type and everything it owned would leave the
graph and the audit.

`children(node)` is `node.children` with each such run replaced by a view of
the declaration it was: an object with the parts of a tree-sitter node the
graph extractor and the audit read (`type`, `children`, `named_children`,
`child_by_field_name`, the start and end). Its members are the real nodes
the grammar did parse. A run is recovered only when it is unambiguous: a type
keyword, a name, then only what a declaration head holds before the opening
brace. Without its closing brace the type runs to the end of the node.
"""
from __future__ import annotations

_DECLARATIONS = {
    "class": "class_declaration",
    "struct": "struct_declaration",
    "interface": "interface_declaration",
    "record": "record_declaration",
}
# What may stand before the keyword, and between the name and the opening brace.
_LEAD = frozenset({"modifier", "attribute_list"})
_HEAD = frozenset({"type_parameter_list", "parameter_list", "base_list", "type_parameter_constraints_clause"})


class RecoveredNode:
    """A declaration rebuilt from the loose children of an ERROR node."""

    is_named = True
    is_missing = False
    has_error = True

    def __init__(self, type_: str, children: list):
        self.type = type_
        self.children = children
        self.named_children = [c for c in children if c.is_named]
        self.start_point = children[0].start_point
        self.end_point = children[-1].end_point
        self.start_byte = children[0].start_byte
        self.end_byte = children[-1].end_byte

    @property
    def child_count(self) -> int:
        return len(self.children)

    def child_by_field_name(self, name: str):
        if name == "name":
            return next((c for c in self.children if c.type == "identifier"), None)
        if name == "body":
            return next((c for c in self.children if c.type == "declaration_list"), None)
        if name == "bases":
            return next((c for c in self.children if c.type == "base_list"), None)
        return None


def children(node) -> list:
    """`node.children`; for an ERROR node, with the type declarations it
    holds in pieces put back together."""
    kids = node.children
    if node.type != "ERROR":
        return kids
    out, i = [], 0
    while i < len(kids):
        decl = _DECLARATIONS.get(kids[i].type)
        if decl is not None:
            name_at = i + 1
            # `record class X` and `record struct X`
            if kids[i].type == "record" and name_at < len(kids) and kids[name_at].type in ("class", "struct"):
                name_at += 1
            if name_at < len(kids) and kids[name_at].type == "identifier":
                j = name_at + 1
                while j < len(kids) and kids[j].type in _HEAD:
                    j += 1
                if j < len(kids) and kids[j].type == "{":
                    depth, k = 0, j
                    while k < len(kids):
                        if kids[k].type == "{":
                            depth += 1
                        elif kids[k].type == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1
                    k = min(k, len(kids) - 1)
                    lead = len(out)
                    while lead > 0 and out[lead - 1].type in _LEAD:
                        lead -= 1
                    head = out[lead:]
                    del out[lead:]
                    body = RecoveredNode("declaration_list", kids[j:k + 1])
                    out.append(RecoveredNode(decl, head + kids[i:j] + [body]))
                    i = k + 1
                    continue
        out.append(kids[i])
        i += 1
    return out
