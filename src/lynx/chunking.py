"""AST-aware chunking via tree-sitter, with SentenceSplitter fallback.

Goal: produce chunks aligned with the *structure* of the code — one chunk per
function / method / class — instead of arbitrary token windows that cut across
syntactic boundaries. Embeddings of structurally-coherent units score
meaningfully on similarity searches; embeddings of half-functions don't.

Public entry point: `chunk_file(abs_path, content) -> list[Chunk]`.
A `Chunk` is a small dict with `text` + metadata (symbol name, kind, line
range, language, chunker). The caller converts it into a LlamaIndex TextNode
in the indexing pipeline.

For unsupported file types (markdown, plain text, shader languages, etc.)
we fall back to the existing SentenceSplitter — same behavior as v0.2.

Bumping `CHUNKER_VERSION` invalidates all stored chunks across every source
(it's part of the drift / SHA-cache snapshot), forcing a clean rebuild on
next start. Do this whenever the chunker logic changes in a way that makes
existing chunks incompatible with new ones (boundaries shift, metadata
changes, etc.).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import tree_sitter
import tree_sitter_c as ts_c
import tree_sitter_c_sharp as ts_cs
import tree_sitter_cpp as ts_cpp
import tree_sitter_go as ts_go
import tree_sitter_java as ts_java
import tree_sitter_javascript as ts_js
import tree_sitter_python as ts_py
import tree_sitter_rust as ts_rs
import tree_sitter_typescript as ts_ts


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Bumping invalidates every cached chunk (existing indexes are flagged with a
# CRITICAL drift and the user is prompted to rebuild). Bump whenever the
# chunking logic changes the boundaries / metadata in a backwards-incompatible
# way.
CHUNKER_VERSION = 3
# History:
#   1 — SentenceSplitter only (pre-M3 era)
#   2 — tree-sitter + fallback, naive _extract_name (took first identifier)
#   3 — _extract_name now uses child_by_field_name("name") so C# method
#       declarations correctly report the method name (not the return type).

# Soft cap on chunk size. Anything bigger gets split with SentenceSplitter so
# huge auto-generated methods don't produce a single 50k-token chunk. ~8000
# chars ≈ ~2000 tokens, which fits comfortably in the embedding model's
# context window (BGE-small is 512 but it truncates fine for our purposes).
MAX_CHUNK_CHARS = 8000

# Defaults for the SentenceSplitter fallback (and for over-sized AST chunks).
FALLBACK_CHUNK_SIZE = 1024     # tokens
FALLBACK_CHUNK_OVERLAP = 100   # tokens


# ---------------------------------------------------------------------------
# Per-language rules
# ---------------------------------------------------------------------------
# For each language we declare:
#   - parser_factory: () → tree_sitter.Language
#   - container_types: AST node types we recurse INTO (class, namespace, ...)
#     The container itself does NOT become a chunk; its inner chunks do.
#   - chunk_types: AST node types we emit AS chunks (function, method, ...)
#     We do not recurse into a chunk node — a method inside a method (nested
#     functions in JS for example) becomes part of the outer chunk.
#   - name_node_types: ordered list of child node types to try when extracting
#     a symbol name from a container or chunk node.


def _ts_language(factory: Callable[[], object]) -> tree_sitter.Language:
    """Wrap a per-language module's `language()` call into a tree_sitter.Language."""
    return tree_sitter.Language(factory())


_LANG_RULES: dict = {
    "c_sharp": {
        "parser_factory": lambda: _ts_language(ts_cs.language),
        "container_types": {
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "record_declaration",
            "namespace_declaration",
            "file_scoped_namespace_declaration",
        },
        "chunk_types": {
            "method_declaration",
            "constructor_declaration",
            "destructor_declaration",
            "property_declaration",
            "indexer_declaration",
            "operator_declaration",
            "delegate_declaration",
            "event_declaration",
            "enum_declaration",
        },
        # `qualified_name` covers C# namespaces like `MyFramework.Damage`.
        "name_node_types": ["identifier", "type_identifier", "qualified_name"],
    },
    "python": {
        "parser_factory": lambda: _ts_language(ts_py.language),
        "container_types": {"class_definition"},
        "chunk_types": {"function_definition", "decorated_definition"},
        "name_node_types": ["identifier"],
    },
    "typescript": {
        "parser_factory": lambda: _ts_language(ts_ts.language_typescript),
        "container_types": {
            "class_declaration",
            "interface_declaration",
            "namespace_declaration",
            "module",
            "internal_module",
        },
        "chunk_types": {
            "function_declaration",
            "method_definition",
            "abstract_method_signature",
            "type_alias_declaration",
            "enum_declaration",
        },
        "name_node_types": ["identifier", "type_identifier", "property_identifier"],
    },
    "tsx": {
        "parser_factory": lambda: _ts_language(ts_ts.language_tsx),
        "container_types": {
            "class_declaration",
            "interface_declaration",
            "namespace_declaration",
        },
        "chunk_types": {
            "function_declaration",
            "method_definition",
            "type_alias_declaration",
            "enum_declaration",
        },
        "name_node_types": ["identifier", "type_identifier", "property_identifier"],
    },
    "javascript": {
        "parser_factory": lambda: _ts_language(ts_js.language),
        "container_types": {"class_declaration"},
        "chunk_types": {
            "function_declaration",
            "method_definition",
            "generator_function_declaration",
        },
        "name_node_types": ["identifier", "property_identifier"],
    },
    "cpp": {
        "parser_factory": lambda: _ts_language(ts_cpp.language),
        "container_types": {
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        },
        "chunk_types": {
            "function_definition",
            "template_declaration",
        },
        "name_node_types": ["identifier", "type_identifier", "field_identifier"],
    },
    "c": {
        "parser_factory": lambda: _ts_language(ts_c.language),
        "container_types": set(),  # C has no nesting that we care about
        "chunk_types": {
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "type_definition",
        },
        "name_node_types": ["identifier", "type_identifier", "field_identifier"],
    },
    "go": {
        "parser_factory": lambda: _ts_language(ts_go.language),
        "container_types": set(),
        "chunk_types": {
            "function_declaration",
            "method_declaration",
            "type_declaration",
        },
        "name_node_types": ["identifier", "type_identifier", "field_identifier"],
    },
    "rust": {
        "parser_factory": lambda: _ts_language(ts_rs.language),
        "container_types": {
            "impl_item",
            "mod_item",
            "trait_item",
        },
        "chunk_types": {
            "function_item",
            "struct_item",
            "enum_item",
            "type_item",
        },
        "name_node_types": ["identifier", "type_identifier", "field_identifier"],
    },
    "java": {
        "parser_factory": lambda: _ts_language(ts_java.language),
        "container_types": {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        },
        "chunk_types": {
            "method_declaration",
            "constructor_declaration",
        },
        "name_node_types": ["identifier", "type_identifier"],
    },
}


# Map file extensions to language keys above.
_EXT_TO_LANG: dict = {
    ".cs":   "c_sharp",
    ".py":   "python",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cpp":  "cpp",
    ".cxx":  "cpp",
    ".cc":   "cpp",
    ".hpp":  "cpp",
    ".hxx":  "cpp",
    ".c":    "c",
    ".h":    "c",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
}


# Lazy parser cache — one Parser per language built on first use, then reused.
_PARSER_CACHE: dict = {}


def _get_parser(lang_key: str) -> tree_sitter.Parser:
    if lang_key not in _PARSER_CACHE:
        rules = _LANG_RULES[lang_key]
        _PARSER_CACHE[lang_key] = tree_sitter.Parser(rules["parser_factory"]())
    return _PARSER_CACHE[lang_key]


def language_for_path(abs_path: str) -> Optional[str]:
    """Return the language key for this path, or None if no AST parser available."""
    return _EXT_TO_LANG.get(Path(abs_path).suffix.lower())


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------


def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_name(node, name_node_types: list) -> Optional[str]:
    """Find the symbol name for a node.

    Strategy:
      1. Try `node.child_by_field_name("name")` — works for most tree-sitter
         grammars where the declaration explicitly tags its name field.
         CRITICAL for C# `method_declaration`, which has two `identifier`
         children (return type AND name); without this we'd pick the return
         type and label every method by its return type instead of its name.
      2. Fall back to the first immediate child whose type is in
         `name_node_types` — for grammars without a `name` field or for
         odd cases where the field is absent (e.g. anonymous functions).
    """
    named = node.child_by_field_name("name")
    if named is not None:
        try:
            return named.text.decode("utf-8", errors="replace")
        except Exception:
            pass

    for child in node.children:
        if child.type in name_node_types:
            try:
                return child.text.decode("utf-8", errors="replace")
            except Exception:
                return None
    return None


def _walk(node, source: bytes, rules: dict, container_path: list) -> list:
    """Recursively walk `node`, returning a list of chunk dicts.

    The traversal rules:
      - When we hit a chunk_types node: emit it as a chunk (do NOT recurse
        into it — nested chunks would over-fragment things like JS closures).
      - When we hit a container_types node: do NOT emit the whole container;
        recurse into it, prefixing the container's name to nested chunk names.
      - Anything else: keep walking down (covers wrapper nodes like
        decorated_definition, modifiers, etc.).
    """
    chunks = []
    container_types = rules["container_types"]
    chunk_types = rules["chunk_types"]
    name_node_types = rules["name_node_types"]

    for child in node.children:
        if child.type in chunk_types:
            name = _extract_name(child, name_node_types) or "<anonymous>"
            full_name = ".".join(container_path + [name]) if container_path else name
            chunks.append({
                "text": _node_text(source, child),
                "symbol_name": full_name,
                "symbol_kind": child.type,
                "start_line": child.start_point[0] + 1,
                "end_line": child.end_point[0] + 1,
            })
        elif child.type in container_types:
            container_name = _extract_name(child, name_node_types) or "<container>"
            inner = _walk(child, source, rules, container_path + [container_name])
            if inner:
                chunks.extend(inner)
            else:
                # Empty / abstract container with no chunkable members — emit
                # the container itself so its declaration isn't lost.
                chunks.append({
                    "text": _node_text(source, child),
                    "symbol_name": ".".join(container_path + [container_name]),
                    "symbol_kind": child.type,
                    "start_line": child.start_point[0] + 1,
                    "end_line": child.end_point[0] + 1,
                })
        else:
            # Generic descent for wrappers like `decorated_definition`, which
            # in Python wraps a function_definition we want to capture.
            chunks.extend(_walk(child, source, rules, container_path))
    return chunks


def _collect_header_text(root, source: bytes, rules: dict) -> str:
    """Concatenate top-level statements that aren't chunk-emitting or
    container-traversing nodes (imports, using statements, top-level vars).

    Treated as one synthetic "module header" chunk — gives us a single place
    to look up imports / `using` clauses, which are essential context for
    interpreting any method we retrieve from the file later.
    """
    skip = rules["container_types"] | rules["chunk_types"]
    parts = []
    for child in root.children:
        if child.type in skip:
            continue
        # Skip comments — they're attached to whatever they precede in our
        # rendering, no need to make them a standalone chunk.
        if child.type in ("comment", "block_comment", "line_comment"):
            continue
        text = _node_text(source, child).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point + fallback
# ---------------------------------------------------------------------------


def _split_oversized(chunk: dict, max_chars: int) -> list:
    """If a chunk is bigger than max_chars, split it with SentenceSplitter.

    Each sub-chunk inherits the parent's metadata but its symbol_name gets a
    `#partN` suffix so they're distinguishable in retrieval results.
    """
    if len(chunk["text"]) <= max_chars:
        return [chunk]
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=max_chars // 4,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(chunk["text"])
    base_name = chunk.get("symbol_name") or "<chunk>"
    return [
        {
            **chunk,
            "text": piece,
            "symbol_name": f"{base_name}#part{i + 1}",
        }
        for i, piece in enumerate(pieces)
    ]


def _fallback_chunks(abs_path: str, content: str) -> list:
    """Chunk using llama-index's SentenceSplitter — for file types without a
    tree-sitter grammar (markdown, txt, shader languages, JSON, etc.)."""
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(content)
    out = []
    line_cursor = 1
    for i, text in enumerate(pieces):
        line_count = text.count("\n")
        out.append({
            "text": text,
            "symbol_name": f"<chunk{i + 1}>",
            "symbol_kind": "text_window",
            "start_line": line_cursor,
            "end_line": line_cursor + line_count,
        })
        line_cursor += line_count
    return out


def chunk_file(abs_path: str, content: str) -> list:
    """Chunk one file. Returns a list of chunk dicts ready to embed.

    Each chunk has:
      text         — the chunk content
      symbol_name  — fully-qualified name (e.g. "MyClass.handleClick")
      symbol_kind  — AST node type ("method_declaration", ...) or "text_window"
      start_line   — 1-based inclusive
      end_line     — 1-based inclusive
      language     — language key ("c_sharp", "python", ...) or "text"
      chunker      — "tree_sitter" or "sentence_splitter"
      file_path    — propagated from caller (set in this function)
      file_name    — propagated from caller (set in this function)

    Pipeline:
      1. Try tree-sitter for the file's extension.
      2. If unsupported, fall back to SentenceSplitter (current v0.2 behavior).
      3. Split any oversized chunks with SentenceSplitter so no single chunk
         exceeds MAX_CHUNK_CHARS.
      4. Annotate every chunk with file_path / file_name / language / chunker.
    """
    lang_key = language_for_path(abs_path)
    abs_path_norm = os.path.normpath(os.path.abspath(abs_path))
    file_name = os.path.basename(abs_path_norm)

    if lang_key is None:
        chunks = _fallback_chunks(abs_path, content)
        for c in chunks:
            c["language"] = "text"
            c["chunker"] = "sentence_splitter"
            c["file_path"] = abs_path_norm
            c["file_name"] = file_name
        return chunks

    # Tree-sitter path.
    parser = _get_parser(lang_key)
    rules = _LANG_RULES[lang_key]
    source = content.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception:
        # Parse failure (rare — tree-sitter recovers from most syntax errors).
        # Fall back rather than dropping the file entirely.
        chunks = _fallback_chunks(abs_path, content)
        for c in chunks:
            c["language"] = lang_key  # we still know what it was supposed to be
            c["chunker"] = "sentence_splitter"
            c["file_path"] = abs_path_norm
            c["file_name"] = file_name
        return chunks

    chunks = _walk(tree.root_node, source, rules, container_path=[])

    # Header chunk: imports / using / top-level statements.
    header_text = _collect_header_text(tree.root_node, source, rules)
    if header_text:
        chunks.insert(0, {
            "text": header_text,
            "symbol_name": "<header>",
            "symbol_kind": "module_header",
            "start_line": 1,
            "end_line": header_text.count("\n") + 1,
        })

    # If tree-sitter found nothing chunkable (very small or unusual file),
    # treat the whole file as one chunk via the fallback.
    if not chunks:
        chunks = _fallback_chunks(abs_path, content)
        for c in chunks:
            c["language"] = lang_key
            c["chunker"] = "sentence_splitter"
            c["file_path"] = abs_path_norm
            c["file_name"] = file_name
        return chunks

    # Split oversized AST chunks.
    sized = []
    for c in chunks:
        sized.extend(_split_oversized(c, MAX_CHUNK_CHARS))

    for c in sized:
        c["language"] = lang_key
        c["chunker"] = "tree_sitter"
        c["file_path"] = abs_path_norm
        c["file_name"] = file_name

    return sized
