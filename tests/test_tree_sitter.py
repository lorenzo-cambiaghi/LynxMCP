"""Unit-level tests for the chunker module (no embedding model, no ChromaDB).

Validates that `chunk_file` produces the expected boundaries / metadata for
each language we claim to support, plus the SentenceSplitter fallback for
unsupported types, plus the oversized-chunk split path.

Scenarios:
  1. Python — module header + function + class methods, qualified names
  2. C# — namespace.class.method qualification, multiple chunks per class
  3. TypeScript — function + class + method
  4. Markdown (.md) — fallback to SentenceSplitter, language="text"
  5. Shader file (.hlsl) — fallback (no AST parser available)
  6. Empty file — graceful, returns []  or one empty fallback chunk
  7. Huge function — split via SentenceSplitter into multiple sub-chunks
     with #partN suffixes
  8. Metadata invariants — every chunk has file_path / file_name /
     symbol_name / symbol_kind / language / chunker / start_line / end_line
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from lynx.chunking import chunk_file, MAX_CHUNK_CHARS

    # ============================================================
    # 1. Python
    # ============================================================
    py_src = '''"""Module docstring."""
import os
from typing import Optional

GLOBAL_X = 42

def greet(name: str) -> str:
    """Return a greeting."""
    return f"hello {name}"

class Greeter:
    """A greeter."""
    def __init__(self, prefix: str):
        self.prefix = prefix

    def hello(self, name: str) -> str:
        return f"{self.prefix} {name}"
'''
    chunks = chunk_file("demo.py", py_src)
    names = [c["symbol_name"] for c in chunks]
    if "greet" not in names:
        print(f"[test] FAIL [1/8]: 'greet' missing from {names}")
        return 1
    if "Greeter.__init__" not in names:
        print(f"[test] FAIL [1/8]: 'Greeter.__init__' missing from {names}")
        return 1
    if "Greeter.hello" not in names:
        print(f"[test] FAIL [1/8]: 'Greeter.hello' missing from {names}")
        return 1
    if not any(c.get("symbol_kind") == "module_header" for c in chunks):
        print(f"[test] FAIL [1/8]: no module_header chunk for Python imports")
        return 1
    print(f"[test] OK [1/8] Python: {len(chunks)} chunks, methods qualified as Greeter.X")

    # ============================================================
    # 2. C# — namespace.class.method qualification
    # ============================================================
    cs_src = """using System;

namespace MyFramework.Damage
{
    public interface IDamageable
    {
        void ApplyDamage(float amount);
    }

    public class HealthSystem : IDamageable
    {
        public void ApplyDamage(float amount) { }
        public void Heal(float amount) { }
    }
}
"""
    chunks = chunk_file("demo.cs", cs_src)
    names = [c["symbol_name"] for c in chunks]
    expected_qualified = "MyFramework.Damage.HealthSystem.ApplyDamage"
    if expected_qualified not in names:
        print(f"[test] FAIL [2/8]: {expected_qualified!r} missing from {names}")
        return 2
    if "MyFramework.Damage.HealthSystem.Heal" not in names:
        print(f"[test] FAIL [2/8]: 'HealthSystem.Heal' missing from {names}")
        return 2
    if "MyFramework.Damage.IDamageable.ApplyDamage" not in names:
        print(f"[test] FAIL [2/8]: 'IDamageable.ApplyDamage' missing from {names}")
        return 2
    print(f"[test] OK [2/8] C#: {len(chunks)} chunks with full namespace.class.method qualification")

    # ============================================================
    # 3. TypeScript — function + class + method
    # ============================================================
    ts_src = """import { Foo } from "./foo";

export function bar(): number {
    return 1;
}

export class MyClass {
    constructor(public name: string) {}

    greet(): string {
        return "hi " + this.name;
    }
}
"""
    chunks = chunk_file("demo.ts", ts_src)
    names = [c["symbol_name"] for c in chunks]
    if "bar" not in names:
        print(f"[test] FAIL [3/8]: 'bar' missing from {names}")
        return 3
    # MyClass.greet should be qualified
    if not any(n.endswith("greet") and "MyClass" in n for n in names):
        print(f"[test] FAIL [3/8]: 'MyClass.greet' missing from {names}")
        return 3
    print(f"[test] OK [3/8] TypeScript: {len(chunks)} chunks, class method qualified")

    # ============================================================
    # 4. Markdown — fallback to SentenceSplitter
    # ============================================================
    md_src = "# Title\n\nThis is a paragraph.\n\n## Section\n\nMore text.\n"
    chunks = chunk_file("readme.md", md_src)
    if not chunks:
        print(f"[test] FAIL [4/8]: markdown returned 0 chunks")
        return 4
    if chunks[0]["chunker"] != "sentence_splitter":
        print(f"[test] FAIL [4/8]: markdown not routed to SentenceSplitter (got {chunks[0]['chunker']!r})")
        return 4
    if chunks[0]["language"] != "text":
        print(f"[test] FAIL [4/8]: markdown language should be 'text', got {chunks[0]['language']!r}")
        return 4
    print(f"[test] OK [4/8] Markdown: fallback to SentenceSplitter, language='text'")

    # ============================================================
    # 5. HLSL shader file — fallback (no parser for .hlsl)
    # ============================================================
    hlsl_src = """float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv.x, uv.y, 0.0, 1.0);
}
"""
    chunks = chunk_file("shader.hlsl", hlsl_src)
    if not chunks:
        print(f"[test] FAIL [5/8]: shader returned 0 chunks")
        return 5
    if chunks[0]["chunker"] != "sentence_splitter":
        print(f"[test] FAIL [5/8]: HLSL should use fallback, got {chunks[0]['chunker']!r}")
        return 5
    print(f"[test] OK [5/8] HLSL: fallback used (no tree-sitter parser)")

    # ============================================================
    # 6. Empty file — must not crash
    # ============================================================
    chunks = chunk_file("empty.py", "")
    # Empty content can produce 0 chunks OR 1 trivial chunk; either is fine
    # as long as no exception is raised.
    print(f"[test] OK [6/8] empty file: returned {len(chunks)} chunks without crashing")

    # ============================================================
    # 7. Oversized function — split into multiple sub-chunks
    # ============================================================
    # Build a single huge Python function body that exceeds MAX_CHUNK_CHARS.
    body = "    x = 0\n" + ("    x += 1  # filler line that bloats the function body\n" * 200)
    big_src = f"def huge():\n{body}    return x\n"
    chunks = chunk_file("big.py", big_src)
    # Find chunks that come from `huge`
    huge_pieces = [c for c in chunks if c["symbol_name"].startswith("huge")]
    if len(huge_pieces) < 2:
        print(f"[test] FAIL [7/8]: oversized 'huge' should split into >=2 chunks, got {len(huge_pieces)} "
              f"(source size {len(big_src)} chars, MAX_CHUNK_CHARS={MAX_CHUNK_CHARS})")
        return 7
    # Each piece should have a #partN suffix
    suffixes = {c["symbol_name"] for c in huge_pieces if "#part" in c["symbol_name"]}
    if not suffixes:
        print(f"[test] FAIL [7/8]: split chunks missing #partN suffix; names: {[c['symbol_name'] for c in huge_pieces]}")
        return 7
    print(f"[test] OK [7/8] huge function split into {len(huge_pieces)} sub-chunks with #partN suffixes")

    # ============================================================
    # 8. Metadata invariants across every chunk produced so far
    # ============================================================
    # Re-chunk Python + C# + markdown, then check each chunk has the
    # required metadata keys.
    all_chunks = (
        chunk_file("demo.py", py_src)
        + chunk_file("demo.cs", cs_src)
        + chunk_file("readme.md", md_src)
    )
    required_keys = {
        "text", "symbol_name", "symbol_kind", "start_line", "end_line",
        "language", "chunker", "file_path", "file_name",
    }
    for i, c in enumerate(all_chunks):
        missing = required_keys - set(c.keys())
        if missing:
            print(f"[test] FAIL [8/8]: chunk {i} ({c.get('symbol_name', '?')}) missing keys: {missing}")
            return 8
        if not isinstance(c["start_line"], int) or not isinstance(c["end_line"], int):
            print(f"[test] FAIL [8/8]: chunk {i} has non-int line numbers: {c['start_line']!r}, {c['end_line']!r}")
            return 8
        if c["start_line"] < 1 or c["end_line"] < c["start_line"]:
            print(f"[test] FAIL [8/8]: chunk {i} has invalid line range L{c['start_line']}-L{c['end_line']}")
            return 8
    print(f"[test] OK [8/8] metadata invariants hold across {len(all_chunks)} chunks")

    print("\n[test] === SUCCESS: tree-sitter chunker works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
