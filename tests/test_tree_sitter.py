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
  9. Ruby — class.method + module.class.method qualification, singleton_method
 10. PHP — namespace.class.method qualification, free function
 11. Kotlin — class.method qualification, object.method, top-level function
 12. Swift — class.method qualification, protocol.method, init/deinit, free func
 13. parse_file helper — returns (tree, lang_key, rules) and is reused by
     chunk_file so a single file is only parsed once.
 14. Bash — shell functions (`foo() {}` and `function foo {}`)
 15. SQL — CREATE TABLE / FUNCTION / VIEW emitted as chunks
 16. Scala — object/class/trait container qualification of def/val
 17. Lua — function_declaration incl. `M.f` and `T:m` forms
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
        print(f"[test] FAIL [1/17]: 'greet' missing from {names}")
        return 1
    if "Greeter.__init__" not in names:
        print(f"[test] FAIL [1/17]: 'Greeter.__init__' missing from {names}")
        return 1
    if "Greeter.hello" not in names:
        print(f"[test] FAIL [1/17]: 'Greeter.hello' missing from {names}")
        return 1
    if not any(c.get("symbol_kind") == "module_header" for c in chunks):
        print(f"[test] FAIL [1/17]: no module_header chunk for Python imports")
        return 1
    print(f"[test] OK [1/17] Python: {len(chunks)} chunks, methods qualified as Greeter.X")

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
        print(f"[test] FAIL [2/17]: {expected_qualified!r} missing from {names}")
        return 2
    if "MyFramework.Damage.HealthSystem.Heal" not in names:
        print(f"[test] FAIL [2/17]: 'HealthSystem.Heal' missing from {names}")
        return 2
    if "MyFramework.Damage.IDamageable.ApplyDamage" not in names:
        print(f"[test] FAIL [2/17]: 'IDamageable.ApplyDamage' missing from {names}")
        return 2
    print(f"[test] OK [2/17] C#: {len(chunks)} chunks with full namespace.class.method qualification")

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
        print(f"[test] FAIL [3/17]: 'bar' missing from {names}")
        return 3
    # MyClass.greet should be qualified
    if not any(n.endswith("greet") and "MyClass" in n for n in names):
        print(f"[test] FAIL [3/17]: 'MyClass.greet' missing from {names}")
        return 3
    print(f"[test] OK [3/17] TypeScript: {len(chunks)} chunks, class method qualified")

    # ============================================================
    # 4. Markdown — fallback to SentenceSplitter
    # ============================================================
    md_src = "# Title\n\nThis is a paragraph.\n\n## Section\n\nMore text.\n"
    chunks = chunk_file("readme.md", md_src)
    if not chunks:
        print(f"[test] FAIL [4/17]: markdown returned 0 chunks")
        return 4
    if chunks[0]["chunker"] != "sentence_splitter":
        print(f"[test] FAIL [4/17]: markdown not routed to SentenceSplitter (got {chunks[0]['chunker']!r})")
        return 4
    if chunks[0]["language"] != "text":
        print(f"[test] FAIL [4/17]: markdown language should be 'text', got {chunks[0]['language']!r}")
        return 4
    print(f"[test] OK [4/17] Markdown: fallback to SentenceSplitter, language='text'")

    # ============================================================
    # 5. HLSL shader file — fallback (no parser for .hlsl)
    # ============================================================
    hlsl_src = """float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv.x, uv.y, 0.0, 1.0);
}
"""
    chunks = chunk_file("shader.hlsl", hlsl_src)
    if not chunks:
        print(f"[test] FAIL [5/17]: shader returned 0 chunks")
        return 5
    if chunks[0]["chunker"] != "sentence_splitter":
        print(f"[test] FAIL [5/17]: HLSL should use fallback, got {chunks[0]['chunker']!r}")
        return 5
    print(f"[test] OK [5/17] HLSL: fallback used (no tree-sitter parser)")

    # ============================================================
    # 6. Empty file — must not crash
    # ============================================================
    chunks = chunk_file("empty.py", "")
    # Empty content can produce 0 chunks OR 1 trivial chunk; either is fine
    # as long as no exception is raised.
    print(f"[test] OK [6/17] empty file: returned {len(chunks)} chunks without crashing")

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
        print(f"[test] FAIL [7/17]: oversized 'huge' should split into >=2 chunks, got {len(huge_pieces)} "
              f"(source size {len(big_src)} chars, MAX_CHUNK_CHARS={MAX_CHUNK_CHARS})")
        return 7
    # Each piece should have a #partN suffix
    suffixes = {c["symbol_name"] for c in huge_pieces if "#part" in c["symbol_name"]}
    if not suffixes:
        print(f"[test] FAIL [7/17]: split chunks missing #partN suffix; names: {[c['symbol_name'] for c in huge_pieces]}")
        return 7
    print(f"[test] OK [7/17] huge function split into {len(huge_pieces)} sub-chunks with #partN suffixes")

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
            print(f"[test] FAIL [8/17]: chunk {i} ({c.get('symbol_name', '?')}) missing keys: {missing}")
            return 8
        if not isinstance(c["start_line"], int) or not isinstance(c["end_line"], int):
            print(f"[test] FAIL [8/17]: chunk {i} has non-int line numbers: {c['start_line']!r}, {c['end_line']!r}")
            return 8
        if c["start_line"] < 1 or c["end_line"] < c["start_line"]:
            print(f"[test] FAIL [8/17]: chunk {i} has invalid line range L{c['start_line']}-L{c['end_line']}")
            return 8
    print(f"[test] OK [8/17] metadata invariants hold across {len(all_chunks)} chunks")

    # ============================================================
    # 9. Ruby — class.method + module.class.method + singleton_method
    # ============================================================
    rb_src = '''require "json"

module MyMod
  class Greeter
    def hello(name)
      "hi #{name}"
    end

    def self.factory
      Greeter.new
    end
  end
end
'''
    chunks = chunk_file("demo.rb", rb_src)
    names = [c["symbol_name"] for c in chunks]
    if "MyMod.Greeter.hello" not in names:
        print(f"[test] FAIL [9/17]: 'MyMod.Greeter.hello' missing from {names}")
        return 9
    if "MyMod.Greeter.factory" not in names:
        print(f"[test] FAIL [9/17]: 'MyMod.Greeter.factory' (singleton_method) missing from {names}")
        return 9
    if chunks[0]["language"] != "ruby":
        print(f"[test] FAIL [9/17]: ruby language tag wrong, got {chunks[0]['language']!r}")
        return 9
    print(f"[test] OK [9/17] Ruby: {len(chunks)} chunks, module.class.method qualified")

    # ============================================================
    # 10. PHP — namespace.class.method + free function
    # ============================================================
    php_src = '''<?php
namespace App\\Models {
    interface IRepo {
        public function find($id);
    }

    class UserRepo implements IRepo {
        public function find($id) { return null; }
        public function all() { return []; }
    }

    function helper() { return 1; }
}
'''
    chunks = chunk_file("demo.php", php_src)
    names = [c["symbol_name"] for c in chunks]
    if not any("UserRepo.find" in n for n in names):
        print(f"[test] FAIL [10/17]: 'UserRepo.find' missing from {names}")
        return 10
    if not any("UserRepo.all" in n for n in names):
        print(f"[test] FAIL [10/17]: 'UserRepo.all' missing from {names}")
        return 10
    if not any("IRepo.find" in n for n in names):
        print(f"[test] FAIL [10/17]: 'IRepo.find' missing from {names}")
        return 10
    if not any(n.endswith("helper") for n in names):
        print(f"[test] FAIL [10/17]: free function 'helper' missing from {names}")
        return 10
    print(f"[test] OK [10/17] PHP: {len(chunks)} chunks, namespace.class.method qualified")

    # ============================================================
    # 11. Kotlin — class.method + object.method + top-level function
    # ============================================================
    kt_src = '''package com.example

class Greeter(val prefix: String) {
    fun hello(name: String): String = "$prefix $name"
    fun bye(name: String): String = "bye $name"
}

object Util {
    fun ping(): String = "pong"
}

fun topLevel(): Int = 42
'''
    chunks = chunk_file("demo.kt", kt_src)
    names = [c["symbol_name"] for c in chunks]
    if "Greeter.hello" not in names:
        print(f"[test] FAIL [11/17]: 'Greeter.hello' missing from {names}")
        return 11
    if "Greeter.bye" not in names:
        print(f"[test] FAIL [11/17]: 'Greeter.bye' missing from {names}")
        return 11
    if "Util.ping" not in names:
        print(f"[test] FAIL [11/17]: 'Util.ping' missing from {names}")
        return 11
    if "topLevel" not in names:
        print(f"[test] FAIL [11/17]: 'topLevel' missing from {names}")
        return 11
    print(f"[test] OK [11/17] Kotlin: {len(chunks)} chunks, class/object/top-level qualified")

    # ============================================================
    # 12. Swift — class.method + protocol.method + init + free func
    # ============================================================
    sw_src = '''import Foundation

protocol IFoo {
    func bar() -> Int
}

class Greeter: IFoo {
    init() {}
    deinit {}
    func bar() -> Int { return 1 }
    func hello(name: String) -> String { return "hi \\(name)" }
}

func topLevel() -> String { return "hi" }
'''
    chunks = chunk_file("demo.swift", sw_src)
    names = [c["symbol_name"] for c in chunks]
    if "Greeter.bar" not in names:
        print(f"[test] FAIL [12/17]: 'Greeter.bar' missing from {names}")
        return 12
    if "Greeter.hello" not in names:
        print(f"[test] FAIL [12/17]: 'Greeter.hello' missing from {names}")
        return 12
    if not any(n.startswith("Greeter.init") or n == "Greeter.init" for n in names):
        print(f"[test] FAIL [12/17]: 'Greeter.init' missing from {names}")
        return 12
    if not any("IFoo" in n and "bar" in n for n in names):
        print(f"[test] FAIL [12/17]: protocol method 'IFoo.bar' missing from {names}")
        return 12
    if "topLevel" not in names:
        print(f"[test] FAIL [12/17]: free function 'topLevel' missing from {names}")
        return 12
    print(f"[test] OK [12/17] Swift: {len(chunks)} chunks, class/protocol/init/free func qualified")

    # ============================================================
    # 13. parse_file helper — must return the same tree chunk_file uses
    # ============================================================
    from lynx.chunking import parse_file

    res = parse_file("demo.py", py_src)
    if res is None:
        print(f"[test] FAIL [13/17]: parse_file returned None on valid Python source")
        return 13
    tree, lang_key, rules = res
    if lang_key != "python":
        print(f"[test] FAIL [13/17]: parse_file returned lang_key={lang_key!r}, expected 'python'")
        return 13
    if "container_types" not in rules or "chunk_types" not in rules:
        print(f"[test] FAIL [13/17]: rules dict from parse_file missing expected keys")
        return 13
    if tree.root_node.type != "module":
        print(f"[test] FAIL [13/17]: parse_file Python root is {tree.root_node.type!r}, expected 'module'")
        return 13
    # Unsupported extension must return None (not raise).
    if parse_file("readme.md", "# Hello\n") is not None:
        print(f"[test] FAIL [13/17]: parse_file should return None for .md, got non-None")
        return 13
    print(f"[test] OK [13/17] parse_file helper: returns (tree, lang_key, rules) and None for unsupported")

    # ============================================================
    # 14. Bash — shell functions, both syntaxes
    # ============================================================
    sh_src = 'greet() { echo "hi $1"; }\nfunction bye { echo bye; }\n'
    chunks = chunk_file("demo.sh", sh_src)
    names = [c["symbol_name"] for c in chunks]
    if "greet" not in names or "bye" not in names:
        print(f"[test] FAIL [14/17]: bash functions missing from {names}")
        return 14
    if chunks[0]["language"] != "bash":
        print(f"[test] FAIL [14/17]: bash language tag wrong, got {chunks[0]['language']!r}")
        return 14
    print(f"[test] OK [14/17] Bash: {len(chunks)} chunks (greet, bye)")

    # ============================================================
    # 15. SQL — CREATE TABLE / FUNCTION / VIEW as chunks
    # ============================================================
    sql_src = (
        "CREATE TABLE users (id INT, name TEXT);\n"
        "CREATE FUNCTION add(a INT) RETURNS INT AS 'SELECT a' LANGUAGE sql;\n"
        "CREATE VIEW v AS SELECT * FROM users;\n"
    )
    chunks = chunk_file("demo.sql", sql_src)
    names = [c["symbol_name"] for c in chunks]
    for want in ("users", "add", "v"):
        if want not in names:
            print(f"[test] FAIL [15/17]: SQL object {want!r} missing from {names}")
            return 15
    if chunks[0]["language"] != "sql":
        print(f"[test] FAIL [15/17]: sql language tag wrong, got {chunks[0]['language']!r}")
        return 15
    print(f"[test] OK [15/17] SQL: {len(chunks)} chunks (table/function/view)")

    # ============================================================
    # 16. Scala — object/class/trait qualification of def/val
    # ============================================================
    scala_src = (
        'object Main { def main(a: Array[String]): Unit = println("hi"); val x = 1 }\n'
        "class Foo(n: Int) { def bar(): Int = n }\n"
        "trait T { def f(): Unit }\n"
    )
    chunks = chunk_file("demo.scala", scala_src)
    names = [c["symbol_name"] for c in chunks]
    for want in ("Main.main", "Foo.bar", "T.f"):
        if want not in names:
            print(f"[test] FAIL [16/17]: Scala symbol {want!r} missing from {names}")
            return 16
    print(f"[test] OK [16/17] Scala: {len(chunks)} chunks, container.method qualified")

    # ============================================================
    # 17. Lua — function_declaration incl. M.f and T:m forms
    # ============================================================
    lua_src = (
        "local function add(a, b) return a + b end\n"
        "function M.greet(name) return name end\n"
        "function T:m() return 1 end\n"
    )
    chunks = chunk_file("demo.lua", lua_src)
    names = [c["symbol_name"] for c in chunks]
    for want in ("add", "M.greet", "T:m"):
        if want not in names:
            print(f"[test] FAIL [17/17]: Lua symbol {want!r} missing from {names}")
            return 17
    print(f"[test] OK [17/17] Lua: {len(chunks)} chunks (add, M.greet, T:m)")

    print("\n[test] === SUCCESS: tree-sitter chunker works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
