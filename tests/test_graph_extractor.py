"""Unit tests for the single-file graph extractor.

Validates that `extract_file` produces correct nodes/edges/raw_calls for
each language family the graph layer supports. No filesystem, no parser
init beyond what `parse_file` provides — pure function tests.

Scenarios:
  1. Python — class.method + free function + import + intra-file call
  2. C# — namespace.class.method, member call resolution, using directive
  3. Go — function + method + selector_expression call (pkg.Func)
  4. Rust — function_item + method_call_expression + use_declaration
  5. Ruby — class.method + singleton_method + require edge
  6. PHP — namespace.class.method + namespace_use + scoped/member calls
  7. Kotlin — class.method + import + intra-file call
  8. Swift — class.method + protocol.method + import + intra-file call
  9. Cross-file unresolved — raw_calls populated when callee is undefined
 10. Unsupported file — extract_file returns None for markdown
 11. Inheritance — raw_inherits (extends/implements) across C#/Python/TS/Java
 12. Scala — functions, calls, extends/with inheritance, import
 13. Lua — functions + intra-file call graph
 14. Bash — functions + call graph (commands resolving to functions)
 15. Objective-C — methods, message-send calls, #import, super/protocol inherit
 16. No self-loop call edges — member-name collision, overload-to-overload
     call, and recursion all collapse to `X calls X` and must be dropped
"""
from __future__ import annotations

import sys


def _ids_by_label(nodes):
    """Map label -> id for assertion convenience."""
    return {n["label"]: n["id"] for n in nodes}


def _edges(result, relation: str):
    return [e for e in result["edges"] if e["relation"] == relation]


def main() -> int:
    from lynx.graph import extract_file

    # ============================================================
    # 1. Python
    # ============================================================
    py_src = '''import os
from typing import Optional

def helper(x):
    return x + 1

class Greeter:
    def __init__(self, prefix):
        self.prefix = prefix

    def hello(self, name):
        # intra-file call: helper is defined in this file
        return helper(name)
'''
    r = extract_file("demo.py", py_src)
    if r is None:
        print("[test] FAIL [1/15]: extract_file returned None on demo.py")
        return 1
    labels = _ids_by_label(r["nodes"])
    if "helper" not in labels:
        print(f"[test] FAIL [1/15]: 'helper' missing from nodes: {list(labels)}")
        return 1
    if "Greeter.hello" not in labels:
        print(f"[test] FAIL [1/15]: 'Greeter.hello' missing from nodes: {list(labels)}")
        return 1
    # Containment: Greeter -> Greeter.hello
    contains = _edges(r, "contains")
    pair = {(e["source"], e["target"]) for e in contains}
    if (labels["Greeter"], labels["Greeter.hello"]) not in pair:
        print(f"[test] FAIL [1/15]: Greeter -> Greeter.hello contains edge missing")
        return 1
    # Intra-file call: Greeter.hello -> helper, RESOLVED in same pass
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["Greeter.hello"] and e["target"] == labels["helper"] for e in calls):
        print(f"[test] FAIL [1/15]: Greeter.hello -> helper call not resolved: calls={calls}")
        return 1
    # Imports: at least one edge from the file to an external module
    imports = _edges(r, "imports") + _edges(r, "imports_from")
    if not imports:
        print(f"[test] FAIL [1/15]: no import edges emitted")
        return 1
    print(f"[test] OK [1/15] Python: {len(r['nodes'])} nodes, {len(calls)} calls, {len(imports)} imports")

    # ============================================================
    # 2. C# — namespace.class.method qualification + member call
    # ============================================================
    cs_src = """using System;

namespace MyApp.Damage
{
    public class HealthSystem
    {
        public void ApplyDamage(float amount) { }
        public void Heal(float amount)
        {
            // intra-file call to ApplyDamage
            ApplyDamage(amount);
        }
    }
}
"""
    r = extract_file("demo.cs", cs_src)
    if r is None:
        print("[test] FAIL [2/15]: extract_file returned None for demo.cs")
        return 2
    labels = _ids_by_label(r["nodes"])
    if "MyApp.Damage.HealthSystem.ApplyDamage" not in labels:
        print(f"[test] FAIL [2/15]: full qualified label missing from {list(labels)}")
        return 2
    if "MyApp.Damage.HealthSystem.Heal" not in labels:
        print(f"[test] FAIL [2/15]: Heal missing from {list(labels)}")
        return 2
    calls = _edges(r, "calls")
    heal = labels["MyApp.Damage.HealthSystem.Heal"]
    apply = labels["MyApp.Damage.HealthSystem.ApplyDamage"]
    if not any(e["source"] == heal and e["target"] == apply for e in calls):
        print(f"[test] FAIL [2/15]: Heal -> ApplyDamage call missing: {calls}")
        return 2
    imports = _edges(r, "imports")
    if not imports:
        print(f"[test] FAIL [2/15]: using System; produced no import edge")
        return 2
    print(f"[test] OK [2/15] C#: {len(calls)} calls resolved, {len(imports)} using directives")

    # ============================================================
    # 3. Go — function + method + selector_expression call
    # ============================================================
    go_src = """package main

import "fmt"

func helper(x int) int {
    return x + 1
}

type T struct{ x int }

func (t *T) Run() {
    helper(t.x)
    fmt.Println(t.x)
}
"""
    r = extract_file("main.go", go_src)
    if r is None:
        print("[test] FAIL [3/15]: extract_file returned None for main.go")
        return 3
    labels = _ids_by_label(r["nodes"])
    if "helper" not in labels:
        print(f"[test] FAIL [3/15]: 'helper' missing from {list(labels)}")
        return 3
    if "Run" not in labels:
        print(f"[test] FAIL [3/15]: 'Run' method missing from {list(labels)}")
        return 3
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["Run"] and e["target"] == labels["helper"] for e in calls):
        print(f"[test] FAIL [3/15]: Run -> helper call not resolved: {calls}")
        return 3
    # fmt.Println should appear as raw_call (Println not defined locally)
    raw_names = {rc["callee_name"] for rc in r["raw_calls"]}
    if "Println" not in raw_names:
        print(f"[test] FAIL [3/15]: 'Println' (member call) not in raw_calls: {raw_names}")
        return 3
    imports = _edges(r, "imports")
    if not any('"fmt"' in (e.get("module") or "") or e.get("module") == "fmt" for e in imports):
        print(f"[test] FAIL [3/15]: 'fmt' import edge missing: {imports}")
        return 3
    print(f"[test] OK [3/15] Go: {len(calls)} resolved + {len(r['raw_calls'])} raw_calls + {len(imports)} imports")

    # ============================================================
    # 4. Rust — function_item + method_call_expression + use_declaration
    # ============================================================
    rs_src = """use std::io::Write;

fn helper(x: i32) -> i32 { x + 1 }

fn main() {
    let _ = helper(1);
    let s = String::from("hi");
    s.len();
}
"""
    r = extract_file("main.rs", rs_src)
    if r is None:
        print("[test] FAIL [4/15]: extract_file returned None for main.rs")
        return 4
    labels = _ids_by_label(r["nodes"])
    if "helper" not in labels or "main" not in labels:
        print(f"[test] FAIL [4/15]: helper/main missing from {list(labels)}")
        return 4
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["main"] and e["target"] == labels["helper"] for e in calls):
        print(f"[test] FAIL [4/15]: main -> helper call not resolved: {calls}")
        return 4
    # `s.len()` (method_call_expression) should produce a raw_call to "len"
    raw_names = {rc["callee_name"] for rc in r["raw_calls"]}
    if "len" not in raw_names:
        print(f"[test] FAIL [4/15]: 'len' (method_call) missing from raw_calls: {raw_names}")
        return 4
    imports = _edges(r, "imports")
    if not imports:
        print(f"[test] FAIL [4/15]: use std::io::Write produced no import")
        return 4
    print(f"[test] OK [4/15] Rust: {len(calls)} resolved + {len(r['raw_calls'])} raw + {len(imports)} imports")

    # ============================================================
    # 5. Ruby — class.method, singleton_method, require
    # ============================================================
    rb_src = '''require "json"

class Greeter
  def hello(name)
    say(name)
  end

  def say(name)
    puts name
  end

  def self.factory
    Greeter.new
  end
end
'''
    r = extract_file("demo.rb", rb_src)
    if r is None:
        print("[test] FAIL [5/15]: extract_file returned None for demo.rb")
        return 5
    labels = _ids_by_label(r["nodes"])
    if "Greeter.hello" not in labels or "Greeter.say" not in labels:
        print(f"[test] FAIL [5/15]: Greeter.{{hello,say}} missing from {list(labels)}")
        return 5
    if "Greeter.factory" not in labels:
        print(f"[test] FAIL [5/15]: singleton_method 'Greeter.factory' missing")
        return 5
    calls = _edges(r, "calls")
    hello = labels["Greeter.hello"]
    say = labels["Greeter.say"]
    if not any(e["source"] == hello and e["target"] == say for e in calls):
        print(f"[test] FAIL [5/15]: Greeter.hello -> Greeter.say call missing: {calls}")
        return 5
    imports = _edges(r, "imports")
    if not any(e.get("module") == "json" for e in imports):
        print(f"[test] FAIL [5/15]: require 'json' produced no import edge: {imports}")
        return 5
    print(f"[test] OK [5/15] Ruby: {len(calls)} calls, {len(imports)} requires, singleton_method handled")

    # ============================================================
    # 6. PHP — namespace.class.method + use + scoped/member call
    # ============================================================
    php_src = """<?php
namespace App\\Repo {
    use App\\Helper;

    class UserRepo {
        public function helper() { return 1; }
        public function find($id) {
            $this->helper();
            return Helper::call();
        }
    }
}
"""
    r = extract_file("demo.php", php_src)
    if r is None:
        print("[test] FAIL [6/15]: extract_file returned None for demo.php")
        return 6
    labels = _ids_by_label(r["nodes"])
    # Method qualification: must include namespace + class
    helper_label = next((l for l in labels if l.endswith("UserRepo.helper")), None)
    find_label = next((l for l in labels if l.endswith("UserRepo.find")), None)
    if helper_label is None or find_label is None:
        print(f"[test] FAIL [6/15]: UserRepo.{{helper,find}} missing from {list(labels)}")
        return 6
    calls = _edges(r, "calls")
    helper_id, find_id = labels[helper_label], labels[find_label]
    if not any(e["source"] == find_id and e["target"] == helper_id for e in calls):
        print(f"[test] FAIL [6/15]: find -> helper call (via $this->helper()) missing: {calls}")
        return 6
    # Helper::call() — scoped call to external class, must be in raw_calls
    raw_names = {rc["callee_name"] for rc in r["raw_calls"]}
    if "call" not in raw_names:
        print(f"[test] FAIL [6/15]: scoped call 'Helper::call' not in raw_calls: {raw_names}")
        return 6
    imports = _edges(r, "imports")
    if not any("Helper" in (e.get("module") or "") for e in imports):
        print(f"[test] FAIL [6/15]: 'use App\\Helper' import missing: {imports}")
        return 6
    print(f"[test] OK [6/15] PHP: {len(calls)} resolved + {len(r['raw_calls'])} raw + {len(imports)} use")

    # ============================================================
    # 7. Kotlin — class.method + intra-file call
    # ============================================================
    kt_src = """package com.example
import kotlin.collections.List

class Greeter {
    fun helper(): Int = 1
    fun hello(): Int {
        return helper()
    }
}
"""
    r = extract_file("demo.kt", kt_src)
    if r is None:
        print("[test] FAIL [7/15]: extract_file returned None for demo.kt")
        return 7
    labels = _ids_by_label(r["nodes"])
    if "Greeter.helper" not in labels or "Greeter.hello" not in labels:
        print(f"[test] FAIL [7/15]: Greeter.{{helper,hello}} missing from {list(labels)}")
        return 7
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["Greeter.hello"] and e["target"] == labels["Greeter.helper"] for e in calls):
        print(f"[test] FAIL [7/15]: Greeter.hello -> Greeter.helper call missing: {calls}")
        return 7
    imports = _edges(r, "imports")
    if not imports:
        print(f"[test] FAIL [7/15]: kotlin import produced no edges")
        return 7
    print(f"[test] OK [7/15] Kotlin: {len(calls)} calls, {len(imports)} imports")

    # ============================================================
    # 8. Swift — class.method + protocol.method + intra-file call
    # ============================================================
    sw_src = """import Foundation

protocol IFoo {
    func bar() -> Int
}

class Greeter: IFoo {
    func helper() -> Int { return 1 }
    func bar() -> Int {
        return helper()
    }
}
"""
    r = extract_file("demo.swift", sw_src)
    if r is None:
        print("[test] FAIL [8/15]: extract_file returned None for demo.swift")
        return 8
    labels = _ids_by_label(r["nodes"])
    if "Greeter.helper" not in labels or "Greeter.bar" not in labels:
        print(f"[test] FAIL [8/15]: Greeter.{{helper,bar}} missing from {list(labels)}")
        return 8
    if not any(l.endswith("IFoo.bar") for l in labels):
        print(f"[test] FAIL [8/15]: protocol method 'IFoo.bar' missing from {list(labels)}")
        return 8
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["Greeter.bar"] and e["target"] == labels["Greeter.helper"] for e in calls):
        print(f"[test] FAIL [8/15]: Greeter.bar -> Greeter.helper call missing: {calls}")
        return 8
    imports = _edges(r, "imports")
    if not any(e.get("module") == "Foundation" for e in imports):
        print(f"[test] FAIL [8/15]: 'import Foundation' missing: {imports}")
        return 8
    print(f"[test] OK [8/15] Swift: {len(calls)} calls, {len(imports)} imports")

    # ============================================================
    # 9. Cross-file unresolved — raw_calls populated
    # ============================================================
    py_src2 = """def my_func():
    external_func(1)   # not defined in this file
    SomeClass.method()
"""
    r = extract_file("client.py", py_src2)
    if r is None:
        print("[test] FAIL [9/15]: extract_file returned None for client.py")
        return 9
    raw_names = {rc["callee_name"] for rc in r["raw_calls"]}
    if "external_func" not in raw_names:
        print(f"[test] FAIL [9/15]: 'external_func' not in raw_calls: {raw_names}")
        return 9
    # SomeClass.method() should also be captured as a raw_call to "method"
    if "method" not in raw_names:
        print(f"[test] FAIL [9/15]: 'method' (member call) not in raw_calls: {raw_names}")
        return 9
    is_member_flags = {rc["callee_name"]: rc["is_member"] for rc in r["raw_calls"]}
    if is_member_flags.get("method") is not True:
        print(f"[test] FAIL [9/15]: 'method' should have is_member=True, got {is_member_flags}")
        return 9
    if is_member_flags.get("external_func") is not False:
        print(f"[test] FAIL [9/15]: 'external_func' should have is_member=False, got {is_member_flags}")
        return 9
    print(f"[test] OK [9/15] cross-file unresolved: {len(r['raw_calls'])} raw_calls with correct is_member flags")

    # ============================================================
    # 10. Unsupported file — None
    # ============================================================
    r = extract_file("readme.md", "# Hello\n")
    if r is not None:
        print(f"[test] FAIL [10/15]: extract_file should return None for .md, got {type(r)}")
        return 10
    print(f"[test] OK [10/15] unsupported file: returns None (markdown)")

    # ============================================================
    # 11. Inheritance extraction — raw_inherits across language families
    # ============================================================
    # C#: undifferentiated base list (grammar can't tell extends from implements).
    cs_src = """namespace Foo {
  public class Base {}
  public interface IFoo {}
  public class Derived : Base, IFoo {}
  public class Generic<T> : System.Collections.Generic.List<T>, IFoo where T : struct {}
}
"""
    r = extract_file("foo.cs", cs_src)
    if r is None or "raw_inherits" not in r:
        print(f"[test] FAIL [11/15]: C# extractor missing raw_inherits")
        return 11
    ri_cs = r["raw_inherits"]
    cs_pairs = {(x["base_name"], x["base_kind"]) for x in ri_cs}
    if not {("Base", "extends_or_implements"), ("IFoo", "extends_or_implements")}.issubset(cs_pairs):
        print(f"[test] FAIL [11/15]: C# raw_inherits missing Base/IFoo: {cs_pairs}")
        return 11
    if not any(x["base_name"] == "List" for x in ri_cs):
        print(f"[test] FAIL [11/15]: C# generic List<T> base did not collapse to 'List': {cs_pairs}")
        return 11

    # Python: every base is `extends`, generic `Container[int]` -> `Container`.
    py_src2 = """class Base: pass
class Mixin: pass
class Derived(Base, Mixin, generic.Container[int]): pass
"""
    r = extract_file("foo.py", py_src2)
    ri_py = r["raw_inherits"]
    py_names = sorted(x["base_name"] for x in ri_py)
    if py_names != ["Base", "Container", "Mixin"]:
        print(f"[test] FAIL [11/15]: Python bases unexpected: {py_names}")
        return 11
    if any(x["base_kind"] != "extends" for x in ri_py):
        print(f"[test] FAIL [11/15]: Python should mark all bases as 'extends': {ri_py}")
        return 11

    # TypeScript: extends vs implements distinguished.
    ts_src2 = """abstract class Base {}
interface IFoo { foo(): void; }
class Derived extends Base implements IFoo { foo(): void {} }
"""
    r = extract_file("foo.ts", ts_src2)
    ri_ts = r["raw_inherits"]
    kinds_by_name = {x["base_name"]: x["base_kind"] for x in ri_ts}
    if kinds_by_name.get("Base") != "extends":
        print(f"[test] FAIL [11/15]: TS Base should be 'extends', got {kinds_by_name}")
        return 11
    if kinds_by_name.get("IFoo") != "implements":
        print(f"[test] FAIL [11/15]: TS IFoo should be 'implements', got {kinds_by_name}")
        return 11

    # Java: superclass (extends) + super_interfaces (implements).
    java_src2 = """package x;
class Base {}
interface IFoo {}
class Derived extends Base implements IFoo, Runnable {}
"""
    r = extract_file("foo.java", java_src2)
    ri_java = r["raw_inherits"]
    java_kinds = {x["base_name"]: x["base_kind"] for x in ri_java}
    if java_kinds.get("Base") != "extends":
        print(f"[test] FAIL [11/15]: Java Base should be 'extends', got {java_kinds}")
        return 11
    if java_kinds.get("IFoo") != "implements" or java_kinds.get("Runnable") != "implements":
        print(f"[test] FAIL [11/15]: Java interfaces should be 'implements', got {java_kinds}")
        return 11

    # Class with no base list should produce zero raw_inherits.
    r = extract_file("none.py", "class Solo:\n    pass\n")
    if r["raw_inherits"]:
        print(f"[test] FAIL [11/15]: class with no bases should not emit raw_inherits: {r['raw_inherits']}")
        return 11

    print(f"[test] OK [11/15] inheritance extracted: C#({len(ri_cs)}), Python({len(ri_py)}), TS({len(ri_ts)}), Java({len(ri_java)})")

    # ============================================================
    # 12. Scala — functions, calls, inheritance (extends / with), import
    # ============================================================
    scala_src = (
        "import scala.collection.mutable.Map\n"
        "class Animal(name: String) extends Base with Trait1 {\n"
        "  def speak(): String = helper(name)\n"
        "  def helper(s: String): String = s\n"
        "}\n"
        "object Main { def run(): Unit = { val a = new Animal(\"x\"); a.speak() } }\n"
        "trait Trait1 { def f(): Unit }\n"
    )
    r = extract_file("demo.scala", scala_src)
    if r is None:
        print("[test] FAIL [12/15]: extract_file returned None for demo.scala")
        return 12
    labels = _ids_by_label(r["nodes"])
    for want in ("Animal.speak", "Animal.helper", "Main.run"):
        if want not in labels:
            print(f"[test] FAIL [12/15]: Scala node {want!r} missing from {list(labels)}")
            return 12
    if len(_edges(r, "calls")) < 2:
        print(f"[test] FAIL [12/15]: Scala expected >=2 call edges, got {len(_edges(r, 'calls'))}")
        return 12
    bases = {i["base_name"] for i in r["raw_inherits"]}
    if "Base" not in bases or "Trait1" not in bases:
        print(f"[test] FAIL [12/15]: Scala inheritance bases missing, got {bases}")
        return 12
    if not _edges(r, "imports"):
        print("[test] FAIL [12/15]: Scala import edge missing")
        return 12
    print(f"[test] OK [12/15] Scala: functions + {len(_edges(r,'calls'))} calls + extends(Base,Trait1) + import")

    # ============================================================
    # 13. Lua — functions + intra-file call graph
    # ============================================================
    lua_src = (
        "local function add(a, b) return a + b end\n"
        "function greet(name) return add(1, 2) end\n"
    )
    r = extract_file("demo.lua", lua_src)
    if r is None:
        print("[test] FAIL [13/15]: extract_file returned None for demo.lua")
        return 13
    labels = _ids_by_label(r["nodes"])
    if "add" not in labels or "greet" not in labels:
        print(f"[test] FAIL [13/15]: Lua functions missing from {list(labels)}")
        return 13
    if len(_edges(r, "calls")) < 1:
        print("[test] FAIL [13/15]: Lua call edge greet->add missing")
        return 13
    print(f"[test] OK [13/15] Lua: {len(labels)} functions, call graph greet->add")

    # ============================================================
    # 14. Bash — functions + call graph (commands resolving to functions)
    # ============================================================
    sh_src = (
        "greet() { echo hi; }\n"
        "helper() { greet; }\n"
        "main() { greet; helper; }\n"
    )
    r = extract_file("demo.sh", sh_src)
    if r is None:
        print("[test] FAIL [14/15]: extract_file returned None for demo.sh")
        return 14
    labels = _ids_by_label(r["nodes"])
    for want in ("greet", "helper", "main"):
        if want not in labels:
            print(f"[test] FAIL [14/15]: Bash function {want!r} missing from {list(labels)}")
            return 14
    if len(_edges(r, "calls")) < 3:
        print(f"[test] FAIL [14/15]: Bash expected >=3 call edges, got {len(_edges(r,'calls'))}")
        return 14
    print(f"[test] OK [14/15] Bash: {len(labels)} functions, {len(_edges(r,'calls'))} call edges, 'echo' dropped")

    # ============================================================
    # 15. Objective-C — methods, message-send call graph, #import,
    #     superclass/protocol inheritance
    # ============================================================
    objc_src = """#import <Foundation/Foundation.h>
#import "Helper.h"

@interface Calc : NSObject <Pingable, NSCopying>
- (int)addValue:(int)v;
@end

@implementation Calc
- (int)helper:(int)x { return x * 2; }
- (int)compute:(int)v {
    return [self helper:v];
}
@end
"""
    r = extract_file("Calc.m", objc_src)
    if r is None:
        print("[test] FAIL [15/15]: extract_file returned None for Calc.m")
        return 15
    labels = _ids_by_label(r["nodes"])
    if "Calc.helper" not in labels or "Calc.compute" not in labels:
        print(f"[test] FAIL [15/15]: Calc.{{helper,compute}} missing from {list(labels)}")
        return 15
    calls = _edges(r, "calls")
    if not any(e["source"] == labels["Calc.compute"] and e["target"] == labels["Calc.helper"] for e in calls):
        print(f"[test] FAIL [15/15]: message-send [self helper:] not resolved to Calc.helper: {calls}")
        return 15
    imports = _edges(r, "imports")
    if not any((e.get("module") or "").endswith("Helper.h") for e in imports):
        print(f"[test] FAIL [15/15]: '#import \"Helper.h\"' produced no import edge: {imports}")
        return 15
    inh = {(x["base_name"], x["base_kind"]) for x in r["raw_inherits"]}
    if ("NSObject", "extends") not in inh:
        print(f"[test] FAIL [15/15]: ObjC superclass NSObject(extends) missing: {inh}")
        return 15
    if ("Pingable", "implements") not in inh:
        print(f"[test] FAIL [15/15]: ObjC protocol Pingable(implements) missing: {inh}")
        return 15
    print(f"[test] OK [15/15] Objective-C: methods + message-send call + #import + super/protocol inherit")

    # ============================================================
    # 16. No `calls` edge may be a self-loop. Three ways a name-only match
    #     resolves to the enclosing function and would self-loop:
    #       - member call on another receiver: obj.GetVoxel() inside GetVoxel
    #       - overload call: GetVoxel(x) from GetVoxel(x, y) (overloads share
    #         one node id since ids are not signature-aware)
    #       - genuine recursion: Countdown(n - 1) inside Countdown
    #     All three are noise as self-loops and must be dropped.
    # ============================================================
    cs_self = """namespace Demo
{
    class VoxelWorld
    {
        public int GetVoxel(int x)
        {
            return chunk.GetVoxel(x);       // member call on ANOTHER receiver
        }

        public int GetVoxel(int x, int y)   // overload -> same node id
        {
            return GetVoxel(x);             // overload call looks like recursion
        }

        public int Countdown(int n)
        {
            return Countdown(n - 1);        // genuine recursion
        }
    }
}
"""
    r = extract_file("VoxelWorld.cs", cs_self)
    if r is None:
        print("[test] FAIL [16/16]: extract_file returned None on VoxelWorld.cs")
        return 16
    calls = _edges(r, "calls")
    self_loops = [e for e in calls if e["source"] == e["target"]]
    if self_loops:
        print(f"[test] FAIL [16/16]: self-loop call edge(s) emitted: {self_loops}")
        return 16
    print(f"[test] OK [16/16] no self-loop call edges (member/overload/recursion all dropped)")

    print("\n[test] === SUCCESS: graph extractor works as expected ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
