"""What a chunk carries besides its declaration: the doc comment above it,
and for a type the skeleton (doc, header, fields) its members leave out.
Also what the embedding model reads of a chunk's metadata."""
from __future__ import annotations

from lynx.chunking import chunk_file
from lynx.outline import doc_of, signature_of


def _by_name(path, src):
    return {c["symbol_name"]: c for c in chunk_file(path, src)}


CS = """using System;
namespace Game {
/// <summary>Keeps a unit's hit points.</summary>
public class Health : Component {
    int current = 10; // trailing note
    public const int Max = 100;

    /// <summary>
    /// Removes <c>{amount}</c> hit points.
    /// </summary>
    [Obsolete]
    public void ApplyDamage(int amount) { current -= amount; }

    // a note about the next region, then a blank line

    public void Heal() { }
}
}
"""


def test_csharp_doc_comment_travels_with_its_method():
    chunks = _by_name("Health.cs", CS)
    apply = chunks["Game.Health.ApplyDamage"]
    assert apply["text"].startswith("/// <summary>")
    assert apply["start_line"] == 8                   # the doc's first line
    # The `{amount}` inside the doc is not where the signature stops.
    assert signature_of(apply["text"]) == "[Obsolete] public void ApplyDamage(int amount)"
    assert doc_of(apply["text"]) == "Removes {amount} hit points."


def test_a_comment_behind_a_blank_line_or_trailing_code_stays_out():
    chunks = _by_name("Health.cs", CS)
    assert chunks["Game.Health.Heal"]["text"].startswith("public void Heal")
    skeleton = chunks["Game.Health"]["text"]
    assert "trailing note" not in skeleton


def test_a_type_skeleton_holds_its_doc_and_fields_not_its_members():
    skeleton = _by_name("Health.cs", CS)["Game.Health"]
    text = skeleton["text"]
    assert text.startswith("/// <summary>Keeps a unit's hit points.</summary>")
    assert "public class Health : Component {" in text
    assert "int current = 10;" in text and "public const int Max = 100;" in text
    assert "ApplyDamage" not in text                  # indexed in its own chunk
    assert skeleton["symbol_kind"] == "class_declaration"


def test_a_bare_namespace_gets_no_skeleton():
    names = _by_name("Health.cs", CS)
    assert "Game" not in names


def test_javadoc_go_rust_and_exported_typescript():
    java = _by_name("RateLimiter.java", """package p;
/**
 * A rate limiter that hands out permits. {@link Foo}
 */
public class RateLimiter {
  private double rate;
  /**
   * Acquires one permit.
   * @param n permits
   */
  public void acquire(int n) { }
}
""")
    assert java["RateLimiter.acquire"]["text"].startswith("/**")
    assert doc_of(java["RateLimiter.acquire"]["text"]) == "Acquires one permit."
    assert doc_of(java["RateLimiter"]["text"]) == "A rate limiter that hands out permits. Foo"

    go = _by_name("a.go", "package p\n\n// Foo does things.\nfunc Foo() {}\n")
    assert go["Foo"]["text"].startswith("// Foo does things.")

    rust = _by_name("a.rs", "/// Rust doc.\nfn foo() {}\n")
    assert rust["foo"]["text"].startswith("/// Rust doc.")
    assert rust["foo"]["start_line"] == 1

    ts = chunk_file("a.ts", "/** TS doc */\nexport function f() {}\nexport class K {\n  m() {}\n}\n")
    f = next(c for c in ts if c["symbol_name"] == "f")
    assert f["text"].startswith("/** TS doc */")
    header = next((c for c in ts if c["symbol_name"] == "<header>"), None)
    # The exported class has chunks of its own; the header used to copy it.
    assert header is None or "class K" not in header["text"]


def test_python_class_docstring_and_fields_are_indexed():
    chunks = _by_name("models.py", '''class Article(Model):
    """A published article."""
    title = CharField(max_length=200)

    def save(self):
        pass
''')
    text = chunks["Article"]["text"]
    assert '"""A published article."""' in text and "title = CharField" in text
    assert "def save" not in text


def test_ruby_class_keyword_is_not_a_container():
    names = _by_name("u.rb", "class User < Base\n  def greet\n  end\nend\n")
    assert not any("<container>" in n for n in names)


# ---------------------------------------------------------------------------
# What the embedding model reads
# ---------------------------------------------------------------------------

def test_embedding_text_carries_path_and_name_not_bookkeeping(tmp_path):
    from llama_index.core.schema import MetadataMode
    from lynx.rag_manager import CodebaseRAG

    root = tmp_path / "proj"
    (root / "Combat").mkdir(parents=True)
    f = root / "Combat" / "Health.cs"
    f.write_text(CS, encoding="utf-8")
    rag = CodebaseRAG.__new__(CodebaseRAG)
    rag.codebase_path = root
    nodes = rag._build_nodes_for_file(str(f))
    method = next(n for n in nodes if n.metadata["symbol_name"] == "Game.Health.ApplyDamage")
    embedded = method.get_content(metadata_mode=MetadataMode.EMBED)
    assert "rel_path: Combat/Health.cs" in embedded
    assert "symbol_name: Game.Health.ApplyDamage" in embedded
    for noise in ("chunker", "start_line", "end_line", "symbol_kind", "file_name", str(root)):
        assert noise not in embedded, noise
    # What the tools hand back stays the code alone.
    assert method.get_content(metadata_mode=MetadataMode.NONE) == method.text
    header = next(n for n in nodes if n.metadata["symbol_name"] == "<header>")
    assert "symbol_name" not in header.get_content(metadata_mode=MetadataMode.EMBED)


def test_doc_above_steps_over_decorators_and_attributes():
    from lynx.outline import doc_above

    gd = ["## Fires the weapon.", '@rpc("any_peer")', "func fire():", "\tpass"]
    assert doc_above(gd, 2) == "Fires the weapon."
    cs = ["/// <summary>Opens the door.</summary>", "[Command]", "[Obsolete]",
          "public void Open() {}"]
    assert doc_above(cs, 3) == "Opens the door."
    rust = ["/// Adds one.", "#[inline]", "fn inc(x: u8) -> u8 { x + 1 }"]
    assert doc_above(rust, 2) == "Adds one."
    # An attribute alone is not a doc.
    assert doc_above(["[Command]", "public void Open() {}"], 1) == ""
