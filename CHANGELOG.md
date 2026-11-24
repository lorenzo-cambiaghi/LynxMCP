# Changelog

## Unreleased

### Added

`graph_query` `subclasses` and `superclasses` take `depth` (1-10): the whole
hierarchy in one call, level by level, each type once with its level. On
Django 5.2 the tree under `Field`, 100 classes over 4 levels, took one call
per class (101); now it takes one. The Playground and `/api/v1/graph` take
it too.

A codebase source can say what its code and comments are written in:
`"language": {"code": "en", "comments": "it"}`, ISO 639-1 codes, English by
default. The instructions Lynx sends at startup tell the agent to write its
search queries in the code's language, whatever language the user writes
in, and `list_sources` shows it. On a private C# library, requests in
Italian found the right file in the first five for none of the 65 that
named no identifier; the same requests in English, for 32.

C# type references are graph edges, `uses_type`, resolved by the same
namespace and `using` rules as bases: `new T()`, `typeof(T)`, generic
arguments (`Add<Bullets>()`, `GetService<T>()`), parameter and return
types, field and property types. A type registered by its name is no
longer an orphan, and who depends on a type no longer needs a text
search. Static member access (`Kind.Rain`, `JsonContext.Default`)
counts too. On the 1.4-million-line library: 134,499 edges, 36,098 of
them from static member access.

`graph_query` `audit` (symbol = a C# package folder) says where the
package breaks SOLID: types far larger than the codebase's own norm and
the jobs their method names point to, long methods, wide interfaces,
members that only throw `NotImplemented`/`NotSupported`, the same `switch`
in several places, mutable static state, a dependency on a less stable
project, deep hierarchies, internal types nothing uses. Thresholds are a
high percentile of the codebase's own sizes. Each finding comes with the
fence of its refactor, from the graph: the members other packages use
and how many callers each has, the projects to build, the test files that
reach the type, how often its files changed in 180 days. The same report
from the command line: `python -m lynx.audit.solid ROOT PACKAGE`. On an
85-file networking package of the private library it took 17 seconds and
named two classes of 2,376 and 1,943 lines with the jobs in their names
(broadcast, drain, frame, peer, handshake, resume), three handshake
methods over 90 lines and a transport whose `ConnectAsync` only throws.

### Changed

With the graph on, the startup instructions send a new type to its
siblings: list the base's subclasses with `graph_query` and `depth`, and
follow how one is built and registered. On six feature tasks of a private
C# library no session asked the graph, and neither arm found the
extension points the author had used.

CI runs the script suites too (the `tests/test_*.py` files with a `main()`
that pytest leaves out), through `tests/run_script_suites.py`, which finds
them as conftest does; the six that need the embedding model run where it
is installed.

### Fixed

Base classes are resolved the way the language resolves them, not by name
alone. A base is a type, never a method or a namespace that shares its
name, nor the class itself. Python looks in the class's file, then in the
files it imports, then one step further (a package re-exporting its
modules). C# looks in the same file, the namespace and the enclosing ones,
the file's `using` lines (aliases included), the project's `global using`
lines and MSBuild `<Using>` items, or the namespace a qualified base
names; a base none of them can see comes from outside the codebase and
gets no edge. Partial declarations count as one type. On Django 5.2 the
ambiguous `inherits` edges fell from 2,085 of 3,300 (63%) to 6 of 1,622,
and the classes linked to several namesake bases from 377 to 2. On a
private C# library of 1.4 million lines they fell from 9,251 of 11,952
(77%, 7,400 of them pointing at methods) to 8 of 3,649, with no real base
lost among the 34 references left unlinked (Avalonia and Microsoft types,
and three generics named like their base).

A C# 14 `extension(T x) { ... }` block is read as the methods it holds,
not as one function named `extension` (one was 669 lines long).

C# calls are bound by type, not by name. `Foo()` and `this.Foo()` look in
the caller's type and its bases, `base.Foo()` in the bases, `x.Foo()` in
the declared type of `x` (field, property, parameter, local, `var x = new
T()`, `foreach`, pattern), `Type.Foo()` in a type the file can see, then
the extension methods on those types; a receiver of unknown type resolves
only when one visible type declares the name. `Add<T>()` is no longer
lost, and a call on a type outside the codebase (`Assert`, `string`) gets
no edge. On the 1.4-million-line library the call edges went from
1,115,634, 91% of them ambiguous (a plain `Foo()` was linked to every
method named `Foo`, 46 on average), to 146,981, none ambiguous; bases
are resolved first, so a call to an inherited member finds it.

## 1.12.0 - 2026-11-23

Upgrading from 1.11.0: the chunker moves to version 6 (see Changed), so the
first start after this upgrade embeds every codebase source again, in the
background, while the source keeps answering. Builds now embed about 1.7
times faster, so count roughly half an hour for a source of 50,000 chunks on
a CPU and a few minutes for a small one. The code graph is rebuilt once too,
because its extractor changed. The tool profiles moved around a little:
`deep_search` and `feedback` are in `full` only, `module_summary` is in
`core`, and the two hook tools are in every profile. The default `standard`
list is smaller than in 1.11.0 all the same.

### Added

`lynx serve PATH` needs no config file. It indexes that folder as one
codebase source, with the extensions and ignores it finds there, keeps the
index in your user cache folder so the repository stays clean, and turns the
code graph on. It refuses a home folder or a drive root.

`lynx connect <client>` registers Lynx with Claude Code, Cursor, Claude
Desktop, Windsurf, Antigravity or VS Code. It adds one entry to the client's
MCP config, keeps a timestamped backup, leaves every other server alone, and
updates an existing Lynx entry instead of adding a second one. Claude Code
gets Lynx at user scope, so every project you open has its own index, plus
the prompt hook in `~/.claude/settings.json` (`--no-hook` leaves it out,
`--disconnect` removes both). LynxManager's Integrations page does the same
with one button per client.

`lynx update` recognises uv, pipx, pip and uvx installs and upgrades them in
place; config and indexes stay as they are. On Windows it lists the running
Lynx processes that hold files open and stops before touching anything. The
update button in LynxManager closes the manager, updates in its own window
and reopens it.

Two tools that Claude Code's hooks call, not the model. `prompt_hook`
answers the `UserPromptSubmit` hook with a short map of the code a request
is about: up to 12 places, with file, lines and signature, from one search
per sentence. `read_guard` answers an optional `PreToolUse` hook on `Read`
and `Bash`: it refuses once a read of more than 200 lines of an indexed
file, with the file's outline as the reason, and lets the same read through
when it is asked again. Both answer with nothing rather than an error when
the index is not ready. `docs/GUIDE.md` has the hook settings.

`search` takes `identifiers`, the names you would grep for (`labelHover,
hoverLabel`), added to the query. Agents wrote such names when there was a
slot for them and never inside the query itself; on 22 bug reports they
raised the MRR of the file to fix from 0.626 to 0.684. `lynx search
--identifiers` does the same from the command line.

`describe_symbol` takes `file`, to pick one of several definitions of a name
(a test helper defined in 15 files). The others are listed, and the callers
shown are those of the chosen definition only.

`lynx status --json` and `lynx list-sources --json`. With a source that
does not exist, `status` answers `"ok": false` and exits with 1.

The LynxManager Playground takes the search mode, identifiers and outline,
and a file for `describe_symbol`. A source can be removed from its page.

`docs/MANAGER.md`, the LynxManager guide, with pictures of the current UI.
`scripts/manager_guide.py` takes them by driving the real interface in a
headless browser on a sample project (first launch, adding a project, a
search, connecting an editor, the Expert pages) and writes the guide and the
README's tour from the same scenes, so they change when the UI does.

### Changed

Code ranks above tests, data and docs. After fusion, the score of a test,
data or doc file is multiplied by `search.secondary_weight` (0.5 by default;
1.0 turns it off). On 76 issue titles of a TypeScript game, with the files
their fixes changed as the answers, MRR went from 0.185 to 0.362. BM25 also
reads the last four segments of each chunk's path and its symbol name.

Chunker 6. A long TypeScript or JavaScript function gives its named nested
functions chunks of their own, so a helper defined inside a 300-line
function is found by its own name. The parts of a function split across
several chunks carry their own line numbers instead of the whole
function's.

The code graph knows much more of a TypeScript or JavaScript project.
Relative imports point at files (an `index` file included), `export ...
from` re-exports are followed to the file that defines the name, and
`paths` and `baseUrl` aliases are read from `tsconfig.json` or
`jsconfig.json`, `extends` included. Arrow functions and function-valued
class fields are nodes. Calls inside closures and callbacks count for the
function they are written in. A call `obj.f()` lands only on a method `f` in
a file the caller imports, and a name defined in several files resolves to
the one it can reach. On a game of 214,000 lines this took the ambiguous
call edges from 3,899 to 467.

Beyond TS/JS: Python imports resolve to the project's own files (package
roots found through `pyproject.toml`, `setup.py`, `setup.cfg`, `.git` or a
`src/` folder; the standard library is left out), imports inside function
bodies count in every language, and Java calls on a receiver are member
calls. On Lynx's own source, resolved calls went from 311 to 448 and imports
of project files from 0 to 165. An incremental update now gives the graph a
full build gives: the files that import a changed file are read again, and a
removed file takes its edges with it.

Index builds embed about 1.7 times faster, with the same vectors. Chunks of
many files go to the embedder together, 512 at a time, sorted by length and
run 32 at a time, where before each file's few chunks went alone. Measured
on 2,202 chunks of a TypeScript project, the old and new way alternating on
a busy machine: 1.70 times faster in wall time, 1.68 in CPU time. The
vectors are equal to the bit. If a group fails, its files are retried one
at a time, so a bad file is still the only one left out.

Tool profiles: `core` is `search`, `describe_symbol`, `module_summary`,
`find_usages`, `impact` and the two hook tools (7); `standard` adds
`find_definition`, `find_similar`, `graph_query`, `repo_overview` and
`search_diff` (12); `full` has all 19. Their `tools/list` weighs 6.0k,
10.6k and 16.2k characters, about 1,500, 2,650 and 4,050 tokens (standard
was 12.9k in 1.11.0). Optional arguments are advertised without the null
branch, and the tool texts and answers are shorter.

Refreshed the README and benchmark results for 1.12.0, focusing on retrieval
quality, code graph capabilities and multi-source search.

### Fixed

Importing Lynx drops an `SSLKEYLOGFILE` that an antivirus injected into the
environment. Python builds without `OPENSSL_Applink`, uv's among them,
crashed at the first TLS connection with it set, so hooks and scripts that
import Lynx died where `lynx serve` did not.

A client folder that cannot be read is now that client's error, instead of
a crash of `lynx connect` and of the Integrations page.

A search mode chosen for one search (`lynx search --mode`, the Playground)
no longer sticks to the source for the searches after it.

LynxManager: Simple mode stays simple in the Playground and on Doctor. Add
source scans a folder the way `lynx serve` does, with the same extensions
and ignores. A running build shows on its source's page, and a build and a
reset of the same source can no longer start together. Config saves from
the page go through the same lock as the rest of Lynx, and a save that
fails shows why. The Update page shows the command's output.

The first project added from LynxManager is built. Right after it was
added, the build and the page's own requests each opened the new index at
once, and the manager reported it as corrupt; now one request opens it and
the others wait for it.

Benchmark scripts now always re-check existing indexes against the current
chunker and count graph and grep calls consistently.

## 1.11.0 - 2026-11-20

Upgrading from 1.10.0: on a source that lists `.gd`, the first start
re-chunks its GDScript files, in place and in the background. Nothing else
is re-embedded. The code graph is rebuilt once, because its extractor
changed: about 17 seconds for a source of 3,500 C# files, while `lynx serve`
already answers.

### Added

GDScript. `.gd` files are chunked as functions, `_init`, enums and inner
classes, each with the `##` doc comment and any annotation (`@rpc`) on the
lines above it. A script is a class, as in Godot: its functions are named
after its `class_name`, or after the file when it has none
(`Player.take_damage`, `enemy.attack`), and the code graph has a class node
for it. Its `extends` is an inheritance edge, whether it names a class or a
script path; a script that declares `class_name BaseEnemy` in
`enemy_base.gd` answers to both names. `preload` / `load` of a constant
path count as imports. The grammar has wheels for Windows x64, Linux x86_64
and macOS on Apple silicon, so it is installed on those three; elsewhere
`.gd` files are still indexed, as text.

`repo_overview` recognizes a Godot project (`project.godot`, listed among
the manifests) and GDScript among the languages. The extensions the web UI
suggests for a new source leave out Godot's `.import` and `.uid` sidecar
files. `find_tests_for` takes a `test_*` file outside a test folder for a
test (pytest, Godot's GUT).

Coral 1.x can use Lynx again. Coral 1.0 dropped the SQL sources Lynx was
one of and builds a provider from an OpenAPI document instead, so
`GET /api/v1/openapi.json` now describes the `/api/v1` API: `search.query`,
`search.batch`, `graph.query` and `sources.list` become typed tools in the
programs Coral's agents run (`coral.providers.lynx.search.query`). The
document covers `/api/v1` only, so the web UI's own routes are never offered
as tools, and the JSON the routes return is unchanged. `docs/CORAL.md`
describes the setup; the SQL manifest stays for Coral 0.x.

`/api/v1` rows carry paths relative to the source root: `rel_path` on a
search hit, `from_rel_path`, `to_rel_path` and `call_site_rel_path` on a
graph edge.

### Fixed

A file whose extension gained a grammar kept the text chunks it was indexed
with until its content changed, and `lynx build` did not help, because the
SHA cache skips unchanged files. That is how Objective-C reached existing
indexes in 1.6.0, against what its changelog entry said. The cache now
records which extensions had a grammar, and when that list changes, the
files of those extensions are re-chunked in place, and only those.

`describe_symbol` and `module_summary` lost the doc comment of a symbol with
a decorator or an attribute between the comment and the declaration
(`@rpc`, `@Override`, `[Command]`, `#[inline]`). Search results had it.

`search` switched to outlines too early when long chunks came first: its
16,000-character budget counted each chunk whole instead of the 4,000
characters it shows.

## 1.10.0 - 2026-11-20

Upgrading from 1.9.0: the first start rebuilds every code graph once,
because node IDs changed (see Fixed). That is about 17 seconds for a
source of 3,500 C# files, spent while `lynx serve` already answers. Every
start now also checks each codebase source
for files changed while no Lynx was running, in the background. The default
`standard` profile loads one more tool, `find_similar`, for about 220 more
tokens of definitions per session.

The chunker moves to version 5 (see Changed), so the first start after this
upgrade embeds every codebase source again, in the background. It replaces
one file at a time and the source keeps answering throughout. Count about 15
to 20 chunks a second on a CPU: close to an hour for a source of 50,000
chunks, a few minutes for a small one. The code graph is rebuilt once too.
The `standard` profile also loads `module_summary`, for about 210 more tokens
of definitions.

### Added

`lynx serve --source NAME`, repeatable, serves only those sources of the
config. One config can list every repository you work on while each project's
MCP entry opens just the indexes it needs; before, every session opened all
of them. An unknown name stops the server with the list of known ones.

The web UI says which code runs where. The sidebar shows the version and a
hash of the code, and every source card has a "Served by" row: this UI, or
the pid of the process that owns the index, with a badge saying whether it
runs the same code or other code. An editable checkout reports the same
version number before and after its code changes, which is why the number
alone never answered "is the MCP server still on the old code?".

`ignored_globs` per codebase source, next to `ignored_path_fragments`:
file-name or relative-path patterns to leave out, such as
`**/20??-??-??-*.txt` for dated chat transcripts kept inside a source. Searches
drop the excluded files at once, before the next build takes them out of the
index. On a 4,868-file Unity source, eleven transcripts (3.4 MB of text) had
been ranking among the first five results of shader queries; with the glob
they no longer appear.

`lynx manager doctor` checks two more things. It loads each source's lexical
index the way a search does and runs a query against it, so a hybrid search
that has quietly become dense-only shows up as an error. And it compares the
files on disk with the files in the index, listing the ones missing and the
ones that are gone or excluded now. On the first real run that found 158
deleted files still searchable in one source and 407 stale or out-of-scope
files in another, behind an "All checks passed".

The doctor also tells you when the Lynx process serving an index runs other
code than the command you typed: an editable checkout behind the MCP client
and the published release on PATH both reported 1.9.0 while one discarded
every graph the other wrote. Ownership claims now carry the version, the
install path and a hash of the code.

`search` and `find_similar` ask clients to load them up front
(`_meta["anthropic/alwaysLoad"]`). Claude Code defers MCP tools behind a tool
search by default, so a session saw only their names while Grep and Glob were
ready to call, and agents reached for those.

A rules snippet for `AGENTS.md` or `CLAUDE.md`,
[`integrations/agent-rules/lynx-rules.md`](integrations/agent-rules/lynx-rules.md),
with today's tool names and the real score scale. The template in the guide
still listed tool names that had not existed since before 1.0.

### Changed

Search finds more of what it is asked for. Three changes to what goes into
the index, measured on the benchmark repositories with hybrid search, top 5,
against the same index built the old way:

| | hit@5 | hit@1 | MRR |
|---|---|---|---|
| Django (20 tasks) | 85% to 100% | 55% to 60% | 0.67 to 0.77 |
| Json.NET (15 tasks) | 73% to 73% | 47% to 67% | 0.58 to 0.70 |
| Guava (15 tasks) | 80% to 87% | 60% to 67% | 0.70 to 0.74 |

The embedding model used to read eight lines of bookkeeping above every
chunk: the absolute path, which repeats the same prefix on every chunk of a
source, the chunker's name, the line numbers. About 60 of its 512 tokens. It
now reads the path inside the source and the symbol's name, and the code
gets the rest. That change alone took Django to 95% at hit@5 and Json.NET
to 87%. A chunk now also starts at the doc comment written above its
declaration (`///` XML docs, Javadoc and JSDoc, `//` in Go and Rust), which
tree-sitter keeps next to the node and the chunker dropped. And every class,
struct or interface gets a chunk of its own with its doc comment, its
declaration and its fields: class docstrings, Javadoc on types and the
fields of a Django model were in no chunk at all. On Json.NET the doc
comments cost two tasks at hit@5 against the cleaner embedding alone, and
won one at hit@1; on Guava, where nearly every method has Javadoc, they won
two at hit@5. Putting the symbol name and the path into the lexical index
as well was measured again and left out, as in October: Django fell from
100% to 90% at hit@5.

The answers cost fewer tokens. Paths are written relative to the source
root, where `search` used to print a bare file name (ambiguous in a large
tree, so the agent went looking with glob) and the graph tools an absolute
path on every line. A graph answer names the queried symbol once and lists
each caller, callee or subclass on one line under its file, instead of three
lines per edge. On Django: callers of `get_prep_value` 6,043 to 1,965
characters, subclasses of `Field` 18,984 to 3,425, an `impact` 2,396 to
1,484. `search` cuts a body past 4,000 characters (1% of chunks are longer)
and says which lines are left, and once an answer holds 16,000 characters of
code the remaining hits come as outlines, still ranked and cited. The 20
Django benchmark queries at `top_k=8` came back 23% shorter in total, the
largest answer half as long. The HTTP API keeps the absolute `file` and
`file_path` and adds `rel_path`.

`find_definition`, `describe_symbol` and `module_summary` bring the code with
them, so the next step is no longer a `read_file`. `find_definition` returns
the body of the first match and the signature of the others.
`describe_symbol` returns the body of a function; for a class, its bases,
its subtypes and the signature of each member, instead of a body that can
run to a thousand lines. `module_summary` lists every symbol of a file with
its signature and doc line, works without the graph layer (from the file's
own chunks), and joins the `standard` profile.

`find_usages` and `find_tests_for` no longer return code that only looks
like the name. They asked the hybrid search, whose dense half brings back
similar code, and kept every hit; now they ask the lexical half and keep a
chunk only when it contains the name as a whole word, with the lines it is
on. `find_usages` also lists the subtypes of a type. `find_tests_for` walks
the call graph up to four calls back, so a test of `HandleHit` that ends up
in `ApplyDamage` comes out as `[d2]` though it never names it, and it ranks
the textual matches among the test files only, instead of filtering the top
of a search over the whole source.

`lynx build`, the Build button in the web UI and `update_source_index` no
longer throw the code graph away. They bring it up to date the way they
already did the vector index: only files whose content changed are parsed
again, cross-file calls are resolved again, and when nothing changed the graph
on disk is left alone. On a C# source of 3,500 files that takes about 2.5
seconds instead of 17, and the first build of a new source stops extracting
the graph twice. Tests compare the result with a graph built from scratch
after edits, additions and deletions. When the extraction code or a
tree-sitter grammar has changed since the graph was built, as after an
upgrade, the build starts over by itself. `lynx reset` and
`lynx graph build --force` still rebuild from scratch.

The graph is saved as compact JSON. Saving the graph of that source took 3.5
seconds and now takes 1.1, and the watcher pays it after every saved file.

New codebase sources ignore `/obj/`, the intermediate output of .NET builds.

`find_similar` is part of the default `standard` profile. It answers "is my
draft a duplicate?" with the parallel implementations of a snippet, and on a
real source it found five copies of one quarter-turn rotation in five
modules, but agents on the default profile could not see it. The handshake
instructions now open with it and with `search`, the two questions Lynx
answers better than grep, and stay under the 2,048 characters Claude Code
keeps (1,889 with two sources; with many sources the paths are dropped first).

A query written like code (`MeshVoxelizer.Voxelize`, `CELL_ON`,
`getUserModel`) ranks the chunks named after it first in the lexical half of
search. A method's body does not contain its class name, so a two-line test
helper that called `MeshVoxelizer.Voxelize` used to outrank the method in a
lexical search; the definition now comes first. Questions in plain English
are not affected: none of the 50 tasks of the repository's benchmarks
contains such an identifier, so their ranking takes the same path as before.

The watcher and the build now agree on which files belong to a source. The
watcher indexed files inside dot directories such as `.claude/` that every
build then removed again.

### Fixed

A fresh install with uv could not index anything. nltk 3.10, which
llama-index pulls in, refuses to open a file with more than one hard link,
and uv installs packages as hard links into its cache: the sentence splitter
failed on the stopwords llama-index ships, every build stopped there, and
the source came up as a corrupt index. `uv tool install lynx-mcp` and the
Windows and macOS installers, which run uv, were affected; pip and pipx copy
the files and were not. nltk is now held below 3.10.

A chunker upgrade no longer empties the index while it rebuilds. The
collection was deleted and filled again from scratch, and since every start
now catches up with the disk, the first start after an upgrade left a large
source with no results until the last file was embedded again. When only the
chunker version changed, the stored vectors are still valid; each file's
chunks are now replaced in place, one file at a time.

In the web UI, error messages never appeared. htmx 2 drops 4xx and 5xx
responses unless told otherwise, so the 409 naming the process that owns an
index, the config validator's 422 and every playground error left the page
unchanged. Saving the config did nothing at all: the request went out empty
and its 400 was dropped too. The dashboard ran the whole doctor, with an
integrity probe per index and a full load of each lexical index, before
showing anything; the health card now loads after the page and names the
checks that are not OK. Badges had no padding, resetting an index did not ask
first, and a source that failed to load linked to a page that answered with
a JSON 404.

The TypeScript header chunk copied every exported class whole, so their code
was indexed twice. In Ruby, the `class` keyword was taken for a class and
came out as a chunk of its own.

On Windows, `lynx serve` could take minutes to open its indexes, and the first
tool calls of a session came back saying Lynx was still loading. The MCP
transport keeps a read pending on stdin, and Windows makes every other query
against that pipe wait behind it. Two things stalled there until the client
happened to send another message: the integrity probe, a child process that
inherits stdin, and native libraries such as numpy, whose DLL checks the
standard handles while it loads. The transport now reads from a private copy
of the pipe and stdin points at the null device. Measured with two indexes of
25,883 and 10,772 chunks and a client that stays quiet after the handshake,
the indexes are ready in 10 seconds; before, they were still loading after
150.

The web UI refused to build any source after a config change, saying another
Lynx process owned the index and naming its own pid. Saving the config, adding
or removing a source and the reload button all dropped the loaded manager
without giving back its claim on the indexes, so the manager built next found
that claim alive and followed its own index until the UI was restarted. The
old manager now releases its claims first.

The code graph kept the node of an external module after the last file that
imported it changed or was deleted, so a graph kept current by the watcher
drifted away from one built from scratch.

A process killed while saving the code graph could leave nodes from one save
and edges from another, and the next start loaded them as they were. A save
is now marked as in progress until its last file is written, and a graph
found in that state is rebuilt.

Two files with the same name in directories with the same name, like the
Runtime/AssemblyInfo.cs every Unity package has, shared one node in the
code graph. So could two same-named functions inside them, which then had
one list of callers and one blast radius between them. Node IDs now carry
a digest of the file path. The labels you read are the same as before.

Hybrid search lost its lexical half on any source past about 32,000 chunks,
and nothing said so. The BM25 corpus was read from ChromaDB in one call, which
binds one SQL variable per chunk and failed past SQLite's limit; the error went
to the log, and every search answered from embeddings alone. On a
53,552-chunk source, `--mode sparse` returned nothing for
"MeshVoxelizer Voxelize VoxelizationResult", hybrid scores never passed 0.0164
(one ranking fused instead of two), and every search paid 2.4 seconds for the
read that failed. The corpus is now read 5,000 chunks at a time: the same
query finds `MeshVoxelizer.Voxelize` first in sparse mode, and hybrid scores
reach 0.032 again. If the lexical index cannot load anyway, `search` and
`deep_search` say so above their results, each hit carries `mode_effective`,
and the read is retried after five minutes or when the index changes instead
of on every query. Reading and tokenizing 53,000 chunks takes about 8 seconds,
so `lynx serve` does it in the background once its tools answer, and the
first search of a session finds it loaded.

A filter could answer "no results" for a file that was indexed.
`path_contains`, `extensions`, `file_glob` and the file list of `search_diff`
were applied to the top 25 to 40 candidates after the ranking, out of 50,000
chunks. Two `.compute` files ranked 76th and 87th for the query that was
filtered to them, so the search said they did not exist, and someone wrote a
duplicate. Filters now choose the files first and the ranking happens among
them: both files are found. If ChromaDB refuses a filtered query (it can,
for a moment, on a store another process has just written to), the search
retries, then runs unfiltered on a wider pool and filters that.

Files deleted or edited while no Lynx was running stayed in the index until
someone ran a build. Only a session taking over from another one caught up;
a server starting as the owner did not. `lynx serve` now scans its sources in
the background once its watchers run.

Catching up after a takeover crawled a webdoc source's whole site again, and
re-extracted every PDF of a pdf source. It now does what an incremental
update does for them: nothing for a webdoc, changed PDFs only.

Commands no longer print `[config] removed SSLKEYLOGFILE=...` (set
`LYNX_VERBOSE=1` to see it) or "LLM is explicitly disabled. Using MockLLM."
once per source into their output.

## 1.9.0 - 2026-09-06

Upgrading from 1.8.0: the index is unchanged, nothing to rebuild. Two things
do change under you. The default tool profile loads 10 of the 17 tools, so set
`tools.profile` to `"full"` in config.json (or run `lynx serve --profile full`)
to keep the whole set; a call to a tool the profile hid answers by naming the
profile and the fix. And the install no longer contains PyTorch, but an
upgrade in place leaves the old wheels behind: reinstall to reclaim the
gigabyte, with `pipx uninstall lynx-mcp && pipx install lynx-mcp` or
`uv tool install --force --reinstall lynx-mcp`. The first run afterwards
fetches the model's ONNX graph (134 MB) into the same HuggingFace cache entry,
or `lynx manager install --model` does it up front.

### Added

Tool profiles. Every tool definition sits in the client's context on every
turn, and the seventeen tools cost about 7,700 tokens per session (30,678
characters of `tools/list` on the wire), more than three retrievals' worth of
the saving Lynx advertises. The tools are now layered: `core` (search,
describe_symbol, find_usages, impact, feedback), `standard` (core plus
find_definition, deep_search, graph_query, repo_overview, search_diff; the
default) and `full` (all seventeen). Pick one with `tools.profile` in
config.json, with `lynx serve --profile`, or with the `LYNX_TOOL_PROFILE`
variable; `tools.include` and `tools.exclude` adjust a profile by name. A
profile hides tools, it never adds one a source cannot support. The handshake
instructions and the `lynx://guide` resource name only the tools the session
has, and say which profile hid the rest and how to get them back.

The descriptions went on a diet at the same time. Each tool used to repeat
the source catalog and re-list its own arguments in prose; the catalog now
appears once, in the handshake instructions, arguments are described only in
the schema, and the schema no longer carries pydantic's generated titles or
an output schema for a `-> str` return (which also stops the SDK from sending
every result twice, as text and as a structured copy). Measured on the wire
with two sources, graph and git enabled: full 15,279 characters (about 3,800
tokens, from 7,700), standard 10,704 (about 2,700), core 4,783 (about 1,200).
A test keeps each profile under a budget, and another keeps the profile table
equal to what the server actually registers.

### Changed

The embedding model and the optional reranker run on ONNX Runtime instead of
PyTorch. The index is untouched: same model, and the vectors are the same to
the sixth decimal on every reference text (cosine 1.000000 against the
previous build, the query instruction BGE models expect included), so an
existing index keeps working without a rebuild. The three reference searches
return the same top five with the same scores, and the cross-encoder gives the
same logits to four decimals.

What changes is the install. `sentence-transformers` brought torch and
transformers, and on Linux the default torch wheel also brings the CUDA
libraries, so `pipx install lynx-mcp` downloaded 4.2 GB for a server that only
ever ran on CPU. With the lockfile regenerated the download is 165 MB on
Linux and 126 MB on Windows and macOS; the installed tool went from 1.2 GB to
462 MB on Windows, and the lock lost 30 packages. Importing the server went
from about 3.2 s to 1.1 s. A cold CLI command on the reference machine, two
indexes of 523 and 279 MB open, went from 32 to 43 s down to 16 to 17 s; the
first search after `lynx serve` starts from 7.8 s to 3.9 s. Indexing
throughput is unchanged (14.2 chunks/s against 13.3 on 200 real C# chunks;
batches are now grouped by length so padding is not wasted).

Model files follow the runtime. `lynx manager install --model` downloads what
the runtime opens, the ONNX graph plus the tokenizer (134 MB for bge-small
instead of the 267 MB of the two PyTorch formats), and `lynx manager doctor`
reports a cached model as incomplete when the graph is missing. A cache
filled by an older install holds the PyTorch weights only, so the first run
after upgrading stays online long enough to fetch the graph into the same
snapshot, then goes back offline as before. The archive on the `models`
GitHub release still carries the old weights until the next tagged release
republishes it; `--from-archive` against that URL needs the new archive
first. Any HuggingFace repo that ships `onnx/model.onnx` can be configured
(the BGE family, `sentence-transformers/all-MiniLM-L6-v2`, the ms-marco
cross-encoders); a model without one is refused with a message that names
the file and the export command.

The MCP handshake no longer waits for the indexes. `lynx serve` used to open
every index (model, Chroma stores, BM25 corpora, graph, the integrity probe)
before reading the first byte of JSON-RPC: about 10 seconds with two indexes,
and on a first run the whole build. Claude Code gives a server 30 seconds to
answer `initialize` and then reports it dead, so a first run inside a client
could not work at all. Now the tools are registered against a preview of the
sources built from the config alone (type, path, graph and git flags are all
there), the transport starts within a second, and the real manager is built
in a background thread. A tool call that arrives early waits up to 20 seconds
(`LYNX_TOOL_WAIT_SECONDS`), then answers with the loading state and asks to
be retried; a failed load answers with the error and a `doctor` hint instead
of a server that exits before saying why. The handshake instructions carry
one sentence about this so the model knows to retry. Measured from process
spawn with two indexes of 523 and 279 MB: `initialize` answered in 1.4 s
against 10.6 s before. Those 10.6 s were the indexes themselves: 3.5 s the
out-of-process integrity probe, 1.3 s the graph, 1.1 s the model session, the
rest Chroma and the BM25 corpus. The same two indexes once took 86 s to open
on a cold machine during these measurements, which no handshake would have
survived; the client saw the 1.4 s handshake and a loading message on each
call until the first search came back. Stdout stays pointed at stderr for the
life of the process and the transport writes to the saved real descriptor, so
a library that prints during the background load cannot corrupt the channel.

The README leads with what grep cannot answer (structure, behaviour, docs past
the model's cutoff) instead of competing with it on ranking, publishes hit@5
next to hit@1 and MRR for all three benchmarks, including the two where grep
ranks higher, and moves the money figures below the measurements. The savings
calculator (CLI and page) counts the saved round trip's context at the
prompt-cache price (`--cache-read-discount`, default 10%) instead of the full
input price; for 25 engineers that is $17,000 to $27,000 a year on a $10
model, where the old chart said $85,000 to $95,000. The three SQL integrations
and the offline-install recipes moved to their own pages. The package and
registry descriptions start with "LynxMCP", the name that search engines can
tell apart from ByteDance's Lynx and from the lynx browser. The Docker image
serves the `full` tool profile, so directory scanners list all 17 tools.

## 1.8.0 — 2026-09-03

### Added
- **Several sessions can share one index.** Two editor windows, or an editor
  plus `lynx manager ui`, used to be one working session and one dead one: the
  second was told the source was "in use by another Lynx process" and got no
  usable sources at all. That was built on a wrong premise. ChromaDB was
  believed to block a second reader forever; measured against a live
  `lynx serve` on 1.5.9, a second process opens the store, counts it and runs a
  KNN query in about a second, and both stay healthy. Only writing is
  exclusive, so that is what is now serialised: the first process to open a
  source claims it in `owner.json` and runs the watcher, and the rest open the
  same store as followers. A claim left by a killed process is taken over on
  the next start.

  A follower reads and never writes. `lynx build`, `lynx reset` and the
  dashboard's build button refuse there, naming the process that owns the
  index, and its watcher stays quiet so two processes never index the same
  file. Its searches stay current on their own: `count()` was always live
  across processes, but retrieval answers from state loaded at open time, so a
  follower checks the store's write log before searching and, when the owner
  has written, reloads just the files that changed. Measured on a 270MB index:
  reattaching to the store costs 0.13s, against 4.4s to rebuild the whole BM25
  corpus, which is why the delta is applied file by file. The graph layer
  follows the same way, reloading when the owner rewrites it.

  Closing the window that happened to start first does not freeze the others
  either. The owner refreshes its claim while it runs, so a follower can tell
  an index that is being written from one whose owner is gone, and takes over
  on its next search: it starts watching the files and scans for what changed
  while nobody was listening, since a watcher only reports what happens after
  it starts. Three cases land there: an owner that exited, one that was
  killed, and one that stopped running without exiting, which is what a
  sleeping laptop looks like from the outside. That last one is why a claim is
  kept fresh rather than trusted for the life of a pid: pids get recycled, and
  the woken-up owner is told to stand down instead of becoming a second
  writer.

  Verified with the sessions actually running in parallel, not by reasoning
  about them: an owner and two followers searching in a loop through 7 rounds
  of live edits, the owner killed mid-stream, 0 errors in 70 searches, every
  new symbol found by both followers within 4 seconds, one claim holder
  throughout and an undrifted index at the end. Also: 8 processes claiming one
  store at the same instant yield exactly one owner; `lynx build` and
  `lynx reset` refuse against a store someone else owns and leave it
  byte-identical; and the web UI opened next to a running server searches
  normally while its build button answers 409 naming the owner.

- **`export_graph` and the feedback log reached the web UI.** They were the
  last two tools with a CLI and an MCP entry point but no place in the
  browser — backwards in both cases: `export_graph` produces a self-contained
  HTML page whose whole purpose is to be opened and attached to a PR, and the
  feedback log's only audience is the person tuning the index. The playground
  now exports a view and links straight to it (served by `GET
  /api/reports/<name>`, validated against a strict filename pattern *and* a
  resolved-parent check), and the dashboard shows the latest reports agents
  filed when the index couldn't answer them.
- **The web UI's playground covers the whole tool surface too.** It had drifted
  to nine tools of seventeen, with a Graph tab wired to three of the ten graph
  operations — no way to ask the UI for subclasses, imports, a shortest path or
  the bridge edges. Those three hand-written panels are now one `graph_query`
  form with an operation selector (the same consolidation the MCP tool and the
  CLI already made, calling the same dispatcher), and `deep_search`,
  `describe_symbol`, `impact`, `module_summary` and `repo_overview` gained
  panels of their own. The composed tools render their `_format_*` text
  verbatim: a playground exists to show what your agent receives, and a
  prettier UI-only view would display something no agent ever sees.
- **Every MCP tool now has a CLI command.** Ten of the seventeen were
  agent-only: `deep_search`, the four `find_*`, `describe_symbol`, `impact`,
  `repo_overview`, `module_summary`, `search_diff`. Anyone without an MCP
  client — and every script — was locked out of most of what Lynx does. The
  names map 1:1 (`find_definition` → `lynx find-definition`), so the tool
  table doubles as the CLI reference, and each command calls the same
  `manager` method and the same renderer as the tool: the terminal shows
  byte-for-byte what the model receives. `--json` on all of them emits the
  raw structure instead.
- **`lynx search` gained the last three things the tool had and it didn't**:
  `--outline` (signatures only, for triaging a broad query), a repeatable
  `--source` to fuse a subset at request time, and `--json`.
- **`lynx source add` / `lynx source remove` — sources without a browser.**
  Adding a source was a web-UI-only operation: `lynx manager init` wrote a
  config with `sources: {}` and told you to open LynxManager, so a headless
  server or a CI job had no option but hand-written JSON. The two commands
  write through the same validate-then-write path as `POST/DELETE
  /api/sources` (extracted to `manager/sources.py`, which no longer imports
  FastAPI), so both front-ends accept and reject exactly the same configs and
  both leave the file untouched — plus a `.bak` — when validation fails.
  Typed flags per source type, `--block` for anything they don't cover, folder
  extension auto-detection when `--ext` is omitted, `--build` to index
  immediately, `--purge` to reclaim the index on removal, and `--json` output
  for scripts.
- **`lynx graph query` — the knowledge graph from a terminal.** The graph was
  asymmetric: agents got all ten `graph_query` operations over MCP, humans got
  `build`, `status`, and `export` with no way to actually ask anything. All ten
  are now on the CLI (`--op callers|callees|subclasses|superclasses|imports|
  neighbors|shortest_path|overview|surprising_connections|status`). The
  rendering moved to `graph/dispatch.py` and both front-ends call it, so the
  terminal shows byte-for-byte what the model receives. `--json` emits the raw
  edges/path/counters instead, for scripts: once the arguments parse, exactly
  one object on stdout in every case — success, usage error, or unexpected
  runtime failure (the CLI wraps the dispatch in the same try/except the MCP
  tool always had) — each carrying `ok`, `operation` and `source` (`null`
  when resolving the source was itself the failure). A caller never has to
  tell "no output" apart from "output saying no".
- **An empty graph answer now says which kind of empty it is.** "No callers"
  and "no such symbol" printed the same `(no results)`, and the tool
  description could only suggest running `status` to guess which. When an
  operation comes back empty the graph is asked whether it knows the name at
  all, and says so — in the text the model reads and as `"matched": false` in
  the JSON. Reported only when the check is possible, so a half-built graph
  can never produce a wrong "that doesn't exist".
- **Coverage drift detection and a surgical heal** (`integrity.inspect_coverage`
  / `heal_coverage`, surfaced as `doctor --heal-coverage SOURCE`). Read-only
  plain-sqlite audit — never a Chroma client, which would contend with a live
  `serve` — that catches the two silent ways an index stops matching reality:
  files the SHA cache claims are indexed but that hold no chunks, and chunks
  whose vector never reached the HNSW segment. Both heal the same way: forget
  just those files so the next pass re-indexes them, no full rebuild. The
  per-source doctor check now fails on either, naming the `Error querying knn`
  symptom so the diagnosis doesn't have to be re-derived. Chunks the WAL can
  still deliver on its own (the tail of a fresh build, waiting on Chroma's
  persistence threshold) are explicitly NOT counted as damage — otherwise the
  check would fire after every build and be trained away.
- **A Restart button in the LynxManager sidebar.** Always-available action that
  drops the cached manager so every source is reloaded — indexes re-opened,
  watchers restarted, integrity re-probed (the same `/api/manager/reload` the
  "index not verified" / "in use" cards already used, now surfaced globally).
  Handy right after a `doctor --heal-wal` or a manual fix, without hunting for a
  problem card. Confirms first; nothing is lost.
- **Scope a search to a subset of sources, per request.** `search` and
  `deep_search` already took ONE `source` (or omitted it for all). Now `source`
  also accepts a LIST of names — the rankings are fused (RRF) across just those
  sources, each hit still tagged with its origin. So a query can be contextualized
  to certain data sources at request time (`source=["skelforge"]` or
  `["skelforge", "framework"]`) with no server restart and no config change; omit
  for all, one name for a single source. `SourceManager.search_all` /
  `deep_search_all` gained an `only=` filter that backs it.
- **Coral: the code graph is now queryable as six functions.** On top of the
  merged `lynx.search` source, `lynx.callers` / `callees` / `subclasses` /
  `superclasses` / `imports` / `neighbors(symbol => '...')` expose the
  GET /api/v1/graph endpoint with a shared flat-edge column set
  (`from_*` → `to_*` with relation + call site), so a search hit's symbol can
  be pivoted to its structural blast radius and JOINed with live data.
  Canonical spec in `integrations/coral/manifest.yaml`; a mirror restyled
  to Coral's house conventions is staged for the community PR.
- **`lynx manager doctor` detects a wedged index WAL — and `--heal-wal` fixes
  it without a rebuild.** A process killed mid-write (Ctrl-C during watcher
  indexing, a killed serve) can leave pending rows in chroma's
  `embeddings_queue` with a segment behind the queue tail; chromadb (observed
  on 1.5.9) then deadlocks forever on every open, which surfaced as integrity
  probe timeouts at every start. The per-source doctor check now reads the
  fingerprint via plain read-only sqlite (never a Chroma client — that's what
  hangs) and reports "index WAL is wedged" with the remedy. `lynx manager
  doctor --heal-wal SOURCE` purges the stuck writes and stale write locks,
  drops the affected files from the SHA cache so the next build / watcher
  pass re-indexes them (nothing silently lost), and proves the result with
  the out-of-process probe. Refuses while any Lynx process holds the store.

### Changed
- **The two biggest modules were split along their seams.** `cli.py` went
  1657 → 1010 lines (the argparse declarations are now `cli_parser.py`,
  which executes nothing), and `manager/ui/routes.py` 1566 → 1036: the HTML
  fragments moved to `ui/render.py` (pure functions over data) and the
  per-tool panels to `ui/playground.py` — the surface that grows every time
  a tool is added, and the reason routes.py kept swelling. Every name is
  re-exported from its old home, so callers and tests are untouched.
- **Three UI endpoints were removed**, folded into the single `graph_query`
  form: `POST /api/playground/get_callers`, `.../get_callees` and
  `.../architectural_overview` now return 404. They are internal to the web
  UI — nothing outside it is expected to call them — but anything that did
  should post to `/api/playground/graph_query` with `operation=` instead.
- **`lynx search` renders through the same formatter as the `search` tool.**
  It used to carry its own thinner version — file, score, six raw lines — so
  the terminal showed strictly less than the model got: no symbol names, no
  line ranges to cite. One renderer now, and `--json` covers anything that
  was parsing the old text.

### Fixed
- **A fresh install could not start the server.** The `mcp` dependency had no
  version bound. The SDK's 2.0 renamed `FastMCP` to `MCPServer` and removed
  `mcp.server.fastmcp`, so anyone installing today resolved 2.x and got a
  `lynx serve` that failed on import, while machines with an older `mcp`
  already resolved kept working and showed nothing. Pinned to `mcp<2` until
  the port to `MCPServer` lands. Verified on a clean Python 3.10 install with
  the newest of everything else: 430 tests pass and `lynx serve` imports.
- **On macOS and Linux, nothing stopped a reset from wiping a store in use.**
  The guard in front of `lynx reset`, `--heal-wal` and `--heal-coverage` asks
  whether another process holds the index, and that question could only be
  answered on Windows, where the OS names the processes holding a file. Off
  Windows it returned "no idea" and the guards let the delete through, which is
  how a running server could have its files pulled out from under it. The
  ownership claim answers it everywhere now.
- **`lynx source remove --purge` had the same hole, one layer down.** It never
  asked at all: it simply deleted, and relied on Windows refusing to unlink
  files a live process holds open. Everywhere else the delete succeeded and
  left a running session reading an index that was no longer there. It now
  asks first and names the session that has it.
- **`doctor` called healthy indexes "wedged".** The WAL fingerprint it used
  (queued writes older than 10 minutes with a segment behind them) is the
  normal end state of any build whose size is not a multiple of the HNSW
  sync threshold: the vector segment replays that tail on its next open and
  the rows sit there while the index is idle. Two healthy, searchable
  indexes were reported wedged for 16 days, with `--heal-wal` as the advice,
  and running it would have purged the very rows the segment still needed
  (recoverable, but only through a re-index). The check now runs the
  out-of-process integrity probe and lets it decide: a healthy store answers
  in about a second, a wedged one never does, so the fingerprint only
  explains a timeout. Same confirmation before `--heal-wal`, which now
  refuses an index that answers. `doctor` also reports a corrupt or
  unverifiable index itself, and `affected_files` names only the files of
  rows no segment has applied, not everything still in the log.
- **A BOM in the feedback log silently deleted a report.** `load_feedback`
  skips malformed lines on purpose — the log is append-only and a
  half-written final line shouldn't break the reader — but read as plain
  utf-8, a BOM makes line 1 unparseable, so the oldest report vanished with
  no message. Lynx writes the file without one; opening it in a Windows
  editor and saving is enough. Now read as utf-8-sig, like the config.
- **The playground's `search_diff` panel never showed a result.** It read
  `payload["results"]`; the backend returns its hits under `hits`, a key
  mismatch that rendered "No results." on every run, including the runs that
  found something. The panel also dropped the backend's `note`, which is how
  an empty answer explains itself ("No files added/modified vs 'main'"),
  leaving the user staring at a blank box. Nothing caught either, because the
  only test touching the playground lives in a file excluded from collection —
  there is now a pytest-collected one.
- **Two front-ends could silently drop each other's config edits.** Both
  read-modify-write the whole file, so `lynx source add` in a terminal while
  the web UI was open was enough for one write to lose the other. The
  mutations now hold an exclusive lockfile; a lock left behind by a killed
  process is broken once it is stale, and failing to acquire proceeds
  unlocked rather than refusing to work.
- **An empty graph answer no longer claims a symbol exists because a file
  shares its name.** The seed check counted file nodes for every operation —
  meant as the cautious choice, since it can only stay quiet — but it
  silenced the hint exactly where it earns its keep: `--op callers --symbol
  main` reported a match on a graph with no function called `main`, because
  `main.py` exists. It now mirrors each operation's own resolution, and only
  `imports` accepts a file path.
- **`lynx source add` explains a validation failure the way `remove` does.**
  The loader validates the whole file, so the rejection is often about a
  sibling source; `remove` said so and `add` printed a bare "Schema
  validation failed".
- **`source add --json` reports the block that was written**, defaults
  included, instead of echoing the input — a script reading
  `.block.ignored_path_fragments` was told there were none while the file on
  disk had the full list. A failed `--build` now also carries an `error`
  field: it was the last `ok: false` with nothing explaining it.
- **A purge that couldn't happen was reported as if it had.** `remove
  --purge` (and `DELETE /api/sources/{name}?purge=true`) dropped the config
  entry first and deleted the index second. On Windows a running `lynx serve`
  holds the ChromaDB file open, so `rmtree` raised WinError 32 with the entry
  already gone: exit code 0, "removed", index still on disk — and no way to
  retry the purge, since the source was no longer in the config to remove.
  The order is now validate → purge → write, each step gated on the previous:
  a locked index aborts with nothing changed (same command works once the
  lock is released), and a config rejected by validation can no longer cost
  the index either — the loader validates the whole file, so a broken
  *sibling* source (a moved folder) used to be enough to delete the index
  and then refuse the write, stranding the operation halfway.
- **The web UI could not purge a source it had ever loaded.** The UI process
  itself holds the source's ChromaDB files open — chromadb caches the client
  system per storage path, so even dropping every Python reference leaves
  `data_level0.bin` locked (measured: WinError 32 on rmtree from the same
  process). `DELETE ?purge=true` therefore failed against the UI's own lock,
  with a message advising the user to stop whatever holds the index — i.e.
  the page they were clicking in. The route (and `/api/manager/reload`) now
  disposes the manager AND clears chromadb's system cache before deleting;
  the manager rebuilds lazily on the next request.
- **`DELETE /api/sources/{name}?purge=true` deleted the wrong directory when
  the UI wasn't launched from the config's folder.** A relative `storage_path`
  was resolved against the process CWD, while the loader resolves it against
  the config file's directory. The purge then found nothing to remove and
  reported success, leaving the index on disk. Now resolved the same way the
  loader does.
- **A config.json saved with a UTF-8 BOM no longer fails to load.** Read as
  plain utf-8 the BOM survives into the string and `json.loads` rejects it
  with "Unexpected UTF-8 BOM ... line 1 column 1" — a file that looks correct
  in any editor, refusing to load. PowerShell's `Out-File -Encoding utf8`
  writes one by default, so on Windows this was one hand-edit away. The
  loader, the UI's config editor, and the source CRUD now decode utf-8-sig
  (identical to utf-8 when there is no BOM).
- **config.json is now written atomically.** A plain `write_text` that died
  midway — full disk, killed process — left a truncated config that nothing
  could load, with the `.bak` written moments earlier as the only copy. The
  content now goes to a sibling tempfile and is moved into place with
  `os.replace`, and the backup is a byte copy so a BOM or unusual encoding
  survives a rollback.
- **A segment stranded on purged rows is now moved forward, not left waiting.**
  A segment consumes the WAL from its own watermark; once the rows it still
  needs are purged they are gone, so it never advances — and every later write
  waits on it. Measured in that state: 5 chunks/minute, vector index frozen,
  process at zero CPU. `heal_wal` now fast-forwards lagging segments (empty
  queue only) and the chunks whose vectors were lost are re-queued by
  `--heal-coverage` rather than silently accepted as missing.
- **A healed WAL no longer turns the store into a write-only void.** `heal_wal`
  purged `embeddings_queue` and left the segments' `max_seq_id` untouched. But
  `embeddings_queue.seq_id` is an INTEGER PRIMARY KEY with **no AUTOINCREMENT**:
  an emptied queue restarts numbering at 1, so segments parked at (say) 64111
  discard every new row as already-applied. The store then accepts writes and
  stores none — no exception, `count()` unchanged, index frozen forever while
  every log line reports success. The heal now realigns the watermarks (only
  with an empty queue: with rows pending, lowering them would replay and
  duplicate), `inspect_wal` reports the state as `discarding_writes`, and the
  doctor fails on it. Found in the field: an index stuck at 2026-07-03 for 25
  days, with a live watcher logging successful re-indexes into nothing.
- **A failed insert no longer marks a file as indexed — forever.** During a
  delta build, the SHA cache was written for every file in the batch, including
  those whose `insert_nodes` had raised (logged, then treated as done) or that
  produced no chunks. Since `_partition_files` trusts that cache to classify a
  file *unchanged*, those files were skipped on every later pass: permanently
  absent from the index, never retried, with a perfectly healthy `count()` and
  no error anywhere. Now only files whose chunks actually landed are recorded
  (with a new `chunks` count), and a failed one has its entry dropped so the
  next pass retries it. Found in the field: 202 files of a 1944-file source,
  invisible to search for 25 days.
- **Writes are now proven, not assumed.** An insert can fail without raising
  (see the watermark bug above), so recording the SHA on a call that "didn't
  throw" was enough to retire a file from every future pass. Both the batch
  build and the watcher path now compare the collection count before and after
  and refuse to update the cache unless it grew by exactly the number of chunks
  sent — a mismatch logs what happened and leaves the files queued.
- **`_drop_from_file_hashes` reached nothing.** It iterated the top level of
  `file_hashes.json`, which under the current schema holds only
  `schema_version` / `config_snapshot` / `files` — so every heal built on it
  (including `doctor --heal-wal`) reported success while dropping zero
  entries. It now edits the `files` mapping, legacy flat caches included.
- **The integrity probe never exercised search.** It opened the collection and
  called `count()` + `get()`, both of which read sqlite and stay green while
  the vector segment is out of step with the metadata — exactly the state where
  every query dies with `Error querying knn` / `Error finding id`. The probe now
  runs a KNN query too, so a search-dead index is reported as corrupt instead of
  healthy.
- **A timed-out integrity probe is no longer branded a corrupt index.** A
  slow/large index, an AV-scanned disk, or another process holding the store
  made the probe time out, and Lynx reported `corrupt` — telling the user to
  reset a healthy index. Only a non-zero probe exit (caught exception or
  native crash) counts as corruption now; a timeout retries once with a
  larger budget and then reports a new `unverified` status: amber "index not
  verified" card with a Re-check button instead of the red corrupt/reset
  card, mirrored in CLI status and build.
- **A store held by another Lynx process no longer alarms the dashboard.**
  While `lynx serve` holds a source's ChromaDB store, an out-of-process
  integrity probe can never finish (chroma's Rust core blocks a second
  process's first read forever), so every manager-UI start burned the full
  60s+120s timeout and showed a scary warning about a healthy index. On
  Windows, `check_index` now detects who holds the store up front (zero-sharing
  open + Restart Manager attribution) and returns a new `in_use` status
  instantly: the dashboard shows a neutral "in use by another Lynx process"
  card (Re-check button, deliberately no Reset), and CLI status/build mirror
  the message. A store held only by the current process (e.g. after a manager
  reload, via chroma's client cache) is reported healthy instead of probed
  into a guaranteed deadlock. POSIX keeps the previous probe-and-timeout
  behavior. A transient non-Lynx holder (AV scan, backup tool) is filtered by
  a short re-check before the store is declared in use.
- **Orphaned integrity probes no longer wedge a store.** Windows doesn't reap
  children when the parent dies (Ctrl-C, closed terminal), and an orphaned
  probe blocked on another process's store lock lingered forever holding a
  handle — making every later probe time out too. The probe child now
  hard-exits on its own once the parent's timeout budget is over.
- **Reset refuses to wipe a store another process holds open.** The wipe path
  deleted the unlocked files first and then failed on the held sqlite,
  half-destroying a healthy (just busy) index. `reset_source` now raises a
  clear error up front; stop the other Lynx process and retry.
- **The dashboard sidebar shows the real version instead of "vdev".** The
  lookup asked importlib.metadata for the distribution `lynx`, but the
  package is published as `lynx-mcp`, so it always fell back to "dev".

## 1.7.5 — 2026-06-27

### Changed
- **Model download fetches only the weights Lynx uses.** `download_model` now
  passes `ignore_patterns` to `snapshot_download`, skipping the ONNX (incl.
  quantized / graph-optimized), TensorFlow, Flax and OpenVINO copies that HF
  repos like `bge-small` ship alongside the Torch weights. Lynx loads via
  sentence-transformers (embedding *and* reranker), which needs only
  `*.safetensors` / `pytorch_model.bin` + configs/tokenizer. Result: a much
  faster download, a smaller published archive, and a quicker `--from-archive`
  fetch for end users. Both Torch formats are kept (some models, e.g. the
  reranker, ship only `pytorch_model.bin`).

### Added
- **Automatic GitHub fallback for the model download.** When `download_model`
  can't reach huggingface.co (firewall, proxy, DNS, flaky network), it now
  automatically pulls the model from the project's GitHub Release archive
  instead of failing — no manual `--from-archive` step needed. Best-effort: the
  fallback only has the models the maintainer published (the default embedding /
  reranker); an unmatched model yields a clear combined error. Forks / private
  mirrors can repoint the host with `LYNX_MODEL_ARCHIVE_BASE_URL`. This covers
  the installer first-run path (`manager init` → `download_model`).
- **`publish-model.yml` now triggers on the `v*` tag push** instead of
  `release: published` — a Release created by `release.yml` with the default
  `GITHUB_TOKEN` does not fire `release` workflows (GitHub recursion guard), so
  the model archive never got published automatically. A human-pushed tag does.
- **`publish-model.yml` re-uploads when the archive size changes.** It rebuilds
  the archive every run and compares its size to the published asset: same size
  → skip the upload, different/new/missing → (re)publish. So a slimmer rebuild
  or an updated model is picked up automatically, with no manual asset deletion.

## 1.7.4 — 2026-06-27

Installation hardening: make the first-run HuggingFace model download survive
restricted networks, and give clear, actionable errors when it can't.

### Added
- **Offline / air-gapped model install.** `lynx manager install --export-archive
  PATH` zips a cached model in the HF hub layout, and `--from-archive PATH_OR_URL`
  imports it into the cache on a machine that can't reach huggingface.co. The URL
  form accepts any direct, unauthenticated download (a public GitHub Release asset
  is the easiest); it rejects an HTML body — an auth page or a Google Drive
  ">100MB virus-scan" interstitial — with a clear message instead of a cryptic
  `BadZipFile`. Path-traversal-safe extraction (zip + tar, `filter="data"` on
  3.12+). Pinned by `tests/test_manager_install.py`.
- **Publish-model GitHub Action.** A workflow downloads the embedding model on
  GitHub's runners and attaches it to a stable `models` Release, so users behind
  a firewall can `--from-archive` it from this repo. Runs on every release and is
  idempotent — the asset name encodes the model, so an unchanged model is skipped
  (no redundant ~130MB re-upload) while a different model is published.
- **`lynx manager doctor` HF endpoint check.** Reports the active hub cache
  directory and whether an `HF_ENDPOINT` mirror is configured.

### Fixed
- **"Model not downloaded" no longer masquerades as a corrupt index.** When a
  source failed to start because the embedding/reranker model wasn't in the cache
  (and couldn't be fetched), Lynx used to suggest `lynx reset` — which rebuilds
  but still needs the model, a loop. It now detects the missing-model case and
  points at the real fix (`--model`, an `HF_ENDPOINT` mirror, or `--from-archive`).
  Pinned by `tests/test_source_manager_model_missing.py`.
- **`doctor` probes the cache the runtime actually uses.** The model-cache check
  now honors `HF_HUB_CACHE` / `HF_HOME` instead of a hardcoded `~/.cache`.

## 1.7.3 — 2026-06-22

Stabilization & bug-fixing release (plus one small tool that closes a loop).

### Added
- **`lynx manager feedback`** — read & summarize the local feedback log. The
  MCP `feedback` tool appends to `<storage>/_feedback/feedback.jsonl` when an
  agent can't find what it needs, but nothing read it back. This command turns
  those reports into a summary (total, time span, the recent "trying to do /
  tried / stuck" entries, and the sources configured at report time). Read-only,
  100% local, and it reads only `storage_path` from the config — so it works
  even if a source folder has since moved. `--limit N`, `--json`. Pinned by
  `tests/test_manager_feedback.py`.

### Fixed
- **`ignored_path_fragments` now excludes files from the vector index too.**
  Previously the file watcher and the graph layer honored the configured ignore
  fragments (`node_modules`, `dist`, `build`, …) but the vector-index build did
  not — so vendored / build dirs were embedded into ChromaDB and BM25 and leaked
  into search results, while the watcher never refreshed them and the graph
  excluded them (the three layers disagreed). The index build now applies the
  same exclusion. New indexes exclude these files immediately. An existing index
  that already embedded them clears them on its **next rebuild** — a `lynx build`
  / `update_source_index(force=true)`, or an update triggered by a new git commit
  (the SHA partitioner sees the now-ignored files leave the candidate set and
  drops their chunks). A plain restart reuses the populated index as-is, so a
  one-time rebuild is the way to clean a pre-existing index. No drift warning is
  emitted (it would fire for every user, including those with no ignore
  fragments).
- **Multi-segment ignore fragments now match on Windows.** Fragment matching is
  separator-agnostic (both path and fragment normalized to `/`), so a fragment
  like `src/generated` is excluded regardless of OS path separators.

### Changed
- **Cross-source reranking in `search_all`.** When the cross-encoder reranker is
  enabled, a search across all sources (the no-`source` path) now reorders the
  *merged* candidate pool by content relevance instead of fusing per-source
  rankings by position only — so the final cross-source order is content-aware,
  not just RRF. Disabled by default, so existing behavior (pure RRF) is
  unchanged. Falls back to RRF if the reranker can't load or errors.
- **BM25: incremental cache on watcher edits.** Editing one file used to
  invalidate the entire BM25 index, so the next search re-read the whole
  collection from ChromaDB and re-ran the (regex-heavy) code tokenizer over
  every chunk. The cache is now refreshed per file — only the edited file's
  chunks are dropped and re-tokenized, and the BM25Okapi is rebuilt from the
  in-memory corpus without touching the store. Bulk (re)builds and resets still
  do a full cold reload. Pinned by `tests/test_bm25_incremental.py`, which
  asserts the incremental cache equals a cold rebuild of the equivalent corpus
  (and that resets clear a warm cache).
- **Graph: incremental cross-file resolution on every save.** A watcher edit
  used to clear *all* resolved/ambiguous call & inheritance edges and re-resolve
  the entire raw-call set (O(total calls) per file change). `update_file` /
  `remove_file` now re-resolve only the symbol names actually touched by the
  edit (the symbols the file defines/removes plus the names it references). The
  result is byte-for-byte identical to a full rebuild — pinned by
  `tests/test_graph_incremental.py`, which asserts the incremental graph equals
  a from-scratch rebuild after add / edit / remove / ambiguous↔resolved
  transitions.
- **Single source of truth for the codebase file walk** (`lynx.fs_scan`). The
  vector index (`rag_manager`) and the graph layer (`graph.builder`) shared two
  near-identical copies of the directory walk + extension/ignore filtering; they
  now both delegate to one module, which is what fixes the divergence above.
  Ignored directories are pruned during the walk (faster on big trees).

### Internal
- **`server.py` slimmed down** (1472 → ~1110 lines): the pure output-formatting
  helpers moved to `lynx/_format.py`, leaving `server.py` to tool registration
  and bootstrap. The names are re-exported from `server.py`, so
  `lynx.server._format_*` still resolves (no caller/test churn).
- Removed an unreachable no-op branch in the cross-encoder reranker and refreshed
  its docstring.
- Added `tests/test_fs_scan.py` (pure, model-free) covering extension/hidden
  filtering, ignore-fragment exclusion, and separator-agnostic matching.

## 1.7.2 — 2026-06-22

### Added
- **Repository-comprehension tools** for AI agents working in an unfamiliar
  codebase:
  - **`repo_overview`** — the "what is this and where do I start" map: detected
    languages, frameworks, manifest files, likely entry points (main/CLI/server),
    and suggested build/test/run commands. Pure filesystem scan, no graph needed.
  - **`describe_symbol`** — DEFINITION + CALLED BY + CALLS + TESTS for a symbol
    in one call, instead of running find_definition + graph callers/callees +
    find_tests_for separately.
  - **`impact`** — the blast radius of a change: everything that reaches a symbol
    transitively through the call graph (with hop distance) + the tests to re-run.
  - **`module_summary`** (graph) — a file's public symbols + what it imports +
    which files depend on it.
- **`export_graph`** — render a slice of the code graph as a single, offline,
  self-contained file (inline SVG, no server, no CDN): a symbol's blast radius
  (`mode=symbol`) or a file as a hub (`mode=module`). MCP tool + CLI
  (`lynx graph export --symbol … | --module …`). Output goes to the new optional
  `reports_path` config key (default `<storage>/reports`).

### Changed
- **Retrieval deduplicates substantial duplicate chunks** — vendored or
  `build/`/`dist/` copies of the same file (identical bodies ≥120 chars) are
  collapsed to the highest-ranked hit. Short identical boilerplate is never
  merged, so distinct symbols that share a trivial body are all kept.
- **New codebase sources added via the UI get default ignore fragments**
  (`.git`, `.venv`, `node_modules`, `dist`, `build`, `target`, …) so build and
  dependency dirs don't pollute the index.

### Fixed
- **Graph: spurious self-loop call edges removed.** A member call `obj.Foo()`
  inside a method also named `Foo`, and a call between overloads (which collapse
  to one node id), both produced a bogus `Foo calls Foo` edge. Self-loop call
  edges are now dropped at both the extractor and the cross-file resolver.
- **Manager UI folder picker — Windows drive paths hardened end-to-end.** The
  backend now re-normalizes every drive shape (`C:`, `C:Users`, `/C:`, `//C:`,
  `\\C:`) to a real absolute path before resolving, so a corrupted breadcrumb
  segment can no longer send a dead path.

## 1.7.1 — 2026-06-21

### Added
- **Token savings, in money.** `benchmarks/savings_calculator.py` translates the
  measured per-query token saving into the monthly/yearly API bill it removes,
  for **all three** benchmarked codebases (Django/Python −58%, Json.NET/C# −77%,
  Guava/Java −86%) × every flagship model, at 2026-06 input prices —
  **Claude Fable 5** ($10/1M),
  **Claude Opus 4.8** ($5/1M), **GPT-5.5** ($5/1M), and more. Reports two figures
  per codebase: a fully-measured *floor* (tool-output delta only, zero
  assumptions) and a *realistic* figure that adds one eliminated grep round-trip
  re-billing a configurable context window. Measured deltas live in
  `benchmarks/measured.json`. Parametric by team size and usage (`--devs`,
  `--queries-per-dev-day`, `--avg-context-tokens`); emits a deterministic SVG
  chart (`docs/img/cost_savings.svg`, grouped by model and codebase, no matplotlib).
- **Editable price config + interactive calculator.** Model prices now live in
  `benchmarks/pricing.json` (Claude Fable 5 / Opus 4.8 / Sonnet 4.6 / Haiku 4.5,
  GPT-5.5 / GPT-5.4) — edit a price or add a model and both the CLI and the new
  `benchmarks/savings_calculator.html` pick it up. The HTML page lets a user
  choose the **codebase and model** from drop-downs, **override the $/1M price
  live**, and set their own team size/usage with no file edits; it works offline
  (embedded fallback) and reads `pricing.json` + `measured.json` live when served.
  CLI gains `--model` and `--input-price` to mirror the same per-model override.

- **Multi-language benchmarks — C# and Java.** `run_benchmark.py` is now
  language-parametric (target `ext` / `name` / `ref` in the tasks file; separate
  `--results-json` / `--results-md` so each language writes its own artifacts).
  Added `benchmarks/tasks_jsonnet.json` (15 questions over `JamesNK/Newtonsoft.Json`)
  and `benchmarks/tasks_guava.json` (15 questions over `google/guava`), both with
  ground-truth verified against the source. Results:
  - **C# / Json.NET** (`RESULTS_csharp.md`): sparser-comment code — Lynx wins
    **every** metric (hit@1 47% vs 33%, MRR 0.58 vs 0.47, **−77%** tokens to answer).
  - **Java / Guava** (`RESULTS_java.md`): Guava's self-documenting class names are
    grep's *best* case, so grep out-ranks (hit@1 73% vs 60%) — yet tokens to answer
    still collapse **−86%** (5,892 → 807). The honest cross-language takeaway:
    ranking parity swings with how self-documenting the code is, but the **token
    cost drops 58–86% every time** (1 call vs match-lines + a follow-up read).

### Changed
- **README is now outcome-first.** New commercial hero leads with what Lynx
  *saves* — the −58% / 2.4× token deltas and the dollar figures they translate to
  at frontier API prices — before the feature list. The "How it works" ASCII
  pipeline is now a rendered **Mermaid** architecture diagram, and the benchmark
  section now carries the C# (Json.NET) result next to Django.

### Fixed
- **Manager UI folder picker corrupted Windows paths.** The breadcrumb rebuilt a
  Windows drive segment as `\C:` / `/C:` (and rendered the root as `//C:`), so
  clicking the drive — or a child under it — sent a non-existent path to the
  backend (e.g. `path does not exist: C:\build`). The breadcrumb now treats a
  leading drive letter as the drive root (`C:\`), accepts either separator, and
  emits real resolvable absolute paths. POSIX paths are unaffected.

## 1.7.0 — 2026-06-19

### Added
- **Outline search — signatures instead of bodies, for cheap triage.** Both the
  MCP `search(query, outline=true)` tool and the HTTP
  `GET /api/v1/search?view=outline` endpoint return each hit's `signature` +
  first `doc` line instead of its body, so an agent triages candidates and reads
  the full code of only the one it picks (via `find_definition` or the cited
  `file:line`). The tool/param descriptions, the MCP handshake instructions, and
  the `lynx://guide` all tell the agent **when** to use it. Derivation is shared
  (`lynx/outline.py`) so both surfaces produce identical signatures. Measured
  **~2.4× fewer tokens** for the search step on a public repo (psf/requests:
  ≈59% triage; ≈53% even after reading the one chosen body) — see
  `docs/OUTLINE.md` for the data + chart, `benchmarks/outline_tokens.py` to
  reproduce. Query-time transform (no reindex, no `CHUNKER_VERSION` bump, no LLM);
  composes with `format=ndjson`; the default search is unchanged, so Coral /
  DuckDB are unaffected.

## 1.6.0 — 2026-06-19

### Added
- **Objective-C AST chunking and code graph.** `.m` / `.mm` files (and `.h`
  headers that contain `@interface` / `@protocol`) are indexed as
  `@interface` / `@implementation` / `@protocol` methods with qualified symbol
  names, instead of falling back to plain-text windows. The **code knowledge
  graph** covers Objective-C too: `#import` edges, the message-send call graph
  (`[self foo]` resolves caller → callee), and inheritance (superclass
  `extends` + adopted-protocol `implements`). Shared `.h` headers still parse
  as C unless Objective-C markers are present. Adding a language is additive, so
  `CHUNKER_VERSION` is unchanged and existing indexes are **not** auto-flagged —
  to pick up Objective-C on an already-indexed source, run
  `lynx build --source <name>` once to re-chunk its `.m`/`.h` files.
- **`GET /api/v1/graph` endpoint.** Exposes the code knowledge graph (`callers`,
  `callees`, `subclasses`, `superclasses`, `imports`, `neighbors`) as flat JSON
  rows over the stable v1 API, so an external consumer (e.g. a Coral graph
  source) can pivot from a `lynx.search` hit's symbol to its structural blast
  radius and JOIN it with live data. Additive to v1; no reindex.
- **`format=ndjson` on the v1 GET endpoints** (`search`, `sources`, `graph`).
  Returns one JSON row per line so the results drop straight into DuckDB
  (`read_json_auto(..., format='newline_delimited')`), `jq` and pandas without
  unwrapping. Opt-in; the default wrapped JSON is unchanged. See `docs/DUCKDB.md`
  for join recipes.
- **PR impact-analysis GitHub Action** (`integrations/github-action/`). On every
  pull request it indexes the repo on the runner (model + index cached) and
  comments with the cross-file callers of the changed symbols (blast radius) and
  the semantically related code, via the `/api/v1` endpoints. Runs 100% on the
  runner; a self-hosted Lynx is the lighter option for big monorepos.
- **Four more languages for AST-aware chunking: Bash/Shell, SQL, Scala, Lua.**
  Lynx now parses **18+ languages** with tree-sitter — `.sh`/`.bash`, `.sql`,
  `.scala`/`.sc`, and `.lua` files are indexed as whole functions / DDL objects
  (`CREATE TABLE`/`FUNCTION`/`VIEW`) with qualified symbol names, instead of
  falling back to plain-text windows. Add the extensions to a source's
  `supported_extensions` to index them (the manager UI auto-detects them).
  The optional **code knowledge graph** also covers the new languages:
  **Scala** with full call + inheritance (`extends`/`with`) edges, **Lua** and
  **Bash** with the call graph between defined functions. (SQL is
  search/chunk-only — its DDL has no call/inheritance/import structure.)
- **Self-host Docker image.** A `Dockerfile` to run Lynx on your own server
  (CPU-only torch, ~2.7 GB); the indexed code and embeddings stay on the
  machine — "local" = local to your server, not a hosted service.

### Changed
- README/docs: the Coral section now leads with cross-source use cases
  (correlate code search with live GitHub/Sentry/… data). Added `ROADMAP.md`
  (items under evaluation + explicit non-goals) and a Glama directory listing.

## 1.5.1 — 2026-06-15

### Changed
- **Per-parameter descriptions on every MCP tool.** Each tool's JSON input
  schema now documents its parameters (`source`, `top_k`, `query`, `symbol`,
  filters, etc.) via `Annotated[..., Field(description=...)]`, which the prose
  tool description alone didn't provide. Improves the in-client parameter hints
  and tool-definition quality scoring. Tool behavior is unchanged.

## 1.5.0 — 2026-06-15

### Added
- **Batch search API** — `POST /api/v1/search` accepts
  `{"queries": [...], "source": ..., "top_k": ...}` and embeds every query in a
  single model call, returning ranked hits per query
  (`{"results": [{"query": ..., "hits": [...]}, ...]}`). Results match the
  single-query `GET` endpoint: both now route through one shared retrieval core,
  and the batched query embedding is self-checked against the single-query path
  (so it's correct for any embedding model, not just BGE). For multi-query
  consumers fanning a question across rows of another data source.
- **Coral Python toolkit** (`integrations/coral/toolkit.py`) — two thin,
  stdlib-only clients (`Lynx`, `Coral`) so you can compose row-driven search and
  joins in a few lines of Python. Coral's SQL can't drive `lynx.search` from
  another table's column (table-function args resolve to constants at plan
  time); these bricks let you do it: query Coral for the rows, batch them into
  Lynx. Ships with a runnable, credential-free demo. See `docs/CORAL.md`.

## 1.4.1 — 2026-06-15

### Fixed
- **Coral integration docs.** The README examples used
  `lynx.search(q => other_table.column)`, but Coral resolves table-function
  arguments to constants at plan time, so row-driven (per-row) search isn't
  possible — `lynx.search`'s query must be a literal. Rewrote the README around
  the verified-valid pattern: a literal search query exposed as a SQL table you
  filter, sort, and join with live data. (No code change; the `/api/v1` API and
  the `manifest.yaml` were already correct.)

## 1.4.0 — 2026-06-14

### Added
- **Stable local JSON API** (`GET /api/v1/search`, `GET /api/v1/sources`)
  served by `lynx manager ui` — versioned, additive-only, 127.0.0.1 only.
  The integration surface for external tools.
- **Coral source spec** (`integrations/coral/manifest.yaml`): use Lynx from
  [Coral](https://github.com/withcoral/coral) as a SQL schema — join semantic
  code search with live GitHub / Sentry / Linear data in one query, e.g.
  `SELECT file, symbol, score FROM lynx.search(q => '...') LIMIT 5`. Recipe and
  worked examples in `docs/CORAL.md` and the README.

This release also carries everything from 1.3.1 below (antivirus startup-crash
fix, corrupt-index detection + `lynx reset`, UI fixes).

## 1.3.1 — 2026-06-14

### Fixed
- **Windows startup crash with HTTPS-inspecting antivirus.** Avast/AVG (and
  similar) inject `SSLKEYLOGFILE` pointing at a device path into every process;
  Python's `ssl` opens it through OpenSSL's file BIO on first TLS use, which
  aborts the bundled interpreter (`no OPENSSL_Applink`) mid-startup with no
  traceback. Lynx now strips `SSLKEYLOGFILE` before any TLS use — it runs
  offline and never needs TLS key logging. Opt out with
  `LYNX_KEEP_SSLKEYLOGFILE=1`.
- **Long config paths no longer overflow the sidebar.** The footer shows the
  config filename, with the full path on hover.

### Added
- **Corrupt-index detection, recovery, and `lynx reset`.** A
  version-incompatible or truncated ChromaDB index used to break the dashboard
  silently (blank page) or crash the process. Lynx now probes each index in a
  subprocess before opening it, so even a segfaulting store can't take down the
  UI/MCP server; the dashboard shows the source as **corrupt** (other sources
  keep working) with a **Reset** button, and the CLI gains
  `lynx reset --source <name>` to wipe and rebuild from scratch. The index is
  disposable derived data — a reset rebuilds it from your files.

## 1.3.0 — 2026-06-12

### Added
- **Usage playbook in the MCP handshake.** The server now sends compact
  `instructions` at initialize (source catalog, how to phrase queries, score
  interpretation, escalation ladder) and exposes the full playbook as the
  `lynx://guide` resource. Every MCP client gets this automatically — no
  rules file installation required (rules files remain available for
  stronger steering).
- **Tool annotations.** All tools now carry MCP annotations
  (`readOnlyHint` / `destructiveHint` / `idempotentHint` / `openWorldHint`),
  so clients can auto-approve the read-only retrieval tools.
- **`feedback` tool.** When the agent can't find what it needs, it can file
  a structured report (what it tried, where it got stuck) BEFORE giving up.
  Reports are appended to `rag_storage/_feedback/feedback.jsonl` — 100%
  local, never uploaded — and give the index owner a concrete signal for
  tuning sources, filters, and chunking.

## 1.2.0 — 2026-06-12

### Added
- **JS rendering for webdoc sources (opt-in).** Set `render_js: true` on a
  webdoc source to crawl SPA / client-side-rendered docs sites through
  headless Chromium (Playwright): pages are extracted *after* client-side
  rendering, and link discovery sees the post-JS DOM. Install the browser
  dependency with `lynx manager install webdoc-js`. New per-source tunables:
  `render_wait_until` (`load`/`domcontentloaded`/`networkidle`) and
  `render_timeout_seconds`. The default install stays browser-free.
- `lynx manager install` now supports extras with post-install steps
  (used by `webdoc-js` to download the Chromium binary).

## 1.1.2 — 2026-06-12

### Added
- `server.json` + `mcp-name` ownership marker in the README for listing on
  the official MCP Registry (registry.modelcontextprotocol.io).

## 1.1.1 — 2026-06-12

### Changed
- Depend on `llama-index-core` instead of the `llama-index` meta-package:
  drops the unused OpenAI bundle (`openai`, `llama-index-llms-openai`,
  `llama-index-embeddings-openai`) from the install. Lynx never calls any
  remote LLM — the install log now reflects that.
- Install instructions point at the published PyPI package
  (`pipx install lynx-mcp`); native installers bootstrap from PyPI too.

## 1.1.0 — 2026-06-12

### Breaking
- **Fixed MCP tool surface.** Tools no longer multiply per source
  (`search_<name>`, `get_callers_<name>`, ...). The server now exposes a
  constant set of ~11 tools that take a `source` argument:
  `search`, `deep_search`, `graph_query`, `find_definition`, `find_usages`,
  `find_tests_for`, `find_similar`, `search_diff`, `list_sources`,
  `get_rag_status`, `update_source_index`. With three sources this drops
  the registered tool count from 50+ to 11 and keeps client context lean.
  `search_all_sources` / `deep_search_all_sources` are folded into
  `search` / `deep_search` (omit `source` to fan out). The ten graph tools
  are folded into `graph_query(operation=...)`.
  If you generated an AI rules file with an older version, regenerate it
  from the manager UI.
- **Distribution renamed to `lynx-mcp`** (the bare `lynx` name on PyPI
  belongs to an unrelated project). The import package and the `lynx` CLI
  are unchanged.

### Fixed
- First run on a clean machine no longer hangs: HuggingFace offline mode
  is now enabled only when the configured models are already in the local
  cache, instead of being hard-coded on (which blocked the initial model
  download unless you knew about `lynx manager install --model`).
- `lynx manager install <extra>` no longer runs `pip install lynx[...]`,
  which resolved against the unrelated `lynx` package on PyPI. It now
  installs the extra's actual requirements.

### Changed
- README rewritten as a concise overview; the full manual moved to
  `docs/GUIDE.md`.
- CI now runs the pytest suite on Linux + Windows across Python 3.10/3.13
  (previously only installers were built).
