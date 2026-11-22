# Lynx

This project is indexed by Lynx, a local MCP server. Its sources:
`<project>` (our code) and `<library-docs>` (the docs of a library we use).

- To find where something happens, call `search` first with a plain
  description ("where the player picks a seat to enter") instead of grep or
  opening whole files. Read only the lines it points to.
- For a name you already know: `find_usages`, `find_definition`,
  `describe_symbol`.
- Who calls it, and what breaks if it changes: `graph_query` and `impact`.
- Before writing a new helper, call `find_similar` with the draft.
- Before calling a library API, search `<library-docs>`: the model's
  knowledge may predate the version this project uses.
- Exact strings (an error message, a config key) and files you already
  know: grep and read them directly.

The full playbook is the `lynx://guide` resource.
