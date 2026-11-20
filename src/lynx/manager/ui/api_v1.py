"""The OpenAPI document of the stable `/api/v1` surface.

Coral 1.x builds a Provider from an OpenAPI document: every operation becomes
a typed tool its agents call from TypeScript (`lynx.search.query`, `lynx.graph.query`).
The models below only describe the rows the routes in `routes.py` already
return. They go into the schema through `responses={200: ...}` and never run
on a response, so the JSON a caller gets is the same with or without them.

Only `/api/v1/*` is published. The app-wide `/openapi.json` stays off, so the
UI's own routes (config writes, source deletion, HTML partials) are never
offered to an agent as tools.

What Coral 1.10 requires of the document, read from its materializer
(`coral-catalog/src/openapi/materialize.rs`): OpenAPI 3.0 or 3.1, a non-empty
`info.title` and `info.version`, and exactly one absolute http(s) URL under
`servers`. An `operationId` with dots becomes a namespace (`graph.query` is
called as `lynx.graph.query`). GET tools start enabled; anything else waits
for approval.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

GRAPH_OPERATIONS = ("callers", "callees", "subclasses", "superclasses",
                    "imports", "neighbors")
SEARCH_VIEWS = ("full", "outline")


class SearchHit(BaseModel):
    """One search result. `view=full` carries `content`; `view=outline`
    carries `signature` and `doc` instead."""
    source: str = Field(description="Source the hit comes from.")
    file: str = Field(description="File name.")
    file_path: str = Field(description="Absolute path of the file.")
    rel_path: str = Field("", description="Path relative to the source root; empty for docs and PDFs.")
    symbol: str = Field(description="Qualified symbol, e.g. `Player.take_damage`; `<header>` or `<chunkN>` for non-symbol chunks.")
    kind: str = Field(description="AST node type of the chunk, or `text_window`.")
    language: str
    start_line: int
    end_line: int
    score: float
    mode_effective: str = Field(description="`dense` on a hybrid source means its lexical index failed to load.")
    content: Optional[str] = Field(None, description="The chunk text (view=full).")
    signature: Optional[str] = Field(None, description="Declaration line (view=outline).")
    doc: Optional[str] = Field(None, description="First doc-comment line (view=outline).")


class SearchResponse(BaseModel):
    results: list[SearchHit]


class BatchSearchResult(BaseModel):
    query: str
    hits: list[SearchHit]


class BatchSearchResponse(BaseModel):
    results: list[BatchSearchResult]


class GraphEdge(BaseModel):
    """One edge of the code graph, flattened: `from_*` -> `to_*`."""
    relation: str = Field(description="`calls`, `inherits`, `imports` or `contains`.")
    base_kind: str = Field(description="For `inherits`: `extends`, `implements` or `extends_or_implements`.")
    confidence: str = Field(description="`extracted` (same file), `resolved` or `ambiguous` (across files).")
    module: str = Field(description="For `imports`: the imported module or path.")
    from_symbol: str
    from_kind: str
    from_file: str
    from_rel_path: str = ""
    from_start_line: int
    from_end_line: int
    to_symbol: str
    to_kind: str
    to_file: str
    to_rel_path: str = ""
    to_start_line: int
    to_end_line: int
    call_site_file: str = Field(description="For `calls`: the file of the call.")
    call_site_rel_path: str = ""
    call_site_line: int = Field(description="For `calls`: the line of the call.")


class GraphEdgesResponse(BaseModel):
    results: list[GraphEdge]


class SourceInfo(BaseModel):
    name: str
    type: str = Field(description="`codebase`, `webdoc` or `pdf`.")
    location: str = Field(description="Folder path or site URL.")
    chunk_count: int
    last_update: str


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]


DOCUMENT_DESCRIPTION = (
    "Lynx: local code search and code graph over the sources configured in "
    "Lynx (codebases, library docs, PDFs). Read-only. Search returns ranked "
    "code with file, lines and symbol; the graph answers who calls a symbol, "
    "what it calls, and what inherits from it."
)


def build_openapi(routes, *, version: str, server_url: str) -> dict:
    """The OpenAPI document of the `/api/v1/*` routes among `routes`, with
    `server_url` as its one server (the address the caller reached)."""
    from fastapi.openapi.utils import get_openapi

    v1 = [r for r in routes
          if getattr(r, "path", "").startswith("/api/v1/")
          and getattr(r, "include_in_schema", False)]
    document = get_openapi(
        title="Lynx",
        version=version,
        description=DOCUMENT_DESCRIPTION,
        routes=v1,
        servers=[{"url": server_url}],
    )
    # "No credentials", said outright. Without the field Coral reads the
    # authentication as missing and makes the owner pick a sign-in method;
    # an empty list is `explicit_anonymous`, and it picks "No authentication"
    # itself (coral-catalog/src/openapi/configuration.rs).
    document["security"] = []
    return document
