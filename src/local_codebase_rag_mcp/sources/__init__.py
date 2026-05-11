"""Per-type source backends for the multi-source server.

Each source type (codebase / webdoc / pdf / ...) is implemented as a subclass
of `SourceBackend` registered in `SOURCE_BACKENDS`. The `SourceManager`
instantiates one backend per entry in `config.sources` at server boot and
dispatches search / build / status calls based on the source name.

Adding a new source type requires:
  1. Subclass `SourceBackend` in a new module under this package.
  2. Set `type_name` to the string used in `config.json` (`type: "..."`).
  3. Register the class in `SOURCE_BACKENDS` below.
  4. (Optional) Add type-specific config validation in `config.py`.
"""
from .base import SourceBackend
from .codebase import CodebaseBackend


SOURCE_BACKENDS = {
    CodebaseBackend.type_name: CodebaseBackend,
    # "webdoc": WebdocBackend,   # added in M2
    # "pdf":    PdfBackend,      # added in M3
}


__all__ = ["SourceBackend", "CodebaseBackend", "SOURCE_BACKENDS"]
