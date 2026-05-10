"""Make `src/` importable when tests are run without `pip install -e .`.

This is a convenience for development: contributors can `python tests/test_X.py`
straight from a fresh clone. CI/PyPI users always go through the installed
package, where this shim is a no-op.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
