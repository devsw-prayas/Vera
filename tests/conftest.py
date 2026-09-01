"""Pytest config for the Vera test suite.

Puts build/<config>/ on sys.path so tests can `import _vera` without a pip
install. The suite is self-contained - no external repos.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# anaconda + pip-torch both ship an OpenMP runtime; without this they abort on import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_VERA_ROOT = Path(__file__).resolve().parents[1]

for _cfg in ("Release", "Debug"):
    _p = _VERA_ROOT / "build" / _cfg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
