"""Pytest configuration for the Vera test suite.

The fluorescence tests cross-check Vera against the DBR R&D Python oracle (a
deterministic, non-Monte-Carlo bispectral transport solver). That oracle lives
in a *separate* repo and is imported read-only:

    R&D/Inverse Spectral Rendering/           <- added to sys.path here
        src/__init__.py                        <- package `src`
        src/kernels.py    kernel_fluorescence(...)   -- the rank-1 operator matrix
        src/forward.py    neumann_forward(...)       -- L = L_e + T L_e + T^2 L_e + ...
        src/spectral_grid.py  make_grid(...)         -- the nu-tilde-uniform grid

We only ever import from `src.*` (the oracle's compute modules). We never import
its `tests.*` package, to avoid colliding with this repo's own `tests/`.

If the oracle repo is not found, the fluorescence-oracle tests are skipped with a
clear message rather than erroring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Anaconda base + pip-torch both ship an OpenMP runtime; without this they abort
# on import with "libiomp5md.dll already initialized". Safe for our CPU-only,
# single-threaded numeric use here.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Vera repo root (this file is <root>/tests/conftest.py).
_VERA_ROOT = Path(__file__).resolve().parents[1]

# The freshly built _vera extension (build/<config>/_vera*.pyd) so tests can
# `import _vera` without a pip install. Harmless if absent.
for _cfg in ("Release", "Debug"):
    _p = _VERA_ROOT / "build" / _cfg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The oracle repo sits alongside Vera under ".../Graphics Programming/".
#   .../Graphics Programming/Prototype/DBR-x-ReSTIR   (Vera)
#   .../Graphics Programming/R&D/Inverse Spectral Rendering   (oracle)
_ORACLE_ROOT = (
    _VERA_ROOT.parents[1] / "R&D" / "Inverse Spectral Rendering"
)

ORACLE_AVAILABLE = (_ORACLE_ROOT / "src" / "kernels.py").is_file()

if ORACLE_AVAILABLE and str(_ORACLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORACLE_ROOT))


def requires_oracle():
    """Decorator: skip a test when the DBR oracle repo is not reachable."""
    return pytest.mark.skipif(
        not ORACLE_AVAILABLE,
        reason=f"DBR oracle not found at {_ORACLE_ROOT}",
    )


@pytest.fixture(scope="session")
def oracle_root() -> Path:
    if not ORACLE_AVAILABLE:
        pytest.skip(f"DBR oracle not found at {_ORACLE_ROOT}")
    return _ORACLE_ROOT
