"""Unit checks on the DBR Python oracle's rank-1 fluorescence kernel itself —
the reference Vera's rendered output is validated against in
test_fluorescence_render.py.

Run:  pytest tests/test_fluorescence_oracle.py
Needs `torch` in the interpreter (use the conda `Spectral` env, or
`pip install torch` into the Vera venv).
"""
from __future__ import annotations

import pytest

from conftest import requires_oracle

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)


@requires_oracle()
def test_oracle_imports():
    """The three oracle entry points we depend on are importable."""
    from src.kernels import kernel_fluorescence  # noqa: F401
    from src.forward import neumann_forward  # noqa: F401
    from src.spectral_grid import make_grid  # noqa: F401


@requires_oracle()
def test_rank1_kernel_is_stokes_and_lower_triangular():
    """Φ shifts energy only to longer wavelengths (upper triangle ~ 0).

    Oracle convention: T[i, j] = QY · e(λ_i) · a(λ_j) · w_j, with e centred at
    λ_em, a centred at λ_ex, and λ_em > λ_ex (Stokes). Row i = output λ_i,
    column j = input λ_j, so energy moving to longer λ populates the lower
    triangle (i > j) and the strict upper triangle should be negligible.
    """
    from src.kernels import kernel_fluorescence
    from src.spectral_grid import make_grid

    grid = make_grid(lam_min=360.0, lam_max=700.0)
    T = kernel_fluorescence(
        grid.lam,
        lam_ex=380.0,
        lam_em=520.0,
        sigma_f=15.0,
        weights=grid.weights,
        quantum_yield=0.9,
    )
    assert T.shape == (grid.N, grid.N)

    lam = grid.lam
    upper = T[lam.unsqueeze(1) < lam.unsqueeze(0) - 3 * 15.0]  # output well below input
    assert upper.abs().max().item() < 1e-6


@requires_oracle()
def test_rank1_kernel_energy_conserving():
    """One fluorescent bounce re-emits exactly QY of what it absorbed, no more.

    For the operator T[i,j] = QY · e_i · a_j · w_j with e normalised to
    Σ_i e_i w_i = 1, applying it to a field L gives
        re-emitted  = Σ_i w_i (T @ L)_i = QY · Σ_j a_j w_j L_j = QY · absorbed.
    So the bounce is strictly lossy for QY < 1 and never amplifies.
    """
    from src.kernels import kernel_fluorescence
    from src.spectral_grid import make_grid

    grid = make_grid(lam_min=360.0, lam_max=700.0)
    qy = 0.75
    a = torch.exp(-0.5 * ((grid.lam - 400.0) / 20.0) ** 2)  # peak-1 absorption
    T = kernel_fluorescence(
        grid.lam, lam_ex=400.0, lam_em=560.0, sigma_f=20.0,
        weights=grid.weights, quantum_yield=qy,
    )

    L = torch.ones_like(grid.lam)
    absorbed = (a * grid.weights * L).sum()
    reemitted = (grid.weights * (T @ L)).sum()

    assert reemitted.item() <= absorbed.item() + 1e-9
    assert reemitted.item() == pytest.approx(qy * absorbed.item(), rel=1e-9)
