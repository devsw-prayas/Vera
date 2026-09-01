"""Rendered-image checks for the rank-1 fluorescent Lambertian (Mojzik et al. 2018),
driving the actual CUDA renderer through the `_vera` extension:

  A. Excitation-wavelength invariance — under a flat illuminant the fluorescent
     response should depend only on the emission profile, not where in the
     absorption band the energy was gathered.
  B. Energy-gain direction — fluorescence should raise luminance on the surface.
  C. Zero-QY reduction — quantum_yield = 0 reproduces the plain Lambertian path.
  D. Oracle ratio — a single fluorescent bounce's fluor/elastic XYZ ratio should
     match the DBR Python oracle's kernel_fluorescence prediction.

Run:  pytest tests/test_fluorescence_render.py
Needs `_vera` (built into build/<config>/); D also needs `torch` + the DBR
oracle repo — see conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_oracle

_vera = pytest.importorskip("_vera")

LAMBDA_MIN, LAMBDA_MAX = 360.0, 700.0

GREY = [0.5, 0.5, 0.5]
SPP = 96
BOUNCES_CAVITY = 12


# ---------------------------------------------------------------------------
# analytic CIE 1931 fit — ported verbatim from src/HWSS/Public/CIE.h
# ---------------------------------------------------------------------------

def _cie_gauss(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    t = (x - mu) / s
    return np.exp(-0.5 * t * t)


def _cie_xyz(lam):
    lam = np.asarray(lam, dtype=np.float64)
    x = (1.056 * _cie_gauss(lam, 599.8, 37.9, 31.0)
         + 0.362 * _cie_gauss(lam, 442.0, 16.0, 26.7)
         - 0.065 * _cie_gauss(lam, 501.1, 20.4, 26.2))
    y = (0.821 * _cie_gauss(lam, 568.8, 46.9, 40.5)
         + 0.286 * _cie_gauss(lam, 530.9, 16.3, 31.1))
    z = (1.217 * _cie_gauss(lam, 437.0, 11.8, 36.0)
         + 0.681 * _cie_gauss(lam, 459.0, 26.0, 13.8))
    return np.stack([x, y, z], axis=-1)


def _cie_y_integral(steps=1024):
    lam = np.linspace(LAMBDA_MIN, LAMBDA_MAX, steps)
    return np.trapz(_cie_xyz(lam)[:, 1], lam)


# ---------------------------------------------------------------------------
# scene helpers
# ---------------------------------------------------------------------------

def _cavity(back_material_id_fn, *, emitter_intensity=6.0):
    """Closed grey box, camera looking at the back wall, ceiling emitter panel.

    `back_material_id_fn(scene) -> int` supplies the back wall's material so the
    caller controls whether it is elastic or fluorescent.
    Returns (scene, camera, back_pixel_slice).
    """
    scene = _vera.Scene()
    grey = scene.add_material(_vera.lambertian(GREY))
    emit = scene.add_material(_vera.emissive([1.0, 1.0, 1.0], emitter_intensity))
    back = back_material_id_fn(scene)

    # floor, ceil, back, left, right
    scene.add_box_walls(grey, grey, back, grey, grey, 1.0)
    # ceiling emitter panel just below the ceiling (y = +0.999)
    s = 0.6
    scene.add_quad([-s, 0.999, -s], [s, 0.999, -s], [s, 0.999, s], [-s, 0.999, s],
                   [0.0, -1.0, 0.0], emit)

    cam = _vera.camera_look_at(eye=[0.0, 0.0, 2.6], target=[0.0, 0.0, -1.0],
                               up=[0.0, 1.0, 0.0], fov_y_deg=40.0,
                               width=64, height=64)
    # central patch of the frame = back wall
    return scene, cam, (slice(24, 40), slice(24, 40))


def _mean_xyz(img, region):
    return np.asarray(img, dtype=np.float64)[region].reshape(-1, 3).mean(axis=0)


# ---------------------------------------------------------------------------
# A. excitation-wavelength invariance
# ---------------------------------------------------------------------------

def test_excitation_wavelength_invariance():
    """Flat illuminant: sweeping lam_ex leaves the fluorescent wall unchanged.

    lam_ex is kept inside the visible band where the emissive white SPD is
    genuinely flat, so integral(a * L) is lam_ex-independent (Gaussian mass N_a
    is the same) and only the emission profile drives the result.
    """
    lam_em = 620.0
    out = []
    for lam_ex in (430.0, 470.0):
        scene, cam, region = _cavity(
            lambda sc: sc.add_material(
                _vera.fluorescent_lambertian(GREY, lam_ex, lam_em, 15.0, 0.8)))
        img = _vera.render_xyz(scene, cam, spp=SPP, max_bounces=BOUNCES_CAVITY)
        out.append(_mean_xyz(img, region))

    a, b = out
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
    # MC noise floor at this spp/res; the two must agree well within it.
    assert rel.max() < 0.06, f"excitation dependence too large: {rel}, {a} vs {b}"


# ---------------------------------------------------------------------------
# B. energy-gain direction
# ---------------------------------------------------------------------------

def test_fluorescence_adds_luminance():
    """QY > 0 raises luminance vs the same wall with QY = 0 (UV/violet -> 620nm)."""
    def make(qy):
        scene, cam, region = _cavity(
            lambda sc: sc.add_material(
                _vera.fluorescent_lambertian(GREY, 400.0, 620.0, 18.0, qy)))
        img = _vera.render_xyz(scene, cam, spp=SPP, max_bounces=BOUNCES_CAVITY)
        return _mean_xyz(img, region)

    off = make(0.0)
    on = make(0.9)
    assert on[1] > off[1] * 1.02, f"fluorescence did not add luminance: Y {off[1]} -> {on[1]}"


# ---------------------------------------------------------------------------
# C. zero-QY reduction
# ---------------------------------------------------------------------------

def test_zero_qy_matches_plain_lambertian():
    """quantum_yield = 0 must reproduce the elastic Lambertian path.

    IsFluorescent(mat) is false when fluorQY == 0, so the shade kernel takes the
    identical branch — same RNG draws, same everything. Any difference beyond
    tight float noise is a regression in the shared path.
    """
    def render(mat_fn):
        scene, cam, region = _cavity(lambda sc: sc.add_material(mat_fn(sc)))
        img = _vera.render_xyz(scene, cam, spp=SPP, max_bounces=BOUNCES_CAVITY)
        return _mean_xyz(img, region)

    plain = render(lambda sc: _vera.lambertian(GREY))
    zqy = render(lambda sc: _vera.fluorescent_lambertian(GREY, 400.0, 620.0, 15.0, 0.0))
    rel = np.abs(plain - zqy) / np.maximum(np.abs(plain), 1e-6)
    assert rel.max() < 1e-4, f"zero-QY diverged from plain Lambertian: {rel}, {plain} vs {zqy}"


# ---------------------------------------------------------------------------
# D. oracle ratio — single fluorescent bounce
# ---------------------------------------------------------------------------

@requires_oracle()
def test_single_bounce_shape_matches_oracle():
    """One fluorescent bounce reproduces kernel_fluorescence's spectral shape.

    A flat wall lit by a big flat white emitter, one bounce (max_bounces=2),
    rendered twice with the same albedo — elastic and fluorescent — and compared
    as the XYZ ratio fluor/elastic, which cancels geometry/pi/exposure/CIE-norm
    against the same ratio computed from the oracle's rank-1 operator.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    from src.kernels import kernel_fluorescence
    from src.spectral_grid import make_grid

    R, QY = 0.5, 0.8
    lam_ex, lam_em, sigma = 440.0, 610.0, 16.0

    # --- Vera: open scene, single bounce, flat white emitter facing the wall ---
    def render_wall(mat_fn):
        scene = _vera.Scene()
        wall = scene.add_material(mat_fn(scene))
        emit = scene.add_material(_vera.emissive([1.0, 1.0, 1.0], 8.0))
        # wall at z = -1 facing +z
        s = 1.0
        scene.add_quad([-s, -s, -1.0], [s, -s, -1.0], [s, s, -1.0], [-s, s, -1.0],
                       [0.0, 0.0, 1.0], wall)
        # big emitter panel at z = -0.2, also facing +z toward the camera but
        # behind it lighting the wall from the front
        e = 2.0
        scene.add_quad([-e, -e, 1.6], [e, -e, 1.6], [e, e, 1.6], [-e, e, 1.6],
                       [0.0, 0.0, -1.0], emit)
        cam = _vera.camera_look_at(eye=[0.0, 0.0, 1.2], target=[0.0, 0.0, -1.0],
                                   up=[0.0, 1.0, 0.0], fov_y_deg=30.0,
                                   width=48, height=48)
        img = _vera.render_xyz(scene, cam, spp=256, max_bounces=2)
        return np.asarray(img, np.float64)[16:32, 16:32].reshape(-1, 3).mean(axis=0)

    xyz_elastic = render_wall(lambda sc: _vera.lambertian([R, R, R]))
    xyz_fluor = render_wall(
        lambda sc: _vera.fluorescent_lambertian([R, R, R], lam_ex, lam_em, sigma, QY))
    vera_ratio = xyz_fluor / xyz_elastic

    # --- Oracle: same ratio from the rank-1 operator, flat unit illuminant ---
    grid = make_grid(lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX)
    lam = grid.lam.numpy()
    w = grid.weights.numpy()
    L_flat = np.ones_like(lam)

    Kfl = kernel_fluorescence(grid.lam, lam_ex, lam_em, sigma, grid.weights, QY).numpy()
    L_elastic = R * L_flat
    L_fluor = R * L_flat + Kfl @ L_flat

    cie = _cie_xyz(lam)                      # (N, 3)
    proj = lambda L: (L[:, None] * cie * w[:, None]).sum(axis=0)
    oracle_ratio = proj(L_fluor) / proj(L_elastic)

    rel = np.abs(vera_ratio - oracle_ratio) / np.abs(oracle_ratio)
    assert rel.max() < 0.05, (
        f"fluor/elastic XYZ ratio disagrees with oracle: "
        f"vera {vera_ratio} vs oracle {oracle_ratio} (rel {rel})")
