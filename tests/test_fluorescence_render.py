"""Rendered-image checks for the rank-1 fluorescent Lambertian (Mojzik et al. 2018),
driving the CUDA renderer through `_vera`. Self-contained - no external repo:

  A. Excitation-wavelength invariance - under a flat illuminant the fluorescent
     response depends only on the emission profile, not where in the absorption
     band the energy was gathered.
  B. Energy-gain direction - fluorescence raises luminance on the surface.
  C. Zero-QY reduction - quantum_yield = 0 reproduces the plain Lambertian path.

The quantitative oracle cross-check (vs the DBR bispectral solver) lives in the
Inverse Spectral Rendering repo and is run manually.

Run:  pytest tests/test_fluorescence_render.py    (needs _vera in build/<config>/)
"""
from __future__ import annotations

import numpy as np
import pytest

_vera = pytest.importorskip("_vera")

GREY = [0.5, 0.5, 0.5]
SPP = 96
BOUNCES = 12


def _cavity(back_material_fn):
    """Closed grey box, camera on the back wall, ceiling emitter panel.
    `back_material_fn(scene) -> matId` picks the back wall material.
    Returns (scene, camera, back_wall_pixel_slice)."""
    scene = _vera.Scene()
    grey = scene.add_material(_vera.lambertian(GREY))
    emit = scene.add_material(_vera.emissive([1.0, 1.0, 1.0], 6.0))
    back = back_material_fn(scene)

    scene.add_box_walls(grey, grey, back, grey, grey, 1.0)  # floor, ceil, back, left, right
    s = 0.6
    scene.add_quad([-s, 0.999, -s], [s, 0.999, -s], [s, 0.999, s], [-s, 0.999, s],
                   [0.0, -1.0, 0.0], emit)

    cam = _vera.camera_look_at(eye=[0.0, 0.0, 2.6], target=[0.0, 0.0, -1.0],
                               up=[0.0, 1.0, 0.0], fov_y_deg=40.0, width=64, height=64)
    return scene, cam, (slice(24, 40), slice(24, 40))


def _mean_xyz(img, region):
    return np.asarray(img, dtype=np.float64)[region].reshape(-1, 3).mean(axis=0)


def _render_back_wall(back_material_fn):
    scene, cam, region = _cavity(back_material_fn)
    img = _vera.render_xyz(scene, cam, spp=SPP, max_bounces=BOUNCES)
    return _mean_xyz(img, region)


def test_excitation_wavelength_invariance():
    """Flat illuminant: sweeping lam_ex must not change the fluorescent wall.
    lam_ex stays inside the visible band where the emitter SPD is flat, so the
    absorbed energy integral(a * L) is lam_ex-independent and only emission drives
    the result."""
    out = [
        _render_back_wall(lambda sc: sc.add_material(
            _vera.fluorescent_lambertian(GREY, lam_ex, 620.0, 15.0, 0.8)))
        for lam_ex in (430.0, 470.0)
    ]
    rel = np.abs(out[0] - out[1]) / np.maximum(np.abs(out[0]), 1e-6)
    assert rel.max() < 0.06, f"excitation dependence too large: {rel}, {out[0]} vs {out[1]}"


def test_fluorescence_adds_luminance():
    """QY > 0 raises luminance vs the same wall with QY = 0 (violet -> 620nm)."""
    off = _render_back_wall(lambda sc: sc.add_material(
        _vera.fluorescent_lambertian(GREY, 400.0, 620.0, 18.0, 0.0)))
    on = _render_back_wall(lambda sc: sc.add_material(
        _vera.fluorescent_lambertian(GREY, 400.0, 620.0, 18.0, 0.9)))
    assert on[1] > off[1] * 1.02, f"fluorescence did not add luminance: Y {off[1]} -> {on[1]}"


def test_zero_qy_matches_plain_lambertian():
    """quantum_yield = 0 must reproduce the elastic Lambertian path exactly.
    IsFluorescent(mat) is false at fluorQY == 0, so the shade kernel takes the
    identical branch - same RNG draws. Any diff beyond float noise is a regression
    in the shared path."""
    plain = _render_back_wall(lambda sc: sc.add_material(_vera.lambertian(GREY)))
    zqy = _render_back_wall(lambda sc: sc.add_material(
        _vera.fluorescent_lambertian(GREY, 400.0, 620.0, 15.0, 0.0)))
    rel = np.abs(plain - zqy) / np.maximum(np.abs(plain), 1e-6)
    assert rel.max() < 1e-4, f"zero-QY diverged from plain Lambertian: {rel}, {plain} vs {zqy}"
