"""Fluorescence-aware distance tracking for participating media (Mojzik et al.
2018, sec 5 + App. A). Self-contained - no external repo.

  A. Free-path CDF - the App. A closed form and its inverse are consistent and
     normalised, across sigma_hat_t > 0, sigma_hat_t = 0, and sigma_hat_s > sigma_hat_t.
  B. Zero-QY reduction - fluorescent_medium(quantum_yield=0) reproduces the plain
     medium path (aware sampler is gated on fluorQY > 0).
  C. Fig 10 - a medium that is clear to the eye but fluorescent still glows in the
     emission band; a non-fluorescent medium with the same (zero) coefficients
     stays black.
  D. Energy - the emission-band response grows with QY and stays finite (no gain).

Run:  pytest tests/test_volume_fluorescence.py    (needs _vera in build/<config>/)
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# --- A. free-path CDF (pure python mirror of SampleFluorFreePath / App. A) -----

def _sample_free_path(ss, st, d, u):
    """Returns (t, n, escaped). App. A eq 7, corrected (st/ss - 1)."""
    if st > 1e-12:
        E = math.exp(-d * st)
        n = (ss + (st - ss) * E) / st
        D = 1.0 + (st / max(ss, 1e-12) - 1.0) * E
        arg = 1.0 - u * D
        if arg <= 1e-12:
            return d, n, True
        t = -math.log(arg) / st
        return (t, n, t >= d)
    if ss > 1e-12:
        n = d * ss + 1.0
        t = u * n / ss
        return (t, n, t >= d)
    return d, 1.0, True


def _cdf(ss, st, d, t):
    """P(T <= t) for t < d (App. A)."""
    if st > 1e-12:
        return (1.0 - math.exp(-t * st)) / (1.0 + (st / ss - 1.0) * math.exp(-d * st))
    return t * ss / (d * ss + 1.0)


_CASES = [
    (0.20, 0.20, 10.0),  # scattering only (non-fluorescent reduction)
    (0.02, 0.20, 10.0),  # absorptive, weak scattering
    (0.02, 0.30, 5.0),   # more absorptive
    (0.60, 0.10, 8.0),   # strongly fluorescent: ss > st
    (0.60, 0.03, 12.0),  # strongly fluorescent, weakly absorptive
    (0.40, 0.00, 6.0),   # clear-but-fluorescent: st == 0
]


@pytest.mark.parametrize("ss,st,d", _CASES)
def test_free_path_normalised(ss, st, d):
    """integral of p'(t)/n over [0,d) plus the Dirac mass at d equals 1."""
    _, n, _ = _sample_free_path(ss, st, d, 0.5)
    ts = np.linspace(0.0, d, 20001)
    pdf = ss * np.exp(-st * ts) / n
    cont = np.trapezoid(pdf, ts)
    dirac = math.exp(-st * d) / n
    assert abs(cont + dirac - 1.0) < 2e-4, (ss, st, d, cont, dirac)


@pytest.mark.parametrize("ss,st,d", _CASES)
def test_free_path_inverse_matches_cdf(ss, st, d):
    """The sampled t inverts the CDF: P(sample(u)) == u on the collision branch."""
    pdlo = _cdf(ss, st, d, d - 1e-9)  # P(d-) = 1 - Dirac mass
    for u in (0.01, 0.1, 0.37, 0.6, 0.85, 0.999):
        t, _, escaped = _sample_free_path(ss, st, d, u)
        if u < pdlo * 0.999:
            assert not escaped
            assert abs(_cdf(ss, st, d, t) - u) < 1e-6, (ss, st, d, u, t)
        # u near/above P(d-) must land on the surface
    t, _, escaped = _sample_free_path(ss, st, d, min(pdlo + (1 - pdlo) * 0.5, 0.9999))
    assert escaped


def test_free_path_reduces_to_exponential():
    """ss == st: the App. A sampler is plain exponential tracking, t = -log(1-u)/st."""
    st = 0.3
    for u in (0.05, 0.3, 0.7, 0.95):
        t, n, _ = _sample_free_path(st, st, 1e9, u)  # d -> inf so never escapes
        assert abs(n - 1.0) < 1e-6
        assert abs(t - (-math.log(1.0 - u) / st)) < 1e-6


# --- CUDA-path checks --------------------------------------------------------

_vera = pytest.importorskip("_vera")

SPP = 128
BOUNCES = 16


def _box(medium_fn):
    """Closed grey box, ceiling near-UV emitter, camera inside. medium_fn() -> Medium
    or None. Returns (scene, cam, medium_idx, center_region)."""
    scene = _vera.Scene()
    grey = scene.add_material(_vera.lambertian([0.5, 0.5, 0.5]))
    emit = scene.add_material(_vera.emissive([0.42, 0.30, 1.0], 16.0))
    idx = 0
    m = medium_fn()
    if m is not None:
        idx = scene.add_medium(m)

    scene.add_box_walls(grey, grey, grey, grey, grey, 1.0)
    s = 0.6
    scene.add_quad([-s, 0.999, -s], [s, 0.999, -s], [s, 0.999, s], [-s, 0.999, s],
                   [0.0, -1.0, 0.0], emit)
    cam = _vera.camera_look_at(eye=[0.0, 0.0, 2.6], target=[0.0, 0.0, -1.0],
                               up=[0.0, 1.0, 0.0], fov_y_deg=40.0, width=64, height=64)
    return scene, cam, idx, (slice(20, 44), slice(20, 44))


def _render_mean_xyz(medium_fn):
    scene, cam, idx, region = _box(medium_fn)
    img = _vera.render_xyz(scene, cam, spp=SPP, max_bounces=BOUNCES, default_medium=idx)
    return np.asarray(img, np.float64)[region].reshape(-1, 3).mean(axis=0)


def test_zero_qy_matches_plain_medium():
    """quantum_yield = 0 -> aware sampler is bypassed (fluorQY > 0 gate), so the
    render must match a plain medium() with the same elastic coefficients."""
    plain = _render_mean_xyz(lambda: _vera.medium(
        [0.7, 0.85, 1.0], 0.15, [1.0, 1.0, 1.0], 0.25, 0.0))
    zqy = _render_mean_xyz(lambda: _vera.fluorescent_medium(
        absorb=[0.7, 0.85, 1.0], absorb_scale=0.15,
        scatter=[1.0, 1.0, 1.0], scatter_scale=0.25,
        g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
        fluor_sigma_s=0.6, quantum_yield=0.0))
    rel = np.abs(plain - zqy) / np.maximum(np.abs(plain), 1e-6)
    assert rel.max() < 1e-4, f"zero-QY diverged from plain medium: {rel}, {plain} vs {zqy}"


def test_clear_fluorescent_medium_glows():
    """Fig 10: a medium with zero elastic absorption/scattering but nonzero
    fluorescence. A non-fluorescent medium with the same (zero) coefficients is
    invisible; the fluorescent one produces measurable radiance."""
    clear_nonfluor = _render_mean_xyz(lambda: _vera.medium(
        [0.0, 0.0, 0.0], 0.0, [1.0, 1.0, 1.0], 0.0, 0.0))
    clear_fluor = _render_mean_xyz(lambda: _vera.fluorescent_medium(
        absorb=[0.0, 0.0, 0.0], absorb_scale=0.0,
        scatter=[1.0, 1.0, 1.0], scatter_scale=0.0,
        g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
        fluor_sigma_s=0.9, quantum_yield=0.9))
    # both see the lit walls; the fluorescent medium must add on top of that.
    assert np.all(np.isfinite(clear_fluor))
    assert clear_fluor.sum() > clear_nonfluor.sum() * 1.05, (
        f"clear fluorescent medium did not glow: {clear_nonfluor} -> {clear_fluor}")


def _blob_density(n=32, radius=0.85):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) / c
    return np.ascontiguousarray(np.clip(1.0 - r / radius, 0.0, 1.0).astype(np.float32))


def test_heterogeneous_fluorescent_blob_shifts_red_and_is_finite():
    """A density-grid fluorophore (fluorescence-aware Woodcock) absorbs the blue/UV
    and re-emits ~620nm: X/Z rises vs the same grid at quantum_yield = 0. Total XYZ
    need not rise (QY < 1 loses energy) - the signature is the spectral shift."""
    dens = _blob_density()

    def mk(qy):
        return lambda: _vera.heterogeneous_fluorescent_medium(
            dens, [-0.7, -0.7, -0.7], [0.7, 0.7, 0.7],
            absorb=[0.7, 0.85, 1.0], absorb_scale=0.6,
            scatter=[1.0, 1.0, 1.0], scatter_scale=0.4,
            g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
            fluor_sigma_s=1.2, quantum_yield=qy)

    off = _render_mean_xyz(mk(0.0))
    on = _render_mean_xyz(mk(0.9))
    assert np.all(np.isfinite(on)), on
    assert on.sum() < off.sum() + 20.0, f"heterogeneous runaway: {off} -> {on}"
    r_off, r_on = off[0] / off[2], on[0] / on[2]
    assert r_on > r_off * 1.05, f"no red shift from the fluorophore: X/Z {r_off:.3f} -> {r_on:.3f}"


def test_emission_grows_with_qy_and_stays_finite():
    """Monotone in QY, bounded (no energy gain)."""
    def mk(qy):
        return lambda: _vera.fluorescent_medium(
            absorb=[0.0, 0.0, 0.0], absorb_scale=0.0,
            scatter=[1.0, 1.0, 1.0], scatter_scale=0.0,
            g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
            fluor_sigma_s=0.8, quantum_yield=qy)
    y = [_render_mean_xyz(mk(qy)).sum() for qy in (0.0, 0.3, 0.6, 0.9)]
    assert all(np.isfinite(y)), y
    assert y[1] >= y[0] and y[2] >= y[1] * 0.98 and y[3] >= y[2] * 0.98, y
    assert y[3] < y[0] + 50.0, f"runaway energy: {y}"
