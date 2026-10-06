"""Snell's window: camera inside a flat BK7 (Sellmeier) slab looking up into air.
Self-contained - no external repo.

  A. Window radius - the visible air disk subtends the critical angle
     asin(1/n), not the camera FOV.
  B. Horizon compression - walls spanning theta_air 80.5..90 deg land in a thin
     ring at the rim; the centre still sees the ceiling.
  C. TIR - everything outside the window sees the dark floor below the slab.

Run:  pytest tests/test_snell_window.py    (needs _vera in build/<config>/)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

_vera = pytest.importorskip("_vera")

RES, FOV_Y, SPP = 512, 120.0, 64
BK7 = dict(sell_b1=1.03961212, sell_c1=0.00600069867, sell_b2=0.231792344, sell_c2=0.0200179144)


def _bk7_ior(lam_um):
    l2 = lam_um * lam_um
    return math.sqrt(1.0 + BK7["sell_b1"] * l2 / (l2 - BK7["sell_c1"])
                     + BK7["sell_b2"] * l2 / (l2 - BK7["sell_c2"]))


@pytest.fixture(scope="module")
def img():
    s = _vera.Scene()
    sky  = s.add_material(_vera.emissive([1.0, 1.0, 1.0], 1.0))
    wall = s.add_material(_vera.emissive([0.8, 0.1, 0.8], 1.2))
    dark = s.add_material(_vera.lambertian([0.03, 0.03, 0.03]))
    glass = s.add_material(_vera.dispersive_dielectric(**BK7))

    h = 12.0
    s.add_quad([-h, 2, -h], [h, 2, -h], [h, 2, h], [-h, 2, h], [0, -1, 0], sky)
    s.add_quad([-h, 0, -h], [-h, 0, h], [-h, 2, h], [-h, 2, -h], [1, 0, 0], wall)
    s.add_quad([h, 0, h], [h, 0, -h], [h, 2, -h], [h, 2, h], [-1, 0, 0], wall)
    s.add_quad([h, 0, -h], [-h, 0, -h], [-h, 2, -h], [h, 2, -h], [0, 0, 1], wall)
    s.add_quad([-h, 0, h], [h, 0, h], [h, 2, h], [-h, 2, h], [0, 0, -1], wall)
    s.add_quad([-h, -3, -h], [-h, -3, h], [h, -3, h], [h, -3, -h], [0, 1, 0], dark)
    g = 15.0  # slab: |x|,|z| < 15, -2 < y < 0
    s.add_quad([-g, 0, -g], [-g, 0, g], [g, 0, g], [g, 0, -g], [0, 1, 0], glass)
    s.add_quad([-g, -2, -g], [g, -2, -g], [g, -2, g], [-g, -2, g], [0, -1, 0], glass)
    s.add_quad([-g, -2, -g], [-g, -2, g], [-g, 0, g], [-g, 0, -g], [-1, 0, 0], glass)
    s.add_quad([g, -2, g], [g, -2, -g], [g, 0, -g], [g, 0, g], [1, 0, 0], glass)
    s.add_quad([g, -2, -g], [-g, -2, -g], [-g, 0, -g], [g, 0, -g], [0, 0, -1], glass)
    s.add_quad([-g, -2, g], [g, -2, g], [g, 0, g], [-g, 0, g], [0, 0, 1], glass)

    cam = _vera.camera_look_at(eye=[0, -0.4, 0], target=[0, 1, 0], up=[0, 0, 1],
                               fov_y_deg=FOV_Y, width=RES, height=RES)
    return np.asarray(_vera.render(s, cam, SPP, 32, "raw", 1.0, False, 0))


def _radial(img):
    c = (RES - 1) / 2.0
    y, x = np.mgrid[0:RES, 0:RES]
    return np.hypot(x - c, y - c)


def _px_radius(theta):
    return (RES / 2.0) * math.tan(theta) / math.tan(math.radians(FOV_Y / 2.0))


def _profile(img):
    # azimuthal mean luminance per 1px annulus; robust to stray fireflies
    r = _radial(img).astype(int).ravel()
    lum = img.sum(axis=2).ravel()
    return np.bincount(r, lum) / np.maximum(np.bincount(r), 1)


def test_window_radius_matches_critical_angle(img):
    prof = _profile(img)
    measured = np.argmax(prof < 0.5 * prof[:10].mean())
    expected = _px_radius(math.asin(1.0 / _bk7_ior(0.5876)))
    assert abs(measured - expected) < 3.0, (measured, expected)


def test_walls_compressed_into_rim(img):
    r = _radial(img)
    rim_in = _px_radius(math.asin(math.sin(math.atan(12.0 / 2.0)) / _bk7_ior(0.5876)))
    rim_out = _px_radius(math.asin(1.0 / _bk7_ior(0.5876)))
    ring = img[(r > rim_in + 0.5) & (r < rim_out - 0.5)].mean(axis=0)
    centre = img[r < 0.25 * rim_in].mean(axis=0)
    g_ratio = lambda c: c[1] / (c[0] + c[2])
    assert g_ratio(ring) < 0.5 * g_ratio(centre), (ring, centre)  # magenta wall vs sky


def test_tir_outside_window(img):
    prof = _profile(img)
    lo = int(_px_radius(math.asin(1.0 / _bk7_ior(0.5876)))) + 4
    assert prof[lo:RES // 2].mean() < 0.02, prof[lo:RES // 2].mean()
