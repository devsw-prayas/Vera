"""Visual check for fluorescence-aware distance tracking (Mojzik et al. 2018, sec 5).

An enclosed box filled with a fluorescent participating medium, lit by a near-UV
ceiling panel. Two variants:

  VARIANT="absorptive"    - the medium also absorbs and scatters elastically; it
    reads as a hazy volume that glows in the emission band (paper Fig 1 / 11).
  VARIANT="clear"         - zero elastic absorption/scattering, purely fluorescent.
    Plain exponential tracking generates NO collision events here, so a classic
    tracer renders it completely transparent (paper Fig 10). The aware sampler
    puts collisions where the fluorescence is, producing the Stokes glow.
  VARIANT="heterogeneous" - a soft density-grid sphere of fluorophore; glows only
    where it is dense (fluorescence-aware Woodcock, paper Fig 15).

    python examples/fluorescent_medium.py [absorptive|clear|heterogeneous]

Drives the freshly built build/<config>/_vera directly.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for cfg in ("Release", "Debug"):
    p = os.path.join(_ROOT, "build", cfg)
    if os.path.isdir(p):
        sys.path.insert(0, p)

import numpy as np
import _vera

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "absorptive"

scene = _vera.Scene()

wall = scene.add_material(_vera.lambertian([0.55, 0.55, 0.6]))
floor = scene.add_material(_vera.lambertian([0.35, 0.35, 0.38]))
lamp = scene.add_material(_vera.emissive([0.42, 0.30, 1.0], 16.0))  # near-UV / deep violet

if VARIANT == "clear":
    fog = _vera.fluorescent_medium(
        absorb=[0.0, 0.0, 0.0], absorb_scale=0.0,
        scatter=[1.0, 1.0, 1.0], scatter_scale=0.0,
        g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
        fluor_sigma_s=0.9, quantum_yield=0.9)
elif VARIANT == "heterogeneous":
    N = 48
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N].astype(np.float32)
    c = (N - 1) / 2
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) / c
    dens = np.ascontiguousarray(np.clip(1.0 - r / 0.85, 0.0, 1.0).astype(np.float32) ** 1.5)
    fog = _vera.heterogeneous_fluorescent_medium(
        dens, [-0.7, -0.7, -0.7], [0.7, 0.7, 0.7],
        absorb=[0.7, 0.85, 1.0], absorb_scale=0.6,
        scatter=[1.0, 1.0, 1.0], scatter_scale=0.4,
        g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
        fluor_sigma_s=1.2, quantum_yield=0.9)
else:
    fog = _vera.fluorescent_medium(
        absorb=[0.7, 0.85, 1.0], absorb_scale=0.15,
        scatter=[1.0, 1.0, 1.0], scatter_scale=0.25,
        g=0.0, lam_ex=360.0, lam_em=620.0, sigma=20.0,
        fluor_sigma_s=0.6, quantum_yield=0.9)
med = scene.add_medium(fog)

scene.add_box_walls(floor, wall, wall, wall, wall, 1.0)
s = 0.55
scene.add_quad([-s, 0.999, -s], [s, 0.999, -s], [s, 0.999, s], [-s, 0.999, s],
               [0.0, -1.0, 0.0], lamp)

cam = _vera.camera_look_at([0.0, 0.0, 3.1], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                           42.0, 900, 700)

# camera starts inside the medium - the whole box is filled.
img = np.asarray(
    _vera.render(scene, cam, 1024, 24, "aces", 1.0, False, med), np.float32)

from PIL import Image
a = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
out = os.path.join(_ROOT, f"fluorescent_medium_{VARIANT}.png")
Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(out)
print("wrote", out)
