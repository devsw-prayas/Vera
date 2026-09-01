"""Visual check for the rank-1 fluorescent Lambertian (Mojzik et al. 2018).

An enclosed room with a violet/near-UV ceiling light. The LEFT sphere is a
fluorophore that absorbs around 400 nm and re-emits (diffusely) around 520 nm -
a green Stokes glow that should read clearly against the dim violet-lit walls,
with no colour cast once converged. The RIGHT sphere is the same grey Lambertian
with fluorescence off, for reference.

    python examples/fluorescent_room.py

Drives the freshly built build/<config>/_vera directly (bypasses the `vera`
editable-install wrapper).
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

scene = _vera.Scene()

wall = scene.add_material(_vera.lambertian([0.55, 0.55, 0.6]))
floor = scene.add_material(_vera.lambertian([0.35, 0.35, 0.38]))
lamp = scene.add_material(_vera.emissive([0.42, 0.30, 1.0], 14.0))   # near-UV / deep violet
fluo = scene.add_material(
    _vera.fluorescent_lambertian([0.30, 0.30, 0.32], 400.0, 520.0, 16.0, 0.9))
plain = scene.add_material(_vera.lambertian([0.30, 0.30, 0.32]))

scene.add_box_walls(floor, wall, wall, wall, wall, 1.0)
s = 0.55
scene.add_quad([-s, 0.999, -s], [s, 0.999, -s], [s, 0.999, s], [-s, 0.999, s],
               [0.0, -1.0, 0.0], lamp)
scene.add_sphere([-0.42, -0.55, 0.0], 0.42, fluo)
scene.add_sphere([0.42, -0.55, 0.0], 0.42, plain)

cam = _vera.camera_look_at([0.0, 0.1, 3.1], [0.0, -0.25, 0.0], [0.0, 1.0, 0.0],
                           42.0, 900, 700)

img = np.asarray(_vera.render(scene, cam, 1024, 24, "aces", 1.0, False), np.float32)

from PIL import Image
a = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
out = os.path.join(_ROOT, "fluorescent_room.png")
Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(out)
print("wrote", out)
