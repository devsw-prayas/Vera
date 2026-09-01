"""Vera - Python frontend over the CUDA spectral path tracer.

Build the `_vera` extension from the repo-root CMakeLists with
-DVERA_BUILD_PYTHON=ON. Usage: build a `Scene`, make a camera with
`camera_look_at`, then `render` (tonemapped), `render_hdr` (linear), or
`render_xyz` (raw CIE XYZ).
"""
from __future__ import annotations

from . import shapes

try:
    # installed / `pip install -e .` layout: _vera sits inside the package
    from . import _vera  # type: ignore
except ImportError:
    # dev layout: the freshly built extension on PYTHONPATH (build/<cfg>/)
    import _vera  # type: ignore

Material = _vera.Material
Medium = _vera.Medium
Camera = _vera.Camera
_Scene = _vera.Scene
lambertian = _vera.lambertian
ggx = _vera.ggx
dielectric = _vera.dielectric
dispersive_dielectric = _vera.dispersive_dielectric
emissive = _vera.emissive
fluorescent_lambertian = _vera.fluorescent_lambertian
medium = _vera.medium
camera_look_at = _vera.camera_look_at
_render = _vera.render
_render_xyz = _vera.render_xyz

__all__ = [
    "Material", "Medium", "Camera", "Scene",
    "lambertian", "ggx", "dielectric", "dispersive_dielectric", "emissive",
    "fluorescent_lambertian", "medium",
    "camera_look_at", "render", "render_hdr", "render_xyz", "save_png", "shapes",
]


class Scene:
    """Scene builder. Wraps `_vera.Scene` (all native methods forwarded) and adds
    the `add_plane/box/disk/cylinder/cone/uv_sphere` helpers from `vera.shapes`."""

    def __init__(self):
        self._s = _Scene()

    def __getattr__(self, name):
        return getattr(self._s, name)

    def __repr__(self):
        return repr(self._s).replace("<vera.Scene", "<vera.Scene(py)")

    # -- standard shapes -------------------------------------------------
    def _mesh(self, gen_out, material):
        v, f, n = gen_out
        self._s.add_mesh(v, f, material, n)

    def add_plane(self, material, size=(1.0, 1.0), center=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)):
        self._mesh(shapes.plane(size, center, normal), material)

    def add_box(self, material, size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
        self._mesh(shapes.box(size, center), material)

    def add_disk(self, material, radius=1.0, center=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), segments=64):
        self._mesh(shapes.disk(radius, center, normal, segments), material)

    def add_cylinder(self, material, radius=1.0, height=1.0, center=(0.0, 0.0, 0.0),
                     axis=(0.0, 1.0, 0.0), segments=64, caps=True):
        self._mesh(shapes.cylinder(radius, height, center, axis, segments, caps), material)

    def add_cone(self, material, radius=1.0, height=1.0, center=(0.0, 0.0, 0.0),
                 axis=(0.0, 1.0, 0.0), segments=64, cap=True):
        self._mesh(shapes.cone(radius, height, center, axis, segments, cap), material)

    def add_uv_sphere(self, material, radius=1.0, center=(0.0, 0.0, 0.0), segments=64, rings=32):
        self._mesh(shapes.uv_sphere(radius, center, segments, rings), material)


def _unwrap(scene):
    return scene._s if isinstance(scene, Scene) else scene


def render(scene, camera, *, spp=256, max_bounces=32, tonemap="aces",
           exposure=1.0, use_optix=False):
    """Render `scene` from `camera`; returns tonemapped (H, W, 3) float32 in [0, 1]."""
    return _render(_unwrap(scene), camera, spp, max_bounces, tonemap, exposure, use_optix)


def render_hdr(scene, camera, *, spp=256, max_bounces=32, exposure=1.0, use_optix=False):
    """Render with no tonemap/clamp - (H, W, 3) float32 of raw linear radiance."""
    return _render(_unwrap(scene), camera, spp, max_bounces, "raw", exposure, use_optix)


def render_xyz(scene, camera, *, spp=256, max_bounces=32, use_optix=False):
    """Raw per-pixel CIE XYZ radiance, (H, W, 3) float32 - no tonemap/exposure/sRGB.

    The space the renderer accumulates in, so the fluorescence tests compare it
    directly against the DBR oracle's spectral radiance.
    """
    return _render_xyz(_unwrap(scene), camera, spp, max_bounces, use_optix)


def save_png(path, img, gamma=False):
    """Write an (H, W, 3) float array in [0, 1] to a PNG (needs Pillow)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img, dtype=np.float32)
    if gamma:
        a = np.clip(a, 0.0, 1.0) ** (1.0 / 2.2)
    Image.fromarray((np.clip(a, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)).save(path)
