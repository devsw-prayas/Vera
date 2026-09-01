"""Vera — experimental spectral path tracer.

Thin Python frontend over the CUDA renderer. Build the native extension `_vera`
from the repo-root CMakeLists with -DVERA_BUILD_PYTHON=ON.

    import vera

    scene = vera.Scene()
    floor  = scene.add_material(vera.lambertian([0.6, 0.6, 0.6]))
    light  = scene.add_material(vera.emissive([1, 0.95, 0.9], 12.0))
    glass  = scene.add_material(vera.dispersive_dielectric(
                 4.3356, 0.1060**2, 0.3306, 0.1750**2))

    scene.add_box_walls(floor, light, floor, floor, floor, scale=1.0)
    scene.add_uv_sphere(glass, radius=0.4, center=[0.0, -0.6, 0.0])
    scene.add_box(floor, size=[0.6, 0.3, 0.6], center=[-0.5, -0.85, 0.2])

    cam = vera.camera_look_at(eye=[0, 0, 3.3], target=[0, -0.3, 0],
                              fov_y_deg=42, width=800, height=600)
    img = vera.render(scene, cam, spp=256)          # (H, W, 3) float32, tonemapped
    hdr = vera.render_hdr(scene, cam, spp=256)      # (H, W, 3) float32, unclamped
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
    """Imperative scene builder. Wraps the native `_vera.Scene`; adds the
    procedural shape helpers (see `vera.shapes`). Every native method
    (`add_material`, `add_quad`, `add_sphere`, `add_diamond`, `add_box_walls`,
    `add_mesh`, `set_env_constant`, `set_env_hdri`, `build`, `triangle_count`, …)
    is forwarded unchanged."""

    def __init__(self):
        self._s = _Scene()

    def __getattr__(self, name):
        return getattr(self._s, name)

    def __repr__(self):
        return repr(self._s).replace("<vera.Scene", "<vera.Scene(py)")

    # ── standard shapes ─────────────────────────────────────────────────
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
    """Render with no tonemap/clamp — (H, W, 3) float32 of raw linear radiance."""
    return _render(_unwrap(scene), camera, spp, max_bounces, "raw", exposure, use_optix)


def render_xyz(scene, camera, *, spp=256, max_bounces=32, use_optix=False):
    """Raw per-pixel CIE XYZ radiance — (H, W, 3) float32, no tonemap/exposure/sRGB.

    This is the space the renderer accumulates in; use it for spectral-oracle
    cross-checks (project a reference L(lambda) through the analytic CIE fit in
    src/HWSS/Public/CIE.h and compare directly)."""
    return _render_xyz(_unwrap(scene), camera, spp, max_bounces, use_optix)


def save_png(path, img, gamma=False):
    """Write an (H, W, 3) float array in [0, 1] to a PNG (needs Pillow)."""
    import numpy as np
    from PIL import Image

    a = np.asarray(img, dtype=np.float32)
    if gamma:
        a = np.clip(a, 0.0, 1.0) ** (1.0 / 2.2)
    Image.fromarray((np.clip(a, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)).save(path)
