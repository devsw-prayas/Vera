"""Procedural triangle-mesh generators for the standard shapes.

Each returns ``(vertices, faces, normals)``:
  * ``vertices`` : (N, 3) float32
  * ``faces``    : (M, 3) uint32  — indices into ``vertices``, wound CCW as seen
                   from outside / from the +normal side
  * ``normals``  : (N, 3) float32 smooth per-vertex normals for curved surfaces,
                   or ``None`` for flat shapes (the renderer then uses geometric
                   face normals)

Feed straight into ``vera.Scene.add_mesh(vertices, faces, material, normals)``,
or use the ``vera.Scene.add_*`` convenience methods which call these for you.
"""
from __future__ import annotations

import numpy as np

__all__ = ["plane", "box", "disk", "cylinder", "cone", "uv_sphere"]


def _f32(a):  return np.ascontiguousarray(a, dtype=np.float32)
def _u32(a):  return np.ascontiguousarray(a, dtype=np.uint32)


def _basis(normal):
    """Orthonormal (tangent, bitangent, unit-normal) for an arbitrary normal."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) or 1.0)
    helper = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    t = np.cross(helper, n); t /= (np.linalg.norm(t) or 1.0)
    b = np.cross(n, t)
    return t, b, n


def plane(size=(1.0, 1.0), center=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0)):
    """A flat rectangle ``size`` (width, depth) centred at ``center``, facing ``normal``."""
    t, b, n = _basis(normal)
    hw, hd = size[0] * 0.5, size[1] * 0.5
    c = np.asarray(center, dtype=np.float64)
    v = np.array([c - hw * t - hd * b, c + hw * t - hd * b,
                  c + hw * t + hd * b, c - hw * t + hd * b])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return _f32(v), _u32(f), None


def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Solid axis-aligned box, outward-facing."""
    sx, sy, sz = (np.asarray(size, dtype=np.float64) * 0.5)
    cx, cy, cz = center
    v = np.array([[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
                  [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
                  [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]])
    f = np.array([
        [0, 3, 2], [0, 2, 1],   # -Z
        [4, 5, 6], [4, 6, 7],   # +Z
        [0, 4, 7], [0, 7, 3],   # -X
        [1, 2, 6], [1, 6, 5],   # +X
        [0, 1, 5], [0, 5, 4],   # -Y
        [3, 7, 6], [3, 6, 2],   # +Y
    ])
    return _f32(v), _u32(f), None


def disk(radius=1.0, center=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0), segments=64):
    """Filled circle, front face toward ``normal``."""
    t, b, n = _basis(normal)
    c = np.asarray(center, dtype=np.float64)
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = c + radius * (np.outer(np.cos(ang), t) + np.outer(np.sin(ang), b))
    v = np.vstack([c[None, :], rim])
    f = np.array([[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)])
    return _f32(v), _u32(f), None


def cylinder(radius=1.0, height=1.0, center=(0.0, 0.0, 0.0),
             axis=(0.0, 1.0, 0.0), segments=64, caps=True):
    """Cylinder centred at ``center``, running along ``axis``. Smooth side normals."""
    t, b, n = _basis(axis)
    c = np.asarray(center, dtype=np.float64)
    half = n * (height * 0.5)
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.outer(np.cos(ang), t) + np.outer(np.sin(ang), b)     # (S,3) unit radial
    bot = c - half + radius * ring
    top = c + half + radius * ring
    v = np.vstack([bot, top])
    nrm = np.vstack([ring, ring])                                  # side normals = radial

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, segments + i, segments + j], [i, segments + j, j]]

    if caps:
        cb = len(v); v = np.vstack([v, (c - half)[None, :]]); nrm = np.vstack([nrm, (-n)[None, :]])
        ct = len(v); v = np.vstack([v, (c + half)[None, :]]); nrm = np.vstack([nrm, n[None, :]])
        for i in range(segments):
            j = (i + 1) % segments
            faces += [[cb, j, i], [ct, segments + i, segments + j]]

    return _f32(v), _u32(np.array(faces)), _f32(nrm)


def cone(radius=1.0, height=1.0, center=(0.0, 0.0, 0.0),
         axis=(0.0, 1.0, 0.0), segments=64, cap=True):
    """Cone with its base centred at ``center``, apex ``height`` along ``axis``."""
    t, b, n = _basis(axis)
    c = np.asarray(center, dtype=np.float64)
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.outer(np.cos(ang), t) + np.outer(np.sin(ang), b)
    base = c + radius * ring
    apex = c + n * height
    v = np.vstack([base, apex[None, :]])

    slope = radius / np.hypot(radius, height)
    side_n = slope * ring + (height / np.hypot(radius, height)) * n
    nrm = np.vstack([side_n, n[None, :]])

    faces = [[i, (i + 1) % segments, segments] for i in range(segments)]
    if cap:
        ci = len(v); v = np.vstack([v, c[None, :]]); nrm = np.vstack([nrm, (-n)[None, :]])
        faces += [[ci, (i + 1) % segments, i] for i in range(segments)]

    return _f32(v), _u32(np.array(faces)), _f32(nrm)


def uv_sphere(radius=1.0, center=(0.0, 0.0, 0.0), segments=64, rings=32):
    """Latitude/longitude sphere with smooth normals."""
    c = np.asarray(center, dtype=np.float64)
    lat = np.linspace(0.0, np.pi, rings + 1)
    lon = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    dirs = np.stack([np.sin(la) * np.cos(lo), np.cos(la), np.sin(la) * np.sin(lo)], axis=-1)
    dirs = dirs.reshape(-1, 3)
    v = c + radius * dirs

    W = segments + 1
    faces = []
    for i in range(rings):
        for j in range(segments):
            a, bb, cc, d = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            faces += [[a, cc, bb], [bb, cc, d]]
    return _f32(v), _u32(np.array(faces)), _f32(dirs)
