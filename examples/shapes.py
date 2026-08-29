"""Every standard shape in one frame, via the vera.Scene.add_* helpers.

    python examples/shapes.py
"""
import vera

s = vera.Scene()
grey  = s.add_material(vera.lambertian([0.62, 0.62, 0.64]))
light = s.add_material(vera.emissive([1.0, 0.96, 0.90], 16.0))
red   = s.add_material(vera.lambertian([0.75, 0.12, 0.12]))
metal = s.add_material(vera.ggx([0.95, 0.66, 0.55], roughness=0.06))
glass = s.add_material(vera.dispersive_dielectric(4.3356, 0.1060**2, 0.3306, 0.1750**2))

s.add_plane(grey,  size=[8, 8], center=[0, -1, 0])
s.add_plane(grey,  size=[8, 5], center=[0, 1.5, -3], normal=[0, 0, 1])
s.add_plane(light, size=[2.6, 2.6], center=[0, 3.49, -0.5], normal=[0, -1, 0])

s.add_box(red,        size=[0.9, 0.9, 0.9], center=[-1.35, -0.55, -0.2])
s.add_uv_sphere(metal, radius=0.6, center=[1.15, -0.4, 0.1])
s.add_cylinder(grey,  radius=0.26, height=1.5, center=[0.15, -0.25, -0.9])
s.add_cone(glass,     radius=0.5, height=1.0, center=[-0.2, -1.0, 1.2])

cam = vera.camera_look_at(eye=[0.3, 0.9, 4.6], target=[0, -0.25, 0],
                          fov_y_deg=44, width=900, height=650)

img = vera.render(s, cam, spp=400, tonemap="aces", exposure=1.15)
vera.save_png("shapes.png", img)
print("wrote shapes.png", img.shape, "|", s)
