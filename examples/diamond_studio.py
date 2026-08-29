"""Port of src/Main.cu's diamond studio scene to the Python API.

    python examples/diamond_studio.py
"""
import vera

DIAMOND_AXIS = [0.0, 0.70, -0.714]

scene = vera.Scene()

floor   = scene.add_material(vera.lambertian([0.035, 0.045, 0.06]))
backdrop = scene.add_material(vera.emissive([0.72, 0.86, 1.0], 7.0))
diamond = scene.add_material(vera.dispersive_dielectric(
    sell_b1=4.3356, sell_c1=0.1060**2, sell_b2=0.3306, sell_c2=0.1750**2))
ceiling = scene.add_material(vera.emissive([1.0, 0.92, 0.78], 3.0))

R, FY, WT = 4.0, -2.0, 4.0
scene.add_quad([-R, FY, -R], [R, FY, -R], [R, FY, R], [-R, FY, R], [0, 1, 0], floor)
scene.add_quad([-R, FY, -3.5], [R, FY, -3.5], [R, WT, -3.5], [-R, WT, -3.5], [0, 0, 1], backdrop)
scene.add_quad([-R, FY, -3.5], [-R, FY, R], [-R, WT, R], [-R, WT, -3.5], [1, 0, 0], floor)
scene.add_quad([R, FY, R], [R, FY, -3.5], [R, WT, -3.5], [R, WT, R], [-1, 0, 0], floor)
scene.add_quad([-R, WT, -3.5], [-R, WT, R], [R, WT, R], [R, WT, -3.5], [0, -1, 0], floor)
scene.add_quad([-1.6, WT - 0.05, -0.8], [1.6, WT - 0.05, -0.8],
              [1.6, WT - 0.05, 1.0], [-1.6, WT - 0.05, 1.0], [0, -1, 0], ceiling)

for center, gr, tr, ch, pd in [
    ([0.00, -1.616, 0.15], 0.52, 0.22, 0.20, 0.49),
    ([-1.35, -1.782, 0.75], 0.33, 0.14, 0.13, 0.31),
    ([1.55, -1.782, 0.85], 0.33, 0.14, 0.13, 0.31),
    ([-1.85, -1.86, 1.75], 0.21, 0.09, 0.08, 0.20),
    ([2.80, -1.86, 1.60], 0.21, 0.09, 0.08, 0.20),
]:
    scene.add_diamond(center, gr, tr, ch, pd, diamond, axis=DIAMOND_AXIS)

# optional image-based fill; harmless if the file is absent -> comment out
try:
    scene.set_env_hdri("src/HWSS/Data/env.hdr", intensity=1.0)
except RuntimeError:
    pass

cam = vera.camera_look_at(eye=[3.6, -1.25, 3.6], target=[0.0, -1.25, 0.45],
                          fov_y_deg=52.0, width=1280, height=720)

img = vera.render(scene, cam, spp=256, max_bounces=32, tonemap="aces", exposure=1.2)
vera.save_png("diamond_studio.png", img)
print("wrote diamond_studio.png", img.shape)
