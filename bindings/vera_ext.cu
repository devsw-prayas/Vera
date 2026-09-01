// Vera Python bindings (nanobind). Compiled as .cu only because the public headers are
// .cuh; this TU has no kernels - it drives the host entry points (UploadScene,
// BuildSceneLighting, RenderHWSS) and marshals the framebuffer into a NumPy array.
// Thin imperative wrapper: Scene builder, material/env factories, Camera, render().

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <Triangle.h>

#include <Scene.cuh>
#include <Camera.cuh>
#include <HdrImage.h>

#include "MaterialHWSS.h"
#include "MediumHWSS.h"
#include "EnvMapHWSS.h"
#include "RGB2Spec.h"
#include "SceneHWSS.h"
#include "FrameBufferHWSS.cuh"
#include "RendererHWSS.cuh"
#include "CIE.h"
#include "SpectralConstants.h"

namespace nb = nanobind;
using namespace Vera::Core;
using namespace Vera::Spectral::HWSS;
using Vec3 = std::array<float, 3>;

namespace {
	inline float3 f3(const Vec3& a) { return make_float3(a[0], a[1], a[2]); }

	inline void cuda_check(const char* where) {
		cudaError_t e = cudaGetLastError();
		if (e != cudaSuccess)
			throw std::runtime_error(std::string("CUDA error in ") + where + ": " + cudaGetErrorString(e));
	}

	ToneMapper parse_tonemap(const std::string& s) {
		if (s == "aces" || s == "ACES")     return ToneMapper::ACES;
		if (s == "agx" || s == "AgX")      return ToneMapper::AgX;
		if (s == "reinhard" || s == "Reinhard") return ToneMapper::Reinhard;
		if (s == "linear" || s == "Linear")   return ToneMapper::Linear;
		if (s == "raw" || s == "none")      return ToneMapper::Raw;   // unclamped linear radiance
		throw std::invalid_argument("unknown tonemapper '" + s + "' (aces|agx|reinhard|linear|raw)");
	}

	// Process-lifetime spectral upsampling table. Loaded once from VERA_DATA_DIR.
	RGB2SpecTable& spec_table() {
		static RGB2SpecTable table = [] {
			RGB2SpecTable t = RGB2SpecTable::Load(VERA_DATA_DIR "srgb.coeff");
			if (t.res == 0)
				throw std::runtime_error("failed to load " VERA_DATA_DIR "srgb.coeff");
			return t;
			}();
		return table;
	}

	// -- Scene: host-side accumulation + lazy GPU upload ----------------------
	struct Scene {
		std::vector<float3>        pos, norms;
		std::vector<NXB::Triangle> tris;
		std::vector<uint16_t>      matIds;
		std::vector<Material>      materials;
		std::vector<MediumHWSS>    media;
		EnvMapHWSS                 envMap{};      // default: black constant env
		float                     bboxPad = 0.05f;

		SceneGPU          gpu{};
		SceneLightingHWSS lighting{};
		bool              built = false;

		~Scene() { release(); }

		int add_material(const Material& m) {
			materials.push_back(m);
			return int(materials.size() - 1);
		}

		// 1-indexed medium slot (0 == vacuum), matching Material::mediumIdx.
		int add_medium(const MediumHWSS& m) {
			media.push_back(m);
			return int(media.size());
		}

		void add_quad(const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3,
					  const Vec3& n, int matId) {
			AppendQuad(pos, norms, tris, matIds, f3(v0), f3(v1), f3(v2), f3(v3), f3(n), uint16_t(matId));
			built = false;
		}
		void add_sphere(const Vec3& center, float radius, int matId) {
			AppendSphereMesh(pos, norms, tris, matIds, f3(center), radius, uint16_t(matId));
			built = false;
		}
		void add_diamond(const Vec3& center, float girdleR, float tableR,
						 float crownH, float pavilionD, int matId, const Vec3& axis) {
			AppendDiamondMesh(pos, norms, tris, matIds, f3(center), girdleR, tableR, crownH, pavilionD,
							  uint16_t(matId), f3(axis));
			built = false;
		}
		void add_box_walls(int floor_, int ceil_, int back_, int left_, int right_, float scale) {
			AppendBoxWalls(pos, norms, tris, matIds,
						   uint16_t(floor_), uint16_t(ceil_), uint16_t(back_),
						   uint16_t(left_), uint16_t(right_), scale);
			built = false;
		}

		using F32Arr = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
		using U32Arr = nb::ndarray<const uint32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

		// Arbitrary indexed triangle mesh. vertices: (N,3) float32; faces: (M,3)
		// uint32 into vertices; normals: optional (N,3) float32 for smooth shading
		// (flat geometric normals if omitted). See vera.shapes for generators.
		void add_mesh(F32Arr vertices, U32Arr faces, int matId, std::optional<F32Arr> normals) {
			if (vertices.shape(1) != 3) throw std::invalid_argument("vertices must be (N, 3)");
			if (faces.shape(1) != 3) throw std::invalid_argument("faces must be (M, 3)");
			if (normals && (normals->shape(0) != vertices.shape(0) || normals->shape(1) != 3))
				throw std::invalid_argument("normals must be (N, 3) matching vertices");

			AppendMesh(pos, norms, tris, matIds,
					   reinterpret_cast<const float3*>(vertices.data()), uint32_t(vertices.shape(0)),
					   faces.data(), uint32_t(faces.shape(0) * 3),
					   normals ? reinterpret_cast<const float3*>(normals->data()) : nullptr,
					   uint16_t(matId));
			built = false;
		}

		size_t triangle_count() const { return matIds.size(); }

		void set_env_constant(const Vec3& rgb, float intensity) {
			FreeImageEnvMap(envMap);
			envMap = MakeConstEnvMap(spec_table(), f3(rgb), intensity);
		}
		void set_env_hdri(const std::string& path, float intensity) {
			FreeImageEnvMap(envMap);
			HdrImage img = LoadHdr(path.c_str());
			if (!img.ok())
				throw std::runtime_error("failed to load HDRI '" + path + "'");
			HdrEnvSource src{ img.pixels.data(), img.width, img.height };
			envMap = MakeImageEnvMap(spec_table(), src, intensity);
		}

		void build() {
			if (built) return;
			if (pos.empty())
				throw std::runtime_error("Scene has no geometry");

			float3 lo = pos[0], hi = pos[0];
			for (const float3& p : pos) {
				lo.x = fminf(lo.x, p.x); lo.y = fminf(lo.y, p.y); lo.z = fminf(lo.z, p.z);
				hi.x = fmaxf(hi.x, p.x); hi.y = fmaxf(hi.y, p.y); hi.z = fmaxf(hi.z, p.z);
			}
			lo.x -= bboxPad; lo.y -= bboxPad; lo.z -= bboxPad;
			hi.x += bboxPad; hi.y += bboxPad; hi.z += bboxPad;

			release_gpu();
			UploadScene(pos, norms, tris, matIds,
						materials.data(), uint32_t(materials.size()), uint32_t(sizeof(Material)),
						lo, hi, gpu);
			lighting = BuildSceneLighting(pos, matIds, materials, media);
			cuda_check("Scene.build");
			built = true;
		}

		void release_gpu() {
			if (gpu.d_pos || gpu.matCount) FreeScene(gpu);
			FreeSceneLighting(lighting);
			built = false;
		}
		void release() {
			release_gpu();
			FreeImageEnvMap(envMap);
		}
	};

	// -- render --------------------------------------------------------------
	nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 3>>
		render(Scene& scene, const Camera& camera,
			   uint32_t spp, uint32_t max_bounces,
			   const std::string& tonemap, float exposure, bool use_optix,
			   uint32_t default_medium) {
		scene.build();

		const uint32_t W = camera.m_Width, H = camera.m_Height;
		if (W == 0 || H == 0)
			throw std::runtime_error("camera width/height not set");

		FrameBufferHWSS fb = AllocFrameBuffer(W, H);
		float4* d_outRGB = nullptr;
		cudaMalloc(&d_outRGB, size_t(W) * H * sizeof(float4));
		cuda_check("render: alloc");

		Material* d_materials = (Material*)scene.gpu.d_mats;
		ToneMapper tm = parse_tonemap(tonemap);

		{
			nb::gil_scoped_release nogil;
			RenderHWSS(scene.gpu.geom, d_materials, scene.lighting.d_media, scene.lighting.lightBvh,
					   scene.envMap, camera, fb, d_outRGB, spp, max_bounces,
					   default_medium, tm, exposure, use_optix);
			cudaDeviceSynchronize();
		}
		cuda_check("render: RenderHWSS");

		std::vector<float4> host(size_t(W) * H);
		cudaMemcpy(host.data(), d_outRGB, host.size() * sizeof(float4), cudaMemcpyDeviceToHost);
		cudaFree(d_outRGB);
		FreeFrameBuffer(fb);
		cuda_check("render: copyback");

		float* out = new float[size_t(W) * H * 3];
		for (size_t i = 0; i < size_t(W) * H; ++i) {
			out[i * 3 + 0] = host[i].x;
			out[i * 3 + 1] = host[i].y;
			out[i * 3 + 2] = host[i].z;
		}
		nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
		return { out, { H, W, 3 }, owner };
	}

	// -- render_xyz ---------------------------------------------------------
	// Raw per-pixel CIE XYZ radiance: the accumulated framebuffer divided by
	// (spp * CIE_Y_Integral), before any tonemap/exposure/sRGB. This is the space
	// the renderer actually integrates in, so it's the target for the fluorescence
	// tests' cross-checks against the DBR oracle's spectral radiance.
	nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 3>>
		render_xyz(Scene& scene, const Camera& camera,
				   uint32_t spp, uint32_t max_bounces, bool use_optix,
				   uint32_t default_medium) {
		scene.build();

		const uint32_t W = camera.m_Width, H = camera.m_Height;
		if (W == 0 || H == 0)
			throw std::runtime_error("camera width/height not set");

		FrameBufferHWSS fb = AllocFrameBuffer(W, H);
		float4* d_outRGB = nullptr;
		cudaMalloc(&d_outRGB, size_t(W) * H * sizeof(float4));
		cuda_check("render_xyz: alloc");

		Material* d_materials = (Material*)scene.gpu.d_mats;
		{
			nb::gil_scoped_release nogil;
			RenderHWSS(scene.gpu.geom, d_materials, scene.lighting.d_media, scene.lighting.lightBvh,
					   scene.envMap, camera, fb, d_outRGB, spp, max_bounces,
					   default_medium, ToneMapper::Raw, 1.f, use_optix);
			cudaDeviceSynchronize();
		}
		cuda_check("render_xyz: RenderHWSS");

		// fb.d_accumXYZ survives RenderHWSS (only the CIE texture is freed there); it
		// holds the running sum over `spp` samples of per-lane XYZ contributions.
		std::vector<float3> host(size_t(W) * H);
		cudaMemcpy(host.data(), fb.d_accumXYZ, host.size() * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaFree(d_outRGB);
		FreeFrameBuffer(fb);
		cuda_check("render_xyz: copyback");

		const float norm = 1.f / (float(spp) * CIE_Y_Integral(LAMBDA_MIN, LAMBDA_MAX));
		float* out = new float[size_t(W) * H * 3];
		for (size_t i = 0; i < size_t(W) * H; ++i) {
			out[i * 3 + 0] = host[i].x * norm;
			out[i * 3 + 1] = host[i].y * norm;
			out[i * 3 + 2] = host[i].z * norm;
		}
		nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
		return { out, { H, W, 3 }, owner };
	}

	// -- Camera convenience constructor -------------------------------------
	Camera camera_look_at(const Vec3& eye, const Vec3& target, const Vec3& up,
						  float fov_y_deg, uint32_t width, uint32_t height) {
		float3 e = f3(eye), t = f3(target), u = f3(up);
		float3 f = t - e;
		f = f * (1.f / sqrtf(dot(f, f)));
		float3 r = cross(f, u); r = r * (1.f / sqrtf(dot(r, r)));
		float3 cu = cross(r, f);

		Camera c{};
		c.m_Origin = e; c.m_Forward = f; c.m_Right = r; c.m_Up = cu;
		c.m_HalfTanFovY = tanf(0.5f * fov_y_deg * 3.14159265358979324f / 180.f);
		c.m_AspectRatio = float(width) / float(height);
		c.m_LensRadius = 0.f;
		c.m_FocusDist = 1.f;
		c.m_Width = width; c.m_Height = height;
		return c;
	}
} // namespace

NB_MODULE(_vera, m) {
	m.doc() = "Vera spectral path tracer - Python frontend (v1 skeleton)";

	nb::class_<Material>(m, "Material");
	nb::class_<MediumHWSS>(m, "Medium");

	// material factories
	m.def("lambertian", [](Vec3 albedo) { return MakeLambertian(spec_table(), f3(albedo)); },
		  nb::arg("albedo"));
	m.def("ggx", [](Vec3 f0, float roughness) { return MakeGGX(spec_table(), f3(f0), roughness); },
		  nb::arg("f0"), nb::arg("roughness"));
	m.def("dielectric", [](float ior, float roughness, int medium) {
		return MakeDielectric(ior, roughness, uint8_t(medium)); },
		nb::arg("ior") = 1.5f, nb::arg("roughness") = 0.f, nb::arg("medium") = 0);
	m.def("dispersive_dielectric", [](float b1, float c1, float b2, float c2, float roughness, int medium) {
		return MakeDispersiveDielectric(b1, c1, b2, c2, roughness, uint8_t(medium)); },
		nb::arg("sell_b1"), nb::arg("sell_c1"), nb::arg("sell_b2"), nb::arg("sell_c2"),
		nb::arg("roughness") = 0.f, nb::arg("medium") = 0);
	m.def("emissive", [](Vec3 color, float intensity) { return MakeEmissive(spec_table(), f3(color), intensity); },
		  nb::arg("color"), nb::arg("intensity"));
	m.def("fluorescent_lambertian",
		  [](Vec3 albedo, float lam_ex, float lam_em, float sigma, float quantum_yield) {
			  return MakeFluorescentLambertian(spec_table(), f3(albedo), lam_ex, lam_em, sigma, quantum_yield); },
			  nb::arg("albedo"), nb::arg("lam_ex"), nb::arg("lam_em"),
			  nb::arg("sigma") = 15.f, nb::arg("quantum_yield") = 1.f);
	m.def("medium", [](Vec3 absorb, float absorb_scale, Vec3 scatter, float scatter_scale, float g) {
		return MakeMedium(spec_table(), f3(absorb), absorb_scale, f3(scatter), scatter_scale, g); },
		nb::arg("absorb"), nb::arg("absorb_scale"), nb::arg("scatter"), nb::arg("scatter_scale"),
		nb::arg("g") = 0.f);
	m.def("fluorescent_medium",
		  [](Vec3 absorb, float absorb_scale, Vec3 scatter, float scatter_scale, float g,
			 float lam_ex, float lam_em, float sigma, float fluor_sigma_s, float quantum_yield) {
			  return MakeFluorescentMedium(spec_table(), f3(absorb), absorb_scale, f3(scatter),
										   scatter_scale, g, lam_ex, lam_em, sigma, fluor_sigma_s, quantum_yield); },
		  nb::arg("absorb") = Vec3{ 0.f, 0.f, 0.f }, nb::arg("absorb_scale") = 0.f,
		  nb::arg("scatter") = Vec3{ 1.f, 1.f, 1.f }, nb::arg("scatter_scale") = 0.f,
		  nb::arg("g") = 0.f, nb::arg("lam_ex"), nb::arg("lam_em"),
		  nb::arg("sigma") = 15.f, nb::arg("fluor_sigma_s"), nb::arg("quantum_yield") = 1.f);
	m.def("heterogeneous_fluorescent_medium",
		  [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> density,
			 Vec3 grid_min, Vec3 grid_max,
			 Vec3 absorb, float absorb_scale, Vec3 scatter, float scatter_scale, float g,
			 float lam_ex, float lam_em, float sigma, float fluor_sigma_s, float quantum_yield) {
			  // EvalDensity indexes x fastest, so a C-contiguous array must be (Z,Y,X).
			  int3 res = make_int3((int)density.shape(2), (int)density.shape(1), (int)density.shape(0));
			  return MakeHeterogeneousFluorescentMedium(
				  spec_table(), f3(absorb), absorb_scale, f3(scatter), scatter_scale, g,
				  lam_ex, lam_em, sigma, fluor_sigma_s, quantum_yield,
				  density.data(), res, f3(grid_min), f3(grid_max)); },
		  nb::arg("density"), nb::arg("grid_min"), nb::arg("grid_max"),
		  nb::arg("absorb") = Vec3{ 0.f, 0.f, 0.f }, nb::arg("absorb_scale") = 0.f,
		  nb::arg("scatter") = Vec3{ 1.f, 1.f, 1.f }, nb::arg("scatter_scale") = 0.f,
		  nb::arg("g") = 0.f, nb::arg("lam_ex"), nb::arg("lam_em"),
		  nb::arg("sigma") = 15.f, nb::arg("fluor_sigma_s"), nb::arg("quantum_yield") = 1.f,
		  "density: (Z,Y,X) float32 grid (C-contiguous -> x fastest, matching EvalDensity).");

	nb::class_<Camera>(m, "Camera")
		.def(nb::init<>())
		.def_rw("half_tan_fov_y", &Camera::m_HalfTanFovY)
		.def_rw("aspect_ratio", &Camera::m_AspectRatio)
		.def_rw("lens_radius", &Camera::m_LensRadius)
		.def_rw("focus_dist", &Camera::m_FocusDist)
		.def_rw("width", &Camera::m_Width)
		.def_rw("height", &Camera::m_Height);
	m.def("camera_look_at", &camera_look_at,
		  nb::arg("eye"), nb::arg("target"), nb::arg("up") = Vec3{ 0.f, 1.f, 0.f },
		  nb::arg("fov_y_deg") = 52.f, nb::arg("width") = 1920, nb::arg("height") = 1080);

	nb::class_<Scene>(m, "Scene")
		.def(nb::init<>())
		.def_rw("bbox_pad", &Scene::bboxPad)
		.def("add_material", &Scene::add_material, nb::arg("material"))
		.def("add_medium", &Scene::add_medium, nb::arg("medium"))
		.def("add_quad", &Scene::add_quad,
			 nb::arg("v0"), nb::arg("v1"), nb::arg("v2"), nb::arg("v3"), nb::arg("normal"), nb::arg("material"))
		.def("add_sphere", &Scene::add_sphere, nb::arg("center"), nb::arg("radius"), nb::arg("material"))
		.def("add_diamond", &Scene::add_diamond,
			 nb::arg("center"), nb::arg("girdle_radius"), nb::arg("table_radius"),
			 nb::arg("crown_height"), nb::arg("pavilion_depth"), nb::arg("material"),
			 nb::arg("axis") = Vec3{ 0.f, 1.f, 0.f })
		.def("add_box_walls", &Scene::add_box_walls,
			 nb::arg("floor"), nb::arg("ceil"), nb::arg("back"), nb::arg("left"), nb::arg("right"),
			 nb::arg("scale") = 1.f)
		.def("add_mesh", &Scene::add_mesh,
			 nb::arg("vertices"), nb::arg("faces"), nb::arg("material"),
			 nb::arg("normals") = nb::none(),
			 "Append an indexed triangle mesh. vertices (N,3) float32, faces (M,3) uint32, "
			 "optional per-vertex normals (N,3) float32 for smooth shading.")
		.def("set_env_constant", &Scene::set_env_constant, nb::arg("rgb"), nb::arg("intensity"))
		.def("set_env_hdri", &Scene::set_env_hdri, nb::arg("path"), nb::arg("intensity") = 1.f)
		.def("build", &Scene::build)
		.def("release", &Scene::release)
		.def_prop_ro("triangle_count", &Scene::triangle_count)
		.def("__repr__", [](const Scene& s) {
		return "<vera.Scene " + std::to_string(s.matIds.size()) + " tris, "
			+ std::to_string(s.materials.size()) + " materials>";
			 });

	m.def("render", &render,
		  nb::arg("scene"), nb::arg("camera"),
		  nb::arg("spp") = 256, nb::arg("max_bounces") = 32,
		  nb::arg("tonemap") = "aces", nb::arg("exposure") = 1.f, nb::arg("use_optix") = false,
		  nb::arg("default_medium") = 0,
		  "Render `scene` from `camera`; returns tonemapped (H, W, 3) float32 in [0, 1]. "
		  "default_medium is the 1-indexed medium the camera starts inside (0 = vacuum).");
	m.def("render_xyz", &render_xyz,
		  nb::arg("scene"), nb::arg("camera"),
		  nb::arg("spp") = 256, nb::arg("max_bounces") = 32, nb::arg("use_optix") = false,
		  nb::arg("default_medium") = 0,
		  "Raw per-pixel CIE XYZ radiance (no tonemap / exposure / sRGB); the space "
		  "the renderer integrates in. Target for spectral-oracle cross-checks.");
}