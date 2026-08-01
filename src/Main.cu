#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <Triangle.h>

#include <Scene.cuh>
#include <Camera.cuh>

#include "MaterialHWSS.h"
#include "EnvMapHWSS.h"
#include "SceneHWSS.h"
#include "RendererHWSS.cuh"

using namespace BSPT::Spectral::HWSS;

static void WritePPM(const char* path, const float4* pixels, uint32_t width, uint32_t height) {
	FILE* f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "Failed to open '%s' for writing\n", path); return; }
	fprintf(f, "P6\n%u %u\n255\n", width, height);
	std::vector<unsigned char> row(width * 3);
	for (uint32_t y = 0; y < height; ++y) {
		for (uint32_t x = 0; x < width; ++x) {
			float4 p = pixels[y * width + x];
			row[x * 3 + 0] = (unsigned char)(fminf(fmaxf(p.x, 0.f), 1.f) * 255.f + 0.5f);
			row[x * 3 + 1] = (unsigned char)(fminf(fmaxf(p.y, 0.f), 1.f) * 255.f + 0.5f);
			row[x * 3 + 2] = (unsigned char)(fminf(fmaxf(p.z, 0.f), 1.f) * 255.f + 0.5f);
		}
		fwrite(row.data(), 1, row.size(), f);
	}
	fclose(f);
}

int main() {
	// ── Materials ──────────────────────────────────────────────────────────
	RGB2SpecTable rgb2spec = RGB2SpecTable::Load(BSPT_DATA_DIR "srgb.coeff");
	if (rgb2spec.res == 0) {
		fprintf(stderr, "Failed to load RGB2Spec table — aborting.\n");
		return 1;
	}

	std::vector<Material> hMaterials;
	hMaterials.push_back(MakeLambertian(rgb2spec, make_float3(0.73f, 0.73f, 0.73f))); // 0: walls
	hMaterials.push_back(MakeEmissive(rgb2spec, make_float3(1.f, 1.f, 1.f), 1.f));     // 1: ceiling light
	hMaterials.push_back(MakeLambertian(rgb2spec, make_float3(0.75f, 0.1f, 0.1f)));    // 2: red sphere
	// 3: dispersive glass sphere — cauchyB is exaggerated well past real glass (~0.007um^2
	// for BK7) so the chromatic splitting HWSS is built for is actually visible in a single
	// render rather than needing a spectroscope.
	hMaterials.push_back(MakeDielectric(/*ior*/ 1.5f, /*roughness*/ 0.f, /*mediumIdx*/ 0, /*cauchyB*/ 0.008f));

	// ── Geometry: five-wall box room + two spheres ───────────────────────────
	std::vector<float3>        hPos, hNorms;
	std::vector<NXB::Triangle> hTris;
	std::vector<uint16_t>      hMatIds;

	Core::AppendBoxWalls(hPos, hNorms, hTris, hMatIds, 0, 1, 0, 0, 0);
	Core::AppendSphereMesh(hPos, hNorms, hTris, hMatIds, make_float3(-0.45f, -0.65f, -0.1f), 0.35f, 2);
	Core::AppendSphereMesh(hPos, hNorms, hTris, hMatIds, make_float3(0.45f, -0.55f, 0.3f), 0.45f, 3);

	Core::SceneGPU scene{};
	Core::UploadScene(
		hPos, hNorms, hTris, hMatIds,
		hMaterials.data(), (uint32_t)hMaterials.size(), (uint32_t)sizeof(Material),
		make_float3(-1.05f, -1.05f, -1.05f), make_float3(1.05f, 1.05f, 1.05f),
		scene);

	Material* d_materials = (Material*)scene.d_mats;

	SceneLightingHWSS lighting = BuildSceneLighting(hPos, hMatIds, hMaterials, {});

	EnvMapHWSS envMap = MakeConstEnvMap(rgb2spec, make_float3(0.f, 0.f, 0.f), 0.f);

	// ── Camera ────────────────────────────────────────────────────────────
	uint32_t width = 1920, height = 1080;
	Core::Camera camera{};
	camera.m_Origin       = make_float3(0.f, 0.f, 3.4f);
	camera.m_Forward      = make_float3(0.f, 0.f, -1.f);
	camera.m_Right        = make_float3(1.f, 0.f, 0.f);
	camera.m_Up           = make_float3(0.f, 1.f, 0.f);
	camera.m_HalfTanFovY  = tanf(0.5f * 45.f * 3.14159265f / 180.f);
	camera.m_AspectRatio  = (float)width / (float)height;
	camera.m_LensRadius   = 0.f;
	camera.m_FocusDist    = 1.f;
	camera.m_Width        = width;
	camera.m_Height       = height;

	// ── Render ────────────────────────────────────────────────────────────
	FrameBufferHWSS fb = AllocFrameBuffer(width, height);
	float4* d_outRGB = nullptr;
	cudaMalloc(&d_outRGB, (size_t)width * height * sizeof(float4));

	printf("Rendering %ux%u...\n", width, height);
	RenderHWSS(scene.geom, d_materials, lighting.d_media, lighting.lightBvh, envMap,
		camera, fb, d_outRGB, /*samplesPerPixel*/ 512, /*maxBounces*/ 16,
		/*defaultMediumIdx*/ 0, Core::ToneMapper::ACES, /*exposure*/ 1.2f);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		return 1;
	}

	std::vector<float4> hOut(width * height);
	cudaMemcpy(hOut.data(), d_outRGB, hOut.size() * sizeof(float4), cudaMemcpyDeviceToHost);
	WritePPM("output.ppm", hOut.data(), width, height);
	printf("Wrote output.ppm\n");

	cudaFree(d_outRGB);
	FreeFrameBuffer(fb);
	FreeSceneLighting(lighting);
	Core::FreeScene(scene);
	RGB2SpecTable::Free(rgb2spec);
	return 0;
}
