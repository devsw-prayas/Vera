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

using namespace Vera::Core;
using namespace Vera::Spectral::HWSS;

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
	RGB2SpecTable rgb2spec = RGB2SpecTable::Load(VERA_DATA_DIR "srgb.coeff");
	if (rgb2spec.res == 0) {
		fprintf(stderr, "Failed to load RGB2Spec table — aborting.\n");
		return 1;
	}

	std::vector<Material> hMaterials;
	hMaterials.push_back(MakeLambertian(rgb2spec, make_float3(0.035f, 0.045f, 0.06f))); // 0: charcoal studio floor
	hMaterials.push_back(MakeEmissive(rgb2spec, make_float3(0.72f, 0.86f, 1.f), 7.f));   // 1: cool luminous backdrop
	// 2: diamond — real 2-term Sellmeier dispersion (Peter 1923), n@587.6nm ~= 2.417.
	// Verified against 3-term Sellmeier ground truth: exact match (diamond's real published
	// dispersion IS this 2-term form), vs. a Cauchy fit's ~18 arcsec refraction-angle error.
	// No interior medium: clear diamond, with color coming only from dispersion and lighting.
	hMaterials.push_back(MakeDispersiveDielectric(
		/*sellB1*/ 4.3356f, /*sellC1*/ 0.1060f * 0.1060f,
		/*sellB2*/ 0.3306f, /*sellC2*/ 0.1750f * 0.1750f,
		/*roughness*/ 0.f, /*mediumIdx*/ 0));
	hMaterials.push_back(MakeEmissive(rgb2spec, make_float3(1.f, 0.92f, 0.78f), 3.f)); // 3: ceiling light

	// ── Geometry: cinematic open studio with one luminous backdrop ────────────
	std::vector<float3>        hPos, hNorms;
	std::vector<NXB::Triangle> hTris;
	std::vector<uint16_t>      hMatIds;

	const float roomScale = 4.f;
	const float floorY = -2.f;
	const float wallTop = 4.f;
	// Cinematic open studio: a dark floor and one luminous backdrop wall.
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(-roomScale, floorY, -roomScale),
		make_float3( roomScale, floorY, -roomScale),
		make_float3( roomScale, floorY,  roomScale),
		make_float3(-roomScale, floorY,  roomScale),
		make_float3(0.f, 1.f, 0.f), 0);
	// Bright wall BL->BR, facing into the room (+Z).
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(-roomScale, floorY, -3.5f),
		make_float3( roomScale, floorY, -3.5f),
		make_float3( roomScale, wallTop, -3.5f),
		make_float3(-roomScale, wallTop, -3.5f),
		make_float3(0.f, 0.f, 1.f), 1);
	// Enclosed room: side walls and ceiling, while the rear wall above remains emissive.
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(-roomScale, floorY, -3.5f), make_float3(-roomScale, floorY, roomScale),
		make_float3(-roomScale, wallTop, roomScale), make_float3(-roomScale, wallTop, -3.5f),
		make_float3(1.f, 0.f, 0.f), 0);
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(roomScale, floorY, roomScale), make_float3(roomScale, floorY, -3.5f),
		make_float3(roomScale, wallTop, -3.5f), make_float3(roomScale, wallTop, roomScale),
		make_float3(-1.f, 0.f, 0.f), 0);
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(-roomScale, wallTop, -3.5f), make_float3(-roomScale, wallTop, roomScale),
		make_float3(roomScale, wallTop, roomScale), make_float3(roomScale, wallTop, -3.5f),
		make_float3(0.f, -1.f, 0.f), 0);
	Vera::Core::AppendQuad(hPos, hNorms, hTris, hMatIds,
		make_float3(-1.6f, wallTop - 0.05f, -0.8f), make_float3(1.6f, wallTop - 0.05f, -0.8f),
		make_float3(1.6f, wallTop - 0.05f, 1.0f), make_float3(-1.6f, wallTop - 0.05f, 1.0f),
		make_float3(0.f, -1.f, 0.f), 3);
	// Five clear diamonds: hero in front, four smaller stones staggered behind it.
	Vera::Core::AppendDiamondMesh(hPos, hNorms, hTris, hMatIds, make_float3(0.f, -1.616f, 0.15f),
		0.52f, 0.22f, 0.20f, 0.49f, 2, make_float3(0.f, 0.70f, -0.714f));
	Vera::Core::AppendDiamondMesh(hPos, hNorms, hTris, hMatIds, make_float3(-1.35f, -1.782f, 0.75f),
		0.33f, 0.14f, 0.13f, 0.31f, 2, make_float3(0.f, 0.70f, -0.714f));
	Vera::Core::AppendDiamondMesh(hPos, hNorms, hTris, hMatIds, make_float3(1.55f, -1.782f, 0.85f),
		0.33f, 0.14f, 0.13f, 0.31f, 2, make_float3(0.f, 0.70f, -0.714f));
	Vera::Core::AppendDiamondMesh(hPos, hNorms, hTris, hMatIds, make_float3(-1.85f, -1.86f, 1.75f),
		0.21f, 0.09f, 0.08f, 0.20f, 2, make_float3(0.f, 0.70f, -0.714f));
	Vera::Core::AppendDiamondMesh(hPos, hNorms, hTris, hMatIds, make_float3(2.80f, -1.86f, 1.60f),
		0.21f, 0.09f, 0.08f, 0.20f, 2, make_float3(0.f, 0.70f, -0.714f));

	Vera::Core::SceneGPU scene{};
	Vera::Core::UploadScene(
		hPos, hNorms, hTris, hMatIds,
		hMaterials.data(), (uint32_t)hMaterials.size(), (uint32_t)sizeof(Material),
		make_float3(-roomScale - 0.05f, -roomScale - 0.05f, -roomScale - 0.05f),
		make_float3( roomScale + 0.05f,  roomScale + 0.05f,  roomScale + 0.05f),
		scene);

	Material* d_materials = (Material*)scene.d_mats;

	std::vector<MediumHWSS> hMedia;
	// No participating medium for the clean comparison render.

	SceneLightingHWSS lighting = BuildSceneLighting(hPos, hMatIds, hMaterials, hMedia);

	EnvMapHWSS envMap = MakeConstEnvMap(rgb2spec, make_float3(0.f, 0.f, 0.f), 0.f);

	// ── Camera: low three-quarter cinematic composition ───────────────────────
	uint32_t width = 1920, height = 1080;
	Vera::Core::Camera camera{};
	float3 camEye    = make_float3(3.6f, -1.25f, 3.6f);
	float3 camTarget = make_float3(0.f, -1.25f, 0.45f);
	float3 worldUp   = make_float3(0.f, 1.f, 0.f);
	float3 camForward = normalize(camTarget - camEye);
	float3 camRight   = normalize(cross(camForward, worldUp));
	float3 camUp      = cross(camRight, camForward);
	camera.m_Origin       = camEye;
	camera.m_Forward      = camForward;
	camera.m_Right        = camRight;
	camera.m_Up           = camUp;
	camera.m_HalfTanFovY  = tanf(0.5f * 52.f * 3.14159265f / 180.f);
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
		camera, fb, d_outRGB, /*samplesPerPixel*/ 1024, /*maxBounces*/ 32,
		/*defaultMediumIdx*/ 0, Vera::Core::ToneMapper::ACES, /*exposure*/ 1.2f);
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
	Vera::Core::FreeScene(scene);
	RGB2SpecTable::Free(rgb2spec);
	return 0;
}
