#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "MaterialHWSS.h"
#include "MediumHWSS.h"
#include "LightBVHHWSS.h"
#include "LightBVHBuilderHWSS.h"

namespace BSPT::Spectral::HWSS {
	// Lighting-side companion to Core::SceneGPU: a LightBVH built from whichever
	// triangles carry an Emissive material, plus an uploaded medium buffer.
	// Built separately from Core::UploadScene because both need the (HWSS-only)
	// Material array, which Core deliberately doesn't know the layout of.
	struct SceneLightingHWSS final {
		LightBVH    lightBvh{};
		MediumHWSS* d_media     = nullptr;
		uint32_t    mediaCount  = 0;
	};

	// hPos/hMatIds must be the same host arrays passed to Core::UploadScene for
	// this scene (unindexed triangle soup: 3 unique verts per triangle) — the
	// implicit identity index buffer is reconstructed here to match.
	inline SceneLightingHWSS BuildSceneLighting(
		const std::vector<float3>&   hPos,
		const std::vector<uint16_t>& hMatIds,
		const std::vector<Material>& hMaterials,
		const std::vector<MediumHWSS>& hMedia)
	{
		SceneLightingHWSS lighting{};

		std::vector<uint32_t> hIdx(hPos.size());
		for (uint32_t i = 0; i < (uint32_t)hPos.size(); ++i) hIdx[i] = i;

		lighting.lightBvh = BuildLightBVH(
			hPos.data(), hIdx.data(), hMatIds.data(),
			(uint32_t)hMatIds.size(), hMaterials.data());

		lighting.mediaCount = (uint32_t)hMedia.size();
		if (lighting.mediaCount > 0) {
			cudaMalloc(&lighting.d_media, lighting.mediaCount * sizeof(MediumHWSS));
			cudaMemcpy(lighting.d_media, hMedia.data(), lighting.mediaCount * sizeof(MediumHWSS), cudaMemcpyHostToDevice);
		}
		return lighting;
	}

	inline void FreeSceneLighting(SceneLightingHWSS& lighting) {
		FreeLightBVH(lighting.lightBvh);
		if (lighting.d_media) cudaFree(lighting.d_media);
		lighting = SceneLightingHWSS{};
	}
}
