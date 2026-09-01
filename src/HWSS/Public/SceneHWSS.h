#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "MaterialHWSS.h"
#include "MediumHWSS.h"
#include "LightBVHHWSS.h"
#include "LightBVHBuilderHWSS.h"

namespace Vera::Spectral::HWSS {
	// Lighting-side companion to Core::SceneGPU: a LightBVH over the Emissive triangles
	// plus an uploaded medium buffer. Separate from Core::UploadScene because it needs the
	// HWSS Material layout, which Core doesn't know.
	struct SceneLightingHWSS final {
		LightBVH    lightBvh{};
		MediumHWSS* d_media = nullptr;
		uint32_t    mediaCount = 0;
	};

	// hPos/hMatIds must be the same arrays passed to Core::UploadScene (unindexed soup);
	// the identity index buffer is rebuilt here to match.
	inline SceneLightingHWSS BuildSceneLighting(
		const std::vector<float3>& hPos,
		const std::vector<uint16_t>& hMatIds,
		const std::vector<Material>& hMaterials,
		const std::vector<MediumHWSS>& hMedia) {
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
