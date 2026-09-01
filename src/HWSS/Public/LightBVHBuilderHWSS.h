#pragma once
#include "LightBVHHWSS.h"
#include "MaterialHWSS.h"

namespace Vera::Spectral::HWSS {
	LightBVH BuildLightBVH(
		const float3* hostVertices,
		const uint32_t* hostIndices,
		const uint16_t* hostTriMatIDs,
		uint32_t        totalTriCount,
		const Material* hostMaterials
	);

	void FreeLightBVH(LightBVH& bvh);
}
