#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

namespace BSPT::Spectral::HWSS {
	struct LightSample {
		float3   position;
		float3   normal;
		uint16_t materialId; // evaluate emission via EvalEmission(materials[materialId], lambda)
		float    pdf;
		uint32_t lightIdx;
	};

	struct LightEntry {
		uint16_t materialId;
		float3   coneAxis;
		float    cosConeAngle;
		float    totalArea;
		float    flux;           // single-wavelength (555nm) proxy, importance-sampling weight only
		uint32_t firstGlobalTri; // global primIdx of first triangle in this group
		uint32_t firstTri;       // index into triAreaCDF[]
		uint32_t triCount;
	};

	struct LightBVHNode {
		float3   bMin;
		float3   bMax;
		float3   coneAxis;
		float    cosConeAngle;
		float    flux;
		uint32_t leftChild;
		uint32_t rightChild;
		uint32_t lightIdx;
	};

	struct LightBVH {
		LightBVHNode* nodes          = nullptr;
		uint32_t      nodeCount      = 0;
		LightEntry*   lights         = nullptr;
		uint32_t      lightCount     = 0;
		float*        triAreaCDF     = nullptr;
		uint32_t*     primToLight    = nullptr; // primToLight[globalPrimIdx] = lightIdx, 0xFFFFFFFF if not a light
		uint32_t      totalPrimCount = 0;
	};
}
