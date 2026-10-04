#pragma once
#include <cuda_runtime.h>
#include <CoreUtils.h>
#include <Camera.cuh>
#include "RaySHW.h"
#include "CompactionSHW.cuh"

// Forward-declare to avoid pulling in the full OptixTraversal.h
#ifdef VERA_ENABLE_OPTIX
namespace Vera::Core { struct OptixTraversalContext; }
#endif

namespace Vera::Core { struct MaterialSorter; }

namespace Vera::SHW {
	// Double-buffered SoA + scratch
	struct SHWBuffers {
		RayCoreSoA coreA, coreB;
		RayExtSoA  extA,  extB;
		Core::WavefrontHitRecord* d_hits = nullptr;
		RayCompactorSHW compactor;
		uint32_t maxRays = 0;
	};

	SHWBuffers AllocSHWBuffers(uint32_t maxRays);
	void FreeSHWBuffers(SHWBuffers& buf);

	// Individual wavefront steps - consumer assembles the loop
	void GenerateSHW(
		const Core::Camera& camera,
		RayCoreSoA core, RayExtSoA ext,
		uint32_t sampleIdx, uint8_t defaultMediumIdx);

	void TraverseSHW(
		const Core::GeometryBuffers& geom,
		RayCoreSoA core,
		Core::WavefrontHitRecord* hits,
		uint32_t rayCount,
		bool useOptix
#ifdef VERA_ENABLE_OPTIX
		, Core::OptixTraversalContext* optixCtx = nullptr
#endif
	);

	// Returns device pointer to the sorted order permutation
	uint32_t* SortSHW(
		Core::MaterialSorter& sorter,
		Core::GeometryBuffers geom,
		Core::WavefrontHitRecord* hits,
		uint32_t count);

	uint32_t CompactSHW(
		RayCompactorSHW& compactor,
		RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
		RayCoreSoA coreOut, RayExtSoA extOut);
}
