// OptiX device programs for the hardware-accelerated traversal pathway. Compiled to PTX only
// (see CMakeLists.txt's VeraOptixPTX custom command) and loaded at runtime by
// OptixTraversal.cpp's optixModuleCreate -- never part of the main -rdc=true device-link unit
// the rest of Vera's wavefront kernels live in.
//
// Each program writes/reads params.hits[idx] directly by re-deriving idx from
// optixGetLaunchIndex() rather than round-tripping through optixTrace payload registers --
// simpler, and there's no cross-program state to pass here since every program already knows
// which ray it's handling.
#include <optix.h>
#include <cfloat>

#include "OptixTraversal.h"

extern "C" {
__constant__ Vera::Core::OptixTraversalParams params;
}

extern "C" __global__ void __raygen__vera()
{
	const uint32_t idx = optixGetLaunchIndex().x;

	// Defensive parity with TraversalKernelWavefront's dead-ray check (Traversal.cu) -- in the
	// current pipeline this is always false here since RayCompactor already removes dead rays
	// between bounces, but costs nothing to keep symmetric with the software path.
	if (params.rayFlags[idx] & Vera::Spectral::HWSS::RAY_FLAG_DEAD) {
		params.hits[idx].m_Hit = false;
		params.hits[idx].t     = FLT_MAX;
		return;
	}

	float3 origin    = params.rayOrigin[idx];
	float3 direction = params.rayDirection[idx];

	optixTrace(
		params.handle,
		origin,
		direction,
		1e-4f,               // tmin, matches TraverseClosestHit's tmin in Traversal.cu
		FLT_MAX,              // tmax
		0.f,                   // rayTime (no motion blur)
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,                     // SBT offset
		1,                     // SBT stride
		0                      // miss SBT index
	);
}

extern "C" __global__ void __miss__vera()
{
	const uint32_t idx = optixGetLaunchIndex().x;
	params.hits[idx].m_Hit = false;
	params.hits[idx].t     = FLT_MAX;
}

extern "C" __global__ void __closesthit__vera()
{
	const uint32_t idx = optixGetLaunchIndex().x;
	const float2 bary  = optixGetTriangleBarycentrics();

	Vera::Core::WavefrontHitRecord h{};
	h.m_Hit     = true;
	h.t         = optixGetRayTmax();
	h.m_PrimIdx = optixGetPrimitiveIndex();
	h.m_U       = bary.x;
	h.m_V       = bary.y;
	h.m_InstIdx = optixGetInstanceIndex();

	params.hits[idx] = h;
}
