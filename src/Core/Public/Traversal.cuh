#pragma once
#include <cuda_runtime.h>
#include <BVH.h>
#include <Triangle.h>

#include "CoreUtils.h"
#include "RayHWSS.h"

namespace Vera::Core {
	// Shared BVH-walk primitives: decoupled from any ray representation.
	// Both functions operate on raw origin/direction/tmin/tmax — no Ray or RayHWSS.

	// Full closest-hit TLAS->BLAS walk. Returns the nearest intersection found
	// (or h.m_Hit==false if none within [tmin, tmax]).
	__device__ WavefrontHitRecord TraverseClosestHit(
		GeometryBuffers geom,
		float3 origin, float3 direction,
		float tmin, float tmax);

	// Any-hit TLAS->BLAS walk. Early-exits and returns true on the first
	// intersection found within [tmin, tmax] (shadow / occlusion test).
	__device__ bool TraverseAnyHit(
		GeometryBuffers geom,
		float3 origin, float3 direction,
		float tmin, float tmax);

	// Wavefront kernel: thin wrapper — checks RAY_FLAG_DEAD, then delegates
	// to TraverseClosestHit.  HWSS-specific because it knows about RayHWSS.
	__global__ void TraversalKernelWavefront(
		GeometryBuffers geom,
		const Spectral::HWSS::RayHWSS* __restrict__ rays,
		WavefrontHitRecord* __restrict__ hits,
		uint32_t rayCount);
}
