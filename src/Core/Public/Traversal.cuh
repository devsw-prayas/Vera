#pragma once
#include <cuda_runtime.h>
#include <BVH.h>
#include <Triangle.h>

#include "CoreUtils.h"
#include "Ray.h"
#include "HitRecord.h"
#include "RayHWSS.h"

namespace BSPT::Spectral::HWSS::Core {
	__global__ void TraversalKernel(NXB::BVH2 bvh, NXB::Triangle* triangles, Ray* rays, HitRecord* hits, uint32_t rayCount);
	__global__ void TraversalKernelWavefront(GeometryBuffers geom, Ray* rays, WavefrontHitRecord* hits, uint32_t rayCount);
	__global__ void TraversalKernelWavefront(GeometryBuffers geom, RayHWSS* rays, WavefrontHitRecord* hits, uint32_t rayCount);
}
