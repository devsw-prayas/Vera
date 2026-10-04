#pragma once
#include <cuda_runtime.h>
#include <CoreUtils.h>
#include "RaySHW.h"
#include "MaterialHWSS.h"
#include "MediumHWSS.h"
#include "LightBVHHWSS.h"
#include "EnvMapHWSS.h"
#include "FrameBufferHWSS.cuh"

namespace Vera::SHW {
	__global__ void ShadeKernelSHWWavefront(
		Core::GeometryBuffers geom,
		RayCoreSoA coreIn,
		RayExtSoA extIn,
		const Core::WavefrontHitRecord* __restrict__ hits,
		const uint32_t* __restrict__ order,
		const Spectral::HWSS::Material* __restrict__ materials,
		const Spectral::HWSS::MediumHWSS* __restrict__ media,
		Spectral::HWSS::LightBVH lightBvh,
		RayCoreSoA coreOut,
		RayExtSoA extOut,
		uint32_t rayCount,
		Spectral::HWSS::FrameBufferHWSS fb,
		Spectral::HWSS::EnvMapHWSS envMap,
		uint32_t maxBounces,
		cudaTextureObject_t cieTex);
}
