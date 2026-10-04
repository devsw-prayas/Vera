#pragma once
#include <cuda_runtime.h>
#include <Camera.cuh>
#include "RaySHW.h"

namespace Vera::SHW {
	__global__ void GeneratePrimaryRaysSHWKernel(
		Core::Camera camera,
		RayCoreSoA core, RayExtSoA ext,
		uint32_t sampleIdx,
		uint8_t defaultMediumIdx);
}
