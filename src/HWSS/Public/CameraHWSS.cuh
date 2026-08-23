#pragma once
#include <cuda_runtime.h>
#include "Camera.cuh"
#include "RayHWSS.h"

namespace Vera::Spectral::HWSS {
	__global__ void GeneratePrimaryRaysHWSSKernel(
		Core::Camera camera, RayCoreSoA core, RayExtSoA ext,
		unsigned int sampleIdx = 0, unsigned char defaultMediumIdx = 0);
}
