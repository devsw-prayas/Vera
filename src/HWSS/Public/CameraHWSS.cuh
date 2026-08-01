#pragma once
#include <cuda_runtime.h>
#include "Camera.cuh"
#include "RayHWSS.h"

namespace BSPT::Spectral::HWSS {
	__global__ void GeneratePrimaryRaysHWSSKernel(
		Core::Camera camera, RayHWSS* rays,
		unsigned int sampleIdx = 0, unsigned char defaultMediumIdx = 0);
}
