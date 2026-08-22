#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <Tonemap.h>

namespace Vera::Spectral::HWSS {
	// Accumulates per-pixel CIE XYZ (not RGB) since each HWSS sample only ever
	// contributes at 4 discrete wavelengths — XYZ is the natural accumulation
	// space, converted to display RGB once at resolve time.
	struct FrameBufferHWSS final {
		float3*  d_accumXYZ = nullptr;
		uint32_t width  = 0;
		uint32_t height = 0;
	};

	FrameBufferHWSS AllocFrameBuffer(uint32_t width, uint32_t height);
	void FreeFrameBuffer(FrameBufferHWSS& fb);

	// spp: total samples accumulated per pixel; cieYIntegral: CIE_Y_Integral(LAMBDA_MIN, LAMBDA_MAX).
	__global__ void ResolveFrameBufferKernel(
		FrameBufferHWSS fb, uint32_t spp, float cieYIntegral,
		Core::ToneMapper tm, float exposure, float4* outRGB);
}
