#include "FrameBufferHWSS.cuh"
#include "CIE.h"
#include <CudaMath.h>

namespace BSPT::Spectral::HWSS {
	FrameBufferHWSS AllocFrameBuffer(uint32_t width, uint32_t height) {
		FrameBufferHWSS fb{};
		fb.width  = width;
		fb.height = height;
		size_t count = (size_t)width * height;
		cudaMalloc(&fb.d_accumXYZ, count * sizeof(float3));
		cudaMemset(fb.d_accumXYZ, 0, count * sizeof(float3));
		return fb;
	}

	void FreeFrameBuffer(FrameBufferHWSS& fb) {
		cudaFree(fb.d_accumXYZ);
		fb = FrameBufferHWSS{};
	}

	__device__ inline float LinearToSRGB(float c) {
		c = fminf(fmaxf(c, 0.f), 1.f);
		return (c <= 0.0031308f) ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
	}

	__global__ void ResolveFrameBufferKernel(
		FrameBufferHWSS fb, uint32_t spp, float cieYIntegral,
		Core::ToneMapper tm, float exposure, float4* outRGB)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= fb.width * fb.height) return;

		float3 xyz = fb.d_accumXYZ[idx];
		float norm = 1.f / (fmaxf((float)spp, 1.f) * fmaxf(cieYIntegral, 1e-8f));
		xyz.x *= norm; xyz.y *= norm; xyz.z *= norm;

		float3 rgb = XYZToLinearSRGB(xyz);
		rgb.x *= exposure; rgb.y *= exposure; rgb.z *= exposure;
		rgb = Core::applyTonemap(tm, rgb);
		outRGB[idx] = make_float4(LinearToSRGB(rgb.x), LinearToSRGB(rgb.y), LinearToSRGB(rgb.z), 1.f);
	}
}
