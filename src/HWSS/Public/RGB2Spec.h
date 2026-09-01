#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Jakob & Hanika 2019 spectral upsampling: an RGB reflectance becomes three
// sigmoid-polynomial coefficients with R(lambda) = Sigmoid(c0*lambda^2 + c1*lambda + c2).
// The RGB->coeffs LUT lookup is host-only (runs when baking Materials); only the
// polynomial eval runs on the GPU.
namespace Vera::Spectral::HWSS {
	struct RGBSigmoidPolynomial final {
		float c0 = 0.f, c1 = 0.f, c2 = 0.f;

		__host__ __device__ inline float operator()(float lambda) const {
			float v = (c0 * lambda + c1) * lambda + c2;
			return 0.5f + 0.5f * v / sqrtf(1.f + v * v);
		}
	};

	// data layout: [maxChannel(3)][zi(res)][yi(res)][xi(res)][coeff(3)]
	struct RGB2SpecTable final {
		int32_t res = 0;
		float* scale = nullptr; // host ptr, length res; monotonic remap for the max channel
		float* data = nullptr; // host ptr, length 3*res*res*res*3

		static RGB2SpecTable Load(const char* path);
		static void Free(RGB2SpecTable& table);
	};

	// Trilinear LUT lookup: linear RGB in [0,1]^3 -> sigmoid-polynomial coeffs. Host-only.
	RGBSigmoidPolynomial RGB2SpecLookup(const RGB2SpecTable& table, float3 rgb);
}
