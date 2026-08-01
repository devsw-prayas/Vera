#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector_types.h>

// Jakob & Hanika 2019 "A Low-Dimensional Function Space for Efficient Spectral
// Upsampling": an RGB reflectance is represented as three sigmoid-polynomial
// coefficients (c0,c1,c2); the reflectance at a wavelength is
//   R(lambda) = Sigmoid(c0*lambda^2 + c1*lambda + c2).
// The LUT lookup that produces (c0,c1,c2) from RGB only ever runs at scene
// build time (baking Material structs on the host); the sigmoid-polynomial
// evaluation itself runs per-shading-sample on the GPU. So the LUT stays
// host-only — no need to keep it device-resident.
namespace BSPT::Spectral::HWSS {

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
		float* data  = nullptr; // host ptr, length 3*res*res*res*3

		static RGB2SpecTable Load(const char* path);
		static void Free(RGB2SpecTable& table);
	};

	// Looks up the sigmoid-polynomial coefficients for a linear RGB reflectance
	// in [0,1]^3 via trilinear interpolation into the LUT. Host-only — called
	// once per material when baking a scene, not per shading sample.
	RGBSigmoidPolynomial RGB2SpecLookup(const RGB2SpecTable& table, float3 rgb);
}
