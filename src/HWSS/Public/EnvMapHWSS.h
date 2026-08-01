#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

#include "RGB2Spec.h"

namespace BSPT::Spectral::HWSS {
	// Constant-radiance environment (uniform sky/background). Baked the same
	// way as an emissive Material: an artist RGB color upsampled once on the
	// host via RGB2SpecLookup, then scaled by intensity per shading sample.
	struct EnvMapHWSS final {
		RGBSigmoidPolynomial colorSpec;
		float                intensity = 0.f;
	};

	__host__ __device__ inline float EvalEnvMap(const EnvMapHWSS& env, float lambda) {
		return env.colorSpec(lambda) * env.intensity;
	}

	inline EnvMapHWSS MakeConstEnvMap(const RGB2SpecTable& table, float3 colorRGB, float intensity) {
		EnvMapHWSS env{};
		env.colorSpec = RGB2SpecLookup(table, colorRGB);
		env.intensity = intensity;
		return env;
	}
}
