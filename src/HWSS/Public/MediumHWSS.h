#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

#include "RGB2Spec.h"

namespace BSPT::Spectral::HWSS {
	// Homogeneous participating medium, spectral. Coefficients are physical
	// (not bounded to [0,1]), so — unlike Material's reflectance — they're
	// baked as a normalized spectral *shape* (RGB2Spec, in [0,1]) times a
	// separate positive scale, the same pattern used for Material::emissiveScale
	// and EnvMapHWSS::intensity.
	struct MediumHWSS final {
		RGBSigmoidPolynomial sigmaAShape;
		float                sigmaAScale = 0.f; // absorption coefficient magnitude
		RGBSigmoidPolynomial sigmaSShape;
		float                sigmaSScale = 0.f; // scattering coefficient magnitude
		float                g = 0.f;           // Henyey-Greenstein anisotropy [-1,1]
	};

	__host__ __device__ inline float EvalSigmaA(const MediumHWSS& m, float lambda) {
		return m.sigmaAShape(lambda) * m.sigmaAScale;
	}

	__host__ __device__ inline float EvalSigmaS(const MediumHWSS& m, float lambda) {
		return m.sigmaSShape(lambda) * m.sigmaSScale;
	}

	__host__ __device__ inline float EvalSigmaT(const MediumHWSS& m, float lambda) {
		return EvalSigmaA(m, lambda) + EvalSigmaS(m, lambda);
	}

	inline MediumHWSS MakeMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g)
	{
		MediumHWSS m{};
		m.sigmaAShape = RGB2SpecLookup(table, absorbColorRGB);
		m.sigmaAScale = absorbScale;
		m.sigmaSShape = RGB2SpecLookup(table, scatterColorRGB);
		m.sigmaSScale = scatterScale;
		m.g = g;
		return m;
	}
}
