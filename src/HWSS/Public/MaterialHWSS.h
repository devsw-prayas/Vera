#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "RGB2Spec.h"

namespace Vera::Spectral::HWSS {
	enum class MaterialType : uint8_t {
		Lambertian,
		GGX,
		Emissive,
		Dielectric,
	};

	// Spectral material. Reflectance/emission color is stored as sigmoid-polynomial
	// coefficients (Jakob & Hanika 2019) baked once on the host from an artist RGB
	// value via RGB2SpecLookup — evaluating a material at a sampled wavelength is
	// then just RGBSigmoidPolynomial::operator()(lambda), cheap enough per-sample.
	//
	// GGX conductors are Schlick-only in this model: exact complex-IOR conductor
	// Fresnel would need real measured spectral eta/kappa curves per metal, which
	// RGB2Spec (designed for [0,1] reflectance) can't represent, so F0 is baked
	// the same way as Lambertian albedo instead.
	//
	// Dielectric dispersion is optional: cauchyB == 0 means a constant IOR; a
	// nonzero cauchyB enables wavelength-dependent IOR via the Cauchy equation
	// (see EvalIOR) for chromatic dispersion (e.g. prism/glass fringing).
	struct Material final {
		RGBSigmoidPolynomial albedoSpec;   // Lambertian reflectance / GGX Schlick F0 / Emissive base color
		float                emissiveScale = 0.f; // Emissive: radiance multiplier; unused otherwise
		float                roughness     = 0.f;
		float                ior           = 1.5f; // Dielectric base IOR (at reference wavelength)
		float                cauchyB       = 0.f;  // Dielectric dispersion strength (um^2); 0 = achromatic
		MaterialType         type          = MaterialType::Lambertian;
		uint8_t              mediumIdx     = 0;    // Dielectric interior medium (1-indexed, 0 = vacuum)
		uint8_t              pad[2]        = {};
	};

	__host__ __device__ inline float EvalReflectance(const Material& mat, float lambda) {
		return mat.albedoSpec(lambda);
	}

	__host__ __device__ inline float EvalEmission(const Material& mat, float lambda) {
		return mat.albedoSpec(lambda) * mat.emissiveScale;
	}

	// Cauchy's equation: n(lambda) = ior + cauchyB / lambda_um^2. lambda is in nm.
	__host__ __device__ inline float EvalIOR(const Material& mat, float lambda) {
		if (mat.cauchyB == 0.f) return mat.ior;
		float lambdaUm = lambda * 0.001f;
		return mat.ior + mat.cauchyB / (lambdaUm * lambdaUm);
	}

	// ── Host-side factories: bake an artist RGB color into spectral coefficients ──

	inline Material MakeLambertian(const RGB2SpecTable& table, float3 albedoRGB) {
		Material m{};
		m.type = MaterialType::Lambertian;
		m.albedoSpec = RGB2SpecLookup(table, albedoRGB);
		return m;
	}

	inline Material MakeGGX(const RGB2SpecTable& table, float3 f0RGB, float roughness) {
		Material m{};
		m.type = MaterialType::GGX;
		m.albedoSpec = RGB2SpecLookup(table, f0RGB);
		m.roughness = roughness;
		return m;
	}

	inline Material MakeDielectric(float ior, float roughness, uint8_t mediumIdx = 0, float cauchyB = 0.f) {
		Material m{};
		m.type = MaterialType::Dielectric;
		m.ior = ior;
		m.roughness = roughness;
		m.mediumIdx = mediumIdx;
		m.cauchyB = cauchyB;
		return m;
	}

	inline Material MakeEmissive(const RGB2SpecTable& table, float3 colorRGB, float intensity) {
		Material m{};
		m.type = MaterialType::Emissive;
		m.albedoSpec = RGB2SpecLookup(table, colorRGB);
		m.emissiveScale = intensity;
		return m;
	}
}
