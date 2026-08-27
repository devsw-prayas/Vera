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
	// Dielectric dispersion is optional: sellB1 == 0 && sellB2 == 0 means a constant IOR
	// (mat.ior); nonzero Sellmeier coefficients enable wavelength-dependent IOR via the
	// 2-term Sellmeier equation (see EvalIOR) for chromatic dispersion (e.g. prism/glass
	// fringing, diamond fire). 2-term beats a Cauchy fit strictly (measured against real
	// 3-term glass/diamond Sellmeier ground truth over 380-700nm) for the same handful of
	// extra coefficients, and — since MaterialSortHWSS groups shading by material id — the
	// extra divisions are warp-uniform within a batch, not a per-lane divergence cost.
	struct Material final {
		RGBSigmoidPolynomial albedoSpec;   // Lambertian reflectance / GGX Schlick F0 / Emissive base color
		float                emissiveScale = 0.f; // Emissive: radiance multiplier; unused otherwise
		float                roughness     = 0.f;
		float                ior           = 1.5f; // Dielectric IOR when dispersion is disabled
		float                sellB1        = 0.f;  // Sellmeier term 1 coefficient; 0 = achromatic
		float                sellC1        = 0.f;  // Sellmeier term 1 resonance (um^2)
		float                sellB2        = 0.f;  // Sellmeier term 2 coefficient
		float                sellC2        = 0.f;  // Sellmeier term 2 resonance (um^2)
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

	__host__ __device__ inline bool IsDispersive(const Material& mat) {
		return mat.sellB1 != 0.f || mat.sellB2 != 0.f;
	}

	// 2-term Sellmeier equation: n^2(lambda) = 1 + B1*l^2/(l^2-C1) + B2*l^2/(l^2-C2),
	// l in um. lambda is in nm.
	__host__ __device__ inline float EvalIOR(const Material& mat, float lambda) {
		if (!IsDispersive(mat)) return mat.ior;
		float l2 = lambda * lambda * 1e-6f;
		float n2 = 1.f + mat.sellB1 * l2 / (l2 - mat.sellC1) + mat.sellB2 * l2 / (l2 - mat.sellC2);
		return sqrtf(fmaxf(n2, 1e-6f));
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

	inline Material MakeDielectric(float ior, float roughness, uint8_t mediumIdx = 0) {
		Material m{};
		m.type = MaterialType::Dielectric;
		m.ior = ior;
		m.roughness = roughness;
		m.mediumIdx = mediumIdx;
		return m;
	}

	// Sellmeier coefficients (um^2) from a real glass/gem catalog fit, e.g. diamond
	// (Peter 1923): B1=4.3356 C1=0.1060^2 B2=0.3306 C2=0.1750^2.
	inline Material MakeDispersiveDielectric(
		float sellB1, float sellC1, float sellB2, float sellC2,
		float roughness, uint8_t mediumIdx = 0)
	{
		Material m{};
		m.type = MaterialType::Dielectric;
		m.sellB1 = sellB1; m.sellC1 = sellC1;
		m.sellB2 = sellB2; m.sellC2 = sellC2;
		m.roughness = roughness;
		m.mediumIdx = mediumIdx;
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
