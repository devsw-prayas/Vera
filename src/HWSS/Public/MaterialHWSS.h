#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "RGB2Spec.h"
#include "FluorescenceHWSS.h"

namespace Vera::Spectral::HWSS {
	enum class MaterialType : uint8_t {
		Lambertian,
		GGX,
		Emissive,
		Dielectric,
	};

	// Spectral material. Albedo/emission stored as sigmoid-polynomial coefficients (Jakob & Hanika 2019).
	// GGX: Schlick F0 only (RGB2Spec can't hold complex eta/kappa).
	// Dielectric: zero Sellmeier coefficients => constant IOR, nonzero => 2-term Sellmeier.
	struct Material final {
		RGBSigmoidPolynomial albedoSpec;   // Lambertian reflectance / GGX F0 / Emissive color
		float                emissiveScale = 0.f;
		float                roughness = 0.f;
		float                ior = 1.5f;
		float                sellB1 = 0.f;  // Sellmeier B1; 0 = achromatic
		float                sellC1 = 0.f;  // Sellmeier C1 (um^2)
		float                sellB2 = 0.f;
		float                sellC2 = 0.f;
		// rank-1 fluorophore; fluorQY == 0 means inert. Stokes: fluorLamEm > fluorLamEx.
		float                fluorLamEx = 0.f;
		float                fluorLamEm = 0.f;
		float                fluorSigma = 0.f;
		float                fluorQY = 0.f;
		float                fluorEmNorm = 1.f;  // precomputed emission normalisation
		float                fluorAbsCdfLo = 0.f;  // precomputed excitation sampling bounds
		float                fluorAbsCdfHi = 1.f;
		float                fluorAbsNorm = 1.f;
		MaterialType         type = MaterialType::Lambertian;
		uint8_t              mediumIdx = 0;    // interior medium index (0 = vacuum)
		uint8_t              pad[2] = {};
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

	__host__ __device__ inline bool IsFluorescent(const Material& mat) {
		return mat.fluorQY > 0.f;
	}

	// 2-term Sellmeier equation: n^2(lambda) = 1 + B1*l^2/(l^2-C1) + B2*l^2/(l^2-C2),
	// l in um. lambda is in nm.
	__host__ __device__ inline float EvalIOR(const Material& mat, float lambda) {
		if (!IsDispersive(mat)) return mat.ior;
		float l2 = lambda * lambda * 1e-6f;
		float n2 = 1.f + mat.sellB1 * l2 / (l2 - mat.sellC1) + mat.sellB2 * l2 / (l2 - mat.sellC2);
		return sqrtf(fmaxf(n2, 1e-6f));
	}

	// host-side factories: bake RGB -> spectral coefficients

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
		float roughness, uint8_t mediumIdx = 0) {
		Material m{};
		m.type = MaterialType::Dielectric;
		m.sellB1 = sellB1; m.sellC1 = sellC1;
		m.sellB2 = sellB2; m.sellC2 = sellC2;
		m.roughness = roughness;
		m.mediumIdx = mediumIdx;
		return m;
	}

	// Elastic Lambertian + rank-1 fluorophore. Keep peak_reflectance + QY < 1 for energy conservation.
	inline Material MakeFluorescentLambertian(
		const RGB2SpecTable& table, float3 albedoRGB,
		float lamEx, float lamEm, float sigma, float quantumYield) {
		Material m = MakeLambertian(table, albedoRGB);
		m.fluorLamEx = lamEx;
		m.fluorLamEm = lamEm;
		m.fluorSigma = sigma;
		m.fluorQY = quantumYield;
		m.fluorEmNorm = FluorEmissionNorm(lamEm, sigma);
		m.fluorAbsCdfLo = FluorAbsCdfLo(lamEx, sigma);
		m.fluorAbsCdfHi = FluorAbsCdfHi(lamEx, sigma);
		m.fluorAbsNorm = FluorAbsNorm(lamEx, sigma);
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
