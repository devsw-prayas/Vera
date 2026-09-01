#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

#include "RGB2Spec.h"
#include "FluorescenceHWSS.h"

namespace Vera::Spectral::HWSS {
	// Spectral participating medium. Coefficients are physical (unbounded), stored as a
	// normalized RGB2Spec shape times a positive scale. d_density == nullptr => homogeneous
	// (existing behaviour); when set it's a trilinear density grid multiplying the shape.
	struct MediumHWSS final {
		RGBSigmoidPolynomial sigmaAShape;
		float                sigmaAScale = 0.f; // absorption coefficient magnitude
		RGBSigmoidPolynomial sigmaSShape;
		float                sigmaSScale = 0.f; // scattering coefficient magnitude
		float                g = 0.f;           // Henyey-Greenstein anisotropy [-1,1]

		// rank-1 fluorophore; fluorQY == 0 means inert. Stokes: fluorLamEm > fluorLamEx.
		// fluorSigmaS is the raw peak fluorescent scattering coefficient.
		float fluorLamEx    = 0.f;
		float fluorLamEm    = 0.f;
		float fluorSigma    = 0.f;
		float fluorSigmaS   = 0.f;
		float fluorQY       = 0.f;
		float fluorEmNorm   = 1.f;  // precomputed emission normalisation
		float fluorAbsCdfLo = 0.f;  // precomputed excitation sampling bounds
		float fluorAbsCdfHi = 1.f;
		float fluorAbsNorm  = 1.f;

		// Heterogeneous extension (nullptr => homogeneous):
		float* d_density = nullptr; // dense grid, gridRes.x*gridRes.y*gridRes.z floats, x-fastest
		int3     gridRes = { 0, 0, 0 };
		float3   gridMin = { 0.f, 0.f, 0.f }; // world-space bounds the grid maps onto
		float3   gridMax = { 0.f, 0.f, 0.f };
		float    maxDensity = 1.f;
		// Woodcock majorant: bounds sigmaT(x,lambda) for every position and visible wavelength.
		float    majorantSigmaT = 0.f;
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

	__host__ __device__ inline bool IsHeterogeneous(const MediumHWSS& m) {
		return m.d_density != nullptr;
	}

	// Trilinear density lookup at a world-space point; 0 outside the grid bounds (vacuum there).
	// Only meaningful when IsHeterogeneous(m).
	__device__ inline float EvalDensity(const MediumHWSS& m, float3 p) {
		float3 rel = make_float3(
			(p.x - m.gridMin.x) / fmaxf(m.gridMax.x - m.gridMin.x, 1e-8f),
			(p.y - m.gridMin.y) / fmaxf(m.gridMax.y - m.gridMin.y, 1e-8f),
			(p.z - m.gridMin.z) / fmaxf(m.gridMax.z - m.gridMin.z, 1e-8f));
		if (rel.x < 0.f || rel.x > 1.f || rel.y < 0.f || rel.y > 1.f || rel.z < 0.f || rel.z > 1.f)
			return 0.f;

		float3 g = make_float3(rel.x * m.gridRes.x - 0.5f, rel.y * m.gridRes.y - 0.5f, rel.z * m.gridRes.z - 0.5f);
		int x0 = (int)floorf(g.x), y0 = (int)floorf(g.y), z0 = (int)floorf(g.z);
		float fx = g.x - x0, fy = g.y - y0, fz = g.z - z0;

		auto sample = [&](int x, int y, int z) -> float {
			x = max(0, min(x, m.gridRes.x - 1));
			y = max(0, min(y, m.gridRes.y - 1));
			z = max(0, min(z, m.gridRes.z - 1));
			return m.d_density[(z * m.gridRes.y + y) * m.gridRes.x + x];
			};

		float c00 = sample(x0, y0, z0) * (1 - fx) + sample(x0 + 1, y0, z0) * fx;
		float c10 = sample(x0, y0 + 1, z0) * (1 - fx) + sample(x0 + 1, y0 + 1, z0) * fx;
		float c01 = sample(x0, y0, z0 + 1) * (1 - fx) + sample(x0 + 1, y0, z0 + 1) * fx;
		float c11 = sample(x0, y0 + 1, z0 + 1) * (1 - fx) + sample(x0 + 1, y0 + 1, z0 + 1) * fx;
		float c0 = c00 * (1 - fy) + c10 * fy;
		float c1 = c01 * (1 - fy) + c11 * fy;
		return c0 * (1 - fz) + c1 * fz;
	}

	// Position-dependent coefficients: density(x) * homogeneous coefficient. Homogeneous
	// media have density 1 everywhere, so these reduce to EvalSigmaA/S/T and are safe to
	// call unconditionally.
	__device__ inline float EvalSigmaAAt(const MediumHWSS& m, float3 p, float lambda) {
		float density = IsHeterogeneous(m) ? EvalDensity(m, p) : 1.f;
		return density * EvalSigmaA(m, lambda);
	}

	__device__ inline float EvalSigmaSAt(const MediumHWSS& m, float3 p, float lambda) {
		float density = IsHeterogeneous(m) ? EvalDensity(m, p) : 1.f;
		return density * EvalSigmaS(m, lambda);
	}

	__device__ inline float EvalSigmaTAt(const MediumHWSS& m, float3 p, float lambda) {
		return EvalSigmaAAt(m, p, lambda) + EvalSigmaSAt(m, p, lambda);
	}

	__host__ __device__ inline bool IsFluorescentMedium(const MediumHWSS& m) {
		return m.fluorQY > 0.f;
	}

	// In-shift term: energy gathered at lambda from shorter wavelengths. The rank-1 integral
	// over l_i of fluorSigmaS*QY*a(l_i)*e(lambda) collapses to fluorSigmaS*QY*N_a*e(lambda).
	__host__ __device__ inline float EvalSigmaFluorIn(const MediumHWSS& m, float lambda) {
		if (m.fluorQY <= 0.f) return 0.f;
		float e = FluorEmissionRaw(lambda, m.fluorLamEm, m.fluorSigma) / fmaxf(m.fluorEmNorm, 1e-8f);
		return m.fluorSigmaS * m.fluorQY * m.fluorAbsNorm * e;
	}

	// Out-shift term: energy shed from lambda to longer wavelengths -> fluorSigmaS*QY*a(lambda).
	__host__ __device__ inline float EvalSigmaFluorOut(const MediumHWSS& m, float lambda) {
		if (m.fluorQY <= 0.f) return 0.f;
		return m.fluorSigmaS * m.fluorQY * FluorAbsorptionRaw(lambda, m.fluorLamEx, m.fluorSigma);
	}

	// Fluorescence-aware coefficients (Mojzik 2018 eq 3): sigma_t / sigma_s plus the out- / in-
	// shift term. Drive transmittance and the free-path CDF. Reduce to EvalSigmaT/S when inert.
	__host__ __device__ inline float EvalSigmaHatT(const MediumHWSS& m, float lambda) {
		return EvalSigmaT(m, lambda) + EvalSigmaFluorOut(m, lambda);
	}

	__host__ __device__ inline float EvalSigmaHatS(const MediumHWSS& m, float lambda) {
		return EvalSigmaS(m, lambda) + EvalSigmaFluorIn(m, lambda);
	}

	// density(x) scales the raw and fluorescent terms alike.
	__device__ inline float EvalSigmaHatTAt(const MediumHWSS& m, float3 p, float lambda) {
		float density = IsHeterogeneous(m) ? EvalDensity(m, p) : 1.f;
		return density * EvalSigmaHatT(m, lambda);
	}

	__device__ inline float EvalSigmaHatSAt(const MediumHWSS& m, float3 p, float lambda) {
		float density = IsHeterogeneous(m) ? EvalDensity(m, p) : 1.f;
		return density * EvalSigmaHatS(m, lambda);
	}

	__device__ inline float EvalSigmaFluorInAt(const MediumHWSS& m, float3 p, float lambda) {
		float density = IsHeterogeneous(m) ? EvalDensity(m, p) : 1.f;
		return density * EvalSigmaFluorIn(m, lambda);
	}

	inline MediumHWSS MakeMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g) {
		MediumHWSS m{};
		m.sigmaAShape = RGB2SpecLookup(table, absorbColorRGB);
		m.sigmaAScale = absorbScale;
		m.sigmaSShape = RGB2SpecLookup(table, scatterColorRGB);
		m.sigmaSScale = scatterScale;
		m.g = g;
		m.majorantSigmaT = absorbScale + scatterScale; // maxDensity implicitly 1 (homogeneous)
		return m;
	}

	// hDensity: host grid, gridRes.x*gridRes.y*gridRes.z floats, x-fastest, values >= 0
	// (a multiplier on the base coefficients; 1.0 == as dense as the base medium).
	inline MediumHWSS MakeHeterogeneousMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g,
		const float* hDensity, int3 gridRes, float3 gridMin, float3 gridMax) {
		MediumHWSS m = MakeMedium(table, absorbColorRGB, absorbScale, scatterColorRGB, scatterScale, g);

		size_t voxelCount = (size_t)gridRes.x * gridRes.y * gridRes.z;
		float maxDensity = 0.f;
		for (size_t i = 0; i < voxelCount; ++i) maxDensity = fmaxf(maxDensity, hDensity[i]);

		cudaMalloc(&m.d_density, voxelCount * sizeof(float));
		cudaMemcpy(m.d_density, hDensity, voxelCount * sizeof(float), cudaMemcpyHostToDevice);
		m.gridRes = gridRes;
		m.gridMin = gridMin;
		m.gridMax = gridMax;
		m.maxDensity = fmaxf(maxDensity, 1e-6f);
		m.majorantSigmaT = m.maxDensity * (absorbScale + scatterScale);
		return m;
	}

	inline void FreeHeterogeneousMedium(MediumHWSS& m) {
		if (m.d_density) cudaFree(m.d_density);
		m.d_density = nullptr;
	}

	// Elastic homogeneous medium + rank-1 fluorophore. Keep peak elastic + fluorescent
	// extinction sane and quantumYield in [0,1]. Stokes requires lamEm > lamEx.
	inline MediumHWSS MakeFluorescentMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g,
		float lamEx, float lamEm, float sigma, float fluorSigmaS, float quantumYield) {
		MediumHWSS m = MakeMedium(table, absorbColorRGB, absorbScale, scatterColorRGB, scatterScale, g);
		m.fluorLamEx    = lamEx;
		m.fluorLamEm    = lamEm;
		m.fluorSigma    = sigma;
		m.fluorSigmaS   = fluorSigmaS;
		m.fluorQY       = quantumYield;
		m.fluorEmNorm   = FluorEmissionNorm(lamEm, sigma);
		m.fluorAbsCdfLo = FluorAbsCdfLo(lamEx, sigma);
		m.fluorAbsCdfHi = FluorAbsCdfHi(lamEx, sigma);
		m.fluorAbsNorm  = FluorAbsNorm(lamEx, sigma);

		// Woodcock majorant bounds max(sigma_hat_s, sigma_hat_t) over the band, not just
		// sigma_t - a strongly fluorescent medium can have sigma_hat_s > sigma_t.
		float maxHat = absorbScale + scatterScale;
		const int steps = 64;
		for (int i = 0; i < steps; ++i) {
			float l = LAMBDA_MIN + (LAMBDA_RANGE * i) / (steps - 1);
			maxHat = fmaxf(maxHat, fmaxf(EvalSigmaHatT(m, l), EvalSigmaHatS(m, l)));
		}
		m.majorantSigmaT = maxHat;
		return m;
	}

	// Density-grid variant. hDensity scales the elastic and fluorescent coefficients alike;
	// the majorant carries maxDensity * max(sigma_hat_s, sigma_hat_t) (Mojzik Fig 15 hotfix).
	inline MediumHWSS MakeHeterogeneousFluorescentMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g,
		float lamEx, float lamEm, float sigma, float fluorSigmaS, float quantumYield,
		const float* hDensity, int3 gridRes, float3 gridMin, float3 gridMax) {
		MediumHWSS m = MakeFluorescentMedium(table, absorbColorRGB, absorbScale, scatterColorRGB,
											 scatterScale, g, lamEx, lamEm, sigma, fluorSigmaS, quantumYield);
		float perUnitMajorant = m.majorantSigmaT; // max(sigma_hat_s, sigma_hat_t) over the band at density 1

		size_t voxelCount = (size_t)gridRes.x * gridRes.y * gridRes.z;
		float maxDensity = 0.f;
		for (size_t i = 0; i < voxelCount; ++i) maxDensity = fmaxf(maxDensity, hDensity[i]);

		cudaMalloc(&m.d_density, voxelCount * sizeof(float));
		cudaMemcpy(m.d_density, hDensity, voxelCount * sizeof(float), cudaMemcpyHostToDevice);
		m.gridRes = gridRes;
		m.gridMin = gridMin;
		m.gridMax = gridMax;
		m.maxDensity = fmaxf(maxDensity, 1e-6f);
		m.majorantSigmaT = m.maxDensity * perUnitMajorant;
		return m;
	}
}
