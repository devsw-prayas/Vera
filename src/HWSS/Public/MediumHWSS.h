#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

#include "RGB2Spec.h"

namespace Vera::Spectral::HWSS {
	// Participating medium, spectral. Coefficients are physical (not bounded to [0,1]), so —
	// unlike Material's reflectance — they're baked as a normalized spectral *shape* (RGB2Spec,
	// in [0,1]) times a separate positive scale, the same pattern used for
	// Material::emissiveScale and EnvMapHWSS::intensity.
	//
	// Heterogeneous media are an additive extension: d_density == nullptr keeps every existing
	// call site's behavior byte-identical (homogeneous, spatially constant), matching this
	// session's established pattern for backward-compatible extension points (e.g.
	// mat.roughness <= 0 gating smooth vs. rough Dielectric). When set, d_density is a dense
	// trilinearly-interpolated grid that multiplies the medium's spectral shape — the standard
	// "homogeneous spectral shape x spatially-varying density" decomposition used for smoke/fog,
	// which avoids needing per-voxel spectral data.
	struct MediumHWSS final {
		RGBSigmoidPolynomial sigmaAShape;
		float                sigmaAScale = 0.f; // absorption coefficient magnitude
		RGBSigmoidPolynomial sigmaSShape;
		float                sigmaSScale = 0.f; // scattering coefficient magnitude
		float                g = 0.f;           // Henyey-Greenstein anisotropy [-1,1]

		// ── Heterogeneous extension (nullptr => homogeneous, existing behavior) ──
		float*   d_density  = nullptr; // dense grid, gridRes.x*gridRes.y*gridRes.z floats, row-major x-fastest
		int3     gridRes    = {0, 0, 0};
		float3   gridMin    = {0.f, 0.f, 0.f}; // world-space bounds the grid maps onto
		float3   gridMax    = {0.f, 0.f, 0.f};
		float    maxDensity = 1.f;  // precomputed max over the whole grid, for the majorant
		// Precomputed Woodcock/spectral-tracking majorant: an upper bound on sigmaT(x,lambda) for
		// EVERY position x and EVERY visible wavelength simultaneously. RGB2SpecLookup's sigmoid
		// polynomials are always <= 1 (they represent [0,1] reflectance-like shapes), so
		// sigmaAScale+sigmaSScale alone already bounds the spectral factor; multiplying by
		// maxDensity bounds the spatial factor too.
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

		float c00 = sample(x0, y0, z0)     * (1 - fx) + sample(x0 + 1, y0, z0)     * fx;
		float c10 = sample(x0, y0 + 1, z0) * (1 - fx) + sample(x0 + 1, y0 + 1, z0) * fx;
		float c01 = sample(x0, y0, z0 + 1)     * (1 - fx) + sample(x0 + 1, y0, z0 + 1)     * fx;
		float c11 = sample(x0, y0 + 1, z0 + 1) * (1 - fx) + sample(x0 + 1, y0 + 1, z0 + 1) * fx;
		float c0  = c00 * (1 - fy) + c10 * fy;
		float c1  = c01 * (1 - fy) + c11 * fy;
		return c0 * (1 - fz) + c1 * fz;
	}

	// Local (position-dependent) sigma_a/sigma_s/sigma_t: density(x) * homogeneous coefficient.
	// For a homogeneous medium (IsHeterogeneous == false) density is implicitly 1 everywhere, so
	// these reduce exactly to EvalSigmaA/S/T -- callers can use these unconditionally.
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
		m.majorantSigmaT = absorbScale + scatterScale; // maxDensity implicitly 1 (homogeneous)
		return m;
	}

	// hDensity: host-side dense grid, gridRes.x*gridRes.y*gridRes.z floats, row-major x-fastest,
	// values >= 0 (a multiplier on the homogeneous spectral coefficients -- 1.0 means "as dense
	// as the base medium", not a physical density unit).
	inline MediumHWSS MakeHeterogeneousMedium(
		const RGB2SpecTable& table,
		float3 absorbColorRGB, float absorbScale,
		float3 scatterColorRGB, float scatterScale,
		float g,
		const float* hDensity, int3 gridRes, float3 gridMin, float3 gridMax)
	{
		MediumHWSS m = MakeMedium(table, absorbColorRGB, absorbScale, scatterColorRGB, scatterScale, g);

		size_t voxelCount = (size_t)gridRes.x * gridRes.y * gridRes.z;
		float maxDensity = 0.f;
		for (size_t i = 0; i < voxelCount; ++i) maxDensity = fmaxf(maxDensity, hDensity[i]);

		cudaMalloc(&m.d_density, voxelCount * sizeof(float));
		cudaMemcpy(m.d_density, hDensity, voxelCount * sizeof(float), cudaMemcpyHostToDevice);
		m.gridRes    = gridRes;
		m.gridMin    = gridMin;
		m.gridMax    = gridMax;
		m.maxDensity = fmaxf(maxDensity, 1e-6f);
		m.majorantSigmaT = m.maxDensity * (absorbScale + scatterScale);
		return m;
	}

	inline void FreeHeterogeneousMedium(MediumHWSS& m) {
		if (m.d_density) cudaFree(m.d_density);
		m.d_density = nullptr;
	}
}
