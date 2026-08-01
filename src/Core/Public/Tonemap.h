#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace BSPT::Spectral::HWSS::Core {
	enum class ToneMapper { Reinhard, ACES, AgX, Linear };

	//  Reinhard
	__host__ __device__ static inline float3 tonemapReinhard(float3 c) {
		return { c.x / (1.f + c.x), c.y / (1.f + c.y), c.z / (1.f + c.z) };
	}

	//  ACES — full RRT + sRGB ODT (Stephen Hill, fitted to Academy reference)
	// Input/output matrices: linear sRGB ↔ AP1-like working space used by RRT fit.
	__host__ __device__ static inline float3 acesIn(float3 v) {
		return {
			v.x * 0.59719f + v.y * 0.35458f + v.z * 0.04823f,
			v.x * 0.07600f + v.y * 0.90834f + v.z * 0.01566f,
			v.x * 0.02840f + v.y * 0.13383f + v.z * 0.83777f
		};
	}
	__host__ __device__ static inline float3 acesOut(float3 v) {
		return {
			v.x * 1.60475f + v.y * -0.53108f + v.z * -0.07367f,
			v.x * -0.10208f + v.y * 1.10813f + v.z * -0.00605f,
			v.x * -0.00327f + v.y * -0.07276f + v.z * 1.07602f
		};
	}
	// Rational polynomial fit to the combined RRT + sRGB ODT tone curve.
	__host__ __device__ static inline float3 rrtOdtFit(float3 v) {
		float3 a = {
			v.x * (v.x + 0.0245786f) - 0.000090537f,
			v.y * (v.y + 0.0245786f) - 0.000090537f,
			v.z * (v.z + 0.0245786f) - 0.000090537f
		};
		float3 b = {
			v.x * (0.983729f * v.x + 0.4329510f) + 0.238081f,
			v.y * (0.983729f * v.y + 0.4329510f) + 0.238081f,
			v.z * (0.983729f * v.z + 0.4329510f) + 0.238081f
		};
		return { a.x / b.x, a.y / b.y, a.z / b.z };
	}
	__host__ __device__ static inline float3 tonemapACES(float3 c) {
		c = acesIn(c);
		c = rrtOdtFit(c);
		c = acesOut(c);
		return {
			fmaxf(0.f, fminf(1.f, c.x)),
			fmaxf(0.f, fminf(1.f, c.y)),
			fmaxf(0.f, fminf(1.f, c.z))
		};
	}

	//  AgX — Troy Sobotka's transform (Blender reference matrices)
	__host__ __device__ static inline float agxSigmoid(float x) {
		float x2 = x * x;
		float x4 = x2 * x2;
		float x6 = x4 * x2;
		return -17.86f * x6 * x
			+ 78.01f * x6
			- 126.7f * x4 * x
			+ 92.06f * x4
			- 28.72f * x2 * x
			+ 4.361f * x2
			- 0.1718f * x
			+ 0.002857f;
	}
	__host__ __device__ static inline float3 agxInset(float3 v) {
		return {
			v.x * 0.856627153315983f + v.y * 0.137318972929847f + v.z * 0.11189821299995f,
			v.x * 0.0951212405381588f + v.y * 0.761241990602591f + v.z * 0.0767994186031903f,
			v.x * 0.0482516061458583f + v.y * 0.101439036467562f + v.z * 0.811302368396859f
		};
	}
	__host__ __device__ static inline float3 agxOutset(float3 v) {
		return {
			v.x * 1.1271005818144368f + v.y * -0.1413297634984383f + v.z * -0.14132976349843826f,
			v.x * -0.11060664309660323f + v.y * 1.157823702216272f + v.z * -0.11060664309660294f,
			v.x * -0.016493938717834573f + v.y * -0.016493938717834257f + v.z * 1.2519364065950405f
		};
	}
	__host__ __device__ static inline float3 tonemapAgX(float3 c) {
		c = agxInset(c);
		// Log2 compress to [-12.47393, 4.026069], remap to [0, 1]
		c = { fmaxf(c.x, 2.2e-13f), fmaxf(c.y, 2.2e-13f), fmaxf(c.z, 2.2e-13f) };
		c = { log2f(c.x), log2f(c.y), log2f(c.z) };
		constexpr float minEV = -12.47393f, maxEV = 4.026069f, range = maxEV - minEV;
		c = {
			fmaxf(0.f, fminf(1.f, (c.x - minEV) / range)),
			fmaxf(0.f, fminf(1.f, (c.y - minEV) / range)),
			fmaxf(0.f, fminf(1.f, (c.z - minEV) / range))
		};
		c = { agxSigmoid(c.x), agxSigmoid(c.y), agxSigmoid(c.z) };
		c = agxOutset(c);
		return {
			fmaxf(0.f, fminf(1.f, c.x)),
			fmaxf(0.f, fminf(1.f, c.y)),
			fmaxf(0.f, fminf(1.f, c.z))
		};
	}

	//  Dispatch
	__host__ __device__ static inline float3 applyTonemap(ToneMapper tm, float3 c) {
		switch (tm) {
		case ToneMapper::ACES:     return tonemapACES(c);
		case ToneMapper::AgX:      return tonemapAgX(c);
		case ToneMapper::Reinhard: return tonemapReinhard(c);
		case ToneMapper::Linear:   return { fmaxf(0.f,fminf(1.f,c.x)), fmaxf(0.f,fminf(1.f,c.y)), fmaxf(0.f,fminf(1.f,c.z)) };
		}
		return tonemapAgX(c);
	}
} // namespace BSPT::Spectral::HWSS::Core
