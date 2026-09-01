#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "PCG32.cuh"

// Randomized (digit-scrambled) Halton QMC for the first kQmcDimBudget dimensions of each
// path, falling back to PCG32 beyond that. Halton over Sobol because the prime bases are
// hand-verifiable. Unscrambled Halton clusters badly in high-index (large-base) dimensions
// at the sample counts used here - worse than random - so each digit gets a
// per-(dimension,pixel,digit) affine scramble mod b (Kollig & Keller 2002), a bijection
// that preserves low discrepancy while decorrelating pixels.
namespace Vera::Spectral::HWSS {
	static constexpr int kQmcDimBudget = 32; // dims [0, kQmcDimBudget) use Halton; beyond, PCG32 fallback
	static constexpr int kQmcCameraDims = 8;  // dims [0, 8) reserved for camera ray-gen (uses 5: uv(2)+lens(2)+hero(1))
	static constexpr int kQmcDimsPerBounce = 12; // dims per bounce (worst case: rough dielectric NEE 7, or elastic
	// Lambertian + fluorescent routing/resample draws). Beyond
	// kQmcDimBudget the PCG32 fallback takes over regardless.
	static constexpr int kQmcScrambleDigits = 12; // digits scrambled per sample: 2^12=4096 covers any realistic spp for base 2, and far more for larger bases

	__device__ static constexpr uint32_t kHaltonPrimes[kQmcDimBudget] = {
		2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
		59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
	};

	__device__ inline uint32_t HashU32(uint32_t x) {
		x ^= x >> 16; x *= 0x7feb352dU;
		x ^= x >> 15; x *= 0x846ca68bU;
		x ^= x >> 16;
		return x;
	}

	// Digit-scrambled radical inverse of `index` in base `b`. Always walks a fixed
	// kQmcScrambleDigits positions, including structurally-zero ones - otherwise index=0
	// stays exactly 0 for every pixel regardless of the scramble.
	__device__ inline float ScrambledRadicalInverse(uint32_t index, uint32_t b, uint32_t seed) {
		float invB = 1.f / (float)b;
		float f = invB;
		float r = 0.f;
		for (int i = 0; i < kQmcScrambleDigits; ++i) {
			uint32_t d = index % b;
			index /= b;
			uint32_t h = HashU32(seed + (uint32_t)i * 0x01000193u);
			// Affine scramble d' = (d*m + s) mod b. b prime => any nonzero m is coprime,
			// so d -> d*m mod b is a bijection; the multiply scatters consecutive inputs,
			// which a translation-only shift did not (measured no better than no scramble).
			uint32_t m = (h % (b - 1)) + 1;
			uint32_t s = HashU32(h) % b;
			uint32_t dScrambled = (d * m + s) % b;
			r += f * (float)dScrambled;
			f *= invB;
		}
		return r;
	}

	__device__ inline float QmcSample(uint32_t sampleIdx, uint32_t dim, uint32_t pixelId) {
		uint32_t seed = HashU32(pixelId * 2654435761u + dim * 0x9E3779B1u);
		return ScrambledRadicalInverse(sampleIdx, kHaltonPrimes[dim], seed);
	}

	// Drop-in replacement for Core::PCG32 (same nextFloat/nextFloat2 interface). `dimBase`
	// is set once per shading call (kQmcCameraDims + bounceCount*kQmcDimsPerBounce; 0 for
	// ray-gen); `localDim` counts draws within the call, so the QMC dimension tracks "how
	// many draws so far" rather than which material branch ran.
	struct HybridRNG {
		Core::PCG32 pcg;
		uint32_t    sampleIdx = 0;
		uint32_t    pixelId = 0;
		uint32_t    dimBase = 0;
		int         localDim = 0;

		__device__ __forceinline__ float nextFloat() {
			uint32_t dim = dimBase + (uint32_t)localDim++;
			if (dim < (uint32_t)kQmcDimBudget) {
				return QmcSample(sampleIdx, dim, pixelId);
			}
			return pcg.nextFloat();
		}

		__device__ __forceinline__ float2 nextFloat2() {
			float a = nextFloat();
			float b = nextFloat();
			return make_float2(a, b);
		}
	};
}
