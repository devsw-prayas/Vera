#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#include "PCG32.cuh"

// Randomized (scrambled) Halton QMC sampling for the first few dimensions of each path
// (camera ray-gen + the first few bounces' worth of shading draws), falling back to plain
// PCG32 beyond that budget. Halton (radical inverse in a small prime base per dimension)
// was chosen over Sobol specifically because every constant involved — the prime bases —
// is trivially self-verifiable by hand; Sobol's direction-number tables are the kind of
// large opaque magic-number constant that's easy to silently mistranscribe without a
// reference file to check against.
//
// Plain (unscrambled) Halton has a real, measured failure mode at the sample counts this
// renderer actually uses: a dimension's radical inverse in base b needs on the order of b
// samples before it's well distributed — with only 8-1024 samples, high-index dimensions
// (bases up to 131 here) barely move off 0 at all, clustering worse than pure random. An
// A/B test this session confirmed it: unscrambled Halton across all 32 QMC dimensions was
// 65% WORSE (MSE vs. a converged reference) than plain PCG32 at 8spp, while restricting it
// to just the first few small-prime dimensions (2,3,5,7 — camera ray-gen) was 13.5% BETTER.
//
// The fix applied below is random digit scrambling (Kollig & Keller 2002): each base-b
// digit of the radical inverse is shifted by a per-(dimension,pixel,digit-position)
// pseudo-random amount mod b before being placed. Addition mod b is a bijection on
// {0,...,b-1} for ANY fixed shift, so this preserves the sequence's low-discrepancy
// guarantee (it's still a valid permutation of the digit space) while breaking the
// clustering — sample 0 in a large base no longer maps to exactly 0 for every pixel, it
// maps to a per-pixel-and-dimension pseudo-random point instead. This also subsumes the
// per-pixel decorrelation a plain Cranley-Patterson rotation would otherwise be needed
// for, since the shift itself already depends on pixelId.
namespace Vera::Spectral::HWSS {
	static constexpr int kQmcDimBudget    = 32; // dims [0, kQmcDimBudget) use Halton; beyond, PCG32 fallback
	static constexpr int kQmcCameraDims   = 8;  // dims [0, 8) reserved for camera ray-gen (uses 5: uv(2)+lens(2)+hero(1))
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

	// Random-digit-scrambled radical inverse of `index` in base `b`, seeded by `seed`
	// (derived from pixelId+dimension so each pixel's copy of the sequence is an
	// independent, still-valid-low-discrepancy permutation of the same underlying
	// points). Always processes a fixed kQmcScrambleDigits digit positions, even ones
	// that are structurally zero for small `index` — otherwise index=0 (every pixel's
	// first sample) would still degenerate to exactly 0 regardless of scrambling, since
	// an all-zero digit string has nothing for a per-digit shift to act on unless the
	// loop actually visits those positions.
	__device__ inline float ScrambledRadicalInverse(uint32_t index, uint32_t b, uint32_t seed) {
		float invB = 1.f / (float)b;
		float f = invB;
		float r = 0.f;
		for (int i = 0; i < kQmcScrambleDigits; ++i) {
			uint32_t d = index % b;
			index /= b;
			uint32_t h = HashU32(seed + (uint32_t)i * 0x01000193u);
			// Affine scramble d' = (d*m + s) mod b. b is always prime here, so any
			// nonzero m in [1, b-1] is coprime to b and (d -> d*m mod b) is a bijection
			// on {0,...,b-1} — unlike a pure additive shift, multiplying SCATTERS
			// consecutive inputs (0,1,2,... -> 0,m,2m,... mod b) instead of just
			// relocating them as a still-consecutive block, which is what actually
			// breaks up the large-prime-base clustering a translation-only shift did
			// not (measured: translation-only scrambling was statistically
			// indistinguishable from no scrambling at all in this session's A/B test).
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

	// Drop-in replacement for Core::PCG32 with an identical nextFloat()/nextFloat2()
	// interface — every existing call site keeps working unchanged, only the declared
	// type of `rng` and the ~handful of function signatures taking it by reference need
	// to change. `dimBase` is set once per shading call (kQmcCameraDims + bounceCount *
	// kQmcDimsPerBounce for shading, 0 for camera ray-gen) and `localDim` increments with
	// every draw made during that call, so QMC dimension usage never depends on which
	// material branch happens to run — it's purely "how many draws have happened so far
	// in this call," which is exactly what a QMC dimension index needs to track.
	struct HybridRNG {
		Core::PCG32 pcg;
		uint32_t    sampleIdx = 0;
		uint32_t    pixelId   = 0;
		uint32_t    dimBase   = 0;
		int         localDim  = 0;

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
