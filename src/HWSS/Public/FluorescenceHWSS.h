#pragma once
#include <cuda_runtime.h>
#include <cmath>

#include "SpectralConstants.h"

// Rank-1 fluorophore model (Mojzik, Fichet & Wilkie 2018): a surface absorbs
// light near one wavelength and re-emits it, diffusely, near a longer one.
//
//     Phi(l_i -> l_o) = QY * a(l_i) * e(l_o)
//
//   a(l) = exp(-1/2 * ((l - lamEx)/sigma)^2)         absorption, peak 1
//   e(l) = exp(-1/2 * ((l - lamEm)/sigma)^2) / N_e   emission density, integrates to 1
//   N_e  = integral of e's numerator over [LAMBDA_MIN, LAMBDA_MAX], host-precomputed
//   QY   = quantum yield in [0, 1]
//
// Matches the R&D DBR oracle's kernel_fluorescence() (R&D/Inverse Spectral
// Rendering/src/kernels.py) exactly, which the Vera tests validate against.
// lamEm > lamEx (Stokes shift). Parameters live on Material; QY == 0 means the
// fluorescent channel is inert and the surface is a plain Lambertian.

namespace Vera::Spectral::HWSS {

	__host__ __device__ inline float FluorAbsorptionRaw(float lambda, float lamEx, float sigma) {
		float t = (lambda - lamEx) / sigma;
		return expf(-0.5f * t * t);
	}

	__host__ __device__ inline float FluorEmissionRaw(float lambda, float lamEm, float sigma) {
		float t = (lambda - lamEm) / sigma;
		return expf(-0.5f * t * t);
	}

	// Normalized emission density, integrates to 1 over the sampled band.
	__host__ __device__ inline float FluorEmissionPdf(float lambda, float lamEm, float sigma, float emNorm) {
		return FluorEmissionRaw(lambda, lamEm, sigma) / fmaxf(emNorm, 1e-8f);
	}

	// Host: N_e by trapezoid over [LAMBDA_MIN, LAMBDA_MAX]. Call once per material
	// (mirrors CIE_Y_Integral in CIE.h).
	inline float FluorEmissionNorm(float lamEm, float sigma, int steps = 601) {
		float step = LAMBDA_RANGE / (steps - 1);
		float sum  = 0.f;
		for (int i = 0; i < steps; ++i) {
			float l = LAMBDA_MIN + i * step;
			float w = (i == 0 || i == steps - 1) ? 0.5f : 1.f;
			sum += w * FluorEmissionRaw(l, lamEm, sigma);
		}
		return sum * step;
	}

	// a(l) is sampled by inverse-CDF as a Gaussian truncated to [LAMBDA_MIN,
	// LAMBDA_MAX]. cdfLo/cdfHi remap a uniform draw into that range; N_a is the
	// truncated integral of a(l). Because the sampling density is a(l)/N_a, a(l)
	// cancels in every estimator that divides Phi by it — no per-sample variance
	// from where in the absorption band the draw landed, and no wasted
	// near-zero-weight samples the way uniform sampling would give.
	inline float StdNormalCdf(float x) { return 0.5f * erfcf(-x * 0.70710678118654752f); }
	inline float FluorAbsCdfLo(float lamEx, float sigma) { return StdNormalCdf((LAMBDA_MIN - lamEx) / sigma); }
	inline float FluorAbsCdfHi(float lamEx, float sigma) { return StdNormalCdf((LAMBDA_MAX - lamEx) / sigma); }
	inline float FluorAbsNorm (float lamEx, float sigma) {
		return sigma * 2.50662827463100050f * (FluorAbsCdfHi(lamEx, sigma) - FluorAbsCdfLo(lamEx, sigma));
	}

	// Device: sample an excitation wavelength from a(l)/N_a given one uniform and
	// the host-precomputed cdfLo/cdfHi.
	__device__ inline float SampleFluorExcitation(float u, float lamEx, float sigma, float cdfLo, float cdfHi) {
		float p = cdfLo + u * (cdfHi - cdfLo);
		return lamEx + sigma * normcdfinvf(p);
	}

	// Probability of routing a lane through the fluorescent channel, proportional
	// to the emission profile at that lane's SENSOR wavelength (where its energy
	// is destined to land, not its current trace wavelength). Near 0 outside the
	// emission band, so out-of-band lanes — and every lane on a non-fluorescent
	// material — always take the elastic channel, unchanged from the plain path.
	__host__ __device__ inline float FluorRoutingProb(
		float lambdaSensor, float lamEm, float sigma, float pMax = 0.9f)
	{
		return FluorEmissionRaw(lambdaSensor, lamEm, sigma) * pMax;
	}
}
