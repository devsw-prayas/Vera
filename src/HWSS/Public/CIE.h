#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

// CIE 1931 2-degree standard observer, analytic multi-Gaussian fit
// (Wyman, Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions", JCGT 2013) — avoids shipping/transcribing a
// full tabulated CMF, accurate enough for path tracer wavelength->XYZ.
namespace Vera::Spectral::HWSS {
	__host__ __device__ inline float CIEGaussian(float x, float mu, float sigma1, float sigma2) {
		float sigma = (x < mu) ? sigma1 : sigma2;
		float t = (x - mu) / sigma;
		return expf(-0.5f * t * t);
	}

	__host__ __device__ inline float CIE_X(float lambda) {
		return 1.056f  * CIEGaussian(lambda, 599.8f, 37.9f, 31.0f)
		     + 0.362f  * CIEGaussian(lambda, 442.0f, 16.0f, 26.7f)
		     - 0.065f  * CIEGaussian(lambda, 501.1f, 20.4f, 26.2f);
	}

	__host__ __device__ inline float CIE_Y(float lambda) {
		return 0.821f * CIEGaussian(lambda, 568.8f, 46.9f, 40.5f)
		     + 0.286f * CIEGaussian(lambda, 530.9f, 16.3f, 31.1f);
	}

	__host__ __device__ inline float CIE_Z(float lambda) {
		return 1.217f * CIEGaussian(lambda, 437.0f, 11.8f, 36.0f)
		     + 0.681f * CIEGaussian(lambda, 459.0f, 26.0f, 13.8f);
	}

	// Numeric integral of CIE_Y over [lambdaMin, lambdaMax] — used to normalize
	// wavelength-sampled XYZ estimates back to radiometric units. Host-only,
	// call once at startup and cache the result.
	inline float CIE_Y_Integral(float lambdaMin, float lambdaMax, int steps = 301) {
		float sum = 0.f;
		float step = (lambdaMax - lambdaMin) / (steps - 1);
		for (int i = 0; i < steps; ++i) {
			float lambda = lambdaMin + i * step;
			float w = (i == 0 || i == steps - 1) ? 0.5f : 1.f;
			sum += w * CIE_Y(lambda);
		}
		return sum * step;
	}

	__host__ __device__ inline float3 XYZToLinearSRGB(float3 xyz) {
		return make_float3(
			 3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
			-0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
			 0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
	}
}
