#pragma once
#include <cuda_runtime.h>
#include <CudaMath.h>

#include "CoreUtils.h"

// Henyey-Greenstein phase function utilities (device-only).
// Convention: wo points AWAY from the scatter event (toward viewer),
//             wi points AWAY from the scatter event (toward next surface).
// cosTheta = dot(wo, wi) = cos of the phase angle.

namespace BSPT::Spectral::HWSS::Core {
	static constexpr float HG_INV_4PI = 0.07957747154594766788f;  // 1 / (4π)

	// p(cosTheta) = (1 - g²) / (4π (1 + g² - 2g·cosTheta)^(3/2))
	__device__ __forceinline__ float HGPhaseEval(float g, float cosTheta) {
		float denom = 1.f + g * g - 2.f * g * cosTheta;
		return (1.f - g * g) / (4.f * PI * denom * sqrtf(denom));
	}

	// Build tangent frame around axis n (Frisvad variant).
	__device__ __forceinline__ void hgBuildONB(const float3& n, float3& T, float3& B) {
		float3 a = (fabsf(n.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f);
		T = normalize(cross(a, n));
		B = cross(n, T);
	}

	// Sample wi from HG phase function given wo (pointing away from scatter point).
	// Returns wi and sets pdf = HGPhaseEval(g, cosTheta_sampled).
	// Since HG sampling is exact, pdf == eval and they cancel in the throughput update.
	__device__ __forceinline__ float3 HGPhaseSample(float g, const float3& wo, float2 u, float& pdf) {
		float cosTheta;
		if (fabsf(g) < 1e-3f) {
			// Isotropic: uniform sphere
			cosTheta = 1.f - 2.f * u.x;
		} else {
			float sq = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
			cosTheta  = (1.f + g * g - sq * sq) / (2.f * g);
		}
		cosTheta = fmaxf(-1.f, fminf(1.f, cosTheta));

		float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
		float phi      = 2.f * PI * u.y;

		// Build frame around wo so that the sampled wi is relative to it
		float3 T, B;
		hgBuildONB(wo, T, B);
		float3 wi = sinTheta * cosf(phi) * T + sinTheta * sinf(phi) * B + cosTheta * wo;
		pdf = HGPhaseEval(g, cosTheta);
		return normalize(wi);
	}
}
