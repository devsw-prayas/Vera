#pragma once
#include <cuda_runtime.h>
#include <CudaMath.h>

#include "CoreUtils.h"

// Henyey-Greenstein phase function. Convention: wo and wi both point away from scatter point.

namespace Vera::Core {
	static constexpr float HG_INV_4PI = 0.07957747154594766788f;  // 1/(4pi)

	// p(cosTheta) = (1 - g^2) / (4pi (1 + g^2 - 2g*cosTheta)^(3/2))
	__device__ __forceinline__ float HGPhaseEval(float g, float cosTheta) {
		float denom = 1.f + g * g - 2.f * g * cosTheta;
		return (1.f - g * g) / (4.f * PI * denom * sqrtf(denom));
	}

	// Frisvad ONB around axis n
	__device__ __forceinline__ void hgBuildONB(const float3& n, float3& T, float3& B) {
		float3 a = (fabsf(n.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f) : make_float3(1.f, 0.f, 0.f);
		T = normalize(cross(a, n));
		B = cross(n, T);
	}

	// Exact HG sample; pdf == eval so they cancel in throughput.
	__device__ __forceinline__ float3 HGPhaseSample(float g, const float3& wo, float2 u, float& pdf) {
		float cosTheta;
		if (fabsf(g) < 1e-3f) {
			cosTheta = 1.f - 2.f * u.x;
		} else {
			float sq = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
			cosTheta = (1.f + g * g - sq * sq) / (2.f * g);
		}
		cosTheta = fmaxf(-1.f, fminf(1.f, cosTheta));

		float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
		float phi = 2.f * PI * u.y;

		float3 T, B;
		hgBuildONB(wo, T, B);
		float3 wi = sinTheta * cosf(phi) * T + sinTheta * sinf(phi) * B + cosTheta * wo;
		pdf = HGPhaseEval(g, cosTheta);
		return normalize(wi);
	}
}