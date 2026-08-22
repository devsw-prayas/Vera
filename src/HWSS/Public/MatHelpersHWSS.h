#pragma once
#include <cuda_runtime.h>
#include <CudaMath.h>

#include "SpectralConstants.h"
#include "PhaseFunction.cuh"

// Scalar (per-wavelength) port of FlashPath-FA2's RGBMatHelpers. The sampling
// geometry (ONB, cosine hemisphere, GGX D/G, VNDF sampling, refraction) is
// wavelength-agnostic and carried over unchanged; only Fresnel/reflectance
// become single-wavelength scalars instead of RGB triples.
namespace BSPT::Spectral::HWSS {
	struct MatHelpersHWSS final {
		__device__ __forceinline__ static void buildONB(const float3& normal, float3& tangent, float3& bitangent) {
			Core::hgBuildONB(normal, tangent, bitangent);
		}

		__device__ __forceinline__ static float3 sampleCosineHemisphere(float2 u) {
			float phi = 2 * PI * u.x;
			float r = sqrtf(u.y);
			float x = r * cosf(phi);
			float y = r * sinf(phi);
			float z = sqrtf(1 - u.y);
			return make_float3(x, y, z);
		}

		__device__ __forceinline__ static float3 toWorld(const float3& localDir, const float3& normal, const float3& tangent, const float3& bitangent) {
			return normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * normal);
		}

		__device__ __forceinline__ static float lambertianPdf(const float3& wi, const float3& normal) {
			return fmaxf(dot(wi, normal), 0.f) / PI;
		}

		__device__ __forceinline__ static float evalLambertian(float albedo) {
			return albedo / PI;
		}

		__device__ __forceinline__ static void sampleLambertian(const float3& normal, float2 u, float3& wi, float& pdf) {
			float3 tangent, bitangent;
			buildONB(normal, tangent, bitangent);
			float3 localDir = sampleCosineHemisphere(u);
			wi = toWorld(localDir, normal, tangent, bitangent);
			pdf = lambertianPdf(wi, normal);
		}

		__device__ __forceinline__ static float schlickFresnel(float f0, float cosTheta) {
			float m  = 1.f - cosTheta;
			float m2 = m * m;
			float t  = m2 * m2 * m;
			return f0 + (1.f - f0) * t;
		}

		__device__ __forceinline__ static float evalGGX(float3 h, float alpha) {
			float a2 = alpha * alpha;
			float d = h.z * h.z * (a2 - 1.f) + 1.f;
			return a2 / (PI * d * d);
		}

		__device__ __forceinline__ static float ggxG2(float3 wi, float3 wo, float alpha) {
			float a2 = alpha * alpha;
			float denom = wo.z * sqrtf(a2 + (1.f - a2) * wi.z * wi.z)
				+ wi.z * sqrtf(a2 + (1.f - a2) * wo.z * wo.z);
			return 0.5f / fmaxf(denom, 1e-7f);
		}

		__device__ __forceinline__ static float3 sampleGGXVNDF(float3 ve, float alpha, float2 u) {
			// Heitz 2018 — "Sampling the GGX Distribution of Visible Normals"
			float3 Vh = normalize(make_float3(alpha * ve.x, alpha * ve.y, ve.z));

			float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(make_float3(0.f, 0.f, 1.f), Vh))
				: make_float3(1.f, 0.f, 0.f);
			float3 T2 = cross(Vh, T1);

			float r = sqrtf(u.x);
			float phi = 2.f * PI * u.y;
			float t1 = r * cosf(phi);
			float t2 = r * sinf(phi);

			float s = 0.5f * (1.f + Vh.z);
			t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1 * t1)) + s * t2;

			float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

			return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.f, Nh.z)));
		}

		__device__ __forceinline__ static float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
			if (cosThetaI < 0) {
				float tmp = etaI; etaI = etaT; etaT = tmp;
				cosThetaI = fabsf(cosThetaI);
			}

			float sinTheta2 = 1 - cosThetaI * cosThetaI;
			float d = etaI / etaT;
			sinTheta2 = d * d * sinTheta2;
			if (sinTheta2 >= 1.f) return 1.f; // TIR

			float cosThetaT = sqrtf(1 - sinTheta2);
			float Rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
			float Rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);
			return 0.5f * (Rs * Rs + Rp * Rp);
		}

		__device__ __forceinline__ static bool refract(float3 wi, float3 n, float eta, float3& wt) {
			float cosThetaI  = dot(-wi, n);
			float sin2ThetaT = eta * eta * fmaxf(0.f, 1.f - cosThetaI * cosThetaI);
			if (sin2ThetaT >= 1.f) return false;
			float cosThetaT = sqrtf(1.f - sin2ThetaT);
			wt = normalize(eta * wi + (eta * cosThetaI - cosThetaT) * n);
			return true;
		}

		__device__ __forceinline__ static bool sampleGGXVNDFTransmission(
			float3 wo_l, float alpha, float eta, float2 u, float3& wt_l, float& pdf)
		{
			float3 h_l = sampleGGXVNDF(wo_l, alpha, u);

			float3 wi_l = -wo_l;
			float3 n_l  = h_l;
			if (!refract(wi_l, n_l, eta, wt_l)) return false;
			if (wt_l.z >= 0.f) return false;

			float  a2      = alpha * alpha;
			float  D       = evalGGX(h_l, alpha);
			float  G1wo    = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			float  vndfPdf = D * G1wo / fmaxf(4.f * wo_l.z, 1e-7f);

			float  woDotH  = fmaxf(dot(wo_l, h_l), 1e-7f);
			float  wtDotH  = fabsf(dot(wt_l, h_l));
			float  denom   = eta * woDotH + wtDotH;
			float  jacobian = wtDotH / fmaxf(denom * denom, 1e-7f);

			pdf = vndfPdf * jacobian;
			return true;
		}
	};
}
