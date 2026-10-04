#pragma once
#include <cuda_runtime.h>
#include <CudaMath.h>
#include <CoreUtils.h>
#include <PhaseFunction.cuh>

// Scalar shading utilities for SHW (Scalar Hero Wavelength): BSDF sampling/eval,
// Fresnel, GGX, IOR dispersion. Same math as HWSS's MatHelpersHWSS, but operating
// on one wavelength at a time instead of a float4 hero-wavelength bundle.
namespace Vera::SHW {
	__device__ __forceinline__ void BuildONB(float3 normal, float3& tangent, float3& bitangent) {
		Core::hgBuildONB(normal, tangent, bitangent);
	}

	__device__ __forceinline__ float3 SampleCosineHemisphere(float2 u) {
		float phi = 2 * Core::PI * u.x;
		float r = sqrtf(u.y);
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1 - u.y);
		return make_float3(x, y, z);
	}

	__device__ __forceinline__ float3 ToWorld(float3 v_local, float3 n, float3 t, float3 b) {
		return normalize(v_local.x * t + v_local.y * b + v_local.z * n);
	}

	__device__ __forceinline__ float LambertianPdf(float3 wi, float3 normal) {
		return fmaxf(dot(wi, normal), 0.f) / Core::PI;
	}

	__device__ __forceinline__ float EvalLambertian(float reflectance) {
		return reflectance / Core::PI;
	}

	__device__ __forceinline__ void SampleLambertian(float3 normal, float2 u, float3& wi, float& pdf) {
		float3 tangent, bitangent;
		BuildONB(normal, tangent, bitangent);
		float3 localDir = SampleCosineHemisphere(u);
		wi = ToWorld(localDir, normal, tangent, bitangent);
		pdf = LambertianPdf(wi, normal);
	}

	__device__ __forceinline__ float SchlickFresnel(float f0, float cosTheta) {
		float m = 1.f - cosTheta;
		float m2 = m * m;
		float t = m2 * m2 * m;
		return f0 + (1.f - f0) * t;
	}

	__device__ __forceinline__ float EvalGGX(float3 h_l, float alpha) {
		float a2 = alpha * alpha;
		float d = h_l.z * h_l.z * (a2 - 1.f) + 1.f;
		return a2 / (Core::PI * d * d);
	}

	__device__ __forceinline__ float GGXG2(float3 wi_l, float3 wo_l, float alpha) {
		float a2 = alpha * alpha;
		float denom = wo_l.z * sqrtf(a2 + (1.f - a2) * wi_l.z * wi_l.z)
			+ wi_l.z * sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z);
		return 0.5f / fmaxf(denom, 1e-7f);
	}

	__device__ __forceinline__ float3 SampleGGXVNDF(float3 wo_l, float alpha, float2 u) {
		// Heitz 2018
		float3 Vh = normalize(make_float3(alpha * wo_l.x, alpha * wo_l.y, wo_l.z));

		float3 T1 = (Vh.z < 0.9999f) ? normalize(cross(make_float3(0.f, 0.f, 1.f), Vh))
			: make_float3(1.f, 0.f, 0.f);
		float3 T2 = cross(Vh, T1);

		float r = sqrtf(u.x);
		float phi = 2.f * Core::PI * u.y;
		float t1 = r * cosf(phi);
		float t2 = r * sinf(phi);

		float s = 0.5f * (1.f + Vh.z);
		t2 = (1.f - s) * sqrtf(fmaxf(0.f, 1.f - t1 * t1)) + s * t2;

		float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1 * t1 - t2 * t2)) * Vh;

		return normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.f, Nh.z)));
	}

	__device__ __forceinline__ float FresnelDielectric(float cosTheta, float etaI, float etaT) {
		if (cosTheta < 0) {
			float tmp = etaI; etaI = etaT; etaT = tmp;
			cosTheta = fabsf(cosTheta);
		}

		float sinTheta2 = 1 - cosTheta * cosTheta;
		float d = etaI / etaT;
		sinTheta2 = d * d * sinTheta2;
		if (sinTheta2 >= 1.f) return 1.f; // TIR

		float cosThetaT = sqrtf(1 - sinTheta2);
		float Rs = (etaI * cosTheta - etaT * cosThetaT) / (etaI * cosTheta + etaT * cosThetaT);
		float Rp = (etaT * cosTheta - etaI * cosThetaT) / (etaT * cosTheta + etaI * cosThetaT);
		return 0.5f * (Rs * Rs + Rp * Rp);
	}

	__device__ __forceinline__ bool Refract(float3 wi, float3 n, float eta, float3& wt) {
		float cosThetaI = dot(-wi, n);
		float sin2ThetaT = eta * eta * fmaxf(0.f, 1.f - cosThetaI * cosThetaI);
		if (sin2ThetaT >= 1.f) return false;
		float cosThetaT = sqrtf(1.f - sin2ThetaT);
		wt = normalize(eta * wi + (eta * cosThetaI - cosThetaT) * n);
		return true;
	}

	// Cauchy dispersion: n = A + B / lambda_um^2. lambda is in nm.
	__device__ __forceinline__ float CauchyIOR(float A, float B, float lambda) {
		float lambda_um = lambda * 1e-3f;
		return A + B / (lambda_um * lambda_um);
	}

	// 2-term Sellmeier equation: n^2(lambda) = 1 + B1*l^2/(l^2-C1) + B2*l^2/(l^2-C2),
	// l in um. lambda is in nm.
	__device__ __forceinline__ float SellmeierIOR(float B1, float C1, float B2, float C2, float lambda) {
		float l2 = lambda * lambda * 1e-6f;
		float n2 = 1.f + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2);
		return sqrtf(fmaxf(n2, 1e-6f));
	}

	__device__ __forceinline__ float PowerHeuristic(float pdfA, float pdfB) {
		float a2 = pdfA * pdfA;
		float b2 = pdfB * pdfB;
		return a2 / fmaxf(a2 + b2, 1e-12f);
	}

	// Scalar Russian roulette: survives with prob q, throughput compensated by 1/q.
	__device__ __forceinline__ bool RussianRouletteSHW(
		float& throughput, float u, uint32_t bounce, uint32_t minBounces = 4) {
		if (bounce < minBounces) return true;
		float q = fminf(fmaxf(throughput, 0.05f), 0.95f);
		if (u > q) return false;
		throughput /= q;
		return true;
	}
}
