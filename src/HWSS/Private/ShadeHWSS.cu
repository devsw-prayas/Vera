#include "ShadeHWSS.cuh"
#include "MatHelpersHWSS.h"
#include "LightBVHSamplerHWSS.cuh"
#include "CIE.h"
#include "PCG32.cuh"
#include "QMCHWSS.h"
#include <PhaseFunction.cuh>
#include <Traversal.cuh>
#include <Triangle.h>
#include <BVH.h>
#include <CudaMath.h>

namespace Vera::Spectral::HWSS {
	using MH = MatHelpersHWSS;

	// XYZ accumulate, weight 1/(activeLanes*pdf_lane). activeLanes is the count of
	// lanes with pdf > 0, NOT HWSS_LANES: a dispersed ray has only the hero lane left,
	// and dividing by the fixed 4 there is a silent per-sample bias.
	// sensorWavelengths != trace wavelengths after a Stokes shift.
	__device__ inline void AccumulateContribution(
		FrameBufferHWSS& fb, uint32_t pixelId, cudaTextureObject_t cieTex,
		const float4& sensorWavelengths, const float4& throughput, const float4& pdf, const float4& Le) {
		float wl[4] = { sensorWavelengths.x, sensorWavelengths.y, sensorWavelengths.z, sensorWavelengths.w };
		float thpt[4] = { throughput.x,  throughput.y,  throughput.z,  throughput.w };
		float pdfArr[4] = { pdf.x, pdf.y, pdf.z, pdf.w };
		float leArr[4] = { Le.x, Le.y, Le.z, Le.w };

		int activeLanes = 0;
		for (int lane = 0; lane < HWSS_LANES; ++lane)
			if (pdfArr[lane] > 0.f) ++activeLanes;
		if (activeLanes == 0) return;

		float3 xyz = make_float3(0.f, 0.f, 0.f);
		for (int lane = 0; lane < HWSS_LANES; ++lane) {
			if (pdfArr[lane] <= 0.f) continue;
			float contrib = thpt[lane] * leArr[lane] / (activeLanes * pdfArr[lane]);
			float3 cie = SampleCIEXYZ(cieTex, wl[lane]);
			xyz.x += contrib * cie.x;
			xyz.y += contrib * cie.y;
			xyz.z += contrib * cie.z;
		}
		atomicAdd(&fb.d_accumXYZ[pixelId].x, xyz.x);
		atomicAdd(&fb.d_accumXYZ[pixelId].y, xyz.y);
		atomicAdd(&fb.d_accumXYZ[pixelId].z, xyz.z);
	}

	__device__ inline float3 Reflect(float3 wo, float3 n) {
		return normalize(2.f * dot(wo, n) * n - wo);
	}

	__device__ inline float PowerHeuristic(float pdfA, float pdfB) {
		float a2 = pdfA * pdfA, b2 = pdfB * pdfB;
		return (a2 + b2 > 0.f) ? a2 / (a2 + b2) : 0.f;
	}

	// no RR below minBounces; survivors rescaled by 1/q
	__device__ inline bool RussianRoulette(float4& thpt, HybridRNG& rng, uint32_t bounceCount, uint32_t minBounces = 4) {
		if (bounceCount < minBounces) return true;
		float maxComp = fmaxf(fmaxf(thpt.x, thpt.y), fmaxf(thpt.z, thpt.w));
		float q = fminf(fmaxf(maxComp, 0.05f), 0.95f);
		if (rng.nextFloat() > q) return false;
		float invQ = 1.f / q;
		thpt.x *= invQ; thpt.y *= invQ; thpt.z *= invQ; thpt.w *= invQ;
		return true;
	}

	// Woodcock tracking (Kutz 2017 Alg.4). Shared majorant sample across all lanes;
	// returns true on real scatter (scatterT set), false on ballistic pass-through.
	__device__ inline bool HeterogeneousFreeFlight(
		const MediumHWSS& med, float3 origin, float3 dir, float tMax,
		const float wl[4], float4& thpt, HybridRNG& rng, float& scatterT) {
		float majorant = med.majorantSigmaT;
		if (majorant <= 0.f) { scatterT = tMax; return false; }

		float* thptArr[4] = { &thpt.x, &thpt.y, &thpt.z, &thpt.w };
		float t = 0.f;

		for (int iter = 0; iter < 1024; ++iter) { // safety cap
			float u = rng.nextFloat();
			t += -logf(fmaxf(1.f - u, 1e-8f)) / majorant;
			if (t >= tMax) { scatterT = tMax; return false; }

			float3 p = origin + t * dir;
			float sigmaS[4], sigmaT_[4];
			float avgS = 0.f, avgN = 0.f;
			for (int lane = 0; lane < HWSS_LANES; ++lane) {
				sigmaT_[lane] = EvalSigmaTAt(med, p, wl[lane]);
				sigmaS[lane] = EvalSigmaSAt(med, p, wl[lane]);
				float sigmaN = fmaxf(majorant - sigmaT_[lane], 0.f);
				avgS += fabsf(sigmaS[lane] * (*thptArr[lane]));
				avgN += fabsf(sigmaN * (*thptArr[lane]));
			}
			avgS *= 0.25f; avgN *= 0.25f;
			float c = avgS + avgN;
			if (c <= 1e-12f) continue; // numerically vacuum

			float pScatter = avgS / c;
			if (rng.nextFloat() < pScatter) {
				for (int lane = 0; lane < HWSS_LANES; ++lane)
					*thptArr[lane] *= sigmaS[lane] / (majorant * pScatter);
				scatterT = t;
				return true;
			} else {
				float pNull = 1.f - pScatter;
				for (int lane = 0; lane < HWSS_LANES; ++lane) {
					float sigmaN = fmaxf(majorant - sigmaT_[lane], 0.f);
					*thptArr[lane] *= sigmaN / (majorant * pNull);
				}
			}
		}
		scatterT = tMax;
		return false; // safety cap hit -> ballistic pass-through
	}

	// Per-lane shadow transmittance. Homogeneous: analytic. Heterogeneous: ratio tracking (Novak 2014).
	__device__ inline float4 EvalTransmittance(
		const MediumHWSS& med, float3 origin, float3 dir, float segLength, const float4& wavelengths, HybridRNG& rng) {
		if (!IsHeterogeneous(med)) {
			return make_float4(
				expf(-EvalSigmaT(med, wavelengths.x) * segLength), expf(-EvalSigmaT(med, wavelengths.y) * segLength),
				expf(-EvalSigmaT(med, wavelengths.z) * segLength), expf(-EvalSigmaT(med, wavelengths.w) * segLength));
		}

		float majorant = med.majorantSigmaT;
		if (majorant <= 0.f) return make_float4(1.f, 1.f, 1.f, 1.f);

		float wl[4] = { wavelengths.x, wavelengths.y, wavelengths.z, wavelengths.w };
		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		float* trArr[4] = { &tr.x, &tr.y, &tr.z, &tr.w };

		float t = 0.f;
		for (int iter = 0; iter < 1024; ++iter) {
			float u = rng.nextFloat();
			t += -logf(fmaxf(1.f - u, 1e-8f)) / majorant;
			if (t >= segLength) break;

			float3 p = origin + t * dir;
			for (int lane = 0; lane < HWSS_LANES; ++lane) {
				float sigmaTLocal = EvalSigmaTAt(med, p, wl[lane]);
				float sigmaN = fmaxf(majorant - sigmaTLocal, 0.f);
				*trArr[lane] *= sigmaN / majorant;
			}
		}
		return tr;
	}

	// Lambertian area-light NEE. MIS vs BSDF pdf.
	__device__ inline void SampleDirectLighting(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float4& reflectance,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (lightBvh.lightCount == 0) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2 = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist = sqrtf(dist2);
		float3 wi = toLight / dist;

		float cosSurf = fmaxf(dot(wi, normal), 0.f);
		float cosLight = fmaxf(dot(ls.normal, -wi), 0.f);
		if (cosSurf <= 0.f || cosLight <= 1e-6f) return;

		float solidPdf = ls.pdf * dist2 / cosLight;
		if (solidPdf <= 0.f) return;

		if (Vera::Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		float bsdfPdfForWi = MH::lambertianPdf(wi, normal);
		float misWeight = PowerHeuristic(solidPdf, bsdfPdfForWi);

		float4 nee = ray.m_Throughput;
		float scale = misWeight * cosSurf / solidPdf;
		nee.x *= MH::evalLambertian(reflectance.x) * scale * tr.x;
		nee.y *= MH::evalLambertian(reflectance.y) * scale * tr.y;
		nee.z *= MH::evalLambertian(reflectance.z) * scale * tr.z;
		nee.w *= MH::evalLambertian(reflectance.w) * scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	// Fluorescent lobe NEE. Shared excitation sample; per-lane emission weight. MIS vs fluoresced BSDF.
	__device__ inline void SampleDirectLightingFluor(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (lightBvh.lightCount == 0) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2 = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist = sqrtf(dist2);
		float3 wi = toLight / dist;

		float cosSurf = fmaxf(dot(wi, normal), 0.f);
		float cosLight = fmaxf(dot(ls.normal, -wi), 0.f);
		if (cosSurf <= 0.f || cosLight <= 1e-6f) return;

		float solidPdf = ls.pdf * dist2 / cosLight;
		if (solidPdf <= 0.f) return;

		if (Vera::Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		float lamI = SampleFluorExcitation(rng.nextFloat(), mat.fluorLamEx, mat.fluorSigma,
										   mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
		const Material& lmat = materials[ls.materialId];
		float LeI = EvalEmission(lmat, lamI); // emission at the excitation wavelength, not l_o
		if (LeI <= 0.f) return;

		float misWeight = PowerHeuristic(solidPdf, MH::lambertianPdf(wi, normal));
		float commonScale = misWeight * cosSurf / solidPdf * LeI * mat.fluorQY * mat.fluorAbsNorm / PI;

		float4 lo = ray.m_Wavelengths; // l_o == sensor wavelength here (pre-shift)
		float4 nee = ray.m_Throughput;
		nee.x *= FluorEmissionPdf(lo.x, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.x;
		nee.y *= FluorEmissionPdf(lo.y, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.y;
		nee.z *= FluorEmissionPdf(lo.z, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.z;
		nee.w *= FluorEmissionPdf(lo.w, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf,
							   make_float4(1.f, 1.f, 1.f, 1.f));
	}

	// GGX area-light NEE. MIS vs VNDF pdf.
	__device__ inline void SampleDirectLightingGGX(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (lightBvh.lightCount == 0 || wo_l.z <= 0.f) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2 = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist = sqrtf(dist2);
		float3 wi = toLight / dist;

		float cosLight = fmaxf(dot(ls.normal, -wi), 0.f);
		if (cosLight <= 1e-6f) return;

		float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
		if (wi_l.z <= 0.f) return;

		float solidPdf = ls.pdf * dist2 / cosLight;
		if (solidPdf <= 0.f) return;

		float3 h_l = normalize(wo_l + wi_l);
		float  Dh = MH::evalGGX(h_l, alpha);
		float  vis = MH::ggxG2(wi_l, wo_l, alpha);

		float a2 = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
		float bsdfPdf = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);

		if (Vera::Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float cosTheta = fmaxf(wo_l.z, 0.f);
		float scale = misWeight * wi_l.z / solidPdf;

		float f0 = EvalReflectance(mat, ray.m_Wavelengths.x);
		float f1 = EvalReflectance(mat, ray.m_Wavelengths.y);
		float f2 = EvalReflectance(mat, ray.m_Wavelengths.z);
		float f3 = EvalReflectance(mat, ray.m_Wavelengths.w);

		float4 nee = ray.m_Throughput;
		nee.x *= Dh * vis * MH::schlickFresnel(f0, cosTheta) * scale * tr.x;
		nee.y *= Dh * vis * MH::schlickFresnel(f1, cosTheta) * scale * tr.y;
		nee.z *= Dh * vis * MH::schlickFresnel(f2, cosTheta) * scale * tr.z;
		nee.w *= Dh * vis * MH::schlickFresnel(f3, cosTheta) * scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	// Rough dielectric area-light NEE (Walter 2007). Evaluates whichever lobe matches wi, MIS vs VNDF.
	// eta^2 omitted from BTDF to match BSDF-sampling path; Jacobian still needs it.
	__device__ inline void SampleDirectLightingDielectricRough(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, float etaI, float etaT,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (lightBvh.lightCount == 0 || wo_l.z <= 0.f) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2 = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist = sqrtf(dist2);
		float3 wi = toLight / dist;

		// A transmitted wi can approach the light from either face, so |dot| unsigned.
		float solidPdf = ls.pdf * dist2 / fmaxf(fabsf(dot(ls.normal, wi)), 1e-6f);
		if (solidPdf <= 0.f) return;

		float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
		if (fabsf(wi_l.z) < 1e-6f) return;

		float a2 = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));

		float value = 0.f;   // achromatic (no per-wavelength albedo)
		float bsdfPdf = 0.f;
		float3 shadowNormal = normal;

		if (wi_l.z > 0.f) {
			// Reflection lobe
			float3 h_l = normalize(wo_l + wi_l);
			float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
			float  F = MH::fresnelDielectric(cosThetaH, etaI, etaT);
			float  Dh = MH::evalGGX(h_l, alpha);
			float  vis = MH::ggxG2(wi_l, wo_l, alpha);
			value = Dh * vis * F;
			bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
		} else {
			// Transmission lobe: generalized half-vector (Walter et al. eq 16), oriented onto wo's side.
			float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
			if (ht.z < 0.f) ht = -ht;
			float cosThetaH = dot(wo_l, ht);
			if (cosThetaH <= 0.f) return;

			float F = MH::fresnelDielectric(cosThetaH, etaI, etaT);
			float Dh = MH::evalGGX(ht, alpha);
			float vis = MH::ggxG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
			float G2 = vis * 4.f * wo_l.z * fabsf(wi_l.z);

			float denomJ = etaI * dot(wo_l, ht) + etaT * dot(wi_l, ht);
			if (fabsf(denomJ) < 1e-7f) return;

			float wiDotH = dot(wi_l, ht);
			value = (1.f - F) * Dh * G2 * fabsf(wiDotH * cosThetaH)
				/ fmaxf(wo_l.z * fabsf(wi_l.z), 1e-7f) / (denomJ * denomJ);

			float dwh_dwi = etaT * etaT * fabsf(wiDotH) / (denomJ * denomJ);
			bsdfPdf = (1.f - F) * (G1wo * Dh * fmaxf(cosThetaH, 0.f) / wo_l.z) * dwh_dwi;

			shadowNormal = -normal; // wi crosses the surface - offset the shadow origin the other way
		}

		if (value <= 0.f || bsdfPdf <= 0.f) return;
		if (Vera::Core::TraverseAnyHit(geom, P + shadowNormal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		// Uses the current medium for both lobes: exact for reflection, approximate for
		// transmission (the shadow ray actually crosses into the far-side medium).
		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + shadowNormal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float scale = misWeight * fabsf(wi_l.z) * value / solidPdf;

		float4 nee = ray.m_Throughput;
		nee.x *= scale * tr.x; nee.y *= scale * tr.y; nee.z *= scale * tr.z; nee.w *= scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	enum EnvBsdfKind { ENV_BSDF_LAMBERT = 0, ENV_BSDF_GGX = 1, ENV_BSDF_ROUGH_DIELECTRIC = 2 };

	// Env-map NEE. Importance-samples luminance, evaluates BSDF, MIS vs BSDF pdf. Skipped inside media.
	__device__ inline void SampleDirectLightingEnv(
		Core::GeometryBuffers& geom, const EnvMapHWSS& envMap,
		const float3& P, const float3& normal,
		int bsdfKind, const float4& lambReflectance,
		const float3& tangent, const float3& bitangent, const float3& wo_l,
		float alpha, float etaI, float etaT, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (envMap.tex == 0 || ray.m_MediumIdx != 0) return;

		float envPdf;
		float3 wi = SampleEnvMap(envMap, rng.nextFloat2(), envPdf);
		if (envPdf <= 0.f) return;

		float3 shadowNormal = normal;
		float  geomTerm = 0.f;   // |cos|, folded into the estimator
		float  bsdfPdf = 0.f;   // solid-angle pdf of wi under BSDF sampling (for MIS)
		float4 fVal = make_float4(0.f, 0.f, 0.f, 0.f); // per-lane BSDF value

		if (bsdfKind == ENV_BSDF_LAMBERT) {
			float cosSurf = dot(wi, normal);
			if (cosSurf <= 0.f) return;
			geomTerm = cosSurf;
			bsdfPdf = MH::lambertianPdf(wi, normal);
			fVal = make_float4(
				MH::evalLambertian(lambReflectance.x), MH::evalLambertian(lambReflectance.y),
				MH::evalLambertian(lambReflectance.z), MH::evalLambertian(lambReflectance.w));
		} else if (bsdfKind == ENV_BSDF_GGX) {
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (wi_l.z <= 0.f) return;

			float3 h_l = normalize(wo_l + wi_l);
			float  Dh = MH::evalGGX(h_l, alpha);
			float  vis = MH::ggxG2(wi_l, wo_l, alpha);
			float  a2 = alpha * alpha;
			float  G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			bsdfPdf = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			geomTerm = wi_l.z;

			float cosTheta = fmaxf(wo_l.z, 0.f); // macro-normal convention
			float f0 = EvalReflectance(mat, ray.m_Wavelengths.x);
			float f1 = EvalReflectance(mat, ray.m_Wavelengths.y);
			float f2 = EvalReflectance(mat, ray.m_Wavelengths.z);
			float f3 = EvalReflectance(mat, ray.m_Wavelengths.w);
			fVal = make_float4(
				Dh * vis * MH::schlickFresnel(f0, cosTheta), Dh * vis * MH::schlickFresnel(f1, cosTheta),
				Dh * vis * MH::schlickFresnel(f2, cosTheta), Dh * vis * MH::schlickFresnel(f3, cosTheta));
		} else { // rough dielectric - achromatic
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (fabsf(wi_l.z) < 1e-6f) return;

			float a2 = alpha * alpha;
			float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			float value = 0.f;

			if (wi_l.z > 0.f) {
				float3 h_l = normalize(wo_l + wi_l);
				float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
				float  F = MH::fresnelDielectric(cosThetaH, etaI, etaT);
				float  Dh = MH::evalGGX(h_l, alpha);
				float  vis = MH::ggxG2(wi_l, wo_l, alpha);
				value = Dh * vis * F;
				bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			} else {
				float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
				if (ht.z < 0.f) ht = -ht;
				float cosThetaH = dot(wo_l, ht);
				if (cosThetaH <= 0.f) return;

				float F = MH::fresnelDielectric(cosThetaH, etaI, etaT);
				float Dh = MH::evalGGX(ht, alpha);
				float vis = MH::ggxG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
				float G2 = vis * 4.f * wo_l.z * fabsf(wi_l.z);

				float denomJ = etaI * dot(wo_l, ht) + etaT * dot(wi_l, ht);
				if (fabsf(denomJ) < 1e-7f) return;

				float wiDotH = dot(wi_l, ht);
				value = (1.f - F) * Dh * G2 * fabsf(wiDotH * cosThetaH)
					/ fmaxf(wo_l.z * fabsf(wi_l.z), 1e-7f) / (denomJ * denomJ);

				float dwh_dwi = etaT * etaT * fabsf(wiDotH) / (denomJ * denomJ);
				bsdfPdf = (1.f - F) * (G1wo * Dh * fmaxf(cosThetaH, 0.f) / wo_l.z) * dwh_dwi;
				shadowNormal = -normal;
			}
			if (value <= 0.f) return;
			geomTerm = fabsf(wi_l.z);
			fVal = make_float4(value, value, value, value);
		}

		if (bsdfPdf <= 0.f || geomTerm <= 0.f) return;
		if (Vera::Core::TraverseAnyHit(geom, P + shadowNormal * 1e-4f, wi, 1e-4f, 1e30f)) return;

		float4 LeVec = make_float4(
			EvalEnvMapDir(envMap, wi, ray.m_Wavelengths.x), EvalEnvMapDir(envMap, wi, ray.m_Wavelengths.y),
			EvalEnvMapDir(envMap, wi, ray.m_Wavelengths.z), EvalEnvMapDir(envMap, wi, ray.m_Wavelengths.w));

		float misWeight = PowerHeuristic(envPdf, bsdfPdf);
		float scale = misWeight * geomTerm / envPdf;

		float4 nee = ray.m_Throughput;
		nee.x *= fVal.x * scale; nee.y *= fVal.y * scale;
		nee.z *= fVal.z * scale; nee.w *= fVal.w * scale;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	__global__ void ShadeKernelHWSSWavefront(
		Core::GeometryBuffers geom,
		RayCoreSoA coreIn,
		RayExtSoA extIn,
		const Core::WavefrontHitRecord* __restrict__ hits,
		const uint32_t* __restrict__ order,
		const Material* __restrict__ materials,
		const MediumHWSS* __restrict__ media,
		LightBVH lightBvh,
		RayCoreSoA coreOut,
		RayExtSoA extOut,
		uint32_t rayCount,
		FrameBufferHWSS fb,
		EnvMapHWSS envMap,
		uint32_t maxBounces,
		cudaTextureObject_t cieTex) {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		unsigned int srcIdx = order[idx];
		RayHWSS ray = LoadRay(coreIn, extIn, srcIdx);

		Core::WavefrontHitRecord hit = hits[srcIdx];

		HybridRNG rng{};
		rng.pcg.m_State = ray.m_RngState;
		rng.sampleIdx = ray.m_SampleIdx;
		rng.pixelId = ray.pixelId;
		// bounce count BEFORE this call's increments - picks this call's QMC dimension range
		rng.dimBase = (uint32_t)kQmcCameraDims + (uint32_t)ray.m_BounceCount * (uint32_t)kQmcDimsPerBounce;

		if (ray.m_MediumIdx != 0) {
			MediumHWSS med = media[ray.m_MediumIdx - 1];
			float tMax = hit.m_Hit ? hit.t : 1e6f;

			if (IsHeterogeneous(med)) {
				if (med.majorantSigmaT > 0.f) {
					float wl[4] = { ray.m_Wavelengths.x, ray.m_Wavelengths.y, ray.m_Wavelengths.z, ray.m_Wavelengths.w };
					float4 thpt = ray.m_Throughput;
					float  scatterT = tMax;
					bool scattered = HeterogeneousFreeFlight(med, ray.m_Origin, ray.m_Direction, tMax, wl, thpt, rng, scatterT);

					if (scattered) {
						float3 scatterP = ray.m_Origin + scatterT * ray.m_Direction;
						float3 wo = -ray.m_Direction;
						float phasePdf;
						float3 wi = Core::HGPhaseSample(med.g, wo, rng.nextFloat2(), phasePdf);

						ray.m_BounceCount += 1;
						if (!RussianRoulette(thpt, rng, ray.m_BounceCount)) {
							ray.flags |= RAY_FLAG_DEAD;
							ray.m_RngState = rng.pcg.m_State;
							StoreRay(coreOut, extOut, idx, ray);
							return;
						}

						ray.m_Origin = scatterP;
						ray.m_Direction = wi;
						ray.m_Throughput = thpt;
						ray.flags &= ~RAY_FLAG_DELTA;
						ray.m_RngState = rng.pcg.m_State;
						if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					} else {
						// ballistic pass-through: the null-collision weight is the transmittance ratio
						ray.m_Throughput = thpt;
					}
				}
			} else {
				float sigmaTHero = EvalSigmaT(med, ray.m_Wavelengths.x);
				if (sigmaTHero > 0.f) {
					float u = rng.nextFloat();
					float sampledDist = -logf(fmaxf(1.f - u, 1e-8f)) / sigmaTHero;

					if (sampledDist < tMax) {
						// volume scatter: ratio-track lanes, HG phase sample, skip surface shading
						float3 scatterP = ray.m_Origin + sampledDist * ray.m_Direction;
						float pdfHero = sigmaTHero * expf(-sigmaTHero * sampledDist);
						float4 thpt = ray.m_Throughput;
						float wl[4] = { ray.m_Wavelengths.x, ray.m_Wavelengths.y, ray.m_Wavelengths.z, ray.m_Wavelengths.w };
						float* thptArr[4] = { &thpt.x, &thpt.y, &thpt.z, &thpt.w };
						for (int lane = 0; lane < HWSS_LANES; ++lane) {
							float sigmaTLane = EvalSigmaT(med, wl[lane]);
							float sigmaSLane = EvalSigmaS(med, wl[lane]);
							float trLane = expf(-sigmaTLane * sampledDist);
							*thptArr[lane] *= sigmaSLane * trLane / fmaxf(pdfHero, 1e-12f);
						}

						float3 wo = -ray.m_Direction;
						float phasePdf;
						float3 wi = Core::HGPhaseSample(med.g, wo, rng.nextFloat2(), phasePdf);

						ray.m_BounceCount += 1;
						if (!RussianRoulette(thpt, rng, ray.m_BounceCount)) {
							ray.flags |= RAY_FLAG_DEAD;
							ray.m_RngState = rng.pcg.m_State;
							StoreRay(coreOut, extOut, idx, ray);
							return;
						}

						ray.m_Origin = scatterP;
						ray.m_Direction = wi;
						ray.m_Throughput = thpt;
						ray.flags &= ~RAY_FLAG_DELTA;
						ray.m_RngState = rng.pcg.m_State;
						if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					} else {
						// no scatter: attenuate to the surface by the per-lane/hero transmittance ratio
						float pNoScatter = expf(-sigmaTHero * tMax);
						float wl[4] = { ray.m_Wavelengths.x, ray.m_Wavelengths.y, ray.m_Wavelengths.z, ray.m_Wavelengths.w };
						float* thptArr[4] = { &ray.m_Throughput.x, &ray.m_Throughput.y, &ray.m_Throughput.z, &ray.m_Throughput.w };
						for (int lane = 0; lane < HWSS_LANES; ++lane) {
							float trLane = expf(-EvalSigmaT(med, wl[lane]) * tMax);
							*thptArr[lane] *= trLane / fmaxf(pNoScatter, 1e-12f);
						}
					}
				}
			}
		}

		if (!hit.m_Hit) {
			float3 missDir = ray.m_Direction;
			float4 LeVec = make_float4(
				EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.x), EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.y),
				EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.z), EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.w));
			// MIS against env NEE from the previous vertex; delta bounces get full weight
			float misScalar = 1.f;
			if (envMap.tex != 0 && !(ray.flags & RAY_FLAG_DELTA)) {
				float envPdf = EnvMapPdf(envMap, missDir);
				misScalar = PowerHeuristic(ray.m_BsdfPdf, envPdf);
			}
			// fluoresced lanes have no env-NEE counterpart yet -> full weight
			unsigned char fm = ray.m_LaneFluoresced;
			float4 wThpt = make_float4(
				ray.m_Throughput.x * ((fm & 1u) ? 1.f : misScalar),
				ray.m_Throughput.y * ((fm & 2u) ? 1.f : misScalar),
				ray.m_Throughput.z * ((fm & 4u) ? 1.f : misScalar),
				ray.m_Throughput.w * ((fm & 8u) ? 1.f : misScalar));
			AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, wThpt, ray.m_Pdf, LeVec);
			ray.flags |= RAY_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		uint16_t matId = geom.m_TriMatID[hit.m_PrimIdx];
		Material mat = materials[matId];

		Core::Instance inst = geom.m_DevInstances[hit.m_InstIdx];
		// interpolated normals - flat normals cause hard shading edges on near-mirror GGX
		uint32_t vi0 = geom.m_DevIndexBuffer[hit.m_PrimIdx * 3 + 0];
		uint32_t vi1 = geom.m_DevIndexBuffer[hit.m_PrimIdx * 3 + 1];
		uint32_t vi2 = geom.m_DevIndexBuffer[hit.m_PrimIdx * 3 + 2];
		float3 n0 = geom.m_DevVertexNorms[vi0];
		float3 n1 = geom.m_DevVertexNorms[vi1];
		float3 n2 = geom.m_DevVertexNorms[vi2];
		float bary0 = 1.f - hit.m_U - hit.m_V;
		float3 nLocal = normalize(bary0 * n0 + hit.m_U * n1 + hit.m_V * n2);
		float3 normal = normalize(make_float3(
			dot(make_float3(inst.m_Transform[0].x, inst.m_Transform[0].y, inst.m_Transform[0].z), nLocal),
			dot(make_float3(inst.m_Transform[1].x, inst.m_Transform[1].y, inst.m_Transform[1].z), nLocal),
			dot(make_float3(inst.m_Transform[2].x, inst.m_Transform[2].y, inst.m_Transform[2].z), nLocal)));

		float3 P = ray.m_Origin + hit.t * ray.m_Direction;
		float3 wo = -ray.m_Direction;
		// geometric normal for dielectric enter/exit test; viewer-facing flip below
		float3 geoNormal = normal;
		if (dot(wo, normal) < 0.f) normal = -normal;

		if (mat.type == MaterialType::Emissive) {
			// delta bounces have no competing NEE -> full weight; otherwise MIS against the
			// NEE the previous vertex did (EmissiveHitPdf gives the light-strategy pdf for wi)
			float misScalar = 1.f;
			if (!(ray.flags & RAY_FLAG_DELTA)) {
				float lightPdfArea = EmissiveHitPdf(lightBvh, hit.m_PrimIdx, ray.m_Origin);
				float cosLight = fabsf(dot(normal, ray.m_Direction));
				float dist2 = hit.t * hit.t;
				float lightPdfSolidAngle = (lightPdfArea > 0.f && cosLight > 1e-6f) ? lightPdfArea * dist2 / cosLight : 0.f;
				misScalar = PowerHeuristic(ray.m_BsdfPdf, lightPdfSolidAngle);
			}
			// a fluoresced lane now has a competing NEE (SampleDirectLightingFluor), so it
			// takes the same MIS weight as the rest
			if (misScalar > 0.f) {
				float4 LeVec = make_float4(
					EvalEmission(mat, ray.m_Wavelengths.x), EvalEmission(mat, ray.m_Wavelengths.y),
					EvalEmission(mat, ray.m_Wavelengths.z), EvalEmission(mat, ray.m_Wavelengths.w));
				float4 thptWeighted = make_float4(
					ray.m_Throughput.x * misScalar, ray.m_Throughput.y * misScalar,
					ray.m_Throughput.z * misScalar, ray.m_Throughput.w * misScalar);
				AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, thptWeighted, ray.m_Pdf, LeVec);
			}
			ray.flags |= RAY_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		float4 lambertianReflectance = make_float4(0.f, 0.f, 0.f, 0.f);
		if (mat.type == MaterialType::Lambertian) {
			lambertianReflectance = make_float4(
				EvalReflectance(mat, ray.m_Wavelengths.x), EvalReflectance(mat, ray.m_Wavelengths.y),
				EvalReflectance(mat, ray.m_Wavelengths.z), EvalReflectance(mat, ray.m_Wavelengths.w));
			SampleDirectLighting(geom, lightBvh, materials, media, P, normal, lambertianReflectance, ray, rng, fb, cieTex);
			SampleDirectLightingEnv(geom, envMap, P, normal, ENV_BSDF_LAMBERT, lambertianReflectance,
									make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f),
									0.f, 1.f, 1.f, mat, ray, rng, fb, cieTex);
			// fluorescent lobe direct lighting (area lights only - env NEE for it isn't built yet)
			if (IsFluorescent(mat))
				SampleDirectLightingFluor(geom, lightBvh, materials, media, P, normal, mat, ray, rng, fb, cieTex);
		}

		float3 ggxTangent, ggxBitangent, ggxWo_l;
		float  ggxAlpha = 0.f;
		if (mat.type == MaterialType::GGX) {
			MH::buildONB(normal, ggxTangent, ggxBitangent);
			ggxWo_l = make_float3(dot(wo, ggxTangent), dot(wo, ggxBitangent), dot(wo, normal));
			ggxAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);
			SampleDirectLightingGGX(geom, lightBvh, materials, media, P, normal, ggxTangent, ggxBitangent,
									ggxWo_l, ggxAlpha, mat, ray, rng, fb, cieTex);
			SampleDirectLightingEnv(geom, envMap, P, normal, ENV_BSDF_GGX, make_float4(0.f, 0.f, 0.f, 0.f),
									ggxTangent, ggxBitangent, ggxWo_l, ggxAlpha, 1.f, 1.f, mat, ray, rng, fb, cieTex);
		}

		float3 wi;
		float  pdfDirectional = 0.f;
		float  bsdfMisPdf = 0.f; // solid-angle pdf of wi, for MIS at a later emissive hit
		float4 thpt = ray.m_Throughput;
		bool   valid = true;
		bool   didFluoresce = false; // any lane took the wavelength-shift channel

		if (mat.type == MaterialType::Lambertian) {
			MH::sampleLambertian(normal, rng.nextFloat2(), wi, pdfDirectional);
			if (pdfDirectional <= 0.f) valid = false;
			else if (!IsFluorescent(mat)) {
				// cosine sampling cancels cosTheta/pdf -> throughput *= albedo
				thpt.x *= lambertianReflectance.x; thpt.y *= lambertianReflectance.y;
				thpt.z *= lambertianReflectance.z; thpt.w *= lambertianReflectance.w;
				bsdfMisPdf = pdfDirectional;
			} else {
				// Fluorescent Lambertian: diffuse re-emission, so `wi` is shared; per lane
				// route to elastic reflection or a wavelength shift. All draws unconditional
				// so a material-sorted batch stays draw-count-uniform.
				bsdfMisPdf = pdfDirectional;
				unsigned char fluorMask = 0;
				float uRoute = rng.nextFloat();
				float lamInU[4] = { rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat() };
				float sWl[4] = { ray.m_SensorWavelengths.x, ray.m_SensorWavelengths.y,
								  ray.m_SensorWavelengths.z, ray.m_SensorWavelengths.w };
				float tWl[4] = { ray.m_Wavelengths.x, ray.m_Wavelengths.y,
								  ray.m_Wavelengths.z, ray.m_Wavelengths.w };
				float refl[4] = { lambertianReflectance.x, lambertianReflectance.y,
								  lambertianReflectance.z, lambertianReflectance.w };
				float* thptArr[4] = { &thpt.x, &thpt.y, &thpt.z, &thpt.w };
				for (int lane = 0; lane < HWSS_LANES; ++lane) {
					// pFl ~ 0 outside the emission band -> always elastic (`*= refl`), bit-identical to the plain path
					float pFl = FluorRoutingProb(sWl[lane], mat.fluorLamEm, mat.fluorSigma);
					if (uRoute < pFl) {
						didFluoresce = true;
						fluorMask |= (unsigned char)(1u << lane);
						// current wavelength is the fluorescent OUTPUT; sample the INPUT from
						// a(lamIn)/N_a (cancels a(lamIn) exactly - see FluorescenceHWSS.h)
						float lamInNm = SampleFluorExcitation(lamInU[lane], mat.fluorLamEx, mat.fluorSigma,
															  mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
						float phiOverQ = mat.fluorQY * mat.fluorAbsNorm
							* FluorEmissionPdf(tWl[lane], mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm);
						*thptArr[lane] *= phiOverQ / pFl;
						tWl[lane] = lamInNm;
					} else {
						*thptArr[lane] *= refl[lane] / (1.f - pFl);
					}
				}
				ray.m_Wavelengths = make_float4(tWl[0], tWl[1], tWl[2], tWl[3]);
				ray.m_LaneFluoresced = fluorMask;
			}
		} else if (mat.type == MaterialType::GGX) {
			float3 h_l = MH::sampleGGXVNDF(ggxWo_l, ggxAlpha, rng.nextFloat2());
			float3 wi_l = 2.f * dot(ggxWo_l, h_l) * h_l - ggxWo_l;
			if (wi_l.z <= 0.f) { valid = false; } else {
				wi = MH::toWorld(wi_l, normal, ggxTangent, ggxBitangent);
				float a2 = ggxAlpha * ggxAlpha;
				float G1wo = 2.f * ggxWo_l.z / (ggxWo_l.z + sqrtf(a2 + (1.f - a2) * ggxWo_l.z * ggxWo_l.z));
				// VNDF sampling cancels D and most of the pdf, leaving throughput *= F*G2/G1
				// (Heitz 2018). MH::ggxG2 returns Vis = G2/(4*wo.z*wi.z), so multiply the
				// 4*wo.z*wi.z factor back before dividing by G1(wo).
				float vis = MH::ggxG2(wi_l, ggxWo_l, ggxAlpha);
				float ratio = (vis * 4.f * ggxWo_l.z * wi_l.z) / fmaxf(G1wo, 1e-7f);

				float cosTheta = fmaxf(dot(wo, normal), 0.f);
				float f0 = EvalReflectance(mat, ray.m_Wavelengths.x);
				float f1 = EvalReflectance(mat, ray.m_Wavelengths.y);
				float f2 = EvalReflectance(mat, ray.m_Wavelengths.z);
				float f3 = EvalReflectance(mat, ray.m_Wavelengths.w);
				thpt.x *= MH::schlickFresnel(f0, cosTheta) * ratio;
				thpt.y *= MH::schlickFresnel(f1, cosTheta) * ratio;
				thpt.z *= MH::schlickFresnel(f2, cosTheta) * ratio;
				thpt.w *= MH::schlickFresnel(f3, cosTheta) * ratio;
				pdfDirectional = 1.f; // pdf cancelled into the ratio above

				float Dh = MH::evalGGX(h_l, ggxAlpha);
				bsdfMisPdf = G1wo * Dh / fmaxf(4.f * ggxWo_l.z, 1e-7f);
			}
		} else { // Dielectric
			float etaHero = EvalIOR(mat, ray.m_Wavelengths.x);
			bool entering = dot(ray.m_Direction, geoNormal) < 0.f;
			float etaI = entering ? ray.m_IorCurr : etaHero;
			float etaT = entering ? etaHero : 1.f;

			if (mat.roughness <= 0.f) {
				// smooth: mirror or Snell - delta BSDF, no NEE
				float F = MH::fresnelDielectric(dot(wo, normal), etaI, etaT);

				if (rng.nextFloat() < F) {
					wi = Reflect(wo, normal);
					pdfDirectional = F;
				} else {
					// `normal` is already on wo's side from the flip above - what refract()
					// wants. Do NOT re-flip it for the exit case (breaks cosThetaI's sign).
					float3 n = normal;
					float eta = etaI / etaT;
					float3 wt;
					if (!MH::refract(-wo, n, eta, wt)) { valid = false; } else {
						wi = wt;
						pdfDirectional = 1.f - F;
						ray.m_IorCurr = etaT;
						ray.m_MediumIdx = entering ? mat.mediumIdx : 0;
						if (IsDispersive(mat)) {
							// collapse non-hero lanes (Wilkie et al.)
							ray.flags |= RAY_FLAG_DISPERSED;
							ray.m_Pdf.y = ray.m_Pdf.z = ray.m_Pdf.w = 0.f;
							thpt.y = thpt.z = thpt.w = 0.f;
						}
					}
				}
			} else {
				// rough dielectric (Walter 2007), VNDF sampled; G2/G1 throughput ratio same as GGX
				float3 dTangent, dBitangent;
				MH::buildONB(normal, dTangent, dBitangent);
				float3 dWo_l = make_float3(dot(wo, dTangent), dot(wo, dBitangent), dot(wo, normal));
				float  dAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);

				SampleDirectLightingDielectricRough(geom, lightBvh, materials, media, P, normal, dTangent, dBitangent,
													dWo_l, dAlpha, etaI, etaT, ray, rng, fb, cieTex);
				SampleDirectLightingEnv(geom, envMap, P, normal, ENV_BSDF_ROUGH_DIELECTRIC, make_float4(0.f, 0.f, 0.f, 0.f),
										dTangent, dBitangent, dWo_l, dAlpha, etaI, etaT, mat, ray, rng, fb, cieTex);

				float3 h_l = MH::sampleGGXVNDF(dWo_l, dAlpha, rng.nextFloat2());
				float  F = MH::fresnelDielectric(dot(dWo_l, h_l), etaI, etaT);
				float  a2 = dAlpha * dAlpha;
				float  G1wo = 2.f * dWo_l.z / (dWo_l.z + sqrtf(a2 + (1.f - a2) * dWo_l.z * dWo_l.z));
				float  Dh = MH::evalGGX(h_l, dAlpha);

				if (rng.nextFloat() < F) {
					float3 wi_l = 2.f * dot(dWo_l, h_l) * h_l - dWo_l;
					if (wi_l.z <= 0.f) { valid = false; } else {
						wi = MH::toWorld(wi_l, normal, dTangent, dBitangent);
						float vis = MH::ggxG2(wi_l, dWo_l, dAlpha);
						float ratio = (vis * 4.f * dWo_l.z * wi_l.z) / fmaxf(G1wo, 1e-7f);
						thpt.x *= ratio; thpt.y *= ratio; thpt.z *= ratio; thpt.w *= ratio;
						pdfDirectional = 1.f; // sentinel; real pdf is bsdfMisPdf below
						bsdfMisPdf = F * G1wo * Dh / fmaxf(4.f * dWo_l.z, 1e-7f);
					}
				} else {
					float eta = etaI / etaT;
					float3 wt_l;
					if (!MH::refract(-dWo_l, h_l, eta, wt_l)) { valid = false; } else if (wt_l.z >= 0.f) { valid = false; } else {
						wi = MH::toWorld(wt_l, normal, dTangent, dBitangent);
						float vis = MH::ggxG2(make_float3(wt_l.x, wt_l.y, fabsf(wt_l.z)), dWo_l, dAlpha);
						float ratio = (vis * 4.f * dWo_l.z * fabsf(wt_l.z)) / fmaxf(G1wo, 1e-7f);
						thpt.x *= ratio; thpt.y *= ratio; thpt.z *= ratio; thpt.w *= ratio;
						pdfDirectional = 1.f;

						ray.m_IorCurr = etaT;
						ray.m_MediumIdx = entering ? mat.mediumIdx : 0;
						if (IsDispersive(mat)) {
							ray.flags |= RAY_FLAG_DISPERSED;
							ray.m_Pdf.y = ray.m_Pdf.z = ray.m_Pdf.w = 0.f;
							thpt.y = thpt.z = thpt.w = 0.f;
						}

						float denomJ = etaI * dot(dWo_l, h_l) + etaT * dot(wt_l, h_l);
						float dwh_dwi = etaT * etaT * fabsf(dot(wt_l, h_l)) / fmaxf(denomJ * denomJ, 1e-9f);
						bsdfMisPdf = (1.f - F) * (G1wo * Dh * fmaxf(dot(dWo_l, h_l), 0.f) / dWo_l.z) * dwh_dwi;
					}
				}
			}
		}

		if (!valid || pdfDirectional <= 0.f) {
			ray.flags |= RAY_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		ray.m_BounceCount += 1;
		if (!RussianRoulette(thpt, rng, ray.m_BounceCount)) {
			ray.flags |= RAY_FLAG_DEAD;
			ray.m_RngState = rng.pcg.m_State;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		ray.m_Origin = P + wi * 1e-4f;
		ray.m_Direction = wi;
		ray.m_Throughput = thpt;
		ray.m_BsdfPdf = bsdfMisPdf; // only meaningful when RAY_FLAG_DELTA is unset
		ray.flags &= ~RAY_FLAG_DELTA;
		if (mat.type == MaterialType::Dielectric && mat.roughness <= 0.f)
			ray.flags |= RAY_FLAG_DELTA; // true delta BSDF, no NEE
		// fluoresced mask is set in the Lambertian branch; any elastic bounce clears it so MIS resumes normal
		if (!didFluoresce) ray.m_LaneFluoresced = 0;
		ray.m_RngState = rng.pcg.m_State;

		if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;

		StoreRay(coreOut, extOut, idx, ray);
	}
}