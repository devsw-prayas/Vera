#include "ShadeSHW.cuh"
#include "ShadingUtilsSHW.cuh"
#include "FluorescenceSHW.h"
#include "PhiloxSampler.cuh"
#include "LightBVHSamplerHWSS.cuh"
#include "CIE.h"
#include <PhaseFunction.cuh>
#include <Traversal.cuh>
#include <Triangle.h>
#include <BVH.h>
#include <CudaMath.h>

namespace Vera::SHW {
	using namespace Vera::Spectral::HWSS;

	__device__ inline void AccumulateScalar(
		FrameBufferHWSS& fb, uint32_t pixelId, cudaTextureObject_t cieTex,
		float sensorLambda, float throughput, float pdf, float Le) {
		if (pdf <= 0.f || Le == 0.f) return;
		float contrib = throughput * Le / pdf;
		float3 cie = SampleCIEXYZ(cieTex, sensorLambda);
		atomicAdd(&fb.d_accumXYZ[pixelId].x, contrib * cie.x);
		atomicAdd(&fb.d_accumXYZ[pixelId].y, contrib * cie.y);
		atomicAdd(&fb.d_accumXYZ[pixelId].z, contrib * cie.z);
	}

	__device__ inline float3 Reflect(float3 wo, float3 n) {
		return normalize(2.f * dot(wo, n) * n - wo);
	}

	__device__ inline float EvalTransmittanceScalar(
		const MediumHWSS& med, float3 origin, float3 dir, float segLength, float lambda, PhiloxSampler& rng) {
		if (!IsHeterogeneous(med))
			return expf(-EvalSigmaHatT(med, lambda) * segLength);

		float majorant = med.majorantSigmaT;
		if (majorant <= 0.f) return 1.f;

		float tr = 1.f;
		float t = 0.f;
		for (int iter = 0; iter < 1024; ++iter) {
			t += -logf(fmaxf(1.f - rng.nextFloat(), 1e-8f)) / majorant;
			if (t >= segLength) break;
			float3 p = origin + t * dir;
			float sigmaTLocal = EvalSigmaHatTAt(med, p, lambda);
			float sigmaN = fmaxf(majorant - sigmaTLocal, 0.f);
			tr *= sigmaN / majorant;
		}
		return tr;
	}

	__device__ inline void SampleDirectLightingLambert(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, float reflectance,
		RaySHW& ray, PhiloxSampler& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
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

		if (Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float tr = 1.f;
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittanceScalar(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Lambda, rng);

		float Le = EvalEmission(materials[ls.materialId], ray.m_Lambda);
		float bsdfPdfForWi = LambertianPdf(wi, normal);
		float misWeight = PowerHeuristic(solidPdf, bsdfPdfForWi);
		float nee = ray.m_Throughput * EvalLambertian(reflectance) * misWeight * cosSurf / solidPdf * tr;
		AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda, nee, ray.m_Pdf, Le);
	}

	__device__ inline void SampleDirectLightingFluorSHW(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const Material& mat,
		RaySHW& ray, PhiloxSampler& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
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

		if (Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float tr = 1.f;
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittanceScalar(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Lambda, rng);

		float lamI = SampleFluorExcitation(rng.nextFloat(), mat.fluorLamEx, mat.fluorSigma,
										   mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
		float LeI = EvalEmission(materials[ls.materialId], lamI);
		if (LeI <= 0.f) return;

		float misWeight = PowerHeuristic(solidPdf, LambertianPdf(wi, normal));
		float emPdf = FluorEmissionPdf(ray.m_Lambda, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm);
		float nee = ray.m_Throughput * emPdf * misWeight * cosSurf / solidPdf * LeI
				  * mat.fluorQY * mat.fluorAbsNorm / Core::PI * tr;
		AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda, nee, ray.m_Pdf, 1.f);
	}

	__device__ inline void SampleDirectLightingGGXSHW(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, const Material& mat,
		RaySHW& ray, PhiloxSampler& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
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
		float  Dh = EvalGGX(h_l, alpha);
		float  vis = GGXG2(wi_l, wo_l, alpha);

		float a2 = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
		float bsdfPdf = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);

		if (Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float tr = 1.f;
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittanceScalar(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Lambda, rng);

		float Le = EvalEmission(materials[ls.materialId], ray.m_Lambda);
		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float f0 = EvalReflectance(mat, ray.m_Lambda);
		float nee = ray.m_Throughput * Dh * vis * SchlickFresnel(f0, fmaxf(wo_l.z, 0.f))
				  * misWeight * wi_l.z / solidPdf * tr;
		AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda, nee, ray.m_Pdf, Le);
	}

	__device__ inline void SampleDirectLightingDielectricRoughSHW(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, float etaI, float etaT,
		RaySHW& ray, PhiloxSampler& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (lightBvh.lightCount == 0 || wo_l.z <= 0.f) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2 = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist = sqrtf(dist2);
		float3 wi = toLight / dist;

		float solidPdf = ls.pdf * dist2 / fmaxf(fabsf(dot(ls.normal, wi)), 1e-6f);
		if (solidPdf <= 0.f) return;

		float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
		if (fabsf(wi_l.z) < 1e-6f) return;

		float a2 = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));

		float value = 0.f;
		float bsdfPdf = 0.f;
		float3 shadowNormal = normal;

		if (wi_l.z > 0.f) {
			float3 h_l = normalize(wo_l + wi_l);
			float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
			float  F = FresnelDielectric(cosThetaH, etaI, etaT);
			float  Dh = EvalGGX(h_l, alpha);
			float  vis = GGXG2(wi_l, wo_l, alpha);
			value = Dh * vis * F;
			bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
		} else {
			float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
			if (ht.z < 0.f) ht = -ht;
			float cosThetaH = dot(wo_l, ht);
			if (cosThetaH <= 0.f) return;

			float F = FresnelDielectric(cosThetaH, etaI, etaT);
			float Dh = EvalGGX(ht, alpha);
			float vis = GGXG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
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

		if (value <= 0.f || bsdfPdf <= 0.f) return;
		if (Core::TraverseAnyHit(geom, P + shadowNormal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		float tr = 1.f;
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittanceScalar(media[ray.m_MediumIdx - 1], P + shadowNormal * 1e-4f, wi, dist - 2e-3f, ray.m_Lambda, rng);

		float Le = EvalEmission(materials[ls.materialId], ray.m_Lambda);
		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float nee = ray.m_Throughput * misWeight * fabsf(wi_l.z) * value / solidPdf * tr;
		AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda, nee, ray.m_Pdf, Le);
	}

	enum EnvBsdfKind { ENV_BSDF_LAMBERT = 0, ENV_BSDF_GGX = 1, ENV_BSDF_ROUGH_DIELECTRIC = 2 };

	__device__ inline void SampleDirectLightingEnvSHW(
		Core::GeometryBuffers& geom, const EnvMapHWSS& envMap,
		const float3& P, const float3& normal,
		int bsdfKind, float lambReflectance,
		const float3& tangent, const float3& bitangent, const float3& wo_l,
		float alpha, float etaI, float etaT, const Material& mat,
		RaySHW& ray, PhiloxSampler& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex) {
		if (envMap.tex == 0 || ray.m_MediumIdx != 0) return;

		float envPdf;
		float3 wi = SampleEnvMap(envMap, rng.nextFloat2(), envPdf);
		if (envPdf <= 0.f) return;

		float3 shadowNormal = normal;
		float  geomTerm = 0.f;
		float  bsdfPdf = 0.f;
		float  fVal = 0.f;

		if (bsdfKind == ENV_BSDF_LAMBERT) {
			float cosSurf = dot(wi, normal);
			if (cosSurf <= 0.f) return;
			geomTerm = cosSurf;
			bsdfPdf = LambertianPdf(wi, normal);
			fVal = EvalLambertian(lambReflectance);
		} else if (bsdfKind == ENV_BSDF_GGX) {
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (wi_l.z <= 0.f) return;

			float3 h_l = normalize(wo_l + wi_l);
			float  Dh = EvalGGX(h_l, alpha);
			float  vis = GGXG2(wi_l, wo_l, alpha);
			float  a2 = alpha * alpha;
			float  G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			bsdfPdf = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			geomTerm = wi_l.z;

			float f0 = EvalReflectance(mat, ray.m_Lambda);
			fVal = Dh * vis * SchlickFresnel(f0, fmaxf(wo_l.z, 0.f));
		} else {
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (fabsf(wi_l.z) < 1e-6f) return;

			float a2 = alpha * alpha;
			float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));

			if (wi_l.z > 0.f) {
				float3 h_l = normalize(wo_l + wi_l);
				float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
				float  F = FresnelDielectric(cosThetaH, etaI, etaT);
				float  Dh = EvalGGX(h_l, alpha);
				float  vis = GGXG2(wi_l, wo_l, alpha);
				fVal = Dh * vis * F;
				bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			} else {
				float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
				if (ht.z < 0.f) ht = -ht;
				float cosThetaH = dot(wo_l, ht);
				if (cosThetaH <= 0.f) return;

				float F = FresnelDielectric(cosThetaH, etaI, etaT);
				float Dh = EvalGGX(ht, alpha);
				float vis = GGXG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
				float G2 = vis * 4.f * wo_l.z * fabsf(wi_l.z);

				float denomJ = etaI * dot(wo_l, ht) + etaT * dot(wi_l, ht);
				if (fabsf(denomJ) < 1e-7f) return;

				float wiDotH = dot(wi_l, ht);
				fVal = (1.f - F) * Dh * G2 * fabsf(wiDotH * cosThetaH)
					/ fmaxf(wo_l.z * fabsf(wi_l.z), 1e-7f) / (denomJ * denomJ);

				float dwh_dwi = etaT * etaT * fabsf(wiDotH) / (denomJ * denomJ);
				bsdfPdf = (1.f - F) * (G1wo * Dh * fmaxf(cosThetaH, 0.f) / wo_l.z) * dwh_dwi;
				shadowNormal = -normal;
			}
			if (fVal <= 0.f) return;
			geomTerm = fabsf(wi_l.z);
		}

		if (bsdfPdf <= 0.f || geomTerm <= 0.f) return;
		if (Core::TraverseAnyHit(geom, P + shadowNormal * 1e-4f, wi, 1e-4f, 1e30f)) return;

		float Le = EvalEnvMapDir(envMap, wi, ray.m_Lambda);
		float misWeight = PowerHeuristic(envPdf, bsdfPdf);
		float nee = ray.m_Throughput * fVal * misWeight * geomTerm / envPdf;
		AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda, nee, ray.m_Pdf, Le);
	}

	__device__ inline bool HeterogeneousFreeFlightScalar(
		const MediumHWSS& med, float3 origin, float3 dir, float tMax,
		float lambda, float& throughput, PhiloxSampler& rng, float& scatterT) {
		float majorant = med.majorantSigmaT;
		if (majorant <= 0.f) { scatterT = tMax; return false; }

		float t = 0.f;
		for (int iter = 0; iter < 1024; ++iter) {
			t += -logf(fmaxf(1.f - rng.nextFloat(), 1e-8f)) / majorant;
			if (t >= tMax) { scatterT = tMax; return false; }

			float3 p = origin + t * dir;
			float sigmaT = EvalSigmaHatTAt(med, p, lambda);
			float sigmaS = EvalSigmaHatSAt(med, p, lambda);
			float sigmaN = fmaxf(majorant - sigmaT, 0.f);

			float pScatter = sigmaS / (sigmaS + sigmaN + 1e-12f);
			if (rng.nextFloat() < pScatter) {
				throughput *= sigmaS / (majorant * pScatter);
				scatterT = t;
				return true;
			} else {
				throughput *= sigmaN / (majorant * (1.f - pScatter));
			}
		}
		scatterT = tMax;
		return false;
	}

	__global__ void ShadeKernelSHWWavefront(
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
		RaySHW ray = LoadRay(coreIn, extIn, srcIdx);

		Core::WavefrontHitRecord hit = hits[srcIdx];

		Core::PhiloxKey pkey{};
		pkey.m_PixelId = ray.m_PixelId;
		pkey.m_FrameIdx = 0;
		pkey.m_SampleIdx = ray.m_SampleIdx;
		pkey.m_NeighborIdx = 0;
		pkey.m_StreamSelect = (uint32_t)ray.m_BounceCount + 1;
		PhiloxSampler rng{pkey, 0};

		if (ray.m_MediumIdx != 0) {
			MediumHWSS med = media[ray.m_MediumIdx - 1];
			float tMax = hit.m_Hit ? hit.t : 1e6f;

			if (IsHeterogeneous(med)) {
				if (med.majorantSigmaT > 0.f) {
					float thpt = ray.m_Throughput;
					float scatterT = tMax;
					bool scattered = HeterogeneousFreeFlightScalar(med, ray.m_Origin, ray.m_Direction, tMax, ray.m_Lambda, thpt, rng, scatterT);

					if (scattered) {
						float3 scatterP = ray.m_Origin + scatterT * ray.m_Direction;

						bool fluorMed = IsFluorescentMedium(med);
						if (fluorMed) {
							float sigSi = EvalSigmaFluorInAt(med, scatterP, ray.m_Lambda);
							float ssHat = EvalSigmaHatSAt(med, scatterP, ray.m_Lambda);
							if (rng.nextFloat() < sigSi / fmaxf(ssHat, 1e-12f)) {
								ray.m_Lambda = SampleFluorExcitation(rng.nextFloat(), med.fluorLamEx, med.fluorSigma,
																	 med.fluorAbsCdfLo, med.fluorAbsCdfHi);
								ray.m_Flags |= SHW_FLAG_FLUORESCED;
							}
						}

						float3 wo = -ray.m_Direction;
						float phasePdf;
						float3 wi = Core::HGPhaseSample(med.g, wo, rng.nextFloat2(), phasePdf);
						if (fluorMed) ray.m_BsdfPdf = phasePdf;

						ray.m_BounceCount += 1;
						if (!RussianRouletteSHW(thpt, rng.nextFloat(), ray.m_BounceCount)) {
							ray.m_Flags |= SHW_FLAG_DEAD;
							StoreRay(coreOut, extOut, idx, ray);
							return;
						}

						ray.m_Origin = scatterP;
						ray.m_Direction = wi;
						ray.m_Throughput = thpt;
						ray.m_Flags &= ~SHW_FLAG_DELTA;
						if (ray.m_BounceCount >= maxBounces) ray.m_Flags |= SHW_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					} else {
						ray.m_Throughput = thpt;
					}
				}
			} else if (hit.m_Hit && IsFluorescentMedium(med)) {
				float d = tMax;
				float ss = EvalSigmaHatS(med, ray.m_Lambda);
				float st = EvalSigmaHatT(med, ray.m_Lambda);

				bool escaped;
				float nWeight;
				float sampledDist = SampleFluorFreePath(ss, st, d, rng.nextFloat(), escaped, nWeight);

				if (!escaped) {
					float3 scatterP = ray.m_Origin + sampledDist * ray.m_Direction;
					float fVal = ss * expf(-st * sampledDist);
					float pDensity = fVal / fmaxf(nWeight, 1e-12f);
					float thpt = ray.m_Throughput * fVal / fmaxf(pDensity, 1e-30f);

					float sigSi = EvalSigmaFluorIn(med, ray.m_Lambda);
					if (rng.nextFloat() < sigSi / fmaxf(ss, 1e-12f)) {
						ray.m_Lambda = SampleFluorExcitation(rng.nextFloat(), med.fluorLamEx, med.fluorSigma,
															 med.fluorAbsCdfLo, med.fluorAbsCdfHi);
						ray.m_Flags |= SHW_FLAG_FLUORESCED;
					}

					float3 wo = -ray.m_Direction;
					float phasePdf;
					float3 wi = Core::HGPhaseSample(med.g, wo, rng.nextFloat2(), phasePdf);
					ray.m_BsdfPdf = phasePdf;

					ray.m_BounceCount += 1;
					if (!RussianRouletteSHW(thpt, rng.nextFloat(), ray.m_BounceCount)) {
						ray.m_Flags |= SHW_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					}

					ray.m_Origin = scatterP;
					ray.m_Direction = wi;
					ray.m_Throughput = thpt;
					ray.m_Flags &= ~SHW_FLAG_DELTA;
					if (ray.m_BounceCount >= maxBounces) ray.m_Flags |= SHW_FLAG_DEAD;
					StoreRay(coreOut, extOut, idx, ray);
					return;
				} else {
					float escapeMass = expf(-st * d);
					float pEscape = escapeMass / fmaxf(nWeight, 1e-12f);
					ray.m_Throughput *= escapeMass / fmaxf(pEscape, 1e-30f);
				}
			} else {
				float sigmaT = EvalSigmaT(med, ray.m_Lambda);
				if (sigmaT > 0.f) {
					float sampledDist = -logf(fmaxf(1.f - rng.nextFloat(), 1e-8f)) / sigmaT;

					if (sampledDist < tMax) {
						float3 scatterP = ray.m_Origin + sampledDist * ray.m_Direction;
						float sigmaS = EvalSigmaS(med, ray.m_Lambda);
						float thpt = ray.m_Throughput * sigmaS / fmaxf(sigmaT, 1e-12f);

						float3 wo = -ray.m_Direction;
						float phasePdf;
						float3 wi = Core::HGPhaseSample(med.g, wo, rng.nextFloat2(), phasePdf);

						ray.m_BounceCount += 1;
						if (!RussianRouletteSHW(thpt, rng.nextFloat(), ray.m_BounceCount)) {
							ray.m_Flags |= SHW_FLAG_DEAD;
							StoreRay(coreOut, extOut, idx, ray);
							return;
						}

						ray.m_Origin = scatterP;
						ray.m_Direction = wi;
						ray.m_Throughput = thpt;
						ray.m_Flags &= ~SHW_FLAG_DELTA;
						if (ray.m_BounceCount >= maxBounces) ray.m_Flags |= SHW_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					} else {
						ray.m_Throughput *= expf(-sigmaT * tMax) / fmaxf(expf(-sigmaT * tMax), 1e-12f);
					}
				}
			}
		}

		if (!hit.m_Hit) {
			float3 missDir = ray.m_Direction;
			float Le = EvalEnvMapDir(envMap, missDir, ray.m_Lambda);
			float misScalar = 1.f;
			if (envMap.tex != 0 && !(ray.m_Flags & SHW_FLAG_DELTA)) {
				float envPdf = EnvMapPdf(envMap, missDir);
				misScalar = PowerHeuristic(ray.m_BsdfPdf, envPdf);
			}
			if (ray.m_Flags & SHW_FLAG_FLUORESCED) misScalar = 1.f;
			AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda,
							 ray.m_Throughput * misScalar, ray.m_Pdf, Le);
			ray.m_Flags |= SHW_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		uint16_t matId = geom.m_TriMatID[hit.m_PrimIdx];
		Material mat = materials[matId];

		Core::Instance inst = geom.m_DevInstances[hit.m_InstIdx];
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
		float3 geoNormal = normal;
		if (dot(wo, normal) < 0.f) normal = -normal;

		if (mat.type == MaterialType::Emissive) {
			float misScalar = 1.f;
			if (!(ray.m_Flags & SHW_FLAG_DELTA)) {
				float lightPdfArea = EmissiveHitPdf(lightBvh, hit.m_PrimIdx, ray.m_Origin);
				float cosLight = fabsf(dot(normal, ray.m_Direction));
				float dist2 = hit.t * hit.t;
				float lightPdfSolidAngle = (lightPdfArea > 0.f && cosLight > 1e-6f) ? lightPdfArea * dist2 / cosLight : 0.f;
				misScalar = PowerHeuristic(ray.m_BsdfPdf, lightPdfSolidAngle);
			}
			if (misScalar > 0.f) {
				float Le = EvalEmission(mat, ray.m_Lambda);
				AccumulateScalar(fb, ray.m_PixelId, cieTex, ray.m_SensorLambda,
								 ray.m_Throughput * misScalar, ray.m_Pdf, Le);
			}
			ray.m_Flags |= SHW_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		float lambReflectance = 0.f;
		if (mat.type == MaterialType::Lambertian) {
			lambReflectance = EvalReflectance(mat, ray.m_Lambda);
			SampleDirectLightingLambert(geom, lightBvh, materials, media, P, normal, lambReflectance, ray, rng, fb, cieTex);
			SampleDirectLightingEnvSHW(geom, envMap, P, normal, ENV_BSDF_LAMBERT, lambReflectance,
									   make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f),
									   0.f, 1.f, 1.f, mat, ray, rng, fb, cieTex);
			if (IsFluorescent(mat))
				SampleDirectLightingFluorSHW(geom, lightBvh, materials, media, P, normal, mat, ray, rng, fb, cieTex);
		}

		float3 ggxTangent, ggxBitangent, ggxWo_l;
		float  ggxAlpha = 0.f;
		if (mat.type == MaterialType::GGX) {
			BuildONB(normal, ggxTangent, ggxBitangent);
			ggxWo_l = make_float3(dot(wo, ggxTangent), dot(wo, ggxBitangent), dot(wo, normal));
			ggxAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);
			SampleDirectLightingGGXSHW(geom, lightBvh, materials, media, P, normal, ggxTangent, ggxBitangent,
									   ggxWo_l, ggxAlpha, mat, ray, rng, fb, cieTex);
			SampleDirectLightingEnvSHW(geom, envMap, P, normal, ENV_BSDF_GGX, 0.f,
									   ggxTangent, ggxBitangent, ggxWo_l, ggxAlpha, 1.f, 1.f, mat, ray, rng, fb, cieTex);
		}

		float3 wi;
		float  pdfDirectional = 0.f;
		float  bsdfMisPdf = 0.f;
		float  thpt = ray.m_Throughput;
		bool   valid = true;
		bool   didFluoresce = false;

		if (mat.type == MaterialType::Lambertian) {
			SampleLambertian(normal, rng.nextFloat2(), wi, pdfDirectional);
			if (pdfDirectional <= 0.f) valid = false;
			else if (!IsFluorescent(mat)) {
				thpt *= lambReflectance;
				bsdfMisPdf = pdfDirectional;
			} else {
				bsdfMisPdf = pdfDirectional;
				float pFl = FluorRoutingProb(ray.m_SensorLambda, mat.fluorLamEm, mat.fluorSigma);
				if (rng.nextFloat() < pFl) {
					didFluoresce = true;
					float lamInNm = SampleFluorExcitation(rng.nextFloat(), mat.fluorLamEx, mat.fluorSigma,
														  mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
					float phiOverQ = mat.fluorQY * mat.fluorAbsNorm
						* FluorEmissionPdf(ray.m_Lambda, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm);
					thpt *= phiOverQ / pFl;
					ray.m_Lambda = lamInNm;
				} else {
					thpt *= lambReflectance / (1.f - pFl);
				}
			}
		} else if (mat.type == MaterialType::GGX) {
			float3 h_l = SampleGGXVNDF(ggxWo_l, ggxAlpha, rng.nextFloat2());
			float3 wi_l = 2.f * dot(ggxWo_l, h_l) * h_l - ggxWo_l;
			if (wi_l.z <= 0.f) { valid = false; } else {
				wi = ToWorld(wi_l, normal, ggxTangent, ggxBitangent);
				float a2 = ggxAlpha * ggxAlpha;
				float G1wo = 2.f * ggxWo_l.z / (ggxWo_l.z + sqrtf(a2 + (1.f - a2) * ggxWo_l.z * ggxWo_l.z));
				float vis = GGXG2(wi_l, ggxWo_l, ggxAlpha);
				float ratio = (vis * 4.f * ggxWo_l.z * wi_l.z) / fmaxf(G1wo, 1e-7f);

				float cosTheta = fmaxf(dot(wo, normal), 0.f);
				float f0 = EvalReflectance(mat, ray.m_Lambda);
				thpt *= SchlickFresnel(f0, cosTheta) * ratio;
				pdfDirectional = 1.f;

				float Dh = EvalGGX(h_l, ggxAlpha);
				bsdfMisPdf = G1wo * Dh / fmaxf(4.f * ggxWo_l.z, 1e-7f);
			}
		} else {
			float etaHero = EvalIOR(mat, ray.m_Lambda);
			bool entering = dot(ray.m_Direction, geoNormal) < 0.f;
			float etaI = entering ? ray.m_IorCurr : etaHero;
			float etaT = entering ? etaHero : 1.f;

			if (mat.roughness <= 0.f) {
				float F = FresnelDielectric(dot(wo, normal), etaI, etaT);
				if (rng.nextFloat() < F) {
					wi = Reflect(wo, normal);
					pdfDirectional = F;
				} else {
					float3 n = normal;
					float eta = etaI / etaT;
					float3 wt;
					if (!Refract(-wo, n, eta, wt)) { valid = false; } else {
						wi = wt;
						pdfDirectional = 1.f - F;
						ray.m_IorCurr = etaT;
						ray.m_MediumIdx = entering ? mat.mediumIdx : 0;
					}
				}
			} else {
				float3 dTangent, dBitangent;
				BuildONB(normal, dTangent, dBitangent);
				float3 dWo_l = make_float3(dot(wo, dTangent), dot(wo, dBitangent), dot(wo, normal));
				float  dAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);

				SampleDirectLightingDielectricRoughSHW(geom, lightBvh, materials, media, P, normal, dTangent, dBitangent,
													   dWo_l, dAlpha, etaI, etaT, ray, rng, fb, cieTex);
				SampleDirectLightingEnvSHW(geom, envMap, P, normal, ENV_BSDF_ROUGH_DIELECTRIC, 0.f,
										   dTangent, dBitangent, dWo_l, dAlpha, etaI, etaT, mat, ray, rng, fb, cieTex);

				float3 h_l = SampleGGXVNDF(dWo_l, dAlpha, rng.nextFloat2());
				float  F = FresnelDielectric(dot(dWo_l, h_l), etaI, etaT);
				float  a2 = dAlpha * dAlpha;
				float  G1wo = 2.f * dWo_l.z / (dWo_l.z + sqrtf(a2 + (1.f - a2) * dWo_l.z * dWo_l.z));
				float  Dh = EvalGGX(h_l, dAlpha);

				if (rng.nextFloat() < F) {
					float3 wi_l = 2.f * dot(dWo_l, h_l) * h_l - dWo_l;
					if (wi_l.z <= 0.f) { valid = false; } else {
						wi = ToWorld(wi_l, normal, dTangent, dBitangent);
						float vis = GGXG2(wi_l, dWo_l, dAlpha);
						float ratio = (vis * 4.f * dWo_l.z * wi_l.z) / fmaxf(G1wo, 1e-7f);
						thpt *= ratio;
						pdfDirectional = 1.f;
						bsdfMisPdf = F * G1wo * Dh / fmaxf(4.f * dWo_l.z, 1e-7f);
					}
				} else {
					float eta = etaI / etaT;
					float3 wt_l;
					if (!Refract(-dWo_l, h_l, eta, wt_l)) { valid = false; } else if (wt_l.z >= 0.f) { valid = false; } else {
						wi = ToWorld(wt_l, normal, dTangent, dBitangent);
						float vis = GGXG2(make_float3(wt_l.x, wt_l.y, fabsf(wt_l.z)), dWo_l, dAlpha);
						float ratio = (vis * 4.f * dWo_l.z * fabsf(wt_l.z)) / fmaxf(G1wo, 1e-7f);
						thpt *= ratio;
						pdfDirectional = 1.f;

						ray.m_IorCurr = etaT;
						ray.m_MediumIdx = entering ? mat.mediumIdx : 0;

						float denomJ = etaI * dot(dWo_l, h_l) + etaT * dot(wt_l, h_l);
						float dwh_dwi = etaT * etaT * fabsf(dot(wt_l, h_l)) / fmaxf(denomJ * denomJ, 1e-9f);
						bsdfMisPdf = (1.f - F) * (G1wo * Dh * fmaxf(dot(dWo_l, h_l), 0.f) / dWo_l.z) * dwh_dwi;
					}
				}
			}
		}

		if (!valid || pdfDirectional <= 0.f) {
			ray.m_Flags |= SHW_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		ray.m_BounceCount += 1;
		if (!RussianRouletteSHW(thpt, rng.nextFloat(), ray.m_BounceCount)) {
			ray.m_Flags |= SHW_FLAG_DEAD;
			StoreRay(coreOut, extOut, idx, ray);
			return;
		}

		ray.m_Origin = P + wi * 1e-4f;
		ray.m_Direction = wi;
		ray.m_Throughput = thpt;
		ray.m_BsdfPdf = bsdfMisPdf;
		ray.m_Flags &= ~SHW_FLAG_DELTA;
		if (mat.type == MaterialType::Dielectric && mat.roughness <= 0.f)
			ray.m_Flags |= SHW_FLAG_DELTA;
		if (!didFluoresce) ray.m_Flags &= ~SHW_FLAG_FLUORESCED;
		else ray.m_Flags |= SHW_FLAG_FLUORESCED;

		if (ray.m_BounceCount >= maxBounces) ray.m_Flags |= SHW_FLAG_DEAD;

		StoreRay(coreOut, extOut, idx, ray);
	}
}
