#include "ShadeHWSS.cuh"
#include "MatHelpersHWSS.h"
#include "LightBVHSamplerHWSS.cuh"
#include "CIE.h"
#include "PCG32.cuh"
#include <PhaseFunction.cuh>
#include <Traversal.cuh>
#include <Triangle.h>
#include <BVH.h>
#include <CudaMath.h>

namespace Vera::Spectral::HWSS {
	using MH = MatHelpersHWSS;

	// Adds a per-lane spectral contribution to the framebuffer: weight = 1/(activeLanes*pdf_lane).
	// activeLanes (lanes with pdf > 0) is NOT always HWSS_LANES: once a path hits a
	// dispersive event, RAY_FLAG_DISPERSED zeroes 3 of the 4 lanes' pdf, leaving only
	// the hero lane valid. Dividing by the fixed lane count instead of the number of
	// lanes actually still contributing would silently under-weight every dispersed
	// sample by that same factor — a real, sample-count-independent bias, not noise.
	__device__ inline void AccumulateContribution(
		FrameBufferHWSS& fb, uint32_t pixelId, cudaTextureObject_t cieTex,
		const float4& wavelengths, const float4& throughput, const float4& pdf, const float4& Le)
	{
		float wl[4]     = { wavelengths.x, wavelengths.y, wavelengths.z, wavelengths.w };
		float thpt[4]   = { throughput.x,  throughput.y,  throughput.z,  throughput.w };
		float pdfArr[4] = { pdf.x, pdf.y, pdf.z, pdf.w };
		float leArr[4]  = { Le.x, Le.y, Le.z, Le.w };

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



	// NEE against LightBVH area lights, Lambertian surfaces only. `reflectance` is the
	// surface's per-lane albedo, precomputed once by the caller and shared with the
	// BSDF-sampling step below so it isn't evaluated twice per lane per shading call.
	__device__ inline void SampleDirectLighting(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials,
		const float3& P, const float3& normal, const float4& reflectance,
		RayHWSS& ray, Core::PCG32& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
	{
		if (lightBvh.lightCount == 0) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2   = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist    = sqrtf(dist2);
		float3 wi      = toLight / dist;

		float cosSurf  = fmaxf(dot(wi, normal), 0.f);
		float cosLight = fmaxf(dot(ls.normal, -wi), 0.f);
		if (cosSurf <= 0.f || cosLight <= 1e-6f) return;

		float solidPdf = ls.pdf * dist2 / cosLight;
		if (solidPdf <= 0.f) return;

		if (Vera::Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		// MIS against BSDF sampling: what pdf would sampling the BSDF have given this
		// same direction wi? (Lambertian is the only NEE-capable material here.)
		float bsdfPdfForWi = MH::lambertianPdf(wi, normal);
		float misWeight = PowerHeuristic(solidPdf, bsdfPdfForWi);

		float4 nee = ray.m_Throughput;
		float scale = misWeight * cosSurf / solidPdf;
		nee.x *= MH::evalLambertian(reflectance.x) * scale;
		nee.y *= MH::evalLambertian(reflectance.y) * scale;
		nee.z *= MH::evalLambertian(reflectance.z) * scale;
		nee.w *= MH::evalLambertian(reflectance.w) * scale;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_Wavelengths, nee, ray.m_Pdf, LeVec);
	}

	// NEE against LightBVH area lights, GGX surfaces. Mirrors SampleDirectLighting but
	// evaluates the microfacet BRDF (D * Vis * F) for the sampled light direction instead
	// of assuming Lambertian, and MIS-weights against the VNDF-sampling pdf for that same
	// direction. Fresnel is evaluated at the macro cosTheta = dot(wo,normal), matching the
	// same approximation the BSDF-sampling path below bakes into its throughput — using the
	// textbook-correct dot(wo,h) here instead would make the two MIS strategies estimate
	// slightly different functions and bias the combined estimate.
	__device__ inline void SampleDirectLightingGGX(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, const Material& mat,
		RayHWSS& ray, Core::PCG32& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
	{
		if (lightBvh.lightCount == 0 || wo_l.z <= 0.f) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2   = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist    = sqrtf(dist2);
		float3 wi      = toLight / dist;

		float cosLight = fmaxf(dot(ls.normal, -wi), 0.f);
		if (cosLight <= 1e-6f) return;

		float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
		if (wi_l.z <= 0.f) return;

		float solidPdf = ls.pdf * dist2 / cosLight;
		if (solidPdf <= 0.f) return;

		float3 h_l = normalize(wo_l + wi_l);
		float  Dh  = MH::evalGGX(h_l, alpha);
		float  vis = MH::ggxG2(wi_l, wo_l, alpha);

		float a2   = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
		float bsdfPdf = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);

		if (Vera::Core::TraverseAnyHit(geom, P + normal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float cosTheta  = fmaxf(wo_l.z, 0.f); // == dot(wo,normal), macro-normal convention (see comment above)
		float scale     = misWeight * wi_l.z / solidPdf;

		float f0 = EvalReflectance(mat, ray.m_Wavelengths.x);
		float f1 = EvalReflectance(mat, ray.m_Wavelengths.y);
		float f2 = EvalReflectance(mat, ray.m_Wavelengths.z);
		float f3 = EvalReflectance(mat, ray.m_Wavelengths.w);

		float4 nee = ray.m_Throughput;
		nee.x *= Dh * vis * MH::schlickFresnel(f0, cosTheta) * scale;
		nee.y *= Dh * vis * MH::schlickFresnel(f1, cosTheta) * scale;
		nee.z *= Dh * vis * MH::schlickFresnel(f2, cosTheta) * scale;
		nee.w *= Dh * vis * MH::schlickFresnel(f3, cosTheta) * scale;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_Wavelengths, nee, ray.m_Pdf, LeVec);
	}

	__global__ void ShadeKernelHWSSWavefront(
		Core::GeometryBuffers geom,
		RayHWSS* raysIn,
		Core::WavefrontHitRecord* hits,
		const uint32_t* order,
		const Material* materials,
		const MediumHWSS* media,
		LightBVH lightBvh,
		RayHWSS* raysOut,
		uint32_t rayCount,
		FrameBufferHWSS fb,
		EnvMapHWSS envMap,
		uint32_t maxBounces,
		cudaTextureObject_t cieTex)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		unsigned int srcIdx = order[idx];
		RayHWSS ray = raysIn[srcIdx];

		Core::WavefrontHitRecord hit = hits[srcIdx];

		Core::PCG32 rng;
		rng.m_State = ray.m_RngState;

		// ── Participating medium: free-flight sample before surface shading ──
		if (ray.m_MediumIdx != 0) {
			MediumHWSS med = media[ray.m_MediumIdx - 1];
			float sigmaTHero = EvalSigmaT(med, ray.m_Wavelengths.x);
			if (sigmaTHero > 0.f) {
				float tMax = hit.m_Hit ? hit.t : 1e6f;
				float u = rng.nextFloat();
				float sampledDist = -logf(fmaxf(1.f - u, 1e-8f)) / sigmaTHero;

				if (sampledDist < tMax) {
					// Volume scatter event: reweight lanes by Tr_lane/Tr_hero (ratio tracking),
					// sample a new direction from the HG phase function, skip surface shading.
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

					ray.m_Origin      = scatterP;
					ray.m_Direction   = wi;
					ray.m_Throughput  = thpt;
					ray.m_BounceCount += 1;
					ray.flags &= ~RAY_FLAG_DELTA;
					ray.m_RngState = rng.m_State;
					if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;
					raysOut[idx] = ray;
					return;
				} else {
					// No scatter: attenuate to the surface (or to infinity on miss) by the
					// per-lane/hero transmittance ratio.
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

		if (!hit.m_Hit) {
			float4 LeVec = make_float4(
				EvalEnvMap(envMap, ray.m_Wavelengths.x), EvalEnvMap(envMap, ray.m_Wavelengths.y),
				EvalEnvMap(envMap, ray.m_Wavelengths.z), EvalEnvMap(envMap, ray.m_Wavelengths.w));
			AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_Wavelengths, ray.m_Throughput, ray.m_Pdf, LeVec);
			ray.flags |= RAY_FLAG_DEAD;
			raysOut[idx] = ray;
			return;
		}

		uint16_t matId = geom.m_TriMatID[hit.m_PrimIdx];
		Material mat = materials[matId];

		Core::Instance inst = geom.m_DevInstances[hit.m_InstIdx];
		// Smooth (Gouraud-interpolated) normal, not the flat per-triangle facet
		// normal — using the flat normal made faceted low-poly meshes show hard,
		// tessellation-aligned shading edges, especially visible on near-mirror
		// GGX surfaces where a single facet's exact normal fully determines the
		// reflection direction.
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

		float3 P  = ray.m_Origin + hit.t * ray.m_Direction;
		float3 wo = -ray.m_Direction;
		// True geometric normal (always points outward from the surface, e.g. away
		// from a sphere's center) — needed for the Dielectric entering/exiting test
		// below, which must not use the "face the viewer" flipped normal: that one
		// is defined to always point against ray.m_Direction, which would make
		// entering/exiting look identical from both sides of the surface.
		float3 geoNormal = normal;
		if (dot(wo, normal) < 0.f) normal = -normal;

		if (mat.type == MaterialType::Emissive) {
			// Camera/specular bounces (RAY_FLAG_DELTA) have no competing NEE
			// strategy, so this BSDF-sampled hit gets full weight. Otherwise the
			// previous (Lambertian) bounce also sampled this light explicitly via
			// NEE — MIS-combine the two strategies via the power heuristic instead
			// of naively picking one, using EmissiveHitPdf for the light strategy's
			// pdf of the exact direction the BSDF sample happened to take.
			float misWeight = 1.f;
			if (!(ray.flags & RAY_FLAG_DELTA)) {
				float lightPdfArea = EmissiveHitPdf(lightBvh, hit.m_PrimIdx, ray.m_Origin);
				float cosLight = fabsf(dot(normal, ray.m_Direction));
				float dist2 = hit.t * hit.t;
				float lightPdfSolidAngle = (lightPdfArea > 0.f && cosLight > 1e-6f) ? lightPdfArea * dist2 / cosLight : 0.f;
				misWeight = PowerHeuristic(ray.m_BsdfPdf, lightPdfSolidAngle);
			}
			if (misWeight > 0.f) {
				float4 LeVec = make_float4(
					EvalEmission(mat, ray.m_Wavelengths.x), EvalEmission(mat, ray.m_Wavelengths.y),
					EvalEmission(mat, ray.m_Wavelengths.z), EvalEmission(mat, ray.m_Wavelengths.w));
				float4 thptWeighted = make_float4(
					ray.m_Throughput.x * misWeight, ray.m_Throughput.y * misWeight,
					ray.m_Throughput.z * misWeight, ray.m_Throughput.w * misWeight);
				AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_Wavelengths, thptWeighted, ray.m_Pdf, LeVec);
			}
			ray.flags |= RAY_FLAG_DEAD;
			raysOut[idx] = ray;
			return;
		}

		float4 lambertianReflectance = make_float4(0.f, 0.f, 0.f, 0.f);
		if (mat.type == MaterialType::Lambertian) {
			lambertianReflectance = make_float4(
				EvalReflectance(mat, ray.m_Wavelengths.x), EvalReflectance(mat, ray.m_Wavelengths.y),
				EvalReflectance(mat, ray.m_Wavelengths.z), EvalReflectance(mat, ray.m_Wavelengths.w));
			SampleDirectLighting(geom, lightBvh, materials, P, normal, lambertianReflectance, ray, rng, fb, cieTex);
		}

		float3 ggxTangent, ggxBitangent, ggxWo_l;
		float  ggxAlpha = 0.f;
		if (mat.type == MaterialType::GGX) {
			MH::buildONB(normal, ggxTangent, ggxBitangent);
			ggxWo_l = make_float3(dot(wo, ggxTangent), dot(wo, ggxBitangent), dot(wo, normal));
			ggxAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);
			SampleDirectLightingGGX(geom, lightBvh, materials, P, normal, ggxTangent, ggxBitangent,
				ggxWo_l, ggxAlpha, mat, ray, rng, fb, cieTex);
		}

		float3 wi;
		float  pdfDirectional = 0.f;
		float  bsdfMisPdf     = 0.f; // real solid-angle pdf of the sampled wi, used to MIS a later emissive hit
		float4 thpt = ray.m_Throughput;
		bool   valid = true;

		if (mat.type == MaterialType::Lambertian) {
			MH::sampleLambertian(normal, rng.nextFloat2(), wi, pdfDirectional);
			if (pdfDirectional <= 0.f) valid = false;
			else {
				// cosine-weighted sampling cancels cosTheta/pdf, leaving throughput *= albedo
				thpt.x *= lambertianReflectance.x; thpt.y *= lambertianReflectance.y;
				thpt.z *= lambertianReflectance.z; thpt.w *= lambertianReflectance.w;
				bsdfMisPdf = pdfDirectional;
			}
		} else if (mat.type == MaterialType::GGX) {
			float3 h_l  = MH::sampleGGXVNDF(ggxWo_l, ggxAlpha, rng.nextFloat2());
			float3 wi_l = 2.f * dot(ggxWo_l, h_l) * h_l - ggxWo_l;
			if (wi_l.z <= 0.f) { valid = false; }
			else {
				wi = MH::toWorld(wi_l, normal, ggxTangent, ggxBitangent);
				float a2   = ggxAlpha * ggxAlpha;
				float G1wo = 2.f * ggxWo_l.z / (ggxWo_l.z + sqrtf(a2 + (1.f - a2) * ggxWo_l.z * ggxWo_l.z));
				// VNDF importance sampling makes D and most of the pdf cancel, leaving
				// throughput *= F * G2(wi,wo)/G1(wo) — see Heitz 2018. MH::ggxG2 returns
				// the visibility term Vis = G2/(4*wo.z*wi.z), not G2 itself, so recover
				// G2 by multiplying back the 4*wo.z*wi.z factor before dividing by G1(wo).
				float vis   = MH::ggxG2(wi_l, ggxWo_l, ggxAlpha);
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
				pdfDirectional = 1.f; // VNDF importance sampling cancels pdf into the ratio above

				float Dh = MH::evalGGX(h_l, ggxAlpha);
				bsdfMisPdf = G1wo * Dh / fmaxf(4.f * ggxWo_l.z, 1e-7f);
			}
		} else { // Dielectric
			float etaHero = EvalIOR(mat, ray.m_Wavelengths.x);
			bool entering = dot(ray.m_Direction, geoNormal) < 0.f;
			float etaI = entering ? ray.m_IorCurr : etaHero;
			float etaT = entering ? etaHero : 1.f;
			float F = MH::fresnelDielectric(dot(wo, normal), etaI, etaT);

			if (rng.nextFloat() < F) {
				wi = Reflect(wo, normal);
				pdfDirectional = F;
				// mirror reflection is achromatic (Snell's law only bends refraction)
			} else {
				// normal is already oriented onto wo's side by the earlier flip (for
				// both entering and exiting), which is exactly what refract() needs —
				// re-flipping here for the exit case put n on the wrong side and broke
				// cosThetaI's sign inside refract().
				float3 n = normal;
				float eta = etaI / etaT;
				float3 wt;
				if (!MH::refract(-wo, n, eta, wt)) { valid = false; }
				else {
					wi = wt;
					pdfDirectional = 1.f - F;
					ray.m_IorCurr = etaT;
					// entering the dielectric's interior medium (0 = vacuum/exit)
					ray.m_MediumIdx = entering ? mat.mediumIdx : 0;
					if (mat.cauchyB != 0.f) {
						// Dispersive: offset lanes would refract at different angles than
						// the hero direction we traced — collapse to hero-only (Wilkie et al.).
						ray.flags |= RAY_FLAG_DISPERSED;
						ray.m_Pdf.y = ray.m_Pdf.z = ray.m_Pdf.w = 0.f;
						thpt.y = thpt.z = thpt.w = 0.f;
					}
				}
			}
		}

		if (!valid || pdfDirectional <= 0.f) {
			ray.flags |= RAY_FLAG_DEAD;
			raysOut[idx] = ray;
			return;
		}

		ray.m_Origin      = P + wi * 1e-4f;
		ray.m_Direction   = wi;
		ray.m_Throughput  = thpt;
		ray.m_BsdfPdf     = bsdfMisPdf; // only meaningful when RAY_FLAG_DELTA is unset (Lambertian, GGX)
		ray.m_BounceCount += 1;
		ray.flags &= ~RAY_FLAG_DELTA;
		if (mat.type == MaterialType::Dielectric)
			ray.flags |= RAY_FLAG_DELTA; // no NEE/eval written for Dielectric yet — treat as delta
		ray.m_RngState = rng.m_State;

		if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;

		raysOut[idx] = ray;
	}
}
