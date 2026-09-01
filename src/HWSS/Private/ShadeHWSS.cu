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

	// Adds a per-lane spectral contribution to the framebuffer: weight = 1/(activeLanes*pdf_lane).
	// activeLanes (lanes with pdf > 0) is NOT always HWSS_LANES: once a path hits a
	// dispersive event, RAY_FLAG_DISPERSED zeroes 3 of the 4 lanes' pdf, leaving only
	// the hero lane valid. Dividing by the fixed lane count instead of the number of
	// lanes actually still contributing would silently under-weight every dispersed
	// sample by that same factor — a real, sample-count-independent bias, not noise.
	//
	// `sensorWavelengths`, not trace wavelengths, set each lane's CIE weight — they
	// diverge once a fluorescent event shifts a lane's trace wavelength (the lane
	// keeps depositing at the wavelength it was born with).
	__device__ inline void AccumulateContribution(
		FrameBufferHWSS& fb, uint32_t pixelId, cudaTextureObject_t cieTex,
		const float4& sensorWavelengths, const float4& throughput, const float4& pdf, const float4& Le)
	{
		float wl[4]     = { sensorWavelengths.x, sensorWavelengths.y, sensorWavelengths.z, sensorWavelengths.w };
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

	// Unbiased Russian roulette path termination: below minBounces every path
	// continues (avoids prematurely truncating short paths where it barely
	// helps), then survival probability tracks the throughput itself (higher
	// remaining energy => more likely to keep tracing) and surviving lanes are
	// rescaled by 1/q so the estimator stays unbiased in expectation.
	__device__ inline bool RussianRoulette(float4& thpt, HybridRNG& rng, uint32_t bounceCount, uint32_t minBounces = 4) {
		if (bounceCount < minBounces) return true;
		float maxComp = fmaxf(fmaxf(thpt.x, thpt.y), fmaxf(thpt.z, thpt.w));
		float q = fminf(fmaxf(maxComp, 0.05f), 0.95f);
		if (rng.nextFloat() > q) return false;
		float invQ = 1.f / q;
		thpt.x *= invQ; thpt.y *= invQ; thpt.z *= invQ; thpt.w *= invQ;
		return true;
	}

	// Woodcock/spectral tracking through a heterogeneous medium (Kutz, Habel, Li & Novák 2017,
	// "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes", Algorithm 4,
	// with the history-aware average-based collision probabilities of their Section 5.1.2 —
	// `thpt` (already the path's accumulated per-lane throughput) IS exactly the "path history"
	// weight vector their formula calls for, so no separate bookkeeping is needed. P_a is folded
	// to 0 per their Section 5.1.3 ("if the medium is non-emissive, we can further reduce the
	// throughput by not simulating absorption collisions") — matching the existing homogeneous
	// scheme's model below, which also has no separate terminal-absorption state, only
	// scatter-vs-ballistic.
	//
	// The majorant (med.majorantSigmaT) must bound sigmaT(x,lambda) for every position AND every
	// visible wavelength simultaneously (see MediumHWSS.h) — distance is sampled once, using
	// this single scalar majorant, shared across all 4 lanes; each lane's own physical
	// coefficient is folded in via the per-lane weight update below, which is what lets 4
	// spectrally-varying lanes share one coherent distance sample instead of needing 4
	// independent free-flight walks.
	//
	// Returns true if a real scattering collision was found before tMax (thpt updated to the
	// accumulated weight through and including that collision; scatterT is the distance); false
	// if the walk reached tMax without one (thpt updated for the ballistic pass-through).
	__device__ inline bool HeterogeneousFreeFlight(
		const MediumHWSS& med, float3 origin, float3 dir, float tMax,
		const float wl[4], float4& thpt, HybridRNG& rng, float& scatterT)
	{
		float majorant = med.majorantSigmaT;
		if (majorant <= 0.f) { scatterT = tMax; return false; }

		float* thptArr[4] = { &thpt.x, &thpt.y, &thpt.z, &thpt.w };
		float t = 0.f;

		for (int iter = 0; iter < 1024; ++iter) { // safety cap, same defensive convention as the BVH traversal stacks
			float u = rng.nextFloat();
			t += -logf(fmaxf(1.f - u, 1e-8f)) / majorant;
			if (t >= tMax) { scatterT = tMax; return false; }

			float3 p = origin + t * dir;
			float sigmaS[4], sigmaT_[4];
			float avgS = 0.f, avgN = 0.f;
			for (int lane = 0; lane < HWSS_LANES; ++lane) {
				sigmaT_[lane] = EvalSigmaTAt(med, p, wl[lane]);
				sigmaS[lane]  = EvalSigmaSAt(med, p, wl[lane]);
				float sigmaN  = fmaxf(majorant - sigmaT_[lane], 0.f);
				avgS += fabsf(sigmaS[lane] * (*thptArr[lane]));
				avgN += fabsf(sigmaN       * (*thptArr[lane]));
			}
			avgS *= 0.25f; avgN *= 0.25f;
			float c = avgS + avgN;
			if (c <= 1e-12f) continue; // numerically vacuum here -- treat as a null collision, keep marching

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
		return false; // safety cap hit -- ballistic pass-through rather than looping forever
	}

	// Per-lane transmittance along a shadow-ray segment through a medium — used for NEE
	// occlusion through participating media, which previously used a purely binary any-hit test
	// with no medium attenuation at all (true for both homogeneous and heterogeneous media; this
	// closes that gap for both). Homogeneous media get the exact analytic transmittance (zero
	// variance, and the current demo scene's only medium usage would be this branch).
	// Heterogeneous media use ratio tracking (Novák et al. 2014): unlike
	// HeterogeneousFreeFlight, this walk never "stops" at a collision — every majorant-rate
	// event along the segment contributes a sigmaN/majorant factor regardless of type, since
	// this estimates pure survival probability, not a scattered path.
	__device__ inline float4 EvalTransmittance(
		const MediumHWSS& med, float3 origin, float3 dir, float segLength, const float4& wavelengths, HybridRNG& rng)
	{
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

	// NEE against LightBVH area lights, Lambertian surfaces only. `reflectance` is the
	// surface's per-lane albedo, precomputed once by the caller and shared with the
	// BSDF-sampling step below so it isn't evaluated twice per lane per shading call.
	__device__ inline void SampleDirectLighting(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float4& reflectance,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
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

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

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
		nee.x *= MH::evalLambertian(reflectance.x) * scale * tr.x;
		nee.y *= MH::evalLambertian(reflectance.y) * scale * tr.y;
		nee.z *= MH::evalLambertian(reflectance.z) * scale * tr.z;
		nee.w *= MH::evalLambertian(reflectance.w) * scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	// NEE for a fluorescent Lambertian's re-emission lobe: direct light converted
	// through Phi instead of reflected. One light + one shared excitation
	// wavelength are sampled per call (shared because the excitation integral
	// doesn't depend on the output wavelength); each lane then weights by its own
	// emission density, so lanes far from the emission band contribute ~0. MIS is
	// paired against the fluoresced BSDF continuation in the caller, which samples
	// the same light via the same cosine-hemisphere direction.
	__device__ inline void SampleDirectLightingFluor(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
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

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		float lamI = SampleFluorExcitation(rng.nextFloat(), mat.fluorLamEx, mat.fluorSigma,
			mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
		const Material& lmat = materials[ls.materialId];
		float LeI = EvalEmission(lmat, lamI); // light emission at the sampled excitation wavelength
		if (LeI <= 0.f) return;

		float misWeight = PowerHeuristic(solidPdf, MH::lambertianPdf(wi, normal));
		float commonScale = misWeight * cosSurf / solidPdf * LeI * mat.fluorQY * mat.fluorAbsNorm / PI;

		float4 lo  = ray.m_Wavelengths; // l_o per lane == sensor wavelength at this (pre-shift) vertex
		float4 nee = ray.m_Throughput;
		nee.x *= FluorEmissionPdf(lo.x, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.x;
		nee.y *= FluorEmissionPdf(lo.y, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.y;
		nee.z *= FluorEmissionPdf(lo.z, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.z;
		nee.w *= FluorEmissionPdf(lo.w, mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm) * commonScale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf,
			make_float4(1.f, 1.f, 1.f, 1.f));
	}

	// NEE against LightBVH area lights, GGX surfaces. Mirrors SampleDirectLighting but
	// evaluates the microfacet BRDF (D * Vis * F) for the sampled light direction instead
	// of assuming Lambertian, and MIS-weights against the VNDF-sampling pdf for that same
	// direction. Fresnel is evaluated at the macro cosTheta = dot(wo,normal), matching the
	// same approximation the BSDF-sampling path below bakes into its throughput — using the
	// textbook-correct dot(wo,h) here instead would make the two MIS strategies estimate
	// slightly different functions and bias the combined estimate.
	__device__ inline void SampleDirectLightingGGX(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
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

		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + normal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

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
		nee.x *= Dh * vis * MH::schlickFresnel(f0, cosTheta) * scale * tr.x;
		nee.y *= Dh * vis * MH::schlickFresnel(f1, cosTheta) * scale * tr.y;
		nee.z *= Dh * vis * MH::schlickFresnel(f2, cosTheta) * scale * tr.z;
		nee.w *= Dh * vis * MH::schlickFresnel(f3, cosTheta) * scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	// NEE against LightBVH area lights, ROUGH dielectric surfaces only (mat.roughness > 0;
	// perfectly smooth dielectric stays delta/BSDF-sampling-only, same as before this
	// feature — NEE is meaningless for a delta BSDF). Evaluates whichever microfacet lobe
	// (reflection or transmission) is geometrically consistent with the light-sampled
	// direction wi, MIS-weighted against the matching VNDF-sampling pdf.
	//
	// Both lobes use Walter et al. 2007's microfacet transmission model. The BTDF value
	// below omits the eta^2 radiance-compression factor a strictly physical derivation
	// would include — this matches the existing BSDF-sampled smooth/rough transmission
	// path (see the throughput derivation comment below), which never applied it either;
	// keeping NEE consistent with that same convention means both MIS strategies estimate
	// the same integrand. The VNDF-sampling PDF Jacobian (which converts the sampled
	// microfacet-normal density into a direction density) does still need the eta^2 term —
	// that's a probability measure-change, not a radiance scaling, so it's unrelated to the
	// value-side convention choice.
	__device__ inline void SampleDirectLightingDielectricRough(
		Core::GeometryBuffers& geom, const LightBVH& lightBvh, const Material* materials, const MediumHWSS* media,
		const float3& P, const float3& normal, const float3& tangent, const float3& bitangent,
		const float3& wo_l, float alpha, float etaI, float etaT,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
	{
		if (lightBvh.lightCount == 0 || wo_l.z <= 0.f) return;

		LightSample ls = SampleLightBVH(lightBvh, geom, P, rng.nextFloat(), rng.nextFloat(), rng.nextFloat2());
		if (ls.pdf <= 0.f) return;

		float3 toLight = ls.position - P;
		float  dist2   = fmaxf(dot(toLight, toLight), 1e-8f);
		float  dist    = sqrtf(dist2);
		float3 wi      = toLight / dist;

		// Unlike Lambertian/GGX's NEE (which only ever sample a reflection direction and
		// so only need |dot(lightNormal, -wi)|), a transmitted wi can approach the light
		// from either face of its emissive geometry, so this uses |dot| unsigned rather
		// than culling the back face.
		float solidPdf = ls.pdf * dist2 / fmaxf(fabsf(dot(ls.normal, wi)), 1e-6f);
		if (solidPdf <= 0.f) return;

		float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
		if (fabsf(wi_l.z) < 1e-6f) return;

		float a2   = alpha * alpha;
		float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));

		float value = 0.f;   // achromatic BSDF value (dielectric has no per-wavelength albedo)
		float bsdfPdf = 0.f;
		float3 shadowNormal = normal;

		if (wi_l.z > 0.f) {
			// Reflection lobe: same construction as GGX's NEE above.
			float3 h_l = normalize(wo_l + wi_l);
			float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
			float  F  = MH::fresnelDielectric(cosThetaH, etaI, etaT);
			float  Dh = MH::evalGGX(h_l, alpha);
			float  vis = MH::ggxG2(wi_l, wo_l, alpha);
			value   = Dh * vis * F;
			bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
		} else {
			// Transmission lobe: reconstruct the generalized half-vector from wi_l/wo_l
			// (Walter et al. eq 16), oriented onto wo's side to match D(h)'s convention.
			float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
			if (ht.z < 0.f) ht = -ht;
			float cosThetaH = dot(wo_l, ht);
			if (cosThetaH <= 0.f) return;

			float F  = MH::fresnelDielectric(cosThetaH, etaI, etaT);
			float Dh = MH::evalGGX(ht, alpha);
			float vis = MH::ggxG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
			float G2  = vis * 4.f * wo_l.z * fabsf(wi_l.z);

			float denomJ = etaI * dot(wo_l, ht) + etaT * dot(wi_l, ht);
			if (fabsf(denomJ) < 1e-7f) return;

			float wiDotH = dot(wi_l, ht);
			value = (1.f - F) * Dh * G2 * fabsf(wiDotH * cosThetaH)
				/ fmaxf(wo_l.z * fabsf(wi_l.z), 1e-7f) / (denomJ * denomJ);

			float dwh_dwi = etaT * etaT * fabsf(wiDotH) / (denomJ * denomJ);
			bsdfPdf = (1.f - F) * (G1wo * Dh * fmaxf(cosThetaH, 0.f) / wo_l.z) * dwh_dwi;

			// wi crosses to the far side of the surface — offset the shadow ray origin the
			// other way, or it starts on the wrong side of the geometry it just left.
			shadowNormal = -normal;
		}

		if (value <= 0.f || bsdfPdf <= 0.f) return;
		if (Vera::Core::TraverseAnyHit(geom, P + shadowNormal * 1e-4f, wi, 1e-4f, dist - 2e-3f)) return;

		// Uses the CURRENT medium (ray.m_MediumIdx) for both lobes. Correct for the reflection
		// lobe (shadow ray stays on the same side); a known approximation for the transmission
		// lobe, where the shadow ray actually crosses into whichever medium is on the far side
		// of the interface — not necessarily the same one. Left as a documented limitation
		// rather than threading mat.mediumIdx-based medium-crossing logic through NEE here.
		float4 tr = make_float4(1.f, 1.f, 1.f, 1.f);
		if (ray.m_MediumIdx != 0)
			tr = EvalTransmittance(media[ray.m_MediumIdx - 1], P + shadowNormal * 1e-4f, wi, dist - 2e-3f, ray.m_Wavelengths, rng);

		const Material& lmat = materials[ls.materialId];
		float4 LeVec = make_float4(
			EvalEmission(lmat, ray.m_Wavelengths.x), EvalEmission(lmat, ray.m_Wavelengths.y),
			EvalEmission(lmat, ray.m_Wavelengths.z), EvalEmission(lmat, ray.m_Wavelengths.w));

		float misWeight = PowerHeuristic(solidPdf, bsdfPdf);
		float scale     = misWeight * fabsf(wi_l.z) * value / solidPdf;

		float4 nee = ray.m_Throughput;
		nee.x *= scale * tr.x; nee.y *= scale * tr.y; nee.z *= scale * tr.z; nee.w *= scale * tr.w;

		AccumulateContribution(fb, ray.pixelId, cieTex, ray.m_SensorWavelengths, nee, ray.m_Pdf, LeVec);
	}

	// Bit tags for the material lobe SampleDirectLightingEnv should evaluate.
	enum EnvBsdfKind { ENV_BSDF_LAMBERT = 0, ENV_BSDF_GGX = 1, ENV_BSDF_ROUGH_DIELECTRIC = 2 };

	// NEE against the environment map (textured mode only — a constant env is
	// picked up fine by BSDF sampling on the miss path, no NEE for it). Importance-
	// samples a direction from the lat-long luminance distribution, evaluates the
	// surface BSDF for that direction, shadow-tests to infinity, and MIS-weights
	// against the same BSDF-sampling pdf the miss-path env hit uses. Mirrors the
	// LightBVH NEE functions above; the reflection/transmission lobe math for the
	// GGX and rough-dielectric branches is identical to SampleDirectLightingGGX /
	// SampleDirectLightingDielectricRough, just driven by an env-sampled wi.
	//
	// Skipped while inside a medium (ray.m_MediumIdx != 0): an env shadow ray from
	// inside a dielectric would need to be tracked across the exit interface first;
	// env NEE overwhelmingly fires on opaque surfaces in open air, so this is left
	// as a documented gap rather than threading medium-crossing logic through here.
	__device__ inline void SampleDirectLightingEnv(
		Core::GeometryBuffers& geom, const EnvMapHWSS& envMap,
		const float3& P, const float3& normal,
		int bsdfKind, const float4& lambReflectance,
		const float3& tangent, const float3& bitangent, const float3& wo_l,
		float alpha, float etaI, float etaT, const Material& mat,
		RayHWSS& ray, HybridRNG& rng, FrameBufferHWSS& fb, cudaTextureObject_t cieTex)
	{
		if (envMap.tex == 0 || ray.m_MediumIdx != 0) return;

		float envPdf;
		float3 wi = SampleEnvMap(envMap, rng.nextFloat2(), envPdf);
		if (envPdf <= 0.f) return;

		float3 shadowNormal = normal;
		float  geomTerm     = 0.f;   // |cos| folded into the estimator
		float  bsdfPdf      = 0.f;   // solid-angle pdf of wi under BSDF sampling (for MIS)
		float4 fVal = make_float4(0.f, 0.f, 0.f, 0.f); // per-lane BSDF value

		if (bsdfKind == ENV_BSDF_LAMBERT) {
			float cosSurf = dot(wi, normal);
			if (cosSurf <= 0.f) return;
			geomTerm = cosSurf;
			bsdfPdf  = MH::lambertianPdf(wi, normal);
			fVal = make_float4(
				MH::evalLambertian(lambReflectance.x), MH::evalLambertian(lambReflectance.y),
				MH::evalLambertian(lambReflectance.z), MH::evalLambertian(lambReflectance.w));
		}
		else if (bsdfKind == ENV_BSDF_GGX) {
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (wi_l.z <= 0.f) return;

			float3 h_l = normalize(wo_l + wi_l);
			float  Dh  = MH::evalGGX(h_l, alpha);
			float  vis = MH::ggxG2(wi_l, wo_l, alpha);
			float  a2  = alpha * alpha;
			float  G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			bsdfPdf  = G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			geomTerm = wi_l.z;

			float cosTheta = fmaxf(wo_l.z, 0.f); // macro-normal convention, matches SampleDirectLightingGGX
			float f0 = EvalReflectance(mat, ray.m_Wavelengths.x);
			float f1 = EvalReflectance(mat, ray.m_Wavelengths.y);
			float f2 = EvalReflectance(mat, ray.m_Wavelengths.z);
			float f3 = EvalReflectance(mat, ray.m_Wavelengths.w);
			fVal = make_float4(
				Dh * vis * MH::schlickFresnel(f0, cosTheta), Dh * vis * MH::schlickFresnel(f1, cosTheta),
				Dh * vis * MH::schlickFresnel(f2, cosTheta), Dh * vis * MH::schlickFresnel(f3, cosTheta));
		}
		else { // ENV_BSDF_ROUGH_DIELECTRIC — achromatic value, mirrors SampleDirectLightingDielectricRough
			if (wo_l.z <= 0.f) return;
			float3 wi_l = make_float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, normal));
			if (fabsf(wi_l.z) < 1e-6f) return;

			float a2   = alpha * alpha;
			float G1wo = 2.f * wo_l.z / (wo_l.z + sqrtf(a2 + (1.f - a2) * wo_l.z * wo_l.z));
			float value = 0.f;

			if (wi_l.z > 0.f) {
				float3 h_l = normalize(wo_l + wi_l);
				float  cosThetaH = fmaxf(dot(wo_l, h_l), 0.f);
				float  F  = MH::fresnelDielectric(cosThetaH, etaI, etaT);
				float  Dh = MH::evalGGX(h_l, alpha);
				float  vis = MH::ggxG2(wi_l, wo_l, alpha);
				value   = Dh * vis * F;
				bsdfPdf = F * G1wo * Dh / fmaxf(4.f * wo_l.z, 1e-7f);
			} else {
				float3 ht = normalize(-(etaT * wi_l + etaI * wo_l));
				if (ht.z < 0.f) ht = -ht;
				float cosThetaH = dot(wo_l, ht);
				if (cosThetaH <= 0.f) return;

				float F  = MH::fresnelDielectric(cosThetaH, etaI, etaT);
				float Dh = MH::evalGGX(ht, alpha);
				float vis = MH::ggxG2(make_float3(wi_l.x, wi_l.y, fabsf(wi_l.z)), wo_l, alpha);
				float G2  = vis * 4.f * wo_l.z * fabsf(wi_l.z);

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
		float scale     = misWeight * geomTerm / envPdf;

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
		cudaTextureObject_t cieTex)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		unsigned int srcIdx = order[idx];
		RayHWSS ray = LoadRay(coreIn, extIn, srcIdx);

		Core::WavefrontHitRecord hit = hits[srcIdx];

		HybridRNG rng{};
		rng.pcg.m_State = ray.m_RngState;
		rng.sampleIdx = ray.m_SampleIdx;
		rng.pixelId   = ray.pixelId;
		// ray.m_BounceCount here is the count BEFORE this shading call's increment(s) below
		// (i.e. how many bounces have already happened) — used once to pick this call's QMC
		// dimension range, consistent regardless of which material branch ends up running.
		rng.dimBase   = (uint32_t)kQmcCameraDims + (uint32_t)ray.m_BounceCount * (uint32_t)kQmcDimsPerBounce;

		// ── Participating medium: free-flight sample before surface shading ──
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
						// Same continuation shape as the homogeneous scatter branch below --
						// HeterogeneousFreeFlight has already folded the per-lane spectral-tracking
						// weight into thpt, so no separate hero-ratio reweighting is needed here.
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

						ray.m_Origin      = scatterP;
						ray.m_Direction   = wi;
						ray.m_Throughput  = thpt;
						ray.flags &= ~RAY_FLAG_DELTA;
						ray.m_RngState = rng.pcg.m_State;
						if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					} else {
						// Ballistic pass-through: the accumulated null-collision weight is the
						// unbiased heterogeneous transmittance ratio, already per-lane.
						ray.m_Throughput = thpt;
					}
				}
			} else {
			float sigmaTHero = EvalSigmaT(med, ray.m_Wavelengths.x);
			if (sigmaTHero > 0.f) {
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

					ray.m_BounceCount += 1;
					if (!RussianRoulette(thpt, rng, ray.m_BounceCount)) {
						ray.flags |= RAY_FLAG_DEAD;
						ray.m_RngState = rng.pcg.m_State;
						StoreRay(coreOut, extOut, idx, ray);
						return;
					}

					ray.m_Origin      = scatterP;
					ray.m_Direction   = wi;
					ray.m_Throughput  = thpt;
					ray.flags &= ~RAY_FLAG_DELTA;
					ray.m_RngState = rng.pcg.m_State;
					if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;
					StoreRay(coreOut, extOut, idx, ray);
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
		}

		if (!hit.m_Hit) {
			float3 missDir = ray.m_Direction;
			float4 LeVec = make_float4(
				EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.x), EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.y),
				EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.z), EvalEnvMapDir(envMap, missDir, ray.m_Wavelengths.w));
			// MIS against env NEE from the previous vertex: a non-delta bounce that
			// missed to a textured env also had that direction explicitly sampled by
			// SampleDirectLightingEnv, so down-weight this BSDF-sampled hit. Delta
			// (camera / mirror) bounces have no competing env-NEE strategy => full weight.
			float misScalar = 1.f;
			if (envMap.tex != 0 && !(ray.flags & RAY_FLAG_DELTA)) {
				float envPdf = EnvMapPdf(envMap, missDir);
				misScalar = PowerHeuristic(ray.m_BsdfPdf, envPdf);
			}
			// Fluoresced lanes have no env-NEE counterpart yet, so they always get full weight.
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
			float misScalar = 1.f;
			if (!(ray.flags & RAY_FLAG_DELTA)) {
				float lightPdfArea = EmissiveHitPdf(lightBvh, hit.m_PrimIdx, ray.m_Origin);
				float cosLight = fabsf(dot(normal, ray.m_Direction));
				float dist2 = hit.t * hit.t;
				float lightPdfSolidAngle = (lightPdfArea > 0.f && cosLight > 1e-6f) ? lightPdfArea * dist2 / cosLight : 0.f;
				misScalar = PowerHeuristic(ray.m_BsdfPdf, lightPdfSolidAngle);
			}
			// A fluoresced lane now has a competing NEE strategy (SampleDirectLightingFluor,
			// called at the previous vertex against this same light and direction pdf), so
			// it uses the same scalar MIS weight as every other lane.
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
			// Fluorescent lobe direct lighting (area lights only — env NEE for it isn't built yet).
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
		float  bsdfMisPdf     = 0.f; // real solid-angle pdf of the sampled wi, used to MIS a later emissive hit
		float4 thpt = ray.m_Throughput;
		bool   valid = true;
		bool   didFluoresce = false; // set if any lane took the wavelength-shift channel this bounce

		if (mat.type == MaterialType::Lambertian) {
			MH::sampleLambertian(normal, rng.nextFloat2(), wi, pdfDirectional);
			if (pdfDirectional <= 0.f) valid = false;
			else if (!IsFluorescent(mat)) {
				// cosine-weighted sampling cancels cosTheta/pdf, leaving throughput *= albedo
				thpt.x *= lambertianReflectance.x; thpt.y *= lambertianReflectance.y;
				thpt.z *= lambertianReflectance.z; thpt.w *= lambertianReflectance.w;
				bsdfMisPdf = pdfDirectional;
			} else {
				// Fluorescent Lambertian: re-emission is diffuse, so the cosine-sampled `wi`
				// is shared — only each lane's wavelength/throughput branches. Per lane we
				// route to elastic reflection or a wavelength shift; draws happen
				// unconditionally so a material-sorted batch stays draw-count-uniform.
				bsdfMisPdf = pdfDirectional;
				unsigned char fluorMask = 0;
				float uRoute   = rng.nextFloat();
				float lamInU[4] = { rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat() };
				float sWl[4]  = { ray.m_SensorWavelengths.x, ray.m_SensorWavelengths.y,
				                  ray.m_SensorWavelengths.z, ray.m_SensorWavelengths.w };
				float tWl[4]  = { ray.m_Wavelengths.x, ray.m_Wavelengths.y,
				                  ray.m_Wavelengths.z, ray.m_Wavelengths.w };
				float refl[4] = { lambertianReflectance.x, lambertianReflectance.y,
				                  lambertianReflectance.z, lambertianReflectance.w };
				float* thptArr[4] = { &thpt.x, &thpt.y, &thpt.z, &thpt.w };
				for (int lane = 0; lane < HWSS_LANES; ++lane) {
					// pFl ~ 0 outside the emission band, so an out-of-band lane (or any
					// lane on a non-fluorescent scene) always takes `*= refl` below —
					// bit-identical to the plain elastic path.
					float pFl = FluorRoutingProb(sWl[lane], mat.fluorLamEm, mat.fluorSigma);
					if (uRoute < pFl) {
						didFluoresce = true;
						fluorMask |= (unsigned char)(1u << lane);
						// Backward path: the current wavelength is the fluorescent OUTPUT;
						// sample the INPUT wavelength from a(lamIn)/N_a (see
						// FluorescenceHWSS.h), which cancels a(lamIn) exactly.
						float lamInNm = SampleFluorExcitation(lamInU[lane], mat.fluorLamEx, mat.fluorSigma,
							mat.fluorAbsCdfLo, mat.fluorAbsCdfHi);
						float phiOverQ = mat.fluorQY * mat.fluorAbsNorm
							* FluorEmissionPdf(tWl[lane], mat.fluorLamEm, mat.fluorSigma, mat.fluorEmNorm);
						*thptArr[lane] *= phiOverQ / pFl;
						tWl[lane] = lamInNm; // the path continues at the input wavelength
					} else {
						*thptArr[lane] *= refl[lane] / (1.f - pFl);
					}
				}
				ray.m_Wavelengths = make_float4(tWl[0], tWl[1], tWl[2], tWl[3]);
				ray.m_LaneFluoresced = fluorMask;
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

			if (mat.roughness <= 0.f) {
				// Perfectly smooth: mirror reflect / Snell refract, delta BSDF (no NEE — see
				// SampleDirectLightingDielectricRough's comment for why NEE needs roughness > 0).
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
						if (IsDispersive(mat)) {
							// Dispersive: offset lanes would refract at different angles than
							// the hero direction we traced — collapse to hero-only (Wilkie et al.).
							ray.flags |= RAY_FLAG_DISPERSED;
							ray.m_Pdf.y = ray.m_Pdf.z = ray.m_Pdf.w = 0.f;
							thpt.y = thpt.z = thpt.w = 0.f;
						}
					}
				}
			} else {
				// Rough (Walter et al. 2007 microfacet transmission, VNDF-sampled — see
				// SampleDirectLightingDielectricRough above for the full derivation). Reflect
				// and refract are both stochastically chosen by Fresnel at the SAMPLED
				// microfacet normal h_l (not the macro normal), same two-lobe structure as the
				// smooth case above but generalized to a rough h_l instead of the fixed normal.
				//
				// With VNDF importance sampling + Fresnel-weighted stochastic lobe selection,
				// both lobes' throughput reduces to the IDENTICAL ratio G2(wi,wo)/G1(wo) that
				// GGX's reflect branch already uses (Fresnel cancels exactly against the
				// selection probability, and — for transmission — the eta^2 Jacobian terms
				// cancel exactly against the BTDF's own eta^2/denominator terms). This is a
				// standard, well-known identity, not a coincidence of this specific derivation.
				float3 dTangent, dBitangent;
				MH::buildONB(normal, dTangent, dBitangent);
				float3 dWo_l = make_float3(dot(wo, dTangent), dot(wo, dBitangent), dot(wo, normal));
				float  dAlpha = fmaxf(mat.roughness * mat.roughness, 1e-4f);

				SampleDirectLightingDielectricRough(geom, lightBvh, materials, media, P, normal, dTangent, dBitangent,
					dWo_l, dAlpha, etaI, etaT, ray, rng, fb, cieTex);
				SampleDirectLightingEnv(geom, envMap, P, normal, ENV_BSDF_ROUGH_DIELECTRIC, make_float4(0.f, 0.f, 0.f, 0.f),
					dTangent, dBitangent, dWo_l, dAlpha, etaI, etaT, mat, ray, rng, fb, cieTex);

				float3 h_l = MH::sampleGGXVNDF(dWo_l, dAlpha, rng.nextFloat2());
				float  F   = MH::fresnelDielectric(dot(dWo_l, h_l), etaI, etaT);
				float  a2  = dAlpha * dAlpha;
				float  G1wo = 2.f * dWo_l.z / (dWo_l.z + sqrtf(a2 + (1.f - a2) * dWo_l.z * dWo_l.z));
				float  Dh   = MH::evalGGX(h_l, dAlpha);

				if (rng.nextFloat() < F) {
					float3 wi_l = 2.f * dot(dWo_l, h_l) * h_l - dWo_l;
					if (wi_l.z <= 0.f) { valid = false; }
					else {
						wi = MH::toWorld(wi_l, normal, dTangent, dBitangent);
						float vis   = MH::ggxG2(wi_l, dWo_l, dAlpha);
						float ratio = (vis * 4.f * dWo_l.z * wi_l.z) / fmaxf(G1wo, 1e-7f);
						thpt.x *= ratio; thpt.y *= ratio; thpt.z *= ratio; thpt.w *= ratio;
						pdfDirectional = 1.f; // sentinel — real pdf is bsdfMisPdf below
						bsdfMisPdf = F * G1wo * Dh / fmaxf(4.f * dWo_l.z, 1e-7f);
					}
				} else {
					float eta = etaI / etaT;
					float3 wt_l;
					if (!MH::refract(-dWo_l, h_l, eta, wt_l)) { valid = false; }
					else if (wt_l.z >= 0.f) { valid = false; }
					else {
						wi = MH::toWorld(wt_l, normal, dTangent, dBitangent);
						float vis   = MH::ggxG2(make_float3(wt_l.x, wt_l.y, fabsf(wt_l.z)), dWo_l, dAlpha);
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

		ray.m_Origin      = P + wi * 1e-4f;
		ray.m_Direction   = wi;
		ray.m_Throughput  = thpt;
		ray.m_BsdfPdf     = bsdfMisPdf; // only meaningful when RAY_FLAG_DELTA is unset (Lambertian, GGX)
		ray.flags &= ~RAY_FLAG_DELTA;
		if (mat.type == MaterialType::Dielectric && mat.roughness <= 0.f)
			ray.flags |= RAY_FLAG_DELTA; // perfectly smooth — still a true delta BSDF, no NEE for it
		// The per-lane fluoresced mask is set in the Lambertian branch; any other
		// (elastic) bounce clears it so a following light hit resumes normal MIS.
		if (!didFluoresce) ray.m_LaneFluoresced = 0;
		ray.m_RngState = rng.pcg.m_State;

		if (ray.m_BounceCount >= maxBounces) ray.flags |= RAY_FLAG_DEAD;

		StoreRay(coreOut, extOut, idx, ray);
	}
}
