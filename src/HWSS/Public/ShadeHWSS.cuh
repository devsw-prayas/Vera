#pragma once
#include <cuda_runtime.h>
#include <CoreUtils.h>

#include "RayHWSS.h"
#include "MaterialHWSS.h"
#include "EnvMapHWSS.h"
#include "MediumHWSS.h"
#include "LightBVHHWSS.h"
#include "FrameBufferHWSS.cuh"

namespace BSPT::Spectral::HWSS {
	// Scatters each ray at its hit (or accumulates env/emissive contribution and
	// kills it on miss/light hit), writing the next bounce's ray into raysOut.
	// Per-lane m_Pdf is the wavelength-sampling pdf (fixed at ray-gen, zeroed on
	// dispersion collapse) — only used to weight framebuffer contributions.
	// Per-lane m_Throughput accumulates f*cosTheta/pdfDirectional every bounce,
	// where pdfDirectional is the single shared scatter-direction pdf (BSDF
	// sampling doesn't fork per lane, except once RAY_FLAG_DISPERSED collapses
	// the path to the hero lane only).
	//
	// NEE (next-event estimation) fires only on Lambertian hits, sampling
	// LightBVH area lights directly (GGX/Dielectric skip NEE — no BSDF *eval*
	// was written for them, only sample). The NEE contribution and any
	// BSDF-sampled ray that later lands on the same light are combined via
	// power-heuristic MIS (RayHWSS::m_BsdfPdf carries the sampling pdf forward;
	// EmissiveHitPdf recovers the light strategy's pdf for the hit direction).
	// Camera/specular (RAY_FLAG_DELTA) bounces have no competing NEE strategy,
	// so their emissive hits always get full weight.
	//
	// Participating media: if the ray is inside a non-vacuum medium
	// (m_MediumIdx != 0), a free-flight distance is sampled from the hero
	// wavelength's sigma_t before surface shading runs; per-lane throughput is
	// corrected by the transmittance ratio to the hero (the same trick HWSS
	// uses at surfaces), matching Wilkie et al.'s treatment of spectral media.
	__global__ void ShadeKernelHWSSWavefront(
		Core::GeometryBuffers geom,
		RayHWSS* raysIn,
		Core::WavefrontHitRecord* hits,
		const Material* materials,
		const MediumHWSS* media,
		LightBVH lightBvh,
		RayHWSS* raysOut,
		uint32_t rayCount,
		FrameBufferHWSS fb,
		EnvMapHWSS envMap,
		uint32_t maxBounces);
}
