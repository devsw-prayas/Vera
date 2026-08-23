#pragma once
#include <vector_types.h>
#include <cstdint>

namespace Vera::Spectral::HWSS {
	static constexpr unsigned char RAY_FLAG_DEAD      = 0x01;
	static constexpr unsigned char RAY_FLAG_DELTA     = 0x02;
	// Set once a wavelength-dependent (dispersive) event is sampled: per Wilkie et al.,
	// the offset lanes are no longer valid samples of the same path and are zeroed out,
	// leaving only the hero lane (m_Wavelengths.x) carrying weight from that point on.
	static constexpr unsigned char RAY_FLAG_DISPERSED = 0x04;

	// Hero + 3 offset wavelengths sampled per path (Hero Wavelength Spectral Sampling).
	static constexpr int HWSS_LANES = 4;

	// Wavefront HWSS ray. Wider than an RGB wavefront ray because MIS across the four
	// lanes requires per-lane throughput and pdf (not just one scalar/triple):
	// Origin(12) Dir(12) Wavelengths(16) Throughput(16) Pdf(16) RngState(8)
	// pixelId(4) BounceCount(1) flags(1) IorCurr(4) MediumIdx(1) BsdfPdf(4) Pad(3).
	//
	// This is the LOGICAL/local representation used inside kernel bodies (built in
	// registers via LoadRay, written back via StoreRay) — actual wavefront-buffer
	// storage is struct-of-arrays (RayCoreSoA/RayExtSoA below), not an array of this
	// struct. See the SoA comment below for why.
	struct alignas(16) RayHWSS final {
		float3             m_Origin;
		float3             m_Direction;
		float4             m_Wavelengths;  // nm; .x = hero, .yzw = offsets (rotation of hero)
		float4             m_Throughput;   // per-lane path throughput
		float4             m_Pdf;          // per-lane wavelength-sampling pdf, for HWSS MIS weighting
		unsigned long long m_RngState;
		unsigned int       pixelId;
		unsigned char      m_BounceCount;
		unsigned char      flags;
		float              m_IorCurr;
		unsigned char      m_MediumIdx;   // 1-indexed (0 = vacuum); max 255 distinct media
		unsigned char      m_Pad[2];
		// Solid-angle pdf of the last BSDF-sampled scatter direction (wavelength-
		// independent — direction sampling isn't per-lane). Used to MIS-weight a
		// BSDF-sampled ray that happens to land on a light against NEE. Meaningless
		// when RAY_FLAG_DELTA is set (no competing NEE strategy for that bounce).
		float              m_BsdfPdf = 1.f;
	};

	// Struct-of-arrays wavefront ray storage. TraversalKernelWavefront only ever reads
	// origin/direction/flags (~28 of RayHWSS's ~96 bytes) — as an AoS array it would still
	// pay for a full RayHWSS load per ray just to use a quarter of it. Splitting into a
	// "core" (origin/direction/flags — read by both Traversal and Shade) and "ext"
	// (everything else — read/written only by Shade) array pair lets Traversal's loads
	// carry only the bytes it actually touches.
	//
	// LoadRay/StoreRay are the only places that know the mapping between individual SoA
	// arrays and the logical RayHWSS fields — every kernel body still operates on a plain
	// local RayHWSS exactly as before, so converting a kernel to SoA storage only changes
	// its prologue (LoadRay) and each exit point (StoreRay), never the logic in between.
	struct RayCoreSoA {
		float3*        origin    = nullptr;
		float3*        direction = nullptr;
		unsigned char* flags     = nullptr;
	};

	struct RayExtSoA {
		float4*             wavelengths = nullptr;
		float4*             throughput  = nullptr;
		float4*             pdf         = nullptr;
		unsigned long long* rngState    = nullptr;
		unsigned int*       pixelId     = nullptr;
		unsigned char*      bounceCount = nullptr;
		float*              iorCurr     = nullptr;
		unsigned char*      mediumIdx   = nullptr;
		float*              bsdfPdf     = nullptr;
	};

	__device__ __forceinline__ RayHWSS LoadRay(const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i) {
		RayHWSS ray{};
		ray.m_Origin      = core.origin[i];
		ray.m_Direction   = core.direction[i];
		ray.flags         = core.flags[i];
		ray.m_Wavelengths = ext.wavelengths[i];
		ray.m_Throughput  = ext.throughput[i];
		ray.m_Pdf         = ext.pdf[i];
		ray.m_RngState    = ext.rngState[i];
		ray.pixelId       = ext.pixelId[i];
		ray.m_BounceCount = ext.bounceCount[i];
		ray.m_IorCurr     = ext.iorCurr[i];
		ray.m_MediumIdx   = ext.mediumIdx[i];
		ray.m_BsdfPdf     = ext.bsdfPdf[i];
		return ray;
	}

	__device__ __forceinline__ void StoreRay(const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i, const RayHWSS& ray) {
		core.origin[i]    = ray.m_Origin;
		core.direction[i] = ray.m_Direction;
		core.flags[i]     = ray.flags;
		ext.wavelengths[i]= ray.m_Wavelengths;
		ext.throughput[i] = ray.m_Throughput;
		ext.pdf[i]        = ray.m_Pdf;
		ext.rngState[i]   = ray.m_RngState;
		ext.pixelId[i]    = ray.pixelId;
		ext.bounceCount[i]= ray.m_BounceCount;
		ext.iorCurr[i]    = ray.m_IorCurr;
		ext.mediumIdx[i]  = ray.m_MediumIdx;
		ext.bsdfPdf[i]    = ray.m_BsdfPdf;
	}
}
