#pragma once
#include <vector_types.h>

namespace BSPT::Spectral::HWSS {
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
	// pixelId(4) MaterialId(2) BounceCount(1) flags(1) IorCurr(4) MediumIdx(1) BsdfPdf(4) Pad(3).
	struct alignas(16) RayHWSS final {
		float3             m_Origin;
		float3             m_Direction;
		float4             m_Wavelengths;  // nm; .x = hero, .yzw = offsets (rotation of hero)
		float4             m_Throughput;   // per-lane path throughput
		float4             m_Pdf;          // per-lane wavelength-sampling pdf, for HWSS MIS weighting
		unsigned long long m_RngState;
		unsigned int       pixelId;
		unsigned short     m_MaterialId;
		unsigned char      m_BounceCount;
		unsigned char      flags;
		float              m_IorCurr;
		unsigned char      m_MediumIdx;   // 1-indexed (0 = vacuum); max 255 distinct media
		unsigned char      m_Pad[3];
		// Solid-angle pdf of the last BSDF-sampled scatter direction (wavelength-
		// independent — direction sampling isn't per-lane). Used to MIS-weight a
		// BSDF-sampled ray that happens to land on a light against NEE. Meaningless
		// when RAY_FLAG_DELTA is set (no competing NEE strategy for that bounce).
		float              m_BsdfPdf = 1.f;
	};
}
