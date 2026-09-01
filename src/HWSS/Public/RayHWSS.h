#pragma once
#include <vector_types.h>
#include <cstdint>

namespace Vera::Spectral::HWSS {
	static constexpr unsigned char RAY_FLAG_DEAD = 0x01;
	static constexpr unsigned char RAY_FLAG_DELTA = 0x02;
	// Set on a dispersive event: the offset lanes stop being valid samples of the same
	// path and are zeroed, leaving only the hero lane (Wilkie et al.).
	static constexpr unsigned char RAY_FLAG_DISPERSED = 0x04;
	// 0x08 reserved: fluorescence state is per-lane, see RayHWSS::m_LaneFluoresced.

	static constexpr int HWSS_LANES = 4; // hero + 3 offset wavelengths per path

	// Logical/local ray, built in registers via LoadRay and written back via StoreRay.
	// Actual wavefront storage is the struct-of-arrays pair below, not an array of this.
	struct alignas(16) RayHWSS final {
		float3             m_Origin;
		float3             m_Direction;
		float4             m_Wavelengths;  // nm; .x = hero, .yzw = offsets. TRACE wavelengths the path
		// currently gathers radiance at; mutated by fluorescent shifts.
		float4             m_SensorWavelengths; // nm; the wavelengths that set each lane's CIE weight.
		// Equal to m_Wavelengths at ray-gen; diverges once a
		// fluorescent event shifts a trace wavelength.
		float4             m_Throughput;   // per-lane path throughput
		float4             m_Pdf;          // per-lane wavelength-sampling pdf, for HWSS MIS weighting
		unsigned long long m_RngState;
		unsigned int       pixelId;
		unsigned char      m_BounceCount;
		unsigned char      flags;
		float              m_IorCurr;
		unsigned char      m_MediumIdx;   // 1-indexed (0 = vacuum); max 255 distinct media
		// Per-lane bitmask (bit k => lane k) marking lanes shifted by a fluorescent event
		// at the last bounce; MIS-weights env-map hits after a shift. Cleared by the next
		// elastic bounce.
		unsigned char      m_LaneFluoresced = 0;
		unsigned char      m_Pad[1];
		// Solid-angle pdf of the last BSDF-sampled direction (not per-lane); MIS-weights a
		// BSDF-sampled ray that lands on a light. Meaningless when RAY_FLAG_DELTA is set.
		float              m_BsdfPdf = 1.f;
		unsigned int       m_SampleIdx; // which spp pass; part of the QMC point index (see QMCHWSS.h)
	};

	// Struct-of-arrays wavefront storage: Traversal only reads origin/direction/flags, so
	// splitting those into a "core" array from the shade-only "ext" fields keeps its loads
	// small. LoadRay/StoreRay are the only places that map SoA arrays to the logical struct.
	struct RayCoreSoA {
		float3* origin = nullptr;
		float3* direction = nullptr;
		unsigned char* flags = nullptr;
	};

	struct RayExtSoA {
		float4* wavelengths = nullptr;
		float4* sensorWavelengths = nullptr;
		float4* throughput = nullptr;
		float4* pdf = nullptr;
		unsigned long long* rngState = nullptr;
		unsigned int* pixelId = nullptr;
		unsigned char* bounceCount = nullptr;
		float* iorCurr = nullptr;
		unsigned char* mediumIdx = nullptr;
		unsigned char* laneFluoresced = nullptr;
		float* bsdfPdf = nullptr;
		unsigned int* sampleIdx = nullptr;
	};

	__device__ __forceinline__ RayHWSS LoadRay(const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i) {
		RayHWSS ray{};
		ray.m_Origin = core.origin[i];
		ray.m_Direction = core.direction[i];
		ray.flags = core.flags[i];
		ray.m_Wavelengths = ext.wavelengths[i];
		ray.m_SensorWavelengths = ext.sensorWavelengths[i];
		ray.m_Throughput = ext.throughput[i];
		ray.m_Pdf = ext.pdf[i];
		ray.m_RngState = ext.rngState[i];
		ray.pixelId = ext.pixelId[i];
		ray.m_BounceCount = ext.bounceCount[i];
		ray.m_IorCurr = ext.iorCurr[i];
		ray.m_MediumIdx = ext.mediumIdx[i];
		ray.m_LaneFluoresced = ext.laneFluoresced[i];
		ray.m_BsdfPdf = ext.bsdfPdf[i];
		ray.m_SampleIdx = ext.sampleIdx[i];
		return ray;
	}

	__device__ __forceinline__ void StoreRay(const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i, const RayHWSS& ray) {
		core.origin[i] = ray.m_Origin;
		core.direction[i] = ray.m_Direction;
		core.flags[i] = ray.flags;
		ext.wavelengths[i] = ray.m_Wavelengths;
		ext.sensorWavelengths[i] = ray.m_SensorWavelengths;
		ext.throughput[i] = ray.m_Throughput;
		ext.pdf[i] = ray.m_Pdf;
		ext.rngState[i] = ray.m_RngState;
		ext.pixelId[i] = ray.pixelId;
		ext.bounceCount[i] = ray.m_BounceCount;
		ext.iorCurr[i] = ray.m_IorCurr;
		ext.mediumIdx[i] = ray.m_MediumIdx;
		ext.laneFluoresced[i] = ray.m_LaneFluoresced;
		ext.bsdfPdf[i] = ray.m_BsdfPdf;
		ext.sampleIdx[i] = ray.m_SampleIdx;
	}
}
