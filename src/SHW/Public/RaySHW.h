#pragma once
#include <vector_types.h>
#include <cstdint>
#include <CoreUtils.h>

namespace Vera::SHW {
	static constexpr unsigned char SHW_FLAG_DEAD  = Core::RAY_FLAG_DEAD;
	static constexpr unsigned char SHW_FLAG_DELTA = 0x02;
	static constexpr unsigned char SHW_FLAG_FLUORESCED = 0x04;

	struct RaySHW final {
		float3   m_Origin;
		float3   m_Direction;
		float    m_Lambda;         // hero wavelength, nm
		float    m_SensorLambda;   // CIE weight wavelength (diverges after fluorescence)
		float    m_Throughput;     // scalar path throughput
		float    m_Pdf;            // wavelength-sampling pdf
		uint32_t m_PixelId;
		uint32_t m_SampleIdx;
		uint8_t  m_BounceCount;
		uint8_t  m_Flags;
		float    m_IorCurr;
		uint8_t  m_MediumIdx;     // 1-indexed (0 = vacuum)
		float    m_BsdfPdf;       // solid-angle pdf of last BSDF sample (for MIS)
	};

	// SoA layout: core (traversal-hot) vs ext (shade-only), same split as HWSS
	struct RayCoreSoA {
		float3*        origin    = nullptr;
		float3*        direction = nullptr;
		unsigned char* flags     = nullptr;
	};

	struct RayExtSoA {
		float*    lambda       = nullptr;
		float*    sensorLambda = nullptr;
		float*    throughput   = nullptr;
		float*    pdf          = nullptr;
		uint32_t* pixelId      = nullptr;
		uint32_t* sampleIdx    = nullptr;
		uint8_t*  bounceCount  = nullptr;
		float*    iorCurr      = nullptr;
		uint8_t*  mediumIdx    = nullptr;
		float*    bsdfPdf      = nullptr;
	};

	__device__ __forceinline__ RaySHW LoadRay(
		const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i) {
		RaySHW ray{};
		ray.m_Origin      = core.origin[i];
		ray.m_Direction   = core.direction[i];
		ray.m_Flags       = core.flags[i];
		ray.m_Lambda      = ext.lambda[i];
		ray.m_SensorLambda= ext.sensorLambda[i];
		ray.m_Throughput  = ext.throughput[i];
		ray.m_Pdf         = ext.pdf[i];
		ray.m_PixelId     = ext.pixelId[i];
		ray.m_SampleIdx   = ext.sampleIdx[i];
		ray.m_BounceCount = ext.bounceCount[i];
		ray.m_IorCurr     = ext.iorCurr[i];
		ray.m_MediumIdx   = ext.mediumIdx[i];
		ray.m_BsdfPdf     = ext.bsdfPdf[i];
		return ray;
	}

	__device__ __forceinline__ void StoreRay(
		const RayCoreSoA& core, const RayExtSoA& ext, uint32_t i, const RaySHW& ray) {
		core.origin[i]    = ray.m_Origin;
		core.direction[i] = ray.m_Direction;
		core.flags[i]     = ray.m_Flags;
		ext.lambda[i]       = ray.m_Lambda;
		ext.sensorLambda[i] = ray.m_SensorLambda;
		ext.throughput[i]   = ray.m_Throughput;
		ext.pdf[i]          = ray.m_Pdf;
		ext.pixelId[i]      = ray.m_PixelId;
		ext.sampleIdx[i]    = ray.m_SampleIdx;
		ext.bounceCount[i]  = ray.m_BounceCount;
		ext.iorCurr[i]      = ray.m_IorCurr;
		ext.mediumIdx[i]    = ray.m_MediumIdx;
		ext.bsdfPdf[i]      = ray.m_BsdfPdf;
	}
}
