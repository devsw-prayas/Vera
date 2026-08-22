#pragma once
#include <cuda_runtime.h>
#include "Ray.h"

namespace BSPT::Spectral::HWSS::Core {

	struct Camera final {
		float3 m_Origin;
		float3 m_Forward;
		float3 m_Right;
		float3 m_Up;

		float m_HalfTanFovY;
		float m_AspectRatio;
		float m_LensRadius;  // 0 = pinhole
		float m_FocusDist;   // distance along forward to focus plane
		unsigned int m_Width;
		unsigned int m_Height;
	};
}
