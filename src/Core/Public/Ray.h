#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

namespace BSPT::Spectral::HWSS::Core {
	struct Ray final {
		float3 m_Origin;
		float3 m_Direction;
		float m_Tmin = 1e-4f;
		float m_Tmax = FLT_MAX;
	};
}
