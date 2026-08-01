#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

namespace BSPT::Spectral::HWSS::Core {
	struct HitRecord final {
		float m_T;
		float3 m_Normal;
		uint32_t m_PrimIdx;
		bool m_Hit;
	};
}
