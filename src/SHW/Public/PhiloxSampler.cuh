#pragma once
#include <Philox.cuh>

namespace Vera::SHW {
	struct PhiloxSampler {
		Core::PhiloxKey baseKey;
		uint32_t drawIdx = 0;

		__device__ __forceinline__ float4 draw4() {
			Core::PhiloxKey k = baseKey;
			k.m_NeighborIdx = drawIdx++;
			return Core::Philox4x32Default::fromKey(k).nextFloat4();
		}

		__device__ __forceinline__ float nextFloat() { return draw4().x; }
		__device__ __forceinline__ float2 nextFloat2() { auto d = draw4(); return make_float2(d.x, d.y); }
	};
}
