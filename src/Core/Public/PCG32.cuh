#pragma once
#include <cuda_runtime.h>

namespace Vera::Core {
	struct PCG32 final {
		unsigned long long m_State;

		__device__ void seed(unsigned long long s) {
			m_State = s + 1442695040888963407ULL;
			m_State = m_State * 6364136223846793005ULL + 1442695040888963407ULL;
		}

		__device__ unsigned int nextInt() {
			m_State = m_State * 6364136223846793005ULL + 1442695040888963407ULL;
			unsigned int xorShift = ((m_State >> 18u) ^ m_State) >> 27u;
			unsigned int rot = m_State >> 59u;
			return (xorShift >> rot) | (xorShift << ((-rot) & 31));
		}

		__device__ float nextFloat() {
			return nextInt() * (1.f / 4294967296.0f);
		}

		__device__ float2 nextFloat2() {
			return make_float2(nextFloat(), nextFloat());
		}
	};
}
