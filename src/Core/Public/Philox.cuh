#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Philox4x32 (Random123) - counter-based RNG for SHW-ReSTIR. Pure function of (counter, key),
// no mutable running state like PCG32. Word layout: c0=pixel_id, c1=frame_idx, c2=sample_idx,
// c3=neighbor_idx (counter words, since only those get the nonlinear multiply-mixing each
// round - this is what prevents axis swap-collisions by construction), k0=stream_select,
// k1=reserved. Validated against src/philox_rng.py + tests/test_philox_rng.py in the
// Spectral-ReSTIR-DI companion repo; this header is a literal translation of that design.
namespace Vera::Core {
	struct PhiloxKey final {
		unsigned int m_PixelId;
		unsigned int m_FrameIdx;
		unsigned int m_SampleIdx;
		unsigned int m_NeighborIdx;
		unsigned int m_StreamSelect; // 0 = geometric, 1 = spectral
	};

	template <int NRounds = 7>
	struct Philox4x32 final {
		static_assert(NRounds == 7 || NRounds == 10, "only 7 or 10 rounds supported");

		__device__ __forceinline__ static Philox4x32 fromKey(const PhiloxKey& key) {
			Philox4x32 rng{};
			rng.m_Counter = make_uint4(key.m_PixelId, key.m_FrameIdx, key.m_SampleIdx, key.m_NeighborIdx);
			rng.m_Key = make_uint2(key.m_StreamSelect, 0u);
			return rng;
		}

		__device__ __forceinline__ uint4 next4() const {
			uint4 c = m_Counter;
			uint2 k = m_Key;

			#pragma unroll
			for (int round = 0; round < NRounds; ++round) {
				unsigned long long product0 = static_cast<unsigned long long>(kMul0) * c.x;
				unsigned long long product1 = static_cast<unsigned long long>(kMul1) * c.z;

				unsigned int hi0 = static_cast<unsigned int>(product0 >> 32);
				unsigned int lo0 = static_cast<unsigned int>(product0);
				unsigned int hi1 = static_cast<unsigned int>(product1 >> 32);
				unsigned int lo1 = static_cast<unsigned int>(product1);

				c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
				k.x += kWeyl0;
				k.y += kWeyl1;
			}

			return c;
		}

		__device__ __forceinline__ float nextFloat() const {
			return next4().x * (1.f / 4294967296.0f);
		}

		__device__ __forceinline__ float2 nextFloat2() const {
			uint4 out = next4();
			return make_float2(out.x * (1.f / 4294967296.0f), out.y * (1.f / 4294967296.0f));
		}

		__device__ __forceinline__ float4 nextFloat4() const {
			uint4 out = next4();
			const float scale = 1.f / 4294967296.0f;
			return make_float4(out.x * scale, out.y * scale, out.z * scale, out.w * scale);
		}

	private:
		static constexpr unsigned int kMul0 = 0xD2511F53u;
		static constexpr unsigned int kMul1 = 0xCD9E8D57u;
		static constexpr unsigned int kWeyl0 = 0x9E3779B9u;
		static constexpr unsigned int kWeyl1 = 0xBB67AE85u;

		uint4 m_Counter;
		uint2 m_Key;
	};

	using Philox4x32Default = Philox4x32<7>;
}
