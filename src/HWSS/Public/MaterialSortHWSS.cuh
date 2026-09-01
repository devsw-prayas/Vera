#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <CoreUtils.h>

namespace Vera::Spectral::HWSS {
	struct MaterialSorter {
		void* d_tempStorage = nullptr;
		size_t    tempStorageBytes = 0;
		uint32_t* d_keysA = nullptr; // material id per active ray (scratch)
		uint32_t* d_keysB = nullptr; // cub radix sort needs distinct in/out key buffers
		uint32_t* d_indicesA = nullptr; // 0..count-1 initially
		uint32_t* d_indicesB = nullptr; // sorted permutation lands here
		uint32_t  maxRayCount = 0;

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Returns a device permutation `order` of length `count` such that hits[order[i]]
		// visits rays grouped by material id (0xFFFFu for misses). Radix-sorts (key, index)
		// pairs only - does not move any ray/hit data.
		uint32_t* Sort(Core::GeometryBuffers geom, Core::WavefrontHitRecord* hits, uint32_t count, cudaStream_t stream = 0);
	};
}