#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <CoreUtils.h>

namespace Vera::Spectral::HWSS {
	struct MaterialSorter {
		void*     d_tempStorage    = nullptr;
		size_t    tempStorageBytes = 0;
		uint32_t* d_keysA          = nullptr; // material id per active ray (scratch)
		uint32_t* d_keysB          = nullptr; // cub radix sort needs distinct in/out key buffers
		uint32_t* d_indicesA       = nullptr; // 0..count-1 initially
		uint32_t* d_indicesB       = nullptr; // sorted permutation lands here
		uint32_t  maxRayCount      = 0;

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Reads hits[0..count), looks up each hit's material id (geom.m_TriMatID[hit.m_PrimIdx],
		// or sentinel 0xFFFFu if the hit missed), sorts a lightweight (key, index) pair array by
		// that key, and returns a device pointer to the resulting permutation `order` of length
		// `count` such that raysIn[order[i]] / hits[order[i]] visits rays grouped by material id.
		// Does NOT move/copy any RayHWSS or WavefrontHitRecord data itself.
		uint32_t* Sort(Core::GeometryBuffers geom, Core::WavefrontHitRecord* hits, uint32_t count, cudaStream_t stream = 0);
	};
}
