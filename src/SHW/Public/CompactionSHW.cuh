#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "RaySHW.h"

namespace Vera::SHW {
	struct RayCompactorSHW {
		void*     d_tempStorage    = nullptr;
		size_t    tempStorageBytes = 0;
		uint32_t* d_numSelected    = nullptr; // device-side count, written by cub, read back to host each call
		uint32_t* d_order          = nullptr; // scratch: compacted (surviving) source indices, densely packed
		uint32_t  maxRayCount      = 0;       // upper bound this was sized for

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Compacts the first `count` (coreIn, extIn) rays into (coreOut, extOut), dropping
		// SHW_FLAG_DEAD. Same two-pass select-then-gather scheme as RayCompactor (HWSS):
		// survivor order isn't preserved. Returns the new live count via a sync readback.
		uint32_t Compact(
			RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
			RayCoreSoA coreOut, RayExtSoA extOut, cudaStream_t stream = 0);
	};
}
