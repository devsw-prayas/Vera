#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "RayHWSS.h"

namespace Vera::Spectral::HWSS {
	struct RayCompactor {
		void* d_tempStorage = nullptr;
		size_t    tempStorageBytes = 0;
		uint32_t* d_numSelected = nullptr; // device-side count, written by cub, read back to host each call
		uint32_t* d_order = nullptr; // scratch: compacted (surviving) source indices, densely packed
		uint32_t  maxRayCount = 0;       // upper bound this was sized for

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Compacts the first `count` (coreIn, extIn) rays into (coreOut, extOut), dropping
		// RAY_FLAG_DEAD. Compacts indices once (cub::DeviceSelect::Flagged keyed by flags)
		// then gathers, so the scan cost is independent of the SoA array count. Survivor
		// order isn't preserved (each ray carries its pixelId). Returns the new live count
		// via a synchronous readback.
		uint32_t Compact(
			RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
			RayCoreSoA coreOut, RayExtSoA extOut, cudaStream_t stream = 0);
	};
}