#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "RayHWSS.h"

namespace Vera::Spectral::HWSS {
	struct RayCompactor {
		void*     d_tempStorage     = nullptr;
		size_t    tempStorageBytes  = 0;
		uint32_t* d_numSelected     = nullptr; // device-side count, written by cub, read back to host each call
		uint32_t* d_order           = nullptr; // scratch: compacted (surviving) source indices, densely packed
		uint32_t  maxRayCount       = 0;       // upper bound this was sized for

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Compacts the first `count` entries of the (coreIn, extIn) SoA ray buffers —
		// dropping entries flagged RAY_FLAG_DEAD — into (coreOut, extOut). Implemented as
		// index compaction (cub::DeviceSelect::Flagged over a counting iterator, keyed by
		// coreIn.flags) followed by a gather kernel, rather than compacting each SoA array
		// separately, so the selection/scan work only happens once per bounce regardless of
		// how many arrays make up a ray. Order of surviving rays does not need to be
		// preserved (each ray carries its own pixelId, so downstream framebuffer writes are
		// unaffected by reordering). Returns the new live ray count via a synchronous
		// device->host readback of *d_numSelected after the cub call completes.
		uint32_t Compact(
			RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
			RayCoreSoA coreOut, RayExtSoA extOut, cudaStream_t stream = 0);
	};
}
