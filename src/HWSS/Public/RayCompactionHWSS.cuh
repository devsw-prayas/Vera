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
		uint32_t  maxRayCount       = 0;       // upper bound this was sized for

		void Init(uint32_t maxRayCount);
		void Destroy();

		// Compacts the first `count` entries of d_in (a RayHWSS* array where some entries
		// are flagged RAY_FLAG_DEAD) into d_out, keeping only the ones that do NOT have
		// RAY_FLAG_DEAD set. Order of surviving rays does not need to be preserved (each
		// RayHWSS carries its own pixelId, so downstream framebuffer writes are unaffected
		// by reordering). Returns the new live ray count via a synchronous device->host
		// readback of *d_numSelected after the cub call completes.
		uint32_t Compact(RayHWSS* d_in, RayHWSS* d_out, uint32_t count, cudaStream_t stream = 0);
	};
}
