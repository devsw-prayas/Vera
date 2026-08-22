#include "RayCompactionHWSS.cuh"
#include <cub/device/device_select.cuh>

namespace Vera::Spectral::HWSS {
	struct AliveOp {
		__device__ bool operator()(const RayHWSS& r) const {
			return !(r.flags & RAY_FLAG_DEAD);
		}
	};

	void RayCompactor::Init(uint32_t inMaxRayCount) {
		maxRayCount = inMaxRayCount;
		cudaMalloc(&d_numSelected, sizeof(uint32_t));

		// Sizing pass for temporary storage
		tempStorageBytes = 0;
		cub::DeviceSelect::If(
			nullptr,
			tempStorageBytes,
			static_cast<RayHWSS*>(nullptr),
			static_cast<RayHWSS*>(nullptr),
			d_numSelected,
			maxRayCount,
			AliveOp{}
		);

		if (tempStorageBytes > 0) {
			cudaMalloc(&d_tempStorage, tempStorageBytes);
		}
	}

	void RayCompactor::Destroy() {
		if (d_tempStorage) {
			cudaFree(d_tempStorage);
			d_tempStorage = nullptr;
		}
		if (d_numSelected) {
			cudaFree(d_numSelected);
			d_numSelected = nullptr;
		}
		tempStorageBytes = 0;
		maxRayCount = 0;
	}

	uint32_t RayCompactor::Compact(RayHWSS* d_in, RayHWSS* d_out, uint32_t count, cudaStream_t stream) {
		if (count == 0) {
			return 0;
		}

		cub::DeviceSelect::If(
			d_tempStorage,
			tempStorageBytes,
			d_in,
			d_out,
			d_numSelected,
			count,
			AliveOp{},
			stream
		);

		uint32_t hostCount = 0;
		cudaMemcpyAsync(&hostCount, d_numSelected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		return hostCount;
	}
}
