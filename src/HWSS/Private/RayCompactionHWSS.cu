#include "RayCompactionHWSS.cuh"
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace Vera::Spectral::HWSS {
	struct AliveFromFlags {
		__device__ __forceinline__ bool operator()(unsigned char flags) const {
			return !(flags & RAY_FLAG_DEAD);
		}
	};

	using FlagIt = thrust::transform_iterator<AliveFromFlags, unsigned char*>;

	// Copies each surviving ray (by its pre-compaction index in `order`) from the
	// (coreIn, extIn) SoA buffers into the dense [0, count) range of (coreOut, extOut).
	__global__ void GatherRaysKernel(
		RayCoreSoA coreIn, RayExtSoA extIn,
		const uint32_t* __restrict__ order, uint32_t count,
		RayCoreSoA coreOut, RayExtSoA extOut)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= count) return;
		StoreRay(coreOut, extOut, idx, LoadRay(coreIn, extIn, order[idx]));
	}

	void RayCompactor::Init(uint32_t inMaxRayCount) {
		maxRayCount = inMaxRayCount;
		cudaMalloc(&d_numSelected, sizeof(uint32_t));
		cudaMalloc(&d_order, maxRayCount * sizeof(uint32_t));

		// Sizing pass for temporary storage
		tempStorageBytes = 0;
		thrust::counting_iterator<uint32_t> countIt(0);
		FlagIt flagIt(nullptr, AliveFromFlags{});
		cub::DeviceSelect::Flagged(
			nullptr,
			tempStorageBytes,
			countIt,
			flagIt,
			d_order,
			d_numSelected,
			maxRayCount
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
		if (d_order) {
			cudaFree(d_order);
			d_order = nullptr;
		}
		tempStorageBytes = 0;
		maxRayCount = 0;
	}

	uint32_t RayCompactor::Compact(
		RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
		RayCoreSoA coreOut, RayExtSoA extOut, cudaStream_t stream)
	{
		if (count == 0) {
			return 0;
		}

		thrust::counting_iterator<uint32_t> countIt(0);
		FlagIt flagIt(coreIn.flags, AliveFromFlags{});

		cub::DeviceSelect::Flagged(
			d_tempStorage,
			tempStorageBytes,
			countIt,
			flagIt,
			d_order,
			d_numSelected,
			count,
			stream
		);

		uint32_t hostCount = 0;
		cudaMemcpyAsync(&hostCount, d_numSelected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		if (hostCount > 0) {
			dim3 block(256);
			dim3 grid((hostCount + block.x - 1) / block.x);
			GatherRaysKernel<<<grid, block, 0, stream>>>(coreIn, extIn, d_order, hostCount, coreOut, extOut);
		}

		return hostCount;
	}
}
