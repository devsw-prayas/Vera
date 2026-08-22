#include "MaterialSortHWSS.cuh"
#include <cub/device/device_radix_sort.cuh>

namespace Vera::Spectral::HWSS {

	__global__ void ExtractMaterialKeysKernel(
		Core::GeometryBuffers geom,
		Core::WavefrontHitRecord* hits,
		uint32_t* keysOut,
		uint32_t* indicesOut,
		uint32_t count)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= count) return;
		indicesOut[idx] = idx;
		keysOut[idx] = hits[idx].m_Hit ? (uint32_t)geom.m_TriMatID[hits[idx].m_PrimIdx] : 0xFFFFu;
	}

	void MaterialSorter::Init(uint32_t inMaxRayCount) {
		maxRayCount = inMaxRayCount;
		cudaMalloc(&d_keysA,    maxRayCount * sizeof(uint32_t));
		cudaMalloc(&d_keysB,    maxRayCount * sizeof(uint32_t));
		cudaMalloc(&d_indicesA, maxRayCount * sizeof(uint32_t));
		cudaMalloc(&d_indicesB, maxRayCount * sizeof(uint32_t));

		// Sizing pass for temporary storage
		tempStorageBytes = 0;
		cub::DeviceRadixSort::SortPairs(
			nullptr,
			tempStorageBytes,
			static_cast<uint32_t*>(nullptr),
			static_cast<uint32_t*>(nullptr),
			static_cast<uint32_t*>(nullptr),
			static_cast<uint32_t*>(nullptr),
			maxRayCount,
			0, sizeof(uint32_t) * 8
		);

		if (tempStorageBytes > 0) {
			cudaMalloc(&d_tempStorage, tempStorageBytes);
		}
	}

	void MaterialSorter::Destroy() {
		if (d_tempStorage) { cudaFree(d_tempStorage); d_tempStorage = nullptr; }
		if (d_keysA)       { cudaFree(d_keysA);       d_keysA       = nullptr; }
		if (d_keysB)       { cudaFree(d_keysB);       d_keysB       = nullptr; }
		if (d_indicesA)    { cudaFree(d_indicesA);    d_indicesA    = nullptr; }
		if (d_indicesB)    { cudaFree(d_indicesB);    d_indicesB    = nullptr; }
		tempStorageBytes = 0;
		maxRayCount = 0;
	}

	uint32_t* MaterialSorter::Sort(Core::GeometryBuffers geom, Core::WavefrontHitRecord* hits, uint32_t count, cudaStream_t stream) {
		if (count == 0) {
			return d_indicesB;
		}

		dim3 block(256);
		dim3 grid((count + block.x - 1) / block.x);
		ExtractMaterialKeysKernel<<<grid, block, 0, stream>>>(geom, hits, d_keysA, d_indicesA, count);

		cub::DeviceRadixSort::SortPairs(
			d_tempStorage,
			tempStorageBytes,
			d_keysA,
			d_keysB,
			d_indicesA,
			d_indicesB,
			count,
			0, sizeof(uint32_t) * 8,
			stream
		);

		return d_indicesB;
	}
}
