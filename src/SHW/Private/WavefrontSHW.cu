#include "WavefrontSHW.cuh"
#include "CameraSHW.cuh"
#include <Traversal.cuh>
#include <MaterialSortHWSS.cuh>
#ifdef VERA_ENABLE_OPTIX
#include <OptixTraversal.h>
#endif

namespace Vera::SHW {
	SHWBuffers AllocSHWBuffers(uint32_t maxRays) {
		SHWBuffers buf{};
		buf.maxRays = maxRays;

		cudaMalloc(&buf.coreA.origin, maxRays * sizeof(float3));
		cudaMalloc(&buf.coreA.direction, maxRays * sizeof(float3));
		cudaMalloc(&buf.coreA.flags, maxRays * sizeof(unsigned char));
		cudaMalloc(&buf.coreB.origin, maxRays * sizeof(float3));
		cudaMalloc(&buf.coreB.direction, maxRays * sizeof(float3));
		cudaMalloc(&buf.coreB.flags, maxRays * sizeof(unsigned char));

		cudaMalloc(&buf.extA.lambda, maxRays * sizeof(float));
		cudaMalloc(&buf.extA.sensorLambda, maxRays * sizeof(float));
		cudaMalloc(&buf.extA.throughput, maxRays * sizeof(float));
		cudaMalloc(&buf.extA.pdf, maxRays * sizeof(float));
		cudaMalloc(&buf.extA.pixelId, maxRays * sizeof(uint32_t));
		cudaMalloc(&buf.extA.sampleIdx, maxRays * sizeof(uint32_t));
		cudaMalloc(&buf.extA.bounceCount, maxRays * sizeof(uint8_t));
		cudaMalloc(&buf.extA.iorCurr, maxRays * sizeof(float));
		cudaMalloc(&buf.extA.mediumIdx, maxRays * sizeof(uint8_t));
		cudaMalloc(&buf.extA.bsdfPdf, maxRays * sizeof(float));

		cudaMalloc(&buf.extB.lambda, maxRays * sizeof(float));
		cudaMalloc(&buf.extB.sensorLambda, maxRays * sizeof(float));
		cudaMalloc(&buf.extB.throughput, maxRays * sizeof(float));
		cudaMalloc(&buf.extB.pdf, maxRays * sizeof(float));
		cudaMalloc(&buf.extB.pixelId, maxRays * sizeof(uint32_t));
		cudaMalloc(&buf.extB.sampleIdx, maxRays * sizeof(uint32_t));
		cudaMalloc(&buf.extB.bounceCount, maxRays * sizeof(uint8_t));
		cudaMalloc(&buf.extB.iorCurr, maxRays * sizeof(float));
		cudaMalloc(&buf.extB.mediumIdx, maxRays * sizeof(uint8_t));
		cudaMalloc(&buf.extB.bsdfPdf, maxRays * sizeof(float));

		cudaMalloc(&buf.d_hits, maxRays * sizeof(Core::WavefrontHitRecord));

		buf.compactor.Init(maxRays);

		return buf;
	}

	void FreeSHWBuffers(SHWBuffers& buf) {
		cudaFree(buf.coreA.origin);   cudaFree(buf.coreA.direction);   cudaFree(buf.coreA.flags);
		cudaFree(buf.coreB.origin);   cudaFree(buf.coreB.direction);   cudaFree(buf.coreB.flags);

		cudaFree(buf.extA.lambda);       cudaFree(buf.extA.sensorLambda); cudaFree(buf.extA.throughput);
		cudaFree(buf.extA.pdf);          cudaFree(buf.extA.pixelId);      cudaFree(buf.extA.sampleIdx);
		cudaFree(buf.extA.bounceCount);  cudaFree(buf.extA.iorCurr);      cudaFree(buf.extA.mediumIdx);
		cudaFree(buf.extA.bsdfPdf);

		cudaFree(buf.extB.lambda);       cudaFree(buf.extB.sensorLambda); cudaFree(buf.extB.throughput);
		cudaFree(buf.extB.pdf);          cudaFree(buf.extB.pixelId);      cudaFree(buf.extB.sampleIdx);
		cudaFree(buf.extB.bounceCount);  cudaFree(buf.extB.iorCurr);      cudaFree(buf.extB.mediumIdx);
		cudaFree(buf.extB.bsdfPdf);

		cudaFree(buf.d_hits);

		buf.compactor.Destroy();
		buf = SHWBuffers{};
	}

	void GenerateSHW(
		const Core::Camera& camera,
		RayCoreSoA core, RayExtSoA ext,
		uint32_t sampleIdx, uint8_t defaultMediumIdx) {
		dim3 block2D(16, 16);
		dim3 grid2D((camera.m_Width + block2D.x - 1) / block2D.x, (camera.m_Height + block2D.y - 1) / block2D.y);
		GeneratePrimaryRaysSHWKernel<<<grid2D, block2D>>>(camera, core, ext, sampleIdx, defaultMediumIdx);
	}

	void TraverseSHW(
		const Core::GeometryBuffers& geom,
		RayCoreSoA core,
		Core::WavefrontHitRecord* hits,
		uint32_t rayCount,
		bool useOptix
#ifdef VERA_ENABLE_OPTIX
		, Core::OptixTraversalContext* optixCtx
#endif
	) {
#ifdef VERA_ENABLE_OPTIX
		if (useOptix && optixCtx) {
			Core::LaunchOptixTraversal(*optixCtx, core.origin, core.direction, core.flags, hits, rayCount, /*stream*/0);
			return;
		}
#else
		(void)useOptix;
#endif
		dim3 block1D(256);
		dim3 grid1D((rayCount + block1D.x - 1) / block1D.x);
		Core::TraversalKernelWavefront<<<grid1D, block1D>>>(geom, core.origin, core.direction, core.flags, hits, rayCount);
	}

	uint32_t* SortSHW(
		Core::MaterialSorter& sorter,
		Core::GeometryBuffers geom,
		Core::WavefrontHitRecord* hits,
		uint32_t count) {
		return sorter.Sort(geom, hits, count, /*stream*/0);
	}

	uint32_t CompactSHW(
		RayCompactorSHW& compactor,
		RayCoreSoA coreIn, RayExtSoA extIn, uint32_t count,
		RayCoreSoA coreOut, RayExtSoA extOut) {
		return compactor.Compact(coreIn, extIn, count, coreOut, extOut, /*stream*/0);
	}
}
