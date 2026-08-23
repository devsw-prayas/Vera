#include "RendererHWSS.cuh"
#include "CameraHWSS.cuh"
#include "ShadeHWSS.cuh"
#include "SpectralConstants.h"
#include "CIE.h"
#include <Traversal.cuh>
#include "RayCompactionHWSS.cuh"
#include "MaterialSortHWSS.cuh"

namespace Vera::Spectral::HWSS {
	void RenderHWSS(
		const Core::GeometryBuffers& geom,
		const Material* d_materials,
		const MediumHWSS* d_media,
		const LightBVH& lightBvh,
		const EnvMapHWSS& envMap,
		const Core::Camera& camera,
		FrameBufferHWSS& fb,
		float4* d_outRGB,
		uint32_t samplesPerPixel,
		uint32_t maxBounces,
		unsigned char defaultMediumIdx,
		Core::ToneMapper tonemapper,
		float exposure)
	{
		uint32_t rayCount = camera.m_Width * camera.m_Height;

		RayCoreSoA coreA{}, coreB{};
		RayExtSoA  extA{},  extB{};
		cudaMalloc(&coreA.origin,      rayCount * sizeof(float3));
		cudaMalloc(&coreA.direction,   rayCount * sizeof(float3));
		cudaMalloc(&coreA.flags,       rayCount * sizeof(unsigned char));
		cudaMalloc(&coreB.origin,      rayCount * sizeof(float3));
		cudaMalloc(&coreB.direction,   rayCount * sizeof(float3));
		cudaMalloc(&coreB.flags,       rayCount * sizeof(unsigned char));
		cudaMalloc(&extA.wavelengths,  rayCount * sizeof(float4));
		cudaMalloc(&extA.throughput,   rayCount * sizeof(float4));
		cudaMalloc(&extA.pdf,          rayCount * sizeof(float4));
		cudaMalloc(&extA.rngState,     rayCount * sizeof(unsigned long long));
		cudaMalloc(&extA.pixelId,      rayCount * sizeof(unsigned int));
		cudaMalloc(&extA.bounceCount,  rayCount * sizeof(unsigned char));
		cudaMalloc(&extA.iorCurr,      rayCount * sizeof(float));
		cudaMalloc(&extA.mediumIdx,    rayCount * sizeof(unsigned char));
		cudaMalloc(&extA.bsdfPdf,      rayCount * sizeof(float));
		cudaMalloc(&extB.wavelengths,  rayCount * sizeof(float4));
		cudaMalloc(&extB.throughput,   rayCount * sizeof(float4));
		cudaMalloc(&extB.pdf,          rayCount * sizeof(float4));
		cudaMalloc(&extB.rngState,     rayCount * sizeof(unsigned long long));
		cudaMalloc(&extB.pixelId,      rayCount * sizeof(unsigned int));
		cudaMalloc(&extB.bounceCount,  rayCount * sizeof(unsigned char));
		cudaMalloc(&extB.iorCurr,      rayCount * sizeof(float));
		cudaMalloc(&extB.mediumIdx,    rayCount * sizeof(unsigned char));
		cudaMalloc(&extB.bsdfPdf,      rayCount * sizeof(float));

		Core::WavefrontHitRecord* d_hits = nullptr;
		cudaMalloc(&d_hits,  rayCount * sizeof(Core::WavefrontHitRecord));

		dim3 block2D(16, 16);
		dim3 grid2D((camera.m_Width + block2D.x - 1) / block2D.x, (camera.m_Height + block2D.y - 1) / block2D.y);
		dim3 block1D(256);

		RayCompactor compactor;
		compactor.Init(rayCount);

		MaterialSorter sorter;
		sorter.Init(rayCount);

		CIETexture cieTex = MakeCIETexture(LAMBDA_MIN, LAMBDA_MAX);

		for (uint32_t sampleIdx = 0; sampleIdx < samplesPerPixel; ++sampleIdx) {
			GeneratePrimaryRaysHWSSKernel<<<grid2D, block2D>>>(camera, coreA, extA, sampleIdx, defaultMediumIdx);

			uint32_t activeCount = rayCount;
			for (uint32_t bounce = 0; bounce < maxBounces && activeCount > 0; ++bounce) {
				dim3 grid1D((activeCount + block1D.x - 1) / block1D.x);
				Core::TraversalKernelWavefront<<<grid1D, block1D>>>(geom, coreA, d_hits, activeCount);
				uint32_t* order = sorter.Sort(geom, d_hits, activeCount, /*stream*/0);
				ShadeKernelHWSSWavefront<<<grid1D, block1D>>>(
					geom, coreA, extA, d_hits, order, d_materials, d_media, lightBvh,
					coreB, extB, activeCount, fb, envMap, maxBounces, cieTex.tex);
				activeCount = compactor.Compact(coreB, extB, activeCount, coreA, extA, /*stream*/0);
			}
		}

		FreeCIETexture(cieTex);
		sorter.Destroy();
		compactor.Destroy();

		cudaFree(coreA.origin);     cudaFree(coreA.direction);   cudaFree(coreA.flags);
		cudaFree(coreB.origin);     cudaFree(coreB.direction);   cudaFree(coreB.flags);
		cudaFree(extA.wavelengths); cudaFree(extA.throughput);   cudaFree(extA.pdf);
		cudaFree(extA.rngState);    cudaFree(extA.pixelId);      cudaFree(extA.bounceCount);
		cudaFree(extA.iorCurr);     cudaFree(extA.mediumIdx);    cudaFree(extA.bsdfPdf);
		cudaFree(extB.wavelengths); cudaFree(extB.throughput);   cudaFree(extB.pdf);
		cudaFree(extB.rngState);    cudaFree(extB.pixelId);      cudaFree(extB.bounceCount);
		cudaFree(extB.iorCurr);     cudaFree(extB.mediumIdx);    cudaFree(extB.bsdfPdf);
		cudaFree(d_hits);

		float cieYIntegral = CIE_Y_Integral(LAMBDA_MIN, LAMBDA_MAX);
		dim3 grid1D((rayCount + block1D.x - 1) / block1D.x);
		ResolveFrameBufferKernel<<<grid1D, block1D>>>(fb, samplesPerPixel, cieYIntegral, tonemapper, exposure, d_outRGB);
	}
}
