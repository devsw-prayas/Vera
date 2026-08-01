#include "RendererHWSS.cuh"
#include "CameraHWSS.cuh"
#include "ShadeHWSS.cuh"
#include "SpectralConstants.h"
#include "CIE.h"
#include <Traversal.cuh>

namespace BSPT::Spectral::HWSS {
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

		RayHWSS* d_raysA = nullptr;
		RayHWSS* d_raysB = nullptr;
		Core::WavefrontHitRecord* d_hits = nullptr;
		cudaMalloc(&d_raysA, rayCount * sizeof(RayHWSS));
		cudaMalloc(&d_raysB, rayCount * sizeof(RayHWSS));
		cudaMalloc(&d_hits,  rayCount * sizeof(Core::WavefrontHitRecord));

		dim3 block2D(16, 16);
		dim3 grid2D((camera.m_Width + block2D.x - 1) / block2D.x, (camera.m_Height + block2D.y - 1) / block2D.y);
		dim3 block1D(256);
		dim3 grid1D((rayCount + block1D.x - 1) / block1D.x);

		for (uint32_t sampleIdx = 0; sampleIdx < samplesPerPixel; ++sampleIdx) {
			GeneratePrimaryRaysHWSSKernel<<<grid2D, block2D>>>(camera, d_raysA, sampleIdx, defaultMediumIdx);

			for (uint32_t bounce = 0; bounce < maxBounces; ++bounce) {
				Core::TraversalKernelWavefront<<<grid1D, block1D>>>(geom, d_raysA, d_hits, rayCount);
				ShadeKernelHWSSWavefront<<<grid1D, block1D>>>(
					geom, d_raysA, d_hits, d_materials, d_media, lightBvh,
					d_raysB, rayCount, fb, envMap, maxBounces);
				RayHWSS* tmp = d_raysA; d_raysA = d_raysB; d_raysB = tmp;
			}
		}

		cudaFree(d_raysA);
		cudaFree(d_raysB);
		cudaFree(d_hits);

		float cieYIntegral = CIE_Y_Integral(LAMBDA_MIN, LAMBDA_MAX);
		ResolveFrameBufferKernel<<<grid1D, block1D>>>(fb, samplesPerPixel, cieYIntegral, tonemapper, exposure, d_outRGB);
	}
}
