#include "RendererSHW.cuh"
#include "WavefrontSHW.cuh"
#include "ShadeSHW.cuh"
#include "CIE.h"
#include <CoreUtils.h>
#include <Traversal.cuh>
#include "MaterialSortHWSS.cuh"
#ifdef VERA_ENABLE_OPTIX
#include <OptixTraversal.h>
#endif

namespace Vera::SHW {
	using namespace Vera::Spectral::HWSS;

	void RenderSHW(
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
		float exposure,
		bool useOptix) {
		uint32_t rayCount = camera.m_Width * camera.m_Height;

		SHWBuffers buf = AllocSHWBuffers(rayCount);

		Core::MaterialSorter sorter;
		sorter.Init(rayCount);

		CIETexture cieTex = MakeCIETexture(LAMBDA_MIN, LAMBDA_MAX);

		dim3 block1D(256);

#ifdef VERA_ENABLE_OPTIX
		Core::OptixTraversalContext optixCtx{};
		if (useOptix) optixCtx = Core::InitOptixTraversal(geom);
#endif

		for (uint32_t sampleIdx = 0; sampleIdx < samplesPerPixel; ++sampleIdx) {
			GenerateSHW(camera, buf.coreA, buf.extA, sampleIdx, defaultMediumIdx);

			uint32_t activeCount = rayCount;
			for (uint32_t bounce = 0; bounce < maxBounces && activeCount > 0; ++bounce) {
				dim3 grid1D((activeCount + block1D.x - 1) / block1D.x);

				TraverseSHW(geom, buf.coreA, buf.d_hits, activeCount, useOptix
#ifdef VERA_ENABLE_OPTIX
					, useOptix ? &optixCtx : nullptr
#endif
				);

				uint32_t* order = sorter.Sort(geom, buf.d_hits, activeCount, /*stream*/0);

				ShadeKernelSHWWavefront<<<grid1D, block1D>>>(
					geom, buf.coreA, buf.extA, buf.d_hits, order, d_materials, d_media, lightBvh,
					buf.coreB, buf.extB, activeCount, fb, envMap, maxBounces, cieTex.tex);

				activeCount = CompactSHW(buf.compactor, buf.coreB, buf.extB, activeCount,
										 buf.coreA, buf.extA);
			}
		}

#ifdef VERA_ENABLE_OPTIX
		if (useOptix) Core::DestroyOptixTraversal(optixCtx);
#endif

		FreeCIETexture(cieTex);
		sorter.Destroy();

		float cieYIntegral = CIE_Y_Integral(LAMBDA_MIN, LAMBDA_MAX);
		dim3 grid1D((rayCount + block1D.x - 1) / block1D.x);
		ResolveFrameBufferKernel<<<grid1D, block1D>>>(fb, samplesPerPixel, cieYIntegral, tonemapper, exposure, d_outRGB);

		FreeSHWBuffers(buf);
	}
}
