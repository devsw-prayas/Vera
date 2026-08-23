#pragma once
#include <cuda_runtime.h>
#include <CoreUtils.h>
#include <Camera.cuh>
#include <Tonemap.h>

#include "MaterialHWSS.h"
#include "EnvMapHWSS.h"
#include "MediumHWSS.h"
#include "LightBVHHWSS.h"
#include "FrameBufferHWSS.cuh"

namespace Vera::Spectral::HWSS {
	// Wavefront dispatch loop: per sample, generates primary rays then ping-pongs
	// two RayHWSS buffers through traversal + shade for maxBounces iterations,
	// accumulating spectral contributions into fb as it goes. On return, fb has
	// been resolved (XYZ -> tonemap -> sRGB) into d_outRGB, a caller-owned device
	// buffer of camera.m_Width * camera.m_Height float4 entries.
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
		unsigned char defaultMediumIdx = 0,
		Core::ToneMapper tonemapper = Core::ToneMapper::AgX,
		float exposure = 1.f,
		// Hardware-accelerated (OptiX) closest-hit traversal instead of the software
		// TraversalKernelWavefront -- only has an effect when built with VERA_ENABLE_OPTIX;
		// silently ignored (falls back to the software path) otherwise. Everything else about
		// the pipeline (shading, NEE shadow rays, compaction, material sort) is unchanged.
		bool useOptix = false);
}
