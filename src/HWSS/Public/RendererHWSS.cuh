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
	// Wavefront loop: per sample, generate primary rays then ping-pong two ray buffers
	// through traversal + shade for maxBounces, accumulating into fb. On return fb is
	// resolved (XYZ -> tonemap -> sRGB) into d_outRGB (caller-owned, W*H float4).
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
		// OptiX closest-hit traversal instead of the software kernel; only effective when
		// built with VERA_ENABLE_OPTIX, otherwise silently falls back.
		bool useOptix = false);
}