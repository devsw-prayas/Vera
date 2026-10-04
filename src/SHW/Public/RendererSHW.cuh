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

namespace Vera::SHW {
	void RenderSHW(
		const Core::GeometryBuffers& geom,
		const Spectral::HWSS::Material* d_materials,
		const Spectral::HWSS::MediumHWSS* d_media,
		const Spectral::HWSS::LightBVH& lightBvh,
		const Spectral::HWSS::EnvMapHWSS& envMap,
		const Core::Camera& camera,
		Spectral::HWSS::FrameBufferHWSS& fb,
		float4* d_outRGB,
		uint32_t samplesPerPixel,
		uint32_t maxBounces,
		unsigned char defaultMediumIdx = 0,
		Core::ToneMapper tonemapper = Core::ToneMapper::AgX,
		float exposure = 1.f,
		bool useOptix = false);
}
