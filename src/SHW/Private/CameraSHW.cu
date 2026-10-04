#include "CameraSHW.cuh"
#include <CoreUtils.h>
#include <Philox.cuh>
#include <cfloat>
#include <CudaMath.h>

namespace Vera::SHW {
	using Vera::Core::LAMBDA_MIN;
	using Vera::Core::LAMBDA_RANGE;
	using Vera::Core::PI;

	__global__ void GeneratePrimaryRaysSHWKernel(
		Core::Camera camera, RayCoreSoA core, RayExtSoA ext,
		uint32_t sampleIdx, uint8_t defaultMediumIdx) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= camera.m_Width || y >= camera.m_Height) return;

		unsigned int pixelId = y * camera.m_Width + x;

		Core::PhiloxKey key{};
		key.m_PixelId = pixelId;
		key.m_FrameIdx = 0;
		key.m_SampleIdx = sampleIdx;
		key.m_NeighborIdx = 0;
		key.m_StreamSelect = 0;
		Core::Philox4x32Default rng = Core::Philox4x32Default::fromKey(key);

		// Sub-pixel jitter + lens sample + wavelength, all from one counter-based draw
		float4 u = rng.nextFloat4();
		float ndcX = (x + u.x) / camera.m_Width * 2.f - 1.f;
		float ndcY = -((y + u.y) / camera.m_Height * 2.f - 1.f);

		float3 d = normalize(camera.m_Forward
							 + ndcX * camera.m_AspectRatio * camera.m_HalfTanFovY * camera.m_Right
							 + ndcY * camera.m_HalfTanFovY * camera.m_Up);

		float3 origin = camera.m_Origin;
		float3 dir = d;

		if (camera.m_LensRadius > 0.f) {
			// Uniform disk sample via concentric mapping
			float  r = sqrtf(u.z) * camera.m_LensRadius;
			float  phi = 2.f * PI * u.w;
			float3 lensOff = r * cosf(phi) * camera.m_Right + r * sinf(phi) * camera.m_Up;

			float  tFocus = camera.m_FocusDist / dot(d, camera.m_Forward);
			float3 focusPt = camera.m_Origin + d * tFocus;

			origin = camera.m_Origin + lensOff;
			dir = normalize(focusPt - origin);
		}

		// Single hero wavelength, uniform over the range (no offset lanes in SHW)
		float wu = rng.nextFloat();
		float lambda = LAMBDA_MIN + wu * LAMBDA_RANGE;

		RaySHW ray{};
		ray.m_Origin = origin;
		ray.m_Direction = dir;
		ray.m_Lambda = lambda;
		ray.m_SensorLambda = lambda; // diverges from m_Lambda only at a fluorescent shift
		ray.m_Throughput = 1.f;
		ray.m_Pdf = 1.f / LAMBDA_RANGE;
		ray.m_PixelId = pixelId;
		ray.m_SampleIdx = sampleIdx;
		ray.m_BounceCount = 0;
		ray.m_BsdfPdf = 1.f; // irrelevant while SHW_FLAG_DELTA is set below
		// Mark as delta so first emissive hit uses w=1 (camera has no competing NEE strategy)
		ray.m_Flags = SHW_FLAG_DELTA;
		ray.m_IorCurr = 1.f;
		ray.m_MediumIdx = defaultMediumIdx;

		StoreRay(core, ext, pixelId, ray);
	}
}
