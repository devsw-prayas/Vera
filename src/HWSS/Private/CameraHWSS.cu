#include "CameraHWSS.cuh"
#include "SpectralConstants.h"
#include "PCG32.cuh"
#include "QMCHWSS.h"
#include <cfloat>
#include <CudaMath.h>

namespace Vera::Spectral::HWSS {
	__global__ void GeneratePrimaryRaysHWSSKernel(
		Core::Camera camera, RayCoreSoA core, RayExtSoA ext,
		unsigned int sampleIdx, unsigned char defaultMediumIdx) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= camera.m_Width || y >= camera.m_Height) return;

		unsigned int pixelId = y * camera.m_Width + x;
		unsigned long long seedValue = static_cast<unsigned long long>(pixelId) * 1973ULL
			+ static_cast<unsigned long long>(sampleIdx) * 9277ULL + 1ULL;

		HybridRNG rng{};
		rng.pcg.seed(seedValue);
		rng.sampleIdx = sampleIdx;
		rng.pixelId = pixelId;
		rng.dimBase = 0; // camera ray-gen owns QMC dims [0, kQmcCameraDims)

		// Sub-pixel jitter
		float2 uv = rng.nextFloat2();
		float ndcX = (x + uv.x) / camera.m_Width * 2.f - 1.f;
		float ndcY = -((y + uv.y) / camera.m_Height * 2.f - 1.f);

		float3 d = normalize(camera.m_Forward
							 + ndcX * camera.m_AspectRatio * camera.m_HalfTanFovY * camera.m_Right
							 + ndcY * camera.m_HalfTanFovY * camera.m_Up);

		float3 origin = camera.m_Origin;
		float3 dir = d;

		if (camera.m_LensRadius > 0.f) {
			// Uniform disk sample via concentric mapping
			float2 u = rng.nextFloat2();
			float  r = sqrtf(u.x) * camera.m_LensRadius;
			float  phi = 2.f * PI * u.y;
			float3 lensOff = r * cosf(phi) * camera.m_Right + r * sinf(phi) * camera.m_Up;

			float  tFocus = camera.m_FocusDist / dot(d, camera.m_Forward);
			float3 focusPt = camera.m_Origin + d * tFocus;

			origin = camera.m_Origin + lensOff;
			dir = normalize(focusPt - origin);
		}

		// Hero + offset wavelengths (Wilkie et al. 2014): hero uniform over the range,
		// offsets a fixed rotation of it wrapped back in - marginal pdf stays 1/LAMBDA_RANGE.
		float hero = LAMBDA_MIN + rng.nextFloat() * LAMBDA_RANGE;
		float4 wavelengths;
		wavelengths.x = hero;
		wavelengths.y = LAMBDA_MIN + fmodf((hero - LAMBDA_MIN) + 1.f * (LAMBDA_RANGE / HWSS_LANES), LAMBDA_RANGE);
		wavelengths.z = LAMBDA_MIN + fmodf((hero - LAMBDA_MIN) + 2.f * (LAMBDA_RANGE / HWSS_LANES), LAMBDA_RANGE);
		wavelengths.w = LAMBDA_MIN + fmodf((hero - LAMBDA_MIN) + 3.f * (LAMBDA_RANGE / HWSS_LANES), LAMBDA_RANGE);

		float lanePdf = 1.f / LAMBDA_RANGE;

		RayHWSS ray{};
		ray.m_Origin = origin;
		ray.m_Direction = dir;
		ray.m_Wavelengths = wavelengths;
		ray.m_SensorWavelengths = wavelengths; // diverges from m_Wavelengths only at a fluorescent shift
		ray.m_LaneFluoresced = 0;
		ray.m_Throughput = make_float4(1.f, 1.f, 1.f, 1.f);
		ray.m_Pdf = make_float4(lanePdf, lanePdf, lanePdf, lanePdf);
		ray.m_RngState = rng.pcg.m_State;
		ray.pixelId = pixelId;
		ray.m_SampleIdx = sampleIdx;
		ray.m_BounceCount = 0;
		ray.m_BsdfPdf = 1.f; // irrelevant while RAY_FLAG_DELTA is set below
		// Mark as delta so first emissive hit uses w=1 (camera has no competing NEE strategy)
		ray.flags = RAY_FLAG_DELTA;
		ray.m_IorCurr = 1.f;
		ray.m_MediumIdx = defaultMediumIdx;

		StoreRay(core, ext, pixelId, ray);
	}
}