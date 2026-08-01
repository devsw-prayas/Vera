#include "Camera.cuh"
#include "PCG32.cuh"
#include <cfloat>
#include <CudaMath.h>

namespace BSPT::Spectral::HWSS::Core {
	static constexpr float CAM_PI = 3.14159265358979323846f;

	__global__ void GeneratePrimaryRaysKernel(Camera camera, Ray* rays) {
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= camera.m_Width || y >= camera.m_Height) return;

		float ndcX = (x + 0.5f) / camera.m_Width * 2.f - 1.f;
		float ndcY = -((y + 0.5f) / camera.m_Height * 2.f - 1.f);

		float3 d = normalize(camera.m_Forward
			+ ndcX * camera.m_AspectRatio * camera.m_HalfTanFovY * camera.m_Right
			+ ndcY * camera.m_HalfTanFovY * camera.m_Up);

		unsigned int idx = y * camera.m_Width + x;
		rays[idx].m_Origin    = camera.m_Origin;
		rays[idx].m_Direction = d;
		rays[idx].m_Tmin      = 1e-4f;
		rays[idx].m_Tmax      = FLT_MAX;
	}
}
