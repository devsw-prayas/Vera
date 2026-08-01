#pragma once
#include <cstdint>
#include <vector_types.h>

#include "BVH.h"
#include "Triangle.h"

namespace BSPT::Spectral::HWSS::Core {
	struct MeshDesc final {
		uint32_t m_FirstTri = 0;
		uint32_t m_TrisCount = 0;
		uint16_t m_MatId = 0;
	};

	struct Instance final {
		float4 m_Transform[3];
		float4 m_InvTransform[3];
		NXB::BVH2 m_Blas;
		uint32_t m_FirstTri = 0;
	};

	struct GeometryBuffers final {
		float3* m_DevVertexPos = nullptr;
		float3* m_DevVertexNorms = nullptr;
		uint32_t* m_DevIndexBuffer = nullptr;
		NXB::Triangle* m_BvhTris = nullptr;
		Instance* m_DevInstances = nullptr;
		uint16_t* m_TriMatID = nullptr;
		uint32_t m_InstCount = 0;
		NXB::BVH2 m_Tlas;
		uint32_t m_VertCount = 0;
		uint32_t m_TrisCount = 0;
	};

	struct WavefrontHitRecord final {
		float t = 0.f;
		uint32_t m_PrimIdx = 0;
		bool m_Hit = false;
		float m_U = 0.f;
		float m_V = 0.f;
		uint32_t m_InstIdx = 0;
	};
}
