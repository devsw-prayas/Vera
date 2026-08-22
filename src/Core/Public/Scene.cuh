#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <BVH.h>
#include <Triangle.h>

#include "CoreUtils.h"

namespace Vera::Core {

	// Single-instance (BLAS==TLAS-instance-of-one) scene container.
	// Material storage is untyped for now (raw bytes, matStride per element) since
	// the spectral material layout isn't finalized yet — swap d_mats/UploadScene's
	// material params for a typed HWSS material buffer once that lands.
	struct SceneGPU final {
		float3*        d_pos    = nullptr;
		float3*        d_norms  = nullptr;
		uint32_t*      d_idx    = nullptr;
		NXB::Triangle* d_tris   = nullptr;
		uint16_t*      d_matIds = nullptr;
		Instance*      d_inst   = nullptr;
		void*          d_mats   = nullptr;
		uint32_t       matCount = 0;
		NXB::BVH2      blas{};
		NXB::BVH2      tlas{};
		GeometryBuffers geom{};
	};

	void FreeScene(SceneGPU& s);

	// Uploads a single-instance scene (identity transform) and builds BLAS + TLAS.
	// hMats/matStride: raw material bytes, matCount elements of matStride bytes each.
	void UploadScene(
		const std::vector<float3>&        hPos,
		const std::vector<float3>&        hNorms,
		const std::vector<NXB::Triangle>& hTris,
		const std::vector<uint16_t>&      hMatIds,
		const void*                       hMats,
		uint32_t                          matCount,
		uint32_t                          matStride,
		float3                            tlasMin,
		float3                            tlasMax,
		SceneGPU&                         s);

	// ── Geometry helpers ─────────────────────────────────────────────────────

	void AppendSphere(float3 center, float radius,
		std::vector<float3>& pos, std::vector<float3>& norms);

	// Two triangles (v0,v1,v2)+(v0,v2,v3) with shared normal n.
	void AppendQuad(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		float3 v0, float3 v1, float3 v2, float3 v3, float3 n, uint16_t matId);

	// Sphere mesh appended in-place.
	void AppendSphereMesh(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		float3 center, float radius, uint16_t matId);

	// Five-wall box room (no front wall) from [-1,-1,-1] to [+1,+1,+1].
	// matFloor/matCeil/matBack/matLeft/matRight are material indices.
	void AppendBoxWalls(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		uint16_t matFloor, uint16_t matCeil, uint16_t matBack,
		uint16_t matLeft,  uint16_t matRight);
}
