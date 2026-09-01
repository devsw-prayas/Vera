#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <BVH.h>
#include <Triangle.h>

#include "CoreUtils.h"

namespace Vera::Core {
	// Single-instance scene container. Material storage is untyped bytes (matStride per
	// element) so Core doesn't depend on the spectral material layout.
	struct SceneGPU final {
		float3* d_pos = nullptr;
		float3* d_norms = nullptr;
		uint32_t* d_idx = nullptr;
		NXB::Triangle* d_tris = nullptr;
		uint16_t* d_matIds = nullptr;
		Instance* d_inst = nullptr;
		void* d_mats = nullptr;
		uint32_t       matCount = 0;
		NXB::BVH2      blas{};
		NXB::BVH2      tlas{};
		GeometryBuffers geom{};
	};

	void FreeScene(SceneGPU& s);

	// Uploads a single-instance scene (identity transform) and builds BLAS + TLAS.
	// hMats/matStride: raw material bytes, matCount elements of matStride bytes each.
	void UploadScene(
		const std::vector<float3>& hPos,
		const std::vector<float3>& hNorms,
		const std::vector<NXB::Triangle>& hTris,
		const std::vector<uint16_t>& hMatIds,
		const void* hMats,
		uint32_t                          matCount,
		uint32_t                          matStride,
		float3                            tlasMin,
		float3                            tlasMax,
		SceneGPU& s);

	// -- Geometry helpers -----------------------------------------------------

	void AppendSphere(float3 center, float radius,
					  std::vector<float3>& pos, std::vector<float3>& norms);

	// Two triangles (v0,v1,v2)+(v0,v2,v3) with shared normal n.
	void AppendQuad(
		std::vector<float3>& hPos, std::vector<float3>& hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>& hMatIds,
		float3 v0, float3 v1, float3 v2, float3 v3, float3 n, uint16_t matId);

	// Sphere mesh appended in-place.
	void AppendSphereMesh(
		std::vector<float3>& hPos, std::vector<float3>& hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>& hMatIds,
		float3 center, float radius, uint16_t matId);

	// Generic indexed triangle mesh, flattened into the triangle soup. indices: indexCount
	// entries (3/face) into verts. vertNormals != nullptr => smooth shading, else flat.
	void AppendMesh(
		std::vector<float3>& hPos, std::vector<float3>& hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>& hMatIds,
		const float3* verts, uint32_t vertCount,
		const uint32_t* indices, uint32_t indexCount,
		const float3* vertNormals, uint16_t matId);

	// Simplified round-brilliant-cut diamond: octagonal table, 8 crown facets to an
	// octagonal girdle, 8 pavilion facets to a culet point. Faceted enough for real
	// per-facet refraction/TIR/dispersion. axis: table-to-culet direction.
	void AppendDiamondMesh(
		std::vector<float3>& hPos, std::vector<float3>& hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>& hMatIds,
		float3 center, float girdleRadius, float tableRadius,
		float crownHeight, float pavilionDepth, uint16_t matId,
		float3 axis = make_float3(0.f, 1.f, 0.f));

	// Five-wall box room (no front wall) from [-scale,-scale,-scale] to [+scale,+scale,+scale].
	// matFloor/matCeil/matBack/matLeft/matRight are material indices.
	void AppendBoxWalls(
		std::vector<float3>& hPos, std::vector<float3>& hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>& hMatIds,
		uint16_t matFloor, uint16_t matCeil, uint16_t matBack,
		uint16_t matLeft, uint16_t matRight, float scale = 1.f);
}