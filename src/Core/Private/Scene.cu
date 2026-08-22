#include "Scene.cuh"
#include <cmath>
#include <BVHBuilder.h>
#include <AABB.h>

namespace Vera::Core {
	static constexpr int   SPHERE_STACKS = 32;
	static constexpr int   SPHERE_SLICES = 64;
	static constexpr int   SPHERE_TRIS   = SPHERE_SLICES * 2 + (SPHERE_STACKS - 2) * SPHERE_SLICES * 2;

	void FreeScene(SceneGPU& s) {
		cudaFree(s.d_pos);  cudaFree(s.d_norms); cudaFree(s.d_idx);
		cudaFree(s.d_tris); cudaFree(s.d_matIds);
		cudaFree(s.d_inst); cudaFree(s.d_mats);
		NXB::FreeDeviceBVH(s.blas);
		NXB::FreeDeviceBVH(s.tlas);
		s = SceneGPU{};
	}

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
		SceneGPU&                         s)
	{
		uint32_t vertCount = (uint32_t)hPos.size();
		uint32_t triCount  = (uint32_t)hTris.size();

		std::vector<uint32_t> hIdx(vertCount);
		for (uint32_t i = 0; i < vertCount; ++i) hIdx[i] = i;

		cudaMalloc(&s.d_pos,    vertCount * sizeof(float3));
		cudaMalloc(&s.d_norms,  vertCount * sizeof(float3));
		cudaMalloc(&s.d_idx,    vertCount * sizeof(uint32_t));
		cudaMalloc(&s.d_tris,   triCount  * sizeof(NXB::Triangle));
		cudaMalloc(&s.d_matIds, triCount  * sizeof(uint16_t));
		cudaMalloc(&s.d_mats,   (size_t)matCount * matStride);

		cudaMemcpy(s.d_pos,    hPos.data(),    vertCount * sizeof(float3),        cudaMemcpyHostToDevice);
		cudaMemcpy(s.d_norms,  hNorms.data(),  vertCount * sizeof(float3),        cudaMemcpyHostToDevice);
		cudaMemcpy(s.d_idx,    hIdx.data(),    vertCount * sizeof(uint32_t),      cudaMemcpyHostToDevice);
		cudaMemcpy(s.d_tris,   hTris.data(),   triCount  * sizeof(NXB::Triangle), cudaMemcpyHostToDevice);
		cudaMemcpy(s.d_matIds, hMatIds.data(), triCount  * sizeof(uint16_t),      cudaMemcpyHostToDevice);
		cudaMemcpy(s.d_mats,   hMats,          (size_t)matCount * matStride,      cudaMemcpyHostToDevice);
		s.matCount = matCount;

		s.blas = NXB::BuildBVH2(s.d_tris, triCount);

		Instance hInst{};
		hInst.m_Transform[0]    = make_float4(1,0,0,0);
		hInst.m_Transform[1]    = make_float4(0,1,0,0);
		hInst.m_Transform[2]    = make_float4(0,0,1,0);
		hInst.m_InvTransform[0] = make_float4(1,0,0,0);
		hInst.m_InvTransform[1] = make_float4(0,1,0,0);
		hInst.m_InvTransform[2] = make_float4(0,0,1,0);
		hInst.m_Blas     = s.blas;
		hInst.m_FirstTri = 0;
		cudaMalloc(&s.d_inst, sizeof(Instance));
		cudaMemcpy(s.d_inst, &hInst, sizeof(Instance), cudaMemcpyHostToDevice);

		NXB::AABB  hTlasAabb{ tlasMin, tlasMax };
		NXB::AABB* d_tlasAabb;
		cudaMalloc(&d_tlasAabb, sizeof(NXB::AABB));
		cudaMemcpy(d_tlasAabb, &hTlasAabb, sizeof(NXB::AABB), cudaMemcpyHostToDevice);
		s.tlas = NXB::BuildBVH2(d_tlasAabb, 1);
		cudaFree(d_tlasAabb);

		s.geom.m_DevVertexPos   = s.d_pos;
		s.geom.m_DevVertexNorms = s.d_norms;
		s.geom.m_DevIndexBuffer = s.d_idx;
		s.geom.m_BvhTris        = s.d_tris;
		s.geom.m_DevInstances   = s.d_inst;
		s.geom.m_TriMatID       = s.d_matIds;
		s.geom.m_VertCount      = vertCount;
		s.geom.m_TrisCount      = triCount;
		s.geom.m_InstCount      = 1;
		s.geom.m_Tlas           = s.tlas;
	}

	// ── Geometry helpers ─────────────────────────────────────────────────────

	void AppendSphere(float3 center, float radius,
		std::vector<float3>& pos, std::vector<float3>& norms)
	{
		auto pt = [&](int s, int j) -> float3 {
			float theta = PI * s / SPHERE_STACKS;
			float phi   = 2.f * PI * j / SPHERE_SLICES;
			return make_float3(
				center.x + radius * sinf(theta) * cosf(phi),
				center.y + radius * cosf(theta),
				center.z + radius * sinf(theta) * sinf(phi));
		};
		auto nm = [&](float3 p) -> float3 {
			float3 d = { p.x - center.x, p.y - center.y, p.z - center.z };
			float  l = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
			return { d.x/l, d.y/l, d.z/l };
		};
		auto push = [&](float3 p) { pos.push_back(p); norms.push_back(nm(p)); };

		for (int j = 0; j < SPHERE_SLICES; ++j)
			{ push(pt(0,0)); push(pt(1,j)); push(pt(1,(j+1)%SPHERE_SLICES)); }
		for (int s = 1; s < SPHERE_STACKS-1; ++s)
			for (int j = 0; j < SPHERE_SLICES; ++j) {
				int jn = (j+1)%SPHERE_SLICES;
				float3 a=pt(s,j),b=pt(s+1,j),c=pt(s+1,jn),d=pt(s,jn);
				push(a); push(b); push(c);
				push(a); push(c); push(d);
			}
		for (int j = 0; j < SPHERE_SLICES; ++j)
			{ push(pt(SPHERE_STACKS-1,j)); push(pt(SPHERE_STACKS,0)); push(pt(SPHERE_STACKS-1,(j+1)%SPHERE_SLICES)); }
	}

	void AppendQuad(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		float3 v0, float3 v1, float3 v2, float3 v3, float3 n, uint16_t matId)
	{
		uint32_t base = (uint32_t)hMatIds.size();
		hPos.push_back(v0); hNorms.push_back(n);
		hPos.push_back(v1); hNorms.push_back(n);
		hPos.push_back(v2); hNorms.push_back(n);
		hPos.push_back(v0); hNorms.push_back(n);
		hPos.push_back(v2); hNorms.push_back(n);
		hPos.push_back(v3); hNorms.push_back(n);
		hMatIds.push_back(matId); hMatIds.push_back(matId);
		hTris.push_back({ hPos[base*3+0], hPos[base*3+1], hPos[base*3+2] });
		hTris.push_back({ hPos[(base+1)*3+0], hPos[(base+1)*3+1], hPos[(base+1)*3+2] });
	}

	void AppendSphereMesh(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		float3 center, float radius, uint16_t matId)
	{
		uint32_t triStart = (uint32_t)hMatIds.size();
		AppendSphere(center, radius, hPos, hNorms);
		for (uint32_t i = 0; i < (uint32_t)SPHERE_TRIS; ++i) {
			uint32_t t = triStart + i;
			hTris.push_back({ hPos[t*3], hPos[t*3+1], hPos[t*3+2] });
			hMatIds.push_back(matId);
		}
	}

	// Five-wall box room (no front wall) from [-1,-1,-1] to [+1,+1,+1].
	void AppendBoxWalls(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		uint16_t matFloor, uint16_t matCeil, uint16_t matBack,
		uint16_t matLeft,  uint16_t matRight)
	{
		float3 bv[30] = {
			{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1},  // floor tri0
			{-1,-1,-1},{ 1,-1, 1},{-1,-1, 1},  // floor tri1
			{-1, 1,-1},{ 1, 1, 1},{ 1, 1,-1},  // ceil tri0
			{-1, 1,-1},{-1, 1, 1},{ 1, 1, 1},  // ceil tri1
			{-1,-1,-1},{ 1, 1,-1},{ 1,-1,-1},  // back tri0
			{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1},  // back tri1
			{-1,-1,-1},{-1,-1, 1},{-1, 1, 1},  // left tri0
			{-1,-1,-1},{-1, 1, 1},{-1, 1,-1},  // left tri1
			{ 1,-1,-1},{ 1, 1, 1},{ 1,-1, 1},  // right tri0
			{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},  // right tri1
		};
		float3 bn[30];
		for (int i=0;  i<6;  ++i) bn[i] = {0, 1,0};
		for (int i=6;  i<12; ++i) bn[i] = {0,-1,0};
		for (int i=12; i<18; ++i) bn[i] = {0, 0,1};
		for (int i=18; i<24; ++i) bn[i] = {1, 0,0};
		for (int i=24; i<30; ++i) bn[i] = {-1,0,0};

		uint32_t base = (uint32_t)hMatIds.size();
		for (int i = 0; i < 30; ++i) { hPos.push_back(bv[i]); hNorms.push_back(bn[i]); }
		hMatIds.push_back(matFloor); hMatIds.push_back(matFloor);
		hMatIds.push_back(matCeil);  hMatIds.push_back(matCeil);
		hMatIds.push_back(matBack);  hMatIds.push_back(matBack);
		hMatIds.push_back(matLeft);  hMatIds.push_back(matLeft);
		hMatIds.push_back(matRight); hMatIds.push_back(matRight);
		for (int i = 0; i < 10; ++i)
			hTris.push_back({ hPos[(base+i)*3], hPos[(base+i)*3+1], hPos[(base+i)*3+2] });
	}
}
