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

	void AppendMesh(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		const float3* verts, uint32_t vertCount,
		const uint32_t* indices, uint32_t indexCount,
		const float3* vertNormals, uint16_t matId)
	{
		for (uint32_t i = 0; i + 2 < indexCount; i += 3) {
			uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
			if (i0 >= vertCount || i1 >= vertCount || i2 >= vertCount) continue;
			float3 a = verts[i0], b = verts[i1], c = verts[i2];

			float3 n0, n1, n2;
			if (vertNormals) {
				n0 = vertNormals[i0]; n1 = vertNormals[i1]; n2 = vertNormals[i2];
			} else {
				float3 e1 = { b.x - a.x, b.y - a.y, b.z - a.z };
				float3 e2 = { c.x - a.x, c.y - a.y, c.z - a.z };
				float3 fn = { e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x };
				float len = sqrtf(fn.x*fn.x + fn.y*fn.y + fn.z*fn.z);
				if (len > 0.f) { fn.x /= len; fn.y /= len; fn.z /= len; }
				n0 = n1 = n2 = fn;
			}

			hPos.push_back(a); hNorms.push_back(n0);
			hPos.push_back(b); hNorms.push_back(n1);
			hPos.push_back(c); hNorms.push_back(n2);
			hTris.push_back({ a, b, c });
			hMatIds.push_back(matId);
		}
	}

	void AppendDiamondMesh(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		float3 center, float girdleRadius, float tableRadius,
		float crownHeight, float pavilionDepth, uint16_t matId, float3 axis)
	{
		// Orthonormal basis around `axis` (the table-to-culet direction; defaults to +Y
		// for "standing" — pass e.g. (1,0,0) to lay the diamond on its side).
		float axisLen = sqrtf(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
		axis = make_float3(axis.x/axisLen, axis.y/axisLen, axis.z/axisLen);
		float3 helper = fabsf(axis.y) < 0.99f ? make_float3(0,1,0) : make_float3(1,0,0);
		float3 tangent = { axis.y*helper.z-axis.z*helper.y, axis.z*helper.x-axis.x*helper.z, axis.x*helper.y-axis.y*helper.x };
		float tlen = sqrtf(tangent.x*tangent.x + tangent.y*tangent.y + tangent.z*tangent.z);
		tangent = make_float3(tangent.x/tlen, tangent.y/tlen, tangent.z/tlen);
		float3 bitangent = { axis.y*tangent.z-axis.z*tangent.y, axis.z*tangent.x-axis.x*tangent.z, axis.x*tangent.y-axis.y*tangent.x };

		constexpr int N = 8;
		float3 girdle[N], table[N];
		for (int k = 0; k < N; ++k) {
			// Rotate the girdle so one pavilion facet sits flat on the floor.
			float ang = 2.f * PI * k / N + 112.5f * PI / 180.f;
			float c = cosf(ang), s = sinf(ang);
			girdle[k] = make_float3(
				center.x + girdleRadius * (c*tangent.x + s*bitangent.x),
				center.y + girdleRadius * (c*tangent.y + s*bitangent.y),
				center.z + girdleRadius * (c*tangent.z + s*bitangent.z));
			float3 tableOff = make_float3(
				tableRadius * (c*tangent.x + s*bitangent.x),
				tableRadius * (c*tangent.y + s*bitangent.y),
				tableRadius * (c*tangent.z + s*bitangent.z));
			table[k] = make_float3(
				center.x + axis.x*crownHeight + tableOff.x,
				center.y + axis.y*crownHeight + tableOff.y,
				center.z + axis.z*crownHeight + tableOff.z);
		}
		float3 tableCenter = make_float3(center.x + axis.x*crownHeight, center.y + axis.y*crownHeight, center.z + axis.z*crownHeight);
		float3 culet       = make_float3(center.x - axis.x*pavilionDepth, center.y - axis.y*pavilionDepth, center.z - axis.z*pavilionDepth);

		auto pushTri = [&](float3 a, float3 b, float3 c, float3 n) {
			hPos.push_back(a); hNorms.push_back(n);
			hPos.push_back(b); hNorms.push_back(n);
			hPos.push_back(c); hNorms.push_back(n);
			hTris.push_back({ a, b, c });
			hMatIds.push_back(matId);
		};

		// Geometric face normal, re-oriented outward from `center` regardless of winding.
		auto outwardNormal = [&](float3 a, float3 b, float3 c) -> float3 {
			float3 e1 = { b.x-a.x, b.y-a.y, b.z-a.z };
			float3 e2 = { c.x-a.x, c.y-a.y, c.z-a.z };
			float3 n  = { e1.y*e2.z-e1.z*e2.y, e1.z*e2.x-e1.x*e2.z, e1.x*e2.y-e1.y*e2.x };
			float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
			n = { n.x/len, n.y/len, n.z/len };
			float3 centroid = { (a.x+b.x+c.x)/3.f, (a.y+b.y+c.y)/3.f, (a.z+b.z+c.z)/3.f };
			float3 out = { centroid.x-center.x, centroid.y-center.y, centroid.z-center.z };
			float d = n.x*out.x + n.y*out.y + n.z*out.z;
			return d < 0.f ? make_float3(-n.x,-n.y,-n.z) : n;
		};

		// Table: flat top octagon.
		for (int k = 0; k < N; ++k) {
			int kn = (k+1) % N;
			pushTri(tableCenter, table[k], table[kn], axis);
		}
		// Crown (bezel) facets: table ring -> girdle ring, 8 quads (2 tris each).
		for (int k = 0; k < N; ++k) {
			int kn = (k+1) % N;
			float3 n = outwardNormal(table[k], girdle[k], girdle[kn]);
			pushTri(table[k], girdle[k], girdle[kn], n);
			pushTri(table[k], girdle[kn], table[kn], n);
		}
		// Pavilion facets: girdle ring -> culet, 8 triangles.
		for (int k = 0; k < N; ++k) {
			int kn = (k+1) % N;
			float3 n = outwardNormal(girdle[k], culet, girdle[kn]);
			pushTri(girdle[k], culet, girdle[kn], n);
		}
	}

	// Five-wall box room (no front wall) from [-scale,-scale,-scale] to [+scale,+scale,+scale].
	void AppendBoxWalls(
		std::vector<float3>&        hPos, std::vector<float3>&        hNorms,
		std::vector<NXB::Triangle>& hTris, std::vector<uint16_t>&     hMatIds,
		uint16_t matFloor, uint16_t matCeil, uint16_t matBack,
		uint16_t matLeft,  uint16_t matRight, float scale)
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
		for (int i = 0; i < 30; ++i) { bv[i].x *= scale; bv[i].y *= scale; bv[i].z *= scale; }
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
