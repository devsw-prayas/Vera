#include "Traversal.cuh"
#include <CudaMath.h>

namespace BSPT::Spectral::HWSS::Core {
	__device__ inline bool RayAABB(const Ray& ray, const NXB::AABB& aabb, float tMax) {
		float tmin = ray.m_Tmin;
		float tmax = tMax;
		float3 invD = make_float3(
			1.0f / ray.m_Direction.x,
			1.0f / ray.m_Direction.y,
			1.0f / ray.m_Direction.z
		);
		invD.x = fabsf(invD.x) > 1e8f ? copysignf(1e8f, invD.x) : invD.x;
		invD.y = fabsf(invD.y) > 1e8f ? copysignf(1e8f, invD.y) : invD.y;
		invD.z = fabsf(invD.z) > 1e8f ? copysignf(1e8f, invD.z) : invD.z;

		float tminx = fminf((aabb.bMin.x - ray.m_Origin.x) * invD.x, (aabb.bMax.x - ray.m_Origin.x) * invD.x);
		float tmaxx = fmaxf((aabb.bMin.x - ray.m_Origin.x) * invD.x, (aabb.bMax.x - ray.m_Origin.x) * invD.x);
		float tminy = fminf((aabb.bMin.y - ray.m_Origin.y) * invD.y, (aabb.bMax.y - ray.m_Origin.y) * invD.y);
		float tmaxy = fmaxf((aabb.bMin.y - ray.m_Origin.y) * invD.y, (aabb.bMax.y - ray.m_Origin.y) * invD.y);
		float tminz = fminf((aabb.bMin.z - ray.m_Origin.z) * invD.z, (aabb.bMax.z - ray.m_Origin.z) * invD.z);
		float tmaxz = fmaxf((aabb.bMin.z - ray.m_Origin.z) * invD.z, (aabb.bMax.z - ray.m_Origin.z) * invD.z);

		tmin = fmaxf(tmin, fmaxf(tminx, fmaxf(tminy, tminz)));
		tmax = fminf(tmax, fminf(tmaxx, fminf(tmaxy, tmaxz)));
		return tmax >= tmin;
	}

	__device__ inline bool RayTriangle(const Ray& ray, const NXB::Triangle& tri, float tMax, float& t, float& U, float& V) {
		float3 edge1 = tri.v1 - tri.v0;
		float3 edge2 = tri.v2 - tri.v0;
		float3 h = cross(ray.m_Direction, edge2);
		float a = dot(edge1, h);
		if (a > -FLT_EPSILON && a < FLT_EPSILON) return false;
		float f = 1.f / a;
		float3 s = ray.m_Origin - tri.v0;
		float u = f * dot(s, h);
		if (u < 0.f || u > 1.f) return false;
		float3 q = cross(s, edge1);
		float v = f * dot(ray.m_Direction, q);
		if (v < 0.f || u + v > 1.f) return false;
		t = f * dot(edge2, q);
		if (t < ray.m_Tmin || t > tMax) return false;
		U = u;
		V = v;
		return true;
	}

	__global__ void TraversalKernelWavefront(GeometryBuffers geom, Ray* rays, WavefrontHitRecord* hits, uint32_t rayCount) {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		Ray r = rays[idx];

		WavefrontHitRecord h{};
		h.m_Hit = false;
		h.t     = FLT_MAX;

		unsigned int tlasStk[32];
		tlasStk[0] = geom.m_Tlas.nodeCount - 1;
		int tlasStkPtr = 0;

		while (tlasStkPtr >= 0) {
			NXB::BVH2::Node tlasNode = geom.m_Tlas.nodes[tlasStk[tlasStkPtr--]];
			if (!RayAABB(r, tlasNode.bounds, h.t)) continue;

			if (tlasNode.leftChild == INVALID_IDX) {
				uint32_t instIdx  = tlasNode.rightChild;
				Instance inst     = geom.m_DevInstances[instIdx];

				float3 wo = r.m_Origin;
				float3 wd = r.m_Direction;

				Ray lr{};
				lr.m_Origin = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), wo) + inst.m_InvTransform[0].w,
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), wo) + inst.m_InvTransform[1].w,
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), wo) + inst.m_InvTransform[2].w
				);
				lr.m_Direction = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), wd),
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), wd),
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), wd)
				);
				lr.m_Tmin = r.m_Tmin;
				lr.m_Tmax = r.m_Tmax;

				unsigned int blasStk[32];
				blasStk[0] = inst.m_Blas.nodeCount - 1;
				int blasStkPtr = 0;

				while (blasStkPtr >= 0) {
					NXB::BVH2::Node blasNode = inst.m_Blas.nodes[blasStk[blasStkPtr--]];
					if (!RayAABB(lr, blasNode.bounds, h.t)) continue;

					if (blasNode.leftChild == INVALID_IDX) {
						uint32_t globalPrim = inst.m_FirstTri + blasNode.rightChild;
						float t, u, v;
						if (RayTriangle(lr, geom.m_BvhTris[globalPrim], h.t, t, u, v)) {
							h.t         = t;
							h.m_U       = u;
							h.m_V       = v;
							h.m_PrimIdx = globalPrim;
							h.m_InstIdx = instIdx;
							h.m_Hit     = true;
						}
					} else {
						blasStk[++blasStkPtr] = blasNode.rightChild;
						blasStk[++blasStkPtr] = blasNode.leftChild;
					}
				}
			} else {
				tlasStk[++tlasStkPtr] = tlasNode.rightChild;
				tlasStk[++tlasStkPtr] = tlasNode.leftChild;
			}
		}

		hits[idx] = h;
	}

	__global__ void TraversalKernelWavefront(GeometryBuffers geom, RayHWSS* rays, WavefrontHitRecord* hits, uint32_t rayCount) {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		RayHWSS ray = rays[idx];

		WavefrontHitRecord h{};
		h.m_Hit = false;
		h.t     = FLT_MAX;

		if (ray.flags & RAY_FLAG_DEAD) { hits[idx] = h; return; }

		Ray r{};
		r.m_Origin    = ray.m_Origin;
		r.m_Direction = ray.m_Direction;
		r.m_Tmin      = 1e-4f;
		r.m_Tmax      = FLT_MAX;

		unsigned int tlasStk[32];
		tlasStk[0] = geom.m_Tlas.nodeCount - 1;
		int tlasStkPtr = 0;

		while (tlasStkPtr >= 0) {
			NXB::BVH2::Node tlasNode = geom.m_Tlas.nodes[tlasStk[tlasStkPtr--]];
			if (!RayAABB(r, tlasNode.bounds, h.t)) continue;

			if (tlasNode.leftChild == INVALID_IDX) {
				uint32_t instIdx  = tlasNode.rightChild;
				Instance inst     = geom.m_DevInstances[instIdx];

				float3 wo = r.m_Origin;
				float3 wd = r.m_Direction;

				Ray lr{};
				lr.m_Origin = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), wo) + inst.m_InvTransform[0].w,
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), wo) + inst.m_InvTransform[1].w,
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), wo) + inst.m_InvTransform[2].w
				);
				lr.m_Direction = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), wd),
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), wd),
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), wd)
				);
				lr.m_Tmin = r.m_Tmin;
				lr.m_Tmax = r.m_Tmax;

				unsigned int blasStk[32];
				blasStk[0] = inst.m_Blas.nodeCount - 1;
				int blasStkPtr = 0;

				while (blasStkPtr >= 0) {
					NXB::BVH2::Node blasNode = inst.m_Blas.nodes[blasStk[blasStkPtr--]];
					if (!RayAABB(lr, blasNode.bounds, h.t)) continue;

					if (blasNode.leftChild == INVALID_IDX) {
						uint32_t globalPrim = inst.m_FirstTri + blasNode.rightChild;
						float t, u, v;
						if (RayTriangle(lr, geom.m_BvhTris[globalPrim], h.t, t, u, v)) {
							h.t         = t;
							h.m_U       = u;
							h.m_V       = v;
							h.m_PrimIdx = globalPrim;
							h.m_InstIdx = instIdx;
							h.m_Hit     = true;
						}
					} else {
						blasStk[++blasStkPtr] = blasNode.rightChild;
						blasStk[++blasStkPtr] = blasNode.leftChild;
					}
				}
			} else {
				tlasStk[++tlasStkPtr] = tlasNode.rightChild;
				tlasStk[++tlasStkPtr] = tlasNode.leftChild;
			}
		}

		hits[idx] = h;
	}
}
