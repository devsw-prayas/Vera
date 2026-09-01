#include "Traversal.cuh"
#include <CudaMath.h>

namespace Vera::Core {
	// -- Shared AABB/triangle intersection primitives (raw float3, no Ray struct) --

	__device__ inline bool RayAABBf(
		float3 o, float3 invD, float tmin, float curTmax, const NXB::AABB& aabb) {
		float tminx = fminf((aabb.bMin.x - o.x) * invD.x, (aabb.bMax.x - o.x) * invD.x);
		float tmaxx = fmaxf((aabb.bMin.x - o.x) * invD.x, (aabb.bMax.x - o.x) * invD.x);
		float tminy = fminf((aabb.bMin.y - o.y) * invD.y, (aabb.bMax.y - o.y) * invD.y);
		float tmaxy = fmaxf((aabb.bMin.y - o.y) * invD.y, (aabb.bMax.y - o.y) * invD.y);
		float tminz = fminf((aabb.bMin.z - o.z) * invD.z, (aabb.bMax.z - o.z) * invD.z);
		float tmaxz = fmaxf((aabb.bMin.z - o.z) * invD.z, (aabb.bMax.z - o.z) * invD.z);
		float lo = fmaxf(tmin, fmaxf(tminx, fmaxf(tminy, tminz)));
		float hi = fminf(curTmax, fminf(tmaxx, fminf(tmaxy, tmaxz)));
		return hi >= lo;
	}

	__device__ inline bool RayTrianglef(
		float3 o, float3 d, float tmin, float tmax, const NXB::Triangle& tri,
		float& outT, float& outU, float& outV) {
		float3 edge1 = tri.v1 - tri.v0;
		float3 edge2 = tri.v2 - tri.v0;
		float3 h = cross(d, edge2);
		float  a = dot(edge1, h);
		if (a > -FLT_EPSILON && a < FLT_EPSILON) return false;
		float  f = 1.f / a;
		float3 s = o - tri.v0;
		float  u = f * dot(s, h);
		if (u < 0.f || u > 1.f) return false;
		float3 q = cross(s, edge1);
		float  v = f * dot(d, q);
		if (v < 0.f || u + v > 1.f) return false;
		float  t = f * dot(edge2, q);
		if (t < tmin || t > tmax) return false;
		outT = t; outU = u; outV = v;
		return true;
	}

	// -- Shared BVH walker, templated on anyHit mode ----------------------------

	template <bool AnyHit>
	__device__ WavefrontHitRecord TraverseBVH(
		GeometryBuffers geom,
		float3 origin, float3 direction,
		float tmin, float tmax) {
		WavefrontHitRecord h{};
		h.m_Hit = false;
		h.t = tmax;

		float3 invD = make_float3(1.f / direction.x, 1.f / direction.y, 1.f / direction.z);

		unsigned int tlasStk[32];
		tlasStk[0] = geom.m_Tlas.nodeCount - 1;
		int tlasStkPtr = 0;

		while (tlasStkPtr >= 0) {
			NXB::BVH2::Node tlasNode = geom.m_Tlas.nodes[tlasStk[tlasStkPtr--]];
			if (!RayAABBf(origin, invD, tmin, h.t, tlasNode.bounds)) continue;

			if (tlasNode.leftChild == INVALID_IDX) {
				uint32_t instIdx = tlasNode.rightChild;
				Instance inst = geom.m_DevInstances[instIdx];

				// Transform ray into instance local space
				float3 lo = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), origin) + inst.m_InvTransform[0].w,
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), origin) + inst.m_InvTransform[1].w,
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), origin) + inst.m_InvTransform[2].w);
				float3 ld = make_float3(
					dot(make_float3(inst.m_InvTransform[0].x, inst.m_InvTransform[0].y, inst.m_InvTransform[0].z), direction),
					dot(make_float3(inst.m_InvTransform[1].x, inst.m_InvTransform[1].y, inst.m_InvTransform[1].z), direction),
					dot(make_float3(inst.m_InvTransform[2].x, inst.m_InvTransform[2].y, inst.m_InvTransform[2].z), direction));
				float3 invLd = make_float3(1.f / ld.x, 1.f / ld.y, 1.f / ld.z);

				unsigned int blasStk[32];
				blasStk[0] = inst.m_Blas.nodeCount - 1;
				int blasStkPtr = 0;

				while (blasStkPtr >= 0) {
					NXB::BVH2::Node blasNode = inst.m_Blas.nodes[blasStk[blasStkPtr--]];
					if (!RayAABBf(lo, invLd, tmin, h.t, blasNode.bounds)) continue;

					if (blasNode.leftChild == INVALID_IDX) {
						uint32_t globalPrim = inst.m_FirstTri + blasNode.rightChild;
						float t, u, v;
						if (RayTrianglef(lo, ld, tmin, h.t, geom.m_BvhTris[globalPrim], t, u, v)) {
							if (AnyHit) {
								// Early exit: set m_Hit and return immediately
								h.m_Hit = true;
								return h;
							}
							h.t = t;
							h.m_U = u;
							h.m_V = v;
							h.m_PrimIdx = globalPrim;
							h.m_InstIdx = instIdx;
							h.m_Hit = true;
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

		return h;
	}

	// -- Public API -------------------------------------------------------------

	__device__ WavefrontHitRecord TraverseClosestHit(
		GeometryBuffers geom,
		float3 origin, float3 direction,
		float tmin, float tmax) {
		return TraverseBVH<false>(geom, origin, direction, tmin, tmax);
	}

	__device__ bool TraverseAnyHit(
		GeometryBuffers geom,
		float3 origin, float3 direction,
		float tmin, float tmax) {
		return TraverseBVH<true>(geom, origin, direction, tmin, tmax).m_Hit;
	}

	// -- Wavefront kernel: thin HWSS wrapper ------------------------------------

	__global__ void TraversalKernelWavefront(
		GeometryBuffers geom,
		Spectral::HWSS::RayCoreSoA rayCore,
		WavefrontHitRecord* __restrict__ hits,
		uint32_t rayCount) {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= rayCount) return;

		unsigned char flags = rayCore.flags[idx];

		WavefrontHitRecord h{};
		h.m_Hit = false;
		h.t = FLT_MAX;

		if (flags & Spectral::HWSS::RAY_FLAG_DEAD) { hits[idx] = h; return; }

		hits[idx] = TraverseClosestHit(geom, rayCore.origin[idx], rayCore.direction[idx], 1e-4f, FLT_MAX);
	}
} // namespace Vera::Core