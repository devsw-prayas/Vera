#pragma once
#include "LightBVHHWSS.h"
#include <CoreUtils.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace BSPT::Spectral::HWSS {

__device__ inline float lightNodeScore(const LightBVHNode& node, const float3& p) {
    float cx = (node.bMin.x + node.bMax.x) * 0.5f;
    float cy = (node.bMin.y + node.bMax.y) * 0.5f;
    float cz = (node.bMin.z + node.bMax.z) * 0.5f;
    float dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
    float dist2 = fmaxf(dx*dx + dy*dy + dz*dz, 1e-4f);
    float dist  = sqrtf(dist2);
    float cosA  = (node.coneAxis.x*dx + node.coneAxis.y*dy + node.coneAxis.z*dz) / dist;
    float cosEff  = fmaxf(cosA - node.sinConeAngle, 0.f);
    // Fallback: never give zero score so traversal never hard-excludes a node
    return node.flux * (cosEff + 1e-3f) / dist2;
}

// Samples a point on a light using the BVH. Returns the sampled position, normal,
// material (evaluate emission via EvalEmission), and area-measure PDF. Caller does
// the shadow test separately.
// uTraversal is rescaled at each split so a single float drives the whole traversal.
__device__ inline LightSample SampleLightBVH(
    const LightBVH&              bvh,
    const Core::GeometryBuffers& geom,
    const float3&                shadingPt,
    float                        uTraversal,
    float                        uTri,
    float2                       uBary)
{
    LightSample ls{};
    ls.pdf = 0.f;

    if (bvh.nodeCount == 0 || bvh.lightCount == 0) return ls;

    // Stochastic BVH traversal — descend until leaf
    uint32_t nodeIdx = 0;
    float traversalPdf = 1.f;

    while (bvh.nodes[nodeIdx].leftChild != 0xFFFFFFFFu) {
        const LightBVHNode& node = bvh.nodes[nodeIdx];
        float scoreL = lightNodeScore(bvh.nodes[node.leftChild],  shadingPt);
        float scoreR = lightNodeScore(bvh.nodes[node.rightChild], shadingPt);
        float total  = scoreL + scoreR;
        float pL     = (total > 0.f) ? scoreL / total : 0.5f;

        if (uTraversal < pL) {
            traversalPdf *= pL;
            uTraversal   /= pL;
            nodeIdx = node.leftChild;
        } else {
            traversalPdf *= (1.f - pL);
            uTraversal    = (uTraversal - pL) / fmaxf(1.f - pL, 1e-7f);
            nodeIdx = node.rightChild;
        }
    }

    const LightEntry& light = bvh.lights[bvh.nodes[nodeIdx].lightIdx];

    // Pick triangle from CDF (linear scan — light triCount is small)
    uint32_t triLocal = light.triCount - 1;
    for (uint32_t k = 0; k < light.triCount; ++k) {
        if (uTri <= bvh.triAreaCDF[light.firstTri + k]) { triLocal = k; break; }
    }

    // Look up world-space vertices
    uint32_t primIdx = light.firstGlobalTri + triLocal;
    uint32_t i0 = geom.m_DevIndexBuffer[primIdx * 3 + 0];
    uint32_t i1 = geom.m_DevIndexBuffer[primIdx * 3 + 1];
    uint32_t i2 = geom.m_DevIndexBuffer[primIdx * 3 + 2];
    float3 v0 = geom.m_DevVertexPos[i0];
    float3 v1 = geom.m_DevVertexPos[i1];
    float3 v2 = geom.m_DevVertexPos[i2];

    // Uniform point on triangle
    float su = sqrtf(uBary.x);
    float b0 = 1.f - su;
    float b1 = su * uBary.y;
    float b2 = 1.f - b0 - b1;
    ls.position = make_float3(
        b0*v0.x + b1*v1.x + b2*v2.x,
        b0*v0.y + b1*v1.y + b2*v2.y,
        b0*v0.z + b1*v1.z + b2*v2.z);

    float3 n0 = geom.m_DevVertexNorms[i0];
    float3 n1 = geom.m_DevVertexNorms[i1];
    float3 n2 = geom.m_DevVertexNorms[i2];
    float3 n  = make_float3(b0*n0.x+b1*n1.x+b2*n2.x, b0*n0.y+b1*n1.y+b2*n2.y, b0*n0.z+b1*n1.z+b2*n2.z);
    float  nLen = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    ls.normal   = (nLen > 1e-7f) ? make_float3(n.x/nLen, n.y/nLen, n.z/nLen) : make_float3(0.f, 1.f, 0.f);

    ls.materialId = light.materialId;
    ls.lightIdx   = bvh.nodes[nodeIdx].lightIdx;
    // Area-measure PDF: traversal probability / totalArea
    ls.pdf      = traversalPdf / light.totalArea;

    return ls;
}

// Returns true if the subtree rooted at nodeIdx contains targetLight as a leaf.
// Light counts are small (typically ≤16), so this DFS is cheap.
__device__ inline bool lightInSubtree(const LightBVH& bvh, uint32_t nodeIdx, uint32_t targetLight) {
    uint32_t stk[32];
    int stkPtr = 0;
    stk[stkPtr++] = nodeIdx;
    while (stkPtr > 0) {
        uint32_t n = stk[--stkPtr];
        if (bvh.nodes[n].leftChild == 0xFFFFFFFFu) {
            if (bvh.nodes[n].lightIdx == targetLight) return true;
        } else {
            stk[stkPtr++] = bvh.nodes[n].leftChild;
            stk[stkPtr++] = bvh.nodes[n].rightChild;
        }
    }
    return false;
}

// Computes the area-measure PDF for a BSDF ray that happened to hit an emissive triangle.
// Replays the stochastic BVH traversal from shadingPt using the same score function as
// SampleLightBVH, accumulating exact branching probabilities instead of the flux-ratio
// approximation (which is wrong for multi-light scenes with distance/angle variation).
__device__ inline float EmissiveHitPdf(const LightBVH& bvh, uint32_t primIdx, const float3& shadingPt) {
    if (primIdx >= bvh.totalPrimCount || bvh.lightCount == 0 || bvh.nodeCount == 0) return 0.f;
    uint32_t targetLight = bvh.primToLight[primIdx];
    if (targetLight == 0xFFFFFFFFu) return 0.f;

    uint32_t nodeIdx      = 0;
    float    traversalPdf = 1.f;

    while (bvh.nodes[nodeIdx].leftChild != 0xFFFFFFFFu) {
        const LightBVHNode& node = bvh.nodes[nodeIdx];
        uint32_t L = node.leftChild;
        uint32_t R = node.rightChild;

        float scoreL = lightNodeScore(bvh.nodes[L], shadingPt);
        float scoreR = lightNodeScore(bvh.nodes[R], shadingPt);
        float total  = scoreL + scoreR;
        float pL     = (total > 0.f) ? scoreL / total : 0.5f;

        if (lightInSubtree(bvh, L, targetLight)) {
            traversalPdf *= pL;
            nodeIdx = L;
        } else {
            traversalPdf *= (1.f - pL);
            nodeIdx = R;
        }
    }

    return traversalPdf / bvh.lights[targetLight].totalArea;
}

}
