#include "LightBVHBuilderHWSS.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

namespace Vera::Spectral::HWSS {
	// flux weighting only; actual radiance is per-lane at shading time
	static constexpr float kFluxLambda = 555.f;

	// Unnormalized edge1 x edge2; area and normal both derive from it.
	static float3 triCross(float3 v0, float3 v1, float3 v2) {
		float ex = v1.x-v0.x, ey = v1.y-v0.y, ez = v1.z-v0.z;
		float fx = v2.x-v0.x, fy = v2.y-v0.y, fz = v2.z-v0.z;
		return float3{ ey*fz - ez*fy, ez*fx - ex*fz, ex*fy - ey*fx };
	}

	static float triAreaFromCross(float3 c) {
		return 0.5f * sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);
	}

	static float3 triNormalFromCross(float3 c) {
		float len = sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);
		return (len > 1e-9f) ? float3{c.x/len, c.y/len, c.z/len} : float3{0.f, 1.f, 0.f};
	}

	static float triArea(float3 v0, float3 v1, float3 v2) {
		return triAreaFromCross(triCross(v0, v1, v2));
	}

	static float3 triNormal(float3 v0, float3 v1, float3 v2) {
		return triNormalFromCross(triCross(v0, v1, v2));
	}

	struct BuildLight {
		LightEntry entry;
		float3     bMin, bMax, centroid;
		uint32_t   lightIdx;
	};

	struct CPUNode {
		float3   bMin, bMax;
		float3   coneAxis;
		float    cosConeAngle;
		float    flux;
		int      left, right;
		uint32_t lightIdx;
	};

	static int buildRecursive(std::vector<BuildLight>& lights, std::vector<CPUNode>& nodes, int start, int end, uint32_t pathBits = 0, uint8_t depth = 0) {
		CPUNode node{};
		node.bMin = {FLT_MAX, FLT_MAX, FLT_MAX};
		node.bMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
		node.flux = 0.f;
		float3 avgAxis = {0.f, 0.f, 0.f};

		for (int i = start; i < end; ++i) {
			const BuildLight& bl = lights[i];
			node.bMin.x = fminf(node.bMin.x, bl.bMin.x); node.bMin.y = fminf(node.bMin.y, bl.bMin.y); node.bMin.z = fminf(node.bMin.z, bl.bMin.z);
			node.bMax.x = fmaxf(node.bMax.x, bl.bMax.x); node.bMax.y = fmaxf(node.bMax.y, bl.bMax.y); node.bMax.z = fmaxf(node.bMax.z, bl.bMax.z);
			node.flux += bl.entry.flux;
			avgAxis.x += bl.entry.coneAxis.x * bl.entry.flux;
			avgAxis.y += bl.entry.coneAxis.y * bl.entry.flux;
			avgAxis.z += bl.entry.coneAxis.z * bl.entry.flux;
		}

		float axLen = sqrtf(avgAxis.x*avgAxis.x + avgAxis.y*avgAxis.y + avgAxis.z*avgAxis.z);
		node.coneAxis = (axLen > 1e-7f) ? float3{avgAxis.x/axLen, avgAxis.y/axLen, avgAxis.z/axLen} : float3{0.f, 1.f, 0.f};

		float maxAngle = 0.f;
		for (int i = start; i < end; ++i) {
			const LightEntry& le = lights[i].entry;
			float d = node.coneAxis.x*le.coneAxis.x + node.coneAxis.y*le.coneAxis.y + node.coneAxis.z*le.coneAxis.z;
			float childHalf = acosf(fmaxf(fminf(le.cosConeAngle, 1.f), -1.f));
			float toChild   = acosf(fmaxf(fminf(d, 1.f), -1.f));
			float total     = fminf(childHalf + toChild, 3.14159265f);
			if (total > maxAngle) maxAngle = total;
		}
		node.cosConeAngle = cosf(maxAngle);

		if (end - start == 1) {
			node.left = node.right = -1;
			node.lightIdx = lights[start].lightIdx;
			lights[start].entry.pathBits  = pathBits;
			lights[start].entry.pathDepth = depth;
			int idx = (int)nodes.size();
			nodes.push_back(node);
			return idx;
		}

		float dx = node.bMax.x - node.bMin.x;
		float dy = node.bMax.y - node.bMin.y;
		float dz = node.bMax.z - node.bMin.z;
		int axis = (dx >= dy && dx >= dz) ? 0 : (dy >= dz) ? 1 : 2;
		int mid  = (start + end) / 2;
		std::nth_element(lights.begin() + start, lights.begin() + mid, lights.begin() + end,
			[axis](const BuildLight& a, const BuildLight& b) {
				return (&a.centroid.x)[axis] < (&b.centroid.x)[axis];
			});

		int nodeIdx = (int)nodes.size();
		nodes.push_back(node);
		int left  = buildRecursive(lights, nodes, start, mid, pathBits, (uint8_t)(depth + 1));
		int right = buildRecursive(lights, nodes, mid, end, pathBits | (1u << depth), (uint8_t)(depth + 1));
		nodes[nodeIdx].left  = left;
		nodes[nodeIdx].right = right;
		return nodeIdx;
	}

	LightBVH BuildLightBVH(
		const float3*   hostVertices,
		const uint32_t* hostIndices,
		const uint16_t* hostTriMatIDs,
		uint32_t        totalTriCount,
		const Material* hostMaterials)
	{
		std::vector<BuildLight> buildLights;
		std::vector<float>      allCDFs;

		uint32_t t = 0;
		while (t < totalTriCount) {
			uint16_t matId = hostTriMatIDs[t];
			if (hostMaterials[matId].type != MaterialType::Emissive) { ++t; continue; }

			const Material& mat = hostMaterials[matId];
			uint32_t groupStart = t;

			BuildLight bl{};
			bl.entry.materialId = matId;
			bl.entry.firstTri = (uint32_t)allCDFs.size();
			bl.bMin = {FLT_MAX, FLT_MAX, FLT_MAX};
			bl.bMax = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

			float3 avgNormal = {0.f, 0.f, 0.f};
			float  cumArea   = 0.f;
			std::vector<float> groupCDFs;

			while (t < totalTriCount && hostTriMatIDs[t] == matId) {
				uint32_t i0 = hostIndices[t * 3 + 0];
				uint32_t i1 = hostIndices[t * 3 + 1];
				uint32_t i2 = hostIndices[t * 3 + 2];
				float3 v0 = hostVertices[i0], v1 = hostVertices[i1], v2 = hostVertices[i2];

				float3 cross_ = triCross(v0, v1, v2);
				float  area = triAreaFromCross(cross_);
				float3 n    = triNormalFromCross(cross_);
				cumArea    += area;
				avgNormal.x += n.x * area;
				avgNormal.y += n.y * area;
				avgNormal.z += n.z * area;
				groupCDFs.push_back(cumArea);

				bl.bMin.x = fminf(bl.bMin.x, fminf(v0.x, fminf(v1.x, v2.x)));
				bl.bMin.y = fminf(bl.bMin.y, fminf(v0.y, fminf(v1.y, v2.y)));
				bl.bMin.z = fminf(bl.bMin.z, fminf(v0.z, fminf(v1.z, v2.z)));
				bl.bMax.x = fmaxf(bl.bMax.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
				bl.bMax.y = fmaxf(bl.bMax.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
				bl.bMax.z = fmaxf(bl.bMax.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
				++t;
			}

			for (float& c : groupCDFs) c /= cumArea;
			for (float  c : groupCDFs) allCDFs.push_back(c);

			float axLen = sqrtf(avgNormal.x*avgNormal.x + avgNormal.y*avgNormal.y + avgNormal.z*avgNormal.z);
			float3 coneAxis = (axLen > 1e-7f) ? float3{avgNormal.x/axLen, avgNormal.y/axLen, avgNormal.z/axLen} : float3{0.f, 1.f, 0.f};

			float maxAngle = 0.f;
			for (uint32_t k = groupStart; k < t; ++k) {
				float3 n = triNormal(hostVertices[hostIndices[k*3]], hostVertices[hostIndices[k*3+1]], hostVertices[hostIndices[k*3+2]]);
				float  d = coneAxis.x*n.x + coneAxis.y*n.y + coneAxis.z*n.z;
				float  a = acosf(fmaxf(fminf(d, 1.f), -1.f));
				if (a > maxAngle) maxAngle = a;
			}

			bl.entry.totalArea     = cumArea;
			bl.entry.flux          = EvalEmission(mat, kFluxLambda) * cumArea;
			bl.entry.coneAxis      = coneAxis;
			bl.entry.cosConeAngle  = cosf(maxAngle);
			bl.entry.firstGlobalTri = groupStart;
			bl.entry.triCount      = t - groupStart;
			bl.centroid = {(bl.bMin.x+bl.bMax.x)*0.5f, (bl.bMin.y+bl.bMax.y)*0.5f, (bl.bMin.z+bl.bMax.z)*0.5f};
			bl.lightIdx = (uint32_t)buildLights.size();
			buildLights.push_back(bl);
		}

		std::vector<CPUNode> cpuNodes;
		if (!buildLights.empty())
			buildRecursive(buildLights, cpuNodes, 0, (int)buildLights.size());

		std::vector<LightEntry> lightEntries(buildLights.size());
		for (size_t i = 0; i < buildLights.size(); ++i)
			lightEntries[i] = buildLights[i].entry;

		std::vector<LightBVHNode> gpuNodes(cpuNodes.size());
		for (size_t i = 0; i < cpuNodes.size(); ++i) {
			const CPUNode& c = cpuNodes[i];
			LightBVHNode&  g = gpuNodes[i];
			g.bMin         = c.bMin;
			g.bMax         = c.bMax;
			g.coneAxis     = c.coneAxis;
			g.cosConeAngle = c.cosConeAngle;
			g.sinConeAngle = sqrtf(fmaxf(1.f - c.cosConeAngle * c.cosConeAngle, 0.f));
			g.flux         = c.flux;
			g.leftChild    = (c.left  == -1) ? 0xFFFFFFFFu : (uint32_t)c.left;
			g.rightChild   = (c.right == -1) ? 0xFFFFFFFFu : (uint32_t)c.right;
			g.lightIdx     = c.lightIdx;
		}

		std::vector<uint32_t> primToLight(totalTriCount, 0xFFFFFFFFu);
		for (const BuildLight& bl : buildLights) {
			for (uint32_t k = 0; k < bl.entry.triCount; ++k)
				primToLight[bl.entry.firstGlobalTri + k] = bl.lightIdx;
		}

		LightBVH bvh{};
		bvh.nodeCount      = (uint32_t)gpuNodes.size();
		bvh.lightCount     = (uint32_t)lightEntries.size();
		bvh.totalPrimCount = totalTriCount;

		if (bvh.nodeCount > 0) {
			cudaMalloc(&bvh.nodes,       bvh.nodeCount      * sizeof(LightBVHNode));
			cudaMalloc(&bvh.lights,      bvh.lightCount     * sizeof(LightEntry));
			cudaMalloc(&bvh.triAreaCDF,  allCDFs.size()     * sizeof(float));
			cudaMalloc(&bvh.primToLight, bvh.totalPrimCount * sizeof(uint32_t));
			cudaMemcpy(bvh.nodes,       gpuNodes.data(),     bvh.nodeCount      * sizeof(LightBVHNode), cudaMemcpyHostToDevice);
			cudaMemcpy(bvh.lights,      lightEntries.data(), bvh.lightCount     * sizeof(LightEntry),   cudaMemcpyHostToDevice);
			cudaMemcpy(bvh.triAreaCDF,  allCDFs.data(),      allCDFs.size()     * sizeof(float),        cudaMemcpyHostToDevice);
			cudaMemcpy(bvh.primToLight, primToLight.data(),  bvh.totalPrimCount * sizeof(uint32_t),     cudaMemcpyHostToDevice);
		}

		return bvh;
	}

	void FreeLightBVH(LightBVH& bvh) {
		if (bvh.nodes)       cudaFree(bvh.nodes);
		if (bvh.lights)      cudaFree(bvh.lights);
		if (bvh.triAreaCDF)  cudaFree(bvh.triAreaCDF);
		if (bvh.primToLight) cudaFree(bvh.primToLight);
		bvh = {};
	}
}
