#pragma once
#ifdef VERA_ENABLE_OPTIX

#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>

#include "CoreUtils.h"
#include "RayHWSS.h"

namespace Vera::Core {
	// Shared host/device launch params - must stay layout-identical on both sides. Fields
	// are unpacked from RayCoreSoA rather than embedded because RayCoreSoA's `= nullptr`
	// member initializers make it non-trivially-constructible, which nvcc rejects for the
	// __constant__ `params` in OptixPrograms.cu.
	struct OptixTraversalParams {
		float3* rayOrigin;
		float3* rayDirection;
		unsigned char* rayFlags;
		WavefrontHitRecord* hits;
		OptixTraversableHandle  handle;
	};

	// Host-side OptiX state. Built once (InitOptixTraversal), reused every bounce
	// (LaunchOptixTraversal), freed once (DestroyOptixTraversal).
	struct OptixTraversalContext {
		OptixDeviceContext context = nullptr;
		OptixModule        module = nullptr;
		OptixProgramGroup  raygenPG = nullptr;
		OptixProgramGroup  missPG = nullptr;
		OptixProgramGroup  hitgroupPG = nullptr;
		OptixPipeline      pipeline = nullptr;

		CUdeviceptr gasOutputBuffer = 0;
		CUdeviceptr iasOutputBuffer = 0;
		OptixTraversableHandle traversableHandle = 0;

		CUdeviceptr sbtRaygenRecord = 0;
		CUdeviceptr sbtMissRecord = 0;
		CUdeviceptr sbtHitgroupRecord = 0;
		OptixShaderBindingTable sbt = {};

		CUdeviceptr d_params = 0; // device copy of OptixTraversalParams, updated + reused each launch
	};

	// Builds a GAS + 1-instance IAS from geom, then the module/pipeline/SBT from the
	// precompiled OptixPrograms PTX. (Single-instance only - see OptixTraversal.cpp.)
	OptixTraversalContext InitOptixTraversal(const GeometryBuffers& geom);
	void DestroyOptixTraversal(OptixTraversalContext& ctx);

	// Traces `rayCount` rays into `hits` - same WavefrontHitRecord layout as
	// Traversal.cuh's TraversalKernelWavefront, so downstream is backend-agnostic.
	void LaunchOptixTraversal(
		OptixTraversalContext& ctx,
		Spectral::HWSS::RayCoreSoA rayCore,
		WavefrontHitRecord* hits,
		uint32_t rayCount,
		cudaStream_t stream);
}

#endif // VERA_ENABLE_OPTIX
