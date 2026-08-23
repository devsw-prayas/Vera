#pragma once
#ifdef VERA_ENABLE_OPTIX

#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>

#include "CoreUtils.h"
#include "RayHWSS.h"

namespace Vera::Core {
	// Shared between host (OptixTraversal.cpp) and device (OptixPrograms.cu) code -- must stay
	// layout-identical on both sides, exactly like every other kernel-launch-params struct in
	// this codebase (e.g. FrameBufferHWSS, GeometryBuffers).
	//
	// Fields are unpacked from RayCoreSoA rather than embedding it directly: RayCoreSoA's
	// pointer members carry in-class default initializers (`= nullptr`), which makes the type
	// non-trivially-default-constructible -- and nvcc rejects any such type for a __constant__
	// global variable ("dynamic initialization is not supported"), which is exactly how
	// OptixPrograms.cu's `params` needs to be declared.
	struct OptixTraversalParams {
		float3*                 rayOrigin;
		float3*                 rayDirection;
		unsigned char*          rayFlags;
		WavefrontHitRecord*     hits;
		OptixTraversableHandle  handle;
	};

	// Host-side OptiX state: context, module, program groups, pipeline, SBT, GAS/IAS device
	// buffers, and the device-side launch-params buffer. Built once per render
	// (InitOptixTraversal), reused every bounce (LaunchOptixTraversal), freed once
	// (DestroyOptixTraversal) -- mirrors RayCompactor/MaterialSorter's Init/.../Destroy shape.
	struct OptixTraversalContext {
		OptixDeviceContext context  = nullptr;
		OptixModule        module   = nullptr;
		OptixProgramGroup  raygenPG   = nullptr;
		OptixProgramGroup  missPG     = nullptr;
		OptixProgramGroup  hitgroupPG = nullptr;
		OptixPipeline      pipeline = nullptr;

		CUdeviceptr gasOutputBuffer = 0;
		CUdeviceptr iasOutputBuffer = 0;
		OptixTraversableHandle traversableHandle = 0;

		CUdeviceptr sbtRaygenRecord   = 0;
		CUdeviceptr sbtMissRecord     = 0;
		CUdeviceptr sbtHitgroupRecord = 0;
		OptixShaderBindingTable sbt = {};

		CUdeviceptr d_params = 0; // device copy of OptixTraversalParams, updated + reused each launch
	};

	// Builds a GAS from geom's triangle buffer and a 1-instance IAS wrapping it with that
	// instance's transform (matches the current single-instance scene structure -- see
	// OptixTraversal.cpp for the known limitation around multi-instance scenes), then sets up
	// the OptiX module/pipeline/SBT from the precompiled OptixPrograms PTX.
	OptixTraversalContext InitOptixTraversal(const GeometryBuffers& geom);
	void DestroyOptixTraversal(OptixTraversalContext& ctx);

	// Traces `rayCount` rays from rayCore through the OptiX-accelerated scene, writing results
	// into `hits` -- same WavefrontHitRecord layout and semantics as Traversal.cuh's
	// TraversalKernelWavefront, so callers downstream don't need to know which backend produced
	// the hit.
	void LaunchOptixTraversal(
		OptixTraversalContext& ctx,
		Spectral::HWSS::RayCoreSoA rayCore,
		WavefrontHitRecord* hits,
		uint32_t rayCount,
		cudaStream_t stream);
}

#endif // VERA_ENABLE_OPTIX
