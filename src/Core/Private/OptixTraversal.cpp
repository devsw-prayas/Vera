#ifdef VERA_ENABLE_OPTIX

#include "OptixTraversal.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

// Minimal local error checking, matching this codebase's existing style (Main.cu: print +
// bail, no exception hierarchy) rather than pulling in the OptiX SDK samples' sutil/Exception.h,
// which drags in GL/GLFW/imgui dependencies this project has no other use for.
#define OPTIX_CHECK(call)                                                                    \
	do {                                                                                      \
		OptixResult res = call;                                                               \
		if (res != OPTIX_SUCCESS) {                                                           \
			fprintf(stderr, "OptiX error %d at %s:%d: %s\n", (int)res, __FILE__, __LINE__, #call); \
			std::abort();                                                                     \
		}                                                                                      \
	} while (0)

#define CUDA_CHECK(call)                                                                     \
	do {                                                                                      \
		cudaError_t err = call;                                                               \
		if (err != cudaSuccess) {                                                             \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
			std::abort();                                                                     \
		}                                                                                      \
	} while (0)

namespace Vera::Core {
	// No per-record data -- everything the device programs need lives in
	// OptixTraversalParams (the pipeline launch params), not the SBT.
	struct EmptySbtRecord {
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	};

	static void OptixLogCallback(unsigned int level, const char* tag, const char* message, void*) {
		fprintf(stderr, "[OptiX][%u][%s]: %s\n", level, tag, message);
	}

	static std::string ReadFile(const char* path) {
		std::ifstream f(path, std::ios::binary);
		if (!f) {
			fprintf(stderr, "Vera: failed to open OptiX PTX file '%s' -- was the VeraOptixPTX build step run?\n", path);
			std::abort();
		}
		std::ostringstream ss;
		ss << f.rdbuf();
		return ss.str();
	}

	OptixTraversalContext InitOptixTraversal(const GeometryBuffers& geom) {
		OptixTraversalContext ctx{};

		// ── Context ──────────────────────────────────────────────────────────
		CUDA_CHECK(cudaFree(0)); // ensure a CUDA context exists before optixInit
		OPTIX_CHECK(optixInit());

		OptixDeviceContextOptions ctxOptions = {};
		ctxOptions.logCallbackFunction = &OptixLogCallback;
		ctxOptions.logCallbackLevel    = 3;
		OPTIX_CHECK(optixDeviceContextCreate(0, &ctxOptions, &ctx.context));

		// ── GAS: built directly from geom's flat (non-indexed) triangle soup. NXB::Triangle is
		// exactly {float3 v0,v1,v2} with no padding (verified against external/NexusBVH/Public/
		// Triangle.h), so it's memory-layout-identical to a flat float3[3*triCount] array --
		// OptiX can read it directly as OPTIX_VERTEX_FORMAT_FLOAT3, no repacking needed. ──
		OptixTraversableHandle gasHandle = 0;
		{
			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

			CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(geom.m_BvhTris);
			uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

			OptixBuildInput triangleInput = {};
			triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
			triangleInput.triangleArray.numVertices   = geom.m_TrisCount * 3;
			triangleInput.triangleArray.vertexBuffers = &d_vertices;
			triangleInput.triangleArray.flags         = triangleInputFlags;
			triangleInput.triangleArray.numSbtRecords  = 1;

			OptixAccelBufferSizes gasBufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.context, &accelOptions, &triangleInput, 1, &gasBufferSizes));

			CUdeviceptr d_tempBuffer;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.gasOutputBuffer), gasBufferSizes.outputSizeInBytes));

			OPTIX_CHECK(optixAccelBuild(
				ctx.context, 0, &accelOptions, &triangleInput, 1,
				d_tempBuffer, gasBufferSizes.tempSizeInBytes,
				ctx.gasOutputBuffer, gasBufferSizes.outputSizeInBytes,
				&gasHandle, nullptr, 0));

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
		}

		// ── IAS: one instance wrapping the GAS, using the scene's existing instance transform.
		// KNOWN LIMITATION: the current scene upload path (Scene.cu's UploadScene) only ever
		// creates a single instance with all geometry sharing one triangle buffer, which is
		// exactly what's built above -- a real multi-instance scene (distinct meshes, multiple
		// transforms) would need one GAS per distinct mesh instead of one GAS for everything,
		// which this first pass does not attempt. ──
		{
			Instance hInst{};
			CUDA_CHECK(cudaMemcpy(&hInst, geom.m_DevInstances, sizeof(Instance), cudaMemcpyDeviceToHost));

			OptixInstance optixInst = {};
			// Instance::m_Transform is 3 float4 rows -- exactly OptixInstance::transform's
			// 3x4 row-major layout, safe to copy directly.
			memcpy(optixInst.transform, &hInst.m_Transform[0], sizeof(float) * 12);
			optixInst.instanceId        = 0;
			optixInst.visibilityMask    = 255;
			optixInst.sbtOffset         = 0;
			optixInst.flags             = OPTIX_INSTANCE_FLAG_NONE;
			optixInst.traversableHandle = gasHandle;

			CUdeviceptr d_instances;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(OptixInstance)));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), &optixInst, sizeof(OptixInstance), cudaMemcpyHostToDevice));

			OptixBuildInput instanceInput = {};
			instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			instanceInput.instanceArray.instances    = d_instances;
			instanceInput.instanceArray.numInstances = 1;

			OptixAccelBuildOptions accelOptions = {};
			accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

			OptixAccelBufferSizes iasBufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx.context, &accelOptions, &instanceInput, 1, &iasBufferSizes));

			CUdeviceptr d_tempBuffer;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), iasBufferSizes.tempSizeInBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.iasOutputBuffer), iasBufferSizes.outputSizeInBytes));

			OPTIX_CHECK(optixAccelBuild(
				ctx.context, 0, &accelOptions, &instanceInput, 1,
				d_tempBuffer, iasBufferSizes.tempSizeInBytes,
				ctx.iasOutputBuffer, iasBufferSizes.outputSizeInBytes,
				&ctx.traversableHandle, nullptr, 0));

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
		}

		// ── Module + pipeline ────────────────────────────────────────────────
		OptixPipelineCompileOptions pipelineCompileOptions = {};
		{
			OptixModuleCompileOptions moduleCompileOptions = {};

			pipelineCompileOptions.usesMotionBlur        = false;
			pipelineCompileOptions.traversableGraphFlags  = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
			pipelineCompileOptions.numPayloadValues       = 0;
			pipelineCompileOptions.numAttributeValues     = 2;
			pipelineCompileOptions.exceptionFlags         = OPTIX_EXCEPTION_FLAG_NONE;
			pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
			pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

			std::string ptx = ReadFile(VERA_OPTIX_PTX_PATH);

			char   log[2048];
			size_t logSize = sizeof(log);
			OPTIX_CHECK(optixModuleCreate(
				ctx.context, &moduleCompileOptions, &pipelineCompileOptions,
				ptx.c_str(), ptx.size(), log, &logSize, &ctx.module));
		}

		{
			OptixProgramGroupOptions pgOptions = {};

			OptixProgramGroupDesc raygenDesc = {};
			raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygenDesc.raygen.module            = ctx.module;
			raygenDesc.raygen.entryFunctionName = "__raygen__vera";
			char log[2048]; size_t logSize = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(ctx.context, &raygenDesc, 1, &pgOptions, log, &logSize, &ctx.raygenPG));

			OptixProgramGroupDesc missDesc = {};
			missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			missDesc.miss.module            = ctx.module;
			missDesc.miss.entryFunctionName = "__miss__vera";
			logSize = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(ctx.context, &missDesc, 1, &pgOptions, log, &logSize, &ctx.missPG));

			OptixProgramGroupDesc hitgroupDesc = {};
			hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroupDesc.hitgroup.moduleCH            = ctx.module;
			hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__vera";
			logSize = sizeof(log);
			OPTIX_CHECK(optixProgramGroupCreate(ctx.context, &hitgroupDesc, 1, &pgOptions, log, &logSize, &ctx.hitgroupPG));
		}

		{
			const uint32_t maxTraceDepth = 1; // raygen -> optixTrace -> {miss,closesthit}, no recursive tracing
			OptixProgramGroup programGroups[] = { ctx.raygenPG, ctx.missPG, ctx.hitgroupPG };

			OptixPipelineLinkOptions linkOptions = {};
			linkOptions.maxTraceDepth = maxTraceDepth;

			char log[2048]; size_t logSize = sizeof(log);
			OPTIX_CHECK(optixPipelineCreate(
				ctx.context, &pipelineCompileOptions, &linkOptions,
				programGroups, 3, log, &logSize, &ctx.pipeline));

			OptixStackSizes stackSizes = {};
			for (auto pg : programGroups)
				OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, ctx.pipeline));

			uint32_t dcStackFromTraversal, dcStackFromState, ccStackSize;
			OPTIX_CHECK(optixUtilComputeStackSizes(
				&stackSizes, maxTraceDepth, 0, 0,
				&dcStackFromTraversal, &dcStackFromState, &ccStackSize));
			OPTIX_CHECK(optixPipelineSetStackSize(
				ctx.pipeline, dcStackFromTraversal, dcStackFromState, ccStackSize,
				2)); // maxTraversableDepth: IAS -> GAS is a 2-level graph
		}

		// ── SBT: no per-record data, everything lives in the launch params ─────
		{
			using EmptyRecord = EmptySbtRecord;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.sbtRaygenRecord), sizeof(EmptyRecord)));
			EmptyRecord rgSbt{};
			OPTIX_CHECK(optixSbtRecordPackHeader(ctx.raygenPG, &rgSbt));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ctx.sbtRaygenRecord), &rgSbt, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.sbtMissRecord), sizeof(EmptyRecord)));
			EmptyRecord msSbt{};
			OPTIX_CHECK(optixSbtRecordPackHeader(ctx.missPG, &msSbt));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ctx.sbtMissRecord), &msSbt, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.sbtHitgroupRecord), sizeof(EmptyRecord)));
			EmptyRecord hgSbt{};
			OPTIX_CHECK(optixSbtRecordPackHeader(ctx.hitgroupPG, &hgSbt));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ctx.sbtHitgroupRecord), &hgSbt, sizeof(EmptyRecord), cudaMemcpyHostToDevice));

			ctx.sbt.raygenRecord                = ctx.sbtRaygenRecord;
			ctx.sbt.missRecordBase              = ctx.sbtMissRecord;
			ctx.sbt.missRecordStrideInBytes     = sizeof(EmptyRecord);
			ctx.sbt.missRecordCount             = 1;
			ctx.sbt.hitgroupRecordBase          = ctx.sbtHitgroupRecord;
			ctx.sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
			ctx.sbt.hitgroupRecordCount         = 1;
		}

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_params), sizeof(OptixTraversalParams)));

		return ctx;
	}

	void LaunchOptixTraversal(
		OptixTraversalContext& ctx,
		Spectral::HWSS::RayCoreSoA rayCore,
		WavefrontHitRecord* hits,
		uint32_t rayCount,
		cudaStream_t stream)
	{
		if (rayCount == 0) return;

		OptixTraversalParams params{};
		params.rayOrigin    = rayCore.origin;
		params.rayDirection = rayCore.direction;
		params.rayFlags     = rayCore.flags;
		params.hits         = hits;
		params.handle       = ctx.traversableHandle;

		CUDA_CHECK(cudaMemcpyAsync(
			reinterpret_cast<void*>(ctx.d_params), &params, sizeof(OptixTraversalParams),
			cudaMemcpyHostToDevice, stream));

		OPTIX_CHECK(optixLaunch(
			ctx.pipeline, stream, ctx.d_params, sizeof(OptixTraversalParams),
			&ctx.sbt, rayCount, 1, 1));
	}

	void DestroyOptixTraversal(OptixTraversalContext& ctx) {
		if (ctx.d_params)          cudaFree(reinterpret_cast<void*>(ctx.d_params));
		if (ctx.sbtRaygenRecord)   cudaFree(reinterpret_cast<void*>(ctx.sbtRaygenRecord));
		if (ctx.sbtMissRecord)     cudaFree(reinterpret_cast<void*>(ctx.sbtMissRecord));
		if (ctx.sbtHitgroupRecord) cudaFree(reinterpret_cast<void*>(ctx.sbtHitgroupRecord));
		if (ctx.gasOutputBuffer)   cudaFree(reinterpret_cast<void*>(ctx.gasOutputBuffer));
		if (ctx.iasOutputBuffer)   cudaFree(reinterpret_cast<void*>(ctx.iasOutputBuffer));

		if (ctx.pipeline)   optixPipelineDestroy(ctx.pipeline);
		if (ctx.hitgroupPG) optixProgramGroupDestroy(ctx.hitgroupPG);
		if (ctx.missPG)     optixProgramGroupDestroy(ctx.missPG);
		if (ctx.raygenPG)   optixProgramGroupDestroy(ctx.raygenPG);
		if (ctx.module)     optixModuleDestroy(ctx.module);
		if (ctx.context)    optixDeviceContextDestroy(ctx.context);

		ctx = OptixTraversalContext{};
	}
}

#endif // VERA_ENABLE_OPTIX
