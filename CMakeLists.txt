cmake_minimum_required(VERSION 3.20)
project(Vera LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB_RECURSE ALL_HEADERS     CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE ALL_CXX_SRC     CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE ALL_INLINE      CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl")
file(GLOB_RECURSE ALL_CUDA_SRC    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE ALL_CUDA_HDRS   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")

# ── Optional OptiX hardware-traversal pathway ───────────────────────────────
# OptixPrograms.cu is NOT a normal wavefront kernel — it holds __raygen__/__miss__/
# __closesthit__ entry points that OptiX loads as PTX at runtime via optixModuleCreate,
# not something the main -rdc=true device-link unit can contain. It's compiled separately
# below and excluded from ALL_CUDA_SRC. The whole feature is optional and additive: on a
# machine without the OptiX SDK, VERA_ENABLE_OPTIX stays off and the software-only
# TraversalKernelWavefront path (existing behavior) is completely unaffected.
set(VERA_OPTIX_PROGRAMS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Private/OptixPrograms.cu")
list(REMOVE_ITEM ALL_CUDA_SRC "${VERA_OPTIX_PROGRAMS_SRC}")

find_path(VERA_OPTIX_INCLUDE_DIR optix.h
    HINTS "$ENV{OPTIX_ROOT}/include" "$ENV{OptiX_INSTALL_DIR}/include"
    PATHS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"
          "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
)

if(VERA_OPTIX_INCLUDE_DIR)
    set(VERA_ENABLE_OPTIX ON)
    message(STATUS "Vera: OptiX SDK found at ${VERA_OPTIX_INCLUDE_DIR} -- OptiX traversal pathway enabled")

    # OptixPrograms.cu compiles to PTX (not linked into the executable) -- OptiX loads it at
    # runtime via optixModuleCreate. Baked as an absolute path compile definition, same
    # convention as VERA_DATA_DIR below, so OptixTraversal.cpp can just read the file.
    set(VERA_OPTIX_PTX_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Data/OptixPrograms.ptx")
    add_custom_command(
        OUTPUT "${VERA_OPTIX_PTX_OUTPUT}"
        COMMAND "${CMAKE_CUDA_COMPILER}" --ptx
                -std=c++17
                --extended-lambda
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public"
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Public"
                "-I${CMAKE_CURRENT_SOURCE_DIR}/external/NexusBVH/Public"
                "-I${VERA_OPTIX_INCLUDE_DIR}"
                --generate-code=arch=compute_86,code=compute_86
                -DVERA_ENABLE_OPTIX
                -o "${VERA_OPTIX_PTX_OUTPUT}"
                "${VERA_OPTIX_PROGRAMS_SRC}"
        DEPENDS "${VERA_OPTIX_PROGRAMS_SRC}"
        COMMENT "Vera: compiling OptixPrograms.cu to PTX"
        VERBATIM
    )
    add_custom_target(VeraOptixPTX DEPENDS "${VERA_OPTIX_PTX_OUTPUT}")
else()
    set(VERA_ENABLE_OPTIX OFF)
    message(STATUS "Vera: OptiX SDK not found -- building software-traversal-only (set OPTIX_ROOT to enable)")
endif()

add_executable(Vera
    ${ALL_HEADERS}
    ${ALL_CXX_SRC}
    ${ALL_INLINE}
    ${ALL_CUDA_SRC}
    ${ALL_CUDA_HDRS}
)

target_include_directories(Vera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Public
)

target_precompile_headers(Vera PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public/PCH.h>
)

target_compile_options(Vera PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --extended-lambda
        -Xcompiler=/Zc:preprocessor
    >
)

target_compile_definitions(Vera PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
    VERA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Data/"
)

set_target_properties(Vera PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(VERA_ENABLE_OPTIX)
    add_dependencies(Vera VeraOptixPTX)
    target_include_directories(Vera PRIVATE ${VERA_OPTIX_INCLUDE_DIR})
    target_compile_definitions(Vera PRIVATE
        VERA_ENABLE_OPTIX
        VERA_OPTIX_PTX_PATH="${VERA_OPTIX_PTX_OUTPUT}"
    )
endif()

add_subdirectory(external/NexusBVH)

target_link_libraries(Vera PRIVATE NexusBVH)
