cmake_minimum_required(VERSION 3.20)
project(BispectralPathTracer LANGUAGES CXX CUDA)

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

add_executable(BispectralPathTracer
    ${ALL_HEADERS}
    ${ALL_CXX_SRC}
    ${ALL_INLINE}
    ${ALL_CUDA_SRC}
    ${ALL_CUDA_HDRS}
)

target_include_directories(BispectralPathTracer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RGB/Public
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Spectral/Public
)

target_precompile_headers(BispectralPathTracer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public/PCH.h>
)

target_compile_options(BispectralPathTracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --extended-lambda
    >
)

target_compile_definitions(BispectralPathTracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
)

set_target_properties(BispectralPathTracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_subdirectory(external/NexusBVH)

target_link_libraries(BispectralPathTracer PRIVATE NexusBVH)
