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
    >
)

target_compile_definitions(Vera PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
    VERA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Data/"
)

set_target_properties(Vera PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_subdirectory(external/NexusBVH)

target_link_libraries(Vera PRIVATE NexusBVH)
