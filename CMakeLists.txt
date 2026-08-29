cmake_minimum_required(VERSION 3.20)
project(Vera LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

option(VERA_BUILD_CLI    "Build the native command-line renderer (src/Main.cu)" ON)
option(VERA_BUILD_PYTHON "Build the nanobind Python extension (_vera)"          OFF)

# ── Source discovery ───────────────────────────────────────────────────────
# Everything under src/ is the core renderer EXCEPT:
#   * src/Main.cu           -> the CLI entry point, only in the `Vera` executable
#   * src/Core/Private/OptixPrograms.cu -> compiled to PTX separately (see below)
file(GLOB_RECURSE ALL_HEADERS   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE ALL_CXX_SRC   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE ALL_INLINE    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl")
file(GLOB_RECURSE ALL_CUDA_SRC  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE ALL_CUDA_HDRS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh")

set(VERA_MAIN_SRC           "${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cu")
set(VERA_OPTIX_PROGRAMS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Private/OptixPrograms.cu")
list(REMOVE_ITEM ALL_CUDA_SRC "${VERA_MAIN_SRC}" "${VERA_OPTIX_PROGRAMS_SRC}")

# ── Optional OptiX hardware-traversal pathway ──────────────────────────────
find_path(VERA_OPTIX_INCLUDE_DIR optix.h
    HINTS "$ENV{OPTIX_ROOT}/include" "$ENV{OptiX_INSTALL_DIR}/include"
    PATHS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"
          "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include"
)

if(VERA_OPTIX_INCLUDE_DIR)
    set(VERA_ENABLE_OPTIX ON)
    message(STATUS "Vera: OptiX SDK found at ${VERA_OPTIX_INCLUDE_DIR} -- OptiX traversal pathway enabled")

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

add_subdirectory(external/NexusBVH)

# ── vera_core: the whole renderer as a reusable static library ─────────────
add_library(vera_core STATIC
    ${ALL_HEADERS}
    ${ALL_CXX_SRC}
    ${ALL_INLINE}
    ${ALL_CUDA_SRC}
    ${ALL_CUDA_HDRS}
)

target_include_directories(vera_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public
    ${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Public
)

target_precompile_headers(vera_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/Core/Public/PCH.h>
)

target_compile_options(vera_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
        --extended-lambda
        -Xcompiler=/Zc:preprocessor
    >
)

target_compile_definitions(vera_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
)
# VERA_DATA_DIR is PUBLIC so the bindings TU (and any other consumer) can locate
# srgb.coeff / the OptiX PTX without re-deriving the path.
target_compile_definitions(vera_core PUBLIC
    VERA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/HWSS/Data/"
)

set_target_properties(vera_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON   # required: vera_core links into the _vera shared module
)

if(VERA_ENABLE_OPTIX)
    add_dependencies(vera_core VeraOptixPTX)
    target_include_directories(vera_core PRIVATE ${VERA_OPTIX_INCLUDE_DIR})
    target_compile_definitions(vera_core PRIVATE
        VERA_ENABLE_OPTIX
        VERA_OPTIX_PTX_PATH="${VERA_OPTIX_PTX_OUTPUT}"
    )
endif()

target_link_libraries(vera_core PUBLIC NexusBVH)

# ── Vera: the native CLI renderer (unchanged entry point) ─────────────────
if(VERA_BUILD_CLI)
    add_executable(Vera ${VERA_MAIN_SRC})
    target_link_libraries(Vera PRIVATE vera_core)
    set_target_properties(Vera PROPERTIES
        CUDA_SEPARABLE_COMPILATION   ON
        CUDA_RESOLVE_DEVICE_SYMBOLS   ON
    )
endif()

# ── _vera: nanobind Python extension ─────────────────────────────────────
if(VERA_BUILD_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # nanobind ships its own CMake config; locate it via the installed package
    # (`pip install nanobind`). Alternative: add_subdirectory(external/nanobind).
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    # vera_ext is a .cu (the renderer's public headers are .cuh) — CMake picks
    # CUDA by extension. nvcc needs the same lambda/preprocessor flags the core
    # uses, and nanobind's hidden-visibility flags must be forwarded to the host
    # compiler with -Xcompiler.
    nanobind_add_module(_vera STABLE_ABI NB_STATIC
        bindings/vera_ext.cu
    )
    target_link_libraries(_vera PRIVATE vera_core)
    target_compile_options(_vera PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -Xcompiler=/Zc:preprocessor>
    )
    set_target_properties(_vera PROPERTIES
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS  ON   # shared objects don't device-link by default
    )
    # KNOWN FRICTION (Windows, Visual Studio generator): vera_core is a static
    # lib with separable compilation, so each consumer (Vera, _vera) device-links
    # it separately and MSBuild's incremental tracking of that step is unreliable
    # -- after editing renderer .cu/.cuh files you can get a STALE device image or
    # a link error about mismatched __fatbinwrap_* hashes. Fix: delete the
    # affected target dirs (build/_vera.dir, build/Vera.dir, build/vera_core.dir)
    # and rebuild. The Ninja generator -- and the `pip install` / scikit-build
    # path -- do not have this problem; prefer `-G Ninja` for iterative work.
    # (A cleaner long-term fix is to make vera_core SHARED so the device link
    # happens once.)

    # so `import _vera` works from the source tree and from an installed wheel
    install(TARGETS _vera LIBRARY DESTINATION vera)
endif()
