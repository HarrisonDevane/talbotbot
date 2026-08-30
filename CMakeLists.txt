# =============================================================================
# Talbot -- top-level CMake configuration.
#
# Design goals (see dependencies/MANIFEST.md for the full story):
#   1. No hardcoded absolute paths. All external dep roots come from env vars
#      set by env\dev.env. Missing env vars -> loud, immediate fail at configure.
#   2. Shared config (includes, libs, flags) lives in ONE INTERFACE library
#      (talbot_common) rather than being duplicated per target. Impossible to
#      have the three main binaries drift on flags.
#   3. Every binary stamps itself with the git commit + configure time via a
#      generated build_info.hpp. Trace any log or artifact back to source.
# =============================================================================

cmake_minimum_required(VERSION 3.21)  # 3.21 for CMakePresets v3 support
project(TalbotEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1. DEPENDENCY PATHS (env-driven, no fallbacks)
#
# Fail-fast if any is missing rather than silently probing default locations.
# A missing env var is almost always "user forgot to `call env\dev.env`" --
# printing which one is missing beats a cryptic "path not found" 200 lines
# into configure.
# -----------------------------------------------------------------------------
foreach(v TALBOT_DEP_ROOT TALBOT_CUDA_PATH TALBOT_CUDNN_PATH TALBOT_TRT_DIR)
    if(NOT DEFINED ENV{${v}})
        message(FATAL_ERROR
            "Required env var '${v}' is not set.\n"
            "  1. Copy env/dev.env.template to env/dev.env\n"
            "  2. Edit env/dev.env with paths for your machine\n"
            "  3. `call env\\dev.env` before running build.bat / cmake\n"
            "See dependencies/MANIFEST.md for expected versions.")
    endif()
endforeach()

set(DEP_ROOT   $ENV{TALBOT_DEP_ROOT})
set(CUDA_PATH  $ENV{TALBOT_CUDA_PATH})
set(CUDNN_PATH $ENV{TALBOT_CUDNN_PATH})
set(TRT_DIR    $ENV{TALBOT_TRT_DIR})

# Derived: all under DEP_ROOT with fixed layout (see MANIFEST.md).
set(LIBTORCH_DIR "${DEP_ROOT}/libtorch")
set(YAML_DIR     "${DEP_ROOT}/yaml-cpp/build")
set(LMDB_DIR     "${DEP_ROOT}/openldap-mdb/libraries/liblmdb")
set(ZSTD_DIR     "${DEP_ROOT}/zstd")
set(FATHOM_DIR   "${DEP_ROOT}/Fathom")

message(STATUS "TALBOT_DEP_ROOT   = ${DEP_ROOT}")
message(STATUS "TALBOT_CUDA_PATH  = ${CUDA_PATH}")
message(STATUS "TALBOT_CUDNN_PATH = ${CUDNN_PATH}")
message(STATUS "TALBOT_TRT_DIR    = ${TRT_DIR}")

# -----------------------------------------------------------------------------
# 2. BUILD METADATA
#
# Runs at CONFIGURE time, not build time. Re-run cmake to pick up a new commit
# hash. That's the right granularity: git state changes tend to correlate with
# wanting a fresh configure anyway (dep updates, new files, etc.), and re-
# stamping every incremental build would trigger unnecessary link rebuilds.
# -----------------------------------------------------------------------------
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_RESULT
    )
    if(NOT GIT_RESULT EQUAL 0)
        set(GIT_COMMIT "unknown")
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DIRTY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(GIT_DIRTY)
        set(GIT_COMMIT "${GIT_COMMIT}-dirty")
    endif()
else()
    set(GIT_COMMIT "unknown")
endif()

string(TIMESTAMP BUILD_TIME_UTC "%Y-%m-%d %H:%M:%S UTC" UTC)
message(STATUS "GIT_COMMIT = ${GIT_COMMIT}")
message(STATUS "BUILD_TIME = ${BUILD_TIME_UTC}")

configure_file(
    "${CMAKE_SOURCE_DIR}/src/core/build_info.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/build_info.hpp"
    @ONLY
)

# -----------------------------------------------------------------------------
# 3. SHARED SOURCES
# -----------------------------------------------------------------------------
set(SHARED_SOURCES
    src/core/action_selector.cpp
    src/core/mcts_engine.cpp
    src/core/mcts_node.cpp
    src/core/inference_batcher.cpp
    src/core/board_utils.cpp
    src/core/pgn_writer.cpp
    "${FATHOM_DIR}/src/tbprobe.c"
)

# -----------------------------------------------------------------------------
# 4. COMMON CONFIG (INTERFACE library)
#
# Everything shared by the three main binaries lives here. Adding a new
# binary that wants the same libs/includes/flags: one `target_link_libraries
# (foo PRIVATE talbot_common)` and it's wired up. Impossible to accidentally
# omit a flag from one target -- the pattern is the same everywhere.
#
# talbot_trt_compile is a lean standalone tool with a smaller dep set; it
# does NOT link talbot_common and stays configured separately below.
# -----------------------------------------------------------------------------
set(INCLUDE_DIRS
    "${CMAKE_BINARY_DIR}/generated"    # for build_info.hpp
    "./src"
    "./src/core"
    "./src/train"
    "./src/play"
    "./src/play/shared"
    "./src/play/uci"
    "./src/third_party"
    "${DEP_ROOT}"
    "${LIBTORCH_DIR}/include"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LMDB_DIR}"
    "${CUDA_PATH}/include"
    "${DEP_ROOT}/yaml-cpp/include"
    "${TRT_DIR}/include"
    "${ZSTD_DIR}/lib"
    "${FATHOM_DIR}/src"
)

set(LINK_DIRS
    "${LIBTORCH_DIR}/lib"
    "${CUDA_PATH}/lib/x64"
    "${YAML_DIR}/Release"
    "${TRT_DIR}/lib"
    "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release"
)

set(LIBS
    torch torch_cpu c10 c10_cuda torch_cuda
    cudart
    nvinfer_10 nvonnxparser_10
    yaml-cpp
    zstd_static
    ntdll advapi32
)

add_library(talbot_common INTERFACE)
target_include_directories(talbot_common INTERFACE ${INCLUDE_DIRS})
target_link_directories(talbot_common    INTERFACE ${LINK_DIRS})
target_link_libraries(talbot_common      INTERFACE ${LIBS})

if(MSVC)
    target_compile_definitions(talbot_common INTERFACE NOMINMAX)
    # /wd4267 4244 4172 4333: harmless narrowing warnings from libtorch / yaml.
    target_compile_options(talbot_common INTERFACE
        /O2 /EHsc /MD /wd4267 /wd4244 /wd4172 /wd4333)
endif()

# -----------------------------------------------------------------------------
# 5. EXECUTABLES
# -----------------------------------------------------------------------------
add_executable(talbot_engine
    ${SHARED_SOURCES}
    src/train/main_train.cpp
    src/train/data_generator.cpp
    src/train/target_generator.cpp
    src/train/io_writer.cpp
    "${LMDB_DIR}/mdb.c"
    "${LMDB_DIR}/midl.c"
)

add_executable(talbot_uci
    ${SHARED_SOURCES}
    src/play/uci/main_uci.cpp
    src/play/shared/time_control.cpp
)

add_executable(talbot_tournament
    ${SHARED_SOURCES}
    src/play/tournament/main_tournament.cpp
    src/play/tournament/self_play_session.cpp
    src/play/shared/time_control.cpp
    src/play/shared/opening_book.cpp
    src/play/shared/game_worker.cpp
)

# TRT compile tool: lean, standalone. Doesn't link talbot_common.
add_executable(talbot_trt_compile
    src/tools/main_trt_compile.cpp
    src/tools/trt_builder.cpp
)

# Wire the three main binaries to the common config + link opts in one pass.
if(MSVC)
    set(LINK_OPTS "/INCLUDE:?warp_size@cuda@at@@YAHXZ /OPT:NOREF /RELEASE")
endif()

foreach(tgt talbot_engine talbot_uci talbot_tournament)
    target_link_libraries(${tgt} PRIVATE talbot_common)
    if(MSVC)
        set_target_properties(${tgt} PROPERTIES LINK_FLAGS ${LINK_OPTS})
    endif()
endforeach()

# TRT compile tool: separate include set + libs (no libtorch).
target_include_directories(talbot_trt_compile PRIVATE
    "./src/tools"
    "${CMAKE_BINARY_DIR}/generated"
    "${CUDA_PATH}/include"
    "${TRT_DIR}/include"
)
target_link_directories(talbot_trt_compile PRIVATE
    "${CUDA_PATH}/lib/x64"
    "${TRT_DIR}/lib"
)
target_link_libraries(talbot_trt_compile PRIVATE
    cudart nvinfer_10 nvonnxparser_10
)
if(MSVC)
    target_compile_definitions(talbot_trt_compile PRIVATE NOMINMAX)
    target_compile_options(talbot_trt_compile PRIVATE
        /O2 /EHsc /MD /wd4267 /wd4244 /wd4172 /wd4333)
endif()

# Fathom's tbprobe.c uses C11 <stdatomic.h>; MSVC requires /std:c11 or c17.
if(MSVC)
    set_source_files_properties(
        "${FATHOM_DIR}/src/tbprobe.c"
        PROPERTIES COMPILE_OPTIONS "/std:c11;/experimental:c11atomics"
    )
endif()

# -----------------------------------------------------------------------------
# 6. DLL DEPLOYMENT (post-build)
#
# copy_directory on the TRT lib dir copies more than we strictly need at
# runtime (all the .lib files too), but it's cheap and it means adding a new
# TRT DLL doesn't require a CMake edit.
# -----------------------------------------------------------------------------
function(deploy_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        # LibTorch core
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10_cuda.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cpu.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cuda.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_global_deps.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/libiomp5md.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/uv.dll" $<TARGET_FILE_DIR:${target_name}>

        # Project deps
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_DIR}/lib" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${YAML_DIR}/Release/yaml-cpp.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release/zstd.dll" $<TARGET_FILE_DIR:${target_name}>

        # CUDA toolkit
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvrtc64_130_0.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvrtc-builtins64_130.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvJitLink_130_0.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublas64_13.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublasLt64_13.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cudart64_13.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/curand64_10.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cusolver64_12.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cufft64_12.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cusparse64_12.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/extras/CUPTI/lib64/cupti64_2025.3.0.dll" $<TARGET_FILE_DIR:${target_name}>

        # cuDNN
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CUDNN_PATH}" $<TARGET_FILE_DIR:${target_name}>

        # NVTX / Sanitizer
        COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/compute-sanitizer/sanitizer-public.dll" $<TARGET_FILE_DIR:${target_name}>
    )
endfunction()

foreach(tgt talbot_engine talbot_uci talbot_tournament)
    deploy_dlls(${tgt})
endforeach()

# TRT compile tool: subset of DLLs (no libtorch, no cuDNN).
add_custom_command(TARGET talbot_trt_compile POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_DIR}/lib" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cudart64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublas64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublasLt64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
)

# -----------------------------------------------------------------------------
# 7. CONFIG FILE DEPLOYMENT
# Ship play_uci.yaml + model.yaml next to talbot_uci.exe. Both required at
# startup; keeping the copies live-synced avoids the "edited config in repo
# but forgot to copy to build dir" trap. model.engine is a build artifact
# and is copied manually.
# -----------------------------------------------------------------------------
add_custom_command(TARGET talbot_uci POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config/play_uci.yaml"
        "$<TARGET_FILE_DIR:talbot_uci>/play_uci.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/config/model.yaml"
        "$<TARGET_FILE_DIR:talbot_uci>/model.yaml"
)