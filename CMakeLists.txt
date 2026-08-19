cmake_minimum_required(VERSION 3.18)
project(TalbotEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. SETUP PATHS
set(DEP_ROOT "D:/Projects/talbot/dependencies")
set(LIBTORCH_DIR "${DEP_ROOT}/libtorch")
set(YAML_DIR "${DEP_ROOT}/yaml-cpp/build")
set(LMDB_DIR "${DEP_ROOT}/openldap-mdb/libraries/liblmdb")
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
set(CUDNN_PATH "C:/Program Files/NVIDIA/CUDNN/v9.20/bin/13.2/x64")
set(TRT_DIR "${DEP_ROOT}/TensorRT-10.10.0.31")
set(ZSTD_DIR "${DEP_ROOT}/zstd")
set(FATHOM_DIR "${DEP_ROOT}/Fathom")

# 2. DEFINE SHARED SOURCES
# Core engine: used by training, play, and UCI executables.
set(SHARED_SOURCES
    src/core/action_selector.cpp
    src/core/mcts_engine.cpp
    src/core/mcts_node.cpp
    src/core/inference_batcher.cpp
    src/core/board_utils.cpp
    "${FATHOM_DIR}/src/tbprobe.c"
)

# 3. DEFINE EXECUTABLES
# Main Training Engine -- training-only sources live in src/train/.
add_executable(talbot_engine
    ${SHARED_SOURCES}
    src/train/main_train.cpp
    src/train/data_generator.cpp
    src/train/target_generator.cpp
    src/train/io_writer.cpp
    "${LMDB_DIR}/mdb.c"
    "${LMDB_DIR}/midl.c"
)

# Standalone UCI Engine: reads play_uci.yaml and communicates via UCI protocol.
add_executable(talbot_uci
    ${SHARED_SOURCES}
    src/play/uci/main_uci.cpp
    src/play/shared/time_control.cpp
)

# TRT Compile Tool: standalone exe that turns an ONNX file into a serialized TensorRT engine.
add_executable(talbot_trt_compile
    src/tools/main_trt_compile.cpp
    src/tools/trt_builder.cpp
)

# 4. SETUP INCLUDES
set(INCLUDE_DIRS
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

target_include_directories(talbot_engine PRIVATE ${INCLUDE_DIRS})
target_include_directories(talbot_uci    PRIVATE ${INCLUDE_DIRS})

# Lean include set for the compile tool
target_include_directories(talbot_trt_compile PRIVATE
    "./src/tools"
    "${CUDA_PATH}/include"
    "${TRT_DIR}/include"
)

# 5. COMPILER FLAGS
if(MSVC)
    set(MSVC_FLAGS /O2 /EHsc /MD /wd4267 /wd4244 /wd4172 /wd4333)

    target_compile_definitions(talbot_engine PRIVATE NOMINMAX)
    target_compile_options(talbot_engine PRIVATE ${MSVC_FLAGS})

    target_compile_definitions(talbot_uci PRIVATE NOMINMAX)
    target_compile_options(talbot_uci PRIVATE ${MSVC_FLAGS})

    target_compile_definitions(talbot_trt_compile PRIVATE NOMINMAX)
    target_compile_options(talbot_trt_compile PRIVATE ${MSVC_FLAGS})

    # Fathom's tbprobe.c uses C11 <stdatomic.h>; MSVC requires /std:c11 or c17.
    set_source_files_properties(
        "${FATHOM_DIR}/src/tbprobe.c"
        PROPERTIES COMPILE_OPTIONS "/std:c11;/experimental:c11atomics"
    )
endif()

# 6. LINKER LIBRARIES
set(LINK_DIRS
    "${LIBTORCH_DIR}/lib"
    "${CUDA_PATH}/lib/x64"
    "${YAML_DIR}/Release"
    "${TRT_DIR}/lib"
    "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release"
)

target_link_directories(talbot_engine PRIVATE ${LINK_DIRS})
target_link_directories(talbot_uci    PRIVATE ${LINK_DIRS})

set(LIBS
    torch torch_cpu c10 c10_cuda torch_cuda
    cudart
    nvinfer_10 nvonnxparser_10
    yaml-cpp
    zstd_static
    ntdll advapi32
)

target_link_libraries(talbot_engine PRIVATE ${LIBS})
target_link_libraries(talbot_uci    PRIVATE ${LIBS})

# Compile tool: lean link.
target_link_directories(talbot_trt_compile PRIVATE
    "${CUDA_PATH}/lib/x64"
    "${TRT_DIR}/lib"
)
target_link_libraries(talbot_trt_compile PRIVATE
    cudart
    nvinfer_10
    nvonnxparser_10
)

# Special linker flags for LibTorch/CUDA compatibility
set(LINK_OPTS "/INCLUDE:?warp_size@cuda@at@@YAHXZ /OPT:NOREF /RELEASE")
set_target_properties(talbot_engine PROPERTIES LINK_FLAGS ${LINK_OPTS})
set_target_properties(talbot_uci    PROPERTIES LINK_FLAGS ${LINK_OPTS})

# 7. POST-BUILD: AUTOMATED DLL DEPLOYMENT
function(deploy_dlls target_name)
    add_custom_command(TARGET ${target_name} POST_BUILD
        # LibTorch Core
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10_cuda.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cpu.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cuda.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_global_deps.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/libiomp5md.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/uv.dll" $<TARGET_FILE_DIR:${target_name}>

        # Project Dependencies
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_DIR}/lib" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${YAML_DIR}/Release/yaml-cpp.dll" $<TARGET_FILE_DIR:${target_name}>
        COMMAND ${CMAKE_COMMAND} -E copy "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release/zstd.dll" $<TARGET_FILE_DIR:${target_name}>

        # CUDA Toolkit
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

deploy_dlls(talbot_engine)
deploy_dlls(talbot_uci)

# TRT compile tool DLL deployment
add_custom_command(TARGET talbot_trt_compile POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_DIR}/lib" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cudart64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublas64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublasLt64_13.dll" $<TARGET_FILE_DIR:talbot_trt_compile>
)