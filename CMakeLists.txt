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

# 2. DEFINE EXECUTABLE AND SOURCES
set(SOURCES
    src/main.cpp
    src/trt_builder.cpp
    src/data_generator.cpp
    src/io_writer.cpp
    src/action_selector.cpp
    src/mcts_engine.cpp
    src/mcts_node.cpp
    src/inference_batcher.cpp
    src/board_utils.cpp
    "${LMDB_DIR}/mdb.c"
    "${LMDB_DIR}/midl.c"
)

add_executable(talbot_engine ${SOURCES})

# 3. SETUP INCLUDES
target_include_directories(talbot_engine PRIVATE
    "./src"
    "${DEP_ROOT}"
    "${LIBTORCH_DIR}/include"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${LMDB_DIR}"
    "${CUDA_PATH}/include"
    "${DEP_ROOT}/yaml-cpp/include" 
    "${TRT_DIR}/include"
    "${ZSTD_DIR}/lib"
)

# 4. COMPILER FLAGS
if(MSVC)
    # CMake handles NOMINMAX here; remove it from your .cpp files!
    target_compile_definitions(talbot_engine PRIVATE NOMINMAX)
    target_compile_options(talbot_engine PRIVATE /O2 /EHsc /MD /wd4267 /wd4244 /wd4172 /wd4333)
endif()

# 5. LINKER LIBRARIES
target_link_directories(talbot_engine PRIVATE
    "${LIBTORCH_DIR}/lib"
    "${CUDA_PATH}/lib/x64"
    "${YAML_DIR}/Release"
    "${TRT_DIR}/lib"
    "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release"
)

target_link_libraries(talbot_engine PRIVATE
    torch torch_cpu c10 c10_cuda torch_cuda      
    cudart                                        
    nvinfer_10 nvonnxparser_10                   
    yaml-cpp                                     
    zstd_static                                  
    ntdll advapi32                               
)

set_target_properties(talbot_engine PROPERTIES 
    LINK_FLAGS "/INCLUDE:?warp_size@cuda@at@@YAHXZ /OPT:NOREF /RELEASE"
)

# 6. POST-BUILD: AUTOMATED DLL DEPLOYMENT
# Updated with verified paths from your 'ls -R' output
# 6. POST_BUILD: AUTOMATED DLL DEPLOYMENT
add_custom_command(TARGET talbot_engine POST_BUILD
    # LibTorch Core
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/c10_cuda.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cpu.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_cuda.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/torch_global_deps.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/libiomp5md.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${LIBTORCH_DIR}/lib/uv.dll" $<TARGET_FILE_DIR:talbot_engine>

    # Project Dependencies
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TRT_DIR}/lib" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${YAML_DIR}/Release/yaml-cpp.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${ZSTD_DIR}/build-cmake/build/cmake/lib/Release/zstd.dll" $<TARGET_FILE_DIR:talbot_engine>
    
    # CUDA Toolkit
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvrtc64_130_0.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvrtc-builtins64_130.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/nvJitLink_130_0.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublas64_13.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cublasLt64_13.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cudart64_13.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/curand64_10.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cusolver64_12.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cufft64_12.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/bin/x64/cusparse64_12.dll" $<TARGET_FILE_DIR:talbot_engine>
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/extras/CUPTI/lib64/cupti64_2025.3.0.dll" $<TARGET_FILE_DIR:talbot_engine>

    # cuDNN (THE FIX: Copy the whole directory to catch all sub-engines)
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CUDNN_PATH}" $<TARGET_FILE_DIR:talbot_engine>

    # NVTX
    COMMAND ${CMAKE_COMMAND} -E copy "${CUDA_PATH}/compute-sanitizer/sanitizer-public.dll" $<TARGET_FILE_DIR:talbot_engine>
)