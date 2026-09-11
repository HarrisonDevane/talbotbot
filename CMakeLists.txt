cmake_minimum_required(VERSION 3.21)
project(TalbotEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 1. DEPENDENCY PATHS
# -----------------------------------------------------------------------------
foreach(v TALBOT_DEP_ROOT TALBOT_CUDA_PATH TALBOT_TRT_DIR)
    if(NOT DEFINED ENV{${v}})
        message(FATAL_ERROR "Required env var '${v}' is not set.")
    endif()
endforeach()

set(DEP_ROOT   $ENV{TALBOT_DEP_ROOT})
set(CUDA_PATH  $ENV{TALBOT_CUDA_PATH})
set(TRT_DIR    $ENV{TALBOT_TRT_DIR})

set(LIBTORCH_DIR "${DEP_ROOT}/libtorch")
set(FATHOM_DIR   "${DEP_ROOT}/Fathom")

# -----------------------------------------------------------------------------
# 2. BUILD METADATA (Git hash)
# -----------------------------------------------------------------------------
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD OUTPUT_VARIABLE GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(GIT_COMMIT "unknown")
endif()

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
# 4. COMMON CONFIG
# -----------------------------------------------------------------------------
set(INCLUDE_DIRS
    "./src"
    "./src/core"
    "./src/train"
    "./src/play"
    "./src/play/shared"
    "./src/play/uci"
    "./src/third_party"
    "${LIBTORCH_DIR}/include"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include"
    "${CUDA_PATH}/include"
    "${TRT_DIR}/include"
    "${TRT_DIR}/include/x86_64-linux-gnu"
    "${FATHOM_DIR}/src"
    "${DEP_ROOT}/chess-library/include"
    "${DEP_ROOT}/concurrentqueue"
)

set(LINK_DIRS
    "${LIBTORCH_DIR}/lib"
    "${CUDA_PATH}/lib64"
    "${TRT_DIR}/lib/x86_64-linux-gnu"
)

set(LIBS
    torch torch_cpu c10 c10_cuda torch_cuda
    cudart nvinfer nvonnxparser
    yaml-cpp zstd lmdb
    pthread
)

add_library(talbot_common INTERFACE)
target_include_directories(talbot_common INTERFACE ${INCLUDE_DIRS})
target_link_directories(talbot_common    INTERFACE ${LINK_DIRS})
target_link_libraries(talbot_common      INTERFACE ${LIBS})

target_compile_options(talbot_common INTERFACE -O3 -Wall -Wno-narrowing -Wno-sign-compare)

# -----------------------------------------------------------------------------
# 5. EXECUTABLES
# -----------------------------------------------------------------------------
add_executable(talbot_data_generator ${SHARED_SOURCES} src/train/main_data_generator.cpp src/train/data_generator.cpp src/train/target_generator.cpp src/train/io_writer.cpp)
add_executable(talbot_uci ${SHARED_SOURCES} src/play/uci/main_uci.cpp src/play/shared/time_control.cpp)
add_executable(talbot_tournament ${SHARED_SOURCES} src/play/tournament/main_tournament.cpp src/play/tournament/self_play_session.cpp src/play/shared/time_control.cpp src/play/shared/opening_book.cpp src/play/shared/game_worker.cpp)
add_executable(talbot_trt_compile src/tools/main_trt_compile.cpp src/tools/trt_builder.cpp)

foreach(tgt talbot_data_generator talbot_uci talbot_tournament)
    target_link_libraries(${tgt} PRIVATE talbot_common)
endforeach()

target_include_directories(talbot_trt_compile PRIVATE "./src/tools" "${CMAKE_BINARY_DIR}/generated" "${CUDA_PATH}/include" "${TRT_DIR}/include" "${TRT_DIR}/include/x86_64-linux-gnu")
target_link_directories(talbot_trt_compile PRIVATE "${CUDA_PATH}/lib64" "${TRT_DIR}/lib/x86_64-linux-gnu")
target_link_libraries(talbot_trt_compile PRIVATE cudart nvinfer nvonnxparser)
target_compile_options(talbot_trt_compile PRIVATE -O3 -Wall)

set_source_files_properties("${FATHOM_DIR}/src/tbprobe.c" PROPERTIES COMPILE_OPTIONS "-std=c11")

# -----------------------------------------------------------------------------
# 6. CONFIG FILE DEPLOYMENT
# -----------------------------------------------------------------------------
add_custom_command(TARGET talbot_uci POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/config/play_uci.yaml" "$<TARGET_FILE_DIR:talbot_uci>/play_uci.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/config/model.yaml" "$<TARGET_FILE_DIR:talbot_uci>/model.yaml"
)