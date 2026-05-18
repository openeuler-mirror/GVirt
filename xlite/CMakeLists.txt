cmake_minimum_required(VERSION 3.16)
project(xlite)

set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
message(STATUS "RUN_MODE: ${RUN_MODE}")
message(STATUS "SOC_VERSION: ${SOC_VERSION}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PATH ${CMAKE_CURRENT_LIST_DIR})
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/out" CACHE STRING "path for install()" FORCE)
else()
    set(CMAKE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
endif()
message(STATUS "CMAKE_INSTALL_PATH: ${CMAKE_INSTALL_PATH}")

# xlite_kernels_common_${RUN_MODE}
file(GLOB KERNEL_FILES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax_topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/topk_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cast_bfloat16_t_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/permutation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/unpermutation_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/all_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/reduce_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/all_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_int8_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_int8_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/experts_counts_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/reorder_moe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/transpose_1_2_float.cpp
)

# xlite_kernels_bf16_${RUN_MODE}
file(GLOB KERNEL_FILES_BF16
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/cast_float_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax_topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/sigmoid_topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/topk_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/unpermutation_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_complex_and_cache_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_attention_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/quant_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/quant_dyn_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_mla_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_scores_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/muls_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/mla_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/transpose_1_2_bfloat16_t.cpp
)

# xlite_kernels_f16_${RUN_MODE}
file(GLOB KERNEL_FILES_F16
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/group_matmul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_complex_and_cache_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/flash_attention_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/dequant_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/indexer_scores_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/muls_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/conv1d_and_silu_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/beta_decay_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/transpose_1_2_float16_t.cpp
)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(xlite_kernels_common_${RUN_MODE} SHARED ${KERNEL_FILES_COMMON})
ascendc_library(xlite_kernels_bf16_${RUN_MODE} SHARED ${KERNEL_FILES_BF16})
ascendc_library(xlite_kernels_f16_${RUN_MODE} SHARED ${KERNEL_FILES_F16})

option(XLITE_DEBUG "Enable xlite debug mode" OFF)

# xlite
add_library(xlite SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/swizzle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/core_assigner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ccl.cpp
)

target_compile_options(xlite PRIVATE
    -std=c++17 -O3 -Wall -Werror
)

target_link_libraries(xlite PRIVATE
    xlite_kernels_common_${RUN_MODE}
    xlite_kernels_bf16_${RUN_MODE}
    xlite_kernels_f16_${RUN_MODE}
    hccl
    torch
)

if(XLITE_DEBUG)
    target_compile_definitions(xlite PRIVATE XLITE_DEBUG_ON)
endif()

# _C
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SYS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYTHON_SYS_DIR)
  # prepend so Python sys dir is searched first; use APPEND if you want it last
  list(PREPEND CMAKE_PREFIX_PATH "${PYTHON_SYS_DIR}")
endif()

find_package(pybind11 REQUIRED)
find_package(Torch COMPONENTS Development REQUIRED)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "pybind11 include: ${pybind11_INCLUDE_DIRS}")
message(STATUS "torch cpp_extension include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "torch cpp_extension lib: ${TORCH_INSTALL_PREFIX}/lib")

pybind11_add_module(
    _C
    csrc/_C.cpp
)

target_compile_options(_C PRIVATE
    -std=c++17 -O3 -Wall -Werror -fPIC
)

target_include_directories(_C PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_directories(_C PRIVATE
    ${TORCH_INSTALL_PREFIX}/lib
)

target_link_libraries(_C PRIVATE
    xlite
    c10
    torch
    torch_python
    torch_cpu
)

set(XLITE_EDITABLE_BUILD "OFF" CACHE STRING "Enable editable build and include torch from source path")
if (XLITE_EDITABLE_BUILD STREQUAL "ON")
    set(TORCH_RPATH "\$ORIGIN/../torch/lib:${TORCH_INSTALL_PREFIX}/lib")
else()
    set(TORCH_RPATH "\$ORIGIN/../torch/lib")
endif()
message(STATUS "TORCH_RPATH: ${TORCH_RPATH}")

set_target_properties(xlite PROPERTIES
    LINK_FLAGS "-Wl,-rpath,\$ORIGIN:\$ORIGIN/lib:${TORCH_RPATH}"
)
set_target_properties(_C PROPERTIES
    LINK_FLAGS "-Wl,-rpath,\$ORIGIN:\$ORIGIN/lib:${TORCH_RPATH}"
)

install(TARGETS xlite_kernels_common_${RUN_MODE} xlite_kernels_bf16_${RUN_MODE} xlite_kernels_f16_${RUN_MODE} xlite
    LIBRARY DESTINATION ${CMAKE_INSTALL_PATH}/xlite/lib
)

install(TARGETS _C
    LIBRARY DESTINATION ${CMAKE_INSTALL_PATH}/xlite
)
