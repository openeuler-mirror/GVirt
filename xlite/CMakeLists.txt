cmake_minimum_required(VERSION 3.16)
project(xlite)

set(RUN_MODE "npu" CACHE STRING "sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

# xlite_kernels_${RUN_MODE}
file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_bfloat16_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/matmul_float.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/decode_att_mix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/prefill_att_mix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_bias.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add.cpp
)

file(GLOB ASCENDC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/embed_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rmsnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/add_and_rmsnorm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/silu_and_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope_and_cache.cpp
)
list(APPEND KERNEL_FILES ${ASCENDC_FILES})


if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(xlite_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

# xlite
add_library(xlite SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/core_assigner.cpp
)

target_compile_options(xlite PRIVATE
    -std=c++17 -O3 -Wall -Werror
)

target_link_libraries(xlite PRIVATE
    xlite_kernels_${RUN_MODE}
    hccl
)

# _C
execute_process(
    COMMAND python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SYS_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${PYTHON_SYS_DIR} CACHE STRING "python sys path")

find_package(pybind11 REQUIRED)
find_package(Torch COMPONENTS Development REQUIRED)

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

target_link_options(_C PRIVATE
    -Wl,-rpath,xlite/lib:${PYTHON_SYS_DIR}/xlite/lib:${TORCH_INSTALL_PREFIX}/lib
)

install(TARGETS xlite_kernels_${RUN_MODE} xlite
    LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR}/xlite/lib
)

install(TARGETS _C
    LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR}/xlite
)
