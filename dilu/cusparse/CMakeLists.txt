cmake_minimum_required(VERSION 3.24)
project(dilu_cusparse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3050 Laptop; override for A100/H100.
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate JAX FFI headers via the live python env (same pattern as Phase 1).
if(NOT DEFINED JAX_FFI_INCLUDE_DIR)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import jax.ffi; print(jax.ffi.include_dir(), end='')"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    RESULT_VARIABLE JAX_FFI_PROBE_RC)
  if(NOT JAX_FFI_PROBE_RC EQUAL 0)
    message(FATAL_ERROR
      "Failed to locate jax.ffi.include_dir() — is /home/yzk/jax-env active?")
  endif()
endif()
message(STATUS "JAX FFI include dir: ${JAX_FFI_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

add_library(dilu_cusparse SHARED
  cpp/plan_registry.cc
  cpp/dilu_factor.cc
  cpp/cusparse_dilu_analyze.cc
  cpp/cusparse_dilu_apply.cc
  cpp/cusparse_dilu_release.cc
  cuda/dilu_factor_kernel.cu
  cuda/scatter_diag_kernel.cu)

target_include_directories(dilu_cusparse PRIVATE
  ${JAX_FFI_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

target_link_libraries(dilu_cusparse PRIVATE
  CUDA::cudart
  CUDA::cusparse)

# Per brief: -O2 for CUDA (NOT -O3), no -ffast-math, no --use_fast_math.
target_compile_options(dilu_cusparse PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)

set_target_properties(dilu_cusparse PROPERTIES
  OUTPUT_NAME "dilu_cusparse"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
