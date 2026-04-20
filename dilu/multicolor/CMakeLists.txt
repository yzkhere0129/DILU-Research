cmake_minimum_required(VERSION 3.24)
project(dilu_multicolor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3050 Laptop; override for A100/H100.
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate JAX FFI headers via the live python env (same pattern as Phase 1/2).
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

# NOTE (arch §6.1): NO cusparse. Only cudart. Phase 3's explicit
# independence from NVIDIA sparse libraries is a design property.
add_library(dilu_multicolor SHARED
  cpp/plan_registry.cc
  cpp/multicolor_analyze.cc
  cpp/multicolor_apply.cc
  cpp/multicolor_refactor.cc
  cpp/multicolor_release.cc
  cuda/dilu_factor_kernel.cu         # copied verbatim from Phase 2
  cuda/elem_scale_kernel.cu           # copied (split off) from Phase 2
  cuda/permute_kernel.cu              # NEW
  cuda/color_stride_forward.cu        # NEW
  cuda/color_stride_backward.cu       # NEW
)

target_include_directories(dilu_multicolor PRIVATE
  ${JAX_FFI_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

target_link_libraries(dilu_multicolor PRIVATE
  CUDA::cudart)

# CLAUDE.md convention: no -ffast-math, -O2 on CUDA, -lineinfo for Nsight.
target_compile_options(dilu_multicolor PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)

set_target_properties(dilu_multicolor PROPERTIES
  OUTPUT_NAME "dilu_multicolor"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
