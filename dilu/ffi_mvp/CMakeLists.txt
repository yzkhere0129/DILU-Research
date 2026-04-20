cmake_minimum_required(VERSION 3.24)
project(dilu_ffi_mvp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # RTX 3050 is compute 8.6; A100 = 80, H100 = 90. Override via
  #   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ...
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate JAX FFI headers by asking the live Python env — no guessing, no manual
# path. If the env changes, rebuild: the path tracks automatically.
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

add_library(dilu_ffi_mvp SHARED
  cpp/jacobi_residual.cc
  cuda/jacobi_residual_kernel.cu)

target_include_directories(dilu_ffi_mvp PRIVATE ${JAX_FFI_INCLUDE_DIR})

target_link_libraries(dilu_ffi_mvp PRIVATE CUDA::cudart)

# Deterministic FP, -lineinfo for Nsight Compute kernel attribution.
# No -ffast-math per CLAUDE.md.
target_compile_options(dilu_ffi_mvp PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
# Note: nvcc --use_fast_math is OFF by default; we deliberately do NOT pass
# it. Per CLAUDE.md we need deterministic FP associativity for regression
# comparability.

set_target_properties(dilu_ffi_mvp PROPERTIES
  OUTPUT_NAME "dilu_ffi_mvp"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
