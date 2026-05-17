cmake_minimum_required(VERSION 3.24)
project(dilu_amgx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Override via cmake -DCMAKE_CUDA_ARCHITECTURES="89;90;120" .. or the
  # DILU_AMGX_CUDA_ARCH env var consumed by build.sh.
  # Default 86 = RTX 3050 Ampere; common alternatives:
  #   89=Ada (RTX 4090 / L40), 90=H100 Hopper,
  #   120=Blackwell (RTX 5060, CUDA 13+).
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate JAX FFI headers via the live python env (Phase 1/2/3 pattern).
if(NOT DEFINED JAX_FFI_INCLUDE_DIR)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import jax.ffi; print(jax.ffi.include_dir(), end='')"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    RESULT_VARIABLE JAX_FFI_PROBE_RC)
  if(NOT JAX_FFI_PROBE_RC EQUAL 0)
    message(FATAL_ERROR
      "Failed to locate jax.ffi.include_dir(). Is your JAX virtualenv "
      "active and importable as `${Python3_EXECUTABLE}`? You can also "
      "bypass the probe with -DJAX_FFI_INCLUDE_DIR=/path/to/jax/ffi/include.")
  endif()
endif()
message(STATUS "JAX FFI include dir: ${JAX_FFI_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

# AMGx installation prefix. Order of resolution:
#   1. -DAMGX_ROOT=... on the cmake command line
#   2. AMGX_ROOT environment variable
#   3. $HOME/amgx-2.5.0   (default for this research repo)
if(NOT DEFINED AMGX_ROOT AND DEFINED ENV{AMGX_ROOT})
  set(AMGX_ROOT "$ENV{AMGX_ROOT}")
endif()
if(NOT DEFINED AMGX_ROOT)
  # Default: user-installed AMGx under $HOME/local/amgx (this repo's canonical
  # location; see phase4_amgx_ffi_architecture.md §7).
  set(AMGX_ROOT "$ENV{HOME}/local/amgx")
endif()
message(STATUS "AMGX_ROOT: ${AMGX_ROOT}")

find_library(AMGX_LIB amgxsh
  PATHS "${AMGX_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED)
if(NOT EXISTS "${AMGX_ROOT}/include/amgx_c.h")
  message(FATAL_ERROR
    "amgx_c.h not found at ${AMGX_ROOT}/include — check AMGX_ROOT.")
endif()
set(AMGX_INCLUDE_DIR "${AMGX_ROOT}/include")
message(STATUS "AMGx library: ${AMGX_LIB}")

add_library(dilu_amgx SHARED
  cpp/plan_registry.cc
  cpp/amgx_setup.cc
  cpp/amgx_update_coefficients.cc
  cpp/amgx_solve.cc
  cpp/amgx_release.cc)

target_include_directories(dilu_amgx PRIVATE
  ${JAX_FFI_INCLUDE_DIR}
  ${AMGX_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

target_link_libraries(dilu_amgx PRIVATE
  ${AMGX_LIB}
  CUDA::cudart)

# Per CLAUDE.md: no -ffast-math, -O2 (not -O3) for CUDA if any .cu were added.
target_compile_options(dilu_amgx PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)

# Optional verbose mode: logs cudaPointerGetAttributes for each FFI buffer.
if(DEFINED DILU_AMGX_VERBOSE AND DILU_AMGX_VERBOSE)
  target_compile_definitions(dilu_amgx PRIVATE DILU_AMGX_VERBOSE=1)
  message(STATUS "DILU_AMGX_VERBOSE=1 (zero-copy pointer instrumentation enabled)")
endif()

set_target_properties(dilu_amgx PROPERTIES
  OUTPUT_NAME "dilu_amgx"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  INSTALL_RPATH "${AMGX_ROOT}/lib"
  BUILD_WITH_INSTALL_RPATH ON)
