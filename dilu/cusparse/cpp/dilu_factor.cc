// FFI handler for `dilu_factor`: compute D_* on device given CSR(A).
//
// Contract (arch doc §4.1): takes CSR triplet + precomputed diag_offset,
// writes d_star[n]. Serial recurrence; one-thread kernel. Phase 3 upgrades.

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" cudaError_t launch_dilu_factor(
    cudaStream_t stream,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values,
    const int32_t* diag_offset,
    double* d_star,
    int32_t n);

static ffi::Error DiluFactorImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> col_idx,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Buffer<ffi::DataType::S32> diag_offset,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> d_star) {
  const size_t n = d_star->element_count();
  if (diag_offset.element_count() != n) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "dilu_factor: diag_offset must have length n");
  }
  if (row_ptr.element_count() != n + 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "dilu_factor: row_ptr must have length n+1");
  }
  if (col_idx.element_count() != values.element_count()) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "dilu_factor: col_idx and values length mismatch");
  }
  if (n > static_cast<size_t>(INT32_MAX)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "dilu_factor: n exceeds int32");
  }
  cudaError_t err = launch_dilu_factor(
      stream,
      row_ptr.typed_data(),
      col_idx.typed_data(),
      values.typed_data(),
      diag_offset.typed_data(),
      d_star->typed_data(),
      static_cast<int32_t>(n));
  if (err != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("dilu_factor launch: ") +
                          cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DiluFactor, DiluFactorImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());
