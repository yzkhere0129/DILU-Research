// XLA FFI handler for the Jacobi-residual MVP kernel.
//
// Contract locked in design doc §2.2. Seven device buffers in, one out,
// all float64 / int32 / row-major. Stream is XLA-supplied — never default.
//
// noexcept boundary (design doc §1.3, landmine #4): all failures are funneled
// through ffi::Error, never a C++ throw.

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" cudaError_t launch_jacobi_residual(
    cudaStream_t stream,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values,
    const double* diag,
    const double* b,
    const double* x,
    double* y,
    int32_t n);

static ffi::Error JacobiResidualImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> col_idx,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Buffer<ffi::DataType::F64> diag,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> y) {
  const size_t n = b.element_count();

  // Shape invariants — XLA does not enforce these, the design doc §1.3 #5 says
  // we must. Catch them before they become UB in the kernel.
  if (diag.element_count() != n || x.element_count() != n ||
      y->element_count() != n) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "JacobiResidual: b/diag/x/y length mismatch");
  }
  if (row_ptr.element_count() != n + 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "JacobiResidual: row_ptr must have length n+1");
  }
  if (col_idx.element_count() != values.element_count()) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "JacobiResidual: col_idx/values length mismatch");
  }
  if (n > static_cast<size_t>(INT32_MAX)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "JacobiResidual: n exceeds int32 range");
  }

  cudaError_t err = launch_jacobi_residual(
      stream,
      row_ptr.typed_data(),
      col_idx.typed_data(),
      values.typed_data(),
      diag.typed_data(),
      b.typed_data(),
      x.typed_data(),
      y->typed_data(),
      static_cast<int32_t>(n));
  if (err != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("JacobiResidual launch failed: ") +
                          cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JacobiResidual, JacobiResidualImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());
