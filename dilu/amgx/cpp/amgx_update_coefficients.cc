// FFI handler: amgx_update_coefficients — values-only refresh.
//
// Contract (arch doc §3.2):
//   Inputs:  token uint64[1], values f64[nnz]
//   Output:  status int32[1]  (0=ok, nonzero=AMGx RC)
//
// Pattern (N, nnz) is fingerprinted; caller guarantees row_ptr/col_idx
// unchanged. AMGx:
//   - AMGX_matrix_replace_coefficients: D→D copy of new values into AMGx's
//     owned matrix buffer. Coarsening hierarchy / interpolators stay.
//   - AMGX_solver_resetup: re-derives operator-dependent smoother state
//     (e.g. Jacobi diagonals, polynomial smoother eigenvalue estimates)
//     WITHOUT rebuilding the C/F graph. Deprecated per header but still
//     functional in v2.5.0 and is the ONLY mechanism for this use case.
//
// This is the K-amortization primitive: run setup once per K timesteps,
// update_coefficients on every timestep in between.

#include <amgx_c.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::amgx::PlanEntry;
using dilu::amgx::plan_cache_lookup;
using dilu::amgx::amgx_rc_str;
using dilu::amgx::handle_called_from_wrong_thread;

static ffi::Error AmgxUpdateCoefficientsImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status_out) {
  if (handle_called_from_wrong_thread()) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      "update_coefficients: called from non-owner thread");
  }
  if (token_buf.element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "update_coefficients: token must be uint64[1]");
  }
  if (status_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "update_coefficients: status output must be int32[1]");
  }

  uint64_t token_host = 0;
  {
    cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                    sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("update_coefficients: token copy: ") +
                            cudaGetErrorString(e));
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("update_coefficients: token sync: ") +
                            cudaGetErrorString(e));
    }
  }

  PlanEntry* entry = plan_cache_lookup(token_host);
  if (!entry) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "update_coefficients: unknown token");
  }

  const int64_t nnz = static_cast<int64_t>(values.element_count());
  if (nnz > INT32_MAX) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "update_coefficients: nnz exceeds int32");
  }
  if (entry->fingerprint.nnz != static_cast<int32_t>(nnz)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "update_coefficients: nnz fingerprint mismatch");
  }

  AMGX_RC rc = AMGX_matrix_replace_coefficients(
      entry->matrix,
      entry->fingerprint.n,
      entry->fingerprint.nnz,
      values.typed_data(),
      /*diag_data=*/nullptr);
  if (rc != AMGX_RC_OK) {
    int32_t status = static_cast<int32_t>(rc);
    cudaMemcpyAsync(status_out->typed_data(), &status, sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);  // status is stack-local; see C1 fix
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("update_coefficients: replace: ") +
                          amgx_rc_str(rc));
  }

  // Resetup: re-derives operator-dependent smoother state. Cost is
  // substantially less than full AMGX_solver_setup because C/F splitting and
  // interpolation are NOT recomputed.
  rc = AMGX_solver_resetup(entry->solver, entry->matrix);
  int32_t status = static_cast<int32_t>(rc);
  cudaError_t e = cudaMemcpyAsync(status_out->typed_data(), &status,
                                  sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("update_coefficients: status write: ") +
                          cudaGetErrorString(e));
  }
  e = cudaStreamSynchronize(stream);  // status is stack-local; see C1 fix
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("update_coefficients: status sync: ") +
                          cudaGetErrorString(e));
  }
  if (rc != AMGX_RC_OK) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("update_coefficients: resetup: ") +
                          amgx_rc_str(rc));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AmgxUpdateCoefficients, AmgxUpdateCoefficientsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // token[1]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // values[nnz]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // status[1]
