// FFI handler: cusparse_dilu_apply.  z = M_DILU^{-1} r.
//
// Contract (arch doc §4.3):
//   Inputs:  token uint64[1], row_ptr int32[n+1], col_idx int32[nnz],
//            values f64[nnz], d_star f64[n], diag_offset int32[n], r f64[n]
//   Output:  z f64[n]
//
// Flow:
//   1. Read token (the ONE tolerated 8-byte D→H copy per apply).
//   2. Verify pattern fingerprint (row_ptr/col_idx pointers + n + nnz all match
//      the analyze-time snapshot). Mismatch → InvalidArgument.
//   3. cusparseSetStream to the XLA stream.
//   4. Build working_values = copy(values) with diagonal replaced by d_star.
//      Rebind mat_L and mat_U onto the updated working_values (in case of
//      buffer reuse this is a no-op, but safe).
//   5. Bind vec_r to `r`; forward SpSV solve:  (D_* + L) y_mid = r  (fill=LOWER).
//   6. y_scaled[i] = d_star[i] * y_mid[i]  (elementwise kernel).
//   7. Bind vec_y_scaled to y_scaled, vec_z to `z`; backward SpSV solve:
//      (D_* + U) z = y_scaled  (fill=UPPER).
//
// Math reference: phase2_cusparse_level_scheduling_math.md §2.2.

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::cusparse::PlanEntry;
using dilu::cusparse::get_handle;
using dilu::cusparse::plan_cache_lookup;
using dilu::cusparse::handle_called_from_wrong_thread;
using dilu::cusparse::cusparse_status_str;

extern "C" cudaError_t launch_build_working_values(
    cudaStream_t stream,
    const double* values,
    double* working_values,
    const int32_t* diag_offset,
    const double* d_star,
    int32_t n,
    int32_t nnz);

extern "C" cudaError_t launch_elem_scale(
    cudaStream_t stream,
    const double* d_star,
    const double* y,
    double* y_scaled,
    int32_t n);

#define CHECK_CUDA_APPLY(expr)                                            \
  do {                                                                     \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cuda: ") + cudaGetErrorString(_e));  \
    }                                                                      \
  } while (0)

#define CHECK_CUSPARSE_APPLY(expr)                                        \
  do {                                                                     \
    cusparseStatus_t _s = (expr);                                          \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                   \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cusparse: ") +                        \
                            cusparse_status_str(_s));                      \
    }                                                                      \
  } while (0)

static ffi::Error CusparseDiluApplyImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Buffer<ffi::DataType::F64> d_star,
    ffi::Buffer<ffi::DataType::F64> r,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> z) {
  // Thread-affinity tripwire (arch §2.6).
  if (handle_called_from_wrong_thread()) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      "apply: cuSPARSE handle called from non-owner thread");
  }

  // 8-byte D→H copy of the opaque token. This is the ONE tolerated host copy
  // per apply (arch §2.3.2). Synchronous here because we need the token
  // before any further dispatch — we cannot schedule SpSV without the plan.
  if (token_buf.element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "apply: token must be uint64[1]");
  }
  uint64_t token_host = 0;
  {
    cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                    sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("apply: token copy: ") +
                            cudaGetErrorString(e));
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("apply: token sync: ") +
                            cudaGetErrorString(e));
    }
  }

  PlanEntry* entry = plan_cache_lookup(token_host);
  if (!entry) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "apply: unknown token (plan not in cache)");
  }

  // Shape + fingerprint checks.
  const size_t n_sz = z->element_count();
  const size_t nnz_sz = values.element_count();
  if (n_sz > static_cast<size_t>(INT32_MAX) ||
      nnz_sz > static_cast<size_t>(INT32_MAX)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "apply: n or nnz exceeds int32");
  }
  const int32_t n = static_cast<int32_t>(n_sz);
  const int32_t nnz = static_cast<int32_t>(nnz_sz);

  if (d_star.element_count() != static_cast<size_t>(n) ||
      r.element_count() != static_cast<size_t>(n)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "apply: shape mismatch among vec inputs");
  }

  if (entry->fingerprint.n != n || entry->fingerprint.nnz != nnz) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "apply: token/(n,nnz) fingerprint mismatch — "
                      "use Plan.apply with matching matrix");
  }

  // Bind the handle to the XLA stream (per-call; cuSPARSE rule).
  cusparseHandle_t handle = nullptr;
  {
    cusparseStatus_t s = get_handle(&handle);
    if (s != CUSPARSE_STATUS_SUCCESS) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("apply: get_handle: ") +
                            cusparse_status_str(s));
    }
  }
  CHECK_CUSPARSE_APPLY(cusparseSetStream(handle, stream));

  // Build working_values = copy(A.values) with diagonal overwritten by d_star.
  // This is the "scatter diag" trap (arch §4.2): cuSPARSE reads diagonal from
  // the CSR values slot, so we must splice d_star in. `diag_offset` was
  // captured and copied at analyze time; using the plan-owned copy insulates
  // us from any XLA buffer shuffling.
  CHECK_CUDA_APPLY(launch_build_working_values(
      stream, values.typed_data(), entry->working_values,
      entry->diag_offset, d_star.typed_data(), n, nnz));

  // IMPORTANT cuSPARSE subtlety (arch §4.2, caught in Phase 2 T6 bring-up):
  // cusparseSpSV_analysis caches the values (notably the diagonal as its
  // reciprocal for the divides) inside the SpSVDescr. Merely overwriting
  // working_values in place does NOT reach the analysis cache.
  // `cusparseSpSV_updateMatrix` is the documented mechanism for refreshing
  // the values after analysis without redoing O(nnz) analysis work. We use
  // UPDATE_GENERAL because both the caller's `values` (off-diagonals) and
  // our scattered diagonal may have changed since analyze time.
  CHECK_CUSPARSE_APPLY(cusparseSpSV_updateMatrix(
      handle, entry->spsv_L, entry->working_values,
      CUSPARSE_SPSV_UPDATE_GENERAL));
  CHECK_CUSPARSE_APPLY(cusparseSpSV_updateMatrix(
      handle, entry->spsv_U, entry->working_values,
      CUSPARSE_SPSV_UPDATE_GENERAL));

  // Bind r and y_mid to the dense-vec descriptors.
  CHECK_CUSPARSE_APPLY(cusparseDnVecSetValues(
      entry->vec_r, const_cast<double*>(r.typed_data())));
  CHECK_CUSPARSE_APPLY(cusparseDnVecSetValues(entry->vec_y_mid, entry->y_mid));

  const double alpha = 1.0;
  // Forward solve: (D_* + L) y_mid = r.
  CHECK_CUSPARSE_APPLY(cusparseSpSV_solve(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_L, entry->vec_r, entry->vec_y_mid,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_L));

  // Middle: y_scaled = D_* .* y_mid.
  CHECK_CUDA_APPLY(launch_elem_scale(
      stream, d_star.typed_data(), entry->y_mid, entry->y_scaled, n));

  CHECK_CUSPARSE_APPLY(cusparseDnVecSetValues(entry->vec_y_scaled, entry->y_scaled));
  CHECK_CUSPARSE_APPLY(cusparseDnVecSetValues(entry->vec_z, z->typed_data()));

  // Backward solve: (D_* + U) z = y_scaled.
  CHECK_CUSPARSE_APPLY(cusparseSpSV_solve(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_U, entry->vec_y_scaled, entry->vec_z,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_U));

  return ffi::Error::Success();
}

#undef CHECK_CUDA_APPLY
#undef CHECK_CUSPARSE_APPLY

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusparseDiluApply, CusparseDiluApplyImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // token[1]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // values
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // d_star
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // r
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // z
