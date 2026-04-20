// FFI handler: multicolor_refactor. Values-only update path.
//
// Contract (arch doc §3.3):
//   When A's values change but pattern and coloring don't, recompute D̃_*
//   without re-running the analyze path.
//
// Steps:
//   1. Read token (8-byte D→H).
//   2. Gather plan-owned values_tilde from the new `values` via nnz_map.
//   3. Run Phase 2 `dilu_factor_kernel` on the permuted CSR to produce
//      d_star_tilde (permuted ordering).
//   4. Inverse-permute d_star_tilde → d_star_out in ORIGINAL ordering for the
//      caller's consumption (matches `dilu_factor`'s contract: d_star lives
//      in original index space).
//
// Why return in original ordering: apply() also accepts d_star in original
// ordering (it permutes internally via perm). So the caller's PCG driver
// threads a single d_star through refactor → apply → apply → ... without
// ever touching the permuted layout.

#include <cuda_runtime.h>
#include <stdint.h>

#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::multicolor::MulticolorPlanEntry;
using dilu::multicolor::plan_cache_lookup;

extern "C" cudaError_t launch_dilu_factor(
    cudaStream_t, const int32_t*, const int32_t*, const double*,
    const int32_t*, double*, int32_t);
extern "C" cudaError_t launch_gather_values_by_nnz_map(
    cudaStream_t, const double*, const int32_t*, double*, int32_t);
extern "C" cudaError_t launch_scatter_by_iperm(
    cudaStream_t, const double*, const int32_t*, double*, int32_t);

#define CHECK(expr)                                                         \
  do {                                                                       \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                 \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                         \
                        std::string("multicolor_refactor: ") +               \
                            cudaGetErrorString(_e));                          \
    }                                                                        \
  } while (0)

static ffi::Error MulticolorRefactorImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> d_star_out) {
  if (token_buf.element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_refactor: token must be uint64[1]");
  }

  uint64_t token_host = 0;
  {
    cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                    sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("multicolor_refactor: token copy: ") +
                            cudaGetErrorString(e));
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("multicolor_refactor: token sync: ") +
                            cudaGetErrorString(e));
    }
  }

  MulticolorPlanEntry* entry = plan_cache_lookup(token_host);
  if (!entry) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_refactor: unknown token");
  }

  const int32_t N = entry->fingerprint.n;
  const int32_t nnz = entry->fingerprint.nnz;
  if (values.element_count() != static_cast<size_t>(nnz)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_refactor: values length != plan nnz");
  }
  if (d_star_out->element_count() != static_cast<size_t>(N)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_refactor: d_star_out length != N");
  }

  // 2. values_tilde ← gather by nnz_map.
  CHECK(launch_gather_values_by_nnz_map(
      stream, values.typed_data(), entry->nnz_map,
      entry->values_tilde, nnz));
  // 3. Run Phase 2 DILU factor on the permuted CSR to produce d_star_tilde.
  CHECK(launch_dilu_factor(
      stream,
      entry->row_ptr_tilde, entry->col_idx_tilde,
      entry->values_tilde, entry->diag_offset_tilde,
      entry->d_star_tilde, N));
  // 4. Inverse-permute d_star_tilde → d_star_out in original ordering.
  //    d_star_out[old] = d_star_tilde[iperm[old]].
  CHECK(launch_scatter_by_iperm(
      stream, entry->d_star_tilde, entry->iperm,
      d_star_out->typed_data(), N));
  return ffi::Error::Success();
}

#undef CHECK

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MulticolorRefactor, MulticolorRefactorImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // token[1]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // values (original ordering)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // d_star (original ordering)
