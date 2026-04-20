// FFI handler: multicolor_apply. Hot path. z = M_mcDILU^{-1} r.
//
// Contract (arch doc §3.2). Caller's I/O vectors live in ORIGINAL ordering;
// the handler permutes in, runs the color-stride solve, inverse-permutes out.
//
// Flow:
//   1. 8-byte D→H copy of the opaque token (arch §2.3.2 tolerated exception).
//   2. Plan lookup + fingerprint check (N, nnz, n_colors).
//   3. Gather values_tilde from original `values` via nnz_map.
//   4. Gather r_tilde from original `r` via perm.
//      (d_star_tilde already lives in the plan — refreshed by refactor, if any.)
//      The caller supplies `d_star` in ORIGINAL ordering too; we pull it into
//      the plan's d_star_tilde via perm. This also lets a caller who computed
//      d_star externally (e.g. with Phase 2's dilu_factor, if they wanted)
//      feed it in, though for Phase 3 the refactor path is canonical.
//   5. Forward sweep: loop colors 0..n_colors-1, one kernel launch each.
//   6. elem_scale: y_scaled_tilde = d_star_tilde * y_tilde.
//   7. Backward sweep: loop colors n_colors-1..0, one kernel launch each.
//   8. Scatter z_tilde to z (caller-visible) via iperm.
//
// Kernel-launch count per apply: 2 * n_colors + 4 (3 gathers + 1 elem_scale)
// For red-black (n_colors = 2): 8 launches per apply.

#include <cuda_runtime.h>
#include <stdint.h>

#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::multicolor::MulticolorPlanEntry;
using dilu::multicolor::plan_cache_lookup;
using dilu::multicolor::plan_map_called_from_wrong_thread;

// Forward decls — all defined in dilu/multicolor/cuda/*.cu.
extern "C" cudaError_t launch_gather_by_perm(
    cudaStream_t, const double*, const int32_t*, double*, int32_t);
extern "C" cudaError_t launch_scatter_by_iperm(
    cudaStream_t, const double*, const int32_t*, double*, int32_t);
extern "C" cudaError_t launch_gather_values_by_nnz_map(
    cudaStream_t, const double*, const int32_t*, double*, int32_t);
extern "C" cudaError_t launch_elem_scale_mc(
    cudaStream_t, const double*, const double*, double*, int32_t);
extern "C" cudaError_t launch_color_stride_forward(
    cudaStream_t, const int32_t*, const int32_t*, const double*,
    const double*, const double*, double*, int32_t, int32_t);
extern "C" cudaError_t launch_color_stride_backward(
    cudaStream_t, const int32_t*, const int32_t*, const double*,
    const double*, const double*, double*, int32_t, int32_t);

#define CHECK(expr)                                                         \
  do {                                                                       \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                 \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                         \
                        std::string("multicolor_apply: ") +                  \
                            cudaGetErrorString(_e));                          \
    }                                                                        \
  } while (0)

static ffi::Error MulticolorApplyImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Buffer<ffi::DataType::F64> d_star,
    ffi::Buffer<ffi::DataType::F64> r,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> z) {
  if (plan_map_called_from_wrong_thread()) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      "multicolor_apply: called from non-owner thread");
  }
  if (token_buf.element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_apply: token must be uint64[1]");
  }

  // (1) 8-byte D→H token copy. Blocks on stream.
  uint64_t token_host = 0;
  {
    cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                    sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("multicolor_apply: token copy: ") +
                            cudaGetErrorString(e));
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                        std::string("multicolor_apply: token sync: ") +
                            cudaGetErrorString(e));
    }
  }

  // (2) Plan lookup + fingerprint.
  MulticolorPlanEntry* entry = plan_cache_lookup(token_host);
  if (!entry) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_apply: unknown token (plan not in cache)");
  }

  const size_t n_sz = z->element_count();
  const size_t nnz_sz = values.element_count();
  if (n_sz > static_cast<size_t>(INT32_MAX) ||
      nnz_sz > static_cast<size_t>(INT32_MAX)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_apply: N or nnz exceeds int32");
  }
  const int32_t N = static_cast<int32_t>(n_sz);
  const int32_t nnz = static_cast<int32_t>(nnz_sz);
  if (d_star.element_count() != static_cast<size_t>(N) ||
      r.element_count() != static_cast<size_t>(N)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_apply: shape mismatch among vec inputs");
  }
  if (entry->fingerprint.n != N || entry->fingerprint.nnz != nnz) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_apply: (N, nnz) fingerprint mismatch");
  }
  const int32_t n_colors = entry->fingerprint.n_colors;

  // (3) values_tilde[k] = values[nnz_map[k]].
  CHECK(launch_gather_values_by_nnz_map(
      stream, values.typed_data(), entry->nnz_map, entry->values_tilde, nnz));
  // (4a) r_tilde[new] = r[perm[new]].
  CHECK(launch_gather_by_perm(
      stream, r.typed_data(), entry->perm, entry->r_tilde, N));
  // (4b) d_star_tilde[new] = d_star[perm[new]]. Overwrites the refactor seed,
  // which is what we want — the caller's d_star is the authoritative source.
  // If the caller keeps calling apply with the same d_star that multicolor_refactor
  // returned, this is an identity re-permute; we eat the launch cost (≤5 µs)
  // for cleanliness. (Optimization: cache a `d_star_permuted_is_fresh` flag.
  // Deferred per arch §4.4.)
  CHECK(launch_gather_by_perm(
      stream, d_star.typed_data(), entry->perm, entry->d_star_tilde, N));

  // (5) Forward sweep — one kernel launch per color, ascending.
  // color_offsets were mirrored to host at analyze time → no D→H copy here.
  // The 8-byte token copy above remains the ONLY tolerated D→H per apply.
  const int32_t* const offsets_host = entry->color_offsets_host;

  for (int32_t c = 0; c < n_colors; ++c) {
    const int32_t off = offsets_host[c];
    const int32_t cnt = offsets_host[c + 1] - off;
    CHECK(launch_color_stride_forward(
        stream,
        entry->row_ptr_tilde, entry->col_idx_tilde, entry->values_tilde,
        entry->d_star_tilde, entry->r_tilde, entry->y_tilde,
        off, cnt));
  }

  // (6) Middle scale: y_scaled = d_star_tilde ⊙ y_tilde.
  CHECK(launch_elem_scale_mc(
      stream, entry->d_star_tilde, entry->y_tilde, entry->y_scaled_tilde, N));

  // (7) Backward sweep — one kernel launch per color, descending.
  for (int32_t c = n_colors - 1; c >= 0; --c) {
    const int32_t off = offsets_host[c];
    const int32_t cnt = offsets_host[c + 1] - off;
    CHECK(launch_color_stride_backward(
        stream,
        entry->row_ptr_tilde, entry->col_idx_tilde, entry->values_tilde,
        entry->d_star_tilde, entry->y_scaled_tilde, entry->z_tilde,
        off, cnt));
  }

  // (8) Inverse permute out: z[old] = z_tilde[iperm[old]].
  CHECK(launch_scatter_by_iperm(
      stream, entry->z_tilde, entry->iperm, z->typed_data(), N));
  return ffi::Error::Success();
}

#undef CHECK

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MulticolorApply, MulticolorApplyImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // token[1]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // values (original ordering)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // d_star (original ordering)
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // r      (original ordering)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()); // z     (original ordering)
