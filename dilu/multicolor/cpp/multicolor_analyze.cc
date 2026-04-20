// FFI handler: multicolor_analyze.
//
// Contract (arch doc §3.1). Python has already:
//   1. Run red-black OR greedy coloring + validate_coloring on host.
//   2. Permuted the CSR via permute_csr → (row_ptr_tilde, col_idx_tilde,
//      values_tilde, nnz_map) with ascending columns per row.
//   3. Computed diag_offset_tilde on the permuted CSR.
//
// So analyze's job is narrowly scoped:
//   (a) Allocate plan-owned device buffers for all 11 arrays.
//   (b) Copy each input buffer (all device-resident) into plan-owned memory.
//   (c) Allocate scratch vectors (d_star_tilde, r_tilde, y_tilde,
//       y_scaled_tilde, z_tilde).
//   (d) Run the Phase 2 `dilu_factor_kernel` on (row_ptr_tilde, col_idx_tilde,
//       values_tilde, diag_offset_tilde) to seed d_star_tilde.
//   (e) Insert into the plan cache; write the opaque token to output.
//
// We deliberately do NOT do coloring or permutation on device. Arch §2.6
// justifies host-side coloring (static grid, one-shot cost, simpler
// correctness validation).
//
// INPUT buffer layout (all device, int32 unless noted, C-contiguous):
//   row_ptr_tilde      int32[N+1]
//   col_idx_tilde      int32[nnz]
//   values_tilde       f64  [nnz]
//   diag_offset_tilde  int32[N]
//   perm               int32[N]
//   iperm              int32[N]
//   color_offsets      int32[n_colors + 1]
//   nnz_map            int32[nnz]
// OUTPUT: token uint64[1]

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::multicolor::MulticolorPlanEntry;
using dilu::multicolor::PatternFingerprint;
using dilu::multicolor::destroy_plan_entry;
using dilu::multicolor::plan_cache_insert;

extern "C" cudaError_t launch_dilu_factor(
    cudaStream_t stream,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values,
    const int32_t* diag_offset,
    double* d_star,
    int32_t n);

#define CHECK_CUDA(expr)                                                    \
  do {                                                                       \
    cudaError_t _e = (expr);                                                 \
    if (_e != cudaSuccess) {                                                 \
      destroy_plan_entry(entry);                                             \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                         \
                        std::string("multicolor_analyze: ") +                \
                            cudaGetErrorString(_e));                          \
    }                                                                        \
  } while (0)

static ffi::Error MulticolorAnalyzeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> row_ptr_tilde,
    ffi::Buffer<ffi::DataType::S32> col_idx_tilde,
    ffi::Buffer<ffi::DataType::F64> values_tilde,
    ffi::Buffer<ffi::DataType::S32> diag_offset_tilde,
    ffi::Buffer<ffi::DataType::S32> perm,
    ffi::Buffer<ffi::DataType::S32> iperm,
    ffi::Buffer<ffi::DataType::S32> color_offsets,
    ffi::Buffer<ffi::DataType::S32> nnz_map,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> token_out) {
  const size_t n_plus_1 = row_ptr_tilde.element_count();
  if (n_plus_1 < 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: row_ptr_tilde empty");
  }
  const int64_t N = static_cast<int64_t>(n_plus_1 - 1);
  const int64_t nnz = static_cast<int64_t>(values_tilde.element_count());
  const int64_t n_colors_plus_1 = static_cast<int64_t>(color_offsets.element_count());

  if (col_idx_tilde.element_count() != static_cast<size_t>(nnz) ||
      nnz_map.element_count() != static_cast<size_t>(nnz)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: nnz-length mismatch "
                      "(col_idx/values/nnz_map)");
  }
  if (diag_offset_tilde.element_count() != static_cast<size_t>(N) ||
      perm.element_count() != static_cast<size_t>(N) ||
      iperm.element_count() != static_cast<size_t>(N)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: N-length mismatch "
                      "(diag_offset/perm/iperm)");
  }
  if (n_colors_plus_1 < 2) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: need at least 1 color");
  }
  if (N > INT32_MAX || nnz > INT32_MAX) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: N or nnz exceeds int32");
  }
  if (token_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: token output must be uint64[1]");
  }

  const int32_t Ni = static_cast<int32_t>(N);
  const int32_t nnzi = static_cast<int32_t>(nnz);
  const int32_t n_colors = static_cast<int32_t>(n_colors_plus_1 - 1);

  MulticolorPlanEntry* entry = new MulticolorPlanEntry();
  entry->fingerprint = PatternFingerprint{Ni, nnzi, n_colors};

  const size_t row_ptr_bytes = (Ni + 1) * sizeof(int32_t);
  const size_t col_idx_bytes = nnzi * sizeof(int32_t);
  const size_t values_bytes  = nnzi * sizeof(double);
  const size_t diag_off_bytes = Ni * sizeof(int32_t);
  const size_t perm_bytes = Ni * sizeof(int32_t);
  const size_t color_offsets_bytes = (n_colors + 1) * sizeof(int32_t);
  const size_t vec_bytes = Ni * sizeof(double);

  // Allocate + copy pattern & metadata. All async on the XLA stream.
  CHECK_CUDA(cudaMallocAsync((void**)&entry->row_ptr_tilde, row_ptr_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->row_ptr_tilde, row_ptr_tilde.typed_data(),
                             row_ptr_bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->col_idx_tilde, col_idx_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->col_idx_tilde, col_idx_tilde.typed_data(),
                             col_idx_bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->values_tilde, values_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->values_tilde, values_tilde.typed_data(),
                             values_bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->diag_offset_tilde, diag_off_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->diag_offset_tilde, diag_offset_tilde.typed_data(),
                             diag_off_bytes, cudaMemcpyDeviceToDevice, stream));

  CHECK_CUDA(cudaMallocAsync((void**)&entry->perm, perm_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->perm, perm.typed_data(),
                             perm_bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->iperm, perm_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->iperm, iperm.typed_data(),
                             perm_bytes, cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->color_offsets, color_offsets_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->color_offsets, color_offsets.typed_data(),
                             color_offsets_bytes, cudaMemcpyDeviceToDevice, stream));
  // Mirror color_offsets to host NOW so apply() launches per-color kernels
  // without a D→H copy. Cap at 64 colors (arch §2.2 worst-case 7-20).
  if (n_colors + 1 > 64) {
    destroy_plan_entry(entry);
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "multicolor_analyze: too many colors (>63) — hard cap");
  }
  CHECK_CUDA(cudaMemcpyAsync(
      entry->color_offsets_host, color_offsets.typed_data(),
      color_offsets_bytes, cudaMemcpyDeviceToHost, stream));
  // Need the host copy available when apply() runs — the stream sync happens
  // naturally before the token is returned (we cudaStreamSynchronize inside
  // the handler's token write? No — analyze writes the token asynchronously
  // too). Do an explicit sync here so the host mirror is valid when we
  // return.
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->nnz_map, col_idx_bytes, stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->nnz_map, nnz_map.typed_data(),
                             col_idx_bytes, cudaMemcpyDeviceToDevice, stream));

  // Scratch vectors.
  CHECK_CUDA(cudaMallocAsync((void**)&entry->d_star_tilde,   vec_bytes, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->r_tilde,        vec_bytes, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->y_tilde,        vec_bytes, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->y_scaled_tilde, vec_bytes, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->z_tilde,        vec_bytes, stream));

  // Seed D̃_* by running Phase 2's verbatim `dilu_factor_kernel` on Ã.
  CHECK_CUDA(launch_dilu_factor(
      stream,
      entry->row_ptr_tilde, entry->col_idx_tilde,
      entry->values_tilde, entry->diag_offset_tilde,
      entry->d_star_tilde, Ni));

  uint64_t token = plan_cache_insert(entry);
  CHECK_CUDA(cudaMemcpyAsync(token_out->typed_data(), &token, sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));
  return ffi::Error::Success();
}

#undef CHECK_CUDA

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MulticolorAnalyze, MulticolorAnalyzeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // row_ptr_tilde
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // col_idx_tilde
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // values_tilde
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // diag_offset_tilde
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // perm
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // iperm
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // color_offsets
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // nnz_map
        .Ret<ffi::Buffer<ffi::DataType::U64>>()); // token[1]
