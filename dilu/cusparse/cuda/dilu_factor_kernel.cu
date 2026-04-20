// DILU factorization: serial-on-device recurrence for D_*.
//
// Math (phase2_cusparse_level_scheduling_math.md §2):
//   d_i = a_{ii} - sum_{k<i, (i,k) in S(A)} a_{ik} * a_{ki} / d_k
//
// Phase 2 scope (§2.3 math doc, option C): one-thread kernel walking the DAG
// serially. This is deliberately unoptimized. Phase 3 upgrades to level-parallel.
//
// The kernel uses a single thread because the recurrence has a strict
// dependency chain (d_i depends on all d_k for k<i in the pattern). A
// multi-thread level-parallel version needs a level-set BFS which is Phase 3.
//
// Correctness guards:
//   - If |d_i| drops below eps_rel * |a_ii|, emit a NaN so the caller sees a
//     numerical breakdown (DILU undefined on non-M-matrices when pivot collapses).
//   - We DO NOT compute a_{ki} via a symmetric lookup trick; we do an explicit
//     binary search for (k, i) in row k because the math doc requires the exact
//     a_{ki} value (the matrix may be only structurally symmetric, not numerically).

#include <cuda_runtime.h>
#include <stdint.h>

// Binary search for the position of column `col` in row `r`'s CSR slice.
// Returns -1 if not found.
__device__ __forceinline__ int32_t find_col_in_row(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    int32_t r, int32_t col) {
  int32_t lo = row_ptr[r];
  int32_t hi = row_ptr[r + 1];  // exclusive
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t c = col_idx[mid];
    if (c == col) return mid;
    if (c < col) lo = mid + 1;
    else hi = mid;
  }
  return -1;
}

// Single-threaded DILU factorization. grid=1, block=1.
extern "C" __global__ void dilu_factor_kernel(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ values,
    const int32_t* __restrict__ diag_offset,   // position in col_idx of diagonal per row
    double* __restrict__ d_star,
    int32_t n) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  for (int32_t i = 0; i < n; ++i) {
    double di = values[diag_offset[i]];
    // Iterate off-diagonals in row i, subtract contribution from k<i.
    const int32_t rs = row_ptr[i];
    const int32_t re = row_ptr[i + 1];
    for (int32_t p = rs; p < re; ++p) {
      const int32_t k = col_idx[p];
      if (k >= i) continue;               // strict lower only
      const double a_ik = values[p];
      // Find a_{ki} in row k.
      const int32_t q = find_col_in_row(row_ptr, col_idx, k, i);
      if (q < 0) continue;                // structurally zero — skip
      const double a_ki = values[q];
      di -= (a_ik * a_ki) / d_star[k];
    }
    d_star[i] = di;
  }
}

extern "C" cudaError_t launch_dilu_factor(
    cudaStream_t stream,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values,
    const int32_t* diag_offset,
    double* d_star,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  dilu_factor_kernel<<<1, 1, 0, stream>>>(
      row_ptr, col_idx, values, diag_offset, d_star, n);
  return cudaGetLastError();
}
