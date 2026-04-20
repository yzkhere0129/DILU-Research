// Forward color-stride triangular solve: (D̃_* + L̃) ỹ = r̃
//
// Math (arch §3.2 step 5, §4.1; math doc (1.8)):
//   For c = 0 .. n_colors - 1:
//     For every row i in color c (color_offsets[c] ≤ i < color_offsets[c+1]):
//        ỹ[i] = ( r̃[i] - Σ_{j < i, (i,j) ∈ L̃} values_tilde[...] * ỹ[j] ) / d̃_*[i]
//
// Correctness invariant (math doc §2.3 #3):
//   Rows in the same color share NO off-diagonal entry. Therefore within one
//   color-sweep kernel launch every row reads only `y[j]` with `j < i`, and
//   those `j` live in color classes 0 .. c-1 — which are already finalized by
//   previous launches. No atomics, no shared memory — pure fan-in per thread.
//
// Kernel launch policy (arch §4.5):
//   Host calls this once per color, in ascending color order. The implicit
//   grid-level sync between launches is what separates stage c from c+1.
//
// Launch shape: 1 thread per row in the color's range. Block 256, grid =
// ceil(color_count / 256).

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void color_stride_forward_kernel(
    const int32_t* __restrict__ row_ptr_tilde,
    const int32_t* __restrict__ col_idx_tilde,
    const double*  __restrict__ values_tilde,
    const double*  __restrict__ d_star_tilde,
    const double*  __restrict__ r_tilde,
    double*        __restrict__ y_tilde,
    int32_t color_offset,
    int32_t color_count) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= color_count) return;
  // Rows within one color are contiguous in the permuted ordering, so
  // row = color_offset + tid (arch §4.2 — coalesced write to y_tilde[row]).
  const int32_t row = color_offset + tid;

  double sum = r_tilde[row];
  const int32_t rs = row_ptr_tilde[row];
  const int32_t re = row_ptr_tilde[row + 1];
  // Strict lower triangle: j < row. Branch cost ~1 instruction (arch §4.4).
  for (int32_t p = rs; p < re; ++p) {
    const int32_t j = col_idx_tilde[p];
    if (j < row) {
      sum -= values_tilde[p] * y_tilde[j];
    }
  }
  y_tilde[row] = sum / d_star_tilde[row];
}

extern "C" cudaError_t launch_color_stride_forward(
    cudaStream_t stream,
    const int32_t* row_ptr_tilde,
    const int32_t* col_idx_tilde,
    const double*  values_tilde,
    const double*  d_star_tilde,
    const double*  r_tilde,
    double*        y_tilde,
    int32_t color_offset,
    int32_t color_count) {
  if (color_count <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (color_count + kBlock - 1) / kBlock;
  color_stride_forward_kernel<<<grid, kBlock, 0, stream>>>(
      row_ptr_tilde, col_idx_tilde, values_tilde, d_star_tilde,
      r_tilde, y_tilde, color_offset, color_count);
  return cudaGetLastError();
}
