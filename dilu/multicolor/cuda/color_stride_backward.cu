// Backward color-stride triangular solve: (D̃_* + Ũ) z̃ = D̃_* ỹ
//
// Mirror of `color_stride_forward_kernel`. Differences (math doc (1.9)):
//   (a) Iterate colors from n_colors-1 down to 0 (host-side launch order).
//   (b) Filter `j > row` instead of `j < row` (strict upper triangle).
//   (c) RHS is `y_scaled_tilde[row]` (= d̃_* ỹ), NOT `r_tilde`.
//   (d) Writes z̃.
//
// Same parallelism guarantee: rows within one color share no off-diagonal
// entries, so per-thread reads of `z[j]` only touch already-finalized rows
// from strictly-later colors. Synchronization is the implicit kernel-launch
// boundary between colors.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void color_stride_backward_kernel(
    const int32_t* __restrict__ row_ptr_tilde,
    const int32_t* __restrict__ col_idx_tilde,
    const double*  __restrict__ values_tilde,
    const double*  __restrict__ d_star_tilde,
    const double*  __restrict__ y_scaled_tilde,
    double*        __restrict__ z_tilde,
    int32_t color_offset,
    int32_t color_count) {
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= color_count) return;
  const int32_t row = color_offset + tid;

  double sum = y_scaled_tilde[row];
  const int32_t rs = row_ptr_tilde[row];
  const int32_t re = row_ptr_tilde[row + 1];
  for (int32_t p = rs; p < re; ++p) {
    const int32_t j = col_idx_tilde[p];
    if (j > row) {
      sum -= values_tilde[p] * z_tilde[j];
    }
  }
  z_tilde[row] = sum / d_star_tilde[row];
}

extern "C" cudaError_t launch_color_stride_backward(
    cudaStream_t stream,
    const int32_t* row_ptr_tilde,
    const int32_t* col_idx_tilde,
    const double*  values_tilde,
    const double*  d_star_tilde,
    const double*  y_scaled_tilde,
    double*        z_tilde,
    int32_t color_offset,
    int32_t color_count) {
  if (color_count <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (color_count + kBlock - 1) / kBlock;
  color_stride_backward_kernel<<<grid, kBlock, 0, stream>>>(
      row_ptr_tilde, col_idx_tilde, values_tilde, d_star_tilde,
      y_scaled_tilde, z_tilde, color_offset, color_count);
  return cudaGetLastError();
}
