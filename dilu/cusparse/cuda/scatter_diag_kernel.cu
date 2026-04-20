// Scatter D_* onto the diagonal of a working copy of A's CSR values.
//
// Why needed (arch doc §4.2): cuSPARSE SpSV reads the diagonal from the CSR
// values array. DILU requires using d_i (not a_ii) at the diagonal, so we
// produce a working_values = values with working_values[diag_offset[i]] = d_i.
//
// The rest of the entries must equal `values[k]` verbatim (L_strict and
// U_strict are reused from A untouched).

#include <cuda_runtime.h>
#include <stdint.h>

// Copy values into working_values. One thread per nnz entry.
extern "C" __global__ void copy_values_kernel(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int32_t nnz) {
  const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nnz) return;
  dst[k] = src[k];
}

// Overwrite diagonal positions with d_star. One thread per row.
extern "C" __global__ void scatter_diag_kernel(
    double* __restrict__ working_values,
    const int32_t* __restrict__ diag_offset,
    const double* __restrict__ d_star,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  working_values[diag_offset[i]] = d_star[i];
}

// Element-wise y_scaled[i] = d_star[i] * y[i]. Used between forward and
// backward SpSV to form the RHS of the U* solve (math doc §2.2 step 4).
extern "C" __global__ void elem_scale_kernel(
    const double* __restrict__ d_star,
    const double* __restrict__ y,
    double* __restrict__ y_scaled,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y_scaled[i] = d_star[i] * y[i];
}

extern "C" cudaError_t launch_build_working_values(
    cudaStream_t stream,
    const double* values,
    double* working_values,
    const int32_t* diag_offset,
    const double* d_star,
    int32_t n,
    int32_t nnz) {
  if (n <= 0 || nnz <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid_nnz = (nnz + kBlock - 1) / kBlock;
  const int grid_n = (n + kBlock - 1) / kBlock;
  copy_values_kernel<<<grid_nnz, kBlock, 0, stream>>>(values, working_values, nnz);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  scatter_diag_kernel<<<grid_n, kBlock, 0, stream>>>(
      working_values, diag_offset, d_star, n);
  return cudaGetLastError();
}

extern "C" cudaError_t launch_elem_scale(
    cudaStream_t stream,
    const double* d_star,
    const double* y,
    double* y_scaled,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  elem_scale_kernel<<<grid, kBlock, 0, stream>>>(d_star, y, y_scaled, n);
  return cudaGetLastError();
}
