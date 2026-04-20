// Gather / scatter kernels used at apply time to cross the permutation
// boundary (arch §3.2).
//
// Contract:
//   - `gather_by_perm`    : dst[new_idx] = src[perm[new_idx]]   (forward)
//   - `scatter_by_iperm`  : dst[old_idx] = src[iperm[old_idx]]  (inverse out)
//       Equivalent to dst[perm[new_idx]] = src[new_idx] — we pick the
//       `iperm[old_idx]` formulation so every thread writes a unique dst
//       slot (no contention).
//   - `gather_values_by_nnz_map` : values_tilde[new_k] = values[nnz_map[new_k]]
//
// All kernels are embarrassingly parallel — one thread per destination slot.

#include <cuda_runtime.h>
#include <stdint.h>

// ---- Vector-length (N) gather --------------------------------------------
extern "C" __global__ void gather_by_perm_kernel(
    const double* __restrict__ src,
    const int32_t* __restrict__ perm,
    double* __restrict__ dst,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[perm[i]];
}

extern "C" cudaError_t launch_gather_by_perm(
    cudaStream_t stream,
    const double* src,
    const int32_t* perm,
    double* dst,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  gather_by_perm_kernel<<<grid, kBlock, 0, stream>>>(src, perm, dst, n);
  return cudaGetLastError();
}

// ---- Inverse-permute out (used for z = P⁻¹ z̃) --------------------------
// dst[i] = src[iperm[i]]: every thread reads from the unique new_idx slot
// assigned to old_idx i.
extern "C" __global__ void scatter_by_iperm_kernel(
    const double* __restrict__ src,
    const int32_t* __restrict__ iperm,
    double* __restrict__ dst,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[iperm[i]];
}

extern "C" cudaError_t launch_scatter_by_iperm(
    cudaStream_t stream,
    const double* src,
    const int32_t* iperm,
    double* dst,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  scatter_by_iperm_kernel<<<grid, kBlock, 0, stream>>>(src, iperm, dst, n);
  return cudaGetLastError();
}

// ---- nnz-length gather (used to rehydrate values_tilde per apply) --------
extern "C" __global__ void gather_values_by_nnz_map_kernel(
    const double* __restrict__ src_values,
    const int32_t* __restrict__ nnz_map,
    double* __restrict__ dst_values_tilde,
    int32_t nnz) {
  const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nnz) return;
  dst_values_tilde[k] = src_values[nnz_map[k]];
}

extern "C" cudaError_t launch_gather_values_by_nnz_map(
    cudaStream_t stream,
    const double* src_values,
    const int32_t* nnz_map,
    double* dst_values_tilde,
    int32_t nnz) {
  if (nnz <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (nnz + kBlock - 1) / kBlock;
  gather_values_by_nnz_map_kernel<<<grid, kBlock, 0, stream>>>(
      src_values, nnz_map, dst_values_tilde, nnz);
  return cudaGetLastError();
}
