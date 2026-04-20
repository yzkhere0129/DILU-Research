// MVP kernel: y[i] = diag[i] * (b[i] - sum_{k in row i} values[k] * x[col_idx[k]])
//
// Invariant (caller-enforced, not re-validated here to keep hot path tight):
//   diag[i] already equals 1.0 / A[i,i]. The design doc §2.2 deliberately
//   precomputes it in JAX so the kernel stays a pure SpMV + AXPBY.
//
// Threading: one thread per row. Phase 1 explicitly does not tune for warp
// efficiency; irregular nnz-per-row balancing is a Phase 2/3 concern.
//
// All loads on read-only buffers go through __ldg to pick up the read-only
// cache path on SM 8.6 (RTX 3050). This is a correctness-neutral hint.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void jacobi_residual_kernel(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ diag,
    const double* __restrict__ b,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const int32_t row_start = __ldg(row_ptr + i);
  const int32_t row_end = __ldg(row_ptr + i + 1);

  // Accumulate A[i,:] * x in a register. Deterministic left-to-right reduction
  // keeps results bitwise-comparable run-to-run (no -ffast-math; CLAUDE.md).
  double acc = 0.0;
  for (int32_t k = row_start; k < row_end; ++k) {
    const int32_t j = __ldg(col_idx + k);
    acc += __ldg(values + k) * __ldg(x + j);
  }

  y[i] = __ldg(diag + i) * (__ldg(b + i) - acc);
}

extern "C" cudaError_t launch_jacobi_residual(
    cudaStream_t stream,
    const int32_t* row_ptr,
    const int32_t* col_idx,
    const double* values,
    const double* diag,
    const double* b,
    const double* x,
    double* y,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((static_cast<int64_t>(n) + kBlock - 1) / kBlock);
  jacobi_residual_kernel<<<grid, kBlock, 0, stream>>>(
      row_ptr, col_idx, values, diag, b, x, y, n);
  return cudaGetLastError();
}
