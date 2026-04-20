// Element-wise y_scaled[i] = d_star[i] * y[i].
//
// Phase 3 middle step between the forward and backward color-stride sweeps
// (arch §3.2 step 6). Semantically identical to the Phase 2 kernel of the
// same name in `dilu/cusparse/cuda/scatter_diag_kernel.cu`; copied here to
// keep `libdilu_multicolor.so` independent of Phase 2 artifacts.
//
// NB: we deliberately do NOT link the Phase 2 library. The math is the same
// but the artifact dependency graph stays clean (arch §6.1 / §6.6).

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void elem_scale_kernel_mc(
    const double* __restrict__ d_star,
    const double* __restrict__ y,
    double* __restrict__ y_scaled,
    int32_t n) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y_scaled[i] = d_star[i] * y[i];
}

extern "C" cudaError_t launch_elem_scale_mc(
    cudaStream_t stream,
    const double* d_star,
    const double* y,
    double* y_scaled,
    int32_t n) {
  if (n <= 0) return cudaSuccess;
  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  elem_scale_kernel_mc<<<grid, kBlock, 0, stream>>>(d_star, y, y_scaled, n);
  return cudaGetLastError();
}
