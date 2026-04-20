// FFI handler: cusparse_dilu_release.  Frees the plan associated with `token`.
//
// Contract (arch §4.4): idempotent; releasing a gone token is a warning, not
// an error.  Returns a dummy int32[1] status so JAX can thread it as a
// dependency through the graph.

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::cusparse::plan_cache_remove;

static ffi::Error CusparseDiluReleaseImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status_out) {
  if (token_buf.element_count() != 1 || status_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "release: token must be uint64[1], status int32[1]");
  }
  uint64_t token_host = 0;
  cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                  sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("release: token copy: ") +
                          cudaGetErrorString(e));
  }
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("release: token sync: ") +
                          cudaGetErrorString(e));
  }
  bool removed = plan_cache_remove(token_host);
  // Write status (1=released, 0=not-present) to output.
  int32_t status = removed ? 1 : 0;
  e = cudaMemcpyAsync(status_out->typed_data(), &status, sizeof(int32_t),
                      cudaMemcpyHostToDevice, stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("release: status write: ") +
                          cudaGetErrorString(e));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusparseDiluRelease, CusparseDiluReleaseImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // token[1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // status[1]
