// FFI handler: amgx_release — teardown for a single plan.
//
// Contract: idempotent; releasing a gone token is a soft-failure (status=0).
// Mirrors Phase 2's release handler exactly, with the plan-cache call
// plumbing into AMGx destructors via destroy_plan_entry().

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::amgx::plan_cache_remove;

static ffi::Error AmgxReleaseImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status_out) {
  if (token_buf.element_count() != 1 || status_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_release: token uint64[1], status int32[1]");
  }
  uint64_t token_host = 0;
  cudaError_t e = cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                                  sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_release: token copy: ") +
                          cudaGetErrorString(e));
  }
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_release: token sync: ") +
                          cudaGetErrorString(e));
  }
  // Idempotent contract (matches Python docstring in wrapper.py:86 and
  // plan.py:90-94): 0 = success, including the case where the token was
  // already absent. Non-zero is reserved for a future "actually failed to
  // tear down" condition. The previous semantics (1 = removed, 0 = unknown)
  // inverted the convention versus the Python layer.
  bool removed = plan_cache_remove(token_host);
  (void)removed;
  int32_t status = 0;
  e = cudaMemcpyAsync(status_out->typed_data(), &status, sizeof(int32_t),
                      cudaMemcpyHostToDevice, stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_release: status write: ") +
                          cudaGetErrorString(e));
  }
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_release: status sync: ") +
                          cudaGetErrorString(e));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AmgxRelease, AmgxReleaseImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>());
