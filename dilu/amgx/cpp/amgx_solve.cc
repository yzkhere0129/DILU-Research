// FFI handler: amgx_solve — the hot path.
//
// Contract (arch doc §3.3):
//   Inputs:  token uint64[1], b f64[n], x0 f64[n]
//   Output:  x f64[n], iters int32[1], status int32[1]
//
// Work:
//   1. 8-byte D→H token read (the ONE tolerated host copy per solve).
//   2. Plan cache lookup + fingerprint check (n).
//   3. AMGX_vector_upload b (D→D copy into AMGx's owned b buffer, ~16 MB at 128³).
//   4. AMGX_vector_upload x0 (initial guess).
//   5. AMGX_solver_solve — runs the full AMG-preconditioned Krylov loop on
//      device; no JAX involvement until it returns.
//   6. AMGX_solver_get_iterations_number + _get_status (tiny host scalars).
//   7. AMGX_vector_download x → JAX output buffer (D→D).
//   8. Write iters/status scalars H→D (4 bytes each).
//
// Stream binding (§3.3.1): AMGx does not expose a public set_stream hook on
// the solver handle, only via config keys that are not universally honored.
// We synchronize the XLA stream BEFORE the AMGx calls so their internal
// stream sees consistent inputs, and synchronize AFTER so subsequent XLA
// ops observe the finished solve. This loses pipelining (small cost at the
// grid sizes we target) but guarantees correctness.

#include <amgx_c.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::amgx::PlanEntry;
using dilu::amgx::plan_cache_lookup;
using dilu::amgx::amgx_rc_str;
using dilu::amgx::handle_called_from_wrong_thread;

#define CHECK_AMGX(expr)                                                  \
  do {                                                                     \
    AMGX_RC _rc = (expr);                                                  \
    if (_rc != AMGX_RC_OK) {                                               \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("amgx: ") + amgx_rc_str(_rc));         \
    }                                                                      \
  } while (0)

#define CHECK_CUDA(expr)                                                  \
  do {                                                                     \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cuda: ") + cudaGetErrorString(_e));  \
    }                                                                      \
  } while (0)

static ffi::Error AmgxSolveImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::U64> token_buf,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Buffer<ffi::DataType::F64> x0,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> iters_out,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status_out) {
  if (handle_called_from_wrong_thread()) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      "amgx_solve: called from non-owner thread");
  }
  if (token_buf.element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: token must be uint64[1]");
  }
  if (iters_out->element_count() != 1 || status_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: iters/status must be int32[1]");
  }

  // 8-byte D→H token read (the tolerated exception).
  uint64_t token_host = 0;
  CHECK_CUDA(cudaMemcpyAsync(&token_host, token_buf.typed_data(),
                             sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  PlanEntry* entry = plan_cache_lookup(token_host);
  if (!entry) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: unknown token");
  }

  const size_t n_sz = b.element_count();
  if (n_sz > static_cast<size_t>(INT32_MAX)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: n exceeds int32");
  }
  const int32_t n = static_cast<int32_t>(n_sz);
  if (x0.element_count() != static_cast<size_t>(n) ||
      x_out->element_count() != static_cast<size_t>(n)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: b/x0/x length mismatch");
  }
  if (entry->fingerprint.n != n) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_solve: token/n fingerprint mismatch");
  }

#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: n=%d, upload b...\n", n);
#endif
  CHECK_AMGX(AMGX_vector_upload(entry->b_vec, n, 1, b.typed_data()));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: upload x0...\n");
#endif
  CHECK_AMGX(AMGX_vector_upload(entry->x_vec, n, 1, x0.typed_data()));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: AMGX_solver_solve...\n");
#endif
  CHECK_AMGX(AMGX_solver_solve(entry->solver, entry->b_vec, entry->x_vec));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: solve done, query iter/status...\n");
#endif
  int n_iter = 0;
  CHECK_AMGX(AMGX_solver_get_iterations_number(entry->solver, &n_iter));
  AMGX_SOLVE_STATUS solve_status = AMGX_SOLVE_FAILED;
  CHECK_AMGX(AMGX_solver_get_status(entry->solver, &solve_status));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: iters=%d status=%d; download x...\n",
               n_iter, (int)solve_status);
#endif
  CHECK_AMGX(AMGX_vector_download(entry->x_vec, x_out->typed_data()));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] solve: download done\n");
#endif

  // Write scalars to output buffers (4-byte H→D each). The stack-local
  // iters_host/status_host go out of scope when this frame unwinds, so we
  // MUST synchronize the stream before returning — otherwise a future XLA
  // scheduler change (pinned-host or CUDA graphs) could fire the copies
  // after the source bytes are gone. cudaStreamSynchronize is microseconds
  // vs the AMGx solve, so the cost is negligible.
  int32_t iters_host = static_cast<int32_t>(n_iter);
  int32_t status_host = static_cast<int32_t>(solve_status);
  CHECK_CUDA(cudaMemcpyAsync(iters_out->typed_data(), &iters_host,
                             sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(status_out->typed_data(), &status_host,
                             sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return ffi::Error::Success();
}

#undef CHECK_AMGX
#undef CHECK_CUDA

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AmgxSolve, AmgxSolveImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // token[1]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // b[n]
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // x0[n]
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // x[n]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // iters[1]
        .Ret<ffi::Buffer<ffi::DataType::S32>>()); // status[1]
