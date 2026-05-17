// FFI handler: amgx_setup.
//
// Contract (arch doc §3.1):
//   Inputs:  row_ptr int32[n+1], col_idx int32[nnz], values f64[nnz]
//   Attr:    config_json : string (passed through XLA FFI attr, NOT traced)
//   Outputs: token uint64[1]
//
// Work:
//   1. AMGX_initialize() once (std::once_flag) → get singleton resources.
//   2. Create ephemeral AMGX_config from supplied JSON. Config is consumed by
//      solver_create; we destroy it right after (AMGx copies state).
//   3. AMGX_matrix_create, AMGX_matrix_upload_all with dEVICE pointers.
//      AMGx's upload uses cudaMemcpyDefault internally → D→D copy, no PCIe.
//      #ifdef DILU_AMGX_VERBOSE — dump cudaPointerGetAttributes of each input
//      pointer so we can verify empirically that the D→D assumption holds.
//   4. AMGX_vector_create for b, x (reused across solves; pointers rebound per
//      call via upload — but avoiding create/destroy churn).
//   5. AMGX_solver_create + AMGX_solver_setup (the expensive hierarchy build).
//   6. Insert PlanEntry → return uint64 token (8-byte H→D memcpy).
//
// Zero-copy verification (arch §5 + §9 step 3): if DILU_AMGX_VERBOSE is
// defined, log cudaMemoryType for each input device pointer. Phase 4 report
// includes this output as empirical confirmation.

#include <amgx_c.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::amgx::PlanEntry;
using dilu::amgx::PatternFingerprint;
using dilu::amgx::amgx_initialize_once;
using dilu::amgx::get_resources;
using dilu::amgx::plan_cache_insert;
using dilu::amgx::destroy_plan_entry;
using dilu::amgx::amgx_rc_str;
using dilu::amgx::handle_called_from_wrong_thread;

// On error the plan owns its own config handle (stored on `entry->config`),
// so `destroy_plan_entry(entry)` is sufficient to release everything. No
// extra `AMGX_config_destroy` call is needed.
#define CHECK_AMGX(expr)                                                  \
  do {                                                                     \
    AMGX_RC _rc = (expr);                                                  \
    if (_rc != AMGX_RC_OK) {                                               \
      destroy_plan_entry(entry);                                           \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("amgx: ") + amgx_rc_str(_rc));         \
    }                                                                      \
  } while (0)

#define CHECK_CUDA(expr)                                                  \
  do {                                                                     \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
      destroy_plan_entry(entry);                                           \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cuda: ") + cudaGetErrorString(_e));  \
    }                                                                      \
  } while (0)

#ifdef DILU_AMGX_VERBOSE
static void dump_pointer_kind(const char* label, const void* ptr) {
  cudaPointerAttributes attrs{};
  cudaError_t e = cudaPointerGetAttributes(&attrs, ptr);
  const char* kind = "unknown";
  if (e == cudaSuccess) {
    switch (attrs.type) {
      case cudaMemoryTypeUnregistered: kind = "cudaMemoryTypeUnregistered"; break;
      case cudaMemoryTypeHost:         kind = "cudaMemoryTypeHost"; break;
      case cudaMemoryTypeDevice:       kind = "cudaMemoryTypeDevice"; break;
      case cudaMemoryTypeManaged:      kind = "cudaMemoryTypeManaged"; break;
    }
    std::fprintf(stderr,
        "[dilu_amgx verbose] %s: ptr=%p type=%s device=%d\n",
        label, ptr, kind, attrs.device);
  } else {
    std::fprintf(stderr,
        "[dilu_amgx verbose] %s: cudaPointerGetAttributes failed: %s\n",
        label, cudaGetErrorString(e));
  }
}
#endif

static ffi::Error AmgxSetupImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> col_idx,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> token_out,
    std::string_view config_json) {
  const size_t n_plus_1 = row_ptr.element_count();
  if (n_plus_1 < 2) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_setup: row_ptr must have length >= 2");
  }
  const int64_t n = static_cast<int64_t>(n_plus_1) - 1;
  const int64_t nnz = static_cast<int64_t>(values.element_count());
  if (col_idx.element_count() != static_cast<size_t>(nnz)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_setup: col_idx/values length mismatch");
  }
  if (n > INT32_MAX || nnz > INT32_MAX) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_setup: n or nnz exceeds int32");
  }
  if (token_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "amgx_setup: token output must be uint64[1]");
  }

  // AMGx's internal upload path may block on the default stream. We
  // synchronize our stream BEFORE the upload so that any preceding JAX
  // ops have committed — cleaner than relying on stream ordering inside AMGx.
  cudaError_t sync0 = cudaStreamSynchronize(stream);
  if (sync0 != cudaSuccess) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_setup: pre-sync: ") +
                          cudaGetErrorString(sync0));
  }

  AMGX_RC rc = amgx_initialize_once();
  if (rc != AMGX_RC_OK) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_setup: init: ") + amgx_rc_str(rc));
  }

  AMGX_resources_handle resources = nullptr;
  rc = get_resources(&resources);
  if (rc != AMGX_RC_OK) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("amgx_setup: resources: ") +
                          amgx_rc_str(rc));
  }

  // config_json comes from the FFI Attr<string_view>. Null-terminate by
  // copying into std::string (string_view is not NUL-terminated in general).
  std::string cfg_str(config_json.data(), config_json.size());

#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] setup: n=%lld nnz=%lld\n",
               static_cast<long long>(n), static_cast<long long>(nnz));
  dump_pointer_kind("row_ptr", row_ptr.typed_data());
  dump_pointer_kind("col_idx", col_idx.typed_data());
  dump_pointer_kind("values",  values.typed_data());
#endif

  // Allocate the plan entry FIRST so every subsequent AMGx call that fails
  // can route through destroy_plan_entry for cleanup. Config is bound into
  // the entry immediately on success and lives as long as the plan does
  // (AMGx's solver keeps a raw pointer to it — see plan_registry.h).
  PlanEntry* entry = new PlanEntry();
  entry->fingerprint = PatternFingerprint{static_cast<int32_t>(n),
                                          static_cast<int32_t>(nnz)};
  entry->config_json = cfg_str;

  rc = AMGX_config_create(&entry->config, cfg_str.c_str());
  if (rc != AMGX_RC_OK) {
    destroy_plan_entry(entry);
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      std::string("amgx_setup: config: ") + amgx_rc_str(rc));
  }

#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] AMGX_matrix_create...\n");
#endif
  CHECK_AMGX(AMGX_matrix_create(&entry->matrix, resources, AMGX_mode_dDDI));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] AMGX_matrix_upload_all...\n");
#endif
  CHECK_AMGX(AMGX_matrix_upload_all(
      entry->matrix,
      static_cast<int>(n),
      static_cast<int>(nnz),
      /*block_dimx=*/1, /*block_dimy=*/1,
      row_ptr.typed_data(),
      col_idx.typed_data(),
      values.typed_data(),
      /*diag_data=*/nullptr));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] AMGX_vector_create b,x...\n");
#endif
  CHECK_AMGX(AMGX_vector_create(&entry->b_vec, resources, AMGX_mode_dDDI));
  CHECK_AMGX(AMGX_vector_create(&entry->x_vec, resources, AMGX_mode_dDDI));

#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] AMGX_solver_create...\n");
#endif
  CHECK_AMGX(AMGX_solver_create(&entry->solver, resources,
                                AMGX_mode_dDDI, entry->config));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] AMGX_solver_setup...\n");
#endif
  CHECK_AMGX(AMGX_solver_setup(entry->solver, entry->matrix));
#ifdef DILU_AMGX_VERBOSE
  std::fprintf(stderr, "[dilu_amgx verbose] setup complete\n");
#endif

  // Register in plan cache, write 8-byte token to device output. token is
  // stack-local; synchronize before returning so the H→D copy is guaranteed
  // to consume it (see C1 fix in amgx_solve.cc for rationale).
  uint64_t token = plan_cache_insert(entry);
  CHECK_CUDA(cudaMemcpyAsync(token_out->typed_data(), &token,
                             sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return ffi::Error::Success();
}

#undef CHECK_AMGX
#undef CHECK_CUDA

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AmgxSetup, AmgxSetupImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // row_ptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // col_idx
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // values
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // token[1]
        .Attr<std::string_view>("config_json"));
