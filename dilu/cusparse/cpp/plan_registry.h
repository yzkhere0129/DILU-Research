// Process-global cuSPARSE handle + SpSV plan cache.
//
// Contract (arch doc §2.3, §2.6):
//   - One cuSPARSE handle per CUDA device, lazily created, NEVER destroyed
//     (process lifetime). Justification: destruction races in XLA shutdown;
//     we accept the handle leak as a tolerated one-shot cost.
//   - HANDLE SHARING RULE: cuSPARSE requires handle-per-thread. JAX 0.9 on
//     single-device runs FFI handlers on a single XLA scheduler thread, so
//     we share the handle across FFI entries safely. If a second thread ever
//     touches the handle we WILL detect it (see plan_registry.cc).
//   - PlanEntry is allocated on heap, keyed by an opaque uint64 token.
//     `analyze` inserts, `apply` looks up, `release` removes + frees.
//   - Pattern fingerprint check: apply() verifies the CSR pointers + sizes
//     match analyze() time, otherwise fail with InvalidArgument (arch §3.3).

#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdint.h>

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dilu {
namespace cusparse {

// Pattern fingerprint — we no longer rely on caller pointers (unstable across
// jit). Instead analyze owns its own row_ptr/col_idx copies; apply just
// verifies n+nnz match (plus a lightweight content-independent sanity check
// by recomputing total nnz against the plan-owned row_ptr[n]).
struct PatternFingerprint {
  int32_t n;
  int32_t nnz;

  bool operator==(const PatternFingerprint& o) const {
    return n == o.n && nnz == o.nnz;
  }
};

// Everything needed to do two SpSV solves on L* and U*.
struct PlanEntry {
  PatternFingerprint fingerprint;

  // Owned SpMat descriptors. `mat_L` is CSR with fill=LOWER, diag=NON_UNIT.
  // `mat_U` is CSR with fill=UPPER, diag=NON_UNIT. Both wrap the SAME
  // working_values buffer we own below (so we need to rebind values per
  // apply — cusparseCsrSetPointers allows this without recreating the desc).
  cusparseSpMatDescr_t mat_L = nullptr;
  cusparseSpMatDescr_t mat_U = nullptr;

  // SpSV plan descriptors (hold the analysis output).
  cusparseSpSVDescr_t spsv_L = nullptr;
  cusparseSpSVDescr_t spsv_U = nullptr;

  // Workspace buffers (cusparseSpSV_bufferSize → cudaMalloc).
  void* buf_L = nullptr;
  void* buf_U = nullptr;
  size_t buf_L_sz = 0;
  size_t buf_U_sz = 0;

  // Plan-owned CSR pattern buffers. Copied from the user's arrays at analyze
  // time. Fixed for the plan's lifetime. This is what the SpMatDescr_t
  // descriptors point at for row_ptr/col_idx — isolating us from XLA's
  // re-buffering of inputs across jit boundaries.
  int32_t* row_ptr = nullptr;     // size n+1
  int32_t* col_idx = nullptr;     // size nnz
  int32_t* diag_offset = nullptr; // size n, owned here (copied from user)

  // Working values buffer that L* and U* descriptors point at.
  // Size = nnz * sizeof(double). Owned. Updated per apply to hold
  // [A's off-diagonals + D_* on the diagonal].
  double* working_values = nullptr;

  // Intermediate vectors for the two-solve sweep. Owned.
  double* y_mid = nullptr;   // output of forward solve
  double* y_scaled = nullptr;  // D_* * y_mid, RHS of backward solve
  // Descriptors that wrap y_mid, y_scaled, and externally-supplied r/z.
  // We create them in analyze with any pointer (they must be bound at apply
  // time via cusparseDnVecSetValues because r and z are XLA-owned).
  cusparseDnVecDescr_t vec_r = nullptr;
  cusparseDnVecDescr_t vec_y_mid = nullptr;
  cusparseDnVecDescr_t vec_y_scaled = nullptr;
  cusparseDnVecDescr_t vec_z = nullptr;
};

// Singleton cuSPARSE handle access. Lazy init on first call. Never destroyed.
// Returns CUSPARSE_STATUS_* on failure.
cusparseStatus_t get_handle(cusparseHandle_t* out_handle);

// Returns true if called from a different thread than the one that first
// created the handle. Arch §2.6 tripwire. If true, handler should fail loudly.
bool handle_called_from_wrong_thread();

// Plan cache operations. All thread-safe (internal mutex), though the expected
// usage is single-threaded.
uint64_t plan_cache_insert(PlanEntry* entry);
PlanEntry* plan_cache_lookup(uint64_t token);
bool plan_cache_remove(uint64_t token);  // returns false if token not present

// Destroys all cuSPARSE/CUDA resources held by an entry, then deletes it.
// Safe on a half-initialized entry (nullptr fields skipped).
void destroy_plan_entry(PlanEntry* entry);

// Human-readable cuSPARSE status string (for error messages).
const char* cusparse_status_str(cusparseStatus_t s);

}  // namespace cusparse
}  // namespace dilu
