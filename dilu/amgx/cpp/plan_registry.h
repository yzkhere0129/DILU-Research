// Process-global AMGx resources + solver-plan cache.
//
// Contract (arch doc §4):
//   - AMGX_initialize() called exactly once per process via std::once_flag.
//   - AMGX_finalize() is NEVER called — process-lifetime leak (inherits
//     Phase 2's cuSPARSE-handle policy; finalize races with XLA shutdown).
//   - One AMGX_resources_handle per device, lazily created, kept for process
//     lifetime. Singleton pattern, mirrors Phase 2.
//   - Per-plan: one (matrix, solver, b_vec, x_vec) tuple, allocated on heap
//     and keyed by an opaque uint64 token in a process-global unordered_map.
//   - Thread-affinity tripwire (phase 2 §4.5 pattern): record creator TID at
//     first AMGX_initialize and assert on every entry.
//
// Handle hierarchy (per arch §4.1):
//   initialize ── resources ── matrices, vectors, solvers
//                             configs (ephemeral; destroyed after solver
//                                      creation — AMGx copies what it needs).

#pragma once

#include <amgx_c.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace dilu {
namespace amgx {

// (N, nnz) pair — pattern-fingerprint. Content-independent; catches
// dimension-mismatch misuse but not pattern-tampering (same guarantee as
// Phase 2/3).
struct PatternFingerprint {
  int32_t n;
  int32_t nnz;

  bool operator==(const PatternFingerprint& o) const {
    return n == o.n && nnz == o.nnz;
  }
};

// AMGx plan. Owns its matrix, solver, config, and the two reused vectors.
//
// Config-handle lifetime note (§BUGFIX from initial skeleton):
// AMGx's `AMGX_solver_create` stores a raw pointer to the config
// (see AMGx source solver.h: `AMG_Config *m_cfg;`). Destroying the config
// after solver_create leaves a dangling pointer, which AMGx dereferences on
// the first `AMGX_solver_solve` call via `getPrintSolveStats()` → SIGSEGV.
// Therefore the PlanEntry OWNS the config handle for its whole lifetime.
struct PlanEntry {
  PatternFingerprint fingerprint;

  // AMGx-owned handles. Lifecycle managed entirely by destroy_plan_entry().
  AMGX_config_handle config = nullptr;
  AMGX_matrix_handle matrix = nullptr;
  AMGX_solver_handle solver = nullptr;
  AMGX_vector_handle b_vec = nullptr;
  AMGX_vector_handle x_vec = nullptr;

  // The JSON source is kept for diagnostics only. Config handle above is the
  // active reference that the solver depends on.
  std::string config_json;
};

// Initialize AMGx exactly once per process. Returns AMGX_RC_OK on success.
// Thread-safe via std::call_once. Idempotent from the caller's perspective.
AMGX_RC amgx_initialize_once();

// Get-or-create the singleton resources handle for device 0 (we are
// single-GPU only). The simple-resources API is adequate for no-MPI builds.
// Returns AMGX_RC_OK on success and writes *out_rsc.
//
// Uses a minimal housekeeping config that AMGx requires to create resources;
// that config is NOT used for actual solves — each plan builds its own config
// from user JSON.
AMGX_RC get_resources(AMGX_resources_handle* out_rsc);

// Returns true iff called from a different thread than the one that
// first initialized AMGx. Arch §4.5 tripwire.
bool handle_called_from_wrong_thread();

// Plan cache — token ↔ PlanEntry*. Thread-safe (mutex). Tokens are a
// monotonic counter starting at 1 (0 reserved as "invalid").
uint64_t plan_cache_insert(PlanEntry* entry);
PlanEntry* plan_cache_lookup(uint64_t token);
bool plan_cache_remove(uint64_t token);  // returns false if absent

// Tear down an entry's AMGx handles in the documented safe order
// (solver → vectors → matrix). Safe on a half-initialized entry (nullptr
// fields skipped). Finally deletes the struct.
void destroy_plan_entry(PlanEntry* entry);

// Human-readable AMGx return-code string (for error reporting).
const char* amgx_rc_str(AMGX_RC rc);

}  // namespace amgx
}  // namespace dilu
