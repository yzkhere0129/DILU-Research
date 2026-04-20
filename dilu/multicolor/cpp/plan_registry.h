// Phase 3 plan registry. Mirrors Phase 2's pattern exactly except:
//   - No cuSPARSE. Plans hold plain-CUDA buffers for the PERMUTED CSR, the
//     coloring metadata, the value-remap, and the 5 scratch vectors.
//   - No handle singleton either (CUDA context is already process-global).
//     We keep the thread-id tripwire for defense in depth.
//
// Contract: arch doc §3.5 (plan entry layout), §5 (lifecycle & fingerprint).
// STOP rules: §9 (plan-map leak, wrong-thread, etc.)

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <mutex>
#include <unordered_map>

namespace dilu {
namespace multicolor {

struct PatternFingerprint {
  int32_t n;
  int32_t nnz;
  int32_t n_colors;
  bool operator==(const PatternFingerprint& o) const {
    return n == o.n && nnz == o.nnz && n_colors == o.n_colors;
  }
};

struct MulticolorPlanEntry {
  PatternFingerprint fingerprint;

  // ---- Permuted CSR (owned). ---------------------------------------------
  int32_t* row_ptr_tilde = nullptr;     // [N+1]
  int32_t* col_idx_tilde = nullptr;     // [nnz]
  double*  values_tilde  = nullptr;     // [nnz] — refreshed per apply/refactor
  int32_t* diag_offset_tilde = nullptr; // [N]

  // ---- Coloring metadata (owned). ----------------------------------------
  int32_t* perm           = nullptr;     // [N]   perm[new] = old
  int32_t* iperm          = nullptr;     // [N]   iperm[old] = new
  int32_t* color_offsets  = nullptr;     // [n_colors + 1]
  // Host mirror of color_offsets, filled at analyze time. Lets apply() drive
  // the per-color launch loop with zero device→host traffic (saves a small
  // but real overhead, and keeps the 8-byte token as the ONLY tolerated D→H
  // copy per apply — matches Phase 2's arch contract exactly).
  int32_t color_offsets_host[64] = {0};

  // ---- Value-remap (owned). ----------------------------------------------
  int32_t* nnz_map        = nullptr;     // [nnz] values_tilde[k] = values[nnz_map[k]]

  // ---- Persistent scratch (owned). ---------------------------------------
  double* d_star_tilde    = nullptr;     // [N]   result of Phase 2 dilu_factor on Ã
  double* r_tilde         = nullptr;     // [N]
  double* y_tilde         = nullptr;     // [N]
  double* y_scaled_tilde  = nullptr;     // [N]
  double* z_tilde         = nullptr;     // [N]
};

// Plan-map primitives (thread-safe via internal mutex).
uint64_t plan_cache_insert(MulticolorPlanEntry* entry);
MulticolorPlanEntry* plan_cache_lookup(uint64_t token);
bool plan_cache_remove(uint64_t token);  // false if missing
void destroy_plan_entry(MulticolorPlanEntry* entry);

// Thread-id tripwire (defense in depth — our single-device JAX setup should
// only ever touch these structures on the XLA scheduler thread).
bool plan_map_called_from_wrong_thread();

}  // namespace multicolor
}  // namespace dilu
