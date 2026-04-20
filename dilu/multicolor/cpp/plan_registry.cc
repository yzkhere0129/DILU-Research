// plan_registry.cc — see header for contract.

#include "plan_registry.h"

#include <atomic>
#include <thread>

namespace dilu {
namespace multicolor {

namespace {

std::mutex g_plan_mutex;
std::unordered_map<uint64_t, MulticolorPlanEntry*> g_plan_map;
std::atomic<uint64_t> g_token_counter{1};  // 0 reserved
std::atomic<bool> g_tid_set{false};
std::thread::id g_tid_owner{};

void register_caller_thread() {
  bool expected = false;
  if (g_tid_set.compare_exchange_strong(expected, true)) {
    g_tid_owner = std::this_thread::get_id();
  }
}

}  // namespace

uint64_t plan_cache_insert(MulticolorPlanEntry* entry) {
  register_caller_thread();
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  uint64_t tok = g_token_counter.fetch_add(1, std::memory_order_relaxed);
  g_plan_map[tok] = entry;
  return tok;
}

MulticolorPlanEntry* plan_cache_lookup(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_plan_map.find(token);
  return (it == g_plan_map.end()) ? nullptr : it->second;
}

bool plan_cache_remove(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_plan_map.find(token);
  if (it == g_plan_map.end()) return false;
  MulticolorPlanEntry* e = it->second;
  g_plan_map.erase(it);
  destroy_plan_entry(e);
  return true;
}

void destroy_plan_entry(MulticolorPlanEntry* entry) {
  if (!entry) return;
  if (entry->row_ptr_tilde)     cudaFree(entry->row_ptr_tilde);
  if (entry->col_idx_tilde)     cudaFree(entry->col_idx_tilde);
  if (entry->values_tilde)      cudaFree(entry->values_tilde);
  if (entry->diag_offset_tilde) cudaFree(entry->diag_offset_tilde);
  if (entry->perm)              cudaFree(entry->perm);
  if (entry->iperm)             cudaFree(entry->iperm);
  if (entry->color_offsets)     cudaFree(entry->color_offsets);
  if (entry->nnz_map)           cudaFree(entry->nnz_map);
  if (entry->d_star_tilde)      cudaFree(entry->d_star_tilde);
  if (entry->r_tilde)           cudaFree(entry->r_tilde);
  if (entry->y_tilde)           cudaFree(entry->y_tilde);
  if (entry->y_scaled_tilde)    cudaFree(entry->y_scaled_tilde);
  if (entry->z_tilde)           cudaFree(entry->z_tilde);
  delete entry;
}

bool plan_map_called_from_wrong_thread() {
  if (!g_tid_set.load(std::memory_order_acquire)) return false;
  return std::this_thread::get_id() != g_tid_owner;
}

}  // namespace multicolor
}  // namespace dilu
