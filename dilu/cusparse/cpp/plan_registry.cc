// Implementation of plan_registry.h.
//
// HANDLE LEAK NOTE: the cuSPARSE handle is intentionally never destroyed.
// Destroying it races with XLA shutdown and with child-process inheritance
// under fork(). Arch doc §2.6 documents this as an accepted cost.

#include "plan_registry.h"

#include <atomic>
#include <thread>

namespace dilu {
namespace cusparse {

namespace {

std::mutex g_handle_mutex;
cusparseHandle_t g_handle = nullptr;
std::thread::id g_handle_owner_tid{};
std::atomic<bool> g_handle_owner_set{false};

std::mutex g_plan_mutex;
std::unordered_map<uint64_t, PlanEntry*> g_plan_map;
std::atomic<uint64_t> g_token_counter{1};  // 0 is reserved / invalid

}  // namespace

cusparseStatus_t get_handle(cusparseHandle_t* out_handle) {
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  if (g_handle == nullptr) {
    cusparseStatus_t s = cusparseCreate(&g_handle);
    if (s != CUSPARSE_STATUS_SUCCESS) {
      g_handle = nullptr;
      return s;
    }
    g_handle_owner_tid = std::this_thread::get_id();
    g_handle_owner_set.store(true, std::memory_order_release);
  }
  *out_handle = g_handle;
  return CUSPARSE_STATUS_SUCCESS;
}

bool handle_called_from_wrong_thread() {
  if (!g_handle_owner_set.load(std::memory_order_acquire)) return false;
  return std::this_thread::get_id() != g_handle_owner_tid;
}

uint64_t plan_cache_insert(PlanEntry* entry) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  uint64_t tok = g_token_counter.fetch_add(1, std::memory_order_relaxed);
  g_plan_map[tok] = entry;
  return tok;
}

PlanEntry* plan_cache_lookup(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_plan_map.find(token);
  return (it == g_plan_map.end()) ? nullptr : it->second;
}

bool plan_cache_remove(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_plan_map.find(token);
  if (it == g_plan_map.end()) return false;
  PlanEntry* e = it->second;
  g_plan_map.erase(it);
  destroy_plan_entry(e);
  return true;
}

void destroy_plan_entry(PlanEntry* entry) {
  if (!entry) return;
  // Order: SpSV descriptors → SpMat descriptors → DnVec descriptors → CUDA buffers.
  if (entry->spsv_L) cusparseSpSV_destroyDescr(entry->spsv_L);
  if (entry->spsv_U) cusparseSpSV_destroyDescr(entry->spsv_U);
  if (entry->mat_L)  cusparseDestroySpMat(entry->mat_L);
  if (entry->mat_U)  cusparseDestroySpMat(entry->mat_U);
  if (entry->vec_r)        cusparseDestroyDnVec(entry->vec_r);
  if (entry->vec_y_mid)    cusparseDestroyDnVec(entry->vec_y_mid);
  if (entry->vec_y_scaled) cusparseDestroyDnVec(entry->vec_y_scaled);
  if (entry->vec_z)        cusparseDestroyDnVec(entry->vec_z);
  if (entry->buf_L)          cudaFree(entry->buf_L);
  if (entry->buf_U)          cudaFree(entry->buf_U);
  if (entry->working_values) cudaFree(entry->working_values);
  if (entry->y_mid)          cudaFree(entry->y_mid);
  if (entry->y_scaled)       cudaFree(entry->y_scaled);
  if (entry->row_ptr)        cudaFree(entry->row_ptr);
  if (entry->col_idx)        cudaFree(entry->col_idx);
  if (entry->diag_offset)    cudaFree(entry->diag_offset);
  delete entry;
}

const char* cusparse_status_str(cusparseStatus_t s) {
  switch (s) {
    case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
      return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default: return "CUSPARSE_STATUS_UNKNOWN";
  }
}

}  // namespace cusparse
}  // namespace dilu
