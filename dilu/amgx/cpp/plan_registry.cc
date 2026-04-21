// Implementation of plan_registry.h.
//
// HANDLE LEAK: we never call AMGX_finalize(). Per arch §4.4, finalize races
// with XLA/CUDA teardown at interpreter exit in ways we cannot reliably
// schedule; we accept the AMGx-init-state leak as a documented cost.
//
// The bootstrap config fed to AMGX_resources_create_simple is the minimal
// one documented in AMGx examples ("{}"). Each per-plan config is created
// and destroyed inside amgx_setup.cc — that's the config consumed by the
// actual solver.

#include "plan_registry.h"

#include <amgx_c.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

// AMGx print callback — routes AMGx diagnostic output to stderr so we can see
// internal errors. Without this, AMGx errors (e.g. "matrix not SPD",
// "NaN in residual") are swallowed.
static void dilu_amgx_print_callback(const char* msg, int length) {
  // length may or may not include a trailing NUL; fprintf with %.*s is safe.
  std::fprintf(stderr, "[amgx] %.*s", length, msg);
}

namespace dilu {
namespace amgx {

namespace {

// One-time init state. Lives for the process.
std::once_flag g_init_once;
AMGX_RC g_init_status = AMGX_RC_UNKNOWN;
std::thread::id g_init_tid{};
std::atomic<bool> g_init_tid_set{false};

std::mutex g_resources_mu;
AMGX_resources_handle g_resources = nullptr;
AMGX_config_handle g_bootstrap_cfg = nullptr;

std::mutex g_plan_mu;
std::unordered_map<uint64_t, PlanEntry*> g_plans;
std::atomic<uint64_t> g_next_token{1};  // 0 reserved invalid

}  // namespace

AMGX_RC amgx_initialize_once() {
  std::call_once(g_init_once, []() {
    g_init_status = AMGX_initialize();
    if (g_init_status == AMGX_RC_OK) {
      // Register print callback so internal AMGx diagnostic messages (setup
      // log, error strings, residual history if enabled) reach stderr.
      AMGX_register_print_callback(dilu_amgx_print_callback);
      // Install AMGx's own SIGSEGV/FPE handler — emits a stack with a C-API
      // error code instead of a silent segfault. This helps pinpoint the
      // difference between "AMGx error" and "JAX/CUDA misuse".
      AMGX_install_signal_handler();
      g_init_tid = std::this_thread::get_id();
      g_init_tid_set.store(true, std::memory_order_release);
    }
  });
  return g_init_status;
}

AMGX_RC get_resources(AMGX_resources_handle* out_rsc) {
  AMGX_RC rc = amgx_initialize_once();
  if (rc != AMGX_RC_OK) return rc;

  std::lock_guard<std::mutex> lock(g_resources_mu);
  if (g_resources != nullptr) {
    *out_rsc = g_resources;
    return AMGX_RC_OK;
  }

  // Bootstrap config. AMGx requires a config to build resources even in the
  // single-GPU no-MPI case. An empty JSON object is the documented minimum.
  rc = AMGX_config_create(&g_bootstrap_cfg, "{}");
  if (rc != AMGX_RC_OK) {
    g_bootstrap_cfg = nullptr;
    return rc;
  }
  rc = AMGX_resources_create_simple(&g_resources, g_bootstrap_cfg);
  if (rc != AMGX_RC_OK) {
    AMGX_config_destroy(g_bootstrap_cfg);
    g_bootstrap_cfg = nullptr;
    g_resources = nullptr;
    return rc;
  }
  *out_rsc = g_resources;
  return AMGX_RC_OK;
}

bool handle_called_from_wrong_thread() {
  if (!g_init_tid_set.load(std::memory_order_acquire)) return false;
  return std::this_thread::get_id() != g_init_tid;
}

uint64_t plan_cache_insert(PlanEntry* entry) {
  std::lock_guard<std::mutex> lock(g_plan_mu);
  uint64_t tok = g_next_token.fetch_add(1, std::memory_order_relaxed);
  g_plans[tok] = entry;
  return tok;
}

PlanEntry* plan_cache_lookup(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_plan_mu);
  auto it = g_plans.find(token);
  return (it == g_plans.end()) ? nullptr : it->second;
}

bool plan_cache_remove(uint64_t token) {
  PlanEntry* to_destroy = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_plan_mu);
    auto it = g_plans.find(token);
    if (it == g_plans.end()) return false;
    to_destroy = it->second;
    g_plans.erase(it);
  }
  // Destroy OUTSIDE the lock — AMGx calls can be slow and we don't want to
  // hold the plan-cache mutex across them.
  destroy_plan_entry(to_destroy);
  return true;
}

void destroy_plan_entry(PlanEntry* entry) {
  if (!entry) return;
  // Safe order per AMGx docs: solver → vectors → matrix → config.
  // Config is destroyed AFTER the solver because the solver holds a raw
  // pointer to it (see plan_registry.h BUGFIX note).
  if (entry->solver) AMGX_solver_destroy(entry->solver);
  if (entry->b_vec)  AMGX_vector_destroy(entry->b_vec);
  if (entry->x_vec)  AMGX_vector_destroy(entry->x_vec);
  if (entry->matrix) AMGX_matrix_destroy(entry->matrix);
  if (entry->config) AMGX_config_destroy(entry->config);
  delete entry;
}

const char* amgx_rc_str(AMGX_RC rc) {
  switch (rc) {
    case AMGX_RC_OK: return "AMGX_RC_OK";
    case AMGX_RC_BAD_PARAMETERS: return "AMGX_RC_BAD_PARAMETERS";
    case AMGX_RC_UNKNOWN: return "AMGX_RC_UNKNOWN";
    case AMGX_RC_NOT_SUPPORTED_TARGET: return "AMGX_RC_NOT_SUPPORTED_TARGET";
    case AMGX_RC_NOT_SUPPORTED_BLOCKSIZE: return "AMGX_RC_NOT_SUPPORTED_BLOCKSIZE";
    case AMGX_RC_CUDA_FAILURE: return "AMGX_RC_CUDA_FAILURE";
    case AMGX_RC_THRUST_FAILURE: return "AMGX_RC_THRUST_FAILURE";
    case AMGX_RC_NO_MEMORY: return "AMGX_RC_NO_MEMORY";
    case AMGX_RC_IO_ERROR: return "AMGX_RC_IO_ERROR";
    case AMGX_RC_BAD_MODE: return "AMGX_RC_BAD_MODE";
    case AMGX_RC_CORE: return "AMGX_RC_CORE";
    case AMGX_RC_PLUGIN: return "AMGX_RC_PLUGIN";
    case AMGX_RC_BAD_CONFIGURATION: return "AMGX_RC_BAD_CONFIGURATION";
    case AMGX_RC_NOT_IMPLEMENTED: return "AMGX_RC_NOT_IMPLEMENTED";
    case AMGX_RC_LICENSE_NOT_FOUND: return "AMGX_RC_LICENSE_NOT_FOUND";
    case AMGX_RC_INTERNAL: return "AMGX_RC_INTERNAL";
    default: return "AMGX_RC_unknown_code";
  }
}

}  // namespace amgx
}  // namespace dilu
