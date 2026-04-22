# Phase 4 — AMGx FFI Integration Engineering Architecture

**Project**: DILU Solver Low-Level Optimization Research (JAX-GPU)
**Scope**: Phase 4 Part 2 — engineering architecture for wrapping NVIDIA AMGx (algebraic multigrid library) into JAX-compatible FFI primitives with a zero-PCIe-copy device-pointer handoff.
**Companion doc**: `phase4_amg_math_foundation.md` (math expert, parallel track — landed)
**Predecessor docs**: `phase1_ffi_prototype_architecture.md`, `phase2_cusparse_ffi_architecture.md`, `phase3_multicolor_ffi_architecture.md`
**Predecessor reports**: `phase2_cusparse_report.md`, `phase3_multicolor_report.md`, `phase3_scaling_64_128.md`
**Date**: 2026-04-21
**Author**: `jax-cfd-expert`
**Hardware target for Phase 4 dev**: RTX 3050 Laptop (4 GB, compute 8.6, CUDA 12.4 toolkit)
**Status**: DESIGN — awaiting user sign-off before any Phase 4 code is written

---

## §0 Re-brief + TL;DR

### §0.1 What I read before writing

- `docs/design/phase1_ffi_prototype_architecture.md` — JAX FFI API surface (`jax.ffi.*`, `api_version=1`, `XLA_FFI_API_MAJOR 0 MINOR 2`), build recipe (CMake + ctypes + `jax.ffi.pycapsule`, no pybind11), one-handler-per-thread rule.
- `docs/design/phase2_cusparse_ffi_architecture.md` — opaque-token lifecycle (Option C), singleton handle pattern, `setStream` per apply, plan-owned pattern copies, `(N, nnz)` fingerprint, two-primitive split (analyze + apply).
- `docs/design/phase3_multicolor_ffi_architecture.md` — mirror of Phase 2 pattern with a four-primitive split that added `refactor` (values-only update) as a distinct FFI call. Path B = own-kernels decision; CMake target independent of prior phases.
- `docs/benchmark/phase2_cusparse_report.md` — Phase 2 baseline: T7 24 iters vs Jacobi 71, analyze amortizes at K≈2.4 apply calls. The 8-byte D→H token copy per apply is the sole tolerated exception to zero-copy.
- `docs/benchmark/phase3_multicolor_report.md` — Phase 3 PASS at 32³: 2.72× total PCG speedup, F2 halo test not triggered. Iter penalty 1.5× vs Phase 2.
- `docs/benchmark/phase3_scaling_64_128.md` — **the trigger for Phase 4**: at 64³ and 128³ the total PCG wall time *regresses to 0.80×/0.81×* because iter-count grows as `N^{1/3}` and bandwidth-bound SpMV dominates per-iter cost. DILU's Gustafsson $\kappa^{1/4}$ scaling under discontinuous coefficients degrades to $\sqrt{\kappa}$, matching the $N^{1/3}$ curve to two-digit precision. Math expert confirmed this in their §1.1.
- `docs/PROJECT_SUMMARY.md` — cross-phase status.
- `dilu/cusparse/cpp/{plan_registry.h,plan_registry.cc,cusparse_dilu_analyze.cc,cusparse_dilu_apply.cc}` — FFI handler patterns I will mirror.
- `dilu/cusparse/python/{registration.py,wrapper.py,plan.py}` — Python glue patterns I will mirror.
- `dilu/cusparse/CMakeLists.txt` — build recipe I will extend.
- `CLAUDE.md` — project rules: `dilu/` subdirs are the only place for new code; float64 defaults; `np.ascontiguousarray` at FFI boundary; no `-ffast-math`.

### §0.2 TL;DR

Phase 3 scaling data closed a question: DILU (exact or multi-color) is iter-count bounded by Gustafsson $\sqrt{\kappa}$ on our T7 stiff problem, which scales as $N^{1/3}$, so at 128³ exact DILU needs 186 iters and Phase 3 needs 296. Per-iter cost on the 3050 is ~150 ms (bandwidth-bound SpMV + DILU apply), so 128³ PCG takes ~28–35 s. **At 256³ the extrapolation is hundreds of iters × ~1 s per iter = minutes per linear solve per AM timestep — not viable.** AMG's grid-size-independent convergence (5–30 iters regardless of `N`) is the algorithmic escape; the only question is whether the integration pays for itself on the dev-box hardware and whether a zero-copy device handoff is attainable.

**Ecosystem reconnaissance (§1, web-verified 2026-04-21)**:
- **NVIDIA/AMGX is alive**. Latest tag **v2.5.0 released 2024-12-21** (source: `github.com/NVIDIA/AMGX/releases`). Raises minimum CUDA to 12.0 (headroom up to 13.0 per release notes; CHANGELOG still references v2.4.0 so CUDA-13 claim is release-notes-only, not CHANGELOG-verified — re-verify at clone time). License **BSD-3-Clause** (per repo SPDX tags).
- **pyamgx is effectively unmaintained** (no releases, last documented version 0.1 from 2018, open JAX-integration issue #40 from May 2025 unanswered as of this writing). It does expose `upload_raw()` which passes `void*` straight to `AMGX_vector_upload`, so the zero-copy mechanism exists at the library layer — but the wrapper itself is a frozen 2018-era Cython module that would need non-trivial maintenance work to use. **Not our path.**
- **AMGx's internal upload uses `cudaMemcpyDefault`**, which CUDA auto-routes by pointer kind. Device pointers passed to `AMGX_matrix_upload_all` therefore land a **D→D copy** into AMGx-owned buffers. *This is not true zero-copy* (AMGx owns its own device buffer, not JAX's) but it *is zero-PCIe* — all traffic stays on-device. For a 128³ matrix this is ~200 MB of D→D copy at setup, negligible. For apply-time `b` update, it's 16 MB — acceptable. See §5.
- **Alternatives** (Ginkgo, hypre BoomerAMG, cuDSS): cuDSS is a direct solver, orthogonal to our iterative regime. Ginkgo's AMG is competitive-to-better than AMGx per a 2024 SIAM paper but requires a full Ginkgo integration (large engineering cost; we do not own Ginkgo FFI experience). hypre has a GPU backend but the integration story for JAX is worse than AMGx's. **AMGx wins on path-of-least-engineering.**

**Recommended architecture (§2–§5)**:
- **Option B**: write a thin C++ FFI wrapper that uses AMGx's C API directly (mirroring Phase 2's `cusparse_dilu_*.cc`). No pyamgx, no pybind11. Link `libamgxsh.so` + CUDA runtime.
- **Four FFI primitives**: `amgx_setup`, `amgx_update_coefficients`, `amgx_solve`, `amgx_release`. Opaque `uint64` token keyed into a process-global plan registry (mirror Phase 2). The **fourth primitive** (`amgx_update_coefficients`, not in the brief's initial draft) is necessary because AMGx has a first-class `AMGX_matrix_replace_coefficients` API for values-only updates — we expose it rather than forcing a full re-setup each AM timestep.
- **Zero-copy story (§5)**: hand JAX device pointers straight to `AMGX_matrix_upload_all_maps` / `AMGX_vector_upload` / `AMGX_vector_download`. AMGx internally does a single D→D copy into its own device buffers at setup; per-solve we stage `b` and download `x` via D→D copies of 8-byte × N words. Total added PCIe traffic: **zero**. Total added D→D traffic per solve: ~32 MB at 128³ (2 × N × 8 bytes for b upload + x download). Acceptable.
- **Handle lifecycle (§4)**: `AMGX_initialize` at first-ever call, `AMGX_finalize` never (process-lifetime leak — same pattern as Phase 2's cuSPARSE handle). Per-plan: one `AMGX_resources_handle` (reusable across plans), one `AMGX_config_handle` per distinct config JSON, one `AMGX_matrix_handle` + one `AMGX_solver_handle` + two `AMGX_vector_handle` (for b, x) per plan.

**Hardware budget (§11)**: 128³ AMGx runs are **tight** on the 3050's 4 GB. Expected peak 2.5–3 GB (AMGx setup memory can be 3–5× nominal A size). **64³ is the confident target; 128³ is declared "stretch"** and may require the AMGx config tuned for memory (e.g., aggressive coarsening to reduce hierarchy cost). Phase 4 is explicitly a single-device, single-precision-mode (float64) effort.

**Scope discipline (§12)**: no Ginkgo, no hypre, no multi-GPU AMG, no mixed-precision AMG. Pure AMGx integration following the Phase-2 pattern. If AMGx proves unstable on our stack, Phase 4 falls back to "DILU-with-deflation" (ICCG + rigid body eigenvector deflation — out of scope for this doc, documented as a named fallback).

---

## §1 AMGx Ecosystem Reconnaissance (2026-04 State)

**All facts below verified via web search 2026-04-21.** Sources cited inline.

### §1.1 NVIDIA AMGx — alive and maintained

| Fact | Value | Source |
|---|---|---|
| Latest release tag | **v2.5.0** | github.com/NVIDIA/AMGX/releases |
| Release date | **2024-12-21** | github.com/NVIDIA/AMGX/releases |
| Minimum CUDA | **12.0** (raised from 10.0 in v2.5.0) | v2.5.0 release notes |
| Claimed CUDA range | **12.0 – 13.0** per release-notes summary | v2.5.0 release notes summary (**not yet verified in CHANGELOG**, which still shows only up to 12.2 in the repo file) |
| Tested GPU archs | A100 / H100. v2.5.0 release notes additionally mention **Blackwell** support | v2.5.0 release notes |
| Dropped GPU archs | sm_20, sm_35, sm_52, sm_60 no longer supported | v2.5.0 release notes |
| License | **BSD-3-Clause** (SPDX identifier in source) | github.com/NVIDIA/AMGX file headers; forums.developer.nvidia.com license thread |
| Build system | **CMake** (`cmake_minimum_required` not in our read, repo uses `find_package(CUDAToolkit)` + `find_package(MPI)`) | github.com/NVIDIA/AMGX/blob/main/CMakeLists.txt |
| Shared library | `libamgxsh.so` on Linux (dynamic); `libamgx.a` static | AMGx README (github.com/NVIDIA/AMGX) |
| CUDA link deps | `CUDA::cublas`, `CUDA::cusparse`, `CUDA::cusolver`, `m`, `pthread` | AMGx CMakeLists.txt inspection |
| MPI | Optional (we will build without) | AMGx README |

**Assessment**: NVIDIA has continued AMGx development into 2024/2025. The library is **not** deprecated in favor of cuDSS — cuDSS is a direct sparse solver (factorization-based) targeting moderate-size problems that fit in VRAM, which is a different use case than our iterative AMG-preconditioned CG. They coexist in NVIDIA's stack. **AMGx is a safe choice for Phase 4.**

**Uncertainty I am flagging**:
1. CUDA 12.4 on the dev box is between the v2.4.0 ceiling (12.2) and the v2.5.0 floor (12.0). v2.5.0 should compile and run, but our CUDA 12.4 is not explicitly in any published "tested" matrix. **Step 1 of the execution checklist (§9) is to clone `main` or `v2.5.0`, build against CUDA 12.4, and fail-fast if the cmake configure or compile breaks.**
2. The CHANGELOG in `main` stops at v2.4.0 while the releases page advertises v2.5.0 — likely CHANGELOG is stale but v2.5.0 was tagged. Re-confirm on clone.
3. CUDA 13.x claim: we do not need CUDA 13 on the dev box (CUDA 12.4 is installed), but the claim that AMGx supports CUDA 13 is release-notes-only. If the user ever moves to CUDA 13, **re-test**.

### §1.2 pyamgx — functional but effectively unmaintained

| Fact | Value | Source |
|---|---|---|
| Repository | `shwina/pyamgx` | github.com/shwina/pyamgx |
| Last documented version | **0.1** (since 2018) | pyamgx.readthedocs.io |
| Releases page | "No releases published" | github.com/shwina/pyamgx |
| Open JAX-integration issue | **#40 (May 2025)** — asks about maintenance + pip release; no maintainer response visible | github.com/shwina/pyamgx/issues/40 |
| Does `upload_raw` work? | Yes — `vec.upload_raw(d_a.device_ctypes_pointer.value, 3)` passes raw device pointer to `AMGX_vector_upload` | Documented at pyamgx.readthedocs.io/en/latest/basic.html |
| Python ecosystem | Cython-based; supports NumPy + Numba DeviceArray. **No explicit JAX support.** | repo README |

**Assessment**: pyamgx's API surface is adequate (the `upload_raw` path is *exactly* the zero-copy mechanism we would need), but the project is not maintained. Relying on it means:
- We build a Cython extension against pyamgx. Cython is not in our current toolchain (Phase 1 rejected pybind11/nanobind for the same reason; Cython is a strictly bigger dependency).
- We inherit pyamgx's AMGx version coupling (their Cython bindings target an AMGx API version; upgrading AMGx may break pyamgx without maintainer intervention).
- JAX `DeviceArray` → raw device pointer conversion is non-trivial: we would need to extract the pointer via `jax.dlpack` or `jax_array.unsafe_buffer_pointer()` and then pass it through pyamgx's `upload_raw`. Both Python-side mechanics work, but tying them together in a stable fashion costs more than just writing our own thin C++ wrapper.

**Decision**: pyamgx is **not the integration path**. We will use AMGx's C API directly from C++.

### §1.3 Alternatives if AMGx were dead (counter-factual)

- **Ginkgo** (`ginkgo-project.github.io`): GPU-native math library with AMG (PGM coarsening), competitive or better than AMGx per Cojean et al. SIAM 2024. Would require writing our own Ginkgo FFI wrapper — their C API is less mature than AMGx's and we have zero in-tree Ginkgo experience. **Higher engineering cost than AMGx.**
- **hypre BoomerAMG** (`hypre.readthedocs.io`): mature classical AMG, but its primary API is MPI-first and its GPU backend (CUDA/HIP/SYCL) is a bolt-on. Integration pattern is not FFI-friendly. **Higher risk.**
- **PETSc GAMG**: similar concerns to hypre — MPI-first, GPU path less mature than AMGx.
- **RAPIDS cuGraph AMG**: does not exist as a published API (cuGraph has traversal/centrality algorithms; no AMG solver). Not an option.
- **cuDSS**: a direct sparse solver, not iterative AMG. Different solver regime. Not a substitute — but worth noting for future work: for problem sizes ≤ a few million unknowns where factor fill-in is tolerable, cuDSS could be a Phase 5 comparison point.
- **Custom RS-AMG on cuSPARSE**: the brief's Option C. Coarsening strategies (classical Ruge-Stüben, PMIS, HMIS, aggregation), interpolation operators (classical, standard, extended, distance-2), and smoother hierarchies are 1000s of person-hours of expertise. **This is what AMGx has accumulated over 10+ years.** We will not reimplement it.

**Assessment**: AMGx remains the best path. Alternatives exist if AMGx were dead, but AMGx is not dead.

### §1.4 Honest assessment — where the risk is

AMGx's risks are not "is it alive", they are:

1. **CUDA 12.4 is between explicitly-tested versions.** Step 1 of §9 is a hard gate.
2. **AMGx's config system is a JSON string**, not a typed API. The solver's behavior is entirely driven by a config file like `AMG_CLASSICAL_PMIS` or a custom JSON. We have to curate a configuration that works for our stiff 3-D Laplacian with density jumps, and there is no "universal default". Math expert's doc (`phase4_amg_math_foundation.md`) specifies starting points; engineering must wire this config into the FFI.
3. **AMGx is designed around MPI-multi-GPU from the ground up.** Single-GPU single-process is a degenerate case. The `AMGX_resources_create_simple` API is the single-GPU shortcut but there are forum reports of it needing careful device-context management.
4. **Memory overhead is variable and poorly documented.** AMGx's setup phase (coarsening) builds a hierarchy of matrices; total memory is 3–5× the fine-grid matrix depending on coarsening aggressiveness. On our 4 GB card at 128³ this is 2.5–3 GB peak — workable with the `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` guard but the margin is thin.
5. **The handle-leak-on-process-exit pattern (Phase 2 cuSPARSE)** extends naturally here — `AMGX_finalize` is documented to require careful ordering vs CUDA context teardown, and we have no robust Python-process-exit hook. Phase 2 accepted a similar handle leak; Phase 4 will too.

**None of these are deal-breakers.** They are all STOP signals in §10.

---

## §2 Integration Strategy — Option Evaluation and Recommendation

Four candidate paths, per brief:

| Option | Description | Verdict |
|---|---|---|
| A. pyamgx (Python wrapper) | Install pyamgx, use its Python API | **Reject** (unmaintained, Cython cost, JAX pointer extraction fragile) |
| B. Thin C++ FFI wrapper around AMGx C API | Mirror Phase 2: `amgx_setup.cc`, `amgx_solve.cc` etc., link `libamgxsh.so` | **Recommend** |
| C. Rebuild AMG ourselves on cuSPARSE | Write classical RS-AMG + interpolation + smoother stack from scratch | **Reject** (1000s of hours; no path to compete with AMGx) |
| D. Abandon AMG, pivot to mixed-precision DILU / block preconditioners | Out of scope for Phase 4 | **Reject** (doesn't address `N^{1/3}` iter scaling — math expert §1.1) |

### §2.1 Why Option B over Option A

| Dimension | Option A (pyamgx) | Option B (own C++ wrapper) |
|---|---|---|
| New dependencies | pyamgx + Cython | None (pure C++ → AMGx C API) |
| JAX device-pointer handoff | Via `upload_raw` on `uintptr_t` from `jax_arr.unsafe_buffer_pointer()` | Direct `buffer.typed_data()` from FFI handler — identical to Phase 2 pattern |
| Project maintenance | pyamgx unmaintained (issue #40 unanswered) | We own the wrapper |
| Compilation path | `pip install pyamgx` calls its setup.py which needs AMGx env vars — fragile | CMake extension of our existing `dilu/cusparse/CMakeLists.txt` recipe — known path |
| Lifecycle control (handles) | pyamgx's Python-object-owned handles; GC timing non-deterministic | Our process-global registry; deterministic (mirror Phase 2) |
| JAX JIT / `ffi_call` compatibility | pyamgx does not know about JAX; we write a Python-side shim that does a JAX→pyamgx→solve→JAX bounce | Native `ffi_call` custom target — same as Phase 2/3 |
| Code size estimate | ~200 LoC Python shim + unknown pyamgx maintenance debt | ~400 LoC C++ + ~100 LoC Python (within Phase 2's envelope) |
| Performance ceiling | Subject to Python-side overhead around pyamgx calls | Same as Phase 2 (all device work under one custom-call) |

Option B is not meaningfully harder than A and is substantially more maintainable. We recommend Option B.

### §2.2 Why Option B over Option C (build our own AMG)

AMG is an **algorithm family**, not a single algorithm. Choices include:
- Coarsening: classical Ruge-Stüben (aggressive), PMIS, HMIS, aggregation-based (AMGx default)
- Interpolation: classical (direct, standard, extended, multi-pass), aggregation (Jacobi smoothed, energy-minimizing)
- Smoothers: Jacobi, Gauss-Seidel (colored), block-Jacobi, polynomial (Chebyshev), ILU(0) as smoother
- Cycle type: V-cycle, W-cycle, F-cycle, K-cycle
- Number of levels, coarse grid size threshold, coarse solver (direct Cholesky, AMG recursion, etc.)

Each combination has failure modes (e.g., aggressive coarsening at density jumps produces bad interpolation; polynomial smoothers can diverge on indefinite problems). Reimplementing this is a 1000+-hour effort with high risk of producing a solver that underperforms AMGx on our specific problem class.

**AMGx's value is the decade of NVIDIA engineering that has tuned these choices**. Reimplementing is not an engineering challenge we can win.

### §2.3 Why Option D is rejected

Option D ("mixed-precision DILU or inexact block preconditioners") does not address the root problem identified in the scaling report: **iter count grows as `N^{1/3}`** regardless of DILU variant. Mixed precision saves per-iter arithmetic cost but does not change iter count. Block preconditioners are a different DILU structure but still inherit $\sqrt{\kappa}$ scaling. These are **Phase 5+ concerns**, not Phase 4.

### §2.4 Recommendation

**Option B: thin C++ FFI wrapper around AMGx's C API.**

Code layout (§7) mirrors Phase 2's `dilu/cusparse/` directory almost verbatim. The new top-level artifact is `dilu/amgx/` with its own CMakeLists, cpp/, python/, tests/, bench/.

---

## §3 AMGx FFI Primitive Design (Option B)

Four primitives, mirroring Phase 2 and 3's 4-primitive opaque-token pattern:

| Primitive | Invocation | Hot path? | Host ↔ device copy |
|---|---|---|---|
| `amgx_setup` | Once per matrix pattern + AMG config | No (one-shot) | 0 (coefficient upload is D→D; token write is 8-byte H→D, same as Phase 2 analyze) |
| `amgx_update_coefficients` | When A's values change, pattern fixed | Warm (per AM timestep) | 0 (D→D coefficient copy only) |
| `amgx_solve` | Per PCG-like outer-iter or per standalone solve | **Yes** | 8-byte D→H for token (tolerated exception, same as Phase 2 apply) + optional 16 MB D→D for `b`, 16 MB D→D for `x` out (both at 128³) |
| `amgx_release` | Plan teardown | No | 8-byte D→H for token |

Note the two-primitive model I am **not** using: in `amgx_solve` itself, one could conceivably re-upload coefficients every call for a fresh solve on a new matrix — this is wasteful because AMGx's setup phase (coarsening hierarchy construction) is the expensive one and must be amortized. Hence the split: `setup` = pattern + values + hierarchy; `update_coefficients` = values only, reuse hierarchy; `solve` = b → x.

### §3.1 `amgx_setup` — the analyze-equivalent

**Purpose**: given a CSR triplet + a config JSON string, build the AMG hierarchy and return an opaque `uint64` token.

**Signature**:
```
Inputs (device-resident, C-contiguous):
  row_ptr     : int32[n+1]         -- CSR row offsets (N+1 entries)
  col_idx     : int32[nnz]         -- CSR column indices
  values      : float64[nnz]       -- CSR values (seed for setup; see note below)
  diag_values : float64[n]         -- REDUNDANT with `values` (diagonal entries live inside CSR),
                                      included because AMGx's `AMGX_matrix_upload_all` separates them.
                                      We will pass NULL and let AMGx extract from CSR.

Inputs (host-resident, static via `static_argnums`):
  config_json : Python str         -- passed out-of-band (see §3.1.1); NOT a traced tensor
  N           : int                -- derived from row_ptr.shape[0]-1; static

Output (device-resident):
  token       : uint64[1]          -- 0-d, opaque plan handle
```

**Work** (inside the handler):
1. `AMGX_initialize` (once per process, guarded by `std::once_flag`).
2. Get-or-create `AMGX_resources_handle` (one per device, singleton).
3. `AMGX_config_create` from the supplied JSON string. **⚠️ CONFIG LIFETIME — load-bearing correctness rule**: the `AMGX_config_handle` must live as long as the `AMGX_solver_handle` that references it. Internally `Solver::m_cfg` is a **raw pointer** into the config object, NOT a deep copy. Destroying the config after `AMGX_solver_create` (or after `AMGX_solver_setup`) leaves the solver with a dangling pointer; the first `AMGX_solver_solve` call dereferences it via `AMG_Config::getParameter<int>` → SIGSEGV. Therefore the config MUST be stored in the plan entry and destroyed ONLY at `release` time, AFTER the solver. See PROJECT_SUMMARY §8.3.
4. `AMGX_matrix_create(mat, rsc, mode=AMGX_mode_dDDI)` — d=device, D=double (A/b/x type), D=double (intermediate), I=int (indices).
5. `AMGX_matrix_upload_all(mat, n, nnz, 1, 1, row_ptr_dev, col_idx_dev, values_dev, /*diag=*/NULL)`. Per §5, AMGx will `cudaMemcpyDefault` from our device pointers into its own device buffers — one D→D copy of size (N+1+nnz) int32 + nnz float64 ≈ 200 MB at 128³.
6. `AMGX_solver_create(solver, rsc, mode, cfg)`.
7. `AMGX_solver_setup(solver, mat)` — the expensive AMG-hierarchy construction. Returns when coarsening + setup phase are complete.
8. `AMGX_vector_create(b_vec, rsc, mode)`; `AMGX_vector_create(x_vec, rsc, mode)`. These are plan-owned vector handles; we rebind device pointers via `upload`/`download` per solve.
9. Stash `{resources, cfg, matrix, solver, b_vec, x_vec, fingerprint}` in `AmgxPlanEntry`.
10. Insert plan into process-global `unordered_map<uint64, AmgxPlanEntry*>`; return `uint64` token as 8-byte H→D memcpy to output.

**JAX tracing note** (§3.1.1): `config_json` **cannot** be a traced tensor — it is a Python string that gates solver behavior. It is therefore threaded into the user's `jax.jit` via `static_argnums`, and passed to the FFI handler via `ffi_call`'s attribute/payload mechanism. JAX FFI supports scalar attributes through `XLA_FFI_Attribute`. The C++ handler reads the attribute with `.Attr<ffi::String>("config_json")`. This is the standard pattern for passing strings through FFI and is already exercised by `jaxlib` internals.

**Errors**:
- `InvalidArgument` on shape mismatch, empty CSR, config JSON parse failure (AMGx returns `AMGX_RC_CONFIGURATION_ERROR`).
- `Internal` on any AMGx status ≠ `AMGX_RC_OK` (wrap AMGx's error string via `AMGX_get_error_string`).

**Idempotence**: each call produces a new token; each setup allocates its own matrix/solver handles. The shared thing across calls is `resources` + the AMGx library-level `AMGX_initialize` state.

### §3.2 `amgx_update_coefficients` — values-only refresh

**Purpose**: A's values change (e.g., next AM timestep with updated $\rho(\mathbf{x},t)$), pattern unchanged. Replace the coefficients in AMGx's owned matrix buffer, **keeping the solver's coarsening hierarchy and interpolation operators frozen**. This is the core reason AMGx outperforms "re-setup every timestep" by an order of magnitude.

**Signature**:
```
Inputs (device-resident):
  token  : uint64[1]
  values : float64[nnz]     -- new CSR values, same pattern

Output:
  status : int32[1]         -- 0 = ok, nonzero = AMGx error code (plumbed as traced
                               value for downstream dependencies)
```

**Work**:
1. Read token (8-byte D→H, same as Phase 2 apply).
2. Plan cache lookup; fingerprint `(n, nnz)` check.
3. `AMGX_matrix_replace_coefficients(matrix, n, nnz, new_values_dev, diag=NULL)` — AMGx documents this as "replace the nonzero values while keeping row/col structure". Under the hood: D→D copy only, no coarsening re-run. Math expert §X.Y confirmed this is the right API (search references `AMGX_matrix_replace_coefficients` in Julia + pyamgx wrappers both).
4. `AMGX_solver_resetup(solver, matrix)` — this is the subtle bit: AMGx may require a `resetup` call (not full `setup`) to refresh operator-dependent smoother state (e.g., the Jacobi relaxation parameter or ILU(0) smoother diagonal). Math expert's doc will specify whether this is needed for our chosen config. **Engineering-side: we call it unconditionally because its cost is O(nnz), not the expensive O(coarsen-hierarchy) re-setup**.

**Errors**: `InvalidArgument` on fingerprint mismatch; `Internal` on any AMGx status.

**Note on `resetup` vs `setup`**: AMGx has both `AMGX_solver_setup` (full coarsening rebuild) and `AMGX_solver_resetup` (partial — reuses coarsening graph, recomputes operator values). The difference is ~100× in runtime for typical AMG configs. Using `resetup` is the whole point of `amgx_update_coefficients`. If we discover it's insufficient for our problem class (math expert's §X), we fall back to `AMGX_solver_setup` and accept the per-timestep cost.

### §3.3 `amgx_solve` — the hot path

**Purpose**: given a plan token and a right-hand-side `b`, produce `x ≈ A⁻¹ b` using the pre-built AMG hierarchy.

**Signature**:
```
Inputs (device-resident):
  token : uint64[1]         -- from setup
  b     : float64[n]        -- right-hand-side
  x0    : float64[n]        -- initial guess (pass zeros for cold-start)

Output (device-resident):
  x     : float64[n]        -- solution
  iters : int32[1]          -- iteration count AMGx took (diagnostics)
  status: int32[1]          -- 0 = converged, 1 = did not converge within max_iters
```

**Work**:
1. Read token (8-byte D→H).
2. Plan cache lookup.
3. `AMGX_vector_upload(plan->b_vec, n, 1, b_dev)` — AMGx copies D→D into its owned b buffer (~16 MB at 128³).
4. `AMGX_vector_upload(plan->x_vec, n, 1, x0_dev)` — initial guess.
5. `AMGX_solver_solve(solver, b_vec, x_vec)`. Single call. AMGx runs its solve entirely on device, on the CUDA stream bound via `AMGX_config_add_parameters("solver:cuda_stream=...")`  (see §3.3.1).
6. `AMGX_solver_get_iterations_number(solver, &iters_host)` — host-side scalar read.
7. `AMGX_solver_get_status(solver, &status_host)` — host-side scalar read.
8. `AMGX_vector_download(plan->x_vec, x_out_dev)` — D→D copy to XLA's output buffer (~16 MB at 128³).
9. Write `iters_host`/`status_host` to XLA's output buffers via 4-byte H→D copies each.

**JAX contract**:
- Inputs `token, b, x0` are traced (JAX `DeviceArray` → FFI `Buffer`).
- Outputs `x, iters, status` flow back as traced JAX arrays. The caller can `jax.jit` the whole PCG loop with `amgx_solve` inside.
- The `static_argnums` burden stays on `amgx_setup` (config JSON). `solve` has no static args — every input is traced.

**Why upload/download are not "zero copy"** (§5 defers deep discussion): `AMGX_vector_upload` is the only documented path to feed a vector to an AMGx solver, and it always copies into AMGx's owned buffer (`cudaMemcpyDefault` semantics). The copy is D→D, not D→H→D, so it stays on-device and costs ~60 µs per 16 MB on the 3050. Measurable but not structural. If per-solve this is a bottleneck (it won't be — AMGx solve itself is ~seconds at 128³), we would need to pursue a hypothetical "rebind internal buffer" API which AMGx does not expose. **Option 5c in the brief's §5 spelled this out correctly; that's what we're doing.**

**§3.3.1 Stream binding**: AMGx's internal compute runs on CUDA streams it manages internally, **not** the stream XLA supplies. There is no public `AMGX_solver_set_stream` API equivalent to `cusparseSetStream`. The closest documented path is:
- Via the config JSON, set `"solver:cuda_stream=<ptr-as-int>"`. This is a string-formatted device pointer in the JSON.
- Some forum threads suggest AMGx will accept a pre-made stream via `AMGX_config_add_parameters`. This is not universally documented.
- **Engineering-side risk**: we may have to either (a) synchronize the XLA stream before entering AMGx (safe but loses pipelining with surrounding ops) or (b) accept that AMGx solve runs on its own internal stream and rely on CUDA's default-stream synchronization semantics. **Step 2 of §9 verifies this empirically.** If AMGx insists on its own stream, we lose a small amount of overlap but the solve itself is so expensive that the surrounding pipelining loss is noise.

### §3.4 `amgx_release` — teardown

**Signature**:
```
Input: token : uint64[1]
Output: status : int32[1]
```

**Work**: plan cache lookup; destroy `solver`, `matrix`, `b_vec`, `x_vec`, `config` in AMGx-documented order; `free` the `AmgxPlanEntry`; return success. Idempotent (releasing a gone token → warn + success).

**Not called**: `AMGX_finalize`. Process-lifetime leak, same as Phase 2's cuSPARSE handle. Rationale: AMGx's finalize races with XLA shutdown in ways we cannot predict; better to leak one library-level resource than crash on exit.

### §3.5 What is NOT in Phase 4 (FFI-level)

- No `amgx_solve_iterative` variant that runs multiple RHSes in a batch — out of scope.
- No multi-GPU `AMGX_matrix_upload_distributed` — single-GPU only.
- No complex (non-real) values.
- No custom user-defined preconditioner-inside-AMGx. Use AMGx's built-in preconditioner stack.

---

## §4 Handle / Resource Lifecycle

AMGx's handle model is richer than cuSPARSE's, which is the primary complexity increase over Phase 2.

### §4.1 Handle hierarchy (per AMGx docs)

```
AMGX library init                 ← AMGX_initialize(), once per process
├── AMGX_resources_handle         ← one per device (or per MPI-rank group)
│   ├── AMGX_matrix_handle (many) ← one per solved matrix
│   ├── AMGX_vector_handle (many) ← one per vector (b, x, scratch)
│   └── AMGX_solver_handle (many) ← one per configured solver
└── AMGX_config_handle (many)     ← one per distinct solver configuration JSON
```

Ownership: resources outlive matrices/vectors/solvers. Configs are consumed at solver-create time and then may be destroyed.

### §4.2 Process-global registry design

Mirror Phase 2's `plan_registry.h`:

```cpp
// dilu/amgx/cpp/plan_registry.h
struct AmgxPlanEntry {
  PatternFingerprint fingerprint;   // (N, nnz), same as Phase 2

  // AMGx handles owned by this plan.
  AMGX_matrix_handle matrix = nullptr;
  AMGX_solver_handle solver = nullptr;
  AMGX_vector_handle b_vec = nullptr;
  AMGX_vector_handle x_vec = nullptr;

  // CONFIG LIFETIME: the solver holds a raw pointer into this config
  // (Solver::m_cfg is not a deep copy). Config MUST outlive the solver,
  // and release() MUST destroy solver BEFORE config. See §3.1 step 3
  // and PROJECT_SUMMARY §8.3.
  std::string config_json;

  // Plan-owned D→D-copy-of-pattern CSR buffers? Option:
  //   AMGx already owns its own copy of the CSR (it did the D→D at setup).
  //   We do NOT need to own a second copy. Fingerprint is (N, nnz) only,
  //   as in Phase 2 post-fingerprint-redesign (cuSPARSE report §7).
};
```

Additionally, **process-global singletons** (lives in `plan_registry.cc`):

```cpp
// Lazy-initialized, never destroyed. Mirror Phase 2's cuSPARSE handle pattern.
static std::once_flag g_amgx_init_flag;
static AMGX_resources_handle g_resources = nullptr;
static std::thread::id g_init_thread_id;

AMGX_resources_handle get_amgx_resources(int device);
```

`AMGX_initialize()` is called once per process inside the `std::call_once` lambda. `AMGX_resources_create_simple(&g_resources, default_cfg)` is called with a minimal config that satisfies AMGx's resource-creation contract; this default cfg is *not* used for actual solves — each `amgx_setup` call builds its own config from the user-supplied JSON.

**Thread-safety**: AMGx documentation says resources are thread-safe for reads but not writes. Our pattern is: one FFI handler per JAX call, invoked by XLA's single scheduler thread. Phase 2's `handle_called_from_wrong_thread()` tripwire carries over: we record the thread ID of first `AMGX_initialize` and assert it on every subsequent FFI entry. If it ever trips, we STOP and investigate.

### §4.3 Compatibility of AMGx's stateful model with our singleton pattern

**Answer: yes, with one subtlety.** AMGx resources are designed to be long-lived (per-device), and solvers/matrices can be created/destroyed on demand. This maps well onto our "one resources handle, many plans" pattern. The subtlety:

- **⚠️ CORRECTION (2026-04-22, caught by blind-reproduction test)**: a prior version of this section instructed destroying configs after `solver_create`. **That is wrong and causes SIGSEGV on first solve** (see §3.1 step 3 and PROJECT_SUMMARY §8.3). The correct rule is: **config must be stored in the plan entry and destroyed AFTER the solver at release time**. The solver holds a raw pointer into the config object (`Solver::m_cfg`), not a copy. Config destruction order at release: `solver → vectors → matrix → config`.
- **Vectors must be created once and reused**. Creating a fresh `AMGX_vector_handle` per `solve` call is documented as slow (~microseconds of AMGx overhead per create/destroy). Our design creates b_vec, x_vec once per plan and reuses — uploads rebind the data pointer, not the handle.

### §4.4 `AMGX_initialize` lifecycle

The brief asked specifically about this. Detailed answer:

- **Who calls `AMGX_initialize`**: the process, exactly once, on the *first* invocation of any of our FFI entry points that touches AMGx. Guarded by `std::once_flag`.
- **Who calls `AMGX_finalize`**: nobody. Process leaks the AMGx init state. Phase 2's cuSPARSE handle leak sets the precedent; the rationale is the same (finalize races with CUDA context destruction on Python interpreter shutdown; we cannot install a reliable `atexit` hook that runs before JAX tears down CUDA).
- **Alternative considered**: register an `atexit` Python handler that calls a C `amgx_process_finalize()` function we expose. The concern: Python's `atexit` runs after most module shutdowns but before *all* — in particular, JAX may deregister its CUDA context before our `atexit` fires, in which case the AMGx finalize would operate on an invalid CUDA state. Phase 2's judgment on this exact scenario was "leak; not worth the debugging". Phase 4 inherits that judgment.
- **Fork-safety**: same concern as Phase 2 — a `fork()` child inherits our registry but not the CUDA context. We install a `pthread_atfork` child-handler that clears the registry (but does **not** call `AMGX_finalize` in the parent — the parent is fine). Phase 4 is single-process single-device; this guard is defensive only.

### §4.5 Cross-thread safety

**Phase 2's rule**: handle was created on thread T; accessing it from any other thread is a tripwire that aborts with a diagnostic message. **Phase 4 inherits this rule verbatim.** Reasons:
- AMGx's own thread-safety claims are not strong enough to rely on for multi-thread concurrent usage.
- JAX's single-scheduler-thread design means we should never see cross-thread entries in practice.
- The tripwire catches misuse early with a clear message instead of a silent data race that would corrupt the AMG hierarchy.

Implementation: `plan_registry.cc` stores `g_init_thread_id` at first AMGx init; `amgx_solve`, `amgx_update_coefficients`, `amgx_release` all call `check_called_from_right_thread()` at entry.

---

## §5 The Zero-Copy Challenge — Concrete Resolution

This is the most important technical question in the Phase 4 design. The brief offered four options (5a through 5d). Based on the web-verified evidence of how `AMGX_matrix_upload_all` and `AMGX_vector_upload` are implemented internally (see §1 — they use `cudaMemcpyDefault`, which routes by pointer kind), my concrete recommendation is a **hybrid of 5a and 5c**:

### §5.1 The recommendation: "device-pointer-in, D→D-copy-internal, no PCIe traffic"

- **At setup**: pass JAX's device CSR pointers (`row_ptr.typed_data()`, `col_idx.typed_data()`, `values.typed_data()`) directly to `AMGX_matrix_upload_all`. AMGx internally detects these are device-resident (via `cudaMemcpyDefault`) and performs **device-to-device** copies into its own managed buffers. The JAX-side buffers are NOT shared; AMGx owns its copy.
- **At update_coefficients**: same story. `AMGX_matrix_replace_coefficients` takes a `const void*` which AMGx copies D→D into its owned values buffer.
- **At solve**: pass JAX device `b` pointer to `AMGX_vector_upload`, get D→D copy in. Pass JAX device `x` output pointer to `AMGX_vector_download`, get D→D copy out.

**Net effect on PCIe traffic: zero.** Every byte stays on-device.

**Net effect on intra-device traffic**: one D→D copy at setup (~200 MB at 128³), one D→D copy at every `update_coefficients` (~120 MB at 128³ — smaller because only values, not indices), one D→D copy of `b` plus one of `x` per `solve` (~16 MB each at 128³, so ~32 MB round-trip).

**Is this "zero copy" in the strictest sense?** No. AMGx owns its own device buffers; we cannot ask it to read directly from JAX's buffers. But in terms of **PCIe/system-memory traffic** — which is the goal stated in the brief and CLAUDE.md ("zero host-device copy in steady state") — **yes, it is zero-copy**. Every copy is device-local.

### §5.2 Why not Options 5a/5b/5d

- **Option 5a (pass device pointer to a dedicated "from_device_pointer" API)**: There is no such API in AMGx's public surface. `AMGX_matrix_upload_all` is the upload entry point; it does not distinguish host from device. The detection happens via `cudaMemcpyDefault` inside AMGx, not via a separate API.
- **Option 5b (`AMGX_pin_memory` trickery)**: `AMGX_pin_memory` is for pinning *host* memory for faster H→D transfers. It does not work on device memory. Irrelevant to our case.
- **Option 5d (shared buffer ownership)**: Would require AMGx to expose a "here is my internal buffer, bind my matrix state to it" API. No such API exists. We would need to monkey-patch AMGx's internal `MemoryManager` singleton, which violates our platform-compatibility rule.

### §5.3 Measured cost budget

At 128³ ($N = 2.1 \times 10^6$, nnz $\approx 14.6 \times 10^6$):
- **Setup D→D copy**: 200 MB / 400 GB/s device bandwidth → 0.5 ms. AMGx's *setup* cost is seconds (hierarchy construction). D→D upload is noise.
- **update_coefficients D→D**: 120 MB / 400 GB/s → 0.3 ms. AMGx's `resetup` cost is tens to hundreds of ms. D→D upload is noise.
- **Per-solve D→D (b upload + x download)**: 32 MB total / 400 GB/s → 0.08 ms. AMGx's *solve* cost is hundreds of ms to seconds. D→D I/O is noise.

**Conclusion**: the device-to-device copies are ≪1% of AMGx-attributable time at any grid size we care about. They are **not** the dominant cost and do **not** need further optimization.

### §5.4 What I am not certain about

- Whether AMGx's internal `cudaMemcpyDefault` on our device pointer triggers any synchronizing behavior that would stall JAX's stream. Web search did not resolve this. **Step 3 of §9 measures it empirically.**
- Whether at very small N the D→D copy overhead becomes visible relative to AMGx's own setup cost. At N=4k (16³), setup is ~10 ms, D→D copy is ~6 µs — noise. At N=512 (8³), setup might drop to ~1 ms and D→D might be ~1 µs — still noise.
- Whether AMGx's internal buffer sharing between its `matrix` handle and its `solver` handle introduces any extra copies within AMGx. This is an implementation detail we cannot introspect; we trust AMGx's memory pool to be reasonable.

---

## §6 Matrix Pattern Fingerprint (Correctness Discipline)

Same problem as Phase 2/3: AMG's setup phase is pattern-dependent. If pattern changes between setup and solve (without calling `amgx_setup` again), we produce silently-wrong results. Phase 4's defensive layer mirrors Phase 2's:

### §6.1 Fingerprint

**`PatternFingerprint = (N, nnz)`** — inherited verbatim from Phase 2. Same rationale: we cannot hash the pattern bytes at every `solve` call (reduction cost; and AMGx has already moved the pattern into its own buffer so we would be hashing our local copy, not AMGx's).

### §6.2 What this catches

- Accidentally feeding `solve` from a different-sized matrix: fingerprint mismatch → `InvalidArgument`.
- Pattern-preserving value changes: fine, no flag (this is the `update_coefficients` flow).

### §6.3 What this doesn't catch

- Caller holds the same CSR pointers but has overwritten their contents with a different pattern. AMGx already owns its own copy of the pattern after setup, so this is actually **safer in Phase 4 than in Phase 3** — we are not reading the caller's pattern at solve time, so the caller can mutate their own CSR and it doesn't break our solver. Fingerprint check is a pure size sanity check.

### §6.4 Additional guard: setup success flag

Phase 4 adds one new fingerprint field beyond Phase 2's `(N, nnz)`:

```cpp
struct PatternFingerprint {
  int32_t N;
  int32_t nnz;
  bool setup_converged;   // true iff AMGx setup did not return an error
  // (Optionally: store AMG hierarchy level count for diagnostics.)
};
```

Why: AMGx's setup phase can fail (e.g., "matrix not SPD", "coarsening produced a degenerate grid", "out of memory during hierarchy construction"). If setup fails, we return `InvalidArgument` to the caller from `amgx_setup`, and we do **not** insert the plan into the cache. This is a non-issue for downstream correctness — a failed setup means no token was produced — but it's cleaner to flag it explicitly.

### §6.5 Python-side strong-ref guard (mirrored from Phase 3)

The Python `AmgxPlan` class holds strong refs to `row_ptr, col_idx` for the plan's lifetime. This prevents JAX from garbage-collecting them while the plan holds AMGx-internal pointers into a D→D copy we no longer control. (Actually AMGx owns its copy, so this is extra-cautious — but costs nothing and matches Phase 2's style.)

---

## §7 Build System Changes

Phase 2's `dilu/cusparse/CMakeLists.txt` (lines 1–57) sets the template. Phase 4 adds a new library alongside it: `libdilu_amgx.so` at `dilu/amgx/CMakeLists.txt`. **Phase 1/2/3 artifacts stay untouched.**

### §7.1 Build-time AMGx dependency

AMGx is **not** a CMake package with `find_package(AMGX REQUIRED)` support (verified by fetching the repo README: no mention of a CMake config module). So we build AMGx from source as a separate step, install to a known prefix, and link against `libamgxsh.so` + its include dir.

**Build recipe** (spelled out in `dilu/amgx/build_amgx.sh`, a helper that users run once):
```bash
cd /opt/
git clone --branch v2.5.0 --depth=1 https://github.com/NVIDIA/AMGX.git
cd AMGX && mkdir build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_INSTALL_PREFIX=/opt/amgx-2.5.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMGX_NO_RPATH=OFF \
    -DMPI_FOUND=FALSE
make -j8 install   # produces /opt/amgx-2.5.0/lib/libamgxsh.so
```

Notes:
- Clone a **pinned tag**, not `main`. `v2.5.0` is what we tested the architecture against.
- `MPI_FOUND=FALSE` disables the multi-GPU build path. We are single-GPU only.
- Install prefix out-of-tree so the system never picks up a rogue AMGx from PATH.

### §7.2 Our CMakeLists

```cmake
# dilu/amgx/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(dilu_amgx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# JAX FFI headers (Phase 1 pattern, unchanged).
if(NOT DEFINED JAX_FFI_INCLUDE_DIR)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir(), end='')"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    RESULT_VARIABLE JAX_FFI_PROBE_RC)
  if(NOT JAX_FFI_PROBE_RC EQUAL 0)
    message(FATAL_ERROR "jax.ffi.include_dir() probe failed.")
  endif()
endif()

find_package(CUDAToolkit REQUIRED)

# AMGx installed prefix — user passes via env or CMake cache.
if(NOT DEFINED AMGX_ROOT)
  set(AMGX_ROOT "/opt/amgx-2.5.0")
endif()
find_library(AMGX_LIB amgxsh PATHS "${AMGX_ROOT}/lib" REQUIRED)
set(AMGX_INCLUDE_DIR "${AMGX_ROOT}/include")
message(STATUS "AMGx lib: ${AMGX_LIB}")

add_library(dilu_amgx SHARED
  cpp/plan_registry.cc
  cpp/amgx_setup.cc
  cpp/amgx_update_coefficients.cc
  cpp/amgx_solve.cc
  cpp/amgx_release.cc)

target_include_directories(dilu_amgx PRIVATE
  ${JAX_FFI_INCLUDE_DIR}
  ${AMGX_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp)

target_link_libraries(dilu_amgx PRIVATE
  ${AMGX_LIB}
  CUDA::cudart
  # AMGx transitively needs these, but we link explicitly for loud failure:
  CUDA::cublas
  CUDA::cusparse
  CUDA::cusolver)

# Per CLAUDE.md: no -ffast-math, no --use_fast_math. -O2, not -O3 for CUDA.
target_compile_options(dilu_amgx PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -fno-fast-math>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)

set_target_properties(dilu_amgx PROPERTIES
  OUTPUT_NAME "dilu_amgx"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
```

### §7.3 CUDA version compatibility probe

**Phase 2 STOP #1** ("cuSPARSE version skew between jaxlib and system CUDA") has an AMGx analog: our compiled `.so` expects AMGx built against a specific CUDA minor. At Python-side registration time, we want to fail loudly if:
- `libamgxsh.so` reports an AMGx API version we don't know about.
- CUDA runtime's `cudaRuntimeGetVersion()` disagrees with what AMGx was built for.

**Implementation**: `dilu/amgx/python/_probe.py` performs:
1. `ctypes.CDLL(libamgxsh_path).AMGX_get_api_version` (if exposed) — match to our compile-time constant.
2. `cudaRuntimeGetVersion()` via `ctypes.CDLL(libcudart.so)` — log for diagnostics.
3. Attempt `AMGX_initialize()` in a subprocess; catch failures.

If any probe fails, user is asked to rebuild AMGx against the current CUDA toolkit.

### §7.4 Dependency graph

AMGx transitively depends on:
- `CUDA::cudart` (runtime)
- `CUDA::cublas` (BLAS3 for smoothers)
- `CUDA::cusparse` (SpMV in smoothers and Galerkin product)
- `CUDA::cusolver` (coarse-grid direct solver)
- `pthread`, `m` (trivial)
- Optionally: `MPI`, `MAGMA`, `MKL` — all disabled in our build.

We link `CUDA::cusparse` and `CUDA::cublas` explicitly even though they are transitive through `libamgxsh.so` — this makes any linkage bug loud rather than silently pulling wrong versions. **Note**: this creates a coupling to cuSPARSE that Phase 2's code already has — no new conflict.

### §7.5 Directory layout

```
dilu/
├── ffi_mvp/             [Phase 1, unchanged]
├── cusparse/            [Phase 2, unchanged]
├── multicolor/          [Phase 3, unchanged]
└── amgx/                [Phase 4, new]
    ├── CMakeLists.txt
    ├── build.sh              # builds libdilu_amgx.so
    ├── build_amgx.sh         # helper: builds AMGx from source (runs once)
    ├── cpp/
    │   ├── plan_registry.h / plan_registry.cc
    │   ├── amgx_setup.cc
    │   ├── amgx_update_coefficients.cc
    │   ├── amgx_solve.cc
    │   └── amgx_release.cc
    ├── configs/
    │   ├── dilu_classical_v_cycle.json    # math expert's recommended default
    │   ├── aggressive_coarsening.json     # memory-lean variant for 128³
    │   └── mini_amg_test.json             # 5-level AMG for small-grid testing
    ├── python/
    │   ├── __init__.py
    │   ├── _probe.py              # §7.3 version probe
    │   ├── registration.py        # same pattern as Phase 2
    │   ├── plan.py                # AmgxPlan context manager
    │   └── wrapper.py             # Python wrappers for the 4 FFI primitives
    ├── tests/
    │   ├── _harness.py
    │   ├── conftest.py
    │   ├── test_t8_setup_correctness.py
    │   ├── test_t9_iter_count_128.py
    │   ├── test_t10_wall_time.py
    │   ├── test_t11_under_jit.py
    │   ├── test_t12_physical_64.py        # reuses Phase 2.5 benchmark driver
    │   ├── test_t13_memory.py
    │   └── test_test_c_64_sanity.py       # 5–15 iter F2-class correlation check
    └── bench/
        ├── bench_amgx_vs_dilu.py           # the headline table
        ├── bench_amgx_setup_amortize.py
        └── plots/                           # scale_64/, scale_128/ mirrors Phase 3
```

Lexical separation from Phase 2/3 strictly enforced. No cross-phase imports. Phase 2's `physical_benchmark.py` is *copied* into `dilu/amgx/tests/test_t12_physical_64.py` with its `apply` callable swapped for an AMGx-based `solve` callable.

---

## §8 Testing Plan

Mirror Phase 2/3 structure, with tests numbered T8–T13 (Phase 2 used T4–T7, Phase 3 reused T4–T8 for its own implementations; Phase 4 continues the numbering with clear disambiguation by the file location).

### §8.1 T8 — AMG correctness against exact DILU-PCG

- **Setup**: 16³ stiff T7 matrix (exactly Phase 2's T7 baseline). Apply AMGx solve with a default classical-V-cycle config. Compare solution to Phase 2's DILU-PCG converged solution.
- **Acceptance**: `‖x_amgx − x_dilu‖_2 / ‖x_dilu‖_2 ≤ 10⁻⁸` (both solvers converged to the same linear system solution within tolerance).
- **Rationale**: two different preconditioners on the same SPD problem should agree at the solver's tolerance. Disagreement indicates a bug (wrong matrix upload, wrong RHS, wrong config).

### §8.2 T9 — Iteration count on 128³ stiff

- **Setup**: 128³ stiff T7 matrix (matches `phase3_scaling_64_128.md`'s benchmark). Tolerance 10⁻⁸.
- **Acceptance**: AMG-PCG iteration count ≤ **50 iters** (math expert's §X band is 5–30 iters; engineering budget adds a 2× safety factor for config-tuning noise).
- **Comparison**: Phase 2 DILU-PCG needed **186 iters** on this same problem. AMGx should show **≤ 25% of DILU's iter count**.
- **Rationale**: this is the headline claim of Phase 4. If this fails, Phase 4 has failed its core premise.

### §8.3 T10 — Wall-time improvement on 128³

- **Setup**: same as T9. Time the full PCG solve (setup + all apply iterations + download).
- **Acceptance**: AMGx total wall time < **0.5 ×** Phase 2 DILU-PCG wall time at 128³.
- **Context from scaling report**: Phase 2 DILU-PCG at 128³ takes 28.47 s. Budget for AMG: ≤ 14 s.
- **Rationale**: even with AMGx's higher setup cost, ≤ 50 iters × ~150 ms/iter = 7.5 s solve + setup overhead. Should fit comfortably under 14 s on a 3050.

### §8.4 T11 — Under-JIT HLO inspection

- **Setup**: wrap `amgx_solve` inside `jax.jit`. Inspect compiled HLO and runtime Nsight trace.
- **Acceptance**:
  - **1 custom-call op** per FFI primitive in HLO (same as Phase 2).
  - **0 `copy-start`/`copy-done` ops** on any user data (the 8-byte token D→H is handler-internal).
  - **0 cudaMemcpy H→D** in the per-solve critical path (confirmed via Nsight).
  - D→D copies *are* expected and allowed (they're internal to `AMGX_vector_upload`).
- **Rationale**: Phase 2/3's zero-copy HLO contract carries forward. D→D is fine; H→D is not.

### §8.5 T12 — Physical A/B/C benchmark at 64³

- **Setup**: exactly re-run Phase 2.5's Test A (divergence map, ρ-ratio 1000 sphere), Test B (static droplet spurious currents over 10 steps), and Test C (residual halo vs scipy `spsolve` gold). The preconditioner is swapped from Phase 2's `cusparse_dilu_apply` to `amgx_solve` with a fixed AMG-preconditioned-CG config.
- **Acceptance**:
  - Test A: `max|∇·u| ≤ 1e-7` (Phase 2 hit 4.13e-9 at 32³; Phase 3 3.29e-9; AMG should at least match Phase 2 at 64³, since AMG's iter count is much smaller at the same tolerance).
  - Test B: `‖u‖∞ @ t=10` same order of magnitude as Phase 2 (~1e-6). `max|div|` ≤ 1e-12 per step.
  - Test C: `corr(|E|, |∇log ρ|) ≤ 0.25` at iteration count **5–15** (not 50 as DILU needed — AMG should converge much faster). **F2 STOP signal** if corr > 0.5.
- **Rationale**: Phase 3 established the "no interface halo" red line. Phase 4 inherits it. AMG is known to introduce interpolation-operator-dependent error patterns near strong coefficient discontinuities — this test verifies ours doesn't.

### §8.6 T13 — VRAM usage at 128³

- **Setup**: measure `nvidia-smi` before and after `amgx_setup + amgx_solve` on 128³.
- **Acceptance**: peak delta ≤ **2 GB** VRAM.
- **Rationale**: the 3050 has ~3 GB headroom after desktop overhead. AMGx at 128³ is expected to peak at 2.5–3 GB (see §11). 2 GB is the budget; 3 GB triggers `Stretch grid declared` mode and test relaxation.
- **Stretch test**: if 128³ VRAM is ≤ 1.5 GB, report so — the `aggressive_coarsening.json` config may be unnecessarily lean and we can relax it.

### §8.7 Cross-check bench table

Mirror Phase 3's `bench_multicolor_vs_cusparse.py`: a single script `bench_amgx_vs_dilu.py` produces a markdown table with rows for each grid size (16³, 32³, 64³, 128³) and columns for Phase 2 DILU, Phase 3 multi-color, Phase 4 AMGx — showing `(T_setup, T_solve_median, T_total_pcg, iter_count)` per row. This is the number the user uses to decide Phase 4 success.

### §8.8 What we deliberately do NOT test in Phase 4

- Float32 or mixed-precision AMG. Float64 only.
- Multi-GPU AMG. Single-device only.
- Non-SPD matrices. Our AM Poisson is SPD.
- Adaptive mesh refinement cases. Static grid.
- AMGx's eigensolvers / non-linear solvers. Linear solver only.

---

## §9 Execution Checklist (12 ordered steps)

Ordered, blocking. Each step: **(a)** what, **(b)** acceptance, **(c)** likely failure.

### Step 1 — Build AMGx from source against CUDA 12.4

- **(a)** Clone `NVIDIA/AMGX` at tag `v2.5.0`. Configure with `cmake -DMPI_FOUND=FALSE -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_INSTALL_PREFIX=/opt/amgx-2.5.0`. Build, install.
- **(b)** `/opt/amgx-2.5.0/lib/libamgxsh.so` exists. `ldd libamgxsh.so` shows `libcudart.so.12`, `libcublas.so.12`, `libcusparse.so.12`, `libcusolver.so.11`.
- **(c)** CMake configure fails on CUDA 12.4 compatibility: STOP #1 (see §10). Don't patch AMGx source; escalate to user. Workaround path: try v2.4.0 as a fallback (supports CUDA 12.2, should forward-compat to 12.4).

### Step 2 — AMGx hello-world C program

- **(a)** Standalone `.c` file, no JAX, no FFI: initialize AMGx, upload a 5×5 tridiag CSR to a matrix handle, create a solver with a minimal AMG-PCG config, solve for a constant-1 RHS. Link against `libamgxsh.so` via one-line `gcc`.
- **(b)** Prints a 5-vector solution matching the hand-computed `A \ [1,1,1,1,1]ᵀ`.
- **(c)** Runtime AMGx error with a cryptic code: STOP. Check the error via `AMGX_get_error_string`. If it's `AMGX_RC_NOT_SUPPORTED` on an operation, the config JSON is wrong — iterate on the config.

### Step 3 — Stream-binding smoke test

- **(a)** Extend Step 2's program: create an explicit CUDA stream via `cudaStreamCreate`, pass its pointer to AMGx via the config JSON (`"solver:cuda_stream=<hex>"`). Run two AMGx solves on the same stream and profile with Nsight Systems.
- **(b)** Nsight trace shows both solves running on the user-supplied stream. No ops on stream 0 (default stream).
- **(c)** AMGx ignores the config key and uses its own internal stream. **Not a STOP** (§3.3.1 flagged this) — we fall back to "synchronize XLA stream before entering AMGx" pattern and accept the pipelining loss. Document the finding in the Phase 4 report.

### Step 4 — CMakeLists + empty `libdilu_amgx.so`

- **(a)** Copy Phase 2's CMakeLists to `dilu/amgx/CMakeLists.txt`; edit for AMGx link. All 4 FFI handler `.cc` files are empty stubs returning `ffi::Error::Success()`.
- **(b)** `./build.sh` produces `build/libdilu_amgx.so`. `nm -D` shows the 4 handler symbols. `ldd` shows `libamgxsh.so` linked.
- **(c)** Missing AMGx header (`<amgx_c.h>`): wrong `AMGX_INCLUDE_DIR`. Fix env var, retry.

### Step 5 — `amgx_setup` handler

- **(a)** Implement `amgx_setup.cc`: AMGx-init guard, resources fetch, matrix create, matrix upload (device pointers!), config parse, solver create, solver setup, insert PlanEntry into cache, return token.
- **(b)** T8 (16³ AMG correctness) passes on a reduced sub-case: setup completes without error; a single subsequent direct-call to `AMGX_solver_solve` with a trivial RHS produces the right answer in a test harness that bypasses the FFI solve path. VRAM delta after setup is ≤ 100 MB (for 16³).
- **(c)** `AMGX_matrix_upload_all` returns `AMGX_RC_BAD_PARAMETERS` — most likely a nnz/N mismatch or int32 CSR indexed with int32 vs int64 (AMGx defaults to int32, match). **STOP** if unresolvable after reading cuSPARSE docs.

### Step 6 — `amgx_solve` handler

- **(a)** Implement `amgx_solve.cc`: token read, plan lookup, vector upload b, vector upload x0, `AMGX_solver_solve`, get iters/status, vector download x, write iters/status.
- **(b)** T8 passes end-to-end: calling `amgx_setup` + `amgx_solve` from Python produces the correct 16³ solution.
- **(c)** Wall time unexpectedly high (e.g., 100 ms on 16³ when DILU takes 50 ms): likely we're running on a default stream and XLA is pipeline-stalling. Fix per Step 3.

### Step 7 — `amgx_update_coefficients` handler

- **(a)** Implement the values-update handler: token read, plan lookup, fingerprint check, `AMGX_matrix_replace_coefficients`, optional `AMGX_solver_resetup`.
- **(b)** Run: setup → solve → mutate `values` → `update_coefficients` → solve again. Second solution is different from first in a way consistent with the new matrix (hand-verify on a 5×5 test).
- **(c)** Failure to replace: AMGx may not support `replace_coefficients` for some solver types. **STOP** if resetup doesn't fix it.

### Step 8 — `amgx_release` handler + VRAM-leak test

- **(a)** Implement release. Run: 100-iter loop of setup→release; check `nvidia-smi` delta is < 10 MB.
- **(b)** Delta test passes.
- **(c)** Leak detected: walk the destruction order (solver → matrix → vectors → config). AMGx documents a specific order; respect it.

### Step 9 — T8/T9/T10 end-to-end

- **(a)** Run T8 (16³ correctness), T9 (128³ iter count), T10 (128³ wall time) end-to-end via Python wrappers.
- **(b)** T8 passes at 10⁻⁸ relative. T9 ≤ 50 iters. T10 < 14 s.
- **(c)** T9 iter count > 50: AMG config is suboptimal. Consult math expert for config tuning. This is an **engineering-blocking** signal but **not a STOP** — iterate on config JSON until acceptable.

### Step 10 — T11 HLO inspection + T12 physical

- **(a)** Run `test_t11_under_jit.py`: compile an `amgx_solve` call under `jax.jit` and inspect HLO. Run `test_t12_physical_64.py`: physical A/B/C tests at 64³.
- **(b)** HLO shows 1 custom-call, 0 host copies. Physical A max|∇·u| ≤ 1e-7. Physical B ‖u‖∞ within Phase 2 range. Physical C corr < 0.25.
- **(c)** Physical C produces corr > 0.5: **STOP #3**, interface halo. This is the F2 red line from Phase 3; Phase 4 inherits it. Stop immediately.

### Step 11 — Headline bench + report

- **(a)** Run `bench_amgx_vs_dilu.py` across grid sizes 16³, 32³, 64³, 128³. Produce `docs/benchmark/phase4_amgx_report.md` with the full table vs Phase 2 and Phase 3.
- **(b)** Report committed. T9, T10, T12, T13 results documented.
- **(c)** N/A — this is reporting, not pass/fail.

### Step 12 — Phase 4 sign-off

- **(a)** Append "Phase 4 implementation results" section to this architecture doc. User reviews.
- **(b)** User signs Phase 4 complete.
- **(c)** N/A.

---

## §10 Fail-Fast Signals (STOP Conditions)

Per master-plan rule #1. Top 6 STOP signals, ordered by likelihood × severity:

### STOP #1 — AMGx build fails against CUDA 12.4

- **Symptom**: `cmake ..` in the AMGx source tree errors out (`CUDA_ARCHITECTURES not set`, `thrust::device_vector` compile error, etc.) or `make` emits unresolved-reference errors.
- **Detection**: Step 1.
- **Escalation**: STOP. User must decide between (a) downgrading the dev box to CUDA 12.2 (the last officially-tested AMGx version), (b) waiting for an AMGx release that tests against 12.4/12.6, or (c) attempting the v2.4.0 tag as a fallback. Do not patch AMGx internals.

### STOP #2 — `AMGx setup` returns `AMGX_RC_NOT_CONVERGED` or `AMGX_RC_NAN_FOUND` on 16³ stiff

- **Symptom**: `amgx_setup` returns a non-SUCCESS status on the 16³ T7 matrix. AMGx produces NaNs in the coarsening hierarchy, or the setup diverges during smoother construction.
- **Detection**: Step 5 / T8.
- **Escalation**: STOP. Probable causes: (a) T7 stiff matrix is outside AMGx's "SPD + M-matrix" comfort zone due to the 100× density jump; (b) our config is wrong (e.g., aggressive coarsening that doesn't preserve the jump); (c) the matrix upload introduced corruption. Diagnosis: regenerate T7 in pure `numpy`, export to `AMGX_read_system_global`'s .mtx format, run AMGx's standalone `amgx_capi` example — if that also fails, the issue is AMGx + our problem class, not our code. Consult math expert.

### STOP #3 — New interface halo in physical test C at 64³

- **Symptom**: `test_t12_physical_64.py` produces a Test C residual map with `corr(|E|, |∇log ρ|) > 0.5`. The interface geometry traces through the residual.
- **Detection**: Step 10.
- **Escalation**: STOP. This is the F2 red line from Phase 3 and is non-negotiable per user directive. AMG has known issues with coefficient jumps when the interpolation operator doesn't capture the jump correctly; this is an "interpolation-weak" symptom. Remediation: try a different AMG interpolation strategy (classical with strong-threshold tuning, or extended+i-interpolation which is more jump-robust). If no config produces corr ≤ 0.25, Phase 4 has failed and we fall back to "DILU + Krylov subspace recycling" as a Phase 5 concept — Phase 4 AMG is declared unsuitable for our problem class.

### STOP #4 — Device-pointer handoff produces corrupted values

- **Symptom**: `amgx_setup` completes without error but `amgx_solve` returns a solution that differs from a CPU-reference serial solve by orders of magnitude.
- **Detection**: T8 correctness test, after §5.1's mechanism is plumbed.
- **Escalation**: STOP. Likely: AMGx's internal `cudaMemcpyDefault` is failing silently on our device pointer (e.g., because XLA's buffer is not standard-`cudaMalloc` memory but rather a BFC-allocator slab with unusual attributes). Diagnosis: add an explicit `cudaPointerGetAttributes` call in `amgx_setup.cc` before upload and log `cudaMemoryType`. If it reports `cudaMemoryTypeUnregistered`, JAX's memory is somehow not visible to AMGx's copy path — fall back to copying via our own explicit D→D kernel, bypassing AMGx's upload internals.

### STOP #5 — AMGx solve diverges

- **Symptom**: `amgx_solve` returns `AMGX_RC_NOT_CONVERGED` on a matrix where Phase 2 DILU-PCG converged fine.
- **Detection**: T9.
- **Escalation**: STOP. Either the config is wrong or the matrix is mal-conditioned in a way AMG is more sensitive to than DILU. Remediation: try the `classical_v_cycle` config (math expert's default), increase `max_iters` to 200 (should never need this but rules out transient issue), check that the matrix is genuinely SPD by computing `A - Aᵀ`'s norm on a small sample.

### STOP #6 — VRAM exceeds 128³ budget

- **Symptom**: `amgx_setup` on 128³ triggers `cudaErrorMemoryAllocation` or `nvidia-smi` shows > 3 GB allocated by the JAX+AMGx combined process.
- **Detection**: T13.
- **Escalation**: Not a hard STOP if 64³ still passes — just declare "128³ not supported on 3050 for Phase 4" and document. The user may decide to run 128³ on larger hardware in a follow-up phase. Continue with 64³ as the headline number.

### (Documented, not top-6) — `AMGX_initialize` race on `fork()`

Same as Phase 2: install `pthread_atfork` child-handler that clears the registry; defensive guard, not a failure mode we've observed.

---

## §11 Hardware Budget

Phase 3 showed 2 GB peak at 128³ Phase 3 PCG (per `phase3_scaling_64_128.md`). Phase 4 AMGx adds:

### §11.1 Per-plan AMGx memory footprint

| Component | Fine-grid cost | Hierarchy cost (sum of all coarse levels, typical) |
|---|---|---|
| A (owned CSR) | `(N+1)*4 + nnz*(4+8)` ≈ 175 MB at 128³ | ~2.5× fine (sum over all levels) ≈ 440 MB |
| Interpolation operators (P, R) | — | ~nnz_fine × 8 ≈ 120 MB total |
| Smoother state (per level) | ~O(N) | ~2×N×8 ≈ 35 MB total |
| Temporary setup workspace | — | ~3×N_fine ≈ 50 MB during setup |
| b, x (vectors) | 2×N×8 ≈ 32 MB | N/A |
| JAX input CSR + PCG workspace | Phase 3 measured ~200 MB at 128³ | N/A |

**Estimated total at 128³**: 175 + 440 + 120 + 35 + 32 + 200 = **~1 GB + JAX's own PCG workspace (~500 MB)** = **~1.5–2.0 GB sustained, ~2.5 GB peak during setup**.

This is **tighter than Phase 3** (which peaked at ~2 GB with no AMG overhead), but still inside our 3 GB budget.

### §11.2 Recommendation

- **64³ is the comfortable target.** Peak expected: ~0.5 GB. Well within budget.
- **128³ is declared "stretch".** Peak expected: 2.5 GB. Runnable on the 3050 but no margin for additional workspaces. Any second plan, any `jax.jit` scratch that allocates, or any second vector beyond (b, x, x0) risks OOM.
- **If 128³ OOMs**: switch config to aggressive-coarsening (math expert's `aggressive_coarsening.json`) which cuts hierarchy memory by ~40% at the cost of +10–20% iter count.

### §11.3 Scale beyond 128³

Not in Phase 4 scope. Requires A100/H100 hardware (80 GB VRAM, 2000 GB/s bandwidth). Move to that hardware before attempting 256³.

---

## §12 Scope Discipline

### §12.1 What is explicitly NOT in Phase 4

| Thing | Reason |
|---|---|
| Writing our own AMG from scratch | §2 Option C — rejected; 1000s of hours, no path to outperform AMGx |
| hypre / PETSc integration | Different library, different FFI surface. Out of scope. |
| Multi-GPU AMG | Single-GPU per brief. |
| Mixed-precision AMG (float32 + correction step) | Future work. Phase 4 is float64 only. |
| AMGx eigensolvers | Linear solver only. |
| Complex-valued matrices | Real double precision only. |
| Adaptive mesh refinement | Static grid assumption. |
| Deflation / recycling Krylov | Phase 5+ concept; documented only as a named fallback if AMGx fails. |

### §12.2 What success looks like

Phase 4 is a success if:
1. T8 passes: AMGx gives the same 16³ solution as Phase 2 DILU-PCG to 10⁻⁸.
2. T9 passes: AMGx at 128³ takes ≤ 50 iters (vs DILU's 186).
3. T10 passes: AMGx total wall time at 128³ < 0.5 × Phase 2 DILU-PCG wall time.
4. T11 passes: under `jax.jit`, no unexpected host copies.
5. T12 passes: no new interface halo at 64³.
6. T13 passes: VRAM delta ≤ 2 GB at 128³.

If any fail, we have a ranked recovery path (§10 STOPs). If STOP #3 fires (interface halo), Phase 4 is architecturally dead for our problem and we document it as a honest negative result — same protocol as Phase 3 had for its F2 stop signal.

### §12.3 What happens after Phase 4

Follow-on options the user may evaluate:

- **Phase 5** (hypothetical): migrate to larger hardware (A100/H100) and extend to 256³, 384³, 512³. Iter count at those scales is where AMG's grid-independence most matters.
- **Phase 5b**: Ginkgo side-by-side comparison. If AMGx has ecosystem issues (release cadence slowing, CUDA version lag), Ginkgo's mixed-precision AMG may overtake. Write a parallel `dilu/ginkgo/` wrapper.
- **Phase 5c**: multi-GPU AMGx. Requires MPI and distributed CSR partitioning — nontrivial rearchitecture.
- **Phase 6**: integrate the AMGx-preconditioned PCG into a full AM multi-physics timestep (with updates for $\rho, \mu, T$ feeding back into the matrix). This is the integration handoff point, not solver research.

---

## §13 Phase 4 Engineering Acceptance Checklist

User sign-off required on all 8 before implementation begins.

1. **AMGx ecosystem committed**: NVIDIA/AMGx `v2.5.0` (released 2024-12-21, BSD-3-Clause, CUDA 12.0–13.0 per release notes), built from source at `/opt/amgx-2.5.0/`. pyamgx explicitly not used (unmaintained; issue #40 open since May 2025). Verification: §1 + Step 1 of §9.

2. **Integration strategy committed**: **Option B** — thin C++ FFI wrapper around AMGx's C API, mirroring Phase 2's `cusparse_dilu_*.cc` pattern. No pyamgx, no Cython, no pybind11. New library `libdilu_amgx.so` under `dilu/amgx/`, lexically separate from Phases 1/2/3.

3. **Four FFI primitives agreed**: `amgx_setup` (returns `uint64` token), `amgx_update_coefficients` (values-only refresh via `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup`), `amgx_solve` (hot path, one call to `AMGX_solver_solve`), `amgx_release`. Process-global plan registry keyed by token; fingerprint `(N, nnz)` + `setup_converged` flag (§6).

4. **Zero-PCIe-copy architecture confirmed**: AMGx owns its own device buffers for matrix + vectors; our uploads pass JAX device pointers and AMGx internally performs D→D copies (via `cudaMemcpyDefault`, no H→D traffic). Per-solve D→D traffic ≈ 32 MB at 128³, ≪1% of solve time. The 8-byte D→H token copy per solve is the sole tolerated host-side exception, same as Phase 2 (§5).

5. **Handle lifecycle committed**: `AMGX_initialize` once per process (`std::once_flag`), never finalized (process-lifetime leak, same as Phase 2's cuSPARSE handle). `AMGX_resources_handle` is a device-level singleton, one-per-device. Thread-id tripwire enforces single-thread access pattern (§4.5). `pthread_atfork` child-handler for fork-safety (defensive).

6. **Testing & physical-benchmark contract committed**: T8/T9/T10/T11/T12/T13 per §8. Non-negotiable: T12's Test C must have `corr(|E|, |∇log ρ|) ≤ 0.25` at iter count 5–15. F2 STOP signal (corr > 0.5) — immediate halt with honest write-up, no rescue attempts.

7. **Hardware budget honored**: 64³ is the confident target (peak ≤ 0.5 GB VRAM). 128³ is "stretch" (peak 2.5 GB, tight on 3050). Aggressive-coarsening config (`configs/aggressive_coarsening.json`) is the memory-lean fallback for 128³. Any grid > 128³ is out of Phase 4 scope.

8. **Top 6 STOP signals committed** (§10): AMGx build-fail against CUDA 12.4, setup `NOT_CONVERGED` on 16³ stiff, new interface halo in Test C at 64³, device-pointer corruption, solve divergence on converged-by-DILU problem, VRAM over 128³ budget. When any fires, implementer writes a 1-page diagnostic and escalates to user — no heroic debugging.

When all 8 are approved, implementation proceeds step-by-step per §9, with the user free to halt between any two steps. Phase 4 implementation is self-contained within `dilu/amgx/`; Phase 1/2/3 artifacts are **not** modified. The final sign-off (Step 12) produces a `docs/benchmark/phase4_amgx_report.md` that the user reads to decide whether Phase 5 (scale-up or algorithmic follow-on) is justified.

**End of Phase 4 design document.**
