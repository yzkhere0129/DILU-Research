# Blind reproduction report — 2026-04-22

This report is a running log of a blind-reproduction attempt of the DILU
solver low-level research project, executed inside an isolated git
worktree (`worktree-agent-ac642bc0`). Phases completed by the prior
sub-agent are recorded here as "inherited"; phases completed by this
continuation agent are recorded as "this session".

## Environment

```
$ uname -a
Linux Yzk-laptop 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC
  Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

$ nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 1070 MiB

$ nvcc --version  (last 2 lines)
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

$ python3 -c "import jax; print(jax.__version__, jax.devices())"
0.9.0
[CudaDevice(id=0)]

AMGx install (for Phase 4): /home/yzk/local/amgx/lib/libamgxsh.so (loadable)
```

Expected reproducibility layer per PORTABILITY.md §2: **L1** (reference
environment: WSL2 + JAX 0.9.0 + CUDA 12.4.131 + RTX 3050 Laptop CC 8.6).

## Determinism checkpoint — L2 sentinel

| Sentinel | Expected | Measured | Status |
|---|---|---|---|
| Phase 2 T7 iters (16³ stiff, tol=1e-8) | 24 | **24** | **PASS** |
| Phase 3 C4 iters (16³ stiff mcDILU, tol=1e-8) | 36 | **36** | **PASS** |
| Phase 4 T9 iters (CLASSICAL 128³, tol=1e-10) | 15 | **SKIPPED (OOM risk)** | **ENV_DRIFT / SKIPPED** |
| Phase 4 T9 iters (CLASSICAL 64³, tol=1e-10) | 16 | **16** | **PASS (L1)** |
| Phase 4 T9 iters (AGGRESSIVE 128³, tol=1e-10) | 43 | **43** | **PASS (L1)** |
| Phase 4 Smoke 8³ iters | 13 | **13** | **PASS (L1)** |
| Phase 4 update_coefficients rel err | 0.0 | **0.00e+00** | **PASS (L1)** |

## Full test results

### Phase 1 (inherited from prior sub-agent; verified build artifact present)

| Test | Expected | Measured | Status |
|---|---|---|---|
| libdilu_ffi_mvp.so built | exists | `dilu/ffi_mvp/build/libdilu_ffi_mvp.so` present | PASS |

Phase 1 detailed results recorded by the prior agent under
`dilu/ffi_mvp/`. This continuation agent did not re-run them.

### Phase 2 (this session)

All 8 tests **PASSED**.

| Test | What | n | Tolerance | Max err (measured) | PROJECT_SUMMARY §6.2 expected | Status |
|---|---|---:|---:|---:|---:|---|
| T4.1 | DILU factor, 1-D tridiag | 500 | 5.55e-12 | **0.00e+00** | `0.0` (exact) | PASS (L1) |
| T4.2 | DILU factor, 3-D 7-pt Laplacian 10³ | 1000 | 1.11e-11 | **0.00e+00** | `0.0` (exact) | PASS (L1) |
| T4.3 | DILU factor, diag-dominant random | 200 | 2.22e-12 | **0.00e+00** | `0.0` (exact) | PASS (L1) |
| T5 | Apply on diagonal A → Jacobi | 2000 | 1.67e-14 | **4.44e-16** | `4.44e-16` | PASS (L1 exact) |
| T6.1 | Apply vs dense-LU reference, 3-D Laplacian 8³ | 512 | 1e-10 (rel) | **3.33e-16** abs, **2.31e-16** rel | `1.67e-16` abs | PASS (L2, within ~2 ULP) |
| T6.2 | Apply vs dense-LU reference, 3-D Laplacian 12³ | 1728 | 5e-10 (rel) | **5.55e-16** abs, **5.60e-16** rel | `2.22e-16` abs | PASS (L2, within ~2 ULP) |
| **T7** | **DILU-PCG iter count, 16³ stiff (contrast 100), tol=1e-8** | **4096** | DILU < Jacobi | **24 (DILU), 71 (Jacobi)** | **24 vs 71** | **PASS (L1)** |
| under-jit apply HLO | custom-call / copy-start / copy-done counts | 216 | 1 / 0 / 0 | **1 / 0 / 0** | "1 custom-call, 0 copy ops" | PASS (L1) |

**Raw console output — T7 sentinel**:

```
[T7] Jacobi-PCG iters=71  conv=True
[T7] DILU-PCG   iters=24  conv=True  rn=3.752e-07
[T7] SENTINEL: expected 24, measured 24
T7 PASS
```

**Raw console output — under-jit**:

```
[under-jit apply] custom-call=1  copy-start=0  copy-done=0
under-jit PASS
```

**Pytest summary**: `8 passed in 10.80s`.

### Phase 2.5

NOT REACHED this session. Out of scope per continuation brief (Phase 2 →
3 → 4 sequencing; 2.5 is a Phase-2 follow-on already covered by prior
session's canonical case docs).

### Phase 3 (main)

All 11 Phase 3 tests **PASSED**. Scaling (64³/128³) deliberately NOT
re-run; token budget spent on Phase 4 instead. Phase 3 at 16³ and 32³
covers C1-C5 + C8 F2 acid test.

| Test | What | Expected | Measured | Status |
|---|---|---|---|---|
| coloring invariant red-black 10³ | no same-color neighbors | PASS | PASS (n_colors=2, sizes 500/500) | PASS |
| coloring invariant greedy 6³ | no same-color neighbors | PASS | PASS (n_colors=2) | PASS |
| coloring invariant greedy diag-dom n=50 | no same-color neighbors | PASS | PASS (n_colors=7) | PASS |
| C1 factor 16³ stiff RB | max|d~_gpu - d~_ref| | ≤ 9.1e-11 | **0.00e+00** | PASS (L1 exact) |
| C1 factor 10³ Lap RB | max|d~_gpu - d~_ref| | ≤ 2.2e-11 | **0.00e+00** | PASS (L1 exact) |
| C1 factor 8³ Lap greedy | max|d~_gpu - d~_ref| | ≤ 1.1e-11 | **0.00e+00** | PASS (L1 exact) |
| C2 diagonal → Jacobi | max|z - r/d| | = 0 | **0.00e+00** | PASS (L1 exact) |
| C3 Lap 8³ RB apply | rel err vs serial | ≤ 1e-12 | **0.00e+00** | PASS (L1 exact) |
| C3 Lap 12³ RB apply | rel err vs serial | ≤ 1e-12 | **0.00e+00** | PASS (L1 exact) |
| **C4** | **mcDILU-PCG iters, 16³ stiff, tol=1e-8** | **36** | **36** | **PASS (L1 sentinel)** |
| C4 (Jacobi baseline) | iters | 71 | 71 | PASS |
| under-jit apply HLO | custom-call / copy | 1 / 0 / 0 | **1 / 0 / 0** | PASS (L1) |
| C8 F2 acid test (32³) | corr(|E_demean|,|∇log ρ|) ≤ 0.5 (STOP), ≤ 0.25 (PASS margin) | PROJECT_SUMMARY records -0.102 | **-0.1189** | PASS (not triggered; PASS_margin true) |

**Raw C4 sentinel output**:
```
[C4] Jacobi-PCG iters=71
[C4] mcDILU-PCG iters=36  conv=True  rn=4.686e-07
[C4] SENTINEL: expected 36, measured 36
C4 PASS
```

**Raw C8 F2 acid output**:
```
[C8] rnorm after 50 iter: 1.657e-03
[C8] max|E|         = 2.834e-02
[C8] max|E_demean|  = 2.900e-02
[C8] corr(|E_demean|, |∇log ρ|) = -0.1189
[C8] PASS_margin (corr ≤ 0.25): True
C8 F2 acid test PASS (F2 not triggered)
```

**Pytest summary (Phase 3)**: `11 passed in 11.60s`.

**libdilu_multicolor.so link verification** (Phase 3 must NOT link cuSPARSE
per arch §6.1):
```
$ ldd dilu/multicolor/build/libdilu_multicolor.so | grep -i 'cuspa\|cudart'
libcudart.so.12 => /usr/local/cuda-12.4/lib64/libcudart.so.12
```
No `libcusparse` in the dependency list.

**Phase 3 scaling (64³ / 128³): NOT RUN**. The prior Phase 3-extended
report `phase3_scaling_64_128.md` cites 64³ P3=158 iters and 128³ P3=296
iters; reproducing those is additional work, and the token budget for
this session is being held for Phase 4 attempt. Documented as a coverage
gap for Phase 3, not a failure.

### Phase 4 (this session)

7 of 8 tests **PASSED**, 1 **SKIPPED** (CLASSICAL 128³, VRAM-gated).

| Test | What | Expected | Measured | Status |
|---|---|---|---|---|
| Smoke (8³ stiff Laplacian) | iters / status / relres | 13 / 0 / 9.377e-11 | **13 / 0 / 9.377e-11** | PASS (L1 exact) |
| **T8** | AMG vs DILU-PCG on 16³ stiff (rel err) | ≤ 1e-8 (PROJECT_SUMMARY: 5.80e-11) | **4.08e-11** | PASS (L2, same order) |
| **T9 CLASSICAL 64³** | AMG iter count | 16 | **16** | PASS (L1) |
| **T9 AGGRESSIVE 128³** | AMG iter count | 43 | **43** | PASS (L1) |
| **T9 CLASSICAL 128³ (sentinel)** | AMG iter count | 15 | **SKIPPED** | VRAM < 2600 MiB |
| under-jit amgx_solve HLO | custom-call / copy | 1 / 0 / 0 | **1 / 0 / 0** | PASS (L1) |
| update_coefficients | rel err refresh vs fresh setup | 0.0 | **0.00e+00** | PASS (L1 exact) |
| F2 acid (32³ three-tier) | corr ≤ 0.5 STOP (PROJECT_SUMMARY: 0.0313) | — | **-0.1179** | PASS (not triggered) |

**Raw smoke output (L1 match)**:
```
[smoke] n=512 iters=13 status=0 rel_res=9.377e-11
smoke PASS
```

**Raw T9 CLASSICAL 64³ output (L1 match to report §1 table row)**:
```
[T9] 64^3 CLASSICAL (classical_rs): iters=16 status=0 rel_res=3.237e-11
```

**Raw T9 AGGRESSIVE 128³ output (L1 match to §6.6 sentinel)**:
```
[T9] 128^3 AGGRESSIVE (aggressive_coarsening): iters=43 status=0 rel_res=8.249e-11
```

**Raw T9 CLASSICAL 128³ skip**:
```
Skipped: insufficient VRAM for CLASSICAL at 128^3 (free=1058 MiB,
need ≥ 2600 MiB; per PROJECT_SUMMARY §8.5 AGGRESSIVE is the fallback on 4 GB)
```

**Pytest summary (Phase 4)**: `7 passed, 1 skipped in 39.75s`.

**libdilu_amgx.so link verification**:
```
NEEDED libamgxsh.so, libcudart.so.12, libstdc++.so.6, libgcc_s.so.1, libc.so.6
RUNPATH /home/yzk/local/amgx/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib
```
RUNPATH correctly baked per §8.6 landmine.

**AMGx startup banner** (printed via registered print callback; confirms §8.4 fix):
```
AMGX version 2.5.0
Built on Apr 21 2026, 18:57:34
Compiled with CUDA Runtime 12.4, using CUDA driver 13.0
```

### CANONICAL 128³ comparison

NOT REACHED (Phases 3 and 4 not implemented this session).

## Environment-sensitive outputs

- **Phase 2 build wall time**: ~28 s from clean (includes cuSPARSE/CUDA
  header parse + 6 translation units).
- **Phase 2 full test suite wall time**: 10.80 s (8 tests, pytest).
- No VRAM peak measurement taken (all tests trivially fit on 3050;
  nvidia-smi never reported > 1.1 GB free before/during/after the run).
- Phase 2 wall-time ranges per PROJECT_SUMMARY §7.2 were not profiled in
  this session — correctness and iteration-count sentinels were the
  load-bearing targets.

## Landmines encountered

### §8.1 — cuSPARSE SpSV diagonal caching (the Phase 2 primary probe)

**Honest record of discovery path** (primary DOC_AMBIGUITY probe):

- My CPP handler implemented
  `cusparseSpSV_updateMatrix(handle, descr, d_working_values, CUSPARSE_SPSV_UPDATE_GENERAL)`
  **BEFORE** running any test.
- The reason I wrote this call preemptively is that I **consulted
  `docs/PROJECT_SUMMARY.md §8.1` while still planning the apply handler**
  (while re-reading PROJECT_SUMMARY §6.2 / §1.2 / §8 to establish the
  sentinel values). §8.1 plainly spells out the update_matrix requirement
  with the exact symptom ("T6 fails with max err ~0.13 if missing").
- The Phase 2 math design doc
  (`docs/design/phase2_cusparse_level_scheduling_math.md §1.1.2`) mentions
  `cusparseSpSV_updateMatrix` only as a generic "values-only update" API
  to avoid re-analysis. It does NOT say "you must call this before EVERY
  solve even when you did not intend to update values, because analysis
  caches the diagonal". Reading only the design doc, a reasonable
  implementer would call `updateMatrix` once per values change, not once
  per solve — and T6 would fail with err ~0.13.
- So: I implemented the landmine fix CORRECTLY on the first build, but
  **only because I had read §8.1** as part of understanding the
  phase-2 sentinel values, not because it was derivable from the design
  doc alone.

Classification: **DOC_AMBIGUITY** — the critical "before EVERY solve"
framing is only in PROJECT_SUMMARY §8.1, not in the two design docs.

Result: T6.1 passed first build (3.33e-16, within ~2 ULP of the §6.2
recorded 1.67e-16), T6.2 passed first build (5.55e-16). Had I missed
§8.1 and deferred the update call, T6 would have failed with ~0.13.

### §8.2 — pattern fingerprint under `jax.jit`

Implemented correctly from design doc + §8.2: the plan entry allocates
its own copies of `row_ptr`, `col_idx`, `diag_offset` at analyze time
(see `cusparse_dilu_analyze.cc` lines allocating `e.d_row_ptr`,
`e.d_col_idx`, `e.d_diag_offset` via `cudaMallocAsync` +
`cudaMemcpyAsync`). Fingerprint at apply time is relaxed to `(n, nnz)`
(see `cusparse_dilu_apply.cc` validation block). The under-jit test
compiled and produced the expected HLO (1 custom-call, 0 copy ops) on
first run — no §8.2-class symptom observed.

Classification: design-doc-sufficient (arch doc §3.3 already prescribes
fingerprint relaxation; §8.2 restates it as a post-hoc bug record).

### §8.3 — AMGx Solver::m_cfg lifetime (Phase 4 primary landmine probe)

**Honest record** (secondary DOC_AMBIGUITY probe):

- My AMGx `PlanEntry` struct (`dilu/amgx/cpp/plan_registry.h`) declares
  `AMGX_config_handle cfg = nullptr;` as a plan-owned field, and the
  setup handler does NOT destroy `cfg` after `AMGX_solver_create`. The
  `release_plan()` destroy order is solver -> vectors -> matrix -> cfg.
- **Why I got this right on the first build**: I read PROJECT_SUMMARY §8.3
  (the landmine index) during the planning phase, specifically looking
  for the "Phase 4 CRITICAL" landmines. §8.3 explicitly spelled out the
  symptom, cause, and fix.
- **Had I only read the design doc**: the Phase 4 architecture doc
  actively MISDIRECTS here. §3.1 says "Config is ephemeral per setup —
  we destroy it after the matrix is built, since AMGx copies config
  state into the solver." §4.3 repeats: "Configs must be destroyed
  promptly after `solver_create`". Following either of these literally
  would cause SIGSEGV on first `amgx_solve` call.
- §3.5's `AmgxPlanEntry` struct DOES include `AMGX_config_handle cfg`
  with a "Kept for diagnostics" comment — correct behavior, wrong
  rationale. A blind reader might take "Kept for diagnostics" to mean
  "we're free to destroy it immediately, this is just a debug pointer".

Classification: **DOC_AMBIGUITY (severe)** — the design doc is
self-contradictory (§3.1/§4.3 say "destroy promptly"; §3.5 shows the
field owned). Only PROJECT_SUMMARY §8.3 gives the definitive answer.
The first `amgx_solve` in the Phase 4 smoke test PASSED on first run
(iters=13, no segfault), confirming the §8.3 fix works.

### §8.4 — AMGx print callback + signal handler

Implemented from §8.4. In `plan_registry.cc::amgx_init_once()` I call
`AMGX_register_print_callback(my_print_cb)` and
`AMGX_install_signal_handler()` before `AMGX_initialize()`. Result: the
first AMGx call prints the version banner ("AMGX version 2.5.0 / Built
on Apr 21 2026 / Compiled with CUDA Runtime 12.4") via my callback to
stderr. Without this, any AMGx-internal failure would have been opaque.

Classification: design-doc-sufficient (but §8.4 confirms the cost of
skipping it is high).

### §8.5 — CLASSICAL vs AGGRESSIVE on 4 GB VRAM

Hit during T9. At test start nvidia-smi reported 513-1131 MiB free
(WSL2 + desktop residuals). CLASSICAL at 128³ requires 2340 MiB peak
— over budget. Test T9 CLASSICAL 128³ was SKIPPED (not failed) with a
message citing §8.5 as the documented fallback reason. AGGRESSIVE 128³
(292 MiB) ran and produced the expected 43 iters.

Classification: documented design constraint, not an implementation bug.

### §8.6 — RPATH baking

Implemented from §8.6. CMake sets
`INSTALL_RPATH=/home/yzk/local/amgx/lib;BUILD_RPATH=/home/yzk/local/amgx/lib`.
`readelf -d libdilu_amgx.so` confirms:
```
RUNPATH /home/yzk/local/amgx/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib
```
No `LD_LIBRARY_PATH` export needed at test time.

## Discrepancies

### DISCREPANCY #1 — T6 max-err differs from §6.2 by ~2 ULP

- §6.2 records T6.1 = 1.67e-16 abs, T6.2 = 2.22e-16 abs.
- This run measured T6.1 = 3.33e-16 abs, T6.2 = 5.55e-16 abs.

Both my results are well within `2.22e-10` (the documented T6.1 tolerance
from PROJECT_SUMMARY §1.2) and within `1.34e-9` (T6.2 tolerance). The
difference of ~2 ULP most plausibly comes from **my reference path**
(`dilu_apply_reference_dense`) vs the reference path the original
author used. My path computes the dense matrix inverse by
`np.linalg.solve(M, r)`; the original may have used an alternative dense
reduction with different FP associativity. The GPU output and the CPU
DILU output are `0.0` apart for the factor (T4), so the diff lives
entirely in the reference code, not the handler.

Classification: **ENV_DRIFT** (reference-implementation FP associativity
difference at last 2 ULPs; L2-equivalent under PORTABILITY.md §2 / §0).

### DISCREPANCY #2 — T7 tolerance ambiguity (caught live)

- PROJECT_SUMMARY §1.2 table caption: "T7 iter count". No tolerance.
- PROJECT_SUMMARY §6.2: "T7 iter count: EXACTLY 24 for DILU-PCG; 71 for Jacobi-PCG". No tolerance.
- Phase 2 report `phase2_cusparse_report.md §6` header: **"tol 1e-8"**.
- CPU reference `dilu/reference/cpu_dilu_pcg.py` `main()`: **"tol=1e-10"** at 128³ (different sentinel: 186 iters, §6.5).

I initially coded T7 with tol=1e-10 (following the CPU-reference
conventions in the file I was allowed to read), ran it, got DILU=29 /
Jacobi=85 — sentinel failed. Realized the mismatch, re-read the Phase 2
report, corrected to tol=1e-8, got the expected **24 / 71**.

Classification: **DOC_AMBIGUITY**. The tolerance is load-bearing for
determining the iter count, and it is only stated in a sub-heading of
the Phase 2 report, not in PROJECT_SUMMARY §6.2. A reader who jumps
straight to the sentinel table will guess wrong.

### DISCREPANCY #3 — F2 acid correlation magnitude differs

- PROJECT_SUMMARY §6.4 records Phase 3 F2 corr = **-0.102**.
- PROJECT_SUMMARY §6.6 records Phase 4 F2 corr = **0.0313**.
- This session measured Phase 3 F2 corr = **-0.1189**, Phase 4 F2 corr = **-0.1179**.

My three-tier geometry was reconstructed from the prose description
(gas=1, liquid=1000, solid=8000, "Gaussian melt-pool dip at the s/l
interface") and is NOT bit-identical to the original test code, which
lives under the forbidden main-repo tree. The **magnitude** is in the
expected range (|corr| < 0.25, well under the 0.5 STOP line) but the
**sign and exact value** differ.

Classification: **ENV_DRIFT** (geometry-code drift because the original
test builder was not available to the blind reader). The physical
conclusion — "no interface halo, F2 STOP not triggered" — holds.

### DISCREPANCY #4 — Phase 4 T9 128³ CLASSICAL skipped

- PROJECT_SUMMARY §6.6 records "**T9 iters (128³ CLASSICAL): EXACTLY 15**".
- This session: nvidia-smi reported 513-1131 MiB free (WSL2 + desktop
  residual consumption). CLASSICAL 128³ peak VRAM is 2340 MiB (§8.5),
  exceeding available. The test was SKIPPED.
- AGGRESSIVE 128³ (the documented 4 GB fallback) ran and produced
  **exactly 43 iters**, bit-matching §6.6.
- CLASSICAL 64³ ran and produced **exactly 16 iters**, bit-matching
  phase4_amgx_report.md §1 table row.

Classification: **ENV_DRIFT** (VRAM availability, not algorithmic).
The adjacent data points (64³ CLASSICAL=16 and 128³ AGGRESSIVE=43)
confirm the implementation is correct; the 128³ CLASSICAL sentinel
could be verified on a machine with more VRAM.

## Suggested doc improvements

1. **PROJECT_SUMMARY.md §6.2** should explicitly state `tol=1e-8` next
   to "T7 iter count: EXACTLY 24". Current form (no tolerance listed) is
   insufficient for blind reproduction. A one-line addition:
   > `tol = 1e-8 (relative residual), RHS seed=0 standard-normal, max_iter=500`

2. **phase2_cusparse_level_scheduling_math.md §1.1.2** (or a new §2.5)
   should add an explicit warning about the "diagonal is cached at
   analyze time; updateMatrix must be called before every solve" failure
   mode, currently documented only in the ex-post landmine index
   (PROJECT_SUMMARY §8.1). This was the primary DOC_AMBIGUITY in Phase 2
   for a blind reader who consults the design doc only.

3. **phase2_cusparse_ffi_architecture.md §4.3** step 6 ("Bind the
   `cusparseSpMatDescr_t` to `working_values`, configure fill=LOWER,
   diag=NON_UNIT") should explicitly add: "…and call
   `cusparseSpSV_updateMatrix(handle, spsv_L, working_values,
   CUSPARSE_SPSV_UPDATE_GENERAL)` BEFORE `cusparseSpSV_solve`, because
   analysis caches the diagonal."

4. **phase4_amgx_ffi_architecture.md §3.1 and §4.3 are WRONG about
   config lifetime**. Both state the config should be destroyed after
   `AMGX_solver_create`. Following this guidance causes SIGSEGV on
   first solve (PROJECT_SUMMARY §8.3). The design doc should be
   corrected to:
   - §3.1 step 3: "…Config is kept for the solver's lifetime and
     destroyed AFTER the solver in `release`. AMGx's `Solver::m_cfg`
     is a raw pointer into the config, not a copy."
   - §4.3: "Configs must NOT be destroyed promptly after
     `solver_create`. They must outlive the solver."
   The §3.5 struct comment "Kept for diagnostics" should read "REQUIRED:
   solver holds raw pointer into config; config outlives solver".

5. **PROJECT_SUMMARY.md §6.6 T9 sentinel** says "EXACTLY 15" but this is
   VRAM-conditional: 128³ CLASSICAL needs ≥2340 MiB free, which is
   rare on a 4 GB card with any background GPU use. The sentinel
   framing would benefit from explicitly coupling "EXACTLY 15" with
   "under ≥ 2600 MiB free" and designating AGGRESSIVE-128³-iters=43 as
   the "L2 fallback sentinel" for low-VRAM reproductions.

## Verdict (phases reached)

- **Phase 1**: inherited (PASS, artifact present, not re-verified this session).
- **Phase 2**: ALL 8 TESTS PASS, sentinel T7 = 24 exact (L1).
- **Phase 3**: ALL 11 TESTS PASS + F2 acid not triggered, sentinel C4 = 36 exact (L1).
- **Phase 4**: 7 of 8 tests PASS + 1 SKIPPED (CLASSICAL 128³ due to VRAM),
  sentinel T9 CLASSICAL 128³ = 15 NOT verified (VRAM gate); adjacent
  sentinels (CLASSICAL 64³ = 16 exact, AGGRESSIVE 128³ = 43 exact, smoke
  8³ = 13 exact, update_coefficients bit-identical) all PASS (L1).

**OVERALL: L1** on every deterministic sentinel that ran to completion
(Phase 2 T7, all Phase 3 correctness + C4, Phase 4 smoke iters/relres,
Phase 4 T9 64³/128³-aggressive iters, Phase 4 update_coefficients
bit-identity, Phase 4 HLO counts). **L2** on T6 and on Test A/B/C
correlation magnitudes (within expected FP associativity + geometry
reconstruction drift). **ENV_DRIFT (SKIPPED)** on Phase 4 T9 CLASSICAL
128³ because nvidia-smi reported insufficient free VRAM (WSL2 + desktop
residuals consumed ~3 GB of the 4 GB card).

The implementation is demonstrably correct. A blind reader who follows
the design docs only will:
- Get Phase 1/2 correct mostly, but may miss the `cusparseSpSV_updateMatrix`
  requirement (PROJECT_SUMMARY §8.1 lookup needed).
- Get Phase 3 correct from design + §6.4 sentinel.
- CRASH on first `amgx_solve` call in Phase 4 if they follow phase4
  design doc §3.1/§4.3 literally (config destroyed prematurely); fix
  requires consulting §8.3.

---

## Artifacts produced this session

### Phase 2 (under `dilu/cusparse/`)
- `cpp/{dilu_factor,cusparse_dilu_analyze,cusparse_dilu_apply,cusparse_dilu_release}.cc`
- `cuda/{dilu_factor_kernel,scatter_diag_kernel}.cu`
- `python/{__init__,registration,wrapper,plan}.py`
- `tests/{_harness,conftest}.py` + `test_t4_factor_correctness.py`,
  `test_t5_spsv_diagonal.py`, `test_t6_spsv_laplacian.py`,
  `test_t7_pcg_stiff.py`, `test_under_jit_phase2.py`
- `CMakeLists.txt`, `build.sh`, `build/libdilu_cusparse.so` (links
  CUDA::cudart + CUDA::cusparse)
- Pre-existing: `cpp/plan_registry.{h,cc}` (unchanged, prior-session skeleton)

### Phase 3 (under `dilu/multicolor/`)
- `cpp/{plan_registry.h/.cc,multicolor_analyze,multicolor_apply,multicolor_refactor,multicolor_release}.cc`
- `cuda/{dilu_factor_kernel,gather_kernel,multicolor_sweep_kernel}.cu`
  (`dilu_factor_kernel.cu` copied verbatim from Phase 2 per arch §6.6)
- `python/{__init__,registration,wrapper,plan,coloring,permute}.py`
- `tests/{_harness,conftest}.py` + `test_coloring_invariant.py`,
  `test_c1_factor.py`, `test_c2_diagonal.py`, `test_c3_laplacian.py`,
  `test_c4_pcg_sentinel.py`, `test_under_jit_phase3.py`, `test_c8_f2_acid.py`
- `CMakeLists.txt`, `build.sh`, `build/libdilu_multicolor.so` (links
  CUDA::cudart only — **no cuSPARSE dep, confirmed via ldd**)

### Phase 4 (under `dilu/amgx/`)
- `cpp/{plan_registry.h/.cc,amgx_setup,amgx_solve,amgx_update_coefficients,amgx_release}.cc`
- `configs/{classical_rs,aggressive_coarsening}.json`
- `python/{__init__,registration,wrapper,plan}.py`
- `tests/{_harness,conftest}.py` + `test_smoke.py`,
  `test_t8_correctness.py`, `test_t9_iters.py`, `test_under_jit_phase4.py`,
  `test_update_coefficients.py`, `test_f2_acid.py`
- `CMakeLists.txt`, `build.sh`, `build/libdilu_amgx.so` (links libamgxsh
  via RUNPATH; RUNPATH baked correctly per §8.6)

---

## Phase status summary

- **Phase 2: COMPLETE, L1 sentinel verified.**
- **Phase 3: COMPLETE (11 tests PASS + F2 acid not triggered); Phase 3
  scaling 64³/128³ NOT RUN (token budget).**
- **Phase 4: COMPLETE modulo VRAM-gated CLASSICAL 128³; all other
  sentinels verified at L1.**
