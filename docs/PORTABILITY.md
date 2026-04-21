# Portability & Cross-Environment Compatibility Guide

**Scope**: this document answers "my JAX / CUDA / AMGx / GPU / OS is not
exactly the reference environment — will the code still run, and what do I
need to change?". It complements `docs/PROJECT_SUMMARY.md` §4 (reference
environment) and §6–§7 (deterministic / environment-sensitive outputs).

**Rule of thumb**: the code is MORE portable than the "bit-identical"
language in PROJECT_SUMMARY suggests. Most acceptance criteria are
**numerically equivalent** across a reasonably wide version window, even
when wall times and VRAM absolutes differ. This file defines that window
precisely.

---

## 0. The three reproducibility layers

Every output produced by this project sits in exactly one of three layers.
Which layer a given measurement belongs to determines how strict the
environment must be.

| Layer | What it guarantees | Environment requirement | Examples from this project |
|-------|--------------------|-------------------------|----------------------------|
| **L1 — Bit-identical** | Same IEEE-754 bytes down to the last ULP | JAX, jaxlib, CUDA, AMGx, compiler, GPU arch all match the reference | `max_err = 2.220446049250313e-16` (T1); HLO byte identity |
| **L2 — Numerically equivalent** | Same discrete decisions (iter counts, branch choices), results within one or two ULPs | Same algorithm generation; minor version drift OK | **T7 = 24 iters**, **T9 = 15 iters**, **F2 corr = 0.0313**, physical A/B thresholds |
| **L3 — Behaviorally equivalent** | Same logical behavior and physical correctness; numbers differ in absolute scale | Compatible API generation; GPU arch may differ | Test A `max\|∇·u\| ≤ 4.1e-8` (value varies); Test B CSF parasitic pattern appears at expected order of magnitude |

**When in doubt, L2 is what matters for scientific correctness.** L1
matters only when you are debugging a numerical diff at ULP precision or
running a byte-level CI gate.

---

## 1. Reference environment (recap)

Authoritative spec: `docs/PROJECT_SUMMARY.md` §4. One-line summary:

> WSL2 on Win11 / Python 3.12 / JAX 0.9.0 / jaxlib 0.9.0 (cuda12) /
> CUDA 12.4.131 / driver 580.97 / gcc 13 / RTX 3050 Laptop (CC 8.6) /
> AMGx v2.5.0 built with `-DCUDA_ARCH=86 -DAMGX_NO_MPI=ON`.

All L1 claims are valid only on this combination. L2 / L3 claims extend
across the version windows in §2.

---

## 2. Tested and expected-compatible version windows

Version windows below are stated per-component and assume the OTHER
components match the reference. Interactions between simultaneous
version shifts are combinatorial and have not been exhaustively tested;
when in doubt run the environment sentinel in §6.

### 2.1 JAX / jaxlib

| Version | Layer | Status | Notes |
|---------|-------|--------|-------|
| 0.9.0 | L1 | **Tested, reference** | The baseline |
| 0.9.1 | L2 | Tested (Phase 1 repro on GTX 1080 / RTX 5060) | L1 might drift by a few ULPs on some HLO graphs; iter counts and correlation values match |
| 0.9.2 – 0.9.x | L2 | Expected-to-work | Same typed-FFI API, no API surface changes in patches |
| 0.10.x | L3 | **Untested, likely needs code change** | `jax.ffi` API signatures can shift; `ffi::Attr<T>` decoded-type contract has been unstable (see `PROJECT_SUMMARY.md` §8.3.1). Run §6 sentinel, expect 1-2 compile errors |
| 0.8.x and older | — | **Not supported** | `jax.ffi` typed API did not exist; legacy `jax.extend.ffi` was renamed-then-replaced |
| jaxlib ≠ jax version | — | **Broken** | Major-minor mismatch crashes at import |

### 2.2 CUDA Toolkit

| Version | Layer | Status | Notes |
|---------|-------|--------|-------|
| 12.4.131 | L1 | **Tested, reference** | The baseline |
| 12.0 – 12.3 | L2 | Expected-to-work | cuSPARSE generic SpSV API stable since 12.0; `cusparseSpSV_updateMatrix` available |
| 12.5 – 12.9 | L2 | Expected-to-work (Phase 1 repro tested on 12.9) | No API surface shifts; cuSPARSE probed on 12.9 still returns 12.3.x runtime |
| 13.0 – 13.x | L2 | Expected-to-work, AMGx rebuild required | Legacy `csrsv2` / `csrilu02` removed in 13.0; **we don't use them**. Generic SpSV intact. But AMGx may need rebuild for CUDA 13 SDK; check NVIDIA/AMGX GitHub for 13.x compat status |
| 11.x and older | — | **Not supported** | cuSPARSE generic API less mature; not tested |

### 2.3 AMGx

| Version | Layer | Status | Notes |
|---------|-------|--------|-------|
| v2.5.0 | L1 | **Tested, reference** | The baseline |
| v2.4.x | L2 | Expected-to-work | `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup` both present; same C API surface |
| v2.3.x and older | — | Untested | May lack `resetup` path; Phase 4 K-amortization may regress |
| v2.6.x (future) | L3 | **Untested, risk of breakage** | `resetup` is already marked deprecated in 2.5. If 2.6 removes it, Phase 4 must fall back to `AMGX_solver_setup` per timestep — functional but loses ~50% of the AM-loop win |
| pyamgx | — | **Not supported** | We explicitly chose C++ FFI over pyamgx (project-sign-off on 2026-04-20) |

### 2.4 GPU compute capability

| Architecture | CC | Status | Rebuild command |
|--------------|----|---------|-----------------|
| Ampere consumer (RTX 30-series) | 8.6 | **Tested, reference** | `CUDA_ARCH=86 bash dilu/*/build.sh` |
| Ampere datacenter (A100) | 8.0 | Expected-to-work | `CUDA_ARCH=80` in build.sh + AMGx rebuild |
| Ada consumer (RTX 40-series) | 8.9 | Expected-to-work | `CUDA_ARCH=89` |
| Hopper datacenter (H100) | 9.0 | Expected-to-work | `CUDA_ARCH=90` + AMGx rebuild |
| Blackwell datacenter (B100/B200) | 10.0 | Untested | `CUDA_ARCH=100`; requires AMGx v2.5+ (claims Blackwell support) |
| Blackwell consumer (RTX 50-series) | 12.0 | Phase 1 FFI tested on RTX 5060 | `CUDA_ARCH=120`; requires CUDA 12.8+ |
| Volta (V100) | 7.0 | Untested | `CUDA_ARCH=70`; AMGx claims sm_70+ |
| Turing (T4) | 7.5 | Untested | `CUDA_ARCH=75` |
| Pascal (GTX 10-series) | 6.1 | Phase 1 FFI tested on GTX 1080 | `CUDA_ARCH=61`; **NOT supported by CUDA 13+** |

### 2.5 Compiler / C++ standard library

| Component | Reference | Known-working window |
|-----------|-----------|----------------------|
| gcc | 13.3 | 11 – 14 (C++17 target) |
| libstdc++ | 13 | 11 – 14 |
| Python | 3.12 | 3.10 – 3.13 |

The `std::string(string_view)` two-arg-form requirement in Phase 4
(`PROJECT_SUMMARY.md` §8.3.2) is specific to libstdc++ 13 missing the
P2499 one-arg overload. On libstdc++ 14+ or Python 3.13 you MAY not
need that compatibility workaround, but keeping it does not hurt.

### 2.6 OS

| OS | Status | Notes |
|----|--------|-------|
| WSL2 Ubuntu on Win11 | **Tested, reference** | |
| Native Linux (Ubuntu 22.04 / 24.04) | Expected-to-work | No WSL-specific code; should be easier (no driver translation) |
| Linux on metal with A100 / H100 | Expected-to-work, untested | Target for next session |
| macOS | — | **Not supported** (no CUDA) |
| Windows native | — | **Not supported** (JAX GPU Windows support is experimental at best) |

---

## 3. Known-broken combinations (do not attempt)

| Combination | What breaks | Why |
|-------------|-------------|-----|
| JAX 0.9 + jaxlib 0.10 (or vice versa) | Import error | Version lockstep required |
| CUDA 13+ targeting Pascal (sm_60/61/62) | nvcc compile error | Pascal support removed in CUDA 13.0 |
| pyamgx as the AMGx entry point | Project decision | Effectively abandoned upstream; issue #40 unanswered since 2025-05 |
| AMGx without MPI libraries present at runtime | `libamgxsh.so` fails to dlopen | `-DAMGX_NO_MPI=ON` at build time does NOT remove the `libmpi.so.40` NEEDED entry; deployment constraint (install `libopenmpi-dev`) |
| `-ffast-math` or `--use_fast_math` | L1 / L2 break | FP associativity changes; iter counts may drift by 1-3 iters; bit-identical is lost |
| Missing `XLA_PYTHON_CLIENT_*` env vars | OOM at 128³ | JAX defaults pre-allocate 75% of VRAM, colliding with AMGx's own allocations |

---

## 4. Migration procedures (by dimension)

### 4.1 Switching GPU architecture (A100 / H100 / RTX 40 / RTX 50)

Effort: **30-60 minutes** if CUDA and AMGx are already installed.

1. Update `CUDA_ARCH` env var to the new compute capability number (see §2.4).
2. Rebuild AMGx from source with the new `-DCUDA_ARCH=XX`.
3. `rm -rf dilu/*/build` and `bash dilu/*/build.sh` for all four phases.
4. Run §6 environment sentinel.
5. Expect L2-level match (iter counts, correlations); wall times and VRAM
   will differ.

### 4.2 Switching CUDA version (12.x → 12.y within 12)

Effort: **5-15 minutes**.

1. Update `nvcc` on PATH.
2. No code changes required.
3. Rebuild the four `.so` files and AMGx.
4. Run sentinel.

### 4.3 Switching CUDA 12 → CUDA 13

Effort: **1-3 hours**.

1. Install CUDA 13 toolkit.
2. Check AMGx upstream for 13.x compatibility; may need to patch or wait
   for new AMGx release. Rebuild AMGx.
3. Rebuild our four `.so`.
4. Run sentinel.
5. Known concern: cuSPARSE 13.x may have behaviour differences we haven't
   tested; verify T6 (apply vs serial) at multiple grid sizes.

### 4.4 Switching JAX 0.9.x → 0.10.x

Effort: **unknown, estimate 0.5-2 days** (depends on how much
`jax.ffi` churned).

1. Update JAX / jaxlib in the virtualenv.
2. Run `python dilu/ffi_mvp/python/_probe.py` (or equivalent) to check
   the FFI API surface.
3. Try to build Phase 1 first (smallest change surface); if it compiles
   and T1 passes, the API is compatible. If not, patch the C++ handler
   signatures per the probe output.
4. Build Phase 2, 3, 4 in order; fix each if needed.
5. Run full sentinel.

### 4.5 Switching AMGx version

Effort: **1-4 hours**.

1. Install new AMGx at `$HOME/local/amgx_newver/` (keep old install
   side-by-side for rollback).
2. Update `dilu/amgx/CMakeLists.txt`'s `AMGX_ROOT` default, or pass
   `-DAMGX_ROOT=...` to CMake.
3. Rebuild `dilu/amgx/build/libdilu_amgx.so`.
4. Check `ldd` on the new `.so` — verify `libamgxsh.so` resolves to
   the new install via embedded RPATH.
5. Run T8, T9, update_coefficients, F2 acid (the AMG-specific tests).
6. If `AMGX_solver_resetup` is removed (future v2.6+ risk), patch
   `dilu/amgx/cpp/amgx_update_coefficients.cc` to call full
   `AMGX_solver_setup` instead. Re-measure amortization break-even K.

### 4.6 Switching host OS (WSL2 → native Linux)

Effort: **2-6 hours** assuming CUDA is already installed on the host.

1. Install JAX + dependencies (`pip install "jax[cuda12]==0.9.0" numpy scipy matplotlib`).
2. Install AMGx per §4.1.
3. Clone the repo, rebuild all four phases.
4. Run sentinel.
5. Expect L2-level match; wall times are usually faster on native Linux
   because WSL2 has some overhead on CUDA dispatch calls (Phase 1
   cross-machine data is consistent with this).

### 4.7 Switching Python 3.12 → 3.13

Effort: **30-90 minutes**.

1. Rebuild the JAX virtualenv with Python 3.13.
2. The `std::string(data, size)` workaround in Phase 4 §8.3.2 may no
   longer be strictly required (P2499 overload available), but leaving
   it is harmless.
3. Rebuild all four `.so`.
4. Run sentinel.

---

## 5. Mixed pip / conda CUDA — special case

If the target machine has **pip-installed `nvidia-cuda-*-cu12` packages
instead of a system CUDA toolkit**, CMake's `find_package(CUDAToolkit)`
may not locate headers / libraries.

**Recommended**: install a proper CUDA toolkit (via NVIDIA's .run file,
apt, or `conda install -c nvidia cuda-toolkit`). The pip `nvidia-cuda-*-cu12`
packages provide runtime but not a unified toolkit layout.

**Alternative**: set `CUDAToolkit_ROOT` environment variable to the
pip packages' merged root. This has not been tested in this project.

This is the exact situation observed on the RTX 5060 machine during
Phase 1 cross-machine repro — `nvcc` was missing until `conda install
-c nvidia cuda-toolkit>=12.8` supplied it. Documented as an environment
landmine; `PROJECT_SUMMARY.md` §4.4 requires a full CUDA toolkit install.

---

## 6. Environment sentinel — "is my stack compatible?"

Run this sequence from a clean checkout on the target machine to verify
L2-level compatibility. **Three numbers must match exactly** regardless
of hardware; wall times may differ by 10-50×.

### 6.1 Prerequisites

1. All 4 `.so` files built (run `bash dilu/*/build.sh` for each phase).
2. AMGx v2.4+ installed at `$HOME/local/amgx/` (or override `AMGX_ROOT`).
3. XLA env vars set per §4.5 of PROJECT_SUMMARY.

### 6.2 The three sentinel outputs

| Test command | Acceptance (MUST match) | What it tests |
|--------------|-------------------------|---------------|
| `python dilu/cusparse/tests/test_t7_pcg_stiff.py` | **DILU-PCG: 24 iters, Jacobi-PCG: 71 iters** | Phase 2 cuSPARSE DILU numerical correctness + PCG driver determinism |
| `python dilu/multicolor/tests/test_t7_pcg_iteration_count.py` | **36 iters** | Phase 3 multi-color DILU |
| `python dilu/amgx/tests/test_t9_iter_count_128.py` (CLASSICAL) | **15 iters** | Phase 4 AMG preconditioner |

One command combining all three sentinels does not currently exist; adding
a `scripts/env_sentinel.sh` that runs these three and checks the expected
integers is a recommended (but not yet implemented) follow-up.

### 6.3 Interpreting sentinel results

| Observed | Interpretation | Action |
|----------|----------------|--------|
| All three match exactly | L2 compatible — safe to trust all other tests | Proceed with full test suite |
| T7 = 24 but T7 (multicolor) ≠ 36 | Phase 3 red-black coloring or permutation is broken | Inspect `dilu/multicolor/python/coloring.py` output |
| T9 ≠ 15 but smoke (8³) passes | AMGx config loaded but convergence differs from reference | Check AMGx version; may be v2.4 vs v2.5 behavioral diff |
| T9 = 15 but F2 acid (`test_t_acid_32.py`) fails with corr > 0.25 | Physical benchmark setup inconsistent (most likely the projection operator consistency rule of §8.7) | Compare against Phase 2.5 plots |
| All tests fail with SIGSEGV at first AMG solve | Phase 4 §8.3.3 config lifetime bug reintroduced | Check `plan_registry.cc` for config destruction order |
| Compile errors on `ffi::Attr<T>` | JAX API drift | See §4.4 JAX upgrade path |

### 6.4 What sentinel does NOT test

- Wall time / dispatch µs — these are L3 and hardware-dependent
- VRAM absolutes — depends on concurrent processes on the GPU
- Physical benchmark image visual identity — may differ by ULP rendering
- Anything involving matplotlib versions — plots are indicative, not canonical

---

## 7. Breakage signal decoder (when something goes wrong)

Map symptoms to probable root cause.

### Compile-time

| Symptom | Most likely cause |
|---------|-------------------|
| `error: 'jax::ffi::Attr' is not a template` | JAX FFI API version shift (JAX 0.10.x?); see §4.4 |
| `error: no matching function for call to 'std::__cxx11::basic_string<char>::basic_string(std::basic_string_view<char>)'` | libstdc++ < 13; use `(data(), size())` form |
| `fatal error: 'amgx_c.h' file not found` | `AMGX_ROOT` not set or AMGx not installed; see §4.5 |
| `nvcc fatal: Unsupported gpu architecture 'compute_86'` | CUDA toolkit older than Ampere support (< 11.1); upgrade |
| `fatal error: 'cusparse.h' file not found` | CUDA toolkit not fully installed (only runtime libs?); install `cuda-toolkit` meta-package |

### Link-time

| Symptom | Most likely cause |
|---------|-------------------|
| `undefined reference to cusparseSpSV_updateMatrix` | CUDA toolkit < 12.0; upgrade to 12.x |
| `undefined reference to AMGX_matrix_replace_coefficients` | AMGx < 2.3 or build issue; rebuild AMGx |
| `cannot find -lamgxsh` | `AMGX_ROOT/lib` not on link path; check CMakeLists library search |

### Runtime

| Symptom | Most likely cause |
|---------|-------------------|
| `error while loading shared libraries: libamgxsh.so` | `libopenmpi` missing; install `libopenmpi-dev` |
| `error while loading shared libraries: libdilu_amgx.so` has RPATH issue | CMake `INSTALL_RPATH` didn't bake; fall back to `export LD_LIBRARY_PATH=$HOME/local/amgx/lib` |
| SIGSEGV on first AMG solve | Phase 4 §8.3.3 config lifetime bug; check plan_registry destruction order |
| SIGSEGV on first cuSPARSE solve | Phase 2 §8.1 updateMatrix call missing; check `cusparse_dilu_apply.cc` |
| T6 apply error 0.13+ | Same as above (cuSPARSE diagonal caching) |
| T7 iter count wrong | Preconditioner is wrong (factor OR apply); check T4/T5/T6 first |
| T9 iter count wrong on CLASSICAL config | AMGx config string not matching; check `dilu/amgx/configs/classical_rs.json` |
| `CUSPARSE_STATUS_NOT_SUPPORTED` on solve | cuSPARSE version too old OR matrix has structural zeros on diagonal |
| `AMGX_STATUS_NOT_CONVERGED` | May be legitimate (check residual); may be bad config; may be matrix outside AMGx assumptions |
| Memory type mismatch in zero-copy probe | JAX allocator not placing CSR on device; check that `jax.device_put` was called |

---

## 8. Migration effort estimates (ballpark)

| Change | Estimate | Risk |
|--------|----------|------|
| Minor CUDA bump (12.4 → 12.6) | < 30 min | Low |
| CUDA 12 → CUDA 13 | 1-3 h | Medium (AMGx rebuild needed) |
| Ampere → Ada (RTX 40) | 30 min | Low (same arch family) |
| Ampere consumer → A100 / H100 | 1 h | Low |
| JAX 0.9 → 0.9.x patch | 5 min | Very low |
| JAX 0.9 → 0.10 | 0.5-2 days | **High** (FFI surface is historically unstable) |
| WSL2 → native Linux | 2-6 h | Low |
| AMGx v2.5 → v2.6 (hypothetical) | 1-4 h | Medium (watch `resetup` fate) |
| AMGx → Hypre BoomerAMG | 1-3 weeks | High (different C API, different config model) |
| AMGx → Ginkgo PGM-AMG | 2-4 weeks | High (new library, mixed-precision testing) |

---

## 9. When to read which document

| Question | Document |
|----------|----------|
| "What does the code actually do?" | `docs/design/phase*.md` + the source files |
| "What numbers should I expect to see?" | `docs/PROJECT_SUMMARY.md` §6 (deterministic) + §7 (ranges) |
| "My environment is different" | **This document** |
| "My environment matches but something is off" | `docs/PROJECT_SUMMARY.md` §8 (critical landmines) |
| "What was the state of the project on 2026-04-21?" | `docs/session_logs/SESSION_HANDOFF_20260421.md` |
| "How do I set up a new A100 from scratch?" | `docs/session_logs/SESSION_HANDOFF_20260421.md` §6 |

---

## 10. Closing guidance

- **Don't aim for L1 on a different machine.** It's not physically possible
  across different GPUs, and usually impossible across minor version bumps.
- **Aim for L2.** If the three sentinel numbers (24 / 36 / 15) match, the
  project is working as designed.
- **Report discrepancies up.** If you find a version combination that
  works but is not in this table, add it. If you find one that fails,
  add it under §3 with the symptom. The portability matrix grows by
  accretion.

*End of portability guide.*
