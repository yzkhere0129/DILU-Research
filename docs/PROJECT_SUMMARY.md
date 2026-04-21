# DILU-Research — Project Summary (Phases 1 / 2 / 2.5 / 3)

**Repo**: `https://github.com/yzkhere0129/DILU-Research` (private)
**Working tree**: `/home/yzk/DILU-Research`
**Reporting window**: 2026-04-19 → 2026-04-21
**Primary hardware**: RTX 3050 Laptop (4 GB VRAM, CC 8.6, FP64 @ 1/32 FP32)
**Status**: Phase 1, Phase 2, Phase 2.5, Phase 3 all PASS. Phase 4 NOT started.

---

## 0. Project goal (verbatim from master plan)

Build a device-resident, zero-host-copy solver for the extremely stiff pressure
Poisson equation in additive-manufacturing (AM) melt-pool CFD. Stack: pure `jax`
+ `jax.ffi` + C++ / CUDA + NVIDIA libraries only. No JAXFLUIDS or other
high-level CFD framework. Hard constraint: condition numbers up to $10^8$,
density ratios $10^3$, viscosity ratios matching AM reality.

The four-phase roadmap progressively breaks the DILU preconditioner's serial
forward/backward-substitution bottleneck:

1. FFI toolchain bring-up (proof of JAX ↔ C++/CUDA pipeline)
2. cuSPARSE SpSV DILU (level-scheduling exact DILU)
3. Multi-color DILU (bypass level scheduling for raw GPU parallelism)
4. AMG / AMGx (Plan B if DILU cannot handle worst-case stiffness)

Phases 1 / 2 / 2.5 / 3 are complete. Phase 4 has not been started.

---

## 1. Phase-by-phase summary

### 1.1 Phase 1 — FFI toolchain bring-up

**Purpose**: prove the JAX ↔ C++/CUDA FFI pipeline works end to end.
**Artifacts root**: `dilu/ffi_mvp/`
**Kernel**: `y = D^{-1} (b - A x)` — Jacobi-preconditioned residual on CSR, float64.
**Design docs**:
- `docs/design/phase1_dilu_math_foundation.md`
- `docs/design/phase1_ffi_prototype_architecture.md`

**Build system**: CMake + C-ABI `.so` + `ctypes` + `jax.ffi.pycapsule`.
Explicitly NOT pybind11 / nanobind — the FFI handler is invoked by XLA directly,
not by Python.

**Acceptance tests** (T1–T3 + under-jit HLO inspection):

| Test | Structure | n | nnz | Max error | Tolerance | Verdict |
|------|-----------|---|-----|-----------|-----------|---------|
| T1 | 1-D Laplacian tridiag | 1000 | 2998 | 2.22e-16 | 3.15e-14 | PASS |
| T2 | 3-D 7-pt Laplacian (10³) | 1000 | 6400 | 8.88e-16 | 5.57e-14 | PASS |
| T3 | Diag-dominant random (diag spread 10³) | 500 | 5486 | 4.44e-16 | 1.48e-13 | PASS |
| under-jit | 3-D 7-pt (8³) inside `jax.jit` | 512 | 3072 | 1× custom-call, 0× copy-start/copy-done | — | PASS |

All errors at float64 ULP floor. HLO inspection confirms zero host-device
traffic in steady state.

**FFI dispatch overhead baseline** (per-call median, n=1000, under `jax.jit`,
inputs pre-placed on device):

| Machine | GPU | CPU | median | min | p99 | Notes |
|---------|-----|-----|-------:|----:|----:|-------|
| yzk-laptop | RTX 3050 Laptop | laptop class | **267 µs** | 224 | 370 (p95) | CUDA 12.4, JAX 0.9.0, idle |
| HR54WV2 | GTX 1080 | Xeon Gold 5120 | **455 µs** | 411 | 716 | CUDA 12.9, JAX 0.9.1, idle |
| ME-6T8XHG4 | RTX 5060 | Intel Ultra 7 265F | 1003 µs* | 130 | 2734 | CUDA 13.2, JAX 0.9.1, **GPU contended** |

*5060 reading is confounded by a non-DILU task occupying the GPU. Not a clean
baseline.

**Revised attribution**: the original Phase 1 report attributed 267 µs to
"consumer-laptop PCIe / FP64 limitations". The three-machine cross-check
**refuted** this — the GTX 1080 on a server-class Xeon is **slower** (455 µs),
not faster. The actual explanation is the JAX + `jax.ffi` + WSL2 + Python
dispatch stack as a whole. This finding is documented in
`docs/benchmark/phase1_repro_HR54WV2_gtx1080.md`.

**VRAM**: peak JAX-attributable < 50 MB on any test. Total Phase-1 footprint
under 500 MB cap by two orders of magnitude.

**Deliverable**: `docs/benchmark/phase1_mvp_report.md`.

---

### 1.2 Phase 2 — cuSPARSE SpSV DILU preconditioner

**Purpose**: integrate NVIDIA's generic `cusparseSpSV` triangular solver as the
exact-DILU preconditioner, using opaque-token lifecycle to avoid per-call
analysis-phase overhead.
**Artifacts root**: `dilu/cusparse/`
**Design docs**:
- `docs/design/phase2_cusparse_level_scheduling_math.md`
- `docs/design/phase2_cusparse_ffi_architecture.md`

**Architecture**: 4 FFI primitives.

| Primitive | Purpose | Hot-path? |
|-----------|---------|-----------|
| `dilu_factor` | Compute $D_*$ on device via the DILU serial recurrence | No (analyze-time only) |
| `cusparse_dilu_analyze` | `cusparseSpSV_analysis`, returns opaque `uint64` token | No (once per matrix pattern) |
| `cusparse_dilu_apply` | Two `cusparseSpSV_solve` + diagonal scale | **Yes** |
| `cusparse_dilu_release` | Free plan registry entry | No |

**Descriptor lifecycle**: Option C (opaque `uint64` token) per arch doc.
One 8-byte D→H scalar copy per `apply` is the sole tolerated host-device
exception to the zero-copy rule.

**Acceptance tests** (T4 / T5 / T6 / T7 + under-jit HLO):

| Test | Content | n | Tolerance | Max err | Verdict |
|------|---------|---:|----------:|---------|---------|
| T4.1 | DILU factor on 1-D tridiag | 500 | 2.22e-11 | **0.00e+00** | PASS |
| T4.2 | DILU factor on 3-D 7-pt Laplacian (10³) | 1000 | 1.33e-10 | **0.00e+00** | PASS |
| T4.3 | DILU factor on diag-dominant random | 200 | 4.43e-11 | **0.00e+00** | PASS |
| T5 | Apply on diagonal A → Jacobi | 2000 | 5.18e-14 | 4.44e-16 | PASS |
| T6.1 | Apply vs serial reference (3-D Laplacian 8³) | 512 | 3.98e-10 | 1.66e-16 | PASS |
| T6.2 | Apply vs serial reference (3-D Laplacian 12³) | 1728 | 1.34e-09 | 2.22e-16 | PASS |
| **T7** | **DILU-PCG iter count on 16³ stiff (ρ contrast 100)** | 4096 | DILU < Jacobi | **24 vs 71** | **PASS (3× reduction)** |
| under-jit apply | Compiled HLO inspection | 216 | 1× custom-call, 0× copy ops | 1 custom-call, 0 copy ops | PASS |
| under-jit factor | Compiled HLO inspection | 64 | 1× custom-call, 0× copy ops | 1 custom-call, 0 copy ops | PASS |

Factor errors are **bit-identical** (0.00e+00) because the GPU kernel and the
NumPy reference walk the same row order with serial left-to-right arithmetic.

**Analysis amortization measurement** (n=1000 tridiag):
- $T_{\text{analyze}} \approx 18$ ms (one-shot cost per matrix pattern)
- $T_{\text{apply}} \approx 6$ ms (per PCG iter)
- Break-even: **K ≈ 2.4–2.8 apply calls** — any realistic PCG run (≥ 24 iters
  at 16³ stiff) amortizes analysis trivially.

**Non-obvious landmines caught during bring-up** (documented as permanent
cautions):

1. **`cusparseSpSV_analysis` caches diagonal values, not just sparsity
   pattern.** Overwriting `working_values` after analyze is invisible to the
   next solve. Fix: call `cusparseSpSV_updateMatrix(CUSPARSE_SPSV_UPDATE_GENERAL)`
   before every solve. Symptom before fix: T6 absolute error 0.134.
2. **Device-pointer-based pattern fingerprint breaks under `jax.jit`.** XLA
   may re-buffer inputs across calls; same pattern produces different pointers.
   Fix: plan entry owns copies of `row_ptr`, `col_idx`, `diag_offset` at analyze
   time. Fingerprint relaxed to `(n, nnz)`.

**VRAM**: peak Phase-2-attributable ≤ 83 MiB on any single test. Platform
allocator returns memory cleanly on subprocess exit.

**cuSPARSE runtime version probed via `cusparseGetProperty`**: 12.3.1.

**Deliverable**: `docs/benchmark/phase2_cusparse_report.md`.

---

### 1.3 Phase 2.5 — Physical sanity benchmark (anti-hallucination)

**Purpose**: ULP-level numerical correctness ≠ physical correctness. Three
tests compare the Phase 2 DILU-PCG against absolute gold standards
(analytical physics laws, dense-exact `scipy.sparse.linalg.spsolve`) and
**visualize** the results so pathologies (interface halo, tentacle vortices,
geometry-correlated residual) become visible.
**Artifacts root**: `dilu/cusparse/tests/physical_benchmark.py`,
`dilu/cusparse/bench/plots/`

**Tests** (all on 32³ = 32 768 cells):

| Test | Setup | Gold standard | DILU measured | Verdict |
|------|-------|---------------|---------------|---------|
| **A** — Divergence map | 32³, ρ-ratio 1000 sphere, random low-k velocity, pressure projection | $\max\|\nabla \cdot \mathbf{u}\| = 0$ | **4.13e-09** (after 115 PCG iters, tol 1e-10) | PASS |
| **B** — Static droplet spurious currents | same geometry, $\mathbf{u}_0 = 0$, CSF surface tension σ=0.07, 10 projection steps | $\|\mathbf{u}\|_\infty = 0$ | **9.64e-07** @ t=10, linear growth | PASS (parasitic-current regime, not solver) |
| **C** — Residual halo vs exact solve | 32³ three-tier (gas 1 / liquid 1000 / solid 8000), melt-pool dip, DILU-PCG fixed 50 iter vs `scipy.sparse.linalg.spsolve` | geometry-uncorrelated error | max\|E\|=18.5 (rel 21%), **no interface correlation** | PASS (mid-convergence, not pathological) |

**Result images** (full path `dilu/cusparse/bench/plots/`):

| File | Content |
|------|---------|
| `divergence_map.png` | z=16 slice of $\nabla \cdot \mathbf{u}$ post-projection. No halo on sphere boundary. |
| `spurious_currents.png` | 4-lobed CSF parasitic-current pattern + $\|u\|_\infty$ vs step (linear, not exponential). |
| `residual_halo.png` | Mean-detrended $p_{\text{dilu}} - p_{\text{exact}}$ at iter 50. Error is a smooth low-frequency gradient from the pinned corner; does NOT trace black interface contours. |

**Methodological note** (documented in report): the divergence operator used
to build the Poisson RHS, the velocity correction operator, and the
post-projection divergence check must share the same face-centered
interpolation. An earlier inconsistency caused a false-FAIL (max\|div\| = 4
looking like solver failure) that was purely a discrete-consistency bug on the
test side. After unifying `cell_to_face → divergence_from_faces`, the discrete
identity closes at machine precision.

**Verdict**: Phase 2 DILU is physically healthy under all three probes at
1000× and 8000× density contrasts. Absolute iter counts at this scale (115 / 82
per step / etc.) are manageable but will scale per the Gustafsson
$O(\kappa^{1/4})$ bound at 128³+. Phase 4 AMG addresses the scaling limit.

**Deliverable**: `docs/benchmark/phase2.5_physical_report.md`.

---

### 1.4 Phase 3 — Multi-color DILU (Path B, no cuSPARSE)

**Purpose**: bypass cuSPARSE's level-scheduling triangular solver entirely.
Use graph coloring to split the DILU dependency DAG into a handful of
independent batches; each batch is a single SpMV-like kernel launch.
**Artifacts root**: `dilu/multicolor/`
**Design docs**:
- `docs/design/phase3_multicoloring_math.md`
- `docs/design/phase3_multicolor_ffi_architecture.md`

**Architecture decisions signed off by user**:

- **Path B**: custom color-stride CUDA kernels. `libdilu_multicolor.so` links
  `libcudart.so.12` only — zero cuSPARSE dependency (verified via `ldd`).
- **Host-side coloring**: pure-Python `red_black_color()` fast path for
  structured 7-pt stencils, `greedy_color_csr()` (first-fit) fallback for
  generic CSR. No new external dependencies (no networkx, no scipy coloring,
  no cuGraph).
- **4 FFI primitives** mirror Phase 2: `multicolor_analyze`, `multicolor_apply`,
  `multicolor_refactor`, `multicolor_release`. Same opaque `uint64` token
  lifecycle.
- **F2 stop signal** (user directive, verbatim): if Test C
  `corr(|E_demean|, |\nabla \log \rho|) > 0.5`, halt immediately — no 4-color,
  no greedy, no rescue. Lock Phase 2 as final DILU baseline. Start Phase 4.

**Acceptance tests** — all 8 criteria PASS:

| # | Criterion | Metric | Threshold | Result | Status |
|---|-----------|--------|-----------|--------|--------|
| C1 | $\tilde D_*$ ULP correctness | max\|Δd̃\| on 16³ stiff | ≤ 1e-13 | **0.00e+00** | PASS |
| C2 | Diagonal A → Jacobi | `|z_P3 − r/d|∞` n=2000 | = 0 | **0.00e+00** | PASS |
| C3 | 16³ Laplacian apply vs serial | `|z_gpu − z_ref|∞` | ≤ 1e-12 | **2.22e-16** (1 ULP) | PASS |
| C4 | T7 iter penalty ≤ 3× | iters / 24 | ≤ 3.0 | **1.50× (36 iters)** | PASS |
| C5 | T7 beats Jacobi | iters < 71 | < 71 | **36 < 71** | PASS |
| C6 | Test A max\|∇·u\| | 32³ projection | ≤ 4.1e-8 | **3.29e-9** | PASS |
| C7 | Test B ‖u‖∞ @ t=10 | AND max\|div\| ≤ 1e-12 | ≤ 1.9e-6 | **9.64e-7, 7.6e-15** | PASS |
| **C8** | **Test C corr(\|E\|, \|∇log ρ\|) — F2 ACID TEST** | Pearson correlation | **≤ 0.25** | **-0.102** | **PASS** |

F2 STOP signal (corr > 0.5): **NOT TRIGGERED** (far below the 0.5 line;
negative correlation with magnitude 0.1).

**Dispatch and wall-time benchmarks** (median / p95 per-apply, 20 warm + 500
timed, single CUDA stream):

| Matrix | N | nnz | Phase 2 apply (µs) | P2 iters | Phase 3 apply (µs) | P3 iters | apply speedup | total PCG speedup |
|--------|---|-----|-------------------:|---------:|-------------------:|---------:|---------------|-------------------|
| n=1000 tridiag | 1000 | 2998 | 5956.5 / 7339.5 | 1 | 963.7 / 2042.1 | 501 | 6.18× | **0.01× (regression)** |
| 16³ stiff (T7) | 4096 | 27 136 | 1853.7 / 2996.9 | 24 | 952.8 / 2454.1 | 36 | 1.95× | **1.30×** |
| **32³ 3-tier (Test C)** | 32 768 | 223 226 | 4342.0 / 5762.9 | 128 | 1035.9 / 2187.2 | 197 | 4.19× | **2.72×** |

Result images (full path `dilu/multicolor/bench/plots/`, side-by-side with
Phase 2.5):

| File | Content |
|------|---------|
| `divergence_map.png` | Phase 3 Test A: clean domain, no sphere-boundary halo. |
| `spurious_currents.png` | Phase 3 Test B: trajectory digit-identical to Phase 2.5. |
| `residual_halo.png` | Phase 3 Test C: no interface geometry signature in error. |

**Kernel launch count** (arch doc §4.3 prediction):
`2 · n_colors + 3 gathers + 1 scale = 2k + 4`. For red-black k=2 → 8
kernels/apply. Phase 2 cuSPARSE `SpSV_solve` dispatches an opaque number of
level barriers internally; on a realistic 256×128×64 grid the Phase 2 math
doc estimated ~446 barriers. **Phase 3 replaces that with a fixed small
constant.**

**Caveats (honest)**:

1. **n=1000 tridiag regresses 81×.** Red-black on a 1-D tight-band matrix
   destroys the natural-ordering locality that makes DILU ≈ exact LU on
   tridiag. Documented failure mode → routing rule: 1-D-dominant problems
   stay on Phase 2, 3-D goes to Phase 3.
2. **12-byte D→H copy at analyze time** (to mirror `color_offsets` for the
   host-side launch loop). One-shot, NOT in the hot path. Apply's hot path
   retains Phase 2's 8-byte-token-only contract.

**Library independence**: `ldd libdilu_multicolor.so` shows `libcudart.so.12`
as the only NVIDIA dependency. This is a hard property, not aspirational.

**Deliverable**: `docs/benchmark/phase3_multicolor_report.md`.

---

## 2. Cross-phase comparison — the headline numbers

### 2.1 Numerical correctness

| Phase | Test | n | Max error | ULP class |
|-------|------|---|-----------|-----------|
| 1 | T1 tridiag | 1000 | 2.22e-16 | 1 ULP |
| 1 | T2 7-pt Laplacian | 1000 | 8.88e-16 | 4 ULP |
| 1 | T3 random stiff | 500 | 4.44e-16 | 2 ULP |
| 2 | T4 factor | up to 1000 | 0.00e+00 | exact (bit-identical) |
| 2 | T6 apply on 3-D | 1728 | 2.22e-16 | 1 ULP |
| 3 | T4 factor | 4096 | 0.00e+00 | exact (bit-identical) |
| 3 | T6 apply on 3-D | 4096 | 2.22e-16 | 1 ULP |

All three implementations are at the floor of float64 associativity on
representative test matrices.

### 2.2 T7 iteration count — the canonical stiff-problem comparison

| Preconditioner | Iterations on 16³ ρ=100 stiff PCG | Notes |
|----------------|----------------------------------:|-------|
| Jacobi (Phase 1 kernel) | **71** | dispatch baseline, no preconditioner structure |
| Phase 2 exact DILU (cuSPARSE SpSV) | **24** | 3× reduction over Jacobi |
| Phase 3 multi-color DILU (red-black) | **36** | 1.5× penalty vs Phase 2, still 2× better than Jacobi |

Phase 3's iteration-count penalty of 1.5× sits in the lower half of the
Duff-Meurant 1989 classical range (median ~2×) and below Li-Saad 2010's
reported 1.8–2.0× for GPU MC-ILU(0).

### 2.3 Physical-benchmark consistency (32³)

| Metric | Phase 2.5 (exact DILU) | Phase 3 (multi-color) | Verdict |
|--------|------------------------|-----------------------|---------|
| Test A max\|∇·u\| | 4.13e-9 | **3.29e-9** | Phase 3 marginally better |
| Test B ‖u‖∞ @ t=10 | 9.64e-7 | 9.64e-7 | digit-identical trajectories |
| Test B max\|div\| per step | ~5e-15 | ~7e-15 | both at machine zero |
| Test C F2 correlation | ~0.1 (estimated baseline) | **-0.102** | both well below 0.25 |

**Phase 3 is physically indistinguishable from Phase 2** on all three acid
tests.

### 2.4 Per-apply dispatch vs total PCG wall time (RTX 3050 Laptop)

Per-apply Phase 3 / Phase 2 speedup grows with grid size and nnz, consistent
with the expected amortization of color-stride kernel launches over more work:

| Grid | Per-apply speedup | Total PCG speedup (incl. iter-count penalty) |
|------|------------------:|---------------------------------------------:|
| n=1000 tridiag | 6.18× | 0.01× (red-black regression, route to Phase 2) |
| 16³ 3-D stiff | 1.95× | 1.30× |
| **32³ 3-tier stiff** (realistic AM-adjacent) | **4.19×** | **2.72×** |

---

## 3. Cumulative artifact inventory

### 3.1 Source code

| Phase | Directory | SLOC (approx) | Purpose |
|-------|-----------|--------------:|---------|
| 1 | `dilu/ffi_mvp/` | ~600 | FFI toolchain MVP (Jacobi residual) |
| 2 | `dilu/cusparse/` | ~2 380 | cuSPARSE SpSV DILU + 4 FFI primitives + T4–T7 + physical bench |
| 3 | `dilu/multicolor/` | ~3 995 | Multi-color DILU, Path B custom kernels, T4–T7 + physical bench + comparison |

Each phase has its own `CMakeLists.txt`, `build.sh`, `cpp/`, `cuda/`,
`python/`, `tests/`, `bench/`. No cross-phase imports — Phase N does not edit
Phase N-1.

### 3.2 Design documents (`docs/design/`)

| File | Phase | Content |
|------|------:|---------|
| `phase1_dilu_math_foundation.md` | 1 | DILU math foundations, serial-nature dissection, three escape routes preview |
| `phase1_ffi_prototype_architecture.md` | 1 | FFI API recon, MVP kernel spec, build-system choice |
| `phase2_cusparse_level_scheduling_math.md` | 2 | cuSPARSE API recon, DILU ≠ ILU(0) integration traps, AM iter-count calibration |
| `phase2_cusparse_ffi_architecture.md` | 2 | Token lifecycle Option C, 4 FFI primitives, pattern fingerprint |
| `phase3_multicoloring_math.md` | 3 | Reordered DILU formalism, Duff-Meurant penalty theory, 8 acceptance criteria, 5 failure modes |
| `phase3_multicolor_ffi_architecture.md` | 3 | Path B commitment, host-side coloring, kernel design, F2 stop protocol |

### 3.3 Benchmark / verification reports (`docs/benchmark/`)

| File | Phase | Content |
|------|------:|---------|
| `phase1_mvp_report.md` | 1 | T1–T3 + under-jit, 3050 baseline 267 µs |
| `phase1_repro_HR54WV2_gtx1080.md` | 1 | Cross-machine repro: 1080 median 455 µs |
| `phase2_cusparse_report.md` | 2 | T4–T7 + under-jit, 24-vs-71 iter headline, analysis amortization K≈2.4 |
| `phase2.5_physical_report.md` | 2.5 | Gold-standard physical benchmarks A/B/C; DILU healthy |
| `phase3_multicolor_report.md` | 3 | 8 acceptance criteria PASS, 32³ total-PCG 2.72× speedup, F2 not triggered |

### 3.4 Result images

| File | Phase | Content |
|------|------:|---------|
| `dilu/cusparse/bench/plots/divergence_map.png` | 2.5 | Test A div(u) slice, no sphere halo |
| `dilu/cusparse/bench/plots/spurious_currents.png` | 2.5 | Test B quiver + ‖u‖∞ linear growth |
| `dilu/cusparse/bench/plots/residual_halo.png` | 2.5 | Test C mean-detrended error, no interface correlation |
| `dilu/multicolor/bench/plots/divergence_map.png` | 3 | Phase 3 Test A, visually identical to Phase 2.5 |
| `dilu/multicolor/bench/plots/spurious_currents.png` | 3 | Phase 3 Test B, identical trajectory |
| `dilu/multicolor/bench/plots/residual_halo.png` | 3 | Phase 3 Test C, corr = -0.102 |

---

## 4. Open questions / next-phase posture

### 4.1 Confirmed by measurement

- JAX + `jax.ffi` + C++/CUDA + `pycapsule` pipeline works end to end on
  JAX 0.9.0. The stack did not break across CUDA 12.4 / 12.9 / 13.2 on the
  three test machines — API contract is stable at the levels Phase 1–3 use.
- Phase 2 (exact DILU via cuSPARSE SpSV) is production-grade at 16³ and up.
- Phase 3 (red-black DILU with custom kernels) delivers 2.72× total-PCG
  speedup at 32³ without physical pathology.
- The F2 risk (multi-color interface halo) did NOT materialize at 8000×
  density contrast.

### 4.2 Deferred to Phase 4 (not started)

- AMG / AMGx integration. The Gustafsson $O(\kappa^{1/4})$ iteration-count
  bound means DILU alone will grow to hundreds of iters at 128³+. Phase 2.5's
  115 iters on Test A (32³) is consistent with this and foreshadows that real
  AM grids need AMG.
- Dispatch characterization on server-class GPUs. The 267 µs / 455 µs consumer
  + WSL2 numbers are NOT a fair reading of the FFI stack — A100 / H100 are
  expected to be sub-100 µs per Phase 2 math doc extrapolation. User has not
  yet run the repro on server hardware.

### 4.3 Deferred (not on any phase roadmap)

- Cross-phase routing (1-D-dominant matrices to Phase 2, 3-D to Phase 3).
  Currently a documented caveat, not automated.
- Pattern-change detection beyond `(n, nnz)` fingerprint. Acceptable for
  static grids; AMR / topology-changing cases would need stronger invariants.

---

## 5. Key constraints respected throughout

- float64 everywhere. No `-ffast-math`, no `--use_fast_math`.
- `np.ascontiguousarray` at every NumPy ↔ FFI / NumPy ↔ scipy boundary.
- XLA env vars set before every `import jax`:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`,
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
- Peak JAX-attributable VRAM across all phases: under 100 MB per test.
  No OOM, no swap activity on 4 GB / 10 GB RAM laptop.
- Handle / descriptor singletons per device, process lifetime.
- Phase N+1 does NOT modify Phase N artifacts (verified via `git status`
  before each phase commit).

---

## 6. Reproducibility

Every phase's benchmark script accepts the same VRAM-safety env prelude and
runs as a single Python command from the repo root:

```
# Phase 1 dispatch repro (any machine)
python dilu/ffi_mvp/bench/repro_cross_machine.py

# Phase 2 T7 PCG iter count
python dilu/cusparse/tests/test_t7_pcg_stiff.py

# Phase 2.5 physical benchmark (all three tests)
python dilu/cusparse/tests/physical_benchmark.py

# Phase 3 multi-color vs Phase 2 comparison
python dilu/multicolor/bench/bench_multicolor_vs_cusparse.py
```

Each writes its own hostname-tagged report under `docs/benchmark/`.

---

*End of project summary. For per-phase full detail see the individual reports
listed in §3.3.*
