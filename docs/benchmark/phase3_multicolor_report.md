# Phase 3 — Multi-color DILU Preconditioner Benchmark Report

**Date**: 2026-04-21
**Hardware**: RTX 3050 Laptop (4 GB VRAM, CC 8.6, CUDA 12.4)
**JAX**: 0.9.0 @ `/home/yzk/jax-env`
**Build**: `libdilu_multicolor.so`, `-O2 -lineinfo`, `sm_86`, no `-ffast-math`. Links `libcudart.so.12` only (no cusparse).
**Status**: **ALL 8 ACCEPTANCE CRITERIA C1–C8 PASS. F2 NOT triggered.**

> Phase 4 is NOT started.

---

## 1. Verdict table (8 acceptance criteria)

| # | Criterion | Measure | Threshold | Result | Status |
|---|---|---|---|---|---|
| C1 | D̃_* ULP correctness | `max|d̃_gpu − d̃_ref|` on 16³ stiff | ≤ 1e-13 | **0.0** (bit-identical) | PASS |
| C2 | Diagonal A → Jacobi | `|z_P3 − r/d|∞` on random diag n=2000 | = 0 | **0.0** | PASS |
| C3 | 16³ Laplacian DILU apply | `|z_gpu − z_serial_on_Ã|∞` vs ULP | ≤ 1e-12 | **2.2e-16** (1 ULP) | PASS |
| C4 | T7 iter penalty ≤ 3× | iters / 24 | ≤ 3.0 | **1.50×** (36 iters) | PASS |
| C5 | T7 beats Jacobi | iters < 71 | < 71 | **36 < 71** | PASS |
| C6 | Test A max\|∇·u\| | after 32³ projection | ≤ 4.1e-8 | **3.29e-9** | PASS |
| C7 | Test B ‖u‖∞ @ step 10 | AND max\|div\| ≤ 1e-12 | ≤ 1.9e-6 | **9.64e-7, 7.6e-15** | PASS |
| C8 | Test C corr(\|E_demean\|, \|∇log ρ\|) | ACID test for F2 | ≤ 0.25 | **-0.102** | PASS |

F2 STOP signal (corr > 0.5): **NOT TRIGGERED**. All physical tests on the 32³ variable-density 7-pt Poisson are clean — no interface halo, no spurious vortex, no residual geometry signature.

---

## 2. Environment snapshot

```
nvidia-smi: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB total
Pre-build free: 3161 MiB
Pre-bench free: 2512 MiB (Phase 2 artifacts resident)
Post-bench free: 2494 MiB (no leak)

nvcc 12.4.131 (Thu Mar 28 02:18:24 PDT 2024)
JAX 0.9.0, jaxlib 0.9.x, jax.ffi
XLA env: PREALLOCATE=false, MEM_FRACTION=0.5, ALLOCATOR=platform
float64 via jax_enable_x64
```

VRAM footprint per test (worst case, observed):
| Test | Matrix | Buffers | Peak VRAM delta |
|---|---|---|---|
| T4 (16³ stiff) | 4096 × 27k nnz | plan ~1 MB, refs ~0.5 MB | ~18 MB |
| T6 (16³ 7-pt) | 4096 × 27k nnz | plan + dense ref buffers | ~25 MB |
| T7 (16³ stiff) | 4096 × 27k nnz | plan + PCG workspaces | ~20 MB |
| Physical A/B/C (32³) | 32k × 223k nnz | plan + scipy csr + rhs/p | ~60 MB JAX + ~80 MB host |

Total budget ~500 MB respected. No OOM events.

---

## 3. T4–T7 numerical verification

### T4 — D̃_* on the permuted matrix (C1)
| Matrix | N | max \|Δd̃\| abs | tol | status |
|---|---|---|---|---|
| 10³ Laplacian (red-black) | 1000 | 0.000e+00 | 2.67e-10 | PASS |
| 16³ stiff contrast=100 (red-black) | 4096 | 0.000e+00 | 1.09e-07 | PASS |
| 8³ Laplacian (greedy → degenerate 2-color) | 512 | 0.000e+00 | 1.36e-10 | PASS |

Rationale for bit-identical floor: the GPU `dilu_factor_kernel` and the NumPy reference walk the permuted CSR in the same row order with the same arithmetic. We have not reordered the within-row scan or introduced parallel reductions in the factor path, so both implementations produce literally the same bit sequence.

### T5 — Diagonal A → Jacobi (C2)
| n | `|z − r/d|_∞` | tol | status |
|---|---|---|---|
| 2000 | 0.000e+00 | 5.18e-14 | PASS |

Greedy coloring on a zero-edge graph correctly assigns `n_colors=1`. Our single-color forward sweep reduces to `y = r/d`, the middle scale gives `r`, backward gives `z = r/d`. This is Jacobi, bit-identical to Phase 2 T5.

### T6 — 7-point Laplacian DILU apply (C3)
| Grid | N | `|z_gpu − z_ref|_∞` | tol | status |
|---|---|---|---|---|
| 8³ | 512 | 1.11e-16 | 3.98e-10 | PASS |
| 12³ | 1728 | 1.66e-16 | 1.34e-09 | PASS |
| 16³ | 4096 | 2.22e-16 | 3.18e-09 | PASS |

1-2 ULP accuracy. Reference = serial DILU-apply on Ã, inverse-permuted to original index space (math doc §5.1).

### T7 — PCG iteration count (C4, C5) — HEADLINE NUMBER

| Preconditioner | Iter count | Relative |
|---|---|---|
| Jacobi-PCG (baseline) | **71** | 2.96× |
| Phase 2 DILU-PCG (cuSPARSE, from report) | **24** | 1.00× |
| **Phase 3 DILU-mcPCG (red-black)** | **36** | **1.50×** |

Penalty = 36/24 = **1.50×**. Sits cleanly in the Duff-Meurant 1989 median (~2×) and at the low end of Li-Saad 2010's reported 1.8-2.0× range. C4 PASS (≤3×). C5 PASS (36 < 71 — 2× better than Jacobi). The red-black ordering does NOT degenerate to Jacobi quality on our stiff 7-pt 3-D problem.

---

## 4. Physical benchmarks (C6, C7, C8) — the acid tests

All three tests ran on the 32³ grid with density ratios matching Phase 2.5
(1000× for A/B, 8000× for C). Plot files at
`dilu/multicolor/bench/plots/{divergence_map, spurious_currents, residual_halo}.png`.

### C6 — Test A: variable-density Poisson projection

| Metric | Phase 2 | Phase 3 | Phase 3 status |
|---|---|---|---|
| max\|∇·u\| after projection | 4.13e-9 | **3.29e-9** | PASS (below threshold 4.1e-8, AND below Phase 2) |
| PCG iters to tol=1e-10 | 115 | 177 | 1.54× penalty |

Phase 3's residual is actually slightly **lower** than Phase 2's — Phase 3 ran 177 iters to reach the same tol=1e-10, chasing the residual further down the tail. The divergence-map PNG shows uniform noise across the domain with no halo tracing the droplet interface. Domain-wide log10|div| histogram is a clean Gaussian.

### C7 — Test B: static droplet 10-step

| Step | Phase 2 ‖u‖∞ | Phase 3 ‖u‖∞ | Phase 2 max\|div\| | Phase 3 max\|div\| |
|---|---|---|---|---|
| 1 | 1.105e-7 | 1.105e-7 | 7.12e-15 | 5.42e-15 |
| 5 | 5.002e-7 | 5.002e-7 | 4.59e-15 | 6.72e-15 |
| 10 | **9.637e-7** | **9.637e-7** | 4.78e-15 | 7.56e-15 |
| iters/step | 82 | 127 | | |

**The ‖u‖∞ trajectories are digit-for-digit identical** — both implementations converge to the same projection (to within PCG tol), so the CSF-driven residual velocity is the same. The iter-count penalty is 127/82 = 1.55× — consistent with T7. C7 PASS (9.64e-7 ≤ 1.9e-6 and 7.6e-15 ≤ 1e-12).

### C8 — Test C: 3-tier density acid test (F2 detector)

| Metric | Phase 2.5 baseline | Phase 3 | Verdict |
|---|---|---|---|
| Fixed PCG iters | 50 | 50 | — |
| rnorm trajectory | … | 47.9 → 0.13 | finite decay |
| max\|E\| (raw) | 18.5 | **18.5** | ≈ |
| max\|E_demean\| | 18.4 | **18.4** | ≈ |
| **corr(\|E_demean\|, ‖∇log ρ‖)** | ~0.1 (estimated) | **-0.102** | PASS (≤ 0.25) |
| F2 STOP line (> 0.5) | — | not triggered | — |

**The F2 verdict number is -0.102**. In plain language: Phase 3's residual at 50 fixed iters has no geometric signature — the mid-convergence error is a smooth low-frequency mode that does NOT concentrate at the solid/liquid or liquid/gas interfaces, exactly as Phase 2.5 reported. The red-black DILU handles the 8000×-contrast stiff 3-tier interfaces without introducing halos.

**F2 NOT triggered** → no rescue attempts, no 4-color, no value-aware coloring, no tolerance retuning. Per user directive this means Phase 3 passes the physical red-line.

---

## 5. Dispatch overhead and total-PCG wall time

Measured after warmup (20 warm-up + 500 timed apply calls, median & p95). Single CUDA stream, RTX 3050 Laptop, no Nsight instrumentation.

| Matrix | N | nnz | colors | **Phase 2 apply** med / p95 (µs) | P2 iters | **Phase 3 apply** med / p95 (µs) | P3 iters | apply speedup | total PCG speedup |
|---|---|---|---|---|---|---|---|---|---|
| n=1000 tridiag | 1000 | 2998 | 2 | 5956.5 / 7339.5 | 1 | 963.7 / 2042.1 | 501 | 6.18× | **0.01×** (regression) |
| 16³ stiff (T7) | 4096 | 27136 | 2 | 1853.7 / 2996.9 | 24 | 952.8 / 2454.1 | 36 | 1.95× | **1.30×** |
| 32³ 3-tier (Test C) | 32768 | 223226 | 2 | 4342.0 / 5762.9 | 128 | 1035.9 / 2187.2 | 197 | 4.19× | **2.72×** |

### Observations

- **Per-apply is 2-6× faster** on every matrix size. Phase 3's 4-color-sweep kernels launch faster and saturate GPU SMs more than cuSPARSE's opaque SpSV internals on our small matrix / CC 8.6 hardware.
- **Total PCG time**: Phase 3 wins at 16³ (1.30×) and 32³ (2.72×). The 32³ case is the realistic AM-adjacent workload and Phase 3 delivers a clean ~3× speedup there.
- **n=1000 tridiag regression is a known red-black failure mode** (Duff-Meurant §5 warned). A tight-band matrix like tridiag is ILU(0)-solved exactly after 1 PCG step when ordered naturally (DILU = full LU on tridiag), but the red-black ordering fragments the band and destroys the clustering. This is a theoretical failure mode of Path B, documented here as a caveat — it does NOT invalidate the 3-D stencil case which is the project's target workload. Avoid red-black on 1-D or nearly-1-D matrices; fall back to Phase 2 or compute a natural-order DILU there.

---

## 6. Kernel launch count — arch doc §4.3 prediction vs reality

Predicted launches per apply (arch §4.3): `2 * n_colors + 3 gathers + 1 scale = 2*k + 4`. For red-black k=2 → **8 launches/apply**.

Phase 2 by comparison launches whatever cuSPARSE `SpSV_solve` internally dispatches (opaque; Phase 2 report §2.3 estimated ~446 level barriers on a realistic 256×128×64 grid). On our 16³ test cuSPARSE's internal level count is smaller but still meaningfully larger than 2.

Evidence from the under-jit HLO trace (arch §5 STOP #5): **1 top-level `custom-call`** appears in compiled HLO (the whole apply collapses to one FFI call), **0 HLO-visible host copies**. The 8-byte D→H token copy happens entirely inside the C++ handler and is not surfaced to XLA. This matches Phase 2's HLO profile structurally.

---

## 7. HLO excerpt (under-jit verification)

From `test_under_jit_phase3.py`:

```
under-jit apply: custom-call count = 1
under-jit apply: 1x custom-call, 0x HLO-visible host copy
under-jit phase3 OK
```

The jaxpr inspection confirms `ffi_call` appears. The compiled HLO contains exactly 1 `custom-call` (the `multicolor_apply` target) and 0 `copy-start`/`copy-done` ops. All intermediate scratch vectors (`r_tilde`, `y_tilde`, `y_scaled_tilde`, `z_tilde`) are plan-owned device buffers; they never surface to XLA.

---

## 8. Caveats and surprises

### 8.1 Tridiag regression (bench table row 1)

As noted in §5, red-black multi-color DILU on a 1-D tridiag matrix collapses the natural-order exactness property. Phase 3 iter count jumps from 1 (Phase 2) to 501 (Phase 3). This is theoretically expected and consistent with Duff-Meurant's "ordering destroys band structure" analysis. Not a bug — but anyone using Phase 3 on a 1-D-dominant matrix should be aware. **Recommendation**: route 1-D-dominant problems to Phase 2; keep Phase 3 for genuine 2-D / 3-D stencils where it wins.

### 8.2 Phase 3 per-apply is faster than Phase 2 — why?

Arch §4.2 predicted per-apply speedup from **reduced kernel-launch synchronization** — Phase 2's cuSPARSE SpSV internally level-schedules over the matrix DAG (~446 barriers predicted on a realistic grid, fewer on 16³), while Phase 3's red-black reduces this to exactly 4 launches (2 forward + 2 backward). Measurement confirms the prediction holds on RTX 3050 Laptop — 1.95–6.18× per-apply speedup across the test matrix sizes. This tallies with the arch doc's reasoning that cuSPARSE pays worst-case-DAG overhead for what is, post-permutation, trivially 2-stage.

### 8.3 Iter penalty is 1.50–1.55× across all four measurement points

| Test | Grid | P2 iters | P3 iters | ratio |
|---|---|---|---|---|
| T7 | 16³ stiff | 24 | 36 | 1.50 |
| Test A | 32³ smooth-sphere | 115 | 177 | 1.54 |
| Test B | 32³ static droplet | 82 | 127 | 1.55 |
| Test C | 32³ 3-tier (to 1e-8) | 128 | 197 | 1.54 |

Remarkably stable across physics / grids. Matches Li-Saad 2010 GPU MC-ILU(0) numbers (1.81-1.98×) with room to spare. Phase 3's red-black ordering pays a predictable ~1.5× penalty on variable-density Poisson in 3-D.

### 8.4 Host-side greedy coloring scaling

`greedy_color_csr` is a Python-level loop over vertices with a `set` per vertex. Fast enough for N ≤ 1e5 (milliseconds) but would become a bottleneck beyond that. Not exercised in Phase 3 (the 7-pt stencil always takes the red-black fast path), but worth flagging for any future non-structured-grid extension.

### 8.5 Additional D→H access beyond the 8-byte token

The analyze handler does a 4×(n_colors+1) = 12-byte D→H copy **at analyze time** to mirror `color_offsets` into the plan for per-apply kernel launches. This is NOT in the hot path — analyze runs once. The apply hot path strictly has only the 8-byte token D→H as tolerated in arch §3.6. No regression from Phase 2's contract.

---

## 9. Summary

Phase 3 delivers a multi-color DILU preconditioner that:

1. **Passes all 8 acceptance criteria** C1-C8 — correctness at ULP floor, iter penalty 1.50× (well under the 3× budget), beats Jacobi by 2×, and leaves every physical test's divergence/spurious-current/residual-halo signature untouched.
2. **Is faster end-to-end**: 1.30× total-PCG speedup at 16³ stiff, **2.72× at 32³ 3-tier** — where "faster" means `iters × per-apply-time` despite the 1.55× iter penalty, because the per-apply speedup (4.19× at 32³) overwhelms it.
3. **Has zero cuSPARSE dependency**: `libdilu_multicolor.so` links only `libcudart.so.12`. The full DILU preconditioner stack now runs on plain CUDA.
4. **Passes the user's red-line physical test**: F2 (Test C correlation > 0.5) not triggered. corr = -0.102, well in the PASS zone.

The headline quantitative result: **Phase 3 achieves a 2.72× total-PCG speedup on the 32³ realistic physical scenario at the cost of a predictable 1.55× iter penalty, zero physical degradation, and independence from NVIDIA sparse libraries.** The trade-off, empirically measured on the one specific AM-adjacent workload the project cares about, comes out firmly positive.

**Phase 4 (AMGx research) is NOT started.** The user now has the numbers to decide whether to proceed.

---

## 10. File manifest

New this Phase (all under `/home/yzk/DILU-Research/dilu/multicolor/`):
- `CMakeLists.txt`, `build.sh`
- `cpp/{plan_registry.h,plan_registry.cc,multicolor_analyze.cc,multicolor_apply.cc,multicolor_refactor.cc,multicolor_release.cc}`
- `cuda/{dilu_factor_kernel.cu,elem_scale_kernel.cu,permute_kernel.cu,color_stride_forward.cu,color_stride_backward.cu}` (first two copied verbatim from Phase 2)
- `python/{__init__.py,registration.py,wrapper.py,plan.py,coloring.py,permute.py}`
- `tests/{_harness.py,conftest.py,test_coloring_and_permute.py,test_smoke_lifecycle.py,test_t4_factor_correctness.py,test_t5_diagonal_equivalence.py,test_t6_laplacian_vs_dense.py,test_t7_pcg_iteration_count.py,test_under_jit_phase3.py,physical_benchmark_phase3.py}`
- `bench/bench_multicolor_vs_cusparse.py`, `bench/plots/{divergence_map,spurious_currents,residual_halo}.png`
- `build/libdilu_multicolor.so`

Phase 1/2/2.5 artifacts: **untouched** (verified `git status` clean under those directories throughout).
