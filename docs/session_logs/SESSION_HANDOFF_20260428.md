# SESSION_HANDOFF_20260428 — OpenFOAM Cross-Check Complete (v1.0 → v3.2)

## State at time of writing

OpenFOAM `laserMeltFoam` cross-check fully complete on the 1M-cell spot melt
case. 150 matrices (50 pd + 100 T) tested across 4 GPU solver paths, all 100%
converged with correct physics. AMGx with DIAGONAL_SYMMETRIC scaling beats
single-thread OpenFOAM by 8.6× (pd) / 2.8× (T) at higher precision (270×-1000×).

This handoff summarizes the multi-day arc from v1.0 (single matrix smoke test)
to v3.2 (full pd+T benchmark with multi-case deploy bundle ready).

## Canonical artifacts (do not recreate)

- **Method spec**: `docs/specs/MATRIX_EXTRACTION_SPEC.md`
- **Canonical benchmark report**: `docs/benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md`
- **Deploy bundle for lab machine**: `dilu/benchmark/openfoam_crosscheck/deploy.tar.gz`
  (30 KB, 18 files: matrixDumper.H + 3 patches + install.sh + patch_case.sh +
   sanity.py + shrink_dump.py + multicase_batch.sh + README + TUTORIAL)
- **Data**: `dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/`
  - 50 pd npz + 100 T npz (1.85 GB total after npz compression from 80 GB ASCII)
  - per-matrix `results/{cusparse,multicolor,amgx_*}.json` with iter/time/rel_vs_OF

## Key technical findings

### Five independent failures and fixes (in order discovered)

| # | Failure | Root cause | Fix | Speed-up |
|---|---------|------------|-----|----------|
| 1 | cuSPARSE PCG 351 s/solve (vs 5 s on canonical case) | Python loop with `float(jnp.linalg.norm(r))` per iter forces GPU→CPU sync × 3/iter | Wrap loop in `jax.lax.while_loop` + `@jax.jit` (single dispatch) | 12.7× single-matrix; 129× total with trace cache |
| 2 | cuSPARSE PCG still 9.76 s after fix #1 | SpMV recomputes `row_of` from `searchsorted` every iter (7M nnz binary search) | Precompute `row_of = np.repeat(arange(n), diff(row_ptr))` once at solve entry, pass into jit closure | 3.1× cuSPARSE / 10.9× Multicolor |
| 3 | AMGx DIVERGES on LPBF pd matrix (5 different configs all hit 200 iter cap with rel_residual stuck at 1e-4) | Matrix is near-singular Neumann Laplacian + setReference (one cell diag doubled). `\|A·1\|_∞ ≈ 6.6e-14`, `λ_min ≈ 1e-13`, plus diag spans 14 orders of magnitude (1e-27 to 1e-13). Classical AMG hierarchy fails on near-singular operators with such heterogeneity | Add `"scaling": "DIAGONAL_SYMMETRIC"` to AMGx config — internally normalizes `A → D⁻¹ᐟ² A D⁻¹ᐟ²` (unit diag) | DIVERGES → 22 iter / 0.34 s / rel_vs_OF=2.5e-6 |
| 4 | AMGx amortization: each matrix re-runs `Plan(rp,ci,vv,cfg)` setup (~1.5 s) wasted | Same sparsity across timesteps; only values change | New `--amortize` mode in `driver_amgx.py` reuses one `AmgxPlan` and calls `update_coefficients(new_values)` per matrix | pd 1.18× / T 12 ms update vs 1.5 s setup |
| 5 | T equation (non-symmetric, advection-dominated) cannot use PCG | `solver: "PCG"` requires SPD; T has `upper != lower` from `fvm::div(rhoCpPhi, T)` | Add `"solver": "BICGSTAB"` outer config; write `driver_cusparse_bicgstab.py` and `driver_multicolor_bicgstab.py` (jit while_loop BiCGStab with same preconds) | Unlocks 100 T matrices |

### Wrong path that ALMOST looked right

`reg=1e-2` uniform ε·I regularization on AMGx for pd:
- Self-consistency `‖A·x − b‖/‖b‖ = 9.25e-11` (looks converged ✓)
- **vs OpenFOAM solution: rel_vs_OF = 1.00 (100% off!)** ❌

Reason: solving `(A + ε·I) x' = b` is a different problem from `A x = b` when
A is near-singular. Uniform ε·I locks the constant-vector nullspace in a
different direction than OpenFOAM's `setReference` does. Same residual, wrong
solution.

**Critical lesson**: always use `‖x_solver − x_OpenFOAM‖∞ / ‖x_OpenFOAM‖∞` as
the primary precision metric for benchmarking, not self-consistency residual.
This was the v2.0 → v2.1 correction.

## Final OpenFOAM-vs-our-solvers comparison

### pd equation (50 matrices)

| Solver | iter med | wallclock med | vs OpenFOAM | rel_vs_OF |
|--------|----------|---------------|-------------|-----------|
| OpenFOAM PCG-DIC (1 Xeon core) | 105 | 2.57 s | 1× | reference |
| our cuSPARSE DILU-PCG | 189 | 2.71 s | 0.95× | 4.4e-04 |
| our Multicolor DILU-PCG | 290 | 1.82 s | 1.41× | 5.5e-04 |
| our AMGx + DIAG_SYM amortized | 22 | **0.30 s** | **8.57×** ⭐ | 2.5e-06 |

### T equation (100 matrices)

| Solver | iter med | wallclock med | vs OpenFOAM | rel_vs_OF |
|--------|----------|---------------|-------------|-----------|
| OpenFOAM PBiCG-DILU (1 Xeon core) | ~5-10 | 0.14 s | 1× | reference |
| our cuSPARSE DILU-BiCGStab | 2 | 0.50 s | 0.28× | 3.6e-07 |
| our Multicolor DILU-BiCGStab | 3 | 0.46 s | 0.30× | 9.7e-08 |
| our AMGx BiCGStab + DIAG_SYM amortized | 7 | **0.05 s** | **2.80×** ⭐ | 1.0e-08 |

cuSPARSE/Multicolor T are slower because they don't yet amortize plan setup.
Each matrix repeats analyze + factor (~0.4 s) which dominates the 2-3 iter
solve. Adding amortization to those drivers (~1-2 h work) would bring T speed
to ~0.10 s/solve, beating OpenFOAM 1.4×.

## OpenFOAM real wallclock measurement

The v2.x estimate of 4 s/pd-solve (theoretical) was replaced in v3.0 by
direct measurement using `std::chrono` patches in `pEqn.H` and `TEqn.H`:

```cpp
const auto t_pd_start = std::chrono::steady_clock::now();
Foam::SolverPerformance<Foam::scalar> pdPerf = pdEqn.solve(...);
const auto t_pd_end = std::chrono::steady_clock::now();
Info<< "[TIMING_pd] "
    << std::chrono::duration<double, std::milli>(t_pd_end - t_pd_start).count()
    << " ms" << endl;
```

Then `grep "TIMING_pd" log.timing | awk '{print $2}'`. 5 timesteps × 3 pd
correctors = 15 samples, median 2.57 s. T median 141 ms.

(The OpenFOAM `profiling` functionObject was attempted first but did not
produce output in our v2506 install — fell through to the manual
`std::chrono` patch.)

## Outstanding work (not blocking)

| Item | Effort | Priority | Notes |
|------|--------|----------|-------|
| Multi-case to reach 1000+ matrices | 1 lab night per 4 cases × ~2-3 nights | High | `multicase_batch.sh` automation ready; user just edits `CASES=(...)` array on lab machine |
| cuSPARSE/Multicolor BiCGStab amortization | 1-2 h coding | Medium | Same template as `driver_amgx.py --amortize` |
| `update_coefficients` perf on pd (0.5 s vs 0.012 s on T) | 0.5-1 day | Low | AMGx BLOCK_JACOBI smoother state recomputation; try `coarse_solver_recompute` AMGx flag |

## What this session did NOT do

- Did not collect any data outside the spot melt case (other laser
  parameters / scan modes / materials still untested)
- Did not write a paper or slides — only the methodology and benchmark
  reports
- Did not test the deploy bundle on the lab machine end-to-end (handed to
  user; the local pipeline_test case validated all components in isolation)

## Next contributor reading order

1. `docs/PROJECT_SUMMARY.md` — overall project context
2. `docs/specs/MATRIX_EXTRACTION_SPEC.md` — how the toolchain works
3. `docs/benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md` — current results
4. `dilu/benchmark/openfoam_crosscheck/` source — drivers and aggregator
5. This handoff — context for any unexplained decisions

## Authority statement

This handoff is a **historical snapshot**. When numbers herein conflict with
`benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md` (canonical) or `specs/MATRIX_EXTRACTION_SPEC.md`,
those win. Do not update this handoff retroactively.
