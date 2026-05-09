LAST_REVIEWED: 2026-05-10T02:30+08:00
ITERATION: v1

# Xeon Validation Plan — single-shot to settle C001-C021

Total budget: 8 hours wall on lab Xeon (HR54WV2).

## Strategy

E01 (laserMeltFoam re-run with solverInfo) takes ~7 h on its own — it can run in nohup background.
E02-E09 are matrix-replay experiments using the existing dumped 6 timesteps × pd_corr0 — fast (~2-3 h serially, much less if 1 GPU + N CPU cores in parallel).

So: **E01 in background + E02-E09 in foreground**. Total wall ≈ 7-8 h.

E10, E12 OPTIONAL — only run if E01-E09 complete with budget left.

## Per-experiment specs

---

## E01 — OF DICPCG actual per-solve wall (settles C006, S2)

TARGETS: C002, C005, C006, S2.

INPUT: ~/cases/single_track_dump (case dir on lab Xeon).

PROCEDURE:
1. Add `solverInfo` function object to `system/controlDict` (writes per-solve wall to `postProcessing/solverInfo/<t>/solverInfo.dat`).
2. Backup current dump dir.
3. Re-run laserMeltFoam in nohup for 1.2 μs.
4. Parse solverInfo.dat for per-pd-solve wall.
5. Compare to estimated wall = iter × 110 ms.

MEASURE:
- per-pd-solve wall_seconds (from solverInfo.dat or via `time` wrapper)
- per-pd-solve iter_count
- ratio wall/iter
- wall stddev across PISO inner iterations

REPETITIONS: 1 full simulation run (≥234 pd_corr0 solves recorded).

STOP_CRITERION: simulation reaches t=1.2 μs OR mean per-pd-solve wall stable across last 50 steps.

EXPECTED_IF_HYPOTHESIS_HOLDS (current 110 ms/iter estimate roughly right):
- per-pd-solve wall ∈ [50, 250] ms × iter (i.e., 5–250 ms on single-iter solves up to a few seconds on 65-iter solves)
- ratio mean wall/iter ∈ [50, 200] ms

EXPECTED_IF_HYPOTHESIS_FAILS:
- ratio wildly different (e.g., < 5 ms/iter or > 500 ms/iter) ⇒ rewrite C006

RUNTIME_BUDGET: 7 h Xeon single-core (background nohup).

OUTPUT: `xeon_validation/results/E01/01/result.json` (only 1 rep — single full sim).

---

## E02 — AMGx PCG cold-start single-shot wall (settles C008 baseline)

TARGETS: C007, C008.

INPUT: 6 single_track npz files (`single_*pd_corr0_*.npz`) — locally cached A, b on lab Xeon.

PROCEDURE: For each timestep t, ≥5 reps:
1. cold-cache (sync; sleep 3)
2. start timer
3. AMGx Plan(rp, ci, vv, cfg) — fresh setup
4. plan.solve(b, x0=zeros)
5. stop timer
6. record setup_s, solve_s, iter, rel_resid

MEASURE: wall_s, setup_s, solve_s, iter, rel_resid_actual.

REPETITIONS: 5 per timestep × 6 timesteps = 30.

STOP_CRITERION: All reps complete OR median stable.

EXPECTED_IF_HYPOTHESIS_HOLDS (C008): per-solve wall ≈ 9-39 s on RTX 5060 (lab GPU is faster than dev RTX 3050, so possibly 5-25 s).

EXPECTED_IF_HYPOTHESIS_FAILS: per-solve wall < 1 s ⇒ AMGx setup is much cheaper than thought (good for amortized story); per-solve > 60 s ⇒ matrix is harder than we thought.

RUNTIME_BUDGET: 30 reps × ~30 s = 15 min.

---

## E03 — AMGx amortized wall (Plan once, update_coefficients per step) (settles S1, C015)

TARGETS: C009, C015.

INPUT: 6 single_track npz, replayed in time order.

PROCEDURE:
1. cold-cache
2. start timer
3. plan = Plan(matrices[0]) — full AMG hierarchy build
4. record setup_ms_first
5. plan.solve(b[0], x0=zeros) — first solve
6. record solve_ms_first
7. for i in 1..5:
   - record start_ts
   - plan.update_coefficients(matrices[i].data)
   - record update_ms[i]
   - plan.solve(b[i], x0=zeros)  # cold init for E03
   - record solve_ms[i]
   - record rel_resid[i]
   - end_ts; total_ms[i] = end - start_ts

REPETITIONS: 5 (each is an entire 6-step sequence).

STOP_CRITERION: completion.

EXPECTED_IF_HYPOTHESIS_HOLDS (C015):
- setup_ms_first ∈ [1500, 5000] ms
- update_ms_per_step ∈ [30, 200] ms
- solve_ms ≈ same as fresh AMGx (because cold init), ~5-25 s
- speedup vs fresh = (setup + N×solve_fresh) / (setup + N×solve_amortized) — should be small (~1-3×) since solve dominates and is unchanged.

EXPECTED_IF_HYPOTHESIS_FAILS:
- update_ms > 5000 ms (close to full setup) ⇒ amortization not effective
- iter rises (numerical reuse of hierarchy doesn't fit new matrix) ⇒ iter increases by >50%

RUNTIME_BUDGET: 5 reps × (1×3s setup + 6×30s solve) = 5 × 183 s = 15 min.

---

## E04 — AMGx amortized + warm-start (settles S1, C016)

TARGETS: C009, C016.

INPUT: same as E03.

PROCEDURE: same as E03 but solve uses x0 = x_prev (the previous step's solution).

ADDITIONAL MEASURE: 
- ‖x_t - x_{t-1}‖∞ / ‖x_t‖∞ (similarity metric)
- iter[t] under cold-start (zeros) vs warm (x_{t-1})
- iter saving = 1 - iter_warm / iter_cold

REPETITIONS: 5 reps × 6-step sequence each.

STOP_CRITERION: completion.

EXPECTED_IF_HYPOTHESIS_HOLDS (C016):
- iter saving ∈ [50%, 90%] ⇒ warm-start drops iter dramatically
- similarity ‖x_t - x_{t-1}‖/‖x_t‖ ∈ [1e-3, 1e-1]

EXPECTED_IF_HYPOTHESIS_FAILS:
- iter saving < 20% ⇒ warm-start not useful for this physics (LPBF transients too fast)
- similarity > 1 ⇒ x_{t-1} not a useful initial guess

RUNTIME_BUDGET: 5 reps × ~150 s (faster than E03 due to fewer iter) = 13 min.

---

## E05 — CHOLMOD fresh refactor wall (settles C011)

TARGETS: C011.

INPUT: 6 single_track A.mm, b.mm.

PROCEDURE: For each timestep t, ≥5 reps:
1. cold-cache
2. start timer
3. factor = sksparse.cholmod.cholesky(A.tocsc())
4. record factor_s
5. x = factor(b)
6. record solve_s
7. record ‖A·x - b‖/‖b‖

REPETITIONS: 5 × 6 = 30.

STOP_CRITERION: completion.

EXPECTED_IF_HYPOTHESIS_HOLDS (C011): factor_s ∈ [50, 80] s on lab Xeon CHOLMOD; solve_s < 0.5 s; rel_resid ~ 1e-15.

EXPECTED_IF_HYPOTHESIS_FAILS: factor_s either drastically different (memory issues) or rel_resid >> 1e-12 (numerical issues).

RUNTIME_BUDGET: 30 reps × 70 s = 35 min.

---

## E06 — CHOLMOD symbolic-reuse (settles C017)

TARGETS: C017.

INPUT: 6 single_track A.mm, b.mm.

PROCEDURE:
1. factor = sksparse.cholmod.cholesky(A_first.tocsc())  # full first
2. record factor_s_first
3. for i in 1..5:
   - factor.cholesky_inplace(A_i.tocsc())  # reuses symbolic
   - record refact_s[i]
   - x = factor(b_i)
   - record solve_s[i]
   - record rel_resid[i]

REPETITIONS: 5 (each is full 6-step sequence).

STOP_CRITERION: completion.

EXPECTED_IF_HYPOTHESIS_HOLDS (C017): refact_s / factor_s_first ∈ [0.2, 0.6] (i.e. 1.7-5× speedup).

EXPECTED_IF_HYPOTHESIS_FAILS: refact_s ≈ factor_s_first ⇒ no benefit; or refact fails (numerical) and forces full factor anyway.

RUNTIME_BUDGET: 5 × (70s + 5×40s) = 5 × 270s = 22 min.

---

## E07 — AMGx + 1 IR vs CHOLMOD LU per-timestep diff (settles C004, S3)

TARGETS: C004, S3, U002, U011.

INPUT: 6 single_track npz with x_AMGx_e12 (= AMGx tol=1e-12 + 1 IR), 6 A,b matrices.

PROCEDURE: For each timestep:
1. Compute x_LU = CHOLMOD(A) × b
2. Verify ‖A·x_LU - b‖/‖b‖ < 1e-13 (sanity)
3. Compute diff = x_AMGx_e12 - x_LU
4. Record max|diff|, ‖diff‖₂, max|diff|/‖x_LU‖∞, ‖diff‖₂/‖x_LU‖₂

REPETITIONS: 1 (deterministic — diff doesn't change with reps).

STOP_CRITERION: 6 timesteps complete.

EXPECTED_IF_HYPOTHESIS_HOLDS (C004, S3): per-timestep max diff/‖x‖∞ ≤ 2e-11; full set: ≤ 1.13e-11 (matching the conversation paste).

EXPECTED_IF_HYPOTHESIS_FAILS: any timestep > 1e-9 ⇒ AMGx + IR has a bug, OR matrix is harder than expected, OR sign-flip wrong.

RUNTIME_BUDGET: 6 timesteps × 70s CHOLMOD = 7 min.

---

## E08 — Per-timestep iter + residual diagnostics (subsumed by E02/E03/E04/E05)

TARGETS: C005, C007, S7.

These data points are already captured by the per-step JSON in E02-E04 + E05-E06. E08 is a parsing step:
1. Walk all xeon_validation/results/E0X/*/result.json
2. Extract iter, residual, wall vectors
3. Plot iter vs timestep_idx for each solver
4. Plot residual vs timestep_idx
5. Save analysis output

REPETITIONS: post-processing step, no solver runs.

OUTPUT: xeon_validation/analysis/iter_residual_diagnostics.csv + .png.

RUNTIME_BUDGET: < 1 min (post-processing).

---

## E09 — Lanczos κ on 6+1 = 7 matrices (settles C012, C021)

TARGETS: C012, C021.

INPUT: 6 single_track A matrices + 1 lab32 (already done tonight, 3.2e+14 — re-verify).

PROCEDURE: For each matrix:
1. Sign-flip to A_pos = -A.
2. eigsh(A_pos, k=1, which='LA', tol=1e-3) → σ_max
3. eigsh(A_pos, k=1, sigma=0, which='LM', tol=1e-3) → σ_min via shift-invert
4. κ = σ_max / σ_min

REPETITIONS: 1 (deterministic).

STOP_CRITERION: 7 done.

EXPECTED_IF_HYPOTHESIS_HOLDS: κ ∈ [1e10, 1e15] on most matrices; 1e6 cited in PROJECT_STATUS_REPORT ⇒ off by 4-9 orders of magnitude.

EXPECTED_IF_HYPOTHESIS_FAILS (i.e. κ really near 1e6): ⇒ revisit our lab32 measurement.

RUNTIME_BUDGET: 7 matrices × 200 s (shift-invert is slow) = 23 min.

---

## E10 (OPTIONAL) — fresh 32-rank dump verify matrixDumper patch on a clean case

TARGETS: C014.

INPUT: copy of single_track_dump (or smaller test case).

PROCEDURE: 
1. Decompose to 32 ranks.
2. mpirun -np 32 laserMeltFoam (short 50ns smoke).
3. Reconstruct global A, b, x_final.
4. Verify ‖A·x_OF - b‖/‖b‖ ≤ 1e-7.

RUNTIME_BUDGET: ~2 h (decompose + smoke + reconstruct).

DEFAULT: skipped unless user enables `--only E10`.

---

## E11 — Phase 0 replay clean retry (settles S6, C013, all replay claims)

TARGETS: S6, C013.

INPUT: 6 single_track npz.

PROCEDURE: This is essentially **bundling** E02 + E03 + E04 + E05 + E06 into a comparable replay test. The Phase 0 replay was supposed to do all 5 modes; we now split them into separate experiments E02-E06 with proper rep counts.

E11 = E02 + E03 + E04 + E05 + E06 reported under one umbrella in the analysis step (E08).

RUNTIME_BUDGET: subsumed.

---

## Master schedule

```
Hour 0    +0:00  E01 starts (background nohup, 7h)
                 E02 starts (foreground, AMGx fresh, ~15 min)
Hour 0:15        E02 done; E03 starts (AMGx amortized, ~15 min)
Hour 0:30        E03 done; E04 starts (AMGx warm, ~13 min)
Hour 0:45        E04 done; E05 starts (CHOLMOD fresh, ~35 min)
Hour 1:20        E05 done; E06 starts (CHOLMOD symbolic, ~22 min)
Hour 1:40        E06 done; E07 starts (LU vs AMGx_IR diff, ~7 min)
Hour 1:50        E07 done; E08 (analysis, ~1 min)
Hour 1:55        E08 done; E09 (Lanczos, ~23 min)
Hour 2:20        E09 done; idle until E01 finishes
Hour 7:00        E01 done (single-core sim ~7h)
Hour 7:00        E11 analysis combines all results
Hour 7:30        compare_to_expected.py runs, summary.md generated
```

If E01 takes longer than 7h, E08-E09 still work. If E01 fails, partial data still useful.
