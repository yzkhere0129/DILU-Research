LAST_REVIEWED: 2026-05-10T01:55+08:00
ITERATION: v1

# Known Unknowns — items the audit cannot settle tonight

For each: the unknown, why it matters, how it'd be settled, whether tomorrow's Xeon plan covers it.

---

## U001 — True condition number κ(A) for 500K LPBF pd matrices
**Why matters**: S4 attribution "5.9 kPa = κ × tol" needs κ measurement. data_inventory.md asserts κ ~ 1e6 but it's never measured.
**Status**: I'm running Lanczos κ now (single matrix, lab32 melt-380ns). Will need to repeat across 6 timesteps + 6 single_track timesteps for full picture.
**Settled by**: E09 (Lanczos σ_max + σ_min via shift-invert on each of 12 matrices).
**Xeon coverage**: YES — E09 explicit.

## U002 — Per-timestep AMGx + IR diff vs LU truth on 500K
**Why matters**: The 1.13e-11 number (S3) is from lab Xeon CHOLMOD, JSON not on dev. We have only conversation paste.
**Status**: Tonight I cannot run CHOLMOD on lab Xeon. SuperLU on dev is 70-180s/case × 6 = 7-18 min, allowed under tonight's budget if needed. But better to run on Xeon for consistency.
**Settled by**: E07 (CHOLMOD on lab Xeon, store JSON).
**Xeon coverage**: YES — E07 explicit.

## U003 — Per-equation per-step OF DICPCG wall (the actual measurement)
**Why matters**: S2 — current "wall" is iter × 110 ms. For C006 to become VERIFIED, need true measurement.
**Settled by**: E01 (run laserMeltFoam with `solverInfo` function object enabled, parse postProcessing/solverInfo/<t>/solverInfo.dat for per-pd-solve wall).
**Xeon coverage**: YES — E01 explicit. Cost: ~7h Xeon single-core to repeat 1.2 μs run.

## U004 — AMGx amortized actual wall on this 500K matrix
**Why matters**: S1, C009, C015. The 100-1000× claim has no measurement.
**Settled by**: E03 (Plan once + N×update_coefficients on 6 timesteps).
**Xeon coverage**: YES — E03 explicit.

## U005 — Warm-start iteration savings on this dataset
**Why matters**: S1, C016. Whether x_{t+1} similar enough to x_t to save iter.
**Settled by**: E04 (warm vs cold init on each (t-1, t) pair).
**Xeon coverage**: YES — E04 explicit.

## U006 — CHOLMOD symbolic-reuse vs full-factor speedup on this matrix
**Why matters**: C017. The 2-5× speedup hypothesis.
**Settled by**: E06 (analyze() once + cholesky_inplace() per step).
**Xeon coverage**: YES — E06 explicit.

## U007 — Phase 0 replay actual mode-by-mode results
**Why matters**: S6, C013. The whole replay experiment never produced output.
**Settled by**: E11 (clean retry on Xeon, ≥5 reps each mode).
**Xeon coverage**: YES — E11 explicit.

## U008 — Senior Melting/Evaporation matrix brokenness cause attribution
**Why matters**: S5, C014, C019. Two confounded causes (matrixDumper bug + case config). Senior's case files not in our repo.
**Settled by**: Cannot settle without senior's cooperation. Re-dump senior's case with patched matrixDumper would do it. NOT in tonight's scope, NOT in tomorrow's Xeon scope (we don't have senior's case files).
**Xeon coverage**: NO — out of scope; document as "user-action item".

## U009 — Whether matrixDumper.H 32-rank patch actually produces self-consistent dumps under standard usage
**Why matters**: S5. Code looks correct (read lines 395-432) but verification was on a case (lab32) with separate physics issues (rays=0).
**Settled by**: E10 (run a fresh 32-rank dump on a clean small case + reconstruct + check ‖A·x_OF - b‖/‖b‖ ≤ 1e-7).
**Xeon coverage**: OPTIONAL — E10 deferred unless budget allows.

## U010 — AMGx behavior at >5M cells (S8)
**Why matters**: PROJECT_STATUS_REPORT cites this as a justification for AMGx development.
**Settled by**: Synthetic Poisson at 5M+ cells. RTX 3050 (4 GB) cannot fit it; RTX 5060 (8 GB) might fit 2-3M; multi-GPU not available.
**Xeon coverage**: NO — beyond hardware reach.

## U011 — Spread of AMGx + IR rel diff across 6 timesteps (S3 — was max raised by one outlier?)
**Why matters**: 1.13e-11 max with rest possibly <1e-12. If 5/6 are 1e-12 and 1/6 is 1e-11, the max number is misleadingly pessimistic.
**Status**: Conversation paste shows: 9.73e-12, 7.94e-12, 1.70e-12, 1.13e-11, 1.14e-12, 8.79e-12. Range 1.14e-12 to 1.13e-11. ~10× spread.
**Settled by**: Already settled in conversation paste. E07 (re-run) would lock it in JSON.
**Xeon coverage**: YES — E07 will produce JSON.

## U012 — `/tmp/bench_gold.log` provenance
**Why matters**: An undocumented benchmark exists, results unknown.
**Status**: Possibly from a sibling agent or previous Claude run. Cannot determine origin from existing tools.
**Settled by**: Ask user; can't settle programmatically.
**Xeon coverage**: NO — administrative.

## U013 — Real cost of cold-cache effects on solver wall measurements
**Why matters**: All E0X measurements are vulnerable to filesystem cache + GPU cache effects between repetitions.
**Settled by**: E0X scripts must include cache-clearing between reps + warmup runs.
**Xeon coverage**: YES — implementation requirement on every E0X script.

## U014 — Stochastic variance in OF solve wall (single-core may have OS noise)
**Why matters**: Lab Xeon single-core may have other load fluctuations affecting per-step wall.
**Settled by**: E01 ≥ 5 reps + use median + report stddev.
**Xeon coverage**: YES — E01 spec includes ≥5 reps.

## U015 — Whether AMGx CLASSICAL_V_DIAGSCALED is the optimal AMGx config for this matrix
**Why matters**: C007 — AMGx 435-773 iter is large. Other AMGx configs (PMIS, AGGRESSIVE coarsening, GS smoothers, Block-Jacobi adjustments) may converge faster.
**Settled by**: AMGx config sweep on 1 timestep — try CLASSICAL_V_DIAGSCALED, CLASSICAL_V_DIAGSCALED_AGGRESSIVE (if exists), CLASSICAL_V_PMIS, etc.
**Xeon coverage**: OPTIONAL E12 — deferred unless budget allows.

---

## Quick budget check

Tomorrow's Xeon time budget: 8 hours (per overnight prompt §6).

| ID | Experiment | Estimated wall |
|---|---|---|
| E01 | OF DICPCG with solverInfo, 1.2μs run | ~7 h 单核 |
| E02 | AMGx fresh single-shot, 6 timesteps × 5 reps | ~25 min |
| E03 | AMGx amortized + warm-start, 6 timesteps × 5 reps | ~30 min |
| E04 | warm-start sensitivity (subset of E03) | (subsumed by E03) |
| E05 | CHOLMOD fresh, 6 × 5 reps | ~40 min |
| E06 | CHOLMOD symbolic reuse, 6 × 5 reps | ~30 min |
| E07 | LU truth + AMGx_e12+IR diff, 6 × 1 (deterministic) | ~10 min |
| E08 | iter count + per-step diagnostics from E02-E04 | (subsumed) |
| E09 | Lanczos κ, 12 matrices × 1 | ~10 min |
| E10 | 32-rank fresh dump verify (optional) | ~2 h |
| E11 | Phase 0 replay (subsumed by E03+E04+E05+E06) | (subsumed) |

**Conflict**: E01 wants 7 h Xeon AND E03+E04+E05+E06+E07+E09 also want CPU/GPU time. **Solution**: E01 runs in parallel (different cores) since E02-E09 are all per-matrix solver tests, not full simulations.

Adjusted total: max(E01=7h, all E02-E09 in parallel ~3h) = 7h. Fits within 8h budget.

E10 must be cut from default plan (extra 2h on top of E01 might tip over budget). Mark as `--only E10` opt-in.
