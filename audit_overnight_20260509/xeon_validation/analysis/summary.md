# Xeon Validation Summary

results_dir: `/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results`
expected_template: `/home/yzk/DILU-Research/audit_overnight_20260509/expected_results_template.json`

---


## E01
  NOT RUN

## E02
  N reps: 6
  wall: mean=4.43s median=4.22s std=0.72s
  iter: mean=580
  resid: max=1.23e-08

## E03
  N reps: 1
  total wall: mean=26.90s median=26.90s
  step 0 (setup): 3883ms
  steps 1..N (amortized): mean=4282ms

## E04
  N reps: 1
  total wall: mean=16.19s median=16.19s
  step 0 (setup): 4513ms
  steps 1..N (amortized): mean=1960ms

## E05
  NOT RUN

## E06
  NOT RUN

## E07
  NOT RUN

## E08

## E09
  NOT RUN

---
## Comparison to expected


### E01 / approx_per_iter_ms (mean wall_per_step / mean total_iter_per_step × 1000)
  Predicted: 110 ∈ [50, 200]
  Rationale: PROJECT_STATUS_REPORT used 110 ms/iter (5s/step ÷ 45 iter/step). Layered estimation has 30% uncertainty per factor. Plausible range 50-200.

### E02 / wall_seconds_mean (AMGx fresh single-shot, 500K LPBF pd, RTX 3050 dev OR RTX 5060 lab)
  Predicted: 20 ∈ [5, 50]
  Rationale: Existing npz meta shows 9-39s on dev RTX 3050. Lab 5060 may be 2-3x faster (8GB vs 4GB, sm_120 vs sm_86). Range 5-50.

### E03 / step_1_to_N_mean_ms_amortized (excluding step 0 full setup)
  Predicted: 800 ∈ [50, 5000]
  Rationale: Hypothesis: update_coefficients ~ 50ms + solve ~ 100-1000ms. If solve dominates and unchanged from fresh, amortized doesn't help much. Predict update saves 1-3s setup overhead per step but solve still 5-20s.

### E03 / speedup_amortized_vs_fresh (total wall ratio)
  Predicted: 1.5 ∈ [1.0, 5.0]
  Rationale: If setup is 2-3s and solve dominates at 5-20s per step, amortized saves only setup × (N-1) = ~10s out of N×solve = ~50-100s ⇒ 10-20% speedup. Far below the asserted 100-1000×.

### E04 / iter_savings_warm_vs_cold_pct
  Predicted: 30 ∈ [0, 70]
  Rationale: LPBF physics has rapid alpha jumps; x_t may differ substantially from x_{t-1}. Warm-start could save 0-70% iter — large uncertainty.

### E04 / similarity_x_t_vs_x_prev_max (rel ‖∞)
  Predicted: 0.05 ∈ [0.001, 0.5]
  Rationale: x evolves with melt pool; rel diff between adjacent timesteps probably 0.1-50%.

### E05 / factor_seconds_mean (CHOLMOD fresh, 500K)
  Predicted: 70 ∈ [40, 150]
  Rationale: Conversation history showed 69-75s. Predict similar ±2x for cold-cache variance.

### E06 / speedup_symbolic_reuse_vs_fresh (total wall ratio)
  Predicted: 1.5 ∈ [0.8, 5.0]
  Rationale: Symbolic part of cholesky is ~30-50% of factor cost; reuse saves that ⇒ 1.4-2× speedup. May fail (numerical issue with near-singular matrix) ⇒ ratio ≈ 1.

### E07 / max_rel_max_AMGx_e12_IR_vs_LU (across 6 timesteps)
  Predicted: 1.13e-11 ∈ [1e-13, 1e-09]
  Rationale: Conversation history showed 1.13e-11. Range allows for numerical variance.

### E07 / max_rel_max_OF_vs_LU (across 6 timesteps)
  Predicted: 3.5e-05 ∈ [1e-06, 0.001]
  Rationale: OF tol=1e-8 × κ ~ 10^3-10^5 typical → 1e-5 to 1e-3. Conversation showed 3.47e-5.

### E09 / kappa_lab32_melt_380ns
  Predicted: 320000000000000.0 ∈ [10000000000.0, 1e+16]
  Rationale: Tonight's audit measured 3.2e14 via Lanczos shift-invert. Re-verify on Xeon with possibly different LAPACK/BLAS — expect within 1-2 orders of magnitude.

### E09 / kappa_single_track_melt_380ns
  Predicted: 10000000000.0 ∈ [1000000.0, 1e+16]
  Rationale: Different matrix from lab32 (rays>0 ≠ rays=0); expect lower κ since b is non-trivial. Wide range — true unknown.