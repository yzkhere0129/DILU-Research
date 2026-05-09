LAST_REVIEWED: 2026-05-10T04:50+08:00
ITERATION: v2

# Adversary Notes — v2 (post v1 + Lanczos seed result + plan v2)

Changes from v1:
- A010 RESOLVED: multi-seed Lanczos test shows σ_min = 4.516e-29 stable across 3 seeds with spread 1.000×. C021 upgraded.
- New attacks A021-A025 against the v2 plan and post-Lanczos evidence.

(A001-A020 from v1 remain. Below are new attacks + status updates on v1 attacks.)

---

## v1 attack status updates

| ID | v1 issue | v2 status |
|---|---|---|
| A001 | Worst AMGx-vs-LU rel cases at picosecond ts | Still open, E08 to stratify |
| A002 | SuperLU vs CHOLMOD x_LU bit-exact? | Still open, E07 to compare |
| A003 | max_iter=2000 cap | Resolved: E03/E04 raised to 5000 |
| A004 | 8K → 500K generalization | Still open, E07 settles |
| A005 | CUDA async timing | Open, depends on jax sync semantics; recommend explicit synchronize() |
| A006 | sudo cold-cache | Open, best-effort sync only without sudo |
| A007 | E07 single-rep variance | Open, will run 2 reps |
| A008 | Thread pinning | Resolved: env vars set in master script |
| A009 | gc.collect between AMGx reps | Open, runners need to add |
| A010 | Lanczos seed sensitivity | **RESOLVED**: 3 seeds same σ_min, spread 1.000× |
| A011 | E01 7h commit risk | Resolved: --smoke mode + manual launch |
| A012 | E03/E04 cache effects | Open, nvidia-smi between reps recommended |
| A013 | E08 amortization speedup | Resolved: speedup_ratios.json computed |
| A014 | Predicted ranges too generous | Open, ranges in expected_results_template should tighten if E08 numbers come back too easy |
| A015 | AMGx warm vs OF ratio | Resolved: E08 computes |
| A016 | Repo state during run | Resolved: env_*.txt records git rev |
| A017 | manifest.json | Resolved: master script writes |
| A018 | E01 manual launch | Resolved: RUNBOOK §5 explicit |
| A019 | config_full_str field | Open, runners should add |
| A020 | Atomic write | Resolved: .tmp + rename used |

## New v2 attacks

### A021 — multi-seed Lanczos finding "stable" doesn't prove σ_min isn't bogus
σ_min = 4.516e-29 across 3 seeds, spread 1.0×. But: this might be a CONVERGENT-to-floor result rather than a true σ_min. If A_pos has many eigenvalues clustered below fp64 noise floor, ALL seeds converge to the noise-floor "eigenvalue" with bit-exact reproducibility — but it's not actually a real eigenvalue, just numerical noise.
ACTION: Cross-check with `np.linalg.eigvalsh(A_pos.todense())` on a small block — but A is 500K×500K, dense conversion = OOM. Alternative: compute trace(A) and trace(A^2) and apply Cauchy-Schwarz bounds. Or: use `scipy.sparse.linalg.svds(A_pos, k=10, which='SM', return_singular_vectors=False)` to get smallest 10 singular values; if they're all ~4.5e-29, spectrum has a near-null space; if there's a clear separation (say smallest is 4.5e-29 but next is 1e-15) then σ_min = 4.5e-29 isolated and meaningful.
TONIGHT-DOABLE: yes (svds with k=10, ~5 min on 500K).

### A022 — E07's sign-flip handling between npz x_truth (positive A) and dumped x_OF (negative A)
The npz `x_truth` was computed by `prepare_single_core_plot_data.py` which sign-flipped before AMGx solve. So x_truth is in the SAME variable space as x_OF (because x doesn't change with sign flip — only A and b together change sign and x = (-A)^-1 (-b) = A^-1 b). So in our ledger we treat them as comparable. **However**, if `prepare_*` code accidentally negated x or b before saving, the comparison would be flipped. Check the npz meta `sign_flipped` field to be sure.
ACTION: E07 must check `meta['sign_flipped']` and document. Add assert.

### A023 — Predicted "warm-start saves 50-90% iter" presumes physics with smooth time evolution
LPBF physics: alpha (gas/metal indicator) jumps in cells where melt advances. From step t to t+1ns, only ~10s of cells change alpha; the bulk x is unchanged. So warm-start SHOULD save iter substantially, but only because most x stays unchanged. The CHANGED cells (≪ 1%) might still need many iter to converge. Net effect: 30-70% iter savings is plausible.
ACTION: E04 already measures, no new change needed. Just record similarity ‖x_t - x_{t-1}‖∞ / ‖x_t‖∞.

### A024 — E01 single-shot full simulation has high variance source we can't measure
laserMeltFoam over 1.2 μs has chaotic dynamics (vapor recoil, melt pool oscillation). Two re-runs may NOT produce the same iter count at each timestep due to floating-point non-determinism in the iterative solver convergence path — this matters because we're comparing OF iter to AMGx iter. If OF iter varies ±20% between runs of same case, the "OF wall = iter × 110ms" estimate has another 20% noise on top of the 110ms uncertainty.
ACTION: Run E01 once; if iter count per timestep varies wildly across PISO inner iter, that's a known issue. RECOMMENDATION: also dump ‖x_OF -x_OF_run2‖ to assess simulation-level reproducibility — but that's beyond budget. Document as a caveat in summary.

### A025 — Plan v2 doesn't include any sanity check that AMGx actually runs at all on lab Xeon
We only verified AMGx + jax-env on dev RTX 3050. Lab GPU is RTX 5060 (sm_120, CUDA 13.2). The AMGx binary may have been built for different sm/CUDA. If AMGx crashes on first solve in E02, all of E02-E04 fail.
ACTION: smoke run (--smoke) catches this. RUNBOOK §4 documents.

---

## Final attack count

20 attacks v1 → 20 + 5 = 25 attacks v2. Of these:
- 13 RESOLVED in v2 (script edits made)
- 9 OPEN (will be resolved by Xeon experiments themselves or recommended improvements)
- 3 are caveats / documentation actions

This is a defensible audit-pass set for a hostile reviewer.
