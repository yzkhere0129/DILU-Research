LAST_REVIEWED: 2026-05-10T00:55+08:00
ITERATION: v1

# Pass 1 Reading Notes

Each entry: 3–5 sentences after reading. `[CITE]` = specific file:line backing.

---

## R01 — CLAUDE.md (project root)
Project says: "DILU solver low-level optimization research", under `dilu/`. VOF/PLIC code is frozen reference. Conventions: Python ≥ 3.10, no -ffast-math, np.ascontiguousarray at FFI. **Important**: project's stated focus is **solver-kernel research**, not CFD app. Recent work has drifted into CFD physics (LPBF case extraction). The drift is intentional (need real matrices) but the report writeup must acknowledge this isn't the original focus.

## R02 — PROJECT_STATUS_REPORT.md
Reads like a polished report but contains the 4 cited bugs (predictions presented as conclusions, estimates not segregated, refuted hypotheses not fully scrubbed, status-as-conclusion claims). Section 1.2 "Settled" claims should be re-examined: each one is a verbal retraction, but I haven't seen evidence that all derived materials (commit messages, fig captions, JSON metadata) were synchronized.

## R03 — solver_comparison_500K.md
**[CITE: solver_comparison_500K.md:18-78]** has 6 per-timestep tables. **[CITE: solver_comparison_500K.md:104]** explicitly admits OF wall is estimated. **[CITE: build_solver_comparison_table.py:104]** I forensically confirmed: `wall_ms = of_iter * 110`. Every OF wall value in the document = `iter × 110 ms`. Table headers do not visually distinguish measured vs estimated; the asterisked footnote at line 104 is mitigation but easy to miss.

## R04 — AMGX_PRECISION_20260504.md
**[CITE: AMGX_PRECISION_20260504.md:9, 36-39]** claims max rel diff vs LU = 2.89e-15 across 48 dumps. I directly verified by reading `precision_results_ir1.json` — max=2.893e-15, median=1.801e-15, min=5.722e-16 across N=48. **TRUTH IS LU (scipy.spsolve)**, not AMGx self-residual. Numbers check out file-cited. NOTE: this is on **8K cells** (LPBF_sanity 2K + dumper_pipeline 8K). Does not by itself prove AMGx works on 500K — that needs separate evidence.

## R05 — data_inventory.md
**[CITE: data_inventory.md:9-65]** has good DISCLAIMER section explicitly listing what is/isn't provable. Claims "5.9 kPa = κ × tol" with κ_local ~ 1e6, but κ is asserted not measured. **[CITE: data_inventory.md:30-49]** lists three senior datasets: Initial_Period (works), Melting (broken: ‖A·x-b‖/‖b‖=18), Evaporation (broken). Ascribes brokenness to matrixDumper bug + case config — these two causes were never decoupled experimentally on the same case.

## R06 — solver_production_comparison.md
**[CITE: solver_production_comparison.md:3-4]** says "Status: Phase 0 pilot replay running". This is **stale** as of 2026-05-10 00:08; PID 466877 not found, JSON not produced. Document title is "Replay Experiment Report" but body is mostly placeholders and TBDs. Does not constitute evidence of anything experimental.

## R07 — EXPERIMENT_DESIGN.md (solver_production_comparison/)
Detailed protocol for 4-phase experiment. Phase 0 was supposed to be pilot. Methodology sound (5 modes incl. amgx_amortized + warm-start, lu_symbolic_reuse). Acceptance criteria specified. But: no actual results produced — Phase 0 didn't write any deliverable JSON.

## R08 — replay_amortized.py source
**[CITE: replay_amortized.py:78-135]** implements amgx amortized via `Plan(...)` once + `plan.update_coefficients(vv)` per step + `plan.solve(b_d, x0_d)`. **[CITE: replay_amortized.py:99-135]** warm-start path uses `x0_d = jnp.asarray(x_prev.astype(np.float64))`. Code is correct in principle. **Untested experimentally** as of 00:55 audit time.

## R09 — lu_truth_lab32_settled.py
**[CITE: lu_truth_lab32_settled.py]** runs SuperLU on lab32 melt-380 ns matrix. Caches result to `/tmp/x_LU_lab32_melting_pd.npy`. **CACHE FILE GONE** (verified `ls /tmp/x_LU_lab32_melting_pd.npy` → No such file). The 5.9 kPa story rested on this cache. Re-running takes ~70s SuperLU. Recommended: re-cache as part of Xeon plan.

## R10 — matrixDumper.H lines 395-432
The patch logic is sound: for coupled (processor) patches use `pbc × ptf.patchNeighbourField()`; for non-coupled use `pbc` directly. **[CITE: matrixDumper.H:413-431]**. This addresses the 32-rank inter-rank halo source contribution that older versions missed. **Verification of the fix** so far: only documented as "lab32 32-rank dump self-consistent at rel ≈ 2.4e-9" — claim not re-verified locally.

## R11 — precision_results_ir1.json (raw)
N=48 dumps. AMGx vs scipy LU truth — max=2.893e-15, median=1.801e-15, min=5.722e-16. The top-5 worst all from **dumper_pipeline 8K case at very short ts (4.4e-12, 1.98e-11)** with iter=22-24. The fact the worst rel is on the smallest matrices is actually surprising — usually larger N → more accumulation. Could indicate a numerical-conditioning artifact at very short time steps. NOT a refutation, but worth marking in ADVERSARY for cross-check.

## R12 — single_*pd_corr0_*.npz files (6 files, 13 MB each)
Each has keys: x_OF, x_AMGx_e8, x_truth (= AMGx tol=1e-12 + 1 IR), b, i,j,k, n, nx,ny,nz, dx, meta. Verified amgx_truth.iters across 6 cases: 559–1123. Verified amgx_truth.rel_resid_actual: 1.019e-15 to 1.025e-15 (algebraic residual, machine ε). NOT the same metric as "rel diff vs LU truth" cited in S3 (1.13e-11). The 1.13e-11 was per-cell max diff, computed on lab Xeon by `lu_truth_single_track.py`, JSON not on dev.

## R13 — Forensic on PID 466877 (S6)
Verified: process not present at 00:08 audit start. No replay_results.json. No /tmp/replay_v2.log. No relevant outputs in /tmp/claude-1000 task output dir. **The Phase 0 replay never produced output we can read**, neither in original PID 466877 attempt nor the re-run b8ypzuhxw. STATUS REFUTED relative to PROJECT_STATUS_REPORT's "is running" claim. Phase 0 pilot results: NONEXISTENT.

## Summary of Pass 1
- Strong: AMGx vs LU on 8K case (R11), matrixDumper patch logic (R10).
- Weak: OF wall numbers (R03 — fully derived), Phase 0 status (R13 — never ran), κ estimate (R09 — inferred from diag spread, not measured), 1.13e-11 5OOK claim (R12 — lab Xeon JSON not on dev).
- Inconsistencies: data_inventory says "lab32 5.9 kPa settled" but the cache backing it is gone (R09); no derived figs were re-checked.

Inconsistencies enumerated:
- D001 OF wall numbers in solver_comparison_500K table do not look like estimates to a casual reader; the asterisk-footnote is at line 104 only
- D002 PROJECT_STATUS_REPORT 1.2 says "lab32 5.9 kPa = κ × tol settled" with no surviving cache to inspect
- D003 The 1.13e-11 claim (S3) cites a JSON not present locally — only output text exists in conversation history
- D004 matrixDumper bug fix never re-verified ON the senior Melting/Evaporation data — only on lab32 case which had a separate physics bug (rays=0)
- D005 PROJECT_STATUS_REPORT 5.1 says "running PID 466877" but as of 2026-05-10 00:08 the PID is dead and no result JSON exists
- D006 dilu/experiments/solver_production_comparison/ has scripts but no validated runs
- D007 The "5s/step ÷ ~45 iter/step" derivation behind 110 ms/iter has at least 30% uncertainty in each factor — multiplied gives ~50% uncertainty in OF wall
- D008 Senior Melting "‖A·x_OF - b‖/‖b‖ = 18" attribution has two causes (matrixDumper bug + case config) never separated
- D009 No measurement of true κ(A) anywhere — only the diag-spread proxy 7.1e+11
- D010 Phase 0 "running" status was published in PROJECT_STATUS_REPORT before being verified
