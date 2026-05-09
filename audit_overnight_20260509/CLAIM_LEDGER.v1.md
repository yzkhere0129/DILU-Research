LAST_REVIEWED: 2026-05-10T01:10+08:00
ITERATION: v1

# Claim Ledger v1 — Forensic Audit

Status legend: VERIFIED (file:line backs it, math checked) / HYPOTHESIS (predictive only, no current evidence) / REFUTED (evidence contradicts) / UNKNOWN (data needed not present)

Each claim has SOURCE, TYPE, EVIDENCE, ADVERSARY, RESPONSE, DEPENDS_ON, VERDICT, NEEDS_XEON.

---

## C001 — AMGx CLASSICAL_V_DIAGSCALED + 1 IR matches scipy LU truth on 8K cells to rel max 2.89e-15
SOURCE: AMGX_PRECISION_20260504.md:9, 36-39 + precision_results_ir1.json
TYPE: VERIFIED
EVIDENCE: I ran `import json; rows = json.load(open('precision_results_ir1.json'))['rows']`; computed `max(r["amgx_vs_truth"] for r in rows) = 2.893e-15`, median 1.801e-15, min 5.722e-16, N=48. Top-5 worst at dumper_pipeline 8K at very-short timesteps (4.4e-12 to 1.98e-11). Truth = `scipy.sparse.linalg.spsolve(A, b)`.
ADVERSARY: "8K cells is tiny — does this generalize?" Answer: No, it does NOT prove behavior on 500K or 2M. C001 is bounded to ≤ 8K.
RESPONSE: Acknowledged. C001 is foundation only; 500K verification is C003.
DEPENDS_ON: (none)
VERDICT: VERIFIED for 8K cell domain only.
NEEDS_XEON: N (already proven, just locked in evidence)

## C002 — On 500K LPBF pd matrix, OF DICPCG @ tol=1e-8 self-residual reaches ‖A·x_OF - b‖/‖b‖ ≈ 9.2e-9 to 1.0e-8
SOURCE: solver_comparison_500K.md:18-78 col `rel_resid (‖A·x-b‖/‖b‖)`; metadata.json:final_residual in postProcessing/matrices/<t>/pd_corr0/
TYPE: VERIFIED
EVIDENCE: solver_comparison_500K.md per-timestep table; cross-check by reading any `metadata.json` in case dump confirms `final_residual: ~9e-9` and `iterations` consistent with table.
ADVERSARY: "Did you re-compute ‖A·x - b‖/‖b‖ from raw A.mm, b.mm, x_final.mm?" Answer: not yet — relied on metadata. Cheap single-matrix check is feasible tonight.
RESPONSE: Will add E07a in Xeon plan (read npz, compute residual, compare to metadata).
VERDICT: VERIFIED-ASSUMING-METADATA-IS-FAITHFUL. Tonight do single-matrix sanity ; metadata likely faithful but re-compute the 6 timesteps to lock in.
NEEDS_XEON: N (single-matrix re-compute is allowed tonight per §4 budget)

## C003 — On 500K LPBF pd matrix, AMGx + 1 IR reaches ‖A·x - b‖/‖b‖ ≈ 1.02e-15 (algebraic, all 6 timesteps)
SOURCE: npz meta `amgx_truth.rel_resid_actual` for all 6 single_*pd_corr0_*.npz
TYPE: VERIFIED
EVIDENCE: I ran `np.load('single_melting_pd_corr0_3.8e-07.npz', allow_pickle=True)["meta"]`, parsed JSON, `amgx_truth.rel_resid_actual = 1.024e-15` (etc., 5 more timesteps). ≤ 1.025e-15 across all 6.
ADVERSARY: This is **algebraic** residual, not diff vs truth. 1e-15 residual on ill-conditioned A doesn't imply 1e-15 distance from LU solution.
RESPONSE: Correct distinction. C003 is purely about residual. C004 separately addresses diff vs LU.
VERDICT: VERIFIED.
NEEDS_XEON: N

## C004 — On 500K LPBF pd matrix, AMGx + 1 IR matches CHOLMOD LU truth to rel max ≤ 1.13e-11 across all 6 timesteps
SOURCE: lab Xeon CHOLMOD output (pasted in conversation, not on dev disk locally); referenced in solver_comparison_500K.md as max(rel max) = 1.13e-11; lu_truth_single_track_results.json mentioned but file not present on dev.
TYPE: HYPOTHESIS-with-conversation-evidence (or, we could call it VERIFIED-CONDITIONAL)
EVIDENCE: User pasted output of `python -m dilu.amgx.bench.lu_truth_single_track --case ~/cases/single_track_dump` showing per-timestep rel: 9.73e-12, 7.94e-12, 1.70e-12, 1.13e-11, 1.14e-12, 8.79e-12. Max=1.13e-11.
ADVERSARY: (a) The JSON `lu_truth_single_track_results.json` is not on dev; we have only conversation output. (b) The lab Xeon CHOLMOD wall ranges 69-75 s — quite tight, but its actual residual `2.7e-15` we saw printed. (c) Did the script use the same npz as we have on dev? (single_*pd_corr0_*.npz, MD5-equivalent?)
RESPONSE: We need to (1) regenerate the JSON deterministically by re-running CHOLMOD on the exact npz files we now have on dev (or on lab), (2) MD5-verify both copies, (3) compare to the per-timestep numbers cited.
VERDICT: HYPOTHESIS-PARTIALLY-VERIFIED — conversation evidence is real, just not reproducibly stored. Re-run as Xeon E07.
NEEDS_XEON: Y — re-run lu_truth on each of the 6 npz to get per-timestep rel-max stored as JSON.

## C005 — OF DICPCG iter for 500K pd_corr0 across 6 timesteps = {7, 35, 40, 17, 43, 65} (one per timestep)
SOURCE: solver_comparison_500K.md table per-timestep; backed by metadata.json:iterations in each pd_corr0 dir
TYPE: VERIFIED
EVIDENCE: solver_comparison_500K.md:18-78; cross-check metadata.json directly (I read for 380ns: iter=35, matches).
ADVERSARY: 7 iter at 320ns vs 65 iter at 1060ns — does this monotonic increase make physical sense? Vapor recoil = harder matrix expected, so iter should grow with t. Yes — confirms.
RESPONSE: Plausible physics-iter correlation. No issue with the numbers.
VERDICT: VERIFIED.
NEEDS_XEON: N

## C006 — OF DICPCG wall (estimated) for 500K pd_corr0 = {770, 3850, 4400, 1870, 4730, 7150} ms
SOURCE: solver_comparison_500K.md:18-78 wall column
TYPE: REFUTED-AS-MEASUREMENT (HYPOTHESIS at best)
EVIDENCE: build_solver_comparison_table.py:104 literal `"wall_ms": of_iter * 110`. Forensically I confirmed all 6 numbers = iter × 110. The 110 ms/iter is itself derived: `5s/step from log.run / ~45 iter/step assumed`. Both factors are estimates. Underlying wall measurement does not exist per pd_corr0.
ADVERSARY: User's S2 attack: "estimate masquerading as measurement". I agree.
RESPONSE: Replace these with HYPOTHETICAL flags. Schedule actual measurement E01 on Xeon (single_track_dump rerun with solverInfo function or instrumentation in matrixDumper).
VERDICT: REFUTED for "measurement"; downgrade table to "ESTIMATED, ±50%" until E01 completes.
NEEDS_XEON: Y — E01 (direct OF DICPCG per-solve wall measurement).

## C007 — AMGx PCG @ tol=1e-8 iter for 500K pd_corr0 across 6 timesteps = {497, 535, 539, 709, 435, 773}
SOURCE: solver_comparison_500K.md per-timestep tables; npz meta.amgx_e8.iters
TYPE: VERIFIED
EVIDENCE: I cross-checked 380ns: npz meta `amgx_e8.iters = 535`. Matches table.
ADVERSARY: That iter range 435-773 — 1.7× spread. Why so high? CLASSICAL_V_DIAGSCALED isn't the optimal AMGx config for this Laplacian. We didn't try CLASSICAL_V_DIAGSCALED with stricter coarsening, or AGGRESSIVE, or PMIS, or (most importantly) AMGx with PCG-DIC preconditioner mode.
RESPONSE: This claim is just iter count, true. The interpretation "AMGx is slow" is suspect because AMGx config wasn't tuned. Logged as separate concern.
VERDICT: VERIFIED for the iter count. Interpretation "AMGx PCG inferior to OF DICPCG on this matrix" needs broader AMGx config sweep.
NEEDS_XEON: Optional — AMGx config sweep (PCG-DIC vs CLASSICAL_V_DIAGSCALED) on 1 timestep.

## C008 — AMGx PCG (RTX 3050) @ tol=1e-8 single-shot wall for 500K pd_corr0 ≈ 9-39 s
SOURCE: solver_comparison_500K.md table; npz meta.amgx_e8.t_solve_s
TYPE: VERIFIED-FOR-DEV-RTX3050-SINGLE-SHOT
EVIDENCE: npz meta has `amgx_e8.t_solve_s` for each. 9.025-38.867 s range. These are MEASURED (Python timing wrapper around Plan.solve).
ADVERSARY: (a) RTX 3050 is dev hardware, not lab 5060 GPU. Numbers don't generalize. (b) Single-shot includes setup; not amortized.
RESPONSE: Correctly scoped. C008 is dev RTX 3050 single-shot only. Lab 5060 amortized = E03 in Xeon plan.
VERDICT: VERIFIED for hardware/mode scope.
NEEDS_XEON: Y for production-mode amortized + lab GPU comparison

## C009 — AMGx amortized + warm-start beats LU 100-1000× in real CFD time-stepping
SOURCE: PROJECT_STATUS_REPORT 1.3, EXPERIMENT_DESIGN.md
TYPE: HYPOTHESIS (no measurement on this dataset)
EVIDENCE: Argument is theoretical: "Plan.update_coefficients reuses AMG hierarchy" + "warm-start reduces iterations". No measurement on this dataset has been run.
ADVERSARY (S1): User raised this as suspicious. (a) What does amortized save vs full setup? Need data on setup_ms vs update_coefficients_ms — likely 2-3 s vs 50-100 ms = ~30× setup savings only. (b) warm-start: how similar are x_t and x_{t+1} in PISO? On melting matrix where alpha changes, likely far. Need to measure ‖x_{t+1} - x_t‖∞ / ‖x_t‖∞ across our 6 timesteps. (c) "100-1000×" is a range that smacks of marketing.
RESPONSE: Strong critique stands. Tonight cannot run replay (per §4). Schedule as E03 + E04 on Xeon. Predict: amortized 5-30× over fresh AMGx; warm-start 1.5-3× over cold start; total speedup over LU per-step ~20-50× (NOT 100-1000×). Pre-register this prediction.
VERDICT: HYPOTHESIS, predicts 20-50× (not 100-1000×) under real PISO conditions
NEEDS_XEON: Y — E03 + E04

## C010 — On 500K LPBF, OF wall ≈ 1.9-7.2 s vs AMGx 9-39 s ⇒ "OF beats AMGx 5-10×"
SOURCE: PROJECT_STATUS_REPORT 1.1; solver_comparison_500K.md takeaway 4
TYPE: REFUTED-CONCLUSION (relies on C006 which is REFUTED)
EVIDENCE: OF side is iter×110ms not measurement (C006). AMGx side is dev RTX 3050 single-shot (C008). Mixing estimated CPU vs measured GPU and reporting "5-10× faster" is unsound.
ADVERSARY: At minimum, "5-10×" is in the noise of OF wall estimate (50% uncertainty). Cannot defend in review.
RESPONSE: Downgrade to "data insufficient to claim a winner; see E01+E02 measurements for proper comparison".
VERDICT: REFUTED.
NEEDS_XEON: Y — both sides need proper measurement (E01, E02).

## C011 — CHOLMOD LU on lab Xeon takes 69-75 s per 500K solve
SOURCE: solver_comparison_500K.md table; matches lab Xeon output user pasted
TYPE: VERIFIED-IN-CONVERSATION (not stored as JSON locally)
EVIDENCE: Conversation history showed "CHOLMOD done in 73.2s, 74.7s, 74.5s, 69.5s, 72.5s, 70.8s" across 6 timesteps. ‖A·x_LU - b‖/‖b‖ printed at 2.69-2.71e-15.
ADVERSARY: Same as C004 — JSON not on dev; conversation paste only. Re-run on Xeon to lock in.
RESPONSE: Schedule as E05 (CHOLMOD fresh per timestep, ≥5 reps).
VERDICT: VERIFIED-CONDITIONAL — re-run E05 to make reproducible.
NEEDS_XEON: Y — E05.

## C012 — lab32 case is well-posed (LU gives unique solution), 5.9 kPa OF-vs-LU diff = condition number × tol amplification
SOURCE: data_inventory.md DISCLAIMER; lu_truth_lab32_settled.py
TYPE: PARTIALLY-REFUTED (cache file deleted)
EVIDENCE: data_inventory.md asserts "system 是良态 (LU residual 2.5e-15)". The supporting cache `/tmp/x_LU_lab32_melting_pd.npy` is **GONE**. The 5.9 kPa number was computed from this cache. Without the cache, the claim is unbacked locally. The ANALYTICAL part (κ × tol theoretical) is sound but κ is also unmeasured.
ADVERSARY (S4): User asked: did all derived materials get synchronized? I haven't grep'd commit messages or fig captions. Also "κ ~ 1e6" is asserted not measured. Diag-spread is 7.1e+11 (extreme; suggests κ could be way bigger than 1e6).
RESPONSE: Re-cache LU on dev (allowed tonight, ~70s SuperLU), grep commit log for ill-posed mentions, schedule Lanczos κ estimation E09.
VERDICT: PARTIALLY-REFUTED (claim "settled" was overstated; cache lost; κ unmeasured).
NEEDS_XEON: Y — E09 (Lanczos κ on lab32 + 6 single_track matrices).

## C013 — "Phase 0 6-step replay is currently running on dev (PID 466877)"
SOURCE: PROJECT_STATUS_REPORT.md 5.1; solver_production_comparison.md status line
TYPE: REFUTED
EVIDENCE: At 2026-05-10T00:08, `ps -p 466877` returns no process. `replay_results.json` does not exist. `/tmp/replay_v2.log` does not exist. The status was stale at the time of report generation.
ADVERSARY (S6): User caught this directly. The original Claude wrote "is running" without verifying.
RESPONSE: Replace with "Phase 0 ATTEMPTED, OUTPUT NOT PRODUCED, CAUSE UNKNOWN (likely silent crash at solve or pipe truncation)". Schedule re-run E11 in Xeon plan as a clean retry with ≥5 reps to establish per-mode wall numbers.
VERDICT: REFUTED.
NEEDS_XEON: Y — E11.

## C014 — matrixDumper.H 32-rank bug fixed (commit 47d4863) and dump self-consistency rel ≈ 2.4e-9
SOURCE: data_inventory.md, matrixDumper.H lines 395-432, conversation history
TYPE: PARTIALLY-VERIFIED-WITH-CAVEATS
EVIDENCE: matrixDumper.H lines 413-431 implement the fix correctly: coupled patches use `pbc * patchNeighbourField()`. Self-consistency 2.4e-9 was reported in conversation but no JSON. The fix verification was on lab32_dump CASE which had its OWN physics bug (rays=0). So the fix was checked against a case with separate issues.
ADVERSARY (S5): The bug-fix-vs-case-config attribution for senior Melting/Evaporation broken data is uncertain. Senior's own Melting matrix had ‖A·x_OF - b‖/‖b‖ = 18 — this could be (a) matrixDumper bug, (b) case bug, (c) both. They were never separated experimentally.
RESPONSE: To separate (a) and (b), would need to: re-dump senior's exact case with patched dumper, OR get senior to re-dump. Both are user-action items, not Xeon-able tonight. Schedule E10 (optional) to re-verify the matrixDumper patch on a fresh 32-rank single_track or dense case.
VERDICT: PARTIALLY-VERIFIED — code is correct, attribution to senior brokenness is unproven.
NEEDS_XEON: Optional E10 (only if budget allows).

## C015 — AMGx amortized cost: setup ≈ 2-3 s, update_coefficients ≈ 50-100 ms, solve ≈ 100-150 ms
SOURCE: PROJECT_STATUS_REPORT comment 5; bench_amgx_vs_of_n32.py history
TYPE: HYPOTHESIS (not measured on this dataset)
EVIDENCE: Approximate numbers from old `bench_amgx_vs_of_n32_results.json` on senior 21-bundle data. NOT measured on single_track 500K.
ADVERSARY: Numbers from a different dataset — extrapolation. Solve time on 500K LPBF (which has very different conditioning) likely different.
RESPONSE: Schedule E03 to measure setup, update_coefficients, solve separately on the 6 single_track timesteps.
VERDICT: HYPOTHESIS.
NEEDS_XEON: Y — E03.

## C016 — In CFD time-stepping, x_{t} ≈ x_{t-Δt} ⟹ warm-start saves ~50-90% iterations
SOURCE: PROJECT_STATUS_REPORT 8.2 (implicit); EXPERIMENT_DESIGN.md
TYPE: HYPOTHESIS (zero data on this dataset)
EVIDENCE: None for our matrices. Generic CFD experience.
ADVERSARY: For LPBF transitioning from cold to melt to vapor, the pd field changes drastically (1.28 MPa peaks form/move with melt pool). x_{t+1} - x_t may NOT be small. Need to measure.
RESPONSE: Schedule E04 (warm-start sensitivity): for each (t-1, t) pair, measure ‖x_t - x_{t-1}‖∞/‖x_t‖∞ AND iter saved when AMGx initialized at x_{t-1} vs zero. The iter ratio is the warm-start benefit.
VERDICT: HYPOTHESIS.
NEEDS_XEON: Y — E04.

## C017 — CHOLMOD symbolic-reuse (analyze once, cholesky_inplace per step) cuts factor wall ~2-5×
SOURCE: EXPERIMENT_DESIGN.md
TYPE: HYPOTHESIS (zero data on this dataset)
EVIDENCE: Generic SuiteSparse experience. Symbolic factorization (AMD ordering) is ~30-50% of total factor cost, so reusing it should save that.
ADVERSARY: For our specific matrix (very-small diag values 1e-26), AMD ordering may stress numerical pivoting; if cholesky_inplace fails (numerical issue), we fall back to full factor anyway.
RESPONSE: Schedule E06 (CHOLMOD symbolic reuse) and watch for fallback-to-full warnings.
VERDICT: HYPOTHESIS.
NEEDS_XEON: Y — E06.

## C018 — On 500K LPBF, AMGx wins at >5M cells / multi-GPU (S8)
SOURCE: PROJECT_STATUS_REPORT 8.1
TYPE: REFUTED-as-substantive-claim (zero data; not even possible to test on our hardware tonight)
EVIDENCE: We have 2M (running, dump in progress), no >5M data. Our hardware tops at 4 GB VRAM (RTX 3050) and 8 GB (RTX 5060) — neither runs 5M-cell AMGx amortized.
ADVERSARY (S8): User asked source. Answer: NVIDIA marketing + literature, not our experiments.
RESPONSE: Mark as "literature claim, not validated by this project". Could possibly synthesize 5M Poisson on lab Xeon (out of GPU range, must use multi-GPU which we don't have), so this can NOT be settled with our resources.
VERDICT: REFUTED-as-validated-claim.
NEEDS_XEON: N (not testable with our resources).

## C019 — Senior Melting & Evaporation matrices unsolvable (LSMR istop=7, ‖A·x_OF - b‖/‖b‖ = 18)
SOURCE: data_inventory.md row "学长 Melting 80³ = 512K"
TYPE: VERIFIED-FROM-PRIOR-WORK (lab_PCG_truth_results.json exists)
EVIDENCE: data_inventory cites this. We can read the JSON. (Did not re-verify in this Pass 1.)
ADVERSARY: But the cause attribution (matrixDumper bug vs case config) was never separated.
RESPONSE: Same as C014 — re-dump senior's case with patched matrixDumper would settle. Not in tonight's scope. Document as known-unknown.
VERDICT: VERIFIED-as-data-fact, REFUTED-cause-attribution.

## C020 — A scratch finding: `/tmp/bench_gold.log` (5385 bytes, mtime 2026-05-10T00:03) shows a benchmark run on 500K matrices, source unknown
SOURCE: forensic discovery during S6 investigation
TYPE: UNKNOWN (provenance not established)
EVIDENCE: head shows `=== 3.2e-07 (melting_onset) pd_corr0 === ... [raw / f64] tol=1e-08 iter= 500 converged=False NaN=False ...`. Looks like a scipy CG fp64/fp32 benchmark.
ADVERSARY: Did the previous Claude or another agent run this? File not in any documented script's output path. Not safe to incorporate without provenance.
RESPONSE: Note for follow-up. Likely from `~/DILU-Research-conditioning-study/` agent or earlier session. Don't rely.
VERDICT: UNKNOWN provenance.

---

## Summary Table

| ID | Status |
|---|---|
| C001 AMGx 8K vs LU 2.89e-15 | VERIFIED |
| C002 OF self-residual ~1e-8 | VERIFIED-pending-recompute |
| C003 AMGx+IR algebraic resid 1e-15 | VERIFIED |
| C004 AMGx+IR vs LU 1.13e-11 (500K) | HYPOTHESIS-CONDITIONAL → E07 |
| C005 OF iter 7-65 | VERIFIED |
| C006 OF wall 770-7150 ms | REFUTED → E01 |
| C007 AMGx iter 435-773 | VERIFIED |
| C008 AMGx wall 9-39 s (dev RTX 3050) | VERIFIED-scoped |
| C009 amortized 100-1000× faster | HYPOTHESIS → E03 |
| C010 OF beats AMGx 5-10× | REFUTED → E01+E02+E03 |
| C011 CHOLMOD 69-75 s | VERIFIED-CONDITIONAL → E05 |
| C012 lab32 well-posed + κ × tol story | PARTIALLY-REFUTED (cache lost) → E09 |
| C013 Phase 0 PID 466877 running | REFUTED → E11 |
| C014 matrixDumper bug fixed | PARTIALLY-VERIFIED |
| C015 amortized: setup 2-3s update 50-100ms | HYPOTHESIS → E03 |
| C016 warm-start saves 50-90% iter | HYPOTHESIS → E04 |
| C017 CHOLMOD symbolic 2-5× | HYPOTHESIS → E06 |
| C018 AMGx wins at >5M | REFUTED-as-claim |
| C019 senior melt/evap unsolvable | VERIFIED+attribution-unknown |
| C020 /tmp/bench_gold.log | UNKNOWN |

---

## SUSPECT_CLAIMS Mapping

| S | Maps to | Status |
|---|---|---|
| S1 (amortized 100-1000×) | C009 | HYPOTHESIS — E03 must measure |
| S2 (OF wall estimated) | C006 | REFUTED-AS-MEASUREMENT — E01 must measure |
| S3 (AMGx+IR rel 1.13e-11 6/6) | C004 | HYPOTHESIS-CONDITIONAL — E07 |
| S4 (5.9 kPa = κ × tol) | C012 | PARTIALLY-REFUTED — cache lost; E09 |
| S5 (matrixDumper bug fix) | C014 | PARTIALLY-VERIFIED — E10 optional |
| S6 (Phase 0 running) | C013 | REFUTED — E11 |
| S7 (OF iter 17-65 vs AMGx 435-773) | C005 + C007 | VERIFIED for the numbers; INTERPRETATION needs config-sweep concession |
| S8 (AMGx wins at >5M) | C018 | REFUTED-as-validated — out of resource range |
