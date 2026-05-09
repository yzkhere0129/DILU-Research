LAST_REVIEWED: 2026-05-10T03:35+08:00
ITERATION: v1

# FINAL_SUMMARY — Forensic Audit Outcome

## Previous Claude's 4 Sins (from overnight prompt §0)

1. Editorialized predictions as conclusions (e.g., "AMGx amortized 100-1000× faster than LU").
2. Mixed estimated and measured values in tables without clear visual separation.
3. Retracted hypotheses verbally without scrubbing all derivative claims.
4. Reported in-flight processes as "running" without verification.

## What this audit produced

### Discovered & locked in

| Finding | Evidence | Disposition |
|---|---|---|
| OF wall numbers in `solver_comparison_500K.md` are arithmetic `iter × 110 ms`, not measured | `build_solver_comparison_table.py:104` literal multiplication | C006 REFUTED-AS-MEASUREMENT; needs E01 |
| Phase 0 PID 466877 was dead at audit start; no replay output was ever produced | `ps -p 466877` no process; no `replay_results.json` | C013 REFUTED |
| `/tmp/x_LU_lab32_melting_pd.npy` cache deleted between work sessions | `ls` confirms missing | Cache restored tonight via SuperLU rerun |
| The asserted `κ ~ 1e6` for lab32 matrix is wrong; actual measured κ via Lanczos = **3.2e+14** | `eigsh(A_pos, k=1, sigma=0)` → σ_min=4.5e-29, σ_max=1.45e-14 | C012 refined; data_inventory.md needs amendment |
| The lab32 pd matrix is numerically near-singular (σ_min at fp64 noise floor) | Lanczos shift-invert | New C021 VERIFIED |
| OF self-residual numbers in metadata.json reproducibly match recompute | direct read of A.mm/b.mm/x_final.mm | C002 VERIFIED |
| 5.9 kPa OF-vs-LU diff confirmed reproducible: 25 cells > 100 Pa, 4 cells > 1 kPa | re-cached LU + diff | Number stands |

### Heuristic findings (real-but-not-fully-locked)

| Finding | Evidence | Action |
|---|---|---|
| The 1.13e-11 max diff between AMGx+IR and LU on 500K | conversation paste (lab Xeon CHOLMOD output) | E07 to lock in JSON |
| Senior melting/evap matrices' brokenness has two confounded causes (matrixDumper bug + case config) | logical analysis of historical events | Cannot settle without senior re-cooperation |
| AMGx amortized + warm-start "100-1000×" is hypothesis | Pure prediction, zero current measurement | E03 + E04 to settle, predicted realistic ratio 1.5-3× (not 100-1000×) |

### Deliverables produced

| # | File | Status |
|---|---|---|
| D1 | READING_NOTES.v1.md | 12 file entries + 10 D-inconsistencies |
| D2 | CLAIM_LEDGER.v1.md → v2.md | 21 claims (C001-C021), all 8 SUSPECT mapped |
| D3 | EVIDENCE_CHAIN.v1.md | DAG with STRONG/MEDIUM/WEAK edges + most-fragile-subgraph identified |
| D4 | KNOWN_UNKNOWNS.v1.md | 15 unknowns (U001-U015), each with settlement path |
| D5 | ADVERSARY_NOTES.v1.md | 20 attacks (A001-A020) on Pass 4 + Pass 6 |
| D6 | XEON_VALIDATION_PLAN.v1.md → v2.md | 11 experiments (E01-E11), schedule, robustness |
| D7 | run_xeon_validation.sh | --dry-run / --smoke / --resume / --only ; passes bash -n + dry-run |
| D8 | xeon_validation/{_common,E0X_runner,compare_to_expected}.py | 11 scripts, all py_compile clean |
| D9 | expected_results_template.json | 12 pre-registered predictions with rationale |
| D10 | RUNBOOK.md | 10-step morning-of instructions |
| D11 | FAILURE_MODES.md | 7 sections covering env / inputs / sanity / smoke / E01 / interpretation / fallback |
| D12 | ITERATION_LOG.md | hourly entries from H0 onward |
| D13 | FINAL_SUMMARY.md | this file |

### Validations passed before audit done

- [x] `bash run_xeon_validation.sh --dry-run` PASS, exit 0
- [x] `python3 -m py_compile xeon_validation/*.py` all OK
- [x] `bash -n run_xeon_validation.sh` syntax OK
- [x] All 8 SUSPECT_CLAIMS S1-S8 mapped to claims with verdicts and (where unsettled) Xeon experiments
- [x] expected_results_template.json predictions have rationale
- [x] RUNBOOK.md commands have been format-tested

## Tomorrow's success criteria

After running `bash run_xeon_validation.sh` on lab Xeon:

| Result | Interpretation |
|---|---|
| AMGx amortized speedup vs fresh ≈ 1.0-2× | Refutes 100-1000× claim. Realistic for our matrix. |
| AMGx amortized speedup vs fresh > 5× | Worth investigating. Hypothesis partially supported. |
| AMGx warm-start saves > 50% iter | Strong support for warm-start utility |
| OF DICPCG measured per-iter ≈ 110 ms | C006 estimate validated |
| OF DICPCG measured per-iter < 50 ms or > 200 ms | C006 measurement was wrong; rewrite |
| AMGx + IR vs LU max rel < 2e-11 | C004 VERIFIED |
| AMGx + IR vs LU max rel > 1e-9 | C004 REFUTED — investigate |
| All 6 single_track κ ∈ [1e6, 1e16] | Confirms ill-conditioning generally |
| Any single_track κ < 1e6 | "Real LPBF" matrix is much better conditioned than lab32 (rays=0) → S4 narrative needs further refinement |

## Items the audit cannot settle without further user action

1. Senior Melting/Evaporation matrix brokenness cause (need senior to re-dump with patched matrixDumper, or run his case ourselves).
2. Whether matrixDumper.H 32-rank patch survives in production multi-rank LPBF cases (lab32 had rays=0; haven't tested patch on a working 32-rank rays>0 case end-to-end).
3. Provenance of `/tmp/bench_gold.log` (timestamp suggests sibling agent, not this session).
4. Behavior at >5M cells (no hardware to test).
5. Whether AMGx config `CLASSICAL_V_DIAGSCALED` is optimal for this matrix family (would require extended config sweep).

## Self-confidence at audit close

70/100. Confidence comes from:
- Strong claim ledger with file:line citations
- Adversary-tested validation plan
- Fail-loud scripts that have passed dry-run + py_compile
- Realistic predictions (e.g., predicting 1.5-3× amortized speedup, not 100-1000×, sets clear refutation criteria)

Reasons to NOT be at 90/100:
- Have not actually run the experiments (that is tomorrow's job)
- Pre-registered predictions could still be wrong in unexpected ways
- Some confounding (senior data attribution, AMGx config space) outside scope

## What user should do next

Morning of:
1. `cd ~/DILU-Research/audit_overnight_20260509`
2. Read `RUNBOOK.md`
3. Follow steps 0–8.

If everything goes per plan, by tomorrow evening:
- `xeon_validation/results/` will have ~50 result.json files
- `xeon_validation/analysis/summary.md` will have all headline ratios
- CLAIM_LEDGER claims marked HYPOTHESIS will be upgraded to VERIFIED or REFUTED based on data

Then a v3 of CLAIM_LEDGER and EVIDENCE_CHAIN can be produced as a "post-validation" artifact.

---

The audit is finished when:
- This FINAL_SUMMARY references all D1-D13
- `bash run_xeon_validation.sh --dry-run` passes
- ITERATION_LOG.md has ≥12 hourly entries (TBD as I keep iterating)

(I am at H6 right now; will continue to H8+ as time permits, focusing on adversary refinements + RUNBOOK polish.)
