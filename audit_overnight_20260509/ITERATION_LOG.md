LAST_REVIEWED: 2026-05-10T03:45+08:00
ITERATION: H6

# Overnight Forensic Audit — Iteration Log

Started: 2026-05-10T00:08:16+08:00 (Beijing local)
Worker: Claude (Opus 4.7, this session)
Working dir: /home/yzk/DILU-Research/audit_overnight_20260509/

---

## H0 — 2026-05-10T00:08:16+08:00 (start)

WALL_SINCE_START: 0h00m
THIS_HOUR_DID:
  - Read overnight prompt fully. Internalized 4-rituals (file:line citations, status enums, no emojis in deliverables, iteration filenames).
  - Created `audit_overnight_20260509/`.
  - Forensic check on S6: PID 466877 NOT FOUND. No replay_results.json. No /tmp/replay_v2.log. **REFUTED** preliminary.
  - Discovered `/tmp/bench_gold.log` mtime 00:03 today, 5385 bytes, source unknown — separate benchmark run not documented.
SUSPECT: S6 REFUTED-PRELIMINARY; rest not-yet-attacked
DELIVERABLES: D12 ITERATION_LOG.md (v0)
SELF_CONFIDENCE_FINAL: 12/100
NEXT_HOUR_PLAN: Pass 1 reading.

---

## H1 — 2026-05-10T01:25+08:00

WALL_SINCE_START: 1h17m
THIS_HOUR_DID:
  - Read CLAUDE.md, PROJECT_STATUS_REPORT.md (re-read), solver_comparison_500K.md (audit), AMGX_PRECISION_20260504.md, data_inventory.md DISCLAIMER.
  - Verified precision_results_ir1.json: 48 dumps, max amgx_vs_truth = 2.893e-15. NUMBER REAL.
  - Verified npz files for 6 single_track timesteps.
  - **S2 forensically PROVEN**: build_solver_comparison_table.py:104 literal `wall_ms = of_iter * 110`. Verified math: all 6 wall numbers = iter × 110 ms.
  - **S6 confirmed**: PID 466877 dead, no JSON.
  - Wrote READING_NOTES.v1.md (12 file entries + D001-D010).
  - Wrote CLAIM_LEDGER.v1.md (20 claims C001-C020, all 8 SUSPECT mapped).
SUSPECT: S2 REFUTED-AS-MEASUREMENT; S6 REFUTED. Others awaiting Xeon.
DELIVERABLES: D1, D2 (v1) complete; D12 v1.
SELF_CONFIDENCE_FINAL: 30/100
NEXT_HOUR_PLAN: Pass 2 — sanity-recompute + cache lab32 LU + EVIDENCE_CHAIN.

---

## H2 — 2026-05-10T02:00+08:00

WALL_SINCE_START: 1h52m
THIS_HOUR_DID:
  - Sanity-recomputed `‖A·x_OF - b‖/‖b‖` from raw .mm for all 6 timesteps. All within 5-15% of metadata. **C002 upgraded VERIFIED**.
  - Re-cached lab32 LU via SuperLU on dev (176.6s, residual 2.752e-15). Verified 5.9 kPa story: max|x_OF - x_LU| = 5914 Pa, 25 cells > 100 Pa, 4 cells > 1 kPa.
  - **Lanczos κ measurement** on lab32: σ_max=1.453e-14, σ_min=4.516e-29, **κ ≈ 3.2e+14** (NOT 1e6 as PROJECT_STATUS_REPORT cited). Matrix is **near-singular at fp64**.
  - Wrote CLAIM_LEDGER.v2.md (refined C002, C012; new C021).
SUSPECT: S4 PARTIALLY-REFINED; lab32 case revealed numerically near-singular.
DELIVERABLES: D2 v2 complete.
SELF_CONFIDENCE_FINAL: 45/100
NEXT_HOUR_PLAN: EVIDENCE_CHAIN + KNOWN_UNKNOWNS, then start XEON_VALIDATION_PLAN + scripts.

---

## H3 — 2026-05-10T02:30+08:00

WALL_SINCE_START: 2h22m
THIS_HOUR_DID:
  - Wrote EVIDENCE_CHAIN.v1.md: DAG with STRONG/MEDIUM/WEAK edges. Identified most-fragile-subgraph: C002 + 110ms/iter assumption → C006 (REFUTED) → C010 (REFUTED conclusion). Headline thesis "AMGx amortized beats LU" is HYPOTHESIS chain end-to-end.
  - Wrote KNOWN_UNKNOWNS.v1.md: 15 unknowns (U001-U015), each with settlement path. 11 are tomorrow-Xeon-able; 2 user-action required (senior data); 2 out of resource range.
  - Wrote XEON_VALIDATION_PLAN.v1.md: 11 experiments (E01-E11) with TARGETS, INPUT, PROCEDURE, MEASURE, REPETITIONS, STOP_CRITERION, EXPECTED_HOLDS / EXPECTED_FAILS, RUNTIME_BUDGET. Total 7-8h on Xeon.
SUSPECT: All 8 mapped to E01-E11.
DELIVERABLES: D3, D4, D6 v1 complete.
SELF_CONFIDENCE_FINAL: 55/100
NEXT_HOUR_PLAN: Implement scripts.

---

## H4 — 2026-05-10T03:00+08:00

WALL_SINCE_START: 2h52m
THIS_HOUR_DID:
  - Wrote run_xeon_validation.sh (bash entrypoint with --dry-run/--smoke/--resume/--only).
  - Wrote xeon_validation/_common.py (utilities: TimedSection, write_result_json, sanity_check_matrix, normalize_sign, find_npz, cold_cache).
  - Wrote 9 runner scripts: E01 (OF wall via solverInfo), E02 (AMGx fresh), E03 (AMGx amortized), E04 (AMGx warm-start), E05 (CHOLMOD fresh), E06 (CHOLMOD symbolic-reuse), E07 (LU vs AMGx_e12_IR diff), E08 (aggregation + headline ratios), E09 (Lanczos κ). compare_to_expected.py for post-run summary.
  - Wrote expected_results_template.json (12 predictions with rationale).
  - Wrote RUNBOOK.md (10-step morning-of instructions) + FAILURE_MODES.md (7 sections).
  - All 11 Python scripts pass `python3 -m py_compile`.
  - `bash -n run_xeon_validation.sh` passes.
  - `bash run_xeon_validation.sh --dry-run` PASSES, exit 0, manifest.json written.
SUSPECT: pipeline structurally sound, awaits tomorrow-Xeon execution.
DELIVERABLES: D7, D8, D9, D10, D11 v1 complete.
SELF_CONFIDENCE_FINAL: 65/100
NEXT_HOUR_PLAN: Adversary Pass 4+6 against ledger and plan.

---

## H5 — 2026-05-10T03:15+08:00

WALL_SINCE_START: 3h07m
THIS_HOUR_DID:
  - Wrote ADVERSARY_NOTES.v1.md: 20 attacks (A001-A020) on CLAIM_LEDGER + XEON_PLAN.
  - Identified 11 high-impact action items.
  - Applied A008 (thread pinning), A013/A015 (E08 headline ratios), A017 (manifest.json) to v2.
  - Edits to _common.py (pin_threads_to_one), run_xeon_validation.sh (env vars + manifest), E08_runner.py (headline ratios calc).
  - Re-verified py_compile + dry-run after edits — all pass.
SUSPECT: more attacks identified; v2 plan addresses majority.
DELIVERABLES: D5 v1 complete; D2/D6/D7/D8 incrementally improved.
SELF_CONFIDENCE_FINAL: 70/100
NEXT_HOUR_PLAN: Plan v2 + Final summary + last verifications.

---

## H6 — 2026-05-10T03:35+08:00

WALL_SINCE_START: 3h27m
THIS_HOUR_DID:
  - Wrote XEON_VALIDATION_PLAN.v2.md (incorporates A001-A020 fixes).
  - Wrote FINAL_SUMMARY.md (D13).
  - Updated ITERATION_LOG.md with H1-H6 entries.
SUSPECT: same status as H5.
DELIVERABLES: D2/D3/D6 v2; D13 v1; D12 v6.
SELF_CONFIDENCE_FINAL: 73/100
NEXT_HOUR_PLAN: Continue iteration H7+ — additional adversary, edge cases, RUNBOOK polish, possible quick measurements (Lanczos on lab32 with multi-seed).

---

## Self-monitoring scoreboard (updates per hour)

| H | Wall | Confidence | Deliverables done |
|---|---|---|---|
| H0 | 0h | 12 | D12 v0 |
| H1 | 1h17m | 30 | D1, D2 v1, D12 v1 |
| H2 | 1h52m | 45 | D2 v2 |
| H3 | 2h22m | 55 | D3, D4, D6 v1 |
| H4 | 2h52m | 65 | D7, D8, D9, D10, D11 v1 |
| H5 | 3h07m | 70 | D5 v1; D2/D6/D7/D8 refined |
| H6 | 3h27m | 73 | D6 v2, D13 v1, D12 v6 |
