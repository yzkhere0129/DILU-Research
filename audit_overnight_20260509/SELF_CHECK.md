LAST_REVIEWED: 2026-05-10T03:55+08:00
ITERATION: v1

# Final Self-Check — verifying all overnight prompt requirements

## §1.1 Forbidden behaviors

| F | Check | Status |
|---|---|---|
| F1 | No "应该会"/"通常"/"按理说" without HYPOTHESIS tag | All hypothesis claims tagged in ledger |
| F2 | All wall/iter/memory numbers tagged MEASURED/ESTIMATED/HYPOTHETICAL | C006 explicitly marked REFUTED-AS-MEASUREMENT; expected_results explicitly predictions |
| F3 | No heavy compute tonight | Only 1× SuperLU (allowed), 2× Lanczos σ_min (allowed). No 384-step replay, no 2M, no 4-solver sweep. |
| F4 | At least 7 iteration passes | H0-H6 done at this point; aiming for H7-H8 |
| F5 | Read all §2 files before analysis | Per Pass 1 in READING_NOTES.v1.md, 12 files read |
| F6 | No one-liner claim summaries | Every claim has SOURCE + EVIDENCE + ADVERSARY + RESPONSE + DEPENDS_ON + VERDICT |
| F7 | No emoji or ✅/❌ in deliverable status | Used VERIFIED/HYPOTHESIS/REFUTED/UNKNOWN throughout. (One mistake: README files I edited tonight in conditioning-study had ✅ but those are not audit deliverables) |
| F8 | Iterative drafts saved as .vN | CLAIM_LEDGER v1, v2; XEON_PLAN v1, v2; ADVERSARY v1; READING_NOTES v1; KNOWN_UNKNOWNS v1; EVIDENCE_CHAIN v1 |
| F9 | ITERATION_LOG entries detailed, not "done" | Every H entry has: this_hour_did, suspect_status, deliverables_status, self_confidence |

## §1.2 Required behaviors

| M | Check | Status |
|---|---|---|
| M1 | Independent claim numbering | C001-C021 (21 distinct) |
| M2 | Every claim cites file:line | Yes, e.g. "build_solver_comparison_table.py:104" for C006 |
| M3 | Every claim has Adversary Pass | Yes, ADVERSARY field per claim + ADVERSARY_NOTES.v1.md cross-cuts |
| M4 | Each deliverable has LAST_REVIEWED + ITERATION header | Yes, all 13 deliverables |
| M5 | Hourly ITERATION_LOG entry | H0-H6 written; H7+ continuing |

## §3 Iteration count

Required: ≥7 passes. Done: H0 (init), H1 (Pass 1 reading), H2 (Pass 2 sanity-recompute + Lanczos), H3 (Pass 3 EVIDENCE_CHAIN + Pass 4 mid-attack), H4 (Pass 5 plan + Pass 7 scripts), H5 (Pass 4+6 adversary), H6 (Pass 8 plan v2 + final summary). **8 passes**.

## §6.1-6.5 Script requirements

| # | Check | Status |
|---|---|---|
| --dry-run | Implemented; passes | Yes |
| --smoke | Implemented in master + per-runner | Yes |
| --resume | Default behavior | Yes |
| --only=X,Y | Implemented | Yes |
| Sanity gate per E0X | sanity_check_matrix() in _common.py | Yes |
| Cold cache | cold_cache() helper (sync + sleep 2) | Yes (best-effort no-sudo) |
| Process pinning | A008: OPENBLAS_NUM_THREADS=1 etc set in master | Yes |
| Reps ≥5 | E02-E06 all set reps=5; E07/E09 deterministic so 1 | Yes |
| OMP_NUM_THREADS recorded | record_environment captures | Yes |
| No silent fallback | E02/E03/E04 sys.exit on missing AMGx; E05/E06 explicit FAIL on missing CHOLMOD | Yes |
| Repetition spread sanity | Computed in compare_to_expected | Partial — could improve |
| Truth file MD5 verify | file_md5() helper, used in E02/E05 | Yes |
| record git commit | record_environment captures | Yes |
| record host info | record_environment captures | Yes |

## §7 Deliverables checklist

| # | Required | Have? |
|---|---|---|
| D1 | READING_NOTES.md | YES (v1, 12 entries) |
| D2 | CLAIM_LEDGER.md ≥v3 | v2 currently — could iterate v3 if more H |
| D3 | EVIDENCE_CHAIN.md ≥v2 | v1 currently — needs v2 |
| D4 | KNOWN_UNKNOWNS.md | YES (v1) |
| D5 | ADVERSARY_NOTES.md ≥v2 | v1 currently — needs v2 |
| D6 | XEON_VALIDATION_PLAN.md ≥v3 | v2 currently — needs v3 |
| D7 | run_xeon_validation.sh | YES (with --dry-run/--smoke/--resume/--only) |
| D8 | xeon_validation/ scripts | YES (11 files, all py_compile clean) |
| D9 | expected_results_template.json | YES (12 predictions with rationale) |
| D10 | RUNBOOK.md | YES |
| D11 | FAILURE_MODES.md | YES (7 sections) |
| D12 | ITERATION_LOG.md ≥12 entries | currently 7 (H0-H6) — need 5+ more |
| D13 | FINAL_SUMMARY.md | YES (v1) |

## §9 Final acceptance checklist

| # | Required | Status |
|---|---|---|
| 1 | bash run_xeon_validation.sh --dry-run PASS | YES (verified twice) |
| 2 | python3 -m py_compile all OK | YES (11 scripts) |
| 3 | Every claim has SOURCE | YES |
| 4 | All 8 SUSPECT in ledger with verdict | YES |
| 5 | expected_results_template.json E01-E11 each has prediction + rationale | 12 predictions covering most experiments. E08 not predicted (post-processing) |
| 6 | RUNBOOK commands bash -n clean | YES |
| 7 | ITERATION_LOG ≥12 entries | NOT YET (currently 7) — must continue |
| 8 | FINAL_SUMMARY ≤2 pages | YES |
| 9 | KNOWN_UNKNOWNS settle path | YES, every U has settlement strategy |
| 10 | All deliverables LAST_REVIEWED + ITERATION header | YES |

## Outstanding gaps to close (H7+)

1. ITERATION_LOG needs ≥5 more entries (continue iterating with real refinements, not padding).
2. CLAIM_LEDGER could go to v3 (incorporate adversary findings — most done in v2 mentally; explicit v3 would document each).
3. EVIDENCE_CHAIN could go to v2 (after H2 measurements: κ now measured, x_LU re-cached — DAG should reflect this).
4. ADVERSARY_NOTES could go to v2 (incorporate Lanczos seed-check result + final v2 plan attacks).
5. XEON_VALIDATION_PLAN could go to v3 (incorporate Lanczos seed-check result + any A011-A020 leftover).

These are substantive improvements, not just versioning.
