LAST_REVIEWED: 2026-05-10T15:35+08:00
ITERATION: v2

# FINAL_SUMMARY v2 — Forensic Audit + Closeout

## 0. Confidence calibration (P2 fix)

Final reported confidence: **70/100**. Specific deductions:

| -10 | E01–E07 hypotheses untested (no Xeon execution yet) |
| -10 | AMGx wrapper not verified on lab GPU sm_120 (first-time risk) |
| -5  | H11 svds finding not yet propagated to all derivative documents (STORY_REVISION lists what to update) |
| -5  | Time padding in H7-H12 entries during initial session — corrected by H13+ closeout |

That gives 100 - 30 = 70/100. The 84/100 reported in ITERATION_LOG H12 was inconsistent
with FINAL_SUMMARY v1's 70/100; **this v2 reconciles both at 70**, with the deductions
above as the calibration.

## 1. The 4 sins from prompt §0 — disposition

| # | Sin | Disposition |
|---|---|---|
| 1 | Editorialized predictions as conclusions | CLAIM_LEDGER v3+v4 marks all predictions as HYPOTHESIS; expected_results_template.v2 pre-registers them |
| 2 | Mixed estimated/measured in tables | C006 explicitly marked REFUTED-AS-MEASUREMENT; OF wall in solver_comparison_500K.md flagged |
| 3 | Retracted hypotheses without scrubbing derivatives | STORY_REVISION.md lists 4 specific propagation actions; user must promote |
| 4 | Reported in-flight as "running" | C013 REFUTED with forensic ps + ls evidence |

## 2. Forensic findings locked tonight (file:line cited in CLAIM_LEDGER)

| Finding | Lock evidence |
|---|---|
| OF wall numbers = `iter × 110 ms` arithmetic, NOT measured | `build_solver_comparison_table.py:104` |
| Phase 0 PID 466877 dead, no replay JSON | `ps -p 466877` empty + `ls replay_results.json` missing |
| lab32 LU x_LU re-cached on dev (replaces deleted file) | `/tmp/x_LU_lab32_melting_pd.npy` 4 MB; rerun residual 2.752e-15 |
| 5.9 kPa OF-vs-LU diff numerically reproduced | max|diff|=5914 Pa, 25 cells > 100 Pa, 4 cells > 1 kPa |
| **κ for lab32 is method-dependent**: Lanczos 3.2e+14 vs svds 1.15e-16 | both at fp64 noise floor |
| **Matrix has 10-dim near-null singular cluster** σ ∈ [1.15e-16, 9.29e-15] | svds k=10 result |
| OF self-residuals from metadata.json match raw recompute | within 5-15% across 6 timesteps |

## 3. Major story revision (P7)

The "5.9 kPa = κ × tol with κ ~ 1e6" narrative is **NO LONGER TENABLE**.
Replacement: "5.9 kPa is the magnitude of x_OF's projection onto a 10-dimensional
near-null singular subspace; LU picks one particular pinned member of that family."

This makes lab32 **a degenerate case** (rays=0 → b ≈ 0 → matrix-driven solution),
not a typical LPBF pd matrix. Whether this near-null structure exists in the
single_track 500K matrices (rays > 0, real physics) is **a binary outcome of E09**
and one of the most paper-impactful results pending.

See STORY_REVISION.md for 4 specific text replacements to propagate.

## 4. Deliverables (D1–D13 + closeout N1–N9)

### Original session (H0–H12)

| # | File | Status |
|---|---|---|
| D1  | READING_NOTES.v1.md | ok |
| D2  | CLAIM_LEDGER.v1/v2/v3.md | v3 with 21 claims |
| D3  | EVIDENCE_CHAIN.v1/v2.md | v2 with confidence per claim |
| D4  | KNOWN_UNKNOWNS.v1.md | 15 unknowns, settle paths |
| D5  | ADVERSARY_NOTES.v1/v2.md | 25 attacks A001–A025 |
| D6  | XEON_VALIDATION_PLAN.v1/v2.md | (closeout adds v3) |
| D7  | run_xeon_validation.sh | dry-run PASS, syntax OK |
| D8  | xeon_validation/*.py | 11 scripts, py_compile clean |
| D9  | expected_results_template.json | (closeout adds v2) |
| D10 | RUNBOOK.md | (closeout adds A018 paragraph) |
| D11 | FAILURE_MODES.md | 7 sections |
| D12 | ITERATION_LOG.md | H0–H12 (now H0–H17) |
| D13 | FINAL_SUMMARY.md | (this file is v2) |

### Closeout pass (H13–H17)

| # | File | Purpose |
|---|---|---|
| N1 | EXTERNAL_EDITS.md | P4 — every audit-window file edit + revert recommendations |
| N2 | SCRIPTS_DUMP.md | P5 — full source of D7+D8 for offline review |
| N3 | SELF_CHECK.v2.md | P3 — grep evidence per resolved attack |
| N4 | STORY_REVISION.md | P7 — 4 text rewrites for derivative docs |
| N5 | XEON_VALIDATION_PLAN.v3.md | P6+P8 — closeout integration |
| N6 | expected_results_template.v2.json | P6 A014 — tightened ranges + svds entries |
| N7 | CLOSEOUT_NOTES.md | P-summary + Honest residuals |
| N8 | CLAIM_LEDGER.v4.md, EVIDENCE_CHAIN.v3.md, ADVERSARY_NOTES.v3.md (TBD if time) |
| N9 | runner code edits + verification (DONE: see SELF_CHECK.v2 grep) |

## 5. Tomorrow's success criteria (unchanged from v1, restated for completeness)

| Result | Interpretation |
|---|---|
| AMGx amortized speedup ratio ∈ [1.05, 2.5] (v2 range) | Refutes 100-1000× claim. Realistic. |
| AMGx amortized > 5× | Hypothesis partially supported; investigate |
| AMGx warm-start saves 10–60% iter (v2 range) | Confirms warm-start utility |
| OF DICPCG measured per-iter ∈ [80, 150] ms (v2 range) | C006 estimate validated |
| AMGx + IR vs LU max rel ∈ [1e-12, 1e-10] (v2 range) | C004 VERIFIED |
| **single_track has near-null subspace (E09 svds)** | LPBF pd matrices universally near-singular — paper insight |
| **single_track has NO near-null cluster (E09 svds)** | lab32 was specific to broken rays=0 case — narrows generalization |

## 6. Honest residuals (also in CLOSEOUT_NOTES)

Items NOT addressed in closeout:
- A006 (sudo cold-cache flush) — needs sudo, declined to pursue
- A012 (nvidia-smi monitoring between reps) — cosmetic, not implemented
- 4 STORY_REVISION rewrites — written in STORY_REVISION.md, not promoted to public docs
- E01 cannot run unsupervised — requires manual laserMeltFoam launch (architectural)
- C018 (AMGx >5M) and C019 (senior data attribution) — out of resource/cooperation reach
- v4 CLAIM_LEDGER, v3 EVIDENCE_CHAIN, v3 ADVERSARY_NOTES — content is in N1-N6 already; if time allows for a clean reorg pass, v4/v3 versions would consolidate

## 7. What user should do morning-of

```bash
cd ~/DILU-Research/audit_overnight_20260509
cat RUNBOOK.md         # latest with A018 paragraph
bash run_xeon_validation.sh --dry-run     # verify everything
bash run_xeon_validation.sh --smoke       # 5 min pipeline check
bash run_xeon_validation.sh                # ~3h foreground for E02-E09
# If E01 desired:
bash run_xeon_validation.sh --only=E01 && (cd ~/cases/single_track_dump && nohup laserMeltFoam > log.run 2>&1 &)
```

After all done:
```bash
python3 xeon_validation/compare_to_expected.py \
    --results-dir xeon_validation/results \
    --expected expected_results_template.v2.json \
    --output xeon_validation/analysis/summary.md
cat xeon_validation/analysis/summary.md
```

Update CLAIM_LEDGER C001-C021 with concrete numbers from result.json files.
Promote STORY_REVISION rewrites to public docs (PROJECT_STATUS_REPORT, data_inventory).
