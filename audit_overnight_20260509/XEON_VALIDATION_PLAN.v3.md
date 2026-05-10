LAST_REVIEWED: 2026-05-10T15:30+08:00
ITERATION: v3

# Xeon Validation Plan v3 — closeout edition

Changes from v2 (per closeout brief P6 + P8):

## Code-level changes APPLIED to runners (verified by grep, see SELF_CHECK.v2.md)

| Attack | Change | Where applied |
|---|---|---|
| A005 | `vv.block_until_ready()` before `time.perf_counter_ns()` start | E02/E03/E04 |
| A009 | `gc.collect()` + `jax.clear_caches()` between reps | E02/E03/E04/E05/E06 |
| A019 | `"config_full_str": repr(cfg)` in result.json | E02/E03/E04/E05/E06 |
| A014 | Tightened `predicted_range` in expected_results | expected_results_template.v2.json |
| P8 (E09 svds upgrade) | `smallest_k_singulars(A_pos, k=10)` + `near_null_dim_estimate` field | E09 |

## E09 spec upgrade (P8)

E09 now records BOTH metrics per matrix:
- `kappa_lanczos`, `sigma_max_lanczos`, `sigma_min_lanczos` — Lanczos shift-invert
- `smallest_10_sigmas_svds: [σ_1, ..., σ_10]` (ascending) — svds k=10
- `has_near_null_subspace`: True if smallest 5 σ all < 1e-12
- `near_null_dim_estimate`: count of σ < 1e-12 in smallest 10

Pre-registered prediction (in expected_results_template.v2.json):
- lab32: confirms near-null structure (smallest 10 all < 1e-13)
- single_track timesteps: BINARY outcome — either confirms LPBF pd matrices universally
  have near-null structure (important paper finding) or confirms it's specific to lab32
  (rays=0 broken case)

## Open attacks deferred to KNOWN_UNKNOWNS

| Attack | Reason for deferral |
|---|---|
| A001 worst AMGx-vs-LU at picosec ts | Settled by E07/E08 stratification; not a separate experiment |
| A002 SuperLU vs CHOLMOD x_LU bit-equal | E07 already runs LU once; cross-validation with SuperLU adds 70s, optional |
| A004 8K → 500K AMGx generalization | Settled by E07 directly |
| A006 sudo cold-cache flush | NEEDS_SUDO; documented limitation |
| A007 E07 2-rep variance | Resolved by A008 (BLAS pinned single-thread → deterministic) |
| A012 nvidia-smi between reps | Cosmetic; not implemented |
| A021 svds k=10 confirmation | RESOLVED: now part of E09 spec |
| A022 E07 sign-flip explicit assert | Could add a 3-line check; deferred |
| A023 LPBF time-evolution similarity | E04 already records similarity_to_prev |
| A024 OF run-to-run iter variance | Single-run E01; documented in FAILURE_MODES |
| A025 AMGx on sm_120 lab GPU | First-time risk; --smoke catches |

## Final master schedule (v3)

```
Step 0  bash run_xeon_validation.sh --dry-run               # ≤30s, sanity
Step 1  bash run_xeon_validation.sh --smoke                 # ≤5min, pipeline check
Step 2  IF E01 needed:
          bash run_xeon_validation.sh --only=E01            # patches OF case + prints launch
          (manual) cd ~/cases/single_track_dump
                   nohup laserMeltFoam > log.run 2>&1 &     # 7h background
        ELSE skip
Step 3  bash run_xeon_validation.sh                         # E02-E09 foreground ~3h
Step 4  Wait for E01 manual launch to finish               # parallel; ~7h
Step 5  bash run_xeon_validation.sh --only=E01              # parses log.run after step 4 done
Step 6  python3 xeon_validation/compare_to_expected.py \
            --results-dir xeon_validation/results \
            --expected expected_results_template.v2.json \
            --output xeon_validation/analysis/summary.md
Step 7  cat xeon_validation/analysis/summary.md
```

Total wall: max(7h E01 + manual launch, 3h E02-E09) ≈ 7-8 h.

## Outputs (v3 added fields)

```
xeon_validation/results/E09/single_track_*/result.json:
  + smallest_10_sigmas_svds: [10 floats ascending]
  + has_near_null_subspace: bool
  + near_null_dim_estimate: int

xeon_validation/results/E0X/*/result.json (X in 2,3,4,5,6):
  + config_full_str: str (full config representation)
```

## Confidence after v3

Per SELF_CHECK.v2.md attack-resolution table:
- 11 attacks RESOLVED-VERIFIED with grep evidence
- 1 RESOLVED-BY-OTHER (A007)
- 2 PARTIALLY-RESOLVED (A011, A018)
- 2 NOT-RESOLVED-DEFERRED (A006, A012)

Plan v3 is reproducibly executable from scratch via the run_xeon_validation.sh
entrypoint. All scripts py_compile clean. dry-run PASSES.
