LAST_REVIEWED: 2026-05-10T03:30+08:00
ITERATION: v2

# Xeon Validation Plan v2 — incorporating Pass-6 adversary feedback

Changes from v1:
- A008: thread pinning (`OPENBLAS_NUM_THREADS=1` etc.) at script entry
- A017: write `manifest.json`
- A013/A015: E08 now computes headline ratios (amortized speedup, AMGx_warm vs LU)
- A011: E01 should be smoke-tested on 5ns first
- A019: each runner records `config_full_str = repr(cfg)` in result.json
- A009: AMGx runners gc.collect() between reps
- E03/E04 max_iter raised to 5000 (from 2000) per A003

Plan structure unchanged. Implementation refined.

(See v1 for the experiment specifications. v2 only adds the above improvements.)

## v2 Master schedule

```
Step 0  bash run_xeon_validation.sh --dry-run             (≤1 min, sanity)
Step 1  bash run_xeon_validation.sh --smoke               (≤5 min, pipeline check)
Step 2  IF E01 needed:                                     
            bash run_xeon_validation.sh --only=E01 --smoke  (5 ns OF, ~5 min)
            ↳ verifies controlDict patch + solverInfo writeable
Step 3  bash run_xeon_validation.sh --only=E01            (full 1.2 μs, ~7h, nohup background)
Step 4  bash run_xeon_validation.sh                       (E02-E09, ~3h foreground)
Step 5  Wait for E01 to finish (if not already done)
Step 6  python3 xeon_validation/compare_to_expected.py    (auto-runs at end of step 4)
Step 7  cat xeon_validation/analysis/summary.md           (review)
```

Total wall: max(7h E01 background, 3h E02-E09 foreground) ≈ 7-8 h.

## Robustness checks built in

- Sanity gate: each E0X verifies `‖A·x_truth - b‖/‖b‖ < 1e-7` on input matrices before solving.
- Atomic writes: result.json written via .tmp + os.fsync + rename.
- Resume: existing result.json files skipped on `--resume` (default).
- Fail-loud: missing solver → SystemExit; no fallback fake numbers.
- Cold-cache: `sync; sleep 2` between reps (best-effort without sudo).
- Repetitions: ≥5 for wall-time experiments; 1 for deterministic (E07, E09).

## Outputs

```
xeon_validation/
├── env_<ts>.txt
├── manifest.json
├── results/
│   ├── E02/{rep}_{t}/result.json
│   ├── E03/rep_NN/result.json
│   ├── E04/rep_NN/result.json
│   ├── E05/{rep}_{t}/result.json
│   ├── E06/rep_NN/result.json
│   ├── E07/{NN}_{t}/result.json + aggregate.json
│   ├── E08/result.json + diagnostics.csv + speedup_ratios.json
│   ├── E09/single_track_*/result.json + lab32_melt_380ns/result.json + aggregate.json
│   └── E01/01/result.json (if E01 enabled)
├── logs/E0X_*.log
└── analysis/summary.md
```
