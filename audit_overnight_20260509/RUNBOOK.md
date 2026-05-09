LAST_REVIEWED: 2026-05-10T02:40+08:00
ITERATION: v1

# RUNBOOK — Xeon Validation Tomorrow

This is the morning instruction sheet. Follow each step in order.

---

## 0. Pre-flight (before launching anything)

```bash
ssh yzk@HR54WV2
cd ~/DILU-Research/audit_overnight_20260509   # or wherever you copy this dir
ls
```

Expected files:
- `run_xeon_validation.sh`   (entrypoint)
- `xeon_validation/`         (scripts)
- `CLAIM_LEDGER.v2.md`       (what we're settling)
- `expected_results_template.json` (pre-registered predictions)

## 1. Activate environment

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc   # OF v2412
source ~/jax-env/bin/activate                       # AMGx wrapper
```

Verify:
```bash
python3 -c "from dilu.amgx.python import Plan; print('AMGx OK')"
python3 -c "from sksparse.cholmod import cholesky; print('CHOLMOD OK')"
which laserMeltFoam
```

If any of these fail, **STOP**. See FAILURE_MODES.md §0.

## 2. Verify case dir + npz availability

```bash
# OF case dir for E01
ls ~/cases/single_track_dump/postProcessing/matrices/ | head
# Expected: 1.06e-06  3.2e-07  3.8e-07  4.1e-07  7e-07  9e-07

# 6 npz with AMGx solutions
ls ~/DILU-Research/dilu/amgx/bench/single_*pd_corr0_*.npz
# Expected: 6 files
```

If missing, **STOP**. See FAILURE_MODES.md §1.

## 3. Dry-run

```bash
bash run_xeon_validation.sh --dry-run
```

Expected: `Sanity gate PASS.` + `[DRY-RUN] All inputs verified. Output dir writable. Exiting 0.`

If FAIL: see FAILURE_MODES.md §2.

## 4. Smoke run (~5 min)

```bash
bash run_xeon_validation.sh --smoke
```

This runs each experiment once with 1 rep. Verifies the pipeline works end-to-end.

Check output:
```bash
ls xeon_validation/results/
# Should see E02/, E03/, E04/, E05/, E06/, E07/, E08/, E09/

cat xeon_validation/results/E02/01_3.2e-07/result.json | head -20
# Should see wall_seconds, iter_count, rel_resid, env, etc.
```

If smoke fails: see FAILURE_MODES.md §3.

## 5. Full run

After smoke passes:

```bash
# Default: runs E02-E09 in foreground (~3h), skips E01 (the 7h one)
bash run_xeon_validation.sh

# To include E01 (7h laserMeltFoam re-run for OF wall measurement):
bash run_xeon_validation.sh --only=E01,E02,E03,E04,E05,E06,E07,E08,E09
# E01 backgrounds itself; E02-E09 run in foreground.

# To run JUST one experiment:
bash run_xeon_validation.sh --only=E07
```

Watch progress in another terminal:
```bash
tail -f xeon_validation/logs/*.log
```

## 6. After completion

```bash
# Check what completed
ls xeon_validation/results/*/   # one folder per experiment

# View summary
cat xeon_validation/analysis/summary.md

# Compare to expectations
python3 xeon_validation/compare_to_expected.py \
    --results-dir xeon_validation/results \
    --expected expected_results_template.json \
    --output xeon_validation/analysis/summary.md
cat xeon_validation/analysis/summary.md
```

## 7. Update CLAIM_LEDGER

For each E0X result:

| Result type | Update CLAIM_LEDGER |
|---|---|
| value within predicted_range | upgrade STATUS to VERIFIED + add E0X cite |
| value within 2× of range | mark MARGINAL with comment |
| value outside 2× of range | downgrade STATUS to REFUTED + investigation note |

The interesting outcomes are the marginal/refute cases. They are evidence either of new physics or a bug; treat them as priority-1 for next session.

## 8. Common monitoring commands during long runs

```bash
# Disk space
df -h ~/cases ~/DILU-Research

# Memory
watch -n 5 'free -h; echo; ps aux --sort=-%mem | head -5'

# Process status
ps aux | grep -E 'laserMeltFoam|python.*runner' | grep -v grep

# Latest results.json
find xeon_validation/results -name "result.json" -newer /tmp -mmin -10 | head
```

## 9. Aborting safely

If you need to abort mid-run:
```bash
# Find python processes:
ps aux | grep python | grep runner

# Kill specific PID (preferred — graceful):
kill <PID>

# If unresponsive:
kill -9 <PID>

# E01 (laserMeltFoam) running in nohup:
ps aux | grep laserMeltFoam
kill <PID>

# Already-completed experiments are safe in xeon_validation/results/.
# Re-run with --resume to skip them.
```

## 10. Hand back to user

Upload these files to user (sibling repo / Google Drive / etc.):
- `xeon_validation/results/` (entire dir)
- `xeon_validation/logs/`
- `xeon_validation/analysis/summary.md`
- `xeon_validation/env_*.txt`

The CLAIM_LEDGER updated with concrete numbers makes the project review-ready.
