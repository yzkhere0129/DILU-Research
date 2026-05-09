LAST_REVIEWED: 2026-05-10T02:50+08:00
ITERATION: v1

# Failure Modes — Diagnostic & Recovery

For each common failure during Xeon validation, the diagnostic command + recovery action.

---

## §0 — Environment activation fails

### Symptom: `python3 -c "from dilu.amgx.python import Plan"` raises ImportError

Diagnostic:
```bash
echo $PYTHONPATH
which python3
python3 -c "import sys; print(sys.path)"
ls ~/DILU-Research/dilu/amgx/python/
```

Causes:
- jax-env not activated → `source ~/jax-env/bin/activate`
- DILU-Research not in PYTHONPATH → `cd ~/DILU-Research; export PYTHONPATH=$(pwd):$PYTHONPATH`
- AMGx shared library missing → check `find ~ -name "libamgx*.so" 2>/dev/null` and set `LD_LIBRARY_PATH` if needed

### Symptom: `from sksparse.cholmod import cholesky` raises ImportError

Diagnostic:
```bash
python3 -c "import sksparse; print(sksparse.__version__)"
pip show scikit-sparse
```

Recovery:
```bash
sudo apt-get install -y libsuitesparse-dev   # one-time
pip install --user scikit-sparse
```

### Symptom: `which laserMeltFoam` empty

Recovery:
```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
# Or for v2506:
source /usr/lib/openfoam/openfoam2506/etc/bashrc
```

---

## §1 — Missing input files

### Symptom: dry-run fails with `FAIL: missing input npz`

Diagnostic:
```bash
ls ~/DILU-Research/dilu/amgx/bench/single_*pd_corr0_*.npz
```

Recovery:
```bash
cd ~/DILU-Research && git pull origin main
# npz files are committed to git (78 MB total); pull should restore them.
```

### Symptom: `FAIL: E01 selected but OF_CASE has no postProcessing/matrices`

Diagnostic:
```bash
ls ~/cases/single_track_dump/
```

Recovery:
- Use `OF_CASE=<other-path> bash run_xeon_validation.sh ...` to override
- Or skip E01: `bash run_xeon_validation.sh --only=E02,E03,E04,E05,E06,E07,E08,E09`

---

## §2 — Sanity gate fails

### Symptom: `SANITY GATE FAILED on input: ‖A·x_truth - b‖/‖b‖ = 1.5e-3 > 1e-7`

Diagnostic:
```bash
ls -la ~/cases/single_track_dump/postProcessing/matrices/3.8e-07/pd_corr0/
md5sum ~/cases/single_track_dump/postProcessing/matrices/3.8e-07/pd_corr0/A.mm
```

Causes:
- A.mm corrupted (partial write) → re-extract from tar/git
- Wrong x_final.mm being checked → script bug, escalate

Recovery:
- Re-rsync OF case dump from a known-good source
- Or skip that timestep: pass `--smoke` to use only one matrix at a time

---

## §3 — Smoke run hangs or crashes

### Symptom: AMGx Plan() hangs during E02 smoke

Diagnostic:
```bash
nvidia-smi    # is GPU oversubscribed?
ps aux | grep python | grep runner
```

Likely cause: GPU memory exhausted or contention with another process.

Recovery:
- `kill <PID>` of the Python runner
- Wait for GPU to clear: `nvidia-smi --query-gpu=memory.used --format=csv` should drop
- Re-run with `--resume`

### Symptom: CHOLMOD `cholesky_inplace` raises numerical exception

Cause: matrix has near-singular pivots; symbolic factorization OK but numerical step fails.

This is captured by E06 itself — script catches the exception, falls back to full factor, records it in `warnings`. **Not a failure**, just diagnostic.

If it falls back ALL 5 times: matrix may be too ill-conditioned for symbolic-reuse to be useful. C017 prediction REFUTED — that's a valid scientific outcome.

### Symptom: result.json is malformed JSON

Diagnostic:
```bash
python3 -m json.tool < xeon_validation/results/E02/01_3.2e-07/result.json
```

Cause: process killed mid-write. The atomic-write pattern (.tmp + rename) should prevent this; if it occurred, check filesystem.

Recovery:
- Delete the malformed result.json
- Re-run with `--resume`

---

## §4 — Long-running E01 issues

### Symptom: laserMeltFoam crashes mid-simulation

Diagnostic:
```bash
tail -100 ~/cases/single_track_dump/log.run
```

Common causes:
- Numerical divergence (NaN in some field) → reduce dt or check initial conditions
- Out of disk → `df -h ~/cases`
- OOM (unlikely on lab Xeon for 500K but possible) → `dmesg | grep -i killed`

Recovery: any partial dumps still in `~/cases/single_track_dump/postProcessing/matrices/` are usable. Run E01 in `--skip-rerun` mode to parse the partial log.

### Symptom: solverInfo function block conflicts with existing controlDict

Diagnostic:
```bash
grep -A 15 "functions" ~/cases/single_track_dump/system/controlDict
```

If there's already a `functions { ... }` block, the script's append doesn't merge — it just appends a second `functions` block, which OF rejects.

Recovery:
- Manually merge the solverInfo block into existing functions{}.
- Or restore backup: `cp system/controlDict.pre_E01 system/controlDict`

---

## §5 — Result interpretation

### Symptom: AMGx amortized speedup is 1.0× (no benefit)

This **refutes** S1 / C009 prediction. Investigate:
- Is `update_coefficients` actually being called? Check JSON `per_step[0].setup_ms` should be ≫ `per_step[1..N].update_ms`
- Is the AMG hierarchy being kept? Watch for warnings about cache invalidation
- Is the matrix structure changing between steps? `nnz` should be identical across all 6 timesteps

If the speedup truly is 1×, write up as a NEGATIVE result — that's still science.

### Symptom: AMGx + 1 IR rel diff vs LU > 1e-9

This **refutes** C004 / S3. Possible causes:
- AMGx implementation bug → re-run E02 with tighter tol, see if rel_resid actually reaches 1e-15
- LU truth itself is wrong → verify `sanity_check_matrix(A, x_LU, b)` passes
- Sign-flip mismatch → check `sign_flipped` field in npz meta

### Symptom: κ measurement varies wildly across reps

Lanczos with `tol=1e-3` gives ±1 order of magnitude variability. Tighten tol to 1e-5 (slower) for production-quality κ. For our purposes order-of-magnitude is enough.

---

## §6 — Reporting back

If experiment NN failed despite recovery attempts:

1. Capture: `xeon_validation/logs/E0NN_*.log` (full stderr/stdout)
2. Capture: `xeon_validation/results/E0NN/.../result.json` (any partial output)
3. Capture: `xeon_validation/env_*.txt` (full environment snapshot)
4. Note in `ITERATION_LOG.md`: which experiment, what symptom, what recovery tried
5. Mark CLAIM_LEDGER claims that depended on it as STATUS: BLOCKED-ON-XEON-FAILURE

---

## §7 — Worst-case fallback

If everything fails on Xeon, drop back to dev:

```bash
ssh yzk@<dev>
cd ~/DILU-Research/audit_overnight_20260509
# Run only what RTX 3050 can handle:
bash run_xeon_validation.sh --only=E02,E07,E09 --smoke
```

E02 (single-shot AMGx), E07 (LU-vs-AMGx diff), E09 (κ) all run on dev hardware. E03/E04 also possible but slower. E01 needs lab Xeon (single-core OF benchmark). E05/E06 need lab Xeon for proper CHOLMOD speed.

This will at least re-establish the most fragile claims. Full deliverable will be reduced.
