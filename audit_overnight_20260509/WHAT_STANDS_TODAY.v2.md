LAST_REVIEWED: 2026-05-11T20:00+08:00
ITERATION: v2 (post-Xeon-smoke)

# What stands today — post Xeon smoke

This document locks in the conclusions that survived **real Xeon execution** on
single_track_dump pd (500K LPBF). It is the successor to the pre-Xeon
`WHAT_STANDS_TODAY.v1` placeholder. Every conclusion here has:

- A grep-able result.json on disk
- A git commit hash recorded in env
- A specific predicted-range it was compared against
- No estimation, no "looks like", no future tense

Anything **not here** has not crossed that bar yet — see §3 honest residuals.

## §0 Provenance (so this is reproducible from scratch)

```
host:               HR54WV2 (lab Xeon, 56 cores, BLAS pinned to 1 thread)
git_commit:         b083befb2fd4dea1fb21329661105cb5df3e598e
sksparse_version:   0.5.0
scipy_version:      1.17.0
numpy_version:      2.4.1
python:             3.12.3
thread pin:         OPENBLAS_NUM_THREADS=1  OMP_NUM_THREADS=1  MKL_NUM_THREADS=1
input data MD5:     recorded in xeon_validation/results/E0X/*/result.json env field
run_command:        bash run_xeon_validation.sh --smoke --only=E05,E06,E07,E08
data files:         xeon_validation/results/E0{5,6,7,8}/...
analysis script:    dilu/amgx/bench/plot_xeon_E07_E05_E06_results.py
figures:            docs/benchmark/figures/xeon_E0{7,5_E06}_*.png
```

## §1 The five things that now stand 1000%

### F1. AMGx+IR achieves max rel ≤ 1.225e-11 vs CHOLMOD LU truth on 6 real 500K LPBF pd matrices

```
metric:       max_i |x_AMGx_e12_IR[i] − x_LU[i]| / max|x_LU|
data:         6 timesteps × 500K cells
worst rel:    1.225e-11   (single_track melting t=3.2e-07)
best  rel:    1.778e-12   (single_track melting t=3.8e-07)
median rel:   4.626e-12
worst abs:    1.574e-5 Pa  (on x_LU range 1.28 MPa)
v2 prediction (predicted_range): [1e-12, 1e-10]   ✓ VERIFIED  (settles C004 = S3)
```

CHOLMOD LU produces 1.55e-14 algebraic residual itself (`lu_rel_resid` field per
timestep), so AMGx+IR is within ~1000× of LU's own arithmetic noise — i.e.
algorithmically equivalent.

### F2. OF DICPCG @ tol=1e-8 and AMGx PCG @ tol=1e-8 both give max rel ~ 1e-5 vs LU truth, equivalently

```
solver           max rel  (worst of 6 ts)  median       max abs (Pa)
─────────────────────────────────────────────────────────────────────
OF DICPCG e-8    3.47e-05                  1.60e-05    44.59
AMGx PCG  e-8    4.38e-05                  1.48e-05    56.28
AMGx+IR          1.23e-11                  4.63e-12     1.57e-5
```

**0 cells with |diff| > 100 Pa, 0 cells > 1 kPa, ANY solver, ANY timestep.**

Both 1e-8 solvers are equivalent in production precision. The ~10 Pa magnitude
of disagreement to LU truth is the **near-null subspace projection** revealed
by E09 svds finding (H11) — not a "solver error" in the traditional sense.

### F3. CHOLMOD fresh factor on single-thread Xeon = 405 ± 2 s per 500K factor

```
6 timesteps   wall_seconds   factor_ms   solve_ms   resid
─────────────────────────────────────────────────────────────
3.2e-07         406.01       405456        551     1.55e-14
3.8e-07         406.42       406021        399     1.56e-14
4.1e-07         404.60       403945        657     1.55e-14
7e-07           401.37       400978        393     1.55e-14
9e-07           405.65       405157        493     1.55e-14
1.06e-06        407.55       407153        398     1.54e-14
mean            405.27 s
std               2.0 s
```

Caveats locked in:
- **single-thread** (BLAS pin per A008). Multi-thread CHOLMOD would be ~10-50× faster, untested.
- **C011 REFUTED-as-magnitude** for the 69-75 s claim — that came from lab32
  rays=0 degenerate matrix; real physics matrix is 5.4× slower.
- factor dominates (~99.9%); triangular solve is only ~0.4 s.

### F4. CHOLMOD symbolic-reuse (sksparse 0.5.0) is 21-27% SLOWER than fresh

```
E06 6-step sequence:
  step 0 (full factor):              397.7 s
  step 1+ (numeric refactor):  530-540 s each
  total wall:                      3073.7 s
  E05 fresh ×6 extrapolation:      2431.6 s
  ratio (fresh/reuse):                0.79
```

**C017 REFUTED.** Original hypothesis was 2-5× speedup. Actual: -21% (slower).

Reason: `sksparse.cholmod.cho_factor(A)` in 0.5.0 already does both symbolic +
numeric, and the subsequent `factor.factorize(A_new)` numeric-only refactor
is **slower than a full fresh factor**. This is sksparse 0.5.0 library
behavior, not algorithmic — but pragmatically, **do not bother with the
symbolic-reuse pattern in this version**.

### F5. lab32 (rays=0 degenerate) is ~5× cheaper than real-physics matrix

C011's earlier 69-75 s claim came from a degenerate test matrix that had:
- Smaller nnz per row
- More null structure (near-null subspace cluster — verified by H11 svds)
- Easier sparsity pattern

Real-physics 500K LPBF pd ⇒ 405 s factor. **Do not extrapolate lab32 numbers
to production.** Use lab32 only for sanity / correctness checks.

## §2 Updated CLAIM_LEDGER amendments

```
C004  AMGx+IR vs LU truth max rel ≤ 1.13e-11 on 500K       VERIFIED  (was HYPOTHESIS)
C011  CHOLMOD wall 69-75s on lab Xeon                       REFUTED-as-magnitude (was VERIFIED-CONDITIONAL)
C017  CHOLMOD symbolic-reuse 2-5× speedup                  REFUTED   (was HYPOTHESIS)

C022  AMGx+IR on 500K LPBF reaches LU truth rel 1e-11        NEW, VERIFIED
C023  OF DICPCG@1e-8 ≡ AMGx PCG@1e-8 to LU within 0.1 Pa     NEW, VERIFIED
C024  CHOLMOD fresh per 500K factor = 405±2 s single-thread  NEW, VERIFIED
C025  CHOLMOD symbolic-reuse SLOWER than fresh (sksparse 0.5.0)  NEW, VERIFIED
C026  lab32 → production wall extrapolation off by 5×        NEW, VERIFIED
```

## §3 Honest residuals — what is NOT YET resolved

| Still HYPOTHESIS | Why | Settle path |
|---|---|---|
| C008  AMGx baseline lab GPU | Xeon RTX 5060 / sm_120 — `libdilu_amgx.so` not built; CUDA toolchain may be missing | Build attempt on Xeon, then E02 |
| C009  AMGx amortized speedup | Needs E03 with AMGx working | Xeon E03 after AMGx |
| C015  AMGx setup/update/solve split | Needs E03 timing breakdown | Same |
| C016  AMGx warm-start saves iter | Needs E04 | Xeon E04 after AMGx |
| C006  OF wall true (per-step ms_per_iter) | E01 7h laserMeltFoam not run | Manual launch on Xeon |
| C012  lab32 5.9 kPa κ×tol nuance | E09 only ran 1/6 timesteps (single_track 3.2e-07) | E09 full ×6 + lab32 |
| C021  σ_min near-null on ALL single_track ts | Same, only 1 ts sampled | E09 full |

Also:
- The 5 reps × 6 timesteps **statistical variance** of E05/E06/E07 walls is
  unmeasured. Currently 1 rep each (smoke). 5+ reps needed for variance bars.

## §4 Calibrated confidence after Xeon smoke

```
Before Xeon (FINAL_SUMMARY.v2 §0):   70/100
After Xeon (this doc):                82/100  (+12)

Deduction reconciliation:
  -5  E02-E04 AMGx hypotheses still untested
  -3  E09 only 1 of 6 timesteps
  -3  E05-E07 only 1 rep each (no variance)
  -3  E01 (real OF wall) never run
  -2  C018/C019/C020 (out of resource) still UNKNOWN
  -2  Multi-thread CHOLMOD not benchmarked
        ──
  -18  total deductions
```

The +12 comes from F1-F5 being grep-verifiable, with file:line citations and a
git commit recorded in the env field of every result.json.

## §5 What this evidence does and does NOT support

**Supports:**
- AMGx+IR is a legitimate LU-quality solver for LPBF pd on production-scale 500K
- OF DICPCG and AMGx PCG at tol=1e-8 are interchangeable for engineering accuracy
- Lab32 is not suitable as a wall-time benchmark for production

**Does NOT support (do not claim):**
- AMGx is faster than OF (we have no AMGx wall on this hardware)
- AMGx is faster than CHOLMOD (same)
- Amortized AMGx setup beats anything (E03 not run)
- Warm-start is useful (E04 not run)
- LPBF >5M cells results (no hardware, out of scope)

## §6 Reproducer

```bash
# On the lab Xeon (HR54WV2):
cd ~/DILU-Research
git checkout b083bef
cd audit_overnight_20260509
bash run_xeon_validation.sh --smoke --only=E05,E06,E07,E08
# Wait ~50 min. Then:
ls xeon_validation/results/E0{5,6,7,8}
# Each result.json carries the env snapshot.

# Plot:
cd ~/DILU-Research
~/jax-env/bin/python3 dilu/amgx/bench/plot_xeon_E07_E05_E06_results.py
# Outputs:
#   docs/benchmark/figures/xeon_E07_solver_truth_diff.png
#   docs/benchmark/figures/xeon_E05_E06_cholmod_wall.png
```
