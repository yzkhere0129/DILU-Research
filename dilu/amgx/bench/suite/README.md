# DILU-Research Benchmark Suite v1

Standardized 50-matrix LPBF solver benchmark for cross-hardware comparison.

## Quick numbers

- **50 matrices** sampled across 4 physical phases (pre_melt 10 / melt 15 / evap_early 10 / evap_late 15)
- Both **pd_corr0** + **T_corr0** equations
- **Binary npz** format (~3 GB total, 10× smaller than ASCII)
- **Cross-HW portable** runner: dev 3050 / lab 5060 / H100 / any CUDA + jax

## Files

```
suite/
├── README.md                   (this file)
├── manifest.json               50 matrices x4 phases breakdown
├── select_50_matrices.py       (Xeon) pool → manifest
├── convert_mm_to_npz.py        (Xeon) ASCII → binary
├── run_benchmark.py            (any HW) 4 protocols × 50 matrices
├── compute_truth_err.py        (any HW) x vs truth diff
└── plot_results.py             (any HW) standard figures
```

## Workflow

### Once (build suite on Xeon)

```bash
# 1. Generate manifest (selects 50 from 384 sane pool)
PYTHONPATH=. python dilu/amgx/bench/suite/select_50_matrices.py \
    --pool validate_dense/sane_pool.txt \
    --out  dilu/amgx/bench/suite/manifest.json

# 2. Convert MM → npz on Xeon (where dense_track_dump_500K lives)
PYTHONPATH=. python dilu/amgx/bench/suite/convert_mm_to_npz.py \
    --case ~/cases/dense_track_dump_500K \
    --manifest dilu/amgx/bench/suite/manifest.json \
    --out ~/benchmark_suite_v1

# 3. tar + distribute to test hardware
cd ~ && tar czf benchmark_suite_v1.tgz benchmark_suite_v1/
# scp benchmark_suite_v1.tgz to lab 5060 / H100 / ...
```

### Per hardware (run benchmark)

```bash
# On any GPU + jax + AMGx FFI built host:
PYTHONPATH=. python dilu/amgx/bench/suite/run_benchmark.py \
    --suite ~/benchmark_suite_v1 \
    --out   results_$(hostname)_$(date +%Y-%m-%d) \
    --equations pd,T \
    --protocols fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR \
    --max-iters 3000 \
    --save-x    # only for truth-diff precision computation

# Then compute precision err
PYTHONPATH=. python dilu/amgx/bench/suite/compute_truth_err.py \
    --suite ~/benchmark_suite_v1 \
    --results results_$(hostname)_$(date +%Y-%m-%d)
```

Estimated wall on 5060: pd 50-matrix < 1 min (no I/O), T 50-matrix < 30 s.

### Cross-HW report

```bash
PYTHONPATH=. python dilu/amgx/bench/suite/plot_results.py \
    --results-dir results/    # contains multiple hostname subdirs
    --output     docs/benchmark/figures/suite_v1_cross_hw.png
```

## Protocols

| Name | tol | n_IR | Amortized | Use |
|---|---:|---:|---|---|
| fresh_e8 | 1e-8 | 0 | no | engineering baseline |
| amortized_e8 | 1e-8 | 0 | yes | production speed |
| fresh_e12_IR | 1e-12 | 1 | no | precision baseline |
| amortized_e12_IR | 1e-12 | 1 | yes | **truth-level (rel_resid 1e-16)** |

## What this gives you

- **wall_compare** plot: 4-5 solvers × 50 matrices, mean wall + variance
- **precision_compare** plot: max |x - x_truth| per protocol per phase
- **per-step trajectory** plot: wall + iter evolution (50 ordered steps)
- **diff slice** plot (LU vs AMGx_e12 style): pick a representative matrix from each phase

## Storage

| Item | Size |
|---|---:|
| ASCII source on Xeon | ~96 GB |
| binary suite npz (50 matrices × 2 eq) | ~3 GB |
| tarball for transfer | ~1.5 GB |
| per-HW results dir | ~50-200 MB (depends on --save-x) |
