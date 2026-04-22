# CANONICAL 128³ — CPU Traditional DILU-PCG Measurement

Added 2026-04-22. Completes the three-way DILU comparison in
CANONICAL_CASE.md §3.1 by adding the CPU single-thread reference point.

## Environment

- Python 3.12.3, numpy 2.4.1, scipy 1.17.0
- CPU: AMD Ryzen 5 5600H with Radeon Graphics, 12 logical cores, 3.29 GHz
- Kernel: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2 Ubuntu on Windows)
- Single-threaded: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
  all set before `import scipy`. `os.environ.setdefault` at top of
  `cpu_dilu_pcg.py` enforces this regardless of the calling shell.
- Driver: `/home/yzk/DILU-Research/dilu/reference/cpu_dilu_pcg.py`
- Matrix: bit-identical (verified at n=4, 8, 16 via `np.array_equal`) to
  `dilu/amgx/tests/_harness.py::stiff_laplacian_3d`. RHS: `np.random.default_rng(0)
  .standard_normal(2097152)`, matching Phase 2 T7 and the multicolor
  64/128 bench.
- Preconditioner: vanilla left-to-right serial DILU factor
  (`D_*[i] = A_ii − Σ_{j<i} A_ij · (1/D_*[j]) · A_ji`) followed by two
  `scipy.sparse.linalg.spsolve_triangular` calls per apply.
- Tolerance: relative residual `‖r‖ / ‖b‖ ≤ 1e-10`. `max_iter = 500`,
  float64.

## Result table

Measured on the 5600H (WSL2, single-threaded):

| Variant              | Iters | Factor (s) | First apply (ms) | PCG wall (s) | Total (factor+PCG) (s) | Ratio vs Phase 2 GPU total (28.47 s) |
|----------------------|------:|-----------:|-----------------:|-------------:|-----------------------:|-------------------------------------:|
| CPU Traditional DILU |   186 |      11.78 |           821.65 |       146.21 |                 157.99 |                    **5.55× slower**  |

Consistency across two independent runs on the same machine: iter count
exact, `rel_res` identical at 8.561e-11, PCG wall 155.0 s / 146.2 s (spread
≈ 6 % — consistent with other-process jitter on a laptop under WSL2).

## Verification

- **Iter count = 186.** Matches Phase 2 cuSPARSE DILU bit-exactly, which is
  the mathematical equivalence test the task brief requested: Phase 2 uses
  cuSPARSE SpSV with level scheduling (parallel), this driver uses scipy's
  sequential triangular solve (serial), and they produce identical DILU
  iterate trajectories because the preconditioner math is the same up to
  FP associativity. The match to the exact count (not ±1, not ±3) is the
  strongest available signal that the 128³ matrix in this script is the
  same matrix Phase 2 solved.
- Final rel residual 8.56e-11 ≤ 1e-10 tolerance. Converged.
- `D_*` range `[5.45, 582.84]` — no zero or negative pivot (required for
  DILU to be a valid preconditioner); matches the expected spread given
  diagonal values ~2–600 on a harmonic-mean-coefficient stencil.

## Interpretation

Two things this data point pins down:

1. **The CPU-vs-GPU gap is modest at this scale, not a chasm.** CPU serial
   DILU costs 5.55× more total wall than Phase 2 cuSPARSE DILU on the RTX
   3050 Laptop. That's a single-order-of-magnitude gap, not the 20–100×
   sometimes quoted for CFD kernels. The reason is that DILU's two triangular
   solves are inherently serial along each level — cuSPARSE parallelizes
   across level width but not across level depth (L_max ≈ N^{1/3} ≈ 128).
   scipy's single-thread solve already captures most of the achievable work
   density on a cached 2M-row matrix. Compare to the 22.1× AMG advantage in
   §3.1: switching from DILU to AMG on GPU beats switching from CPU-DILU to
   GPU-DILU by ~4×. The punchline of §3.1 ("algorithm over hardware") is
   reinforced, not weakened, by this CPU number.

2. **Per-iter cost breakdown on CPU.** PCG wall / iters = 146.2 s / 186 ≈
   786 ms per iter. First-call apply alone is ~820 ms, so >99 % of the
   per-iter time is in the two `spsolve_triangular` calls; SpMV and vector
   ops are noise. This is consistent with the 64³ calibration in-script
   (apply ≈ 92 ms at 64³ → linear-in-N predicted 735 ms at 128³; measured
   820 ms, within 12 %). No surprise, no anomaly.

The factor time (11.78 s) is noteworthy in isolation — at 2M cells the
pure-Python recurrence is already dominating the one-time setup. For a
production CPU DILU we'd drop this to ≤1 s with Cython / Numba / a small
C extension. But factor amortizes to ≈1 s/185 iter ≈ 5 ms/iter, so even
the 12 s Python factor here is <8 % of total wall; not worth optimizing
for a one-shot reference measurement.

**Scope note.** This single number lives under CANONICAL_CASE.md §3.1.
No claim is made about other grids, other physics, or CPU/GPU crossover
for larger problems — the AM-relevant regime is 256³+ where DILU's N^{1/3}
iter-count growth dominates both hardware paths, and AMG is the answer
either way.
