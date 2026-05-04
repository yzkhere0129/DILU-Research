# 04 — Code, Data, and Deploy Bundle Pointers

All paths are relative to the repository root `/home/yzk/DILU-Research/`.

## C++ side (OpenFOAM patches)

| File | Purpose |
|------|---------|
| `dilu/benchmark/openfoam_crosscheck/deploy/matrixDumper.H` | The header defining the dumper class. Drop-in include. |
| `dilu/benchmark/openfoam_crosscheck/deploy/laserMeltFoam.C.diff` | Patch: includes the header + creates `matrixDumper_` instance |
| `dilu/benchmark/openfoam_crosscheck/deploy/pEqn.H.diff`        | Patch: wraps `pdEqn.solve()` with `dumpPreSolve` / `dumpPostSolve` + `std::chrono` timing |
| `dilu/benchmark/openfoam_crosscheck/deploy/TEqn.H.diff`        | Same for `TEqn.solve()` |

The `.pristine` and `.patched` files in the same directory let you verify a
clean apply (`diff` them).

The header itself is also exposed at `LaserbeamFoam/applications/solvers/laserMeltFoam/matrixDumper.H`
in the user's working build tree (outside the repo, but kept for reference).

## Python drivers (our GPU solvers)

All under `dilu/benchmark/openfoam_crosscheck/`:

| File | Solver | Notes |
|------|--------|-------|
| `driver_cusparse.py`            | cuSPARSE PCG (DILU precond)         | `jax.lax.while_loop` + `@jax.jit`. `SPMV_BACKEND=seg_static` is the fast path (precomputed `row_of`). |
| `driver_multicolor.py`          | Multicolor DILU PCG                 | Same template as cuSPARSE, swap precond. |
| `driver_cusparse_bicgstab.py`   | cuSPARSE BiCGStab                   | For non-symmetric matrices (T equation). |
| `driver_multicolor_bicgstab.py` | Multicolor DILU BiCGStab            | Same. |
| `driver_amgx.py`                | NVIDIA AMGx                         | `--amortize` reuses one `AmgxPlan` across timesteps via `update_coefficients`. `--allow-t-eq` for BiCGStab outer. |
| `driver_scipy.py`               | scipy reference (CPU)               | sanity check only. |
| `reader.py`                     | npz/MTX loader, sign normalize      | Auto-negate when all `diag ≤ 0` (OpenFOAM Laplacian convention). |
| `aggregate_results.py`          | walks `<case>/<time>/<eq>_corr*/results/*.json` and prints medians |
| `compare.py`, `run_all.py`      | top-level orchestrators            |

AMGx config additions are in `dilu/amgx/python/config.py`:
- `CLASSICAL_V_DIAGSCALED` — the **only** config that gives correct physics on `pd`
- `CLASSICAL_V_DIAGSCALED_BICGSTAB` — same for `T`
- The "wrong" configs are kept (registered but documented as failing) so the
  history of what was tried is recoverable.

## Data

`dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/`

```
<time>/
├── pd_corr0/    pd_matrix.npz, pd_rhs.npz, pd_x_final.npz
├── pd_corr1/    ...
├── pd_corr2/    ...
└── T_subiter*/  T_matrix.npz, T_rhs.npz, T_x_final.npz
```

50 pd npz + 100 T npz, total 1.85 GB after compression (compared to ~80 GB if
left as MatrixMarket ASCII; `shrink_dump.py` does the conversion).

Per-matrix solver outputs land in `<time>/<eq>_corr*/results/{cusparse,multicolor,amgx_*}.json`
each containing `iter`, `solve_s`, `rel_vs_OF`, `selfres`.

## Deploy bundle (for lab machine, unattended overnight runs)

`dilu/benchmark/openfoam_crosscheck/deploy.tar.gz` (30 KB, 18 files):

```
deploy/
├── matrixDumper.H           # the header
├── *.diff                   # 3 patches
├── *.pristine, *.patched    # verification baselines
├── install.sh               # apply patches + build laserMeltFoam
├── patch_case.sh            # turn an OpenFOAM case into a "dump-mode" case
├── sanity.py                # one-matrix consistency check (Ax_OF ≈ b ?)
├── shrink_dump.py           # MTX → npz conversion
├── multicase_batch.sh       # iterate over a CASES=(...) array, dump each
├── README.md                # bundle overview
└── TUTORIAL.md              # step-by-step recipe for a new case
```

The lab operator only needs: `tar xzf deploy.tar.gz && bash install.sh` once,
then for each case `bash patch_case.sh <caseDir>` and run as normal. The
`multicase_batch.sh` automates the whole CASES array.

## Specs and reports (the documents themselves)

- `docs/specs/MATRIX_EXTRACTION_SPEC.md` — the methodology, reproduction-grade
- `docs/benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md` — canonical results
- `docs/session_logs/SESSION_HANDOFF_20260428.md` — full multi-day handoff
- `docs/reference/Matrix_Output_Guide_external.md` — the PCG-internal-hook
  variant from a colleague (Mei Yang); kept for cross-check / comparison

## Quick-start for the partner

If they want to reproduce on a **new case**:

```bash
# on the lab machine, with OpenFOAM v2506 sourced
tar xzf deploy.tar.gz
cd deploy
bash install.sh                              # patches + builds laserMeltFoam
bash patch_case.sh /path/to/their/caseDir
cd /path/to/their/caseDir
laserMeltFoam                                # writes raw MTX dumps
python3 .../shrink_dump.py                   # MTX → npz
python3 .../sanity.py <one_npz>              # verify A·x_OF ≈ b
```

Then on the GPU machine:

```bash
cd dilu/benchmark/openfoam_crosscheck
python3 driver_amgx.py <case_npz_dir> --amortize       # for pd
python3 driver_amgx.py <case_npz_dir> --amortize --allow-t-eq   # for T
python3 aggregate_results.py <case_npz_dir>            # print medians
```

Total wall time on a new case from scratch: ~30 min OpenFOAM dump + ~5 min npz
shrink + ~10 min our-solver runs ≈ 45 min for 50+100 matrices.
