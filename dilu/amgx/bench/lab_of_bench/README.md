# Lab-machine deliverable: time OpenFOAM PCG-DIC on senior's 21 pd matrices

Three different routes to get OF wall time on senior data, in increasing precision:

## Option 1 (cheapest, 0 lab work) — cite scaling test baseline

`docs/benchmark/OPENFOAM_SCALING_20260503.md` already reports

| OpenFOAM, lab Xeon Gold 5120 N=32, spot_melt_150W (500K cells, similar mesh to senior 512K) |
|---|
| **pd_corr0 median wall = 44.1 ms** |

This is our current benchmark baseline used in `bench_threeway_senior.py`. The matrices are not the *same* matrices, but **same mesh size, same algorithm (PCG+DIC), same machine, same core count**. Within ±10% of what OF would do on senior matrices.

## Option 2 (medium, 1-2h lab work) — re-run dumper on a 80³ case

If we want OF wall time on the *exact* senior matrix structure:

1. SSH lab machine
2. Take spot_melt case, modify `system/blockMeshDict` to 80×80×80
3. Run with the existing matrixDumper that writes `[TIMING_pd]` lines
4. Read first 11 timesteps × 3 correctors = 33 pd timings (matches senior's 21 effective + 12 trivial-iter zeros)
5. Send `log.laserMeltFoam` back

Wall time: 80³ × 30 timesteps ≈ 5 minutes.

## Option 3 (most precise, 4-8h work) — build C++ matrix loader

Write `applications/utilities/matrixSolveTimer/`:
- Reads senior's CSV (matrix_pd_*.csv, source_pd_*.csv)
- Reconstructs `Foam::lduMatrix`
- Calls `Foam::PCG::solve(...)` with same DIC preconditioner
- Reports `[ TIMING_pd ] X ms` per matrix
- Wraps with `mpirun -np 32` for the N=32 baseline

This gives **byte-exact same matrices, OF native algorithm, lab hardware**.

## Recommended

Start with **Option 1** (already done). If results are surprising, escalate to Option 2 (cheap). Option 3 is overkill unless we need to publish.
