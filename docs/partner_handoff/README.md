# Partner Handoff — OpenFOAM × DILU/AMGx Cross-Check

This folder is a curated entry point for a collaborator who has not been
involved in the day-to-day work but needs to **fully understand what was done
and where things stand**.

This folder is **self-contained** — every file is a real copy you can read
directly. The canonical (always-up-to-date) versions live elsewhere in
`docs/`; if you want to track future updates, follow the path noted next to
each file in the reading order below.

---

## TL;DR (one screen)

**Goal**: extract the linear systems `A·x = b` that OpenFOAM's `laserMeltFoam`
solves at every PIMPLE corrector, run them through our self-developed GPU
solvers (cuSPARSE DILU, multicolor DILU, AMGx), and benchmark accuracy +
speed against OpenFOAM itself.

**Test case**: 1M-cell 316L spot melt (LPBF), real LaserbeamFoam physics, not
synthetic Poisson.

**Coverage**: 50 pressure-displacement (`pd`, SPD) matrices + 100 temperature
(`T`, non-symmetric) matrices = **150 matrices**, all 4 GPU solver paths
×100 % converged with the correct physics.

**Headline result** (single Xeon core OpenFOAM vs single RTX-class GPU, ours):

| Equation | OpenFOAM (median) | Our best (AMGx + DIAGONAL_SYMMETRIC) | Speed-up | Precision vs OF |
|----------|------------------:|-------------------------------------:|---------:|-----------------|
| pd       |   2.57 s / solve  |                  **0.30 s / solve**  |  **8.6×** | rel = 2.5e-6    |
| T        |   0.14 s / solve  |                  **0.05 s / solve**  |  **2.8×** | rel = 1.0e-8    |

`rel = ‖x_ours − x_OF‖∞ / ‖x_OF‖∞` — true precision metric, NOT
self-consistency residual.

**Key engineering wins** (each one solved a real failure):
1. cuSPARSE 351 s → 2.71 s (129× via `jax.lax.while_loop` + `@jax.jit`)
2. AMGx **diverges** on near-singular Neumann Laplacian → fixed by adding
   `"scaling": "DIAGONAL_SYMMETRIC"` to AMGx config (one line)
3. AMGx setup amortization — reuse `Plan` across timesteps (`update_coefficients`)
4. T equation BiCGStab outer (PCG would fail — non-symmetric)
5. Real OpenFOAM wallclock via `std::chrono` patches in `pEqn.H` / `TEqn.H`
   (the `profiling` functionObject did not produce output in our v2506)

**Wrong path that almost looked right**: `reg = 1e-2` uniform `ε·I`
regularization gave self-consistency residual `9.25e-11` (looked converged ✓)
but `rel_vs_OF = 1.00` (100 % wrong physics ❌). This is why we always use
`x_OF` as ground truth, never just `‖A·x − b‖`.

---

## Reading order (≈ 30 min total)

| # | File | What you get | Time | Canonical source |
|---|------|--------------|------|------------------|
| 1 | `01_methodology.md` | How we patch `laserMeltFoam` to dump `(A, b, x_OF)` and how we serialize/verify them. Full reproduction recipe. | 10 min | `specs/MATRIX_EXTRACTION_SPEC.md` |
| 2 | `02_results.md`     | Final numbers. 150 matrices × 4 solvers. Side-by-side tables, configs, validation. **The canonical report — always cite this.** | 10 min | `benchmark/OPENFOAM_CROSSCHECK_20260427_v3.2.md` |
| 3 | `03_full_handoff.md`| The narrative: 5 fixes in order discovered, the wrong path, real-wallclock measurement, outstanding work. | 10 min | `session_logs/SESSION_HANDOFF_20260428.md` |
| 4 | `04_code_and_data.md`| Pointers: where the C++ patch, Python drivers, npz datasets, deploy bundle live. | 5 min | (this folder)    |
| 5 | `05_case_config.md`  | The OpenFOAM case configuration table — geometry, BCs, laser, materials, time, solvers, what was extracted, caveats. | 5 min | (this folder)    |

If you only have 5 minutes, read this README + the headline tables in
`02_results.md`.

---

## Status as of 2026-04-28

**Done**
- Toolchain: `matrixDumper.H` + 3 patches + sanity check + npz transport
- Drivers: cuSPARSE PCG, Multicolor PCG, cuSPARSE BiCGStab, Multicolor BiCGStab,
  AMGx (PCG and BiCGStab, both with amortization mode)
- Data: 50 pd + 100 T matrices on spot melt case, 1.85 GB npz
- Deploy bundle for lab machine: `dilu/benchmark/openfoam_crosscheck/deploy.tar.gz`
  (30 KB, includes `multicase_batch.sh` for unattended overnight runs)

**Outstanding (not blocking, listed in `03_full_handoff.md`)**
- Multi-case to reach 1000+ matrices (1 lab night × 2-3 nights)
- cuSPARSE/Multicolor BiCGStab amortization (1-2 h coding, would beat
  OpenFOAM on T as well)
- AMGx `update_coefficients` perf on pd (smoother flag tuning)

**Not done in this session**
- Slides / paper writing
- Cases other than spot melt
- End-to-end validation of deploy bundle on lab machine (handed to operator)

---

## A note on extraction methodology

We use a **solver-level hook** (modify `laserMeltFoam` application; capture
inside `pEqn.H` / `TEqn.H` before `solve()`). A different approach in the
literature/colleagues' work is the **PCG-internal hook** (modify
`src/OpenFOAM/.../PCG.C`; capture inside the solver). Both methods see the
same matrix `A, b` after boundary fold, but ours requires **manually folding
the boundary source** (because `addBoundarySource()` is `protected` in
`fvMatrix`). For the BC types in this case (`fixedValue`, `zeroGradient`)
our manual fold matches OpenFOAM's internal fold to `‖A·x_OF − b‖/‖b‖ ≈ 1e-7`
— i.e., the extraction is faithful. For more exotic BCs (mixed, cyclic,
processor) this has not been verified. See
`reference/Matrix_Output_Guide_external.md` for the PCG-internal-hook
reference, and `specs/MATRIX_EXTRACTION_SPEC.md` for our variant.
