# 05 — OpenFOAM Case Configuration Table

The case used to generate all 50 pd + 100 T matrices in the v3.2 benchmark.

**Case directory**: `/home/yzk/lbm_d3q27_mrt_cumulant/openfoam_laserMeltFoam_spot/`
(this lives outside `DILU-Research/`; the path is on the development machine)

**Solver**: `laserMeltFoam` (LaserbeamFoam fork of OpenFOAM v2506),
patched with `matrixDumper.H` to dump `(A, b, x_OF)` per `solve()` call.

**Physics**: stationary-spot laser melt of a 316L-like substrate, no scan
motion, no ray tracing. Designed to mirror an LBM reference case for
cross-platform validation.

---

## 1. Geometry & Mesh

| Item                   | Value                          |
|------------------------|--------------------------------|
| Domain (X × Y × Z)     | 200 × 400 × 200 μm             |
| Cell count             | 80 × 160 × 80 = **1,024,000** (~1 M) |
| Cell size (uniform)    | 2.5 μm                         |
| `convertToMeters`      | 1.0                            |
| Substrate fill (initial `alpha.metal = 1`) | `box (-1 -1 -1) (200e-6 400e-6 100e-6)` — metal occupies Z ∈ [0, 100] μm (lower half) |

## 2. Patches & Boundary Conditions

| Patch        | Type   | T               | U                            | pd                |
|--------------|--------|-----------------|------------------------------|-------------------|
| `topWall`    | patch  | zeroGradient    | pressureInletOutletVelocity  | totalPressure (= 0)|
| `bottomWall` | patch  | zeroGradient    | fixedValue (0 0 0)           | zeroGradient      |
| `leftWall`   | wall   | zeroGradient    | fixedValue (0 0 0)           | zeroGradient      |
| `rightWall`  | patch  | zeroGradient    | fixedValue (0 0 0)           | zeroGradient      |
| `front`      | wall   | zeroGradient    | fixedValue (0 0 0)           | zeroGradient      |
| `back`       | wall   | zeroGradient    | fixedValue (0 0 0)           | zeroGradient      |

→ pd is **near-pure-Neumann** (only `topWall` is Dirichlet via `totalPressure`),
which combined with `pRefCell 0; pRefValue 0;` produces the 14-orders-of-magnitude
diag span and λ_min ≈ 1e-13 we observed. This is the property that breaks
classical AMG until `DIAGONAL_SYMMETRIC` scaling is applied.

→ T is non-symmetric due to advection (`fvm::div(rhoCpPhi, T)`) → BiCGStab outer.

## 3. Laser

| Parameter                | Value                                                  |
|--------------------------|--------------------------------------------------------|
| Power                    | 150 W (ramped on 0 → 50 μs, off after 50 μs)           |
| Spot radius `laserRadius`| 25 μm                                                  |
| Position (stationary)    | (X=100, Y=200, Z=100) μm — domain centre top of metal   |
| Absorptivity `kappaM`    | 0.35                                                   |
| Surface emissivity `emS` | 0.4                                                    |
| Liquid emissivity `emL`  | 0.1                                                    |
| Ray tracing              | **off** in this case (despite `ray_tracing_on true` flag, used as standard heat source) |
| Wavelength `λ`           | 1.064 μm (Nd:YAG; ignored when ray tracing off)        |
| Source patch             | `topWall`                                              |

`timeVsLaserPosition` clamps to a single point — true spot, no scan.
`timeVsLaserPower` step turns laser off at t = 50 μs to let pool relax.

## 4. Material Properties (316L-ish)

### Metal phase

| Property              | Value                              | Units             |
|-----------------------|------------------------------------|-------------------|
| Density (solid) ρ_S   | 7900                               | kg/m³             |
| Density (liquid) ρ_L  | 7900 (no thermal expansion)        | kg/m³             |
| Kinematic viscosity ν | 6.33e-7                            | m²/s (μ/ρ ≈ 5e-3) |
| Specific heat c_p     | 700 (constant for solid and liquid)| J/(kg·K)          |
| Thermal conductivity k| 20 (constant for solid and liquid) | W/(m·K)           |
| Solidus T_s           | 1650                               | K                 |
| Liquidus T_l          | 1700                               | K                 |
| Latent heat fusion H_f| 2.7e5                              | J/kg              |
| Vaporisation T_vap    | 3000                               | K                 |
| Latent heat vap. H_v  | 7.5e6                              | J/kg              |
| Mush perm. coeff `C`  | 1e6                                | kg/(m³·s)         |

### Gas phase (argon)

| Property | Value | Units |
|----------|-------|-------|
| ν        | 1.5e-5 | m²/s |
| ρ_G      | 1.225  | kg/m³ |
| k_G      | 0.0177 | W/(m·K) |
| c_p,G    | 520    | J/(kg·K) |

### Interface

| Property                         | Value          | Units           | Note                              |
|----------------------------------|----------------|-----------------|-----------------------------------|
| Surface tension σ                | 1.6            | N/m             | at T = T_l                        |
| dσ/dT (`STgrad`)                 | **+1.0e-4**    | N/(m·K)         | **inward-pulling** Marangoni (deep V pool) |
| Reference temperature `TRef`     | 300            | K               |                                   |
| Convective HTC `hC`              | **0**          | W/(m²·K)        | no BC cooling (match LBM)         |

## 5. Time Integration

| Item                  | Value           |
|-----------------------|-----------------|
| `application`         | `laserMeltFoam` |
| `startTime`           | 0               |
| `endTime`             | 100 μs          |
| Initial `deltaT`      | 1e-12 s         |
| `adjustTimeStep`      | yes             |
| `maxCo`               | 0.5             |
| `maxAlphaCo`          | 0.5             |
| `maxDeltaT`           | 2e-8 s          |
| `writeControl`        | `adjustableRunTime` |
| `writeInterval`       | 5 μs (20 VTK snapshots) |
| `writeFormat`         | ASCII           |
| `writePrecision`      | 6               |

Steady-running `dt` lands around 2e-8 s once the melt pool establishes
(maxCo binding); per timestep PIMPLE drives **3 pd correctors + ~117 T
sub-iterations** (T tolerance 1e-12 is very tight).

## 6. PIMPLE / Solver Controls (`fvSolution`)

### Outer loop

| Control                      | Value |
|------------------------------|-------|
| `nOuterCorrectors`           | 1     |
| `nCorrectors` (pd correctors)| 3     |
| `nNonOrthogonalCorrectors`   | 0     |
| `nTCorrectors` (T inner)     | 250 (cap; usually exits via tolerance) |
| `momentumPredictor`          | no    |
| `pRefCell` / `pRefValue`     | 0 / 0 |

### Per-equation solvers (OpenFOAM's own choices)

| Field        | Solver  | Preconditioner | Tolerance | relTol |
|--------------|---------|----------------|-----------|--------|
| `pd`         | PCG     | DIC            | 1e-8      | 0      |
| `pdFinal`    | PCG     | DIC            | 1e-8      | 0      |
| `T` / `TFinal`| PBiCG  | DILU           | 1e-12     | 0      |
| `U`          | PBiCGStab | DILU         | 1e-8      | 0      |
| `pcorr.*`    | PCG     | DIC            | 1e-10     | 0      |
| `alpha.metal` | isoAlpha (isoAdvector) | — | snap 1e-12 / surf 1e-6 | — |

These OpenFOAM solvers produce the `x_OF` we use as the ground truth in
`rel_vs_OF = ‖x_solver − x_OF‖∞ / ‖x_OF‖∞`.

### Residual control (terminate PIMPLE)

| Field group                         | Tolerance   | relTol |
|-------------------------------------|-------------|--------|
| U, pd, temperatureTol, alternativeTol, absTol | 1e-8 | 0 |
| gTTol                                          | 1e-7 | 0 |

## 7. What was extracted

Per timestep we capture the linear systems at exactly these solve-points:

| System    | When                   | Per timestep | Sym? | Solver in OF |
|-----------|------------------------|--------------|------|---------------|
| `pd_corr0`| 1st PIMPLE pd correct  | 1            | SPD  | PCG-DIC       |
| `pd_corr1`| 2nd                    | 1            | SPD  | PCG-DIC       |
| `pd_corr2`| 3rd (`pdFinal`)        | 1            | SPD  | PCG-DIC       |
| `T_corr*` | T inner sub-iterations | ~2 (early-exit on tolerance) | non-sym | PBiCG-DILU |

Sampled timesteps: 50 consecutive `dt` snapshots between **t = 2.25 μs and
t = 3.23 μs**, which is the early melt-pool growth phase (laser already hot,
pool just forming). Total data:

- 50 pd matrices (50 timesteps × ~1 capture per timestep — first corrector only)
- 100 T matrices (50 timesteps × ~2 inner iterations)
- 1.85 GB on disk (compressed npz; ~80 GB if left as MatrixMarket ASCII)

## 8. Why this case (decision rationale)

1. **Real LPBF physics, not synthetic Poisson** — the matrix nonzero pattern,
   diag span, and near-singularity all come from real boundary conditions and
   real coefficient fields, not from a constructed test problem.
2. **Stationary spot** keeps the matrix sparsity pattern fixed across timesteps
   → enables AMGx setup amortization (one `Plan`, reuse via `update_coefficients`).
3. **Cross-platform reference** — same physics also runs in our LBM solver;
   benchmarking GPU CFD solvers on a matrix that both platforms agree on.
4. **1 M cells** is "small enough to iterate fast on a workstation, large
   enough that GPU acceleration matters" — single AMGx solve drops from 2.57 s
   (CPU PCG-DIC) to 0.30 s (GPU AMGx + DIAG_SYM).

## 9. Caveats / things to know

- **Mesh is uniform 2.5 μm** — refinement effects on solver convergence are
  not characterized. AMGx coarsening efficiency may differ on graded meshes.
- **k, c_p are constants**, not the polynomial T-dependence in some 316L data
  sheets. Done deliberately to match the LBM solver.
- **No ray tracing** — surface heat source is the standard top-hat-like Gaussian
  on `topWall`. If switched on, T-equation stiffness changes.
- **`hC = 0`** (no convective cooling boundary). If you turn this on, T diag
  picks up boundary fold contributions and the BC-fold validation should be
  re-checked.
- **Marangoni `dσ/dT = +1e-4`** is **inward-pulling** (deep narrow pool). Most
  316L tables list `−4e-4` (outward, shallow wide pool). This sign was chosen
  to match the LBM reference, not real 316L.
- **OpenFOAM v2506 quirk**: `profiling` functionObject didn't emit output here;
  we measured wallclock with `std::chrono` patches in `pEqn.H` / `TEqn.H`
  instead. Patch is part of the deploy bundle.

## 10. How to re-run

```bash
cd /home/yzk/lbm_d3q27_mrt_cumulant/openfoam_laserMeltFoam_spot
./Allclean
./Allrun                    # ~30 min on the dev machine for full 100 μs
# → writes case.foam + per-timestep VTK + raw matrix dumps
```

To restrict to the 1 μs window we sampled (50 pd + 100 T matrices), edit
`system/controlDict` `endTime` to `3.3e-6` and use `startFrom latestTime`
after the matrix dumper has been activated by patching the case (see
`04_code_and_data.md` Quick-start section).

After dump completes:

```bash
python3 .../shrink_dump.py    # MTX → npz, ~5 min
python3 .../sanity.py <one_npz>   # verify A·x_OF ≈ b, residual ≈ 1e-7
```
