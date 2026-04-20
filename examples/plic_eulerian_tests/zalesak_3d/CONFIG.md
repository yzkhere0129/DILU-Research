# Zalesak 3D Slotted Sphere — Reproduction Configuration

Self-contained specification for the Zalesak 3D rotation benchmark run in
`JAX-LaserAM-plic-research` (branch `yzk/plic-research`). Any independent
team using PLIC / isoAdvector / VoF-compression should be able to reproduce
the numbers in §10 by following the parameters below verbatim.

- JAX driver: `examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py`
- OpenFOAM case (reference implementation): `/home/yzk/OpenFOAM/zalesak_3d_of/`
- OpenFOAM extraction script: `examples/plic_eulerian_tests/zalesak_3d/extract_openfoam.py`

Volume fraction is written as **F** (synonymous with OpenFOAM's `alpha.water`
or generic α); F ∈ [0, 1] with F = 1 inside the slotted sphere.

---

## 1. Domain & Grid

| Quantity | Value |
|---|---|
| Domain | `[0, 1] × [0, 1] × [0, 1]` (dimensionless cube) |
| Grid | `128 × 128 × 128` = **2,097,152** cells |
| Mesh type | uniform hex, `simpleGrading (1 1 1)` |
| `dx = dy = dz` | `1/128 = 0.0078125` |
| Cell volume | `dx³ = 4.76837158e-7` |
| Halo (JAX) | `NH = 1` ghost layer on each side → buffer shape `(130, 130, 130)` |

### 1.1 Boundary conditions

The sphere never touches any boundary, so BC choice is immaterial as long as
it preserves F = 0 outside. The reference implementations use:

| Field | JAX (PLIC) | OpenFOAM (interIsoFoam) |
|---|---|---|
| F / `alpha.water` | halo copy (zero-gradient) on all 6 faces | `zeroGradient` on `xMin xMax yMin yMax zMin zMax` |
| U | prescribed analytic (not solved) | `zeroGradient`; overwritten every step by `freezeU` coded function object |
| `p_rgh` | — | `zeroGradient` (unused: pure advection, no p solve) |

Source refs:
- JAX halo closure: `run_zalesak_3d.py` lines 81–85
- OpenFOAM boundary patches: `/home/yzk/OpenFOAM/zalesak_3d_of/system/blockMeshDict` lines 38–46

---

## 2. Initial Slotted-Sphere α Field

### 2.1 Geometry

| Parameter | Symbol | Value |
|---|---|---|
| Sphere centre | `(cx, cy, cz)` | `(0.50, 0.50, 0.75)` |
| Sphere radius | `R` | `0.15` |
| Slot width (x) | `SLOT_W` | `0.05` (half-width `0.025`; slot spans `x ∈ [0.475, 0.525]`) |
| Slot width (y) | `SLOT_W` | `0.05` (slot spans `y ∈ [0.475, 0.525]`) |
| Slot depth (z) | `SLOT_D` | `0.25` (slot spans `z ∈ [cz + R − SLOT_D, cz + R] = [0.65, 0.90]`) |

Verified source: `run_zalesak_3d.py` lines 35–38; mirrored in
`/home/yzk/OpenFOAM/zalesak_3d_of/init_alpha.py` lines 14–16.

### 2.2 Sub-cell sampling rule (**critical**)

**4 × 4 × 4 = 64 uniform sub-points per cell**, no smoothing / tanh:

```
For each cell (i, j, k) with centre (xi, yj, zk):
  F[i,j,k] = (# sub-points inside SPHERE AND NOT in SLOT) / 64

where the 64 sub-points are
  xs = xi + ((a + 0.5)/4 − 0.5) · dx   for a = 0, 1, 2, 3
  ys = yj + ((b + 0.5)/4 − 0.5) · dx   for b = 0, 1, 2, 3
  zs = zk + ((c + 0.5)/4 − 0.5) · dx   for c = 0, 1, 2, 3
```

Inside predicates (evaluated at each sub-point):
```
in_sphere = (xs − cx)² + (ys − cy)² + (zs − cz)² ≤ R²
in_slot   = (|xs − cx| < SLOT_W/2) ∧ (|ys − cy| < SLOT_W/2)
            ∧ (cz + R − SLOT_D < zs < cz + R)

F_sub = in_sphere ∧ ¬in_slot
F[i,j,k] = mean(F_sub over the 64 sub-points)
```

Source refs: `run_zalesak_3d.py` lines 50–79 (`n_sub = 4`);
`/home/yzk/OpenFOAM/zalesak_3d_of/init_alpha.py` lines 18–60 (identical predicate on 64 sub-points). Both initialisers produce bit-identical F on the 128³ grid (`initial_field_max_abs_diff (JAX vs OF) = 0.0`, `compare_metrics.json`).

### 2.3 Initial volume

| Quantity | Value |
|---|---|
| V₀ (numerical, 4³ sub-sample) | **1.349673e-02** |
| V_sphere (analytic `4/3 π R³`) | 1.413717e-02 |
| V_slot-box (`0.05 × 0.05 × 0.25`) | 6.25e-04 |
| V_sphere − V_slot-box (analytic upper bound) | 1.351217e-02 |
| F range at t = 0 | `[0.000000, 1.000000]` |

V₀ agreement between JAX and OpenFOAM: identical to 15 digits — both
yield `V_init = 0.013496726751327515` (`compare_metrics.json`).

---

## 3. Velocity Field (Rigid-Body Rotation)

Analytic solid-body rotation about the **x-axis** at `(y, z) = (0.5, 0.5)`:

```
u(x,y,z) = 0
v(x,y,z) = −ω · (z − 0.5)
w(x,y,z) = +ω · (y − 0.5)
```

| Quantity | Value |
|---|---|
| Angular velocity ω | `1.0 rad/s` |
| Rotation axis | x-axis through `(y, z) = (0.5, 0.5)` |
| `max \|v\|` at cell centres (y, z ∈ [dx/2, 1−dx/2]) | `0.49609375` |
| `max \|v\|` including halo centres (y, z ∈ [−dx/2, 1+dx/2]) | `0.50390625` *(JAX uses this for CFL)* |
| Divergence `∇·u` | 0 exactly, continuously and discretely |
| Full-rotation period `T = 2π` | `6.283185307179586` |

Velocity locations differ between reference implementations:
- **JAX**: face-centred `v_face` (on y-faces) and `w_face` (on z-faces), `u_face` = 0
  (see `run_zalesak_3d.py` lines 91–108).
- **OpenFOAM interIsoFoam**: cell-centred U overwritten every step by the
  `coded` function object `freezeU`; `phi = fvc::flux(U)` is refreshed to a
  discretely divergence-free face flux (`system/controlDict` lines 36–77).

---

## 4. Time Integration

| Quantity | Value | Note |
|---|---|---|
| CFL | `0.45` | based on `max\|v\|` including halo |
| dt (JAX) | `6.9737e-3 s` | `dt = CFL · dx / 0.50390625 ≈ 6.9767e-3`, then rescaled so `n_steps · dt = T` |
| n_steps | **901** | `ceil(T / dt)` |
| Final dt (rescaled) | `T / 901 = 6.973569e-3` | ensures exact end time |
| Integrator | Strang operator split **y/2 → z → y/2** | rotation is in y-z plane, so no x sub-sweep |
| OpenFOAM `deltaT` | `6.9737e-3` (fixed; `adjustTimeStep no`) | `maxCo = 0.5`, `maxAlphaCo = 0.5` declared but not enforced |
| `endTime` | `6.2832` | `system/controlDict` line 16 |

Source refs:
- JAX dt computation: `run_zalesak_3d.py` lines 215–218
- OpenFOAM: `/home/yzk/OpenFOAM/zalesak_3d_of/system/controlDict` lines 17–30

---

## 5. Solver Settings — JAX Eulerian PLIC

| Stage | Method |
|---|---|
| Normal estimation | Parker–Youngs 3 × 3 × 3 convolution (`compute_youngs_normal_3d`) |
| Intercept solver | Scardovelli–Zaleski analytic formula + 5 Newton iterations (jvp autodiff) (`analytic_intercept`) |
| Flux sweep | Geometric-flux SL scheme (`sweep_flux_{y,z}`, `apply_flux_{y,z}`) |
| Halo update | halo copy / zero-gradient after every sub-sweep |
| Bound preservation (default) | `jnp.clip(F, 0, 1)` after each sub-sweep |
| Bound preservation (alternative) | Weymouth–Zaleski redistribute, `n_iter = 3` (`apply_flux_*_conservative`) |
| Precision | `float32` |
| JIT | one `@jax.jit` wrapping the full Strang step; closure captures `u_face, v_face, w_face, dt` |

Exact operator sequence per step (`plic_step_3d`, lines 113–133):
```
sub_y(F, dt/2) → sub_z(F, dt) → sub_y(F, dt/2)

sub_axis(F, dt_s):
  F ← halo(F)
  (nx, ny, nz) ← compute_youngs_normal_3d(F, dx, dy, dz)
  C ← analytic_intercept(nx, ny, nz, F, dx, dy, dz)
  flux ← sweep_flux_axis(F, nx, ny, nz, C, vel_face, dt_s, dx, dy, dz)
  F ← clip(apply_flux_axis(F, flux, dx, dy, dz), 0, 1)
  return halo(F)
```

---

## 6. Solver Settings — OpenFOAM interIsoFoam (reference)

OpenFOAM 2506. Solver: `interIsoFoam` (geometric isoAdvector). All files
under `/home/yzk/OpenFOAM/zalesak_3d_of/system/`.

### 6.1 `controlDict` essentials

```
application      interIsoFoam;
startTime        0;
endTime          6.2832;
deltaT           6.9737e-3;
adjustTimeStep   no;
writeControl     adjustable;
writeInterval    1.5708;       // every quarter turn
writeFormat      binary;
timePrecision    8;
maxCo            0.5;          maxAlphaCo 0.5;
functions {
    freezeU { type coded; executeControl timeStep; ... }   // rewrites U & phi each step
    volAlpha { type volFieldValue; operation volIntegrate;
               fields (alpha.water); writeInterval 10; }
}
```

### 6.2 `fvSolution` — `alpha.water.*` block (isoAdvector)

```
isoFaceTol            1e-8
surfCellTol           1e-6
nAlphaBounds          3
snapTol               1e-12
clip                  true
reconstructionScheme  isoAlpha
nAlphaCorr            1
nAlphaSubCycles       1
cAlpha                1
```

`PIMPLE: momentumPredictor no; nOuterCorrectors 1; nCorrectors 1; nNonOrthogonalCorrectors 0`.

### 6.3 `fvSchemes`

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { div(phi,alpha) Gauss vanLeer;
                  div(phirb,alpha) Gauss linear;
                  div(rhoPhi,U)    Gauss linearUpwind grad(U);
                  default          none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
```

### 6.4 Parallel decomposition

```
numberOfSubdomains  8
method              simple
coeffs { n (2 2 2); }      // 2×2×2 cartesian split → 8 MPI ranks
```

### 6.5 Transport properties

`water` and `air` both: `nu = 0`, `rho = 1`, Newtonian; `sigma = 0`
(pure advection, no viscosity, no gravity, no surface tension).

---

## 7. Diagnostic Metrics

All three reported side-by-side between JAX and OpenFOAM using identical
definitions, operating on the 128³ interior array (F on cell centres, not
face-averaged):

| Metric | Definition | Units |
|---|---|---|
| V(t) | `Σ F(t) · dx³` over interior cells | dimensionless volume |
| V drift | `(V(t) − V₀) / V₀ × 100` | % |
| L1 shape error | `Σ \|F(t) − F(0)\| · dx³ / V₀ × 100` | % |
| F_min, F_max | cell-wise min/max of F(t) | dimensionless |
| ms/step | average wall time per full Strang step (JIT excluded, JAX) | ms |

---

## 8. Hardware & Runtime

| Item | JAX run | OpenFOAM run |
|---|---|---|
| Device | RTX 3050 Laptop GPU, 4 GB VRAM | 8 CPU cores (WSL2) |
| Parallelism | single-GPU | MPI, 8 ranks (`mpirun -np 8`) |
| CUDA | 12.x | — |
| JAX | 0.9.0, Python 3.12.3 | — |
| Precision | float32 | float64 (OpenFOAM default) |
| Wall time (full rotation, 901 steps) | **137.96 s** | not logged; functionObject output spans all 901 steps |
| ms / step (mean, post-JIT) | **153.1** | — |

Sources: `results/zalesak_3d_final.npz` (keys `wall_time`, `n_steps`);
`docs/PLIC_PERFORMANCE_PROGRESSION.md` §3.2.

---

## 9. Results — Head-to-Head

### 9.1 Summary (values verified from repo artifacts)

| Method | V drift | L1 shape error | F range | ms/step |
|---|---:|---:|---|---:|
| **JAX Eulerian PLIC (clip)** | **−0.0542 %** | **1.529 %** | `[0.0, 1.0]` | **153.1** |
| **OpenFOAM interIsoFoam** | **−2.19e-7 %** | **2.442 %** | `[0.0, 1.0]` | n/a |
| JAX Eulerian PLIC (redistribute, n_iter=3) *[spec target, not yet re-run on 128³ full rotation]* | target < 1e-6 % | target < 2.5 % | — | target < 180 |

Exact verified numbers:
- JAX clip: `V_init = 0.013496726751327515`, `V_final = 0.013489408984095976`, `drift_abs = −7.3178e-6`, `drift_pct = −5.4219e-2`, `L1_pct = 1.52944` (`compare_metrics.json`).
- OpenFOAM: `V_final = 0.013496726721780513`, `drift_abs = −2.9547e-11`, `drift_pct = −2.1892e-7`, `L1_pct = 2.44175`, `n_interface_cells_final = 7878` (`metrics.json`).
- Cross L1 between JAX-final and OF-final fields: `2.2845 %` (same file).

The redistribute target row is reproduced from
`docs/specs/CONSERVATIVE_REDISTRIBUTE_SPEC.md` and the targets encoded in
`examples/plic_eulerian_tests/compare_clip_vs_redistribute.py` lines 258–272.
A full-rotation run under `redistribute` on 128³ is **not** present in
`results/`; the redistribute variant has been validated on the 1 M-cell 45°
translation and the `Rider-Kothe` test.

---

## 10. Reproduction Commands

### 10.1 JAX Eulerian PLIC

```bash
cd /home/yzk/JAX-LaserAM-plic-research
python examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py
# outputs → examples/plic_eulerian_tests/zalesak_3d/results/
#   zalesak_3d_final.npz, zalesak_3d_result.png
```

### 10.2 OpenFOAM interIsoFoam

```bash
source /usr/lib/openfoam/openfoam2506/etc/bashrc
cd /home/yzk/OpenFOAM/zalesak_3d_of
python3 init_alpha.py                       # write 0/alpha.water with 4x4x4 sampling
blockMesh
decomposePar
mpirun -np 8 interIsoFoam -parallel
./finalize.sh                               # reconstructPar + post-process
```

### 10.3 Extract OpenFOAM result and compare

```bash
cd /home/yzk/JAX-LaserAM-plic-research
python examples/plic_eulerian_tests/zalesak_3d/extract_openfoam.py
# writes  results/zalesak_3d_openfoam.npz
```

### 10.4 Visualisation

```bash
# ParaView VTI exports (initial, final):
python examples/plic_eulerian_tests/zalesak_3d/export_for_paraview.py

# 2x3 multi-slice comparison panels:
python examples/plic_eulerian_tests/zalesak_3d/plot_openfoam_overview.png.py
```

---

## 11. File Inventory

All paths absolute.

| Path | Content |
|---|---|
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py` | JAX PLIC driver (128³ full rotation) |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/extract_openfoam.py` | OF reconstructPar + binary-scalarField → npz |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/export_for_paraview.py` | npz → `.vti` |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/plot_openfoam_overview.png.py` | Overview slice comparison plot |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/OPENFOAM_REPRODUCTION.md` | Earlier, more verbose OF reproduction notes |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_final.npz` | JAX F_init, F_final, n_steps=901, wall_time=137.96 s |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_openfoam.npz` | OF F at t=0, π/2, π, 3π/2, 2π |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_initial.vti` | ParaView initial F |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_final.vti` | ParaView final F |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_overview.png` | JAX 2×3 slice comparison |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_openfoam_overview.png` | OpenFOAM 2×3 slice comparison |
| `/home/yzk/JAX-LaserAM-plic-research/examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_isosurface.png` | F = 0.5 isosurface render |
| `/home/yzk/OpenFOAM/zalesak_3d_of/system/{controlDict,blockMeshDict,fvSchemes,fvSolution,decomposeParDict}` | OF case dictionaries |
| `/home/yzk/OpenFOAM/zalesak_3d_of/constant/transportProperties` | `nu = 0, rho = 1` for both phases; `sigma = 0` |
| `/home/yzk/OpenFOAM/zalesak_3d_of/0/{alpha.water, U, p_rgh}` | OF initial fields |
| `/home/yzk/OpenFOAM/zalesak_3d_of/init_alpha.py` | 4×4×4 sampled α writer (binary OF format) |
| `/home/yzk/OpenFOAM/zalesak_3d_of/metrics.json` | OF V₀, V_final, drift, L1 metrics |
| `/home/yzk/OpenFOAM/zalesak_3d_of/compare_metrics.json` | joint JAX ↔ OF comparison numbers |
| `/home/yzk/OpenFOAM/zalesak_3d_of/postProcessing/volAlpha/0/volFieldValue_0.dat` | per-step `∫ α dV` log (90 rows × every 10 steps) |

---

## 12. Validation Checklist

- [ ] Grid exactly 128³ uniform hex on `[0,1]³`; `dx = 1/128`
- [ ] Initial α from 4×4×4 sub-cell sampling (NOT tanh/smoothed); V₀ = 1.349673e-02 to 6 digits
- [ ] Velocity prescribed analytically every step (not advected)
- [ ] `ν = 0`, `ρ = 1` for both phases, `σ = 0`, no gravity
- [ ] CFL ≤ 0.45; `dt ≈ 6.97e-3`; 901 steps over `T = 2π`
- [ ] Strang split **y/2 → z → y/2** (rotation is in y-z plane — no x sub-sweep)
- [ ] α ∈ [0, 1] strictly at every step (post-clip or post-redistribute)
- [ ] |V drift| < 0.1 % (geometric PLIC / isoAdvector class)
- [ ] L1 < 3 % on 128³
- [ ] Cross-compare JAX and OF final fields via ParaView side-by-side on F = 0.5 contour
