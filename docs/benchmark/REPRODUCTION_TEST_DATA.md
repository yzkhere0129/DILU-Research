# Eulerian PLIC Reproduction Test Data — Definitive Baseline

> **Purpose**: Bit-identical reproduction validation. A reimplemented PLIC
> pipeline must match **every number** in this file when run with the
> **unmodified** test scripts on the same hardware / JAX version.
>
> **Date**: 2026-04-17  
> **Branch**: `yzk/plic-research`  
> **Commit**: run `git rev-parse HEAD` in the worktree for exact hash

---

## Hardware & Software

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| CUDA | 12.x |
| JAX | 0.4.x (check `jax.__version__`) |
| Python | 3.12.3 |
| Precision | float32 (default) |
| OS | Linux 6.6.87.2-microsoft-standard-WSL2 |

---

## Test 1: Zalesak 3D Slotted Sphere — 128³ (2.1M cells)

### 1.1 Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Script | `examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py` | copy verbatim |
| Grid | 128 × 128 × 128 = 2,097,152 interior cells | `N = 128` |
| Halo | NH = 1 (total array: 130³) | `NH = 1` |
| dx = dy = dz | 1/128 = 0.0078125 | `DX = DY = DZ = 1.0 / N` |
| Domain | [0, 1]³ | implicit |

#### Initial condition: slotted sphere

| Parameter | Value |
|-----------|-------|
| Sphere center | (0.50, 0.50, 0.75) |
| Sphere radius R | 0.15 |
| Slot width (x and y) | 0.05 |
| Slot depth (z, from top) | 0.25 |
| Sub-sampling | 4 × 4 × 4 = 64 sub-cells per cell |
| Initial volume V₀ | **1.349673e-02** |
| Initial interface cells | **5,924** |

#### Velocity field: solid-body rotation

| Parameter | Value |
|-----------|-------|
| Rotation axis | x-axis through (y, z) = (0.50, 0.50) |
| Angular velocity ω | 1.0 rad/s |
| u(x,y,z) | 0 |
| v(x,y,z) | −ω (z − 0.50) |
| w(x,y,z) | +ω (y − 0.50) |
| max\|vel\| | 0.503906 (includes halo cell positions) |
| Period T | 2π ≈ 6.2832 s |

#### Time stepping

| Parameter | Value |
|-----------|-------|
| CFL | 0.45 |
| dt | 6.9736e-03 s |
| n_steps (one full rotation) | **901** |
| Strang split | y/2 → z → y/2 (rotation is in y-z plane) |
| Bounds mode | `jnp.clip(F, 0, 1)` (inline in script) |

#### Solver

| Component | Method |
|-----------|--------|
| Normal | Parker-Youngs 3×3×3 Prewitt conv via `lax.conv_general_dilated` |
| Intercept | Scardovelli-Zaleski 2000 analytic (Cardano + 5 Newton via `jax.jvp`) |
| Sparse intercept | Padded gather, `MAX_FRAC = 0.15` (unused in this script — script uses dense mode) |
| Flux | Donor-box geometric (PLIC plane ∩ sweep box) |
| Apply flux | Conservative divergence-form update |
| Clip | `jnp.clip(F, 0, 1)` after each sub-sweep |

**[IMPORTANT]** `run_zalesak_3d.py` does NOT import `strang_sweep.py` or `conservative_bounds.py`. It implements the Strang step inline with `jnp.clip`. The modules it imports:
- `jax_laseram.vof.plic.normal_youngs.compute_youngs_normal_3d`
- `jax_laseram.vof.plic.analytic_intercept.analytic_intercept`
- `jax_laseram.vof.plic.geometric_flux.{sweep_flux_y, apply_flux_y, sweep_flux_z, apply_flux_z}`
- `jax_laseram.vof.plic.diagnostics.{compute_stats, format_stats}`

### 1.2 Results — Acceptance Criteria

| Metric | Definitive Value | Tolerance for bit-identical |
|--------|------------------|-----------------------------|
| L1 shape error | **1.53%** | exact to 2 dp (±0.005%) |
| Volume drift (V_final − V₀)/V₀ | **−0.054223%** | exact to 4 significant figures |
| V at step 901 | **1.348941e-02** | exact to 6 significant figures |
| F range after all steps | **[0.000, 1.000]** | no values < 0 or > 1 |
| n_interface at step 901 | **7,832** | ±100 (depends on boundary layer) |
| n_below (F < 0 count) | **0** | must be exactly 0 |
| n_above (F > 1 count) | **0** | must be exactly 0 |

### 1.3 Timing Reference (informational, not acceptance criteria)

| Metric | Value |
|--------|-------|
| Wall time (including diagnostics) | ~231 s |
| ms/step (with diagnostics sync) | ~256 ms |
| GPU peak VRAM | 1824 MB |
| JIT compile (first step) | ~50-100 s |

### 1.4 Volume Drift Trajectory (per checkpoint)

| Step | V | dV/V₀ | n_interface |
|------|---|-------|-------------|
| 0 | 1.349673e-02 | 0 | 5,924 |
| 180 | 1.349491e-02 | −1.35e-04 | 7,913 |
| 360 | 1.349336e-02 | −2.50e-04 | 7,893 |
| 540 | 1.349202e-02 | −3.49e-04 | 7,867 |
| 720 | 1.349048e-02 | −4.63e-04 | 7,844 |
| 900 | 1.348941e-02 | −5.42e-04 | 7,749 |
| 901 | 1.348941e-02 | −5.42e-04 | 7,832 |

---

## Test 2: Rider-Kothe Reversed Vortex — T=10, 128×128×3

### 2.1 Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Script | `examples/plic_eulerian_tests/reversed_vortex/run_comparison.py` | copy verbatim |
| Grid | 128 × 128 × 3 interior cells | `N = 128, NZ = 3` |
| Halo | NH = 1 (total array: 130 × 130 × 5) | `NH = 1` |
| dx = dy = dz | 1/128 = 0.0078125 | `DX = 1.0 / N` |
| Domain | [0, 1] × [0, 1] × [0, 3·dx] (quasi-2D) | implicit |

#### Initial condition: circle

| Parameter | Value |
|-----------|-------|
| Shape | Circle (z-extruded) |
| Center | (0.50, 0.75) |
| Radius R | 0.15 |
| Init method | Sub-cell 4×4 sampling |
| V₀ | **1.65796280e-03** |
| n_interface (initial) | **540** |

#### Velocity field: time-reversed single vortex

Stream function: ψ = (1/π) sin²(πx) sin²(πy) cos(πt/T)

| Component | Formula |
|-----------|---------|
| u(x,y,t) | −sin²(πx) · sin(2πy) · cos(πt/T) |
| v(x,y,t) | +sin(2πx) · sin²(πy) · cos(πt/T) |
| w | 0 |
| max\|u\| | 0.9997 |
| T (period) | **10.0** (user-configured) |

At t = T/2 = 5.0: velocity = 0 (maximum deformation — spiral filament).
At t = T = 10.0: velocity has fully reversed, fluid should return to initial circle.

#### Time stepping

| Parameter | Value |
|-----------|-------|
| CFL | 0.5 |
| dt | 3.91e-03 s |
| n_steps | **2560** |
| Strang split | x/2 → y → x/2 |
| Midpoint time scaling | `cos(π(t + dt/2)/T)` — 2nd-order temporal |
| Bounds mode | `jnp.clip(F, 0, 1)` (inline, both methods) |

#### Solver (both Eulerian PLIC and Lagrangian VOF)

Eulerian PLIC uses the same pipeline as Test 1 (Youngs + analytic intercept + geometric flux + clip). Lagrangian VOF uses `advect_vof_lagrangian_3d` from `jax_laseram.vof.lagrangian_3d`.

**Modules imported by test script:**
- `jax_laseram.vof.plic.normal_youngs.compute_youngs_normal_3d`
- `jax_laseram.vof.plic.analytic_intercept.analytic_intercept`
- `jax_laseram.vof.plic.geometric_flux.{sweep_flux_x, apply_flux_x, sweep_flux_y, apply_flux_y}`
- `jax_laseram.data_types.GridInfo`
- `jax_laseram.vof.lagrangian_3d.advect_vof_lagrangian_3d`

### 2.2 Results — Acceptance Criteria

#### Eulerian PLIC

| Metric | Definitive Value | Tolerance |
|--------|------------------|-----------|
| L1 error | **98.281%** | exact to 3 dp |
| V drift | **−87.2046%** | exact to 4 dp |
| Wall time (2560 steps) | ~9.5 s | informational |
| ms/step | ~3.7 | informational |

#### Lagrangian VOF

| Metric | Definitive Value | Tolerance |
|--------|------------------|-----------|
| L1 error | **156.916%** | exact to 3 dp |
| V drift | **−9.5321%** | exact to 4 dp |
| Wall time (2560 steps) | ~483 s | informational |
| ms/step | ~188.7 | informational |

#### Head-to-Head Winner

| Metric | Winner |
|--------|--------|
| L1 error | Eulerian (98.3% < 156.9%) |
| V drift (absolute) | **Lagrangian** (9.5% < 87.2%) |
| Wall time | Eulerian (9.5s < 483s = **51× faster**) |

### 2.3 Physical Interpretation

- At T=10, the vortex creates 5 full spiral turns before reversing. Sub-cell filaments are thinner than dx.
- Eulerian PLIC: `jnp.clip` destroys filaments → 87% mass loss. L1 is "good" only because almost nothing is left.
- Lagrangian: preserves filament geometry via polygon tracking → only 9.5% mass loss. L1 worse because residual mass is spatially misplaced (overlay diffusion) but mass-conserving.
- **This is the canonical demonstration of Lagrangian's accuracy advantage on strong-deformation flows.**

---

## Output Artifacts

### Test 1 (Zalesak 3D)

| Path | Content |
|------|---------|
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_final.npz` | F_init, F_final, N, DX, n_steps, wall_time |
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_result.png` | 3-slice visualization |
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_overview.png` | 2×3 panel overview |

### Test 2 (Rider-Kothe)

| Path | Content |
|------|---------|
| `examples/plic_eulerian_tests/reversed_vortex/results/reversed_vortex_comparison.png` | 2×3 Eulerian vs Lagrangian static |
| `examples/plic_eulerian_tests/reversed_vortex/results/reversed_vortex_comparison.gif` | 81-frame animation (12 fps) |

---

## Reproduction Instructions

```bash
# In the worktree where plic/ has been reimplemented from spec:

# Test 1: Zalesak 3D
python examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py

# Test 2: Rider-Kothe
python examples/plic_eulerian_tests/reversed_vortex/run_comparison.py

# Acceptance: compare L1 and V drift to the tables above
```

The test scripts are copied **unmodified** from the original branch. The only
code that differs is `src/jax_laseram/vof/plic/` (reimplemented from spec).
`src/jax_laseram/vof/lagrangian_3d/` is kept intact (shared dependency).
