---
name: vof-test-suite
description: |
  Standard VOF (Volume-of-Fluid) validation test suite for interface tracking algorithms. Use this skill whenever the user is developing, debugging, or benchmarking a VOF advection method (Lagrangian, Eulerian, PLIC, SLIC, algebraic, geometric) and wants to know what tests to run, what parameters to use, or what the acceptance criteria are. Also trigger when the user mentions: "run benchmark", "validate VOF", "test interface tracking", "Zalesak", "advection test", "rotation test", "RT instability", "Rayleigh-Taylor", "volume conservation test", "shape preservation", "interface accuracy", or any phrase suggesting they want to verify a VOF implementation against known reference solutions.
---

# VOF Validation Test Suite

A standardized battery of tests for validating VOF interface tracking methods.
Each test targets a specific capability; together they cover the full range
of challenges a VOF method must handle in production.

## Test Selection Guide

When the user asks to validate a VOF method, choose tests based on what
capability they're developing or debugging:

| User's situation | Recommended tests |
|-----------------|-------------------|
| "Just got basic advection working" | **T1** (translation) only |
| "Adding PLIC reconstruction" | **T1** + **T2** (rotation) |
| "Testing obstacle/wall interaction" | **T1** + **T3** (obstacle wrapping) |
| "Full validation for paper/report" | **T1** + **T2** + **T3** |
| "Multi-phase / gravity / density" | **T4** (Rayleigh-Taylor) |
| "Production readiness check" | All four: **T1** + **T2** + **T3** + **T4** |

Run tests in order T1 → T2 → T3 → T4. Each builds on the previous —
if T1 fails, don't bother with T2.

---

## T1: Diagonal Translation (45°)

**What it tests**: Basic advection accuracy, volume conservation, shape preservation
under uniform velocity. This is the MINIMUM test — if this fails, nothing else will work.

**Source**: Barkhudarov (2004) §4.1

### Standard Configuration

```
Domain:     [0, 1]² × [0, Lz]     (Lz = NZ × dx)
Grid:       100 × 100 × 5          (dx = 0.01, 50K cells)
Droplet:    Cylinder R = 0.05 (= D/2), center (0.20, 0.20)
            10 cells per diameter
Velocity:   (cos45°, sin45°, 0) × 1.0 m/s = (0.707, 0.707, 0)
CFL:        0.45
Travel:     5 diameters = 0.50 m → t_end = 0.50 s
Steps:      ~112
Init:       Sub-cell linear ramp (hd = √2 · dx / 2)
```

### Acceptance Criteria

| Metric | Symbol | Formula | PASS threshold | EXCELLENT |
|--------|--------|---------|---------------|-----------|
| Volume error | \|ΔV/V₀\| | \|Σ(F_fin) - Σ(F_init)\| / Σ(F_init) | < 0.5% | < 0.01% |
| Perimeter error | \|ΔP/P₀\| | Perimeter change from F=0.5 contour | < 5% | < 0.5% |
| Shape | visual | F=0.5 contour should be circular | circular | — |

### Multi-angle variant

Run at 0°, 6°, 30°, 45° to expose grid-alignment bias. The 45° case is
the hardest for operator-split methods; Lagrangian methods should be
roughly angle-independent.

### Reference script
`examples/barkhudarov_tests/barkhudarov_v2/test1_advection/barkhudarov_test1_3d.py`

---

## T2: Zalesak Slotted Disk Rotation

**What it tests**: Interface tracking under SHEAR flow (non-uniform velocity).
Rotation produces hex deformation that challenges PLIC reconstruction and
overlay accuracy. The slot is a sub-grid feature that tests sharp-corner preservation.

**Source**: Zalesak (1979), Rider & Kothe (1998)

### Standard Configuration (2D-extruded)

```
Domain:     [0, 1]² × [0, 0.05]
Grid:       100 × 100 × 5          (dx = 0.01)
Disk:       R = 0.15, center (0.50, 0.75)
Slot:       Width 0.05 (5 cells), depth 0.25
Rotation:   ω = 1 rad/s around (0.5, 0.5)
            Velocity: u = -ω(y-0.5), v = ω(x-0.5), w = 0
CFL:        0.45
Duration:   One full rotation: T = 2π ≈ 6.28 s
Steps:      ~988
Init:       4×4 sub-sampling per cell
```

### True 3D variant

```
Domain:     [0, 1]³              (cubic)
Grid:       100 × 100 × 100     (dx = 0.01, 1M cells)
Sphere:     R = 0.15, center (0.5, 0.5, 0.75)
Slot:       Width 0.05 in x and y, depth 0.25 in z
Rotation:   Around x-axis at (*, 0.5, 0.5)
            Velocity: u = 0, v = -ω(z-0.5), w = ω(y-0.5)
Steps:      ~988
Init:       3×3×3 sub-sampling per cell
```

### RK2 Midpoint Velocity Correction (recommended)

For solid-body rotation, replace the velocity field with the
analytical midpoint value to reduce trajectory curvature error
from O((ωdt)²) to O((ωdt)³) per step:

```
u_rk2 = u₀ - 0.5·dt·ω²·(perpendicular component)
```

For rotation around z: `u_rk2 = u₀ - 0.5·dt·ω²·(x-xc)`, `v_rk2 = v₀ - 0.5·dt·ω²·(y-yc)`
For rotation around x: `v_rk2 = v₀ - 0.5·dt·ω²·(y-yc)`, `w_rk2 = w₀ - 0.5·dt·ω²·(z-zc)`

This is a zero-cost improvement (only changes velocity pre-compute),
typically halving L1 error.

### Acceptance Criteria

| Metric | Formula | PASS (100²) | EXCELLENT (200²) |
|--------|---------|-------------|------------------|
| L1 error | ∫\|F_fin - F_init\| dV / V₀ | < 10% | < 3% |
| Volume error | \|ΔV/V₀\| | < 1% | < 0.5% |
| Slot preservation | visual: slot open, corners visible | yes | sharp corners |
| Perimeter | \|ΔP/P₀\| | < 100% | < 10% |

### Grid refinement study

Run at multiple resolutions to verify convergence rate:
- 50² → 100² → 200² (2D-extruded)
- 64³ → 100³ (true 3D)
Expected: L1 ∝ dx¹ (first-order convergence for PLIC methods)

### Literature reference values (100², one rotation)

| Method | L1 error |
|--------|---------|
| First-order upwind | ~30% |
| Zalesak (1979) FCT | ~10% |
| Parker-Youngs PLIC | ~8% |
| Parker-Youngs + RK2 | ~4% |
| Rudman (1997) VOF-PLIC | ~3-5% |
| ELVIRA (Pilliod-Puckett 2004) | ~2-3% |

### Reference scripts
```
examples/barkhudarov_tests/barkhudarov_v2/test3_rotation/
├── zalesak_rotation_3d.py      (2D-extruded baseline)
├── zalesak_rk2.py              (2D-extruded + RK2)
├── zalesak_200_rk2.py          (2D 200² + RK2)
├── zalesak_true_3d.py          (true 3D, 64³)
└── zalesak_true_3d_100.py      (true 3D, 100³)
```

---

## T3: Droplet Impacting Obstacle

**What it tests**: Wall interaction, vertex clamping, volume conservation
near solid boundaries. The fluid must wrap around a corner smoothly.

**Source**: Barkhudarov (2004) §4.2, Figure 7

### Standard Configuration

```
Domain:     [0, 1]² × [0, 0.10]
Grid:       50 × 50 × 5            (dx = 0.02)
Obstacle:   [0, 0.5] × [0, 0.5] × [0, 0.10]  (lower-left quadrant)
Droplet:    Sphere R = 0.20, center (0.70, 0.70, 0.05)
Velocity:   Potential flow (∇²φ=0) with far-field (-1, -1, 0)
            Solve: _solve_potential_flow_2d on domain grid, 5000 Jacobi iter
CFL:        0.45
Duration:   t_end = 0.35 (droplet reaches corner and wraps)
Steps:      ~39
```

### Key implementation details

- **Velocity**: Potential flow gives smooth deceleration at obstacle walls
  and tangential velocity reversal at the corner (essential for wrapping).
  Uniform velocity does NOT work — fluid piles up at the corner instead
  of wrapping.
- **F clip**: Clip F to [0, 1] after each step. Track cumulative overfill
  separately as Barkhudarov's "positive volume error":
  `V_total = V_fluid + V_overfill_cumulative`
- **Obstacle treatment**: Redistribute boundary obstacle cell fluid to
  adjacent fluid cells, then zero all obstacle cells.

### Acceptance Criteria

| Metric | PASS | EXCELLENT |
|--------|------|-----------|
| V_total (including tracked overfill) | \|ΔV_total\| < 1% | \|ΔV_total\| < 0.1% |
| Volume error sign | positive (overfill) | — |
| Shape | symmetric wrapping | matches Fig 7 contour |
| Slot between fluid and obstacle | clean separation | — |

### Reference script
`examples/barkhudarov_tests/barkhudarov_v2/test2_obstacle/test2_obstacle_3d.py`

---

## T4: Rayleigh-Taylor Instability

**What it tests**: Coupled VOF + momentum (gravity-driven multi-phase flow).
Tests whether the VOF method works correctly within a full Navier-Stokes
solver, including density-driven instability, mushroom cap formation, and
long-time interface evolution.

**Source**: Tryggvason (1988), He et al. (1999)

### Standard Configuration

```
Domain:     [0, 1] × [0, 4]        (aspect ratio 1:4)
Grid:       128 × 512              (dx = 0.0078125)
Heavy fluid: ρ_H = 1.225 kg/m³    (above, alpha=1)
Light fluid: ρ_L = 0.169 kg/m³    (below, alpha=0)
Atwood:     At = (ρ_H - ρ_L) / (ρ_H + ρ_L) ≈ 0.758
Interface:  y = 2.0 + 0.05 · cos(2πx)  (single-mode perturbation)
Gravity:    g = -9.81 m/s²  (pointing -y)
BCs:        Top/bottom: fixed value (alpha=1/0)
            Left/right: symmetry
Duration:   t_end = 1.0 s
Init:       8×8 sub-sampling per cell
```

### Metrics

| Metric | Symbol | How to measure |
|--------|--------|----------------|
| Bubble tip position | h1(t) | Highest y where alpha=0.5 at x=0 |
| Spike tip position | h2(t) | Lowest y where alpha=0.5 at x=0.5 |
| Mass conservation | ΔM/M₀ | Sum of (alpha·ρ_H + (1-alpha)·ρ_L) × cell_vol |
| Mushroom shape | visual | Should form symmetric mushroom cap |

### Acceptance Criteria

| Metric | PASS | EXCELLENT |
|--------|------|-----------|
| h1(t) vs reference | within 5% of He et al. (1999) | within 2% |
| h2(t) vs reference | within 5% | within 2% |
| Mass conservation | \|ΔM/M₀\| < 0.1% | < 0.01% |
| Mushroom cap | visible, symmetric | sharp roll-up, thin filament |

### Reference data
`examples/RT_air_helium/of_h1_h2.npz` — OpenFOAM reference for h1(t), h2(t)

### Notes
- This test requires a FULL NS solver (not just VOF advection)
- Run time is significant (~minutes to hours depending on solver)
- Grid refinement: 64×256 → 128×512 → 256×1024
- The RT test validates the VOF module within the coupled solver context,
  not the VOF advection kernel in isolation

### Reference scripts
```
examples/RT_air_helium/
├── initAlpha.py        (initialize alpha field)
└── postProcess.py      (extract h1, h2, mass, generate plots)
```

---

## Test Execution Protocol

When running any test from this suite:

### Before running
1. Verify the code compiles/imports without errors
2. Check GPU/CPU availability (VRAM for large grids)
3. Estimate wall time based on grid size and hardware

### During running
1. Print per-step diagnostics: volume, max F, step time
2. Save intermediate snapshots for evolution visualization

### After running
1. Compute ALL metrics from the acceptance criteria table
2. Generate comparison plots (initial vs final, overlay)
3. Save results to JSON for reproducibility
4. Compare against reference values / previous runs
5. Print a clear PASS / FAIL / EXCELLENT verdict

### Reporting format

```
═══════════════════════════════════════════════
  VOF Test Report: [Test Name]
═══════════════════════════════════════════════
  Grid:       [NX × NY × NZ]
  Method:     [PLIC, SLIC, etc.]
  Hardware:   [GPU model]
  Wall time:  [seconds]

  METRICS:
    Volume error:     [value]%    [PASS/FAIL]
    L1 error:         [value]%    [PASS/FAIL]
    Perimeter error:  [value]%    [PASS/FAIL]

  VERDICT: [PASS / FAIL / EXCELLENT]
═══════════════════════════════════════════════
```

---

## Quick Reference: Which test catches which bug

| Bug type | T1 catches | T2 catches | T3 catches | T4 catches |
|----------|-----------|-----------|-----------|-----------|
| Wrong advection formula | ✓ | ✓ | ✓ | ✓ |
| Poor PLIC normals | — | ✓ | — | ✓ |
| Grid-aligned bias | ✓(multi-angle) | ✓ | — | — |
| Hex compression errors | — | ✓ | ✓ | — |
| Obstacle BC errors | — | — | ✓ | — |
| F>1 clip issues | — | — | ✓ | — |
| Shear flow errors | — | ✓ | — | ✓ |
| Density coupling errors | — | — | — | ✓ |
| Long-time stability | — | ✓(988 steps) | — | ✓ |
| Float32 precision | ✓ | ✓ | ✓ | — |
