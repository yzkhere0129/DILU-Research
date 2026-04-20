# Rider-Kothe Reversed Vortex — Test Configuration

## Geometry & Grid

| Item | Value |
|---|---|
| Domain | `[0, 1] × [0, 1] × [0, 3·dx]` (quasi-2D) |
| Grid | 128 × 128 × 3 |
| `dx = dy = dz` | 1/128 ≈ 0.00781 |
| Halo cells | 1 |

## Initial Condition

| Item | Value |
|---|---|
| Shape | Circle (z-extruded) |
| Center | (0.5, 0.75) |
| Radius | R = 0.15 |
| Init method | Sub-cell 4×4 sampling (anti-aliased F) |
| V₀ | 1.658e-3 |

## Velocity Field (divergence-free, time-reversing)

Stream function `ψ = (1/π) sin²(πx) sin²(πy) cos(πt/T)`

```
u(x,y,t) = -sin²(πx) · sin(2πy) · cos(πt/T)
v(x,y,t) =  sin(2πx) · sin²(πy) · cos(πt/T)
w        =  0
```

- At `t = T/2`, velocity = 0 (max deformation, sub-cell filament)
- At `t = T`, velocity fully reversed → ideal solution returns to initial circle
- `max|u| ≈ 1.0`

## Time Stepping

| Item | Value |
|---|---|
| Period T | 4.0 (also runs at 2.0) |
| CFL | 0.5 |
| dt | CFL·dx / max\|u\| ≈ 3.91e-3 |
| Steps | 1024 for T=4 (512 for T=2) |
| Time scaling | Midpoint `cos(π(t+dt/2)/T)` — 2nd-order temporal |

## Boundary Conditions

All external boundaries: **zero-gradient (Neumann)** via halo copy.

## Metrics

| Metric | Formula |
|---|---|
| L1 shape error | `∫|F(T) − F(0)| dV / V₀ × 100%` |
| Volume drift | `(V(T) − V₀) / V₀ × 100%` |

## Solver Settings

**Eulerian PLIC**
- Normal: Parker-Youngs 3×3×3
- Intercept: Scardovelli-Zaleski analytic + 5 Newton (via `jax.jvp`)
- Padded gather: `MAX_FRAC = 0.15` of total cells
- Strang split: x/2 → y → x/2
- Clip `F ∈ [0,1]` after each sub-sweep

**Lagrangian VOF**
- `advect_vof_lagrangian_3d` from `jax_laseram.vof.lagrangian_3d`
- Hex-box Sutherland-Hodgman overlay (C FFI)

## Hardware

RTX 3050 Laptop (4 GB), float32.

## Results (T=4, 1024 steps)

| | Eulerian PLIC | Lagrangian VOF |
|---|---|---|
| L1 error | 63.6% | 29.7% |
| V drift | −63.5% | −3.7% |
| ms/step | 2.3 | 27.4 |
| JIT compile | 16.4 s | 7.4 s |

## Reproduce

```bash
python examples/plic_eulerian_tests/reversed_vortex/run_comparison.py
```

Outputs `results/reversed_vortex_comparison.png` and `.gif`.
