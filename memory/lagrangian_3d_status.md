---
name: 3D Lagrangian VOF development status
description: Complete status of 3D VOF extension — architecture, performance, benchmark results, C PLIC overlay
type: project
originSessionId: 5e32a9bc-e0e1-4c93-90d7-d06a3fda388e
---
## 3D Lagrangian VOF — Development Status (2026-04-10)

**Branch**: `yzk/lagrangian-3d` on worktree `/home/yzk/JAX-LaserAM-3d-lagrangian`

### Barkhudarov Benchmark Results

**Test 1 — PASSED** (circular droplet 45° advection, 5D travel):
- |dV/V| = 0.00025% (20 steps), shape: circular
- Grid: 100x100x5, dx=0.01, 10 cells/diameter, CFL=0.45

**Test 2 — PASSED** (droplet impacting rectangular obstacle):
- ΔV_total = +0.0000% (exact conservation, overfill tracked separately)
- ΔV_fluid = -2.42% (from F>1 clip), Cumulative overfill = +2.42%
- Smooth symmetric crescent wrapping, matches paper Figure 7
- Grid: 50x50x5, dx=0.02, R=0.20, potential flow velocity

### C/OpenMP PLIC Overlay (2026-04-10)

**~334x total speedup** over JAX batched at 1M cells (measured 2026-04-11).
Overlay alone is ~625x faster; total speedup is limited by other stages.
- 9/9 unit tests pass (C vs JAX exact match to float32)
- Code-quality reviewed, all findings fixed
- Files: `csrc/hex_box_clip.c`, `overlay_native.py`

### 1M cells per-stage timing (2026-04-11, RTX 3050 + 5600H)

| Stage | ms | % |
|-------|-----|---|
| PLIC intercept (bisection, GPU) | 603 | 37% |
| Overlay (C SH PLIC, CPU) | 643 | 40% |
| Clip + halo (GPU) | 227 | 14% |
| PLIC normals + move + clamp | 149 | 9% |
| **Total** | **1621** | 100% |

Bottleneck shifted from overlay-only (96%) to balanced overlay+bisection.
Next optimization: PLIC intercept → C implementation.
| C/OpenMP no-PLIC | ~870 ms | 1 s |

### Critical Bugs Fixed (2026-04-09/10)

1. **jnp.sort(axis=0)** in _volume_below_3d — PLIC bisection completely wrong under vmap
2. **Caller-level local-coordinate shift** — 40x precision improvement
3. **Sobel kernel** for PLIC normals — shape: square → circular
4. **PLIC transfer dV bug**: `F*dV*overlap/fluid_vol` → `F*overlap/fluid_vol`
   dV is wrong in PLIC formula; overlap/fluid_vol already normalizes (Σ=F)
5. **F>1 clip removal**: overlay clip discarded overfill before tracking.
   Now clip in __init__.py, track cumulative overfill = Barkhudarov convention
6. **C PLIC code review fixes**: zero-init keep[], nlen/e1len NaN guards

### Key Architecture

- Pipeline: PLIC (GPU) → face move (GPU) → vertex clamp (GPU) → overlay (CPU/C) → redist → clip → halos
- PLIC path uses C SH overlay via `jax.pure_callback`
- Non-PLIC path uses C 6-tet decomposition
- F clipped to [0,1] in __init__.py; overfill tracked externally in test scripts
- Obstacle: redistribution pushes boundary fluid to neighbors, then zero all obs cells
