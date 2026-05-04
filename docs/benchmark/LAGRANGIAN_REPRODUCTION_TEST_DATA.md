# Lagrangian VOF Reproduction Test Data — Definitive Baseline

> **Purpose**: Bit-identical reproduction validation for the Lagrangian VOF
> pipeline (`src/jax_laseram/vof/lagrangian_3d/`). A reimplementation must
> match **every number** in this file when run with the **unmodified** test
> scripts.
>
> **Date**: 2026-04-17  
> **Branch**: `yzk/plic-research`

---

## Hardware & Software

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM) |
| CUDA | 12.x |
| JAX | 0.4.x |
| Python | 3.12.3 |
| Precision | float32 |
| Overlay path | **C native** (`libhexboxclip.so` via `jax.pure_callback`) |

**Note**: The JAX-only overlay fallback (`overlay_3d.py` Sutherland-Hodgman
path) exhibits catastrophic JIT compile time (~2656 s documented) due to
nested `vmap(scan(...))` on variable-length polygon vertex lists. The C
native path MUST be available. The `csrc/` directory (with
`libhexboxclip.so`) is NOT part of the reproduction target — it is
preserved as a compiled dependency.

---

## Test 1: Zalesak 3D (validates shared `reconstruction_3d.py`)

Zalesak 3D uses only Eulerian PLIC, but PLIC imports `_volume_below_3d` and
`compute_plic_normals_3d` from `lagrangian_3d/reconstruction_3d.py`. Matching
Zalesak results confirms the shared primitives are bit-identical.

### Configuration
Identical to Eulerian PLIC reproduction (see `REPRODUCTION_TEST_DATA.md`).

### Acceptance
| Metric | Value |
|--------|-------|
| L1 error | **1.53%** |
| V drift | **−0.054223%** |
| V @ step 901 | 1.348941e-02 |
| n_interface @ 901 | 7832 |

---

## Test 2: Rider-Kothe Reversed Vortex T=2 (validates full Lagrangian pipeline)

### Configuration

| Parameter | Value |
|-----------|-------|
| Script | `examples/plic_eulerian_tests/reversed_vortex/run_comparison.py` |
| Grid | 128 × 128 × 3 |
| Halo | NH = 1 |
| dx = dy = dz | 1/128 |
| Circle center | (0.50, 0.75) |
| Circle radius | 0.15 |
| V₀ | 1.65796280e-03 |
| Initial interface cells | 540 |
| Period T | **2.0** (standard Rider-Kothe) |
| CFL | 0.5 |
| dt | 3.91e-03 |
| n_steps | 512 |
| Strang split | x/2 → y → x/2 |
| Bounds | `jnp.clip(F, 0, 1)` (inline) |

### Stream function

```
ψ(x, y, t) = (1/π) sin²(πx) sin²(πy) cos(πt/T)

u(x,y,t) = −sin²(πx) · sin(2πy) · cos(πt/T)
v(x,y,t) = +sin(2πx) · sin²(πy) · cos(πt/T)
w        = 0
```

### Acceptance (definitive numbers)

| Method | L1 error | V drift | Wall time |
|--------|---------:|--------:|----------:|
| **Eulerian PLIC**   | **41.621%**  | **−41.6194%** | ~2 s |
| **Lagrangian VOF**  | **9.681%**   | **−1.8288%**  | ~27 s |

### Head-to-head interpretation

| Metric | Winner | Advantage |
|--------|--------|-----------|
| L1 shape error | Lagrangian | 4.3× better |
| V drift (absolute) | Lagrangian | 23× better |
| Wall time | Eulerian | 15× faster |

Textbook demonstration: Lagrangian wins precision, Eulerian wins speed on
strong-deformation (sub-cell filament) flows.

---

## Output Artifacts

### Test 1

| Path | Content |
|------|---------|
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_final.npz` | F_init, F_final |
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_result.png` | 3-slice visualization |
| `examples/plic_eulerian_tests/zalesak_3d/results/zalesak_3d_overview.png` | 2×3 overview |

### Test 2

| Path | Content |
|------|---------|
| `examples/plic_eulerian_tests/reversed_vortex/results/reversed_vortex_comparison.png` | Eulerian vs Lagrangian static |
| `examples/plic_eulerian_tests/reversed_vortex/results/reversed_vortex_comparison.gif` | 87-frame animation |

---

## Reproduction Protocol

### Worktree preparation

Delete all `.py` files under `src/jax_laseram/vof/lagrangian_3d/` (but keep
`csrc/`). Then:

```bash
cd <worktree>
python examples/plic_eulerian_tests/zalesak_3d/run_zalesak_3d.py          # Test 1
python examples/plic_eulerian_tests/reversed_vortex/run_comparison.py     # Test 2
```

### Files to reimplement (5 modules, ~1536 lines)

| Module | Lines | Role |
|--------|------:|------|
| `reconstruction_3d.py` | ~280 | Shared: Youngs normals, SZ volume formula, intercept bisection |
| `move_3d.py` | ~240 | Vertex displacement from face velocities |
| `overlay_3d.py` | ~560 | Sutherland-Hodgman hex-box clipping (JAX fallback) |
| `overlay_native.py` | ~250 | C FFI wrapper via `jax.pure_callback` + ctypes |
| `__init__.py` | ~210 | Pipeline orchestrator (`advect_vof_lagrangian_3d`) |

Dead-code modules (NOT required): `overlay_batched.py`, `overlay_fast.py`,
`overlay_sparse.py`, `tet_clip_analytic.py`.

### Reproduction validation — PASSED 2026-04-17

A blind AI agent reproduced all 5 modules (1536 lines total) from
`LAGRANGIAN_VOF_3D_SPEC.md` alone, producing **bit-identical results** on
both tests on the first attempt.
