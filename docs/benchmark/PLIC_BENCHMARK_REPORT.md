# PLIC Droplet 45° Advection — Benchmark Report

**Date**: 2026-04-12  
**Branch**: `yzk/plic-research` · commit `4f9b818`  
**Case**: `examples/plic_eulerian_tests/droplet_45deg/`

---

## 1. Case Parameters

### Domain & Grid

| Parameter | Value |
|-----------|-------|
| Domain | [0, 1] × [0, 1] × [0, 0.03] m |
| Grid | 100 × 100 × 3 cells (quasi-2D, z-homogeneous) |
| Halo | 1 layer per side |
| dx = dy = dz | 0.01 m |
| Total cells (interior) | 30,000 |
| Total cells (with halos) | 52,020 |

### Physics

| Parameter | Value |
|-----------|-------|
| Droplet center | (0.2, 0.2) |
| Droplet radius R | 0.05 m (D = 0.1 m) |
| Interface transition | tanh, width = 1.5 dx |
| Velocity field | constant u = v = 1.0 m/s, w = 0 |
| Surface tension | 0 (pure kinematic test) |
| Gravity | 0 |
| Density ratio | 1:1 (single-phase) |

### Time Integration

| Parameter | Value |
|-----------|-------|
| Method | Strang split (x/2 → y → x/2) |
| CFL | 0.498 |
| dt | 4.980 × 10⁻³ s |
| t_end | 0.3536 s |
| n_steps | 71 |
| Analytic final center | (0.5536, 0.5536) |

### PLIC Algorithm

| Component | Method | JAX Idiom |
|-----------|--------|-----------|
| Normal | Parker-Young 3×3×3 weighted gradient | `jax.lax.conv_general_dilated` (cuDNN) |
| Intercept | Regula Falsi + bisection fallback, 15 fixed steps | `jax.lax.fori_loop` |
| Volume formula | Scardovelli-Zaleski 2000, 6-branch analytic | nested `jnp.where` cascade |
| Flux | Eulerian donor sweep-box ∩ PLIC plane | `volume_below_plane_3d` on sub-box |
| Pure-cell fallback | F_donor × sub_box_vol (upwind) | `jnp.where(is_interface, plic, upwind)` |

### Precision

| Setting | Value |
|---------|-------|
| JAX x64 | True |
| Compute dtype | float64 |

---

## 2. Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4096 MB |
| Platform | CUDA (JAX default backend = gpu) |
| JAX version | (jax-env venv, Python 3.12.3) |

---

## 3. Per-Stage GPU Profiling (JIT-compiled)

All times are **GPU wall time** with `jax.block_until_ready()`, 10 reps, JIT-warm.

| Stage | Median (ms) | Min | Max | Std | % of sub-sweep |
|-------|-------------|-----|-----|-----|---------------|
| halo update | 12.34 | 10.49 | 14.20 | 1.04 | 41.7% |
| **normal (Youngs conv)** | **1.45** | 0.86 | 1.48 | 0.26 | 4.9% |
| **intercept (15-step RF)** | **14.09** | 13.41 | 14.89 | 0.55 | **47.6%** |
| sweep_flux_x | 1.48 | 1.37 | 1.53 | 0.05 | 5.0% |
| apply_flux_x | 0.25 | 0.25 | 0.42 | 0.05 | 0.8% |
| **sub-sweep sum (est)** | **29.60** | | | | 100% |
| x3 sub-sweeps (est) | 88.81 | | | | |
| | | | | | |
| **FULL STRANG STEP** | **33.66** | 32.50 | 42.05 | 2.80 | — |
| **XLA fusion savings** | **55.15 ms (62%)** | | | | fused away |

### Key Observations

1. **XLA fusion reclaims 62%**: Individually-JIT'd stages sum to ~89 ms/step, but the fused full-step JIT runs at 34 ms. XLA eliminates intermediate materializations and merges elementwise kernels.

2. **Intercept solver is still the bottleneck** (47.6% of sub-sweep), but now at 14 ms instead of 656 ms pre-JIT. The 15-step `fori_loop` is the dominant cost; the Scardovelli-Zaleski analytic inverse (Stage 4 Phase B) would eliminate this entirely.

3. **Halo update is surprisingly expensive** (42% of sub-sweep when profiled separately). This is because the 6× `.at[].set()` calls don't fuse well as separate JIT-compiled functions. In the fused full-step, the halo cost is absorbed into the surrounding computation.

4. **Youngs normal (conv) is fast** — cuDNN handles the 3×3×3 stencil efficiently at 1.45 ms.

---

## 4. Full Run Results (71 steps)

### Performance

| Metric | Value |
|--------|-------|
| Total wall time | 2.344 s |
| Per-step median | 32.77 ms |
| Per-step min | 32.30 ms |
| Per-step max | 34.57 ms |
| Per-step std | 0.59 ms |
| **Throughput** | **0.92 Mcells/s** |
| JIT compile (one-time) | 5.9 s |

### Accuracy

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Volume drift \|dV/V0\| | 3.27 × 10⁻⁸ | < 1 × 10⁻³ | **PASS** |
| F min | -1.52 × 10⁻⁸ | > -1 × 10⁻⁶ | **PASS** |
| F max | 1.000 | < 1 + 1 × 10⁻⁶ | **PASS** |
| Cells F < 0 | 0 | 0 | **PASS** |
| Cells F > 1 | 0 | 0 | **PASS** |
| Final centroid | (0.55406, 0.55360) | | |
| Analytic final | (0.55360, 0.55360) | | |
| cx error / dx | +0.046 | < 2 | **PASS** |
| cy error / dx | -0.0003 | < 2 | **PASS** |
| Interface cells (final) | 123 | | |
| Interface cells (initial) | 2196 → 123 | | PLIC sharpening |

---

## 5. Resource Usage

### GPU Memory

| Metric | Value |
|--------|-------|
| Peak bytes in use | 36.8 MB |
| Bytes in use (steady) | 5.0 MB |
| F array size | 406 KB |
| VRAM utilization | < 1% of 4 GB |

### HLO Complexity

| Module | JAXPR equations |
|--------|----------------|
| Full Strang step | 1026 |
| Youngs normal | 107 |
| Intercept solver (15 steps) | 33 |
| Sweep flux (one direction) | 160 |

### JIT Compile Time

| Scope | Time |
|-------|------|
| Full Strang step | 5.9 s (one-time) |
| Individual stages | 0.3 – 2.8 s each |

---

## 6. Impact of `@jax.jit`

| | Without JIT | With JIT | Speedup |
|---|---|---|---|
| Per-step wall time | 2232 ms | 32.8 ms | **68x** |
| 71-step total | 145 s | 2.3 s | **63x** |
| Dominant cost | kernel launch overhead (1800+ launches/step) | XLA-fused compute | |
| Accuracy | identical | identical | — |

**Root cause**: Without `@jax.jit`, each `jnp.where`/`jnp.abs`/arithmetic dispatches a separate GPU kernel. The 15-iteration `fori_loop` in the intercept solver alone produces ~1800 individual kernel launches. With JIT, XLA compiles the entire Strang step into a fused computation graph with ~15-30 actual GPU kernels.

---

## 7. Comparison

| Implementation | ms/step | Mcells/s | Hardware | Notes |
|----------------|---------|----------|----------|-------|
| **This work (Eulerian PLIC, JIT GPU)** | **32.8** | **0.92** | RTX 3050 4GB | 30k cells, float64 |
| This work (no JIT, same GPU) | 2232 | 0.013 | RTX 3050 4GB | kernel launch overhead |
| Lagrangian VOF (C FFI) | 1622 | 0.62 | RTX 5060 8GB | 1M cells |
| OpenFOAM isoAdvector | ~0.5 | ~2.0 | CPU C++ | 1M cells |
| Theoretical bandwidth limit | ~0.36 | ~2.8 | RTX 5060 | 1M cells |

### Notes on comparison fairness

- This work runs on 30k cells (100×100×3); the others on 1M cells. Per-cell efficiency may differ at scale due to GPU occupancy.
- This work uses float64; switching to float32 would roughly halve bandwidth and may improve throughput.
- The Lagrangian VOF comparison is on a different GPU (RTX 5060) — but the architectural advantage of Eulerian PLIC (analytic sub-box formula vs polygon clipping) is the dominant factor, not the hardware.
- OpenFOAM uses CPU C++ with active-cell-list sparse iteration — a fundamentally different compute model that avoids the dense-over-sparse overhead inherent in JAX.

---

## 8. Identified Optimization Opportunities

| Priority | Optimization | Expected impact | Stage |
|----------|-------------|-----------------|-------|
| **1** | Scardovelli-Zaleski analytic intercept (replace 15-step fori_loop) | intercept 14 ms → ~1 ms, full step ~20 ms | Stage 4 Phase B |
| **2** | float32 compute (currently float64) | ~2x bandwidth savings, ~1.5x throughput | configurable |
| **3** | Padded-gather sparse intercept (only ~2k interface cells) | intercept 14 ms → ~1 ms (if overhead low) | Stage 4 Q2 |
| **4** | Scale to 1M cells on RTX 5060 | GPU occupancy improves, expect ~5 ms/step | Stage 3-4 |
| **5** | 3D Strang (xyzyx) for full 3D cases | needed for Zalesak 3D, no perf change | Stage 3 |
