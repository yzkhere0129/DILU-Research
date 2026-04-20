#!/usr/bin/env python3
"""Benchmark 4 progressively-optimized variants of Eulerian PLIC at 1M cells.

Each variant isolates ONE optimization to quantify its contribution:

  V0 "Naive":       No JIT,  Dense intercept,  Phase A Regula Falsi  (15 steps)
  V1 "+JIT":        @jit,    Dense intercept,  Phase A RF
  V2 "+Gather":     @jit,    Padded gather,    Phase A RF
  V3 "Current":     @jit,    Padded gather,    Phase B analytic + Newton

Same 1M-cell case as profile_1M_eulerian.py:
  Grid 250x250x16, 45-deg translation, CFL=0.45.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))

import jax
import jax.numpy as jnp
import numpy as np

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.intercept_solver import solve_intercept           # Phase A
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept       # Phase B
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x, sweep_flux_y, apply_flux_y,
)

# Config
NX, NY, NZ = 250, 250, 16
DX = 0.01
CFL = 0.45
SPEED = 1.0
N_STEPS = 3  # measure only 3 steps to avoid hour-long naive variant
NH = 1
MAX_FRAC = 0.15
dt = CFL * DX / SPEED
UX = np.cos(np.radians(45)) * SPEED
UY = np.sin(np.radians(45)) * SPEED

print("=" * 70)
print("  Eulerian PLIC Variants Benchmark — 1M cells, 45-deg, RTX 3050")
print("=" * 70)


def build_F():
    """Cylindrical droplet init, matching profile_1M_eulerian.py."""
    Ntx, Nty, Ntz = NX + 2*NH, NY + 2*NH, NZ + 2*NH
    x = np.linspace(DX/2, NX*DX - DX/2, NX)
    y = np.linspace(DX/2, NY*DX - DX/2, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = 0.25; cx = cy = 1.25
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    hd = np.sqrt(2)*DX/2
    F2d = np.clip((R - dist + hd)/(2*hd), 0, 1).astype(np.float32)
    F = np.zeros((Ntx, Nty, Ntz), dtype=np.float32)
    F[NH:-NH, NH:-NH, NH:-NH] = F2d[:, :, None]
    F[0]=F[1]; F[-1]=F[-2]; F[:,0]=F[:,1]; F[:,-1]=F[:,-2]
    F[:,:,0]=F[:,:,1]; F[:,:,-1]=F[:,:,-2]
    return jnp.asarray(F)


def halo(F):
    F = F.at[0].set(F[1]);   F = F.at[-1].set(F[-2])
    F = F.at[:,0].set(F[:,1]); F = F.at[:,-1].set(F[:,-2])
    F = F.at[:,:,0].set(F[:,:,1]); F = F.at[:,:,-1].set(F[:,:,-2])
    return F


# ────────────────────────────────────────────────────────────────
# V0: Naive — no JIT, dense intercept, Phase A
# ────────────────────────────────────────────────────────────────
def step_naive(F):
    """No JIT, dense intercept (all cells), Phase A Regula Falsi (15 iter)."""
    def sub_x(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        C = solve_intercept(nx, ny, nz, F, DX, DX, DX)  # dense, 15 iters on ALL cells
        flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt_s, DX, DX, DX)
        return halo(apply_flux_x(F, flux, DX, DX, DX))  # no clip

    def sub_y(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        C = solve_intercept(nx, ny, nz, F, DX, DX, DX)
        flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt_s, DX, DX, DX)
        return halo(apply_flux_y(F, flux, DX, DX, DX))

    F = sub_x(F, dt/2)
    F = sub_y(F, dt)
    F = sub_x(F, dt/2)
    return F


# ────────────────────────────────────────────────────────────────
# V1: +JIT only — dense, Phase A, but whole step JIT-compiled
# ────────────────────────────────────────────────────────────────
def step_v1(F):
    return step_naive(F)  # same body, will be JIT'd externally


# ────────────────────────────────────────────────────────────────
# V2: +padded gather — JIT, Phase A on compact, clip added
# ────────────────────────────────────────────────────────────────
def step_v2(F):
    def sub_x(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)

        # padded gather
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * MAX_FRAC)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]

        # Phase A on compact
        C_c = solve_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)

        # scatter
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)

        flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    def sub_y(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * MAX_FRAC)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
        C_c = solve_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
        flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_y(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    F = sub_x(F, dt/2)
    F = sub_y(F, dt)
    F = sub_x(F, dt/2)
    return F


# ────────────────────────────────────────────────────────────────
# V3: Current — JIT, padded gather, Phase B analytic
# ────────────────────────────────────────────────────────────────
def step_v3(F):
    def sub_x(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * MAX_FRAC)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
        C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)  # Phase B
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
        flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    def sub_y(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * MAX_FRAC)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
        C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
        flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_y(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    F = sub_x(F, dt/2)
    F = sub_y(F, dt)
    F = sub_x(F, dt/2)
    return F


# ────────────────────────────────────────────────────────────────
# Run each variant
# ────────────────────────────────────────────────────────────────
results = {}

# V0: naive (no JIT, dense, Phase A) — 1 step only (too slow for 5)
print(f"\n[V0] Naive: no JIT, dense intercept, Phase A RF")
F = build_F()
print(f"  Running 1 step (no JIT — each op is a separate GPU launch)...", flush=True)
t0 = time.time()
F = step_naive(F)
F.block_until_ready()
t_naive = time.time() - t0
print(f"  Step 1: {t_naive*1000:.1f} ms")
# Extrapolate: naive has no variance between steps (no JIT overhead)
results["V0 Naive (no JIT, dense, Phase A)"] = t_naive * 1000

# V1: +JIT (still dense, Phase A)
print(f"\n[V1] +@jax.jit: dense, Phase A RF")
F = build_F()
jit_v1 = jax.jit(step_v1)
t0 = time.time()
_ = jit_v1(F); _.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
times = []
for i in range(N_STEPS):
    t0 = time.time()
    F = jit_v1(F); F.block_until_ready()
    times.append((time.time()-t0)*1000)
med = float(np.median(times))
print(f"  Per step (median of {N_STEPS}): {med:.1f} ms")
results["V1 +JIT (dense, Phase A)"] = med

# V2: +gather (Phase A on compact)
print(f"\n[V2] +padded gather (15%), Phase A RF")
F = build_F()
jit_v2 = jax.jit(step_v2)
t0 = time.time()
_ = jit_v2(F); _.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
times = []
for i in range(N_STEPS):
    t0 = time.time()
    F = jit_v2(F); F.block_until_ready()
    times.append((time.time()-t0)*1000)
med = float(np.median(times))
print(f"  Per step (median of {N_STEPS}): {med:.1f} ms")
results["V2 +gather (Phase A on compact)"] = med

# V3: +Phase B
print(f"\n[V3] Current: gather + Phase B analytic + Newton")
F = build_F()
jit_v3 = jax.jit(step_v3)
t0 = time.time()
_ = jit_v3(F); _.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.1f}s")
times = []
for i in range(N_STEPS):
    t0 = time.time()
    F = jit_v3(F); F.block_until_ready()
    times.append((time.time()-t0)*1000)
med = float(np.median(times))
print(f"  Per step (median of {N_STEPS}): {med:.1f} ms")
results["V3 Current (gather + Phase B)"] = med

# ────────────────────────────────────────────────────────────────
# Summary table
# ────────────────────────────────────────────────────────────────
print()
print("=" * 80)
print(f"  1M-Cell Eulerian PLIC Variant Comparison (RTX 3050, float32)")
print("=" * 80)
print(f"  {'Variant':45s}  {'ms/step':>10s}  {'Mcells/s':>10s}  {'vs naive':>9s}")
print(f"  {'-'*45}  {'-'*10}  {'-'*10}  {'-'*9}")
naive_ms = results["V0 Naive (no JIT, dense, Phase A)"]
for label, ms in results.items():
    mcps = 1_000_000 / (ms/1000) / 1e6
    speedup = naive_ms / ms
    print(f"  {label:45s}  {ms:10.1f}  {mcps:10.2f}  {speedup:8.1f}x")

print()
print("  Lagrangian VOF baseline (C FFI, same hardware): 1621.6 ms/step")
print(f"  → Current (V3) vs Lagrangian baseline: {1621.6 / results['V3 Current (gather + Phase B)']:.1f}x")
print(f"  → Current (V3) vs most naive (V0):     {naive_ms / results['V3 Current (gather + Phase B)']:.1f}x")
print("=" * 80)
