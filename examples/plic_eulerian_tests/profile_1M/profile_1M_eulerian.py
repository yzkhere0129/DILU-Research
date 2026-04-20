#!/usr/bin/env python3
"""Profile 1M-cell Eulerian PLIC pipeline — apples-to-apples comparison
with the Lagrangian VOF profile_1M_v2.py reference.

Same case configuration:
  Grid: 250x250x16 = 1,000,000 cells
  dx=0.01, CFL=0.45, dt=4.5e-3
  Droplet: R=0.25, cylinder at (1.25, 1.25) through z
  Velocity: 45 deg, SPEED=1.0
  N_STEPS = 5 (averaged for stable timing)

Stage breakdown (matches Lagrangian stages where possible):
  Stage 1: Normal (Youngs 3x3x3 conv)         [GPU]
  Stage 2: Gather (interface cells, size=15%) [GPU]
  Stage 3: Intercept (analytic + 5 Newton)    [GPU]
  Stage 4: Scatter (compact → full)           [GPU]
  Stage 5: Sweep flux (3 sub-sweeps)          [GPU]
  Stage 6: Apply flux + clip + halo           [GPU]

Lagrangian reference on same hardware (RTX 3050) — for direct comparison:
  Overlay (C PLIC):              642.6 ms  40%  CPU (C FFI callback)
  PLIC intercept bisection:      602.7 ms  37%  GPU
  Clip + halo:                   227.3 ms  14%  GPU
  Face move + normals + clamp:   149.0 ms   9%  GPU
  ─────────────────────────────────────────────
  TOTAL:                         1621.6 ms 100%

Expected Eulerian PLIC total: ~60-80 ms (no polygon clipping needed).
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
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x,
    sweep_flux_y, apply_flux_y,
)


# ─── Config (matches profile_1M_v2.py) ───────────────────────────
NX, NY, NZ = 250, 250, 16
DOMAIN_XY = 2.5
DX = DOMAIN_XY / NX        # 0.01
DOMAIN_Z = NZ * DX         # 0.16
CFL = 0.45
SPEED = 1.0
ANGLE_DEG = 45
N_STEPS = 5
NH = 1
MAX_FRAC = 0.15            # padded gather ceiling

dt = CFL * DX / SPEED
theta = np.radians(ANGLE_DEG)
UX, UY = np.cos(theta) * SPEED, np.sin(theta) * SPEED


def gpu_mem_mb():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return None, None


print("=" * 70)
print("  1M-Cell Per-Stage Profiling — EULERIAN PLIC")
print("=" * 70)
print(f"  Grid: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} cells")
print(f"  dx={DX:.4f}, CFL={CFL}, dt={dt:.6f}")
print(f"  Velocity: {ANGLE_DEG} deg, (ux, uy) = ({UX:.4f}, {UY:.4f})")
print(f"  JAX backend: {jax.default_backend()}, device: {jax.devices()[0]}")
print("=" * 70)


# ─── Init: cylindrical droplet, sub-cell linear ramp (same as Lagrangian ref) ───
mem0, mem_total = gpu_mem_mb()

x_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
y_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NY)
X2, Y2 = np.meshgrid(x_c, y_c, indexing="ij")
R = 0.25; cx, cy = 1.25, 1.25
dist = np.sqrt((X2 - cx)**2 + (Y2 - cy)**2)
hd = np.sqrt(2) * DX / 2
F2d = np.clip((R - dist + hd) / (2*hd), 0, 1).astype(np.float32)

Ntx, Nty, Ntz = NX + 2*NH, NY + 2*NH, NZ + 2*NH
F = np.zeros((Ntx, Nty, Ntz), dtype=np.float32)
F[NH:-NH, NH:-NH, NH:-NH] = F2d[:, :, None]
F[0] = F[1]; F[-1] = F[-2]
F[:, 0] = F[:, 1]; F[:, -1] = F[:, -2]
F[:, :, 0] = F[:, :, 1]; F[:, :, -1] = F[:, :, -2]
F = jnp.asarray(F)

cell_vol = DX**3
V0 = float(F[NH:-NH, NH:-NH, NH:-NH].sum() * cell_vol)
n_active = int(jnp.sum(F[NH:-NH, NH:-NH, NH:-NH] > 1e-6))
print(f"\n  V0 = {V0:.10f}, active cells = {n_active:,}")

mem1, _ = gpu_mem_mb()
if mem0:
    print(f"  VRAM: {mem0} → {mem1} MB (+{mem1-mem0} MB alloc)")


# ─── Halo closure ────────────────────────────────────────────────
def halo(F):
    F = F.at[0].set(F[1]);    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F


# ─── Stage primitives (each JIT'd for profiling isolation) ─────────
TOTAL_CELLS = F.size
MAX_N = int(TOTAL_CELLS * MAX_FRAC)

@jax.jit
def stage_halo(F):
    return halo(F)

@jax.jit
def stage_normal(F):
    return compute_youngs_normal_3d(F, DX, DX, DX)

@jax.jit
def stage_gather(F, nx, ny, nz):
    is_if = (F > 1e-6) & (F < 1.0 - 1e-6)
    idx = jnp.where(is_if.ravel(), size=MAX_N, fill_value=0)[0]
    F_c = F.ravel()[idx]
    nx_c = nx.ravel()[idx]
    ny_c = ny.ravel()[idx]
    nz_c = nz.ravel()[idx]
    return idx, F_c, nx_c, ny_c, nz_c

@jax.jit
def stage_intercept(nx_c, ny_c, nz_c, F_c):
    return analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)

@jax.jit
def stage_scatter(C_c, idx):
    C_flat = jnp.zeros(TOTAL_CELLS, dtype=jnp.float32)
    C_flat = C_flat.at[idx].set(C_c)
    return C_flat.reshape(F.shape)

@jax.jit
def stage_flux_x(F, nx, ny, nz, C):
    return sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)

@jax.jit
def stage_flux_y(F, nx, ny, nz, C):
    return sweep_flux_y(F, nx, ny, nz, C, UY, dt, DX, DX, DX)

@jax.jit
def stage_apply_x(F, flux):
    Fo = apply_flux_x(F, flux, DX, DX, DX)
    return halo(jnp.clip(Fo, 0.0, 1.0))

@jax.jit
def stage_apply_y(F, flux):
    Fo = apply_flux_y(F, flux, DX, DX, DX)
    return halo(jnp.clip(Fo, 0.0, 1.0))


# ─── JIT warmup ─────────────────────────────────────────────────
print("\n  JIT compiling pipeline stages...", flush=True)
t0 = time.time()
F_h = stage_halo(F); F_h.block_until_ready()
nx, ny, nz = stage_normal(F_h); nz.block_until_ready()
idx, F_c, nx_c, ny_c, nz_c = stage_gather(F_h, nx, ny, nz); F_c.block_until_ready()
C_c = stage_intercept(nx_c, ny_c, nz_c, F_c); C_c.block_until_ready()
C = stage_scatter(C_c, idx); C.block_until_ready()
flux_x_val = stage_flux_x(F_h, nx, ny, nz, C); flux_x_val.block_until_ready()
flux_y_val = stage_flux_y(F_h, nx, ny, nz, C); flux_y_val.block_until_ready()
F_tmp = stage_apply_x(F_h, flux_x_val); F_tmp.block_until_ready()
F_tmp = stage_apply_y(F_h, flux_y_val); F_tmp.block_until_ready()
jit_time = time.time() - t0
print(f"  JIT compile time (all stages): {jit_time:.1f}s")

mem_jit, _ = gpu_mem_mb()
if mem0:
    print(f"  VRAM after JIT: {mem_jit} MB (+{mem_jit-mem0} MB)")


# ─── Per-stage profiling ────────────────────────────────────────
print(f"\n  Per-stage profiling ({N_STEPS} Strang steps)...\n")

stage_times = {
    "halo": [],
    "normal": [],
    "gather": [],
    "intercept": [],
    "scatter": [],
    "flux_x": [],
    "apply_x": [],
    "flux_y": [],
    "apply_y": [],
    "total": [],
}


def time_call(fn):
    """Block-after-call GPU timing."""
    t0 = time.time()
    r = fn()
    if isinstance(r, tuple):
        r[0].block_until_ready()
    else:
        r.block_until_ready()
    return r, time.time() - t0


for step in range(N_STEPS):
    t_step0 = time.time()

    # Strang step: x/2 → y → x/2 (3 sub-sweeps, each with full reconstruction)
    for sub_label, apply_fn, flux_fn, dt_s in [
        ("x", stage_apply_x, stage_flux_x, dt/2),
        ("y", stage_apply_y, stage_flux_y, dt),
        ("x", stage_apply_x, stage_flux_x, dt/2),
    ]:
        # --- halo ---
        F, t = time_call(lambda: stage_halo(F))
        stage_times["halo"].append(t)

        # --- normal ---
        (nx, ny, nz), t = time_call(lambda: stage_normal(F))
        stage_times["normal"].append(t)

        # --- gather ---
        (idx, F_c, nx_c, ny_c, nz_c), t = time_call(lambda: stage_gather(F, nx, ny, nz))
        stage_times["gather"].append(t)

        # --- intercept on compact ---
        C_c, t = time_call(lambda: stage_intercept(nx_c, ny_c, nz_c, F_c))
        stage_times["intercept"].append(t)

        # --- scatter back ---
        C, t = time_call(lambda: stage_scatter(C_c, idx))
        stage_times["scatter"].append(t)

        # --- flux ---
        if sub_label == "x":
            flux, t = time_call(lambda: stage_flux_x(F, nx, ny, nz, C))
            stage_times["flux_x"].append(t)
            # --- apply + clip + halo ---
            F, t = time_call(lambda: stage_apply_x(F, flux))
            stage_times["apply_x"].append(t)
        else:
            flux, t = time_call(lambda: stage_flux_y(F, nx, ny, nz, C))
            stage_times["flux_y"].append(t)
            F, t = time_call(lambda: stage_apply_y(F, flux))
            stage_times["apply_y"].append(t)

    t_step = time.time() - t_step0
    stage_times["total"].append(t_step)

    V = float(F[NH:-NH, NH:-NH, NH:-NH].sum() * cell_vol)
    dV = (V - V0) / V0 * 100
    print(f"  Step {step+1}: total={t_step*1000:.1f}ms  dV/V={dV:+.6f}%")


# ─── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  PER-STAGE TIMING (avg over {N_STEPS} steps, {NX*NY*NZ:,} cells)")
print("=" * 70)
print(f"  {'Stage':30s}  {'Avg (ms)':>10s}  {'% total':>8s}  {'Dev':>6s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*6}")

# Per full Strang step each primitive is called 3× (once per sub-sweep).
# Report the PER-STEP total (sum over 3 sub-sweeps).
avg_total = np.mean(stage_times["total"]) * 1000  # median per full Strang step
totals_per_stage = {}
for stage, times in stage_times.items():
    if stage == "total":
        continue
    # Each step executes the stage 3 times (or 2x-1x for flux_x vs flux_y).
    # Since we appended once per execution, group by step:
    n_per_step = len(times) // N_STEPS
    per_step_ms = np.mean(times) * 1000 * n_per_step  # mean * calls-per-step
    totals_per_stage[stage] = per_step_ms
    pct = per_step_ms / avg_total * 100
    print(f"  {stage:30s}  {per_step_ms:10.1f}  {pct:7.1f}%  GPU")

print(f"  {'─'*30}  {'─'*10}  {'─'*8}  {'─'*6}")
print(f"  {'TOTAL (Strang step)':30s}  {avg_total:10.1f}  {'100.0':>7s}%")
print()
print(f"  Throughput: {NX*NY*NZ / (avg_total/1000) / 1e6:.2f} Mcells/s")

V_fin = float(F[NH:-NH, NH:-NH, NH:-NH].sum() * cell_vol)
print(f"  Volume drift: {((V_fin-V0)/V0*100):+.6f}%")
Fi = F[NH:-NH, NH:-NH, NH:-NH]
print(f"  F range: [{float(Fi.min()):+.3e}, {float(Fi.max()):.6f}]")

if mem0:
    mem_end, _ = gpu_mem_mb()
    print(f"\n  VRAM: {mem0} → {mem_end} MB (+{mem_end-mem0} MB), total {mem_total} MB")

print("=" * 70)
print()
print("=" * 70)
print("  HEAD-TO-HEAD: Lagrangian VOF  vs  Eulerian PLIC  (1M cells, 45deg)")
print("=" * 70)
print(f"""
  ┌─────────────────────────────┬──────────┬──────┬──────┐
  │ LAGRANGIAN STAGE            │ Time     │  %   │ Loc  │
  ├─────────────────────────────┼──────────┼──────┼──────┤
  │ Overlay (C PLIC)            │ 642.6 ms │ 40%  │ CPU  │
  │ PLIC intercept bisection    │ 602.7 ms │ 37%  │ GPU  │
  │ Clip + halo                 │ 227.3 ms │ 14%  │ GPU  │
  │ Face move + normals + clamp │ 149.0 ms │  9%  │ GPU  │
  │ TOTAL                       │1621.6 ms │ 100% │      │
  └─────────────────────────────┴──────────┴──────┴──────┘

  ┌─────────────────────────────┬──────────┬──────┬──────┐
  │ EULERIAN PLIC STAGE         │ Time     │  %   │ Loc  │
  ├─────────────────────────────┼──────────┼──────┼──────┤""")
for stage in ["halo", "normal", "gather", "intercept", "scatter",
              "flux_x", "flux_y", "apply_x", "apply_y"]:
    t_ms = totals_per_stage[stage]
    pct = t_ms / avg_total * 100
    print(f"  │ {stage:27s} │ {t_ms:6.1f} ms │ {pct:4.1f}%│ GPU  │")
print(f"  │ {'TOTAL':27s} │ {avg_total:6.1f} ms │100.0%│      │")
print("  └─────────────────────────────┴──────────┴──────┴──────┘")
print()
print(f"  ➜ Eulerian PLIC speedup vs Lagrangian: {1621.6/avg_total:.1f}x")
print("=" * 70)
