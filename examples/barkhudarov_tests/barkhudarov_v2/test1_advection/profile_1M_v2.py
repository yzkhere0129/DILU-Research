#!/usr/bin/env python3
"""Profile 1M-cell pipeline: per-stage timing breakdown.

Grid: 250×250×16 = 1,000,000 cells
Measures each pipeline stage separately:
  Stage 1: PLIC reconstruction (GPU)
  Stage 2: Face move + vertex displacement (GPU)
  Stage 3: Overlay (CPU/C via callback)
  Stage 4: Halo + clip (GPU)

Also compares full pipeline (advect_vof_lagrangian_3d) total.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

import numpy as np
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import (
    compute_plic_normals_3d, compute_intercept_C_3d)
from jax_laseram.vof.lagrangian_3d.move_3d import lagrangian_move_faces_3d
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native, is_native_available)

# ── Config ─────────────────────────────────────────────────────
NX, NY, NZ = 250, 250, 16
DOMAIN_XY = 2.5
DX = DOMAIN_XY / NX
DOMAIN_Z = NZ * DX
CFL = 0.45
SPEED = 1.0
ANGLE_DEG = 45
N_STEPS = 5  # 5 steps averaged for stable per-stage timing

dt = CFL * DX / SPEED
theta = np.radians(ANGLE_DEG)
ux, uy = np.cos(theta) * SPEED, np.sin(theta) * SPEED

print("=" * 70)
print("  1M-Cell Per-Stage Profiling")
print("=" * 70)
print(f"  Grid: {NX}×{NY}×{NZ} = {NX*NY*NZ:,} cells")
print(f"  dx={DX:.4f}, CFL={CFL}, dt={dt:.6f}")
print(f"  C PLIC overlay: {is_native_available()}")
print(f"  JAX backend: {jax.default_backend()}")
print("=" * 70)

# ── GPU memory ─────────────────────────────────────────────────
def gpu_mem_mb():
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return None, None

mem0, mem_total = gpu_mem_mb()

# ── Init ───────────────────────────────────────────────────────
grid = GridInfo(nx=NX, ny=NY, nz=NZ, nh=1, dx=DX, dy=DX, dz=DX,
                x_range=(0, DOMAIN_XY), y_range=(0, DOMAIN_XY), z_range=(0, DOMAIN_Z))
nh = 1
Ntx, Nty, Ntz = NX + 2, NY + 2, NZ + 2
cell_vol = DX**3

# Cylindrical droplet
x = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
y = np.linspace(DX/2, DOMAIN_XY - DX/2, NY)
X, Y = np.meshgrid(x, y, indexing='ij')
R = 0.25; cx, cy = 1.25, 1.25
dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
hd = np.sqrt(2) * DX / 2
F2d = np.clip((R - dist + hd) / (2*hd), 0, 1)

F = np.zeros((Ntx, Nty, Ntz, 1), dtype=np.float32)
F[1:-1, 1:-1, 1:-1, 0] = F2d[:, :, None]
for ax in range(3):
    s = [slice(None)]*4; s2 = list(s)
    s[ax] = 0; s2[ax] = 1; F[tuple(s)] = F[tuple(s2)]
    s[ax] = -1; s2[ax] = -2; F[tuple(s)] = F[tuple(s2)]
F = jnp.array(F)

u_f = jnp.full((Ntx-1, Nty, Ntz), ux, jnp.float32)
v_f = jnp.full((Ntx, Nty-1, Ntz), uy, jnp.float32)
w_f = jnp.zeros((Ntx, Nty, Ntz-1), jnp.float32)

V0 = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
n_active = int(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0]) > 1e-6))
print(f"\n  V0 = {V0:.10f}, active cells: {n_active:,}")

mem1, _ = gpu_mem_mb()
if mem0: print(f"  VRAM: {mem0} → {mem1} MB (+{mem1-mem0} MB alloc)")

# ── JIT warmup (full pipeline) ─────────────────────────────────
print("\n  JIT compiling full pipeline...", end=" ", flush=True)
t0 = time.time()
_ = advect_vof_lagrangian_3d(F, u_f, v_f, w_f, grid, dt)
_.block_until_ready()
jit_full = time.time() - t0
print(f"{jit_full:.1f}s")

mem_jit, _ = gpu_mem_mb()
if mem0: print(f"  VRAM after JIT: {mem_jit} MB (+{mem_jit-mem0} MB)")

# ── Per-stage profiling ────────────────────────────────────────
print(f"\n  Per-stage profiling ({N_STEPS} steps)...\n")

stage_times = {
    'plic_normals': [], 'plic_intercept': [],
    'face_move': [], 'vertex_clamp': [],
    'overlay': [], 'clip_halo': [],
    'total': [],
}

for step in range(N_STEPS):
    t_total_0 = time.time()

    # Stage 1a: PLIC normals
    t0 = time.time()
    nx_f, ny_f, nz_f = compute_plic_normals_3d(F, DX, DX, DX)
    # Gradient gating
    raw_gx = jnp.zeros_like(F)
    raw_gy = jnp.zeros_like(F)
    raw_gz = jnp.zeros_like(F)
    raw_gx = raw_gx.at[1:-1].set((F[2:] - F[:-2]) / (2*DX))
    raw_gy = raw_gy.at[:, 1:-1].set((F[:, 2:] - F[:, :-2]) / (2*DX))
    raw_gz = raw_gz.at[:, :, 1:-1].set((F[:, :, 2:] - F[:, :, :-2]) / (2*DX))
    raw_grad = jnp.sqrt(raw_gx**2 + raw_gy**2 + raw_gz**2)
    interface_mask = raw_grad > 0.5
    nx_f = jnp.where(interface_mask, nx_f, 0.0)
    ny_f = jnp.where(interface_mask, ny_f, 0.0)
    nz_f = jnp.where(interface_mask, nz_f, 0.0)
    nz_f.block_until_ready()
    stage_times['plic_normals'].append(time.time() - t0)

    # Stage 1b: PLIC intercepts
    t0 = time.time()
    C_f = compute_intercept_C_3d(F, nx_f, ny_f, nz_f, DX, DX, DX)
    C_f.block_until_ready()
    stage_times['plic_intercept'].append(time.time() - t0)

    # Stage 2a: Face move
    t0 = time.time()
    xv, yv, zv = lagrangian_move_faces_3d(u_f, v_f, w_f, grid, dt)
    zv.block_until_ready()
    stage_times['face_move'].append(time.time() - t0)

    # Stage 2b: Vertex clamp
    t0 = time.time()
    xv = jnp.clip(xv, 0, DOMAIN_XY)
    yv = jnp.clip(yv, 0, DOMAIN_XY)
    zv = jnp.clip(zv, 0, DOMAIN_Z)
    zv.block_until_ready()
    stage_times['vertex_clamp'].append(time.time() - t0)

    # Stage 3: Overlay (C/OpenMP)
    t0 = time.time()
    F_new = overlay_lagrangian_3d_native(
        F, xv, yv, zv, grid,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
        obs_mask=None)
    F_new.block_until_ready()
    stage_times['overlay'].append(time.time() - t0)

    # Stage 4: Clip + halo
    t0 = time.time()
    F_new = jnp.clip(F_new, 0, 1)
    F_out = jnp.zeros_like(F)
    F_out = F_out.at[1:-1, 1:-1, 1:-1, 0].set(F_new)
    F_out = F_out.at[0].set(F_out[1])
    F_out = F_out.at[-1].set(F_out[-2])
    F_out = F_out.at[:, 0].set(F_out[:, 1])
    F_out = F_out.at[:, -1].set(F_out[:, -2])
    F_out = F_out.at[:, :, 0].set(F_out[:, :, 1])
    F_out = F_out.at[:, :, -1].set(F_out[:, :, -2])
    F_out.block_until_ready()
    stage_times['clip_halo'].append(time.time() - t0)

    t_total = time.time() - t_total_0
    stage_times['total'].append(t_total)

    F = F_out

    V = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
    print(f"  Step {step+1}: total={t_total*1000:.0f}ms  dV={((V-V0)/V0*100):+.6f}%")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-STAGE TIMING (avg over {N} steps, {C} cells)".format(
    N=N_STEPS, C=f"{NX*NY*NZ:,}"))
print("=" * 70)
print(f"  {'Stage':30s}  {'Avg (ms)':>10s}  {'% of total':>10s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}")

avg_total = np.mean(stage_times['total']) * 1000
for stage in ['plic_normals', 'plic_intercept', 'face_move', 'vertex_clamp',
              'overlay', 'clip_halo']:
    avg = np.mean(stage_times[stage]) * 1000
    pct = avg / avg_total * 100
    print(f"  {stage:30s}  {avg:10.1f}  {pct:9.1f}%")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
print(f"  {'TOTAL':30s}  {avg_total:10.1f}  {'100.0':>9s}%")
print(f"\n  Throughput: {NX*NY*NZ / (avg_total/1000) / 1e6:.2f} Mcells/s")

V_fin = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
print(f"  Volume: dV/V0 = {((V_fin-V0)/V0*100):+.6f}%")

if mem0:
    mem_end, _ = gpu_mem_mb()
    print(f"\n  VRAM: {mem0} → {mem_end} MB (+{mem_end-mem0} MB), total {mem_total} MB")
print("=" * 70)
