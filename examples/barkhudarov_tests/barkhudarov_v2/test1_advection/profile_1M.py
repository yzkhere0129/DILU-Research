#!/usr/bin/env python3
"""Profile 1M-cell droplet advection: GPU vs Hybrid (GPU+CPU) pipeline.

Grid: 250×250×16 = 1,000,000 cells
Droplet: R=0.25, center=(1.25,1.25), 45° advection
Runs 5 steps to measure per-step timing and VRAM usage.

Measures:
  1. JIT compilation time
  2. Per-step wall time (PLIC + move + overlay)
  3. GPU VRAM usage (peak)
  4. Volume conservation
"""
import sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d

# ── Configuration ──────────────────────────────────────────────
NX, NY, NZ = 250, 250, 16
DOMAIN_XY = 2.5
DX = DOMAIN_XY / NX  # 0.01
DOMAIN_Z = NZ * DX
D = 0.50  # droplet diameter
R = D / 2
CX, CY = 1.25, 1.25
ANGLE_DEG = 45
SPEED = 1.0
CFL = 0.45
N_STEPS = 5

dt = CFL * DX / SPEED
theta = np.radians(ANGLE_DEG)
ux, uy = np.cos(theta) * SPEED, np.sin(theta) * SPEED

print("=" * 70)
print("  1M-Cell Profiling: 3D Lagrangian VOF Pipeline")
print("=" * 70)
print(f"  Grid: {NX}×{NY}×{NZ} = {NX*NY*NZ:,} cells")
print(f"  Domain: [{DOMAIN_XY}×{DOMAIN_XY}×{DOMAIN_Z:.2f}]")
print(f"  dx=dy=dz={DX:.4f}")
print(f"  Droplet: D={D}, R={R}, center=({CX},{CY})")
print(f"  Velocity: ({ux:.4f},{uy:.4f},0) at {ANGLE_DEG}°")
print(f"  CFL={CFL}, dt={dt:.6f}")
print(f"  Steps: {N_STEPS}")
print(f"  JAX backend: {jax.default_backend()}")
print(f"  Devices: {jax.devices()}")
print("=" * 70)

# ── GPU memory before ──────────────────────────────────────────
def get_gpu_mem_mb():
    """Get current GPU memory usage in MB (NVIDIA only)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        pass
    return None, None

mem_used_0, mem_total = get_gpu_mem_mb()
if mem_used_0 is not None:
    print(f"\n  GPU VRAM before: {mem_used_0} / {mem_total} MB")

# ── Init grid + F ──────────────────────────────────────────────
grid = GridInfo(
    nx=NX, ny=NY, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN_XY),
    y_range=(0.0, DOMAIN_XY),
    z_range=(0.0, DOMAIN_Z),
)
nh = grid.nh
Ntx, Nty, Ntz = NX + 2, NY + 2, NZ + 2
cell_vol = DX**3

# Cylindrical droplet (uniform in z)
x = np.linspace(DX / 2, DOMAIN_XY - DX / 2, NX)
y = np.linspace(DX / 2, DOMAIN_XY - DX / 2, NY)
X, Y = np.meshgrid(x, y, indexing="ij")
dist = np.sqrt((X - CX) ** 2 + (Y - CY) ** 2)
hd = np.sqrt(2) * DX / 2
F_2d = np.where(dist + hd <= R, 1.0,
        np.where(dist - hd >= R, 0.0,
                 np.clip((R - dist + hd) / (2 * hd), 0, 1)))

F = np.zeros((Ntx, Nty, Ntz, 1), dtype=np.float32)
F[1:-1, 1:-1, 1:-1, 0] = F_2d[:, :, None]
F[0] = F[1]; F[-1] = F[-2]
F[:, 0] = F[:, 1]; F[:, -1] = F[:, -2]
F[:, :, 0] = F[:, :, 1]; F[:, :, -1] = F[:, :, -2]
F = jnp.array(F)

# Uniform velocity
u_face = jnp.full((Ntx - 1, Nty, Ntz), ux, jnp.float32)
v_face = jnp.full((Ntx, Nty - 1, Ntz), uy, jnp.float32)
w_face = jnp.zeros((Ntx, Nty, Ntz - 1), jnp.float32)

V0 = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
n_interface = int(np.sum((np.array(F[1:-1, 1:-1, 1:-1, 0]) > 1e-6) &
                          (np.array(F[1:-1, 1:-1, 1:-1, 0]) < 1 - 1e-6)))
n_nonzero = int(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0]) > 1e-6))
print(f"\n  V₀ = {V0:.10f}")
print(f"  Active cells (F>0): {n_nonzero:,}")
print(f"  Interface cells: {n_interface:,}")
print(f"  Total array size: F={F.shape}, u={u_face.shape}")
print(f"  F memory: {F.nbytes / 1e6:.1f} MB")
print(f"  u+v+w memory: {(u_face.nbytes + v_face.nbytes + w_face.nbytes) / 1e6:.1f} MB")

# ── GPU memory after allocation ────────────────────────────────
mem_used_1, _ = get_gpu_mem_mb()
if mem_used_0 is not None and mem_used_1 is not None:
    print(f"  GPU VRAM after alloc: {mem_used_1} MB (+{mem_used_1 - mem_used_0} MB)")

# ── Import sub-components for per-stage profiling ──────────────
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import (
    compute_plic_normals_3d, compute_intercept_C_3d)
from jax_laseram.vof.lagrangian_3d.move_3d import lagrangian_move_faces_3d
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native, is_native_available)
from jax_laseram.vof.lagrangian_3d.overlay_batched import (
    overlay_lagrangian_3d_batched)

print(f"\n  Native C overlay available: {is_native_available()}")

# ═══════════════════════════════════════════════════════════════
# MODE A: Full pipeline (PLIC → falls back to JAX batched overlay)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  MODE A: Full PLIC pipeline (JAX batched overlay)")
print("=" * 70)

F_a = jnp.array(F)  # fresh copy

print("  JIT compiling...", end=" ", flush=True)
t0 = time.time()
_ = advect_vof_lagrangian_3d(F_a, u_face, v_face, w_face, grid, dt)
_.block_until_ready()
jit_a = time.time() - t0
print(f"{jit_a:.1f}s")

mem_a, _ = get_gpu_mem_mb()

times_a = []
for s in range(N_STEPS):
    t0 = time.time()
    F_a = advect_vof_lagrangian_3d(F_a, u_face, v_face, w_face, grid, dt)
    F_a.block_until_ready()
    times_a.append(time.time() - t0)
    V = float(np.sum(np.array(F_a[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
    print(f"    step {s+1}: {times_a[-1]*1000:.0f} ms  ΔV={((V-V0)/V0*100):+.6f}%")

avg_a = np.mean(times_a) * 1000

# ═══════════════════════════════════════════════════════════════
# MODE B: No-PLIC pipeline (C/OpenMP native overlay)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  MODE B: No-PLIC pipeline (C/OpenMP native overlay)")
print("=" * 70)

if not is_native_available():
    print("  *** C overlay not available — skipping Mode B ***")
    avg_b = None
    jit_b = None
else:
    F_b = jnp.array(F)  # fresh copy

    # Manual pipeline without PLIC: move → clamp → native overlay
    def step_no_plic(F_in):
        nh_ = grid.nh
        xv, yv, zv = lagrangian_move_faces_3d(u_face, v_face, w_face, grid, dt)
        xv = jnp.clip(xv, grid.x_range[0], grid.x_range[1])
        yv = jnp.clip(yv, grid.y_range[0], grid.y_range[1])
        zv = jnp.clip(zv, grid.z_range[0], grid.z_range[1])
        F_new = overlay_lagrangian_3d_native(
            F_in, xv, yv, zv, grid,
            nx_f=None, ny_f=None, nz_f=None, C_f=None,
            obs_mask=None)
        F_new = jnp.clip(F_new, 0.0, 1.0)
        nx_, ny_, nz_ = grid.nx, grid.ny, grid.nz
        F_out = jnp.zeros_like(F_in)
        F_out = F_out.at[nh_:nh_+nx_, nh_:nh_+ny_, nh_:nh_+nz_, 0].set(F_new)
        F_out = F_out.at[:nh_].set(F_out[nh_:nh_+1])
        F_out = F_out.at[-nh_:].set(F_out[-nh_-1:-nh_])
        F_out = F_out.at[:, :nh_].set(F_out[:, nh_:nh_+1])
        F_out = F_out.at[:, -nh_:].set(F_out[:, -nh_-1:-nh_])
        F_out = F_out.at[:, :, :nh_].set(F_out[:, :, nh_:nh_+1])
        F_out = F_out.at[:, :, -nh_:].set(F_out[:, :, -nh_-1:-nh_])
        return F_out

    print("  JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    _ = step_no_plic(F_b)
    _.block_until_ready()
    jit_b = time.time() - t0
    print(f"{jit_b:.1f}s")

    mem_b, _ = get_gpu_mem_mb()

    times_b = []
    for s in range(N_STEPS):
        t0 = time.time()
        F_b = step_no_plic(F_b)
        F_b.block_until_ready()
        times_b.append(time.time() - t0)
        V = float(np.sum(np.array(F_b[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
        print(f"    step {s+1}: {times_b[-1]*1000:.0f} ms  ΔV={((V-V0)/V0*100):+.6f}%")

    avg_b = np.mean(times_b) * 1000

# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  COMPARISON: 1M cells ({NX}×{NY}×{NZ})".format(NX=NX, NY=NY, NZ=NZ))
print("=" * 70)
print(f"  {'':30s} {'PLIC (JAX)':>15s}  {'No-PLIC (C)':>15s}  {'Speedup':>8s}")
print(f"  {'JIT compile':30s} {jit_a:>12.1f} s   "
      f"{jit_b:>12.1f} s   " if jit_b else "", end="")
if jit_b:
    print(f"{jit_a/jit_b:>6.1f}x")
else:
    print()
print(f"  {'Per-step avg':30s} {avg_a:>12.0f} ms  ", end="")
if avg_b:
    print(f"{avg_b:>12.0f} ms  {avg_a/avg_b:>6.0f}x")
else:
    print(f"{'N/A':>12s}")
if avg_b:
    print(f"  {'Throughput':30s} "
          f"{NX*NY*NZ/(avg_a/1000)/1e6:>10.2f} Mc/s  "
          f"{NX*NY*NZ/(avg_b/1000)/1e6:>10.2f} Mc/s")

if mem_used_0 is not None:
    print(f"\n  GPU VRAM: {mem_used_0} MB base → "
          f"{mem_a} MB (PLIC) / "
          f"{mem_b if 'mem_b' in dir() else '?'} MB (C)  "
          f"/ {mem_total} MB total")
print("=" * 70)
