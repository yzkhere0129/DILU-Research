#!/usr/bin/env python3
"""Profile Lagrangian VOF pipeline: Pure JAX (GPU) vs Hybrid (GPU+CPU).

Measures per-step time for each pipeline stage, GPU memory, and total throughput.
Two separate tables for 50x50x3 and 100x100x5 grids.
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import (
    compute_plic_normals_3d, compute_intercept_C_3d,
)
from jax_laseram.vof.lagrangian_3d.move_3d import (
    lagrangian_move_faces_3d, build_deformed_hexahedra,
)
from jax_laseram.vof.lagrangian_3d.overlay_batched import overlay_lagrangian_3d_batched
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native, is_native_available,
)
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d


def get_gpu_mem_mb():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except Exception:
        return 0


def setup(NX, NZ):
    DX = 1.0 / NX
    grid = GridInfo(nx=NX, ny=NX, nz=NZ, nh=1, dx=DX, dy=DX, dz=DX,
                    x_range=(0., 1.), y_range=(0., 1.), z_range=(0., NZ*DX))
    nh = 1; Ntx = NX+2; Nty = NX+2; Ntz = NZ+2
    x = np.linspace(DX/2, 1-DX/2, NX); y = x
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X-0.20)**2 + (Y-0.20)**2)
    hd = np.sqrt(2)*DX/2; R = 0.05
    F_2d = np.where(dist+hd<=R, 1., np.where(dist-hd>=R, 0., np.clip((R-dist+hd)/(2*hd),0,1)))
    F = np.zeros((Ntx,Nty,Ntz,1)); F[1:-1,1:-1,1:-1,0] = F_2d[:,:,None]
    F[:1]=F[1:2]; F[-1:]=F[-2:-1]; F[:,:1]=F[:,1:2]; F[:,-1:]=F[:,-2:-1]
    F[:,:,:1]=F[:,:,1:2]; F[:,:,-1:]=F[:,:,-2:-1]
    F = jnp.array(F, jnp.float32)
    th = np.radians(45)
    u = jnp.full((Ntx-1,Nty,Ntz), np.cos(th), jnp.float32)
    v = jnp.full((Ntx,Nty-1,Ntz), np.sin(th), jnp.float32)
    w = jnp.zeros((Ntx,Nty,Ntz-1), jnp.float32)
    return grid, F, u, v, w, 0.45*DX


def time_fn(fn, n_warmup=2, n_repeat=5):
    for _ in range(n_warmup):
        r = fn(); jax.block_until_ready(r)
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        r = fn(); jax.block_until_ready(r)
        ts.append(time.perf_counter() - t0)
    return np.median(ts)


def profile_grid(NX, NZ):
    grid, F, u, v, w, dt = setup(NX, NZ)
    nh = 1; dx = grid.dx; dy = grid.dy; dz = grid.dz
    N = NX * NX * NZ

    mem_before = get_gpu_mem_mb()

    # Step 1: PLIC
    def do_plic():
        nx_f, ny_f, nz_f = compute_plic_normals_3d(F, dx, dy, dz)
        gx = jnp.zeros_like(F); gy = jnp.zeros_like(F); gz = jnp.zeros_like(F)
        gx = gx.at[1:-1].set((F[2:]-F[:-2])/(2*dx))
        gy = gy.at[:,1:-1].set((F[:,2:]-F[:,:-2])/(2*dy))
        gz = gz.at[:,:,1:-1].set((F[:,:,2:]-F[:,:,:-2])/(2*dz))
        grad = jnp.sqrt(gx**2+gy**2+gz**2); m = grad > 0.5
        nx_f=jnp.where(m,nx_f,0.); ny_f=jnp.where(m,ny_f,0.); nz_f=jnp.where(m,nz_f,0.)
        C_f = compute_intercept_C_3d(F, nx_f, ny_f, nz_f, dx, dy, dz)
        return nx_f, ny_f, nz_f, C_f
    t_plic = time_fn(do_plic)
    nx_f, ny_f, nz_f, C_f = do_plic(); jax.block_until_ready(C_f)

    # Step 2: Move
    def do_move():
        xv, yv, zv = lagrangian_move_faces_3d(u, v, w, grid, dt)
        return (jnp.clip(xv, 0, 1), jnp.clip(yv, 0, 1), jnp.clip(zv, 0, NZ*dx))
    t_move = time_fn(do_move)
    xv, yv, zv = do_move(); jax.block_until_ready(zv)

    # Step 3 GPU: JAX batched overlay WITH PLIC
    def do_overlay_gpu():
        return overlay_lagrangian_3d_batched(F, xv, yv, zv, grid,
                                             nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f)
    t_overlay_gpu = time_fn(do_overlay_gpu)

    mem_gpu_peak = get_gpu_mem_mb()

    # Step 3 CPU: Native C overlay (no PLIC — geometric hex∩box only)
    t_overlay_cpu = None
    if is_native_available():
        def do_overlay_cpu():
            return overlay_lagrangian_3d_native(F, xv, yv, zv, grid)
        t_overlay_cpu = time_fn(do_overlay_cpu)

    # Full pipeline (actual production path)
    def do_full():
        return advect_vof_lagrangian_3d(F, u, v, w, grid, dt)
    t_full = time_fn(do_full)

    mem_after = get_gpu_mem_mb()

    return {
        "N": N,
        "t_plic": t_plic,
        "t_move": t_move,
        "t_overlay_gpu": t_overlay_gpu,
        "t_overlay_cpu": t_overlay_cpu,
        "t_full": t_full,
        "mem_peak_mb": mem_gpu_peak,
        "mem_base_mb": mem_before,
    }


def print_table(label, NX, NZ, r):
    N = r["N"]
    t_gpu_total = r["t_full"]
    t_cpu_overlay = r["t_overlay_cpu"]
    t_cpu_total = r["t_plic"] + r["t_move"] + (t_cpu_overlay or 0) + max(t_gpu_total - r["t_plic"] - r["t_move"] - r["t_overlay_gpu"], 0)

    print(f"\n{'─'*72}")
    print(f"  {label}: {NX}×{NX}×{NZ} = {N:,} cells")
    print(f"  GPU: {jax.devices()[0]}  |  VRAM: {r['mem_peak_mb']} MB peak")
    print(f"{'─'*72}")
    print(f"  {'Step':<25} {'Pure JAX (GPU)':>15} {'Hybrid (GPU+CPU)':>17} {'Speedup':>8}")
    print(f"  {'─'*25} {'─'*15} {'─'*17} {'─'*8}")

    # PLIC: same for both (always GPU)
    print(f"  {'1. PLIC reconstruction':<25} {r['t_plic']*1000:>12.1f} ms {r['t_plic']*1000:>14.1f} ms {'1.0x':>8}")
    # Move: same for both (always GPU)
    print(f"  {'2. Face move + hex':<25} {r['t_move']*1000:>12.1f} ms {r['t_move']*1000:>14.1f} ms {'1.0x':>8}")
    # Overlay: GPU vs CPU
    cpu_str = f"{t_cpu_overlay*1000:.1f} ms" if t_cpu_overlay else "N/A"
    sp_str = f"{r['t_overlay_gpu']/t_cpu_overlay:.0f}x" if t_cpu_overlay else "N/A"
    print(f"  {'3. Overlay (bottleneck)':<25} {r['t_overlay_gpu']*1000:>12.1f} ms {cpu_str:>14} {sp_str:>8}")
    # Other
    other = max(t_gpu_total - r["t_plic"] - r["t_move"] - r["t_overlay_gpu"], 0)
    print(f"  {'4. Halo + cleanup':<25} {other*1000:>12.1f} ms {other*1000:>14.1f} ms {'1.0x':>8}")
    print(f"  {'─'*25} {'─'*15} {'─'*17} {'─'*8}")
    print(f"  {'TOTAL per step':<25} {t_gpu_total*1000:>12.1f} ms {t_cpu_total*1000:>14.1f} ms {t_gpu_total/t_cpu_total:>7.1f}x")
    print(f"  {'Steps/minute':<25} {60/t_gpu_total:>12.1f}    {60/t_cpu_total:>14.1f}")

    # VRAM
    print(f"\n  GPU VRAM: {r['mem_base_mb']} MB base → {r['mem_peak_mb']} MB peak "
          f"(+{r['mem_peak_mb']-r['mem_base_mb']} MB)")


# ── Main ──
print("=" * 72)
print("  Lagrangian VOF Pipeline: Pure JAX (GPU) vs Hybrid (GPU+CPU)")
print(f"  Device: {jax.devices()}")
print(f"  Native C/OpenMP: {'Available' if is_native_available() else 'NOT available'}")
print("=" * 72)

for NX, NZ, label in [(50, 3, "Small grid"), (100, 5, "Production grid")]:
    r = profile_grid(NX, NZ)
    print_table(label, NX, NZ, r)
