#!/usr/bin/env python3
"""Profile each step of the Lagrangian VOF pipeline — GPU(JAX) vs CPU(native C).

Measures:
  Step 1: PLIC reconstruction (normals + intercepts)
  Step 2: Lagrangian face move + hex build
  Step 3: Overlay (the suspected bottleneck)
  Step 4: Halo + cleanup

Tests two grid sizes: 50x50x5 (small) and 100x100x5 (production).
"""
import sys, os, time
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
from jax_laseram.vof.lagrangian_3d.overlay_3d import overlay_lagrangian_3d
from jax_laseram.vof.lagrangian_3d.overlay_batched import overlay_lagrangian_3d_batched
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native, is_native_available,
)
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d


def setup_problem(NX, NZ):
    DX = 1.0 / NX
    DOMAIN_Z = NZ * DX
    grid = GridInfo(
        nx=NX, ny=NX, nz=NZ, nh=1,
        dx=DX, dy=DX, dz=DX,
        x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, DOMAIN_Z),
    )
    nh = grid.nh
    Ntx = NX + 2 * nh
    Nty = NX + 2 * nh
    Ntz = NZ + 2 * nh

    x = np.linspace(DX / 2, 1.0 - DX / 2, NX)
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X - 0.20) ** 2 + (Y - 0.20) ** 2)
    hd = np.sqrt(2) * DX / 2
    R = 0.05
    F_2d = np.where(dist + hd <= R, 1.0,
                    np.where(dist - hd >= R, 0.0,
                             np.clip((R - dist + hd) / (2 * hd), 0.0, 1.0)))

    F = np.zeros((Ntx, Nty, Ntz, 1))
    F[nh:nh+NX, nh:nh+NX, nh:nh+NZ, 0] = F_2d[:, :, None]
    F[:nh] = F[nh:nh+1]; F[-nh:] = F[-nh-1:-nh]
    F[:, :nh] = F[:, nh:nh+1]; F[:, -nh:] = F[:, -nh-1:-nh]
    F[:, :, :nh] = F[:, :, nh:nh+1]; F[:, :, -nh:] = F[:, :, -nh-1:-nh]
    F = jnp.array(F, dtype=jnp.float32)

    theta = np.radians(45)
    u = jnp.full((Ntx-1, Nty, Ntz), np.cos(theta), jnp.float32)
    v = jnp.full((Ntx, Nty-1, Ntz), np.sin(theta), jnp.float32)
    w = jnp.zeros((Ntx, Nty, Ntz-1), jnp.float32)

    dt = 0.45 * DX
    return grid, F, u, v, w, dt


def profile_steps(grid, F, u, v, w, dt, n_warmup=1, n_repeat=5, label=""):
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    N_cells = nx * ny * nz

    print(f"\n{'='*60}")
    print(f"  Profile: {label}  ({nx}x{ny}x{nz} = {N_cells:,} cells)")
    print(f"{'='*60}")

    # ── Step 1: PLIC Reconstruction ──
    def step1(F_):
        nx_f, ny_f, nz_f = compute_plic_normals_3d(F_, dx, dy, dz)
        raw_gx = jnp.zeros_like(F_)
        raw_gy = jnp.zeros_like(F_)
        raw_gz = jnp.zeros_like(F_)
        raw_gx = raw_gx.at[1:-1].set((F_[2:]-F_[:-2])/(2*dx))
        raw_gy = raw_gy.at[:,1:-1].set((F_[:,2:]-F_[:,:-2])/(2*dy))
        raw_gz = raw_gz.at[:,:,1:-1].set((F_[:,:,2:]-F_[:,:,:-2])/(2*dz))
        raw_grad = jnp.sqrt(raw_gx**2+raw_gy**2+raw_gz**2)
        mask = raw_grad > 0.5
        nx_f = jnp.where(mask, nx_f, 0.0)
        ny_f = jnp.where(mask, ny_f, 0.0)
        nz_f = jnp.where(mask, nz_f, 0.0)
        C_f = compute_intercept_C_3d(F_, nx_f, ny_f, nz_f, dx, dy, dz)
        return nx_f, ny_f, nz_f, C_f

    # Warmup
    for _ in range(n_warmup):
        nx_f, ny_f, nz_f, C_f = step1(F)
        jax.block_until_ready(C_f)

    times_s1 = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        nx_f, ny_f, nz_f, C_f = step1(F)
        jax.block_until_ready(C_f)
        times_s1.append(time.perf_counter() - t0)
    t_s1 = np.median(times_s1)
    print(f"  Step 1 (PLIC recon):   {t_s1*1000:8.1f} ms")

    # ── Step 2: Face move + hex build ──
    def step2(u_, v_, w_):
        xv, yv, zv = lagrangian_move_faces_3d(u_, v_, w_, grid, dt)
        xv = jnp.clip(xv, grid.x_range[0], grid.x_range[1])
        yv = jnp.clip(yv, grid.y_range[0], grid.y_range[1])
        zv = jnp.clip(zv, grid.z_range[0], grid.z_range[1])
        return xv, yv, zv

    for _ in range(n_warmup):
        xv, yv, zv = step2(u, v, w)
        jax.block_until_ready(zv)

    times_s2 = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        xv, yv, zv = step2(u, v, w)
        jax.block_until_ready(zv)
        times_s2.append(time.perf_counter() - t0)
    t_s2 = np.median(times_s2)
    print(f"  Step 2 (face move):    {t_s2*1000:8.1f} ms")

    # ── Step 3a: Overlay — JAX batched (GPU) ──
    def step3_jax(F_, xv_, yv_, zv_, nx_f_, ny_f_, nz_f_, C_f_):
        return overlay_lagrangian_3d_batched(
            F_, xv_, yv_, zv_, grid,
            nx_f=nx_f_, ny_f=ny_f_, nz_f=nz_f_, C_f=C_f_,
        )

    for _ in range(n_warmup):
        F_jax = step3_jax(F, xv, yv, zv, nx_f, ny_f, nz_f, C_f)
        jax.block_until_ready(F_jax)

    times_s3j = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        F_jax = step3_jax(F, xv, yv, zv, nx_f, ny_f, nz_f, C_f)
        jax.block_until_ready(F_jax)
        times_s3j.append(time.perf_counter() - t0)
    t_s3j = np.median(times_s3j)
    print(f"  Step 3a (overlay JAX): {t_s3j*1000:8.1f} ms  ← GPU PLIC overlay")

    # ── Step 3b: Overlay — Native C/OpenMP (CPU, no PLIC) ──
    if is_native_available():
        def step3_native(F_, xv_, yv_, zv_):
            return overlay_lagrangian_3d_native(F_, xv_, yv_, zv_, grid)

        for _ in range(n_warmup):
            F_nat = step3_native(F, xv, yv, zv)
            jax.block_until_ready(F_nat)

        times_s3n = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            F_nat = step3_native(F, xv, yv, zv)
            jax.block_until_ready(F_nat)
            times_s3n.append(time.perf_counter() - t0)
        t_s3n = np.median(times_s3n)
        print(f"  Step 3b (overlay C):   {t_s3n*1000:8.1f} ms  ← CPU native (no PLIC)")
        speedup = t_s3j / max(t_s3n, 1e-9)
        print(f"  Speedup (C vs JAX):    {speedup:.1f}x")
    else:
        t_s3n = None
        print(f"  Step 3b (overlay C):   N/A (libhexboxclip.so not available)")

    # ── Step 3c: Overlay — Native fallback to JAX batched (with PLIC) ──
    # This is the actual path used in advect_vof_lagrangian_3d when native
    # is available but PLIC is enabled — it falls back to batched JAX.
    # Already measured as Step 3a.

    # ── Full pipeline (end-to-end) ──
    def full_step(F_):
        return advect_vof_lagrangian_3d(F_, u, v, w, grid, dt)

    for _ in range(n_warmup):
        F_out = full_step(F)
        jax.block_until_ready(F_out)

    times_full = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        F_out = full_step(F)
        jax.block_until_ready(F_out)
        times_full.append(time.perf_counter() - t0)
    t_full = np.median(times_full)
    print(f"  Full pipeline:         {t_full*1000:8.1f} ms")

    # ── Summary ──
    other = t_full - t_s1 - t_s2 - t_s3j
    print(f"\n  Breakdown:")
    print(f"    PLIC recon:  {t_s1/t_full*100:5.1f}%  ({t_s1*1000:.1f} ms)")
    print(f"    Face move:   {t_s2/t_full*100:5.1f}%  ({t_s2*1000:.1f} ms)")
    print(f"    Overlay:     {t_s3j/t_full*100:5.1f}%  ({t_s3j*1000:.1f} ms)")
    print(f"    Other:       {other/t_full*100:5.1f}%  ({other*1000:.1f} ms)")
    if t_s3n is not None:
        t_full_native = t_s1 + t_s2 + t_s3n + max(other, 0)
        print(f"\n  If overlay used native C: {t_full_native*1000:.1f} ms ({t_full/t_full_native:.1f}x faster)")

    return {
        "grid": f"{nx}x{ny}x{nz}",
        "n_cells": N_cells,
        "step1_ms": t_s1 * 1000,
        "step2_ms": t_s2 * 1000,
        "step3_jax_ms": t_s3j * 1000,
        "step3_native_ms": t_s3n * 1000 if t_s3n else None,
        "full_ms": t_full * 1000,
    }


print("=" * 60)
print("  Lagrangian VOF Pipeline Profiling")
print(f"  Device: {jax.devices()}")
print(f"  Native C available: {is_native_available()}")
print("=" * 60)

results = {}
for NX, NZ in [(50, 3), (100, 5)]:
    grid, F, u, v, w, dt = setup_problem(NX, NZ)
    r = profile_steps(grid, F, u, v, w, dt,
                      label=f"{NX}x{NX}x{NZ}",
                      n_warmup=2, n_repeat=5)
    results[f"{NX}x{NX}x{NZ}"] = r

print("\n" + "=" * 60)
print("  SUMMARY TABLE")
print("=" * 60)
print(f"  {'Grid':<12} {'Cells':>8} {'PLIC':>8} {'Move':>8} {'Overlay':>10} {'Native':>10} {'Full':>10}")
for k, v in results.items():
    nat_str = f"{v['step3_native_ms']:.1f}" if v['step3_native_ms'] else "N/A"
    print(f"  {k:<12} {v['n_cells']:>8,} {v['step1_ms']:>7.1f}ms {v['step2_ms']:>7.1f}ms "
          f"{v['step3_jax_ms']:>9.1f}ms {nat_str:>9}ms {v['full_ms']:>9.1f}ms")
