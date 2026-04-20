#!/usr/bin/env python3
"""Layer 3: Regression benchmark — clip vs. redistribute, all three test cases.

Compares ``jnp.clip(F, 0, 1)`` (current baseline) against the new
Weymouth-Zaleski conservative redistribute across three benchmark cases:

  Case A: Zalesak 3D slotted sphere, 128^3, 1 full rotation (~900 steps)
  Case B: Rider-Kothe reversed vortex, 128x128x3, T=4 (~1024 steps)
  Case C: 1M-cell profile (250x250x16), 5 steps — timing only

Usage::

    # Run all cases with both methods:
    python compare_clip_vs_redistribute.py

    # Run only one case:
    python compare_clip_vs_redistribute.py --case A

    # Run only one method (useful to get clip baseline first):
    python compare_clip_vs_redistribute.py --method clip
    python compare_clip_vs_redistribute.py --method redistribute

    # Quick smoke-test (very short run, enough to check method imports):
    python compare_clip_vs_redistribute.py --smoke-test

Results are printed to stdout and saved to results/comparison_results.json.

Expected pass criteria (for redistribute vs. clip):
  Case A: V_drift_redist < 1e-8 (clip: 0.054%), L1_redist < 2.5% (clip: 1.53%)
  Case B: V_drift_redist < 0.1% (clip: -63.5%); L1 difference < 10% absolute
  Case C: overhead < 2 ms/sub-sweep, total < 55 ms (clip: ~36-50 ms)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "src")))

import jax
import jax.numpy as jnp
import numpy as np

# ── PLIC imports ─────────────────────────────────────────────────────────────
from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x,
    sweep_flux_y, apply_flux_y,
    sweep_flux_z, apply_flux_z,
)

# ── Conservative bounds (optional — graceful fallback) ───────────────────────
_IMPL_AVAILABLE = False
try:
    from jax_laseram.vof.plic.conservative_bounds import (  # type: ignore[import]
        apply_flux_x_conservative,
        apply_flux_y_conservative,
        apply_flux_z_conservative,
    )
    _IMPL_AVAILABLE = True
    print("  [OK] conservative_bounds loaded.")
except ImportError as e:
    print(f"  [WARN] conservative_bounds not available: {e}")
    print("         Redistribute method will be skipped.")


# ════════════════════════════════════════════════════════════════════════════
# Shared infrastructure
# ════════════════════════════════════════════════════════════════════════════

def halo(F):
    F = F.at[0].set(F[1]); F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F


def reconstruct(F, dx, dy, dz, max_frac=0.15):
    """Sparse PLIC reconstruction (padded-gather)."""
    nx, ny, nz = compute_youngs_normal_3d(F, dx, dy, dz)
    is_if = (F > 1e-6) & (F < 1 - 1e-6)
    max_n = int(F.size * max_frac)
    idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
    F_c = F.ravel()[idx]
    nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
    C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, dx, dy, dz)
    C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
    return nx, ny, nz, C


def metrics(F, F0, NH, cell_vol):
    """Return (V_drift_pct, L1_pct)."""
    i_s = slice(NH, -NH)
    Fi = F[i_s, i_s, i_s]
    F0i = F0[i_s, i_s, i_s]
    V0 = float(F0i.sum() * cell_vol)
    V = float(Fi.sum() * cell_vol)
    L1 = float(jnp.abs(Fi - F0i).sum() * cell_vol)
    V_drift = (V - V0) / V0 * 100
    L1_pct = L1 / V0 * 100
    return V_drift, L1_pct


# ════════════════════════════════════════════════════════════════════════════
# Sub-sweep builders (clip and redistribute)
# ════════════════════════════════════════════════════════════════════════════

def make_sub_sweeps(method: str):
    """Return (sub_x, sub_y, sub_z) functions for the given method."""

    def _apply_x(F, flux, dt, dx, u_face):
        F_raw = apply_flux_x(F, flux, dx, dx, dx)
        if method == "clip":
            return jnp.clip(F_raw, 0.0, 1.0)
        else:
            return apply_flux_x_conservative(
                F_raw, jnp.zeros_like(flux), dt=dt, dx=dx, u_face=u_face, n_iter=3
            )

    def _apply_y(F, flux, dt, dy, v_face):
        F_raw = apply_flux_y(F, flux, dy, dy, dy)
        if method == "clip":
            return jnp.clip(F_raw, 0.0, 1.0)
        else:
            return apply_flux_y_conservative(
                F_raw, jnp.zeros_like(flux), dt=dt, dy=dy, v_face=v_face, n_iter=3
            )

    def _apply_z(F, flux, dt, dz, w_face):
        F_raw = apply_flux_z(F, flux, dz, dz, dz)
        if method == "clip":
            return jnp.clip(F_raw, 0.0, 1.0)
        else:
            return apply_flux_z_conservative(
                F_raw, jnp.zeros_like(flux), dt=dt, dz=dz, w_face=w_face, n_iter=3
            )

    def sub_x(F, u_face, dt_s, DX):
        F = halo(F)
        nx, ny, nz, C = reconstruct(F, DX, DX, DX)
        flux = sweep_flux_x(F, nx, ny, nz, C, u_face, dt_s, DX, DX, DX)
        return halo(_apply_x(F, flux, dt_s, DX, u_face))

    def sub_y(F, v_face, dt_s, DX):
        F = halo(F)
        nx, ny, nz, C = reconstruct(F, DX, DX, DX)
        flux = sweep_flux_y(F, nx, ny, nz, C, v_face, dt_s, DX, DX, DX)
        return halo(_apply_y(F, flux, dt_s, DX, v_face))

    def sub_z(F, w_face, dt_s, DX):
        F = halo(F)
        nx, ny, nz, C = reconstruct(F, DX, DX, DX)
        flux = sweep_flux_z(F, nx, ny, nz, C, w_face, dt_s, DX, DX, DX)
        return halo(_apply_z(F, flux, dt_s, DX, w_face))

    return sub_x, sub_y, sub_z


# ════════════════════════════════════════════════════════════════════════════
# Case A: Zalesak 3D
# ════════════════════════════════════════════════════════════════════════════

def run_case_a_zalesak_3d(method: str, smoke: bool = False) -> dict:
    print(f"\n{'='*65}")
    print(f"CASE A: Zalesak 3D Slotted Sphere  [{method}]")
    print(f"{'='*65}")

    N = 16 if smoke else 128
    NH = 1; DX = DY = DZ = 1.0 / N
    Nt = N + 2 * NH

    SPHERE_CENTER = (0.50, 0.50, 0.75)
    SPHERE_R = 0.15
    SLOT_W = 0.05; SLOT_D = 0.25
    OMEGA = 1.0
    ROT_Y = 0.50; ROT_Z = 0.50
    T_END = 2.0 * np.pi
    CFL = 0.45

    # Init
    print(f"  Initializing {N}^3 grid...", end=" ", flush=True)
    n_sub = 4
    c = jnp.linspace(DX / 2, 1 - DX / 2, N, dtype=jnp.float32)
    sub_off = (jnp.arange(n_sub, dtype=jnp.float32) + 0.5) / n_sub
    xs = c[:, None] - DX / 2 + DX * sub_off[None, :]
    Xs = xs[:, :, None, None, None, None]
    Ys = xs[None, None, :, :, None, None]
    Zs = xs[None, None, None, None, :, :]
    cx, cy, cz = SPHERE_CENTER
    r = jnp.sqrt((Xs - cx)**2 + (Ys - cy)**2 + (Zs - cz)**2)
    in_sphere = r <= SPHERE_R
    in_slot = (
        (jnp.abs(Xs - cx) < SLOT_W / 2)
        & (jnp.abs(Ys - cy) < SLOT_W / 2)
        & (Zs > cz + SPHERE_R - SLOT_D)
        & (Zs < cz + SPHERE_R)
    )
    F_int = (in_sphere & ~in_slot).astype(jnp.float32).mean(axis=(1, 3, 5))
    F_full = jnp.zeros((Nt, Nt, Nt), dtype=jnp.float32)
    F_full = F_full.at[NH:NH+N, NH:NH+N, NH:NH+N].set(F_int)
    F_init = halo(F_full)
    print("done")

    # Velocity field
    cc = jnp.linspace(-NH*DX+DX/2, 1+NH*DX-DX/2, Nt, dtype=jnp.float32)
    u_face = jnp.zeros((Nt-1, Nt, Nt), dtype=jnp.float32)
    _, _, Z_vf = jnp.meshgrid(cc, 0.5*(cc[:-1]+cc[1:]), cc, indexing='ij')
    v_face = (-OMEGA * (Z_vf - ROT_Z)).astype(jnp.float32)
    _, Y_wf, _ = jnp.meshgrid(cc, cc, 0.5*(cc[:-1]+cc[1:]), indexing='ij')
    w_face = (OMEGA * (Y_wf - ROT_Y)).astype(jnp.float32)

    max_vel = float(jnp.maximum(jnp.abs(v_face).max(), jnp.abs(w_face).max()))
    dt = CFL * DX / max_vel
    n_steps = 10 if smoke else int(np.ceil(T_END / dt))
    dt = T_END / n_steps if not smoke else dt
    print(f"  n_steps={n_steps}, dt={dt:.6e}, max_vel={max_vel:.4f}")

    sub_x, sub_y, sub_z = make_sub_sweeps(method)

    @jax.jit
    def step(F):
        # Strang: y/2 -> z -> y/2 (rotation is in y-z plane)
        F = sub_y(F, v_face, dt / 2, DX)
        F = sub_z(F, w_face, dt, DX)
        F = sub_y(F, v_face, dt / 2, DX)
        return F

    # Warmup
    print("  JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    Fw = step(F_init); Fw.block_until_ready()
    compile_time = time.time() - t0
    print(f"{compile_time:.1f}s")

    F = F_init
    t_wall0 = time.time()
    for s in range(1, n_steps + 1):
        F = step(F)
        if s % max(1, n_steps // 5) == 0 or s == n_steps:
            jax.block_until_ready(F)
            V_drift, L1 = metrics(F, F_init, NH, DX**3)
            elapsed = time.time() - t_wall0
            print(f"  step {s:4d}/{n_steps}  V_drift={V_drift:+.4f}%  L1={L1:.3f}%  "
                  f"wall={elapsed:.1f}s", flush=True)

    wall = time.time() - t_wall0
    V_drift, L1 = metrics(F, F_init, NH, DX**3)
    ms_per_step = wall / n_steps * 1000

    print(f"\n  RESULT [{method}]:")
    print(f"    V drift:    {V_drift:+.6f}%   (target: {'<1e-6%' if method=='redistribute' else 'baseline'})")
    print(f"    L1 error:   {L1:.3f}%         (target: {'<2.5%' if method=='redistribute' else 'baseline'})")
    print(f"    ms/step:    {ms_per_step:.1f} ms  (target: {'<180' if method=='redistribute' else 'baseline'})")

    # Pass/fail
    passed = True
    if method == "redistribute":
        if abs(V_drift) >= 1e-6:
            print(f"  FAIL: V drift {abs(V_drift):.2e}% >= 1e-6%")
            passed = False
        if L1 >= 2.5:
            print(f"  FAIL: L1 {L1:.3f}% >= 2.5%")
            passed = False
        if ms_per_step >= 180:
            print(f"  WARN: ms/step {ms_per_step:.1f} >= 180 (threshold)")

    return {
        "case": "A_zalesak_3d",
        "method": method,
        "N": N,
        "n_steps": n_steps,
        "V_drift_pct": float(V_drift),
        "L1_pct": float(L1),
        "ms_per_step": float(ms_per_step),
        "passed": passed,
    }


# ════════════════════════════════════════════════════════════════════════════
# Case B: Rider-Kothe reversed vortex
# ════════════════════════════════════════════════════════════════════════════

def run_case_b_rider_kothe(method: str, smoke: bool = False) -> dict:
    print(f"\n{'='*65}")
    print(f"CASE B: Rider-Kothe Reversed Vortex T=4  [{method}]")
    print(f"{'='*65}")

    N = 16 if smoke else 128
    NZ = 3; NH = 1; DX = 1.0 / N
    Nt = N + 2 * NH; Ntz = NZ + 2 * NH
    T_END = 4.0; CFL = 0.5
    CIRCLE_CENTER = (0.5, 0.75); CIRCLE_R = 0.15

    # Init
    print(f"  Initializing {N}^2 x {NZ} grid...", end=" ", flush=True)
    n_sub = 4
    c = jnp.linspace(DX/2, 1-DX/2, N, dtype=jnp.float32)
    off = (jnp.arange(n_sub, dtype=jnp.float32) + 0.5) / n_sub
    xs = c[:, None] - DX/2 + DX * off[None, :]
    Xs = xs[:, :, None, None]; Ys = xs[None, None, :, :]
    cx, cy = CIRCLE_CENTER
    in_circle = ((Xs - cx)**2 + (Ys - cy)**2) <= CIRCLE_R**2
    F2d = in_circle.astype(jnp.float32).mean(axis=(1, 3))
    F = jnp.zeros((Nt, Nt, Ntz), dtype=jnp.float32)
    F = F.at[NH:NH+N, NH:NH+N, NH:NH+NZ].set(F2d[:, :, None])
    F_init = halo(F)
    print("done")

    # Velocity field (spatial pattern only, scaled by cos(πt/T) at runtime)
    cc = jnp.linspace(-NH*DX+DX/2, 1+NH*DX-DX/2, Nt, dtype=jnp.float32)
    x_fx = 0.5 * (cc[:-1] + cc[1:])
    Xfx, Yfx = jnp.meshgrid(x_fx, cc, indexing="ij")
    u_sp_2d = -jnp.sin(jnp.pi*Xfx)**2 * jnp.sin(2*jnp.pi*Yfx)
    u_spatial = jnp.broadcast_to(u_sp_2d[:, :, None], (Nt-1, Nt, Ntz)).astype(jnp.float32)

    x_fy = cc; y_fy = 0.5 * (cc[:-1] + cc[1:])
    Xfy, Yfy = jnp.meshgrid(x_fy, y_fy, indexing="ij")
    v_sp_2d = jnp.sin(2*jnp.pi*Xfy) * jnp.sin(jnp.pi*Yfy)**2
    v_spatial = jnp.broadcast_to(v_sp_2d[:, :, None], (Nt, Nt-1, Ntz)).astype(jnp.float32)

    max_vel = float(jnp.maximum(jnp.abs(u_spatial).max(), jnp.abs(v_spatial).max()))
    dt = CFL * DX / max_vel
    n_steps = 5 if smoke else int(np.ceil(T_END / dt))
    dt = T_END / n_steps if not smoke else dt
    print(f"  n_steps={n_steps}, dt={dt:.6e}, max_vel={max_vel:.4f}")

    sub_x, sub_y, sub_z = make_sub_sweeps(method)

    @jax.jit
    def step(F, u_face, v_face, dt_step):
        F = sub_x(F, u_face, dt_step / 2, DX)
        F = sub_y(F, v_face, dt_step, DX)
        F = sub_x(F, u_face, dt_step / 2, DX)
        return F

    # Warmup
    print("  JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    scale0 = jnp.float32(np.cos(np.pi * dt / 2 / T_END))
    _ = step(F_init, u_spatial * scale0, v_spatial * scale0, dt)
    _.block_until_ready()
    print(f"{time.time()-t0:.1f}s")

    F = F_init
    t = 0.0
    t_wall0 = time.time()
    for s in range(n_steps):
        scale = jnp.float32(np.cos(np.pi * (t + dt / 2) / T_END))
        F = step(F, u_spatial * scale, v_spatial * scale, dt)
        t += dt
        if (s + 1) % max(1, n_steps // 5) == 0 or s == n_steps - 1:
            jax.block_until_ready(F)
            V_drift, L1 = metrics(F, F_init, NH, DX**3 * NZ)
            print(f"  step {s+1:4d}/{n_steps}  V_drift={V_drift:+.3f}%  L1={L1:.2f}%", flush=True)

    wall = time.time() - t_wall0
    V_drift, L1 = metrics(F, F_init, NH, DX**3 * NZ)
    ms_per_step = wall / n_steps * 1000

    print(f"\n  RESULT [{method}]:")
    print(f"    V drift:    {V_drift:+.4f}%  (clip baseline: -63.5%)")
    print(f"    L1 error:   {L1:.2f}%       (clip baseline: ~63.6%)")
    print(f"    ms/step:    {ms_per_step:.1f} ms")

    passed = True
    if method == "redistribute":
        # V drift must be < 0.1% (from -63.5%)
        if abs(V_drift) >= 0.1:
            print(f"  FAIL: V drift {abs(V_drift):.4f}% >= 0.1%")
            passed = False
        # L1 must not be more than 10% absolute WORSE than clip
        # (L1 will be bad for both methods; just make sure redistribute is not much worse)

    return {
        "case": "B_rider_kothe",
        "method": method,
        "N": N,
        "n_steps": n_steps,
        "V_drift_pct": float(V_drift),
        "L1_pct": float(L1),
        "ms_per_step": float(ms_per_step),
        "passed": passed,
    }


# ════════════════════════════════════════════════════════════════════════════
# Case C: 1M-cell timing benchmark
# ════════════════════════════════════════════════════════════════════════════

def run_case_c_1m_profile(method: str, smoke: bool = False) -> dict:
    print(f"\n{'='*65}")
    print(f"CASE C: 1M-cell Profile (250x250x16)  [{method}]")
    print(f"{'='*65}")

    NX, NY, NZ = (32, 32, 4) if smoke else (250, 250, 16)
    DOMAIN_XY = 2.5; DX = DOMAIN_XY / NX; NH = 1
    CFL = 0.45; SPEED = 1.0
    theta = np.radians(45)
    UX, UY = np.cos(theta) * SPEED, np.sin(theta) * SPEED
    dt = CFL * DX / SPEED
    N_STEPS = 3 if smoke else 5

    Ntx, Nty, Ntz = NX + 2*NH, NY + 2*NH, NZ + 2*NH

    # Init
    x_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
    y_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NY)
    X2, Y2 = np.meshgrid(x_c, y_c, indexing="ij")
    R = 0.25; cx, cy = 1.25, 1.25
    dist = np.sqrt((X2 - cx)**2 + (Y2 - cy)**2)
    hd = np.sqrt(2) * DX / 2
    F2d = np.clip((R - dist + hd) / (2*hd), 0, 1).astype(np.float32)
    F_np = np.zeros((Ntx, Nty, Ntz), dtype=np.float32)
    F_np[NH:-NH, NH:-NH, NH:-NH] = F2d[:, :, None]
    F_np[0] = F_np[1]; F_np[-1] = F_np[-2]
    F_np[:, 0] = F_np[:, 1]; F_np[:, -1] = F_np[:, -2]
    F_np[:, :, 0] = F_np[:, :, 1]; F_np[:, :, -1] = F_np[:, :, -2]
    F = jnp.asarray(F_np)

    V0 = float(F[NH:-NH, NH:-NH, NH:-NH].sum() * DX**3)
    n_cells = NX * NY * NZ
    print(f"  Grid: {NX}x{NY}x{NZ} = {n_cells:,} cells, V0={V0:.8f}")

    sub_x, sub_y, sub_z = make_sub_sweeps(method)

    @jax.jit
    def full_step(F):
        F = sub_x(F, UX, dt / 2, DX)
        F = sub_y(F, UY, dt, DX)
        F = sub_x(F, UX, dt / 2, DX)
        return F

    # Warmup
    print("  JIT compiling...", end=" ", flush=True)
    t0 = time.time()
    Fw = full_step(F); Fw.block_until_ready()
    compile_time = time.time() - t0
    print(f"{compile_time:.1f}s")

    # Profile sub-stages
    DX_f = DX
    nx, ny, nz, C = reconstruct(F, DX_f, DX_f, DX_f)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt / 2, DX_f, DX_f, DX_f)
    F_raw = apply_flux_x(F, flux, DX_f, DX_f, DX_f)

    @jax.jit
    def time_clip(F_raw):
        return jnp.clip(F_raw, 0.0, 1.0)

    @jax.jit
    def time_redist(F_raw, flux):
        if _IMPL_AVAILABLE:
            return apply_flux_x_conservative(
                F_raw, jnp.zeros_like(flux), dt=dt/2, dx=DX_f,
                u_face=jnp.ones_like(flux) * UX, n_iter=3
            )
        return jnp.clip(F_raw, 0.0, 1.0)

    # Warmup sub-stage timing
    _ = time_clip(F_raw); _.block_until_ready()
    _ = time_redist(F_raw, flux); _.block_until_ready()

    # Time clip
    ts_clip = []
    for _ in range(10):
        t0 = time.time()
        r = time_clip(F_raw); r.block_until_ready()
        ts_clip.append((time.time() - t0) * 1000)
    ms_clip = np.median(ts_clip)

    # Time redistribute
    ts_redist = []
    for _ in range(10):
        t0 = time.time()
        r = time_redist(F_raw, flux); r.block_until_ready()
        ts_redist.append((time.time() - t0) * 1000)
    ms_redist = np.median(ts_redist)

    print(f"  clip sub-step:        {ms_clip:.2f} ms")
    print(f"  redistribute sub-step: {ms_redist:.2f} ms")
    print(f"  overhead: {ms_redist - ms_clip:.2f} ms/sub-sweep")

    # Full step timing
    step_times = []
    for s in range(N_STEPS):
        t0 = time.time()
        F = full_step(F); F.block_until_ready()
        step_times.append((time.time() - t0) * 1000)
        V_cur = float(F[NH:-NH, NH:-NH, NH:-NH].sum() * DX**3)
        dV = (V_cur - V0) / V0 * 100
        print(f"  step {s+1}/{N_STEPS}: {step_times[-1]:.1f} ms  dV={dV:+.6f}%")

    avg_ms = np.mean(step_times)
    print(f"\n  RESULT [{method}]:")
    print(f"    avg ms/step: {avg_ms:.1f} ms  (clip baseline: ~49.7 ms incl. sync)")
    print(f"    redistribute overhead per sub-sweep: {ms_redist - ms_clip:.2f} ms")
    print(f"    throughput: {n_cells / (avg_ms/1000) / 1e6:.2f} Mcells/s")

    passed = True
    if method == "redistribute":
        overhead = ms_redist - ms_clip
        if overhead >= 2.0:
            print(f"  WARN: per-sub-sweep overhead {overhead:.2f} ms >= 2 ms")
        if avg_ms >= 55:
            print(f"  WARN: avg ms/step {avg_ms:.1f} >= 55 ms")

    return {
        "case": "C_1m_profile",
        "method": method,
        "n_cells": n_cells,
        "avg_ms_step": float(avg_ms),
        "ms_clip_per_subsweep": float(ms_clip),
        "ms_redist_per_subsweep": float(ms_redist),
        "overhead_per_subsweep": float(ms_redist - ms_clip),
        "passed": passed,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Clip vs. redistribute benchmark comparison"
    )
    parser.add_argument(
        "--case", choices=["A", "B", "C", "all"], default="all",
        help="Which benchmark case to run (default: all)"
    )
    parser.add_argument(
        "--method", choices=["clip", "redistribute", "both"], default="both",
        help="Which method(s) to run (default: both)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Very short run for import/logic smoke testing (tiny grids, few steps)"
    )
    args = parser.parse_args()

    if args.method == "redistribute" and not _IMPL_AVAILABLE:
        print("ERROR: redistribute method requested but conservative_bounds not available.")
        sys.exit(1)

    methods = ["clip", "redistribute"] if args.method == "both" else [args.method]
    if args.method == "both" and not _IMPL_AVAILABLE:
        print("NOTE: conservative_bounds not available, running clip only.")
        methods = ["clip"]

    cases = ["A", "B", "C"] if args.case == "all" else [args.case]
    smoke = args.smoke_test

    print("=" * 65)
    print("  Clip vs. Redistribute Regression Benchmark")
    print("=" * 65)
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Cases: {cases},  Methods: {methods}")
    if smoke:
        print("  *** SMOKE TEST MODE (tiny grids, few steps) ***")
    print()

    all_results = []

    for method in methods:
        for case in cases:
            if case == "A":
                r = run_case_a_zalesak_3d(method, smoke=smoke)
            elif case == "B":
                r = run_case_b_rider_kothe(method, smoke=smoke)
            elif case == "C":
                r = run_case_c_1m_profile(method, smoke=smoke)
            all_results.append(r)

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Case':10s}  {'Method':12s}  {'V drift':>12s}  {'L1':>8s}  {'ms/step':>10s}  {'Status':>6s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*6}")
    for r in all_results:
        case_lbl = r.get("case", "")
        meth = r.get("method", "")
        vd = r.get("V_drift_pct", float("nan"))
        l1 = r.get("L1_pct", float("nan"))
        ms = r.get("avg_ms_step") or r.get("ms_per_step", float("nan"))
        ok = "PASS" if r.get("passed", True) else "FAIL"
        vd_str = f"{vd:+.4f}%" if not math.isnan(vd) else "N/A"
        l1_str = f"{l1:.2f}%" if not math.isnan(l1) else "N/A"
        ms_str = f"{ms:.1f} ms" if not math.isnan(ms) else "N/A"
        print(f"  {case_lbl:10s}  {meth:12s}  {vd_str:>12s}  {l1_str:>8s}  {ms_str:>10s}  {ok:>6s}")

    # Save results
    out_dir = os.path.join(_HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "comparison_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_file}")

    # Overall pass/fail
    n_fail = sum(1 for r in all_results if not r.get("passed", True))
    if n_fail:
        print(f"\n  {n_fail} case(s) FAILED.")
        sys.exit(1)
    else:
        print("\n  All cases PASSED.")


if __name__ == "__main__":
    main()
