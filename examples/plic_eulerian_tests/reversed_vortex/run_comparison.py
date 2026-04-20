#!/usr/bin/env python3
"""Rider-Kothe (1998) reversed single vortex — Eulerian PLIC vs Lagrangian VOF.

Test case:
  Initial:  circle at (0.5, 0.75), R = 0.15
  Velocity: u = -sin²(πx) sin(2πy) cos(πt/T)
            v =  sin(2πx) sin²(πy) cos(πt/T)
  Period:   T = 2  (at t=T/2 velocity is zero, max deformation)
  At t=T:   velocity has reversed, fluid should return to the initial circle
  Metric:   L1 error = ∫|F(T) − F(0)| dV / V0  — pure numerical diffusion

This is the canonical test where Lagrangian VOF is supposed to beat
operator-split Eulerian PLIC, because the spiral filament at t=T/2 is
sub-cell thick and Eulerian PLIC's 1st-order reconstruction cannot track
features below the grid scale.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d
from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x, sweep_flux_y, apply_flux_y,
)

# ─── Parameters ─────────────────────────────────────────────
N = 128                    # 128×128 grid, nz=3 (quasi-2D)
NZ = 3
NH = 1
DX = 1.0 / N
T_END = 2.0                  # T=2 (standard Rider-Kothe period: 1 spiral turn + full reversal)
CFL = 0.5

CIRCLE_CENTER = (0.5, 0.75)
CIRCLE_R = 0.15

print("=" * 72)
print("  Rider-Kothe Reversed Vortex — Eulerian PLIC vs Lagrangian VOF")
print("=" * 72)
print(f"  Grid: {N}×{N}×{NZ}, dx={DX:.5f}, T={T_END}, CFL={CFL}")
print(f"  Circle: center={CIRCLE_CENTER}, R={CIRCLE_R}")
print(f"  Device: {jax.devices()[0]}")


# ─── Init: sub-cell 4×4 sampling ─────────────────────────────
def build_F():
    n_sub = 4
    c = jnp.linspace(DX/2, 1-DX/2, N, dtype=jnp.float32)
    off = (jnp.arange(n_sub, dtype=jnp.float32) + 0.5) / n_sub
    xs = c[:, None] - DX/2 + DX * off[None, :]
    Xs = xs[:, :, None, None]
    Ys = xs[None, None, :, :]
    cx, cy = CIRCLE_CENTER
    in_circle = ((Xs - cx)**2 + (Ys - cy)**2) <= CIRCLE_R**2
    F2d = in_circle.astype(jnp.float32).mean(axis=(1, 3))
    Nt = N + 2*NH
    Ntz = NZ + 2*NH
    F = jnp.zeros((Nt, Nt, Ntz), dtype=jnp.float32)
    F = F.at[NH:NH+N, NH:NH+N, NH:NH+NZ].set(F2d[:, :, None])
    return halo(F)


def halo(F):
    F = F.at[0].set(F[1]); F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F


# ─── Velocity field (time-stepped) ───────────────────────────
cc = jnp.linspace(-NH*DX + DX/2, 1 + NH*DX - DX/2, N + 2*NH, dtype=jnp.float32)

# Spatial pattern at x-face centers (shape Nt-1, Nt, Ntz)
x_fx = 0.5 * (cc[:-1] + cc[1:])         # Nt-1
y_fx = cc                                # Nt
Xfx, Yfx = jnp.meshgrid(x_fx, y_fx, indexing="ij")
u_spatial_2d = -jnp.sin(jnp.pi*Xfx)**2 * jnp.sin(2*jnp.pi*Yfx)

# Spatial pattern at y-face centers (shape Nt, Nt-1, Ntz)
x_fy = cc
y_fy = 0.5 * (cc[:-1] + cc[1:])
Xfy, Yfy = jnp.meshgrid(x_fy, y_fy, indexing="ij")
v_spatial_2d = jnp.sin(2*jnp.pi*Xfy) * jnp.sin(jnp.pi*Yfy)**2

# Broadcast to 3D with z dimension
Ntz = NZ + 2*NH
u_spatial = jnp.broadcast_to(u_spatial_2d[:, :, None], (N+2*NH-1, N+2*NH, Ntz)).astype(jnp.float32)
v_spatial = jnp.broadcast_to(v_spatial_2d[:, :, None], (N+2*NH, N+2*NH-1, Ntz)).astype(jnp.float32)
w_zero = jnp.zeros((N+2*NH, N+2*NH, Ntz-1), dtype=jnp.float32)

# Max vel magnitude (spatial only, modulated by cos(πt/T) ≤ 1)
max_vel = float(jnp.maximum(jnp.abs(u_spatial).max(), jnp.abs(v_spatial).max()))
dt = CFL * DX / max_vel
n_steps = int(np.ceil(T_END / dt))
dt = T_END / n_steps
print(f"  max|u|={max_vel:.4f}, dt={dt:.5f}, n_steps={n_steps}")


# ─── Eulerian PLIC step (uses scalar-like array velocity) ────
@jax.jit
def plic_step(F, u_face, v_face, dt_step):
    def sub_x(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        # padded gather
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * 0.15)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
        C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
        flux = sweep_flux_x(F, nx, ny, nz, C, u_face, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    def sub_y(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        is_if = (F > 1e-6) & (F < 1 - 1e-6)
        max_n = int(F.size * 0.15)
        idx = jnp.where(is_if.ravel(), size=max_n, fill_value=0)[0]
        F_c = F.ravel()[idx]
        nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
        C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)
        C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
        flux = sweep_flux_y(F, nx, ny, nz, C, v_face, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_y(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo(F)

    F = sub_x(F, dt_step/2)
    F = sub_y(F, dt_step)
    F = sub_x(F, dt_step/2)
    return F


# ─── Lagrangian (F shape differs: (Nt, Nt, Ntz, 1)) ─────────
grid = GridInfo(nx=N, ny=N, nz=NZ, nh=NH, dx=DX, dy=DX, dz=DX,
                x_range=(0, 1), y_range=(0, 1), z_range=(0, NZ*DX))

@jax.jit
def lagr_step(F, u_face, v_face, dt_step):
    return advect_vof_lagrangian_3d(F, u_face, v_face, w_zero, grid, dt_step)


# ─── Run Eulerian PLIC ───────────────────────────────────────
print("\n" + "─"*72)
print("  [1/2] Eulerian PLIC")
print("─"*72)
F_ini = build_F()
V0 = float(F_ini[NH:-NH, NH:-NH, NH:-NH].sum() * DX**3)
print(f"  V0 = {V0:.8e}, n_interface = {int(jnp.sum((F_ini > 1e-6) & (F_ini < 1-1e-6)))}")

print("  JIT compiling...", end=" ", flush=True)
t0 = time.time()
scale0 = jnp.float32(np.cos(np.pi * dt / 2 / T_END))
_ = plic_step(F_ini, u_spatial * scale0, v_spatial * scale0, dt)
_.block_until_ready()
print(f"{time.time()-t0:.1f}s")

# Frame capture for GIF (target ~80 frames)
FRAME_EVERY = max(1, n_steps // 80)
plic_frames = [np.asarray(F_ini)]
snapshot_half_plic = None
F_plic = F_ini
t = 0.0
t_wall0 = time.time()
for step in range(n_steps):
    # midpoint-in-time scaling for 2nd-order accuracy
    scale = jnp.float32(np.cos(np.pi * (t + dt/2) / T_END))
    F_plic = plic_step(F_plic, u_spatial * scale, v_spatial * scale, dt)
    t += dt
    if abs(t - T_END/2) < dt/1.5 and snapshot_half_plic is None:
        jax.block_until_ready(F_plic)
        snapshot_half_plic = np.asarray(F_plic)
    if (step + 1) % FRAME_EVERY == 0 or step == n_steps - 1:
        jax.block_until_ready(F_plic)
        plic_frames.append(np.asarray(F_plic))

jax.block_until_ready(F_plic)
wall_plic = time.time() - t_wall0
F_plic_np = np.asarray(F_plic)

V_plic = float(F_plic_np[NH:-NH, NH:-NH, NH:-NH].sum() * DX**3)
L1_plic = float(np.abs(F_plic_np[NH:-NH, NH:-NH, NH:-NH]
                        - np.asarray(F_ini[NH:-NH, NH:-NH, NH:-NH])).sum() * DX**3)
L1_plic_rel = L1_plic / V0 * 100
V_plic_drift = (V_plic - V0) / V0 * 100
print(f"  Wall time: {wall_plic:.1f}s  ({wall_plic/n_steps*1000:.1f} ms/step)")
print(f"  L1 error:   {L1_plic_rel:.3f}%")
print(f"  V drift:    {V_plic_drift:+.4f}%")


# ─── Run Lagrangian VOF ──────────────────────────────────────
print("\n" + "─"*72)
print("  [2/2] Lagrangian VOF")
print("─"*72)

# Lagrangian needs F shape (Nt, Nt, Ntz, 1)
def F_to_lagr(F):
    return F[..., None]

def F_from_lagr(F):
    return F[..., 0]

F_lag = F_to_lagr(F_ini)
print("  JIT compiling...", end=" ", flush=True)
t0 = time.time()
_ = lagr_step(F_lag, u_spatial * scale0, v_spatial * scale0, dt)
_.block_until_ready()
print(f"{time.time()-t0:.1f}s")

lagr_frames = [np.asarray(F_ini)]
snapshot_half_lagr = None
t = 0.0
t_wall0 = time.time()
for step in range(n_steps):
    scale = jnp.float32(np.cos(np.pi * (t + dt/2) / T_END))
    F_lag = lagr_step(F_lag, u_spatial * scale, v_spatial * scale, dt)
    t += dt
    if abs(t - T_END/2) < dt/1.5 and snapshot_half_lagr is None:
        jax.block_until_ready(F_lag)
        snapshot_half_lagr = np.asarray(F_from_lagr(F_lag))
    if (step + 1) % FRAME_EVERY == 0 or step == n_steps - 1:
        jax.block_until_ready(F_lag)
        lagr_frames.append(np.asarray(F_from_lagr(F_lag)))

jax.block_until_ready(F_lag)
wall_lagr = time.time() - t_wall0
F_lagr_np = np.asarray(F_from_lagr(F_lag))

V_lagr = float(F_lagr_np[NH:-NH, NH:-NH, NH:-NH].sum() * DX**3)
L1_lagr = float(np.abs(F_lagr_np[NH:-NH, NH:-NH, NH:-NH]
                        - np.asarray(F_ini[NH:-NH, NH:-NH, NH:-NH])).sum() * DX**3)
L1_lagr_rel = L1_lagr / V0 * 100
V_lagr_drift = (V_lagr - V0) / V0 * 100
print(f"  Wall time: {wall_lagr:.1f}s  ({wall_lagr/n_steps*1000:.1f} ms/step)")
print(f"  L1 error:   {L1_lagr_rel:.3f}%")
print(f"  V drift:    {V_lagr_drift:+.4f}%")


# ─── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  HEAD-TO-HEAD:  Rider-Kothe reversed vortex T={T_END}, {N}×{N} grid")
print("=" * 72)
print(f"  {'Metric':30s}  {'Eulerian PLIC':>15s}  {'Lagrangian VOF':>15s}  {'Winner':>10s}")
print(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*10}")
def winner(a, b, lower_is_better=True):
    if lower_is_better:
        return "Lagrangian" if b < a else "Eulerian"
    return "Lagrangian" if b > a else "Eulerian"
print(f"  {'L1 error (%)':30s}  {L1_plic_rel:15.3f}  {L1_lagr_rel:15.3f}  "
      f"{winner(L1_plic_rel, L1_lagr_rel):>10s}")
print(f"  {'V drift (%)':30s}  {V_plic_drift:+15.4f}  {V_lagr_drift:+15.4f}  "
      f"{winner(abs(V_plic_drift), abs(V_lagr_drift)):>10s}")
print(f"  {'Wall time (s)':30s}  {wall_plic:15.1f}  {wall_lagr:15.1f}  "
      f"{winner(wall_plic, wall_lagr):>10s}")
print("=" * 72)


# ─── Plot ────────────────────────────────────────────────────
os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
z_mid = NH + NZ // 2

x = np.linspace(DX/2, 1-DX/2, N)
X, Y = np.meshgrid(x, x, indexing="ij")

F_ini_slice = np.asarray(F_ini[NH:-NH, NH:-NH, z_mid])
F_plic_slice = F_plic_np[NH:-NH, NH:-NH, z_mid]
F_lagr_slice = F_lagr_np[NH:-NH, NH:-NH, z_mid]
snap_plic_slice = snapshot_half_plic[NH:-NH, NH:-NH, z_mid] if snapshot_half_plic is not None else F_plic_slice
snap_lagr_slice = snapshot_half_lagr[NH:-NH, NH:-NH, z_mid] if snapshot_half_lagr is not None else F_lagr_slice

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

def render(ax, data, title):
    ax.pcolormesh(X, Y, (data > 0.01).astype(float),
                  cmap="Blues", vmin=0, vmax=1, shading="auto")
    ax.contour(X, Y, data, levels=[0.5], colors="k", linewidths=1)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=11)

# Row 0: Eulerian PLIC
render(axes[0, 0], F_ini_slice, "Initial  (t=0)")
render(axes[0, 1], snap_plic_slice, "Eulerian PLIC  (t=T/2, max deformation)")
render(axes[0, 2], F_plic_slice, f"Eulerian PLIC  (t=T, L1={L1_plic_rel:.2f}%)")

# Row 1: Lagrangian
render(axes[1, 0], F_ini_slice, "Initial  (t=0)")
render(axes[1, 1], snap_lagr_slice, "Lagrangian  (t=T/2, max deformation)")
render(axes[1, 2], F_lagr_slice, f"Lagrangian  (t=T, L1={L1_lagr_rel:.2f}%)")

fig.suptitle(
    f"Rider-Kothe reversed vortex T={T_END}  —  {N}×{N} grid, {n_steps} steps\n"
    f"Top: Eulerian PLIC  |  Bottom: Lagrangian VOF  |  "
    f"Winner: {'Lagrangian' if L1_lagr_rel < L1_plic_rel else 'Eulerian'}",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = os.path.join(_HERE, "results", "reversed_vortex_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─── Animated GIF: side-by-side Eulerian | Lagrangian ─────────
import matplotlib.animation as animation

n_frames = min(len(plic_frames), len(lagr_frames))
print(f"Building GIF from {n_frames} frames...")

fig_a, axes_a = plt.subplots(1, 2, figsize=(11, 5.5))
ax_p, ax_l = axes_a
for ax in axes_a:
    ax.set_aspect("equal"); ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)

def frame_slice(frm):
    return frm[NH:-NH, NH:-NH, z_mid]

im_p = ax_p.pcolormesh(X, Y, (frame_slice(plic_frames[0]) > 0.01).astype(float),
                       cmap="Blues", vmin=0, vmax=1, shading="auto")
im_l = ax_l.pcolormesh(X, Y, (frame_slice(lagr_frames[0]) > 0.01).astype(float),
                       cmap="Blues", vmin=0, vmax=1, shading="auto")
title_p = ax_p.set_title("Eulerian PLIC  t=0.000", fontsize=11)
title_l = ax_l.set_title("Lagrangian VOF  t=0.000", fontsize=11)
sup = fig_a.suptitle(f"Rider-Kothe T={T_END}, {N}×{N}", fontsize=12, fontweight="bold")

def animate(i):
    tt = T_END * i / (n_frames - 1)
    data_p = (frame_slice(plic_frames[i]) > 0.01).astype(float)
    data_l = (frame_slice(lagr_frames[i]) > 0.01).astype(float)
    im_p.set_array(data_p.ravel())
    im_l.set_array(data_l.ravel())
    title_p.set_text(f"Eulerian PLIC  t={tt:.3f}")
    title_l.set_text(f"Lagrangian VOF  t={tt:.3f}")
    return im_p, im_l, title_p, title_l

anim = animation.FuncAnimation(fig_a, animate, frames=n_frames, interval=80, blit=False)
gif_out = os.path.join(_HERE, "results", "reversed_vortex_comparison.gif")
anim.save(gif_out, writer=animation.PillowWriter(fps=12))
plt.close()
print(f"Saved: {gif_out}")
