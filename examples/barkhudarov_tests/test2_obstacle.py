#!/usr/bin/env python3
"""Barkhudarov Test 2: 2D circular droplet impacting rectangular obstacle.

Paper: Barkhudarov (2004) §4.2
Domain: [0,1]×[0,1], obstacle [0,0.5]×[0,0.5]
Droplet: R=0.19, center=(0.70,0.70), velocity=(−1,−1)
Expected: volume error always positive (overfill), symmetric smooth shape.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_laseram.grid import create_grid
from jax_laseram.vof.lagrangian import advect_vof_lagrangian

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ────────────────────────────────────────────────
DOMAIN = 1.0
OBS_X, OBS_Y = 0.5, 0.5
DROP_CX, DROP_CY = 0.70, 0.70
DROP_R = 0.19
U_INF, V_INF = -1.0, -1.0
NX = 50          # 50×50: dx=0.02, D/dx=19 cells/diameter
CFL = 0.45


def make_field(grid):
    nh, nx, ny = grid.nh, grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    x = np.linspace(dx/2, DOMAIN-dx/2, nx)
    y = np.linspace(dy/2, DOMAIN-dy/2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X-DROP_CX)**2 + (Y-DROP_CY)**2)
    hd = np.sqrt(2)*dx/2
    F_int = np.where(dist+hd<=DROP_R, 1.0,
            np.where(dist-hd>=DROP_R, 0.0,
                     np.clip((DROP_R-dist+hd)/(2*hd), 0, 1)))
    obs = (X < OBS_X) & (Y < OBS_Y)
    F_int[obs] = 0.0
    shape = (nx+2*nh, ny+2*nh, 1)
    F = np.zeros(shape)
    F[nh:nh+nx, nh:nh+ny, 0] = F_int
    F[:nh,:,:] = F[nh:nh+1,:,:]; F[-nh:,:,:] = F[-nh-1:-nh,:,:]
    F[:,:nh,:] = F[:,nh:nh+1,:]; F[:,-nh:,:] = F[:,-nh-1:-nh,:]
    return jnp.array(F)


def make_velocity(grid):
    nh, nx, ny = grid.nh, grid.nx, grid.ny
    Ntx, Nty = nx+2*nh, ny+2*nh
    dx, dy = grid.dx, grid.dy
    x_cc = np.array([grid.x_range[0] + (i-nh+0.5)*dx for i in range(Ntx)])
    y_cc = np.array([grid.y_range[0] + (j-nh+0.5)*dy for j in range(Nty)])
    obs_cell = (x_cc[:,None] < OBS_X) & (y_cc[None,:] < OBS_Y)
    u = np.full((Ntx-1, Nty, 1), U_INF)
    v = np.full((Ntx, Nty-1, 1), V_INF)
    u[obs_cell[:-1,:]|obs_cell[1:,:], 0] = 0.0
    v[obs_cell[:,:-1]|obs_cell[:,1:], 0] = 0.0
    return jnp.array(u), jnp.array(v)


def perim(F_int, dx, dy):
    v = np.array(F_int)
    hc = ((v[:-1,:]-0.5)*(v[1:,:]-0.5)) < 0
    vc = ((v[:,:-1]-0.5)*(v[:,1:]-0.5)) < 0
    return float(np.sum(hc)*dy + np.sum(vc)*dx)


grid = create_grid(NX, NX, nz=1, x_range=(0.0, DOMAIN), y_range=(0.0, DOMAIN), nh=1)
dx, dy = grid.dx, grid.dy
nh = grid.nh
ic = slice(nh, -nh)

F = make_field(grid)
u_face, v_face = make_velocity(grid)

# Time: droplet needs to travel ~(0.7-0.5)=0.2 to reach obstacle
# then deform, run a bit more. Total travel ≈ 0.5 diagonal = 0.35 each axis.
t_end = 0.35
dt = CFL * dx / (abs(U_INF) + abs(V_INF)) * 2  # use max wave speed
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

F_int0 = np.array(F[ic, ic, 0])
V0 = float(np.sum(F_int0)*dx*dy)
P0 = perim(F_int0, dx, dy)

print(f"Test 2: Droplet Impacting Obstacle")
print(f"  Grid {NX}×{NX}, dx={dx:.3f}, D/dx={2*DROP_R/dx:.1f}")
print(f"  Droplet R={DROP_R}, center=({DROP_CX},{DROP_CY})")
print(f"  Velocity: ({U_INF},{V_INF})")
print(f"  dt={dt:.6f}, n_steps={n_steps}, t_end={t_end}")
print(f"  V0={V0:.8f}, P0={P0:.5f}")
print(f"  Gap to obstacle: {min(DROP_CX-OBS_X, DROP_CY-OBS_Y)-DROP_R:.4f}")

# JIT warmup
_ = advect_vof_lagrangian(F, u_face, v_face, grid, dt, obs_x_max=OBS_X, obs_y_max=OBS_Y)

snap_every = max(1, n_steps // 8)
snapshots = [(0.0, F_int0.copy())]
vol_hist = [V0]
per_hist = [P0]
t_hist = [0.0]

t0_wall = time.time()
for step in range(n_steps):
    F = advect_vof_lagrangian(F, u_face, v_face, grid, dt, obs_x_max=OBS_X, obs_y_max=OBS_Y)
    t = (step+1)*dt
    if (step+1) % snap_every == 0 or step == n_steps-1:
        Fi = np.array(F[ic, ic, 0])
        V = float(np.sum(Fi)*dx*dy)
        P = perim(Fi, dx, dy)
        snapshots.append((t, Fi.copy()))
        vol_hist.append(V)
        per_hist.append(P)
        t_hist.append(t)
        sign = "+" if V > V0 else "-"
        print(f"  step {step+1:4d}/{n_steps}  t={t:.4f}  "
              f"ΔV/V₀={sign}{abs(V-V0)/V0*100:.4f}%  P={P:.5f}", flush=True)

wall = time.time() - t0_wall
V_fin = vol_hist[-1]
P_fin = per_hist[-1]
dV = (V_fin - V0) / V0 * 100
print(f"\nDone {wall:.1f}s")
print(f"  Volume: {V0:.8f} → {V_fin:.8f}  ΔV/V₀={dV:+.5f}% ({'POSITIVE=overfill ✓' if dV>=0 else 'NEGATIVE'})")
print(f"  Perimeter: {P0:.5f} → {P_fin:.5f}  ΔP/P₀={(P_fin-P0)/P0*100:+.3f}%")

# ── Plot 1: Phase evolution snapshots ────────────────────────────
x = np.linspace(dx/2, DOMAIN-dx/2, NX)
y = x.copy()
X, Y = np.meshgrid(x, y, indexing="ij")

n_show = min(6, len(snapshots))
idx_show = np.round(np.linspace(0, len(snapshots)-1, n_show)).astype(int)
fig, axes = plt.subplots(1, n_show, figsize=(3.5*n_show, 4))
if n_show == 1: axes = [axes]

for ax, i in zip(axes, idx_show):
    t_snap, F_snap = snapshots[i]
    ax.contourf(X, Y, F_snap, levels=np.linspace(0,1,21), cmap="Blues", alpha=0.8)
    ax.contour(X, Y, F_snap, levels=[0.5], colors="navy", lw=1.5)
    obs_p = patches.Rectangle((0,0), OBS_X, OBS_Y, lw=2, ec="black", fc="#888888")
    ax.add_patch(obs_p)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.set_title(f"t={t_snap:.3f}", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")

fig.suptitle("Test 2: Droplet Impacting Obstacle (Lagrangian VOF)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_evolution.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_evolution.png")

# ── Plot 2: Volume error over time ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
dV_arr = [(V-V0)/V0*100 for V in vol_hist]
dP_arr = [(P-P0)/P0*100 for P in per_hist]

ax1.plot(t_hist, dV_arr, "r-o", ms=5, lw=2)
ax1.axhline(0, color="gray", ls="--")
ax1.fill_between(t_hist, dV_arr, 0,
                  where=[d>=0 for d in dV_arr], alpha=0.2, color="red", label="Overfill (+)")
ax1.fill_between(t_hist, dV_arr, 0,
                  where=[d<0 for d in dV_arr], alpha=0.2, color="blue", label="Underfill (−)")
ax1.set_xlabel("Time"); ax1.set_ylabel("ΔV/V₀ (%)")
ax1.set_title("Volume Error (should be ≥ 0)"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(t_hist, dP_arr, "b-s", ms=5, lw=2)
ax2.axhline(0, color="gray", ls="--")
ax2.set_xlabel("Time"); ax2.set_ylabel("ΔP/P₀ (%)")
ax2.set_title("Perimeter Change"); ax2.grid(True, alpha=0.3)

fig.suptitle("Test 2: Droplet Impacting Obstacle — Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_metrics.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_metrics.png")

# ── Plot 3: Final interface close-up ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.contourf(X, Y, np.array(F[ic,ic,0]), levels=np.linspace(0,1,21), cmap="Blues", alpha=0.85)
ax.contour(X, Y, np.array(F[ic,ic,0]), levels=[0.5], colors="navy", lw=2)
obs_p = patches.Rectangle((0,0), OBS_X, OBS_Y, lw=2, ec="black", fc="#888888", alpha=0.9)
ax.add_patch(obs_p)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
ax.set_xlabel("x",fontsize=12); ax.set_ylabel("y",fontsize=12)
ax.set_title(f"Final State (t={t_end:.3f})\nΔV/V₀={dV:+.4f}%, Lagrangian VOF", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_final.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_final.png")
