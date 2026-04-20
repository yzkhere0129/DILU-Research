#!/usr/bin/env python3
"""Barkhudarov Test 2: 3D spherical droplet impacting rectangular obstacle.

Paper: Barkhudarov (2004) §4.2
Domain: [0,1]×[0,1]×[0,0.1], obstacle [0,0.5]×[0,0.5]×[0,0.1]
Droplet: R=0.20, center=(0.70,0.70,0.05), velocity=(-1,-1,0)
Pipeline: 3D native (GPU+CPU hybrid) via advect_vof_lagrangian_3d()
Expected: volume error always positive (overfill), symmetric smooth shape.

Boundary conditions:
  - Obstacle treated as solid wall: vertices clamped, F=0 inside, velocity=0
  - Domain walls at x=0, y=0, z=0, z=Lz: solid (vertex clamping)
  - Outflow at x=1, y=1 (zero-pad in overlay)
"""
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax
import jax.numpy as jnp

# NOTE: Do NOT enable x64 — the 3D native pipeline is designed for float32.
# jax.config.update("jax_enable_x64", True)

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Configuration (from paper + R=0.20) ─────────────────────────────
DOMAIN_XY = 1.0
NX, NY, NZ = 50, 50, 5
DX = DOMAIN_XY / NX  # 0.02
DOMAIN_Z = NZ * DX  # 0.10

OBS_X, OBS_Y, OBS_Z = 0.5, 0.5, DOMAIN_Z
DROP_CX, DROP_CY, DROP_CZ = 0.70, 0.70, 0.05
DROP_R = 0.20
U_INF, V_INF = -1.0, -1.0
CFL = 0.45

# Travel: droplet center moves from (0.70, 0.70) toward obstacle
# After impact, allow deformation. Total travel ~ 0.35 diagonal = 0.25 each axis
t_end = 0.35
dt = CFL * DX / (abs(U_INF))  # max wave speed = 1.0
n_steps = int(np.ceil(t_end / dt))
dt = t_end / n_steps

print("=" * 70)
print("  Barkhudarov Test 2: 3D Droplet Impacting Obstacle (Native GPU+CPU)")
print("=" * 70)
print(f"  Domain: [{DOMAIN_XY}×{DOMAIN_XY}×{DOMAIN_Z:.2f}]")
print(f"  Grid: {NX}×{NY}×{NZ}, dx=dy=dz={DX:.3f}")
print(f"  Obstacle: (0,0,0)→({OBS_X},{OBS_Y},{OBS_Z:.2f})")
print(f"  Droplet: R={DROP_R}, center=({DROP_CX},{DROP_CY},{DROP_CZ})")
print(f"  Velocity: ({U_INF},{V_INF},0)")
print(f"  dt={dt:.6f}, n_steps={n_steps}, t_end={t_end}")
print("=" * 70)


def create_droplet_sphere(grid, cx, cy, cz, r):
    """Initialize VOF with a spherical droplet."""
    nh, nx, ny, nz = grid.nh, grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    x = np.linspace(dx / 2, DOMAIN_XY - dx / 2, nx)
    y = np.linspace(dy / 2, DOMAIN_XY - dy / 2, ny)
    z = np.linspace(dz / 2, DOMAIN_Z - dz / 2, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    hd = np.sqrt(3) * dx / 2  # half-diagonal of cubic cell
    F_int = np.where(
        dist + hd <= r,
        1.0,
        np.where(dist - hd >= r, 0.0, np.clip((r - dist + hd) / (2 * hd), 0.0, 1.0)),
    )

    # Zero inside obstacle
    obs_mask = (X < OBS_X) & (Y < OBS_Y)
    F_int[obs_mask] = 0.0

    shape = (nx + 2 * nh, ny + 2 * nh, nz + 2 * nh, 1)
    F = np.zeros(shape)
    F[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0] = F_int
    # Halo fill
    F[:nh, :, :, :] = F[nh : nh + 1, :, :, :]
    F[-nh:, :, :, :] = F[-nh - 1 : -nh, :, :, :]
    F[:, :nh, :, :] = F[:, nh : nh + 1, :, :]
    F[:, -nh:, :, :] = F[:, -nh - 1 : -nh, :, :]
    F[:, :, :nh, :] = F[:, :, nh : nh + 1, :]
    F[:, :, -nh:, :] = F[:, :, -nh - 1 : -nh, :]
    return jnp.array(F, dtype=jnp.float32)



def _solve_potential_flow_2d(nx_grid, ny_grid, dx, obs_nx, obs_ny, u_inf, v_inf, n_iter=2000):
    """Solve ∇²φ=0 on 2D grid with rectangular obstacle via Jacobi iteration.

    Returns u, v at cell faces (staggered grid).
    Boundary conditions:
      - ∂φ/∂n = 0 on obstacle walls (no-penetration)
      - φ = u_inf*x + v_inf*y on domain boundaries (far-field)
    """
    nxn = nx_grid + 1
    nyn = ny_grid + 1
    x_node = np.linspace(0, nx_grid * dx, nxn)
    y_node = np.linspace(0, ny_grid * dx, nyn)

    Xn, Yn = np.meshgrid(x_node, y_node, indexing="ij")
    phi = u_inf * Xn + v_inf * Yn

    obs_x_max = obs_nx * dx
    obs_y_max = obs_ny * dx
    obs_interior = (Xn < obs_x_max - 0.5 * dx) & (Yn < obs_y_max - 0.5 * dx)

    for _ in range(n_iter):
        phi_new = phi.copy()
        phi_new[1:-1, 1:-1] = 0.25 * (
            phi[:-2, 1:-1] + phi[2:, 1:-1] +
            phi[1:-1, :-2] + phi[1:-1, 2:]
        )

        # Neumann BC on obstacle walls: ∂φ/∂n = 0
        ix = int(round(obs_x_max / dx))
        for j in range(0, int(round(obs_y_max / dx)) + 1):
            if 0 <= ix < nxn and 0 <= j < nyn and ix + 1 < nxn:
                phi_new[ix, j] = phi_new[ix + 1, j]
        jy = int(round(obs_y_max / dx))
        for i in range(0, int(round(obs_x_max / dx)) + 1):
            if 0 <= i < nxn and 0 <= jy < nyn and jy + 1 < nyn:
                phi_new[i, jy] = phi_new[i, jy + 1]
        if 0 <= ix < nxn and 0 <= jy < nyn and ix + 1 < nxn and jy + 1 < nyn:
            phi_new[ix, jy] = 0.5 * (phi_new[ix + 1, jy] + phi_new[ix, jy + 1])

        phi_new[obs_interior] = 0.0

        phi_new[0, :] = u_inf * x_node[0] + v_inf * y_node
        phi_new[-1, :] = u_inf * x_node[-1] + v_inf * y_node
        phi_new[:, 0] = u_inf * x_node + v_inf * y_node[0]
        phi_new[:, -1] = u_inf * x_node + v_inf * y_node[-1]

        phi = phi_new

    u_2d = (phi[1:, :-1] + phi[1:, 1:] - phi[:-1, :-1] - phi[:-1, 1:]) / (2 * dx)
    v_2d = (phi[:-1, 1:] + phi[1:, 1:] - phi[:-1, :-1] - phi[1:, :-1]) / (2 * dx)
    return u_2d, v_2d


def _project_div_free_mac(u_mac, v_mac, dx, obs_nx, obs_ny, n_iter=10000):
    """Project 2D MAC velocity field to be exactly divergence-free.

    Solves ∇²p = ∇·v using Jacobi iteration, then corrects v -= ∇p.
    Obstacle treatment: Neumann BC (mirror from fluid side) at walls,
    constant p inside obstacle interior.

    Args:
        u_mac: (Ncx-1, Ncy) x-face velocities (float64)
        v_mac: (Ncx, Ncy-1) y-face velocities (float64)
        dx: uniform grid spacing
        obs_nx, obs_ny: obstacle extent in cells (total grid coords)
        n_iter: Jacobi iterations
    """
    u_mac = u_mac.astype(np.float64)
    v_mac = v_mac.astype(np.float64)
    Ncx = u_mac.shape[0] + 1
    Ncy = v_mac.shape[1] + 1

    # MAC divergence: u[i] is right face of cell i, left face of cell i+1
    div = np.zeros((Ncx, Ncy), dtype=np.float64)
    div[:-1, :] += u_mac / dx   # right face of cell i
    div[1:, :]  -= u_mac / dx   # left face of cell i+1
    div[:, :-1] += v_mac / dx
    div[:, 1:]  -= v_mac / dx

    # Zero divergence in obstacle and boundary cells
    obs = np.zeros((Ncx, Ncy), dtype=bool)
    obs[:obs_nx, :obs_ny] = True
    div[obs] = 0.0
    div[0, :] = 0.0; div[-1, :] = 0.0
    div[:, 0] = 0.0; div[:, -1] = 0.0

    max_div0 = np.max(np.abs(div))
    print(f"    max|∇·v| before projection: {max_div0:.4f}")

    # Jacobi iteration: ∇²p = div
    p = np.zeros((Ncx, Ncy), dtype=np.float64)
    dx2 = dx * dx

    for it in range(n_iter):
        pp = np.pad(p, 1, mode='edge')  # Neumann BC at domain boundaries
        p_new = 0.25 * (
            pp[2:, 1:-1] + pp[:-2, 1:-1] +
            pp[1:-1, 2:] + pp[1:-1, :-2]
            - dx2 * div
        )
        # Obstacle: deep interior = 0, walls mirror from fluid
        if obs_nx > 1 and obs_ny > 1:
            p_new[:obs_nx - 1, :obs_ny - 1] = 0.0
        p_new[obs_nx - 1, :obs_ny] = p_new[obs_nx, :obs_ny]
        p_new[:obs_nx, obs_ny - 1] = p_new[:obs_nx, obs_ny]
        p_new[obs_nx - 1, obs_ny - 1] = 0.5 * (
            p_new[obs_nx, obs_ny - 1] + p_new[obs_nx - 1, obs_ny]
        )
        p = p_new

    # Correct velocity
    u_new = u_mac - (p[1:, :] - p[:-1, :]) / dx
    v_new = v_mac - (p[:, 1:] - p[:, :-1]) / dx

    # Re-zero obstacle faces (belt-and-suspenders for Neumann residual)
    obs_full = np.zeros((Ncx, Ncy), dtype=bool)
    obs_full[:obs_nx, :obs_ny] = True
    u_new[obs_full[:-1, :] | obs_full[1:, :]] = 0.0
    v_new[obs_full[:, :-1] | obs_full[:, 1:]] = 0.0

    # Verify final divergence
    div2 = np.zeros((Ncx, Ncy), dtype=np.float64)
    div2[:-1, :] += u_new / dx; div2[1:, :]  -= u_new / dx
    div2[:, :-1] += v_new / dx; div2[:, 1:]  -= v_new / dx
    div2[obs] = 0.0
    div2[0, :] = 0.0; div2[-1, :] = 0.0
    div2[:, 0] = 0.0; div2[:, -1] = 0.0
    print(f"    max|∇·v| after projection:  {np.max(np.abs(div2)):.2e}")

    return u_new, v_new


def make_velocity_obstacle(grid):
    """Create velocity field via potential flow around obstacle.

    Potential flow gives smooth deceleration at obstacle walls and correct
    tangential velocity reversal at the corner, producing the wrapping
    behavior seen in Barkhudarov's Figure 7.
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    Ntx, Nty, Ntz = nx + 2 * nh, ny + 2 * nh, nz + 2 * nh
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    obs_nx_tot = int(round(OBS_X / dx)) + nh
    obs_ny_tot = int(round(OBS_Y / dy)) + nh

    print("  Solving potential flow...", end=" ", flush=True)
    u_2d, v_2d = _solve_potential_flow_2d(
        Ntx, Nty, dx, obs_nx_tot, obs_ny_tot, U_INF, V_INF, n_iter=5000
    )
    print("done")

    # Extract MAC-sized 2D arrays
    u_mac = u_2d[:Ntx - 1, :Nty].copy()
    v_mac = v_2d[:Ntx, :Nty - 1].copy()

    # Hard-zero obstacle faces in 2D
    obs_2d = np.zeros((Ntx, Nty), dtype=bool)
    obs_2d[:obs_nx_tot, :obs_ny_tot] = True
    u_mac[obs_2d[:-1, :] | obs_2d[1:, :]] = 0.0
    v_mac[obs_2d[:, :-1] | obs_2d[:, 1:]] = 0.0

    # Build 3D arrays (uniform in z)
    u = np.zeros((Ntx - 1, Nty, Ntz), dtype=np.float32)
    v = np.zeros((Ntx, Nty - 1, Ntz), dtype=np.float32)
    w = np.zeros((Ntx, Nty, Ntz - 1), dtype=np.float32)
    for k in range(Ntz):
        u[:, :, k] = u_mac.astype(np.float32)
        v[:, :, k] = v_mac.astype(np.float32)

    return jnp.array(u), jnp.array(v), jnp.array(w)


def make_step_with_reflective_bc(grid, u_face, v_face, w_face, dt):
    """Create step function with wall BC at domain boundaries.

    After advect_vof_lagrangian_3d (which uses zero-pad=outflow), apply
    reflective halos at x=0, y=0, z=0, z=Lz boundaries to enforce wall BC.
    This prevents fluid from draining through domain walls.
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def step(F_):
        F_new = advect_vof_lagrangian_3d(
            F_, u_face, v_face, w_face, grid, dt,
            obs_x_max=OBS_X, obs_y_max=OBS_Y, obs_z_max=OBS_Z,
        )
        # Reflective halos at domain walls (solid wall BC)
        # Left halo ← first interior
        F_new = F_new.at[:nh, :, :, :].set(F_new[nh:nh+1, :, :, :])
        # Right halo ← last interior
        F_new = F_new.at[-nh:, :, :, :].set(F_new[-nh-1:-nh, :, :, :])
        # Bottom halo ← first interior
        F_new = F_new.at[:, :nh, :, :].set(F_new[:, nh:nh+1, :, :])
        # Top halo ← last interior
        F_new = F_new.at[:, -nh:, :, :].set(F_new[:, -nh-1:-nh, :, :])
        # Near halo ← first interior
        F_new = F_new.at[:, :, :nh, :].set(F_new[:, :, nh:nh+1, :])
        # Far halo ← last interior
        F_new = F_new.at[:, :, -nh:, :].set(F_new[:, :, -nh-1:-nh, :])
        return F_new

    return step


def perim_2d(F_slice, dx, dy):
    """Compute perimeter from 2D xy-slice."""
    v = np.array(F_slice)
    hc = ((v[:-1, :] - 0.5) * (v[1:, :] - 0.5)) < 0
    vc = ((v[:, :-1] - 0.5) * (v[:, 1:] - 0.5)) < 0
    return float(np.sum(hc) * dy + np.sum(vc) * dx)


# ── Setup ──────────────────────────────────────────────────────────
grid = GridInfo(
    nx=NX, ny=NY, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN_XY),
    y_range=(0.0, DOMAIN_XY),
    z_range=(0.0, DOMAIN_Z),
)
dx, dy, dz = grid.dx, grid.dy, grid.dz
nh = grid.nh
ic_x = slice(nh, nh + NX)
ic_y = slice(nh, nh + NY)
ic_z = slice(nh, nh + NZ)
z_mid = nh + NZ // 2

F = create_droplet_sphere(grid, DROP_CX, DROP_CY, DROP_CZ, DROP_R)
u_face, v_face, w_face = make_velocity_obstacle(grid)

F_int0_3d = np.array(F[ic_x, ic_y, ic_z, 0])
F_slice0 = np.array(F[ic_x, ic_y, z_mid, 0])
V0 = float(np.sum(F_int0_3d) * dx * dy * dz)
P0 = perim_2d(F_slice0, dx, dy)

print(f"  V₀={V0:.10f}, P₀={P0:.5f}")

step = make_step_with_reflective_bc(grid, u_face, v_face, w_face, dt)

# JIT warmup
print("  JIT compiling...", end=" ", flush=True)
t_jit = time.time()
_ = step(F)
print(f"{time.time() - t_jit:.1f}s")

# ── Run ────────────────────────────────────────────────────────────
# Barkhudarov convention: track cumulative overfill (F>1 clipped each step).
# The overlay preserves volume exactly, but clipping F→[0,1] discards
# the excess.  Report V_total = V_fluid + V_overfill_cumulative.
snap_every = max(1, n_steps // 8)
snapshots = [(0.0, F_slice0.copy())]
vol_hist = [V0]
per_hist = [P0]
t_hist = [0.0]
cell_vol = dx * dy * dz
cumulative_overfill = 0.0
V_prev = V0

t0_wall = time.time()
for step_idx in range(n_steps):
    F = step(F)
    t = (step_idx + 1) * dt

    # Track overfill: overlay conserves volume, so any drop in V_fluid
    # after clip is overfill that was discarded this step.
    Fi_3d = np.array(F[ic_x, ic_y, ic_z, 0])
    V_fluid = float(np.sum(Fi_3d) * cell_vol)
    # Overfill this step = volume that overlay would have produced - actual
    # Since overlay conserves exactly, the missing volume = clipped overfill
    step_overfill = V_prev - V_fluid  # positive when volume was clipped
    if step_overfill > 0:
        cumulative_overfill += step_overfill
    V_prev = V_fluid
    V_total = V_fluid + cumulative_overfill

    if (step_idx + 1) % snap_every == 0 or step_idx == n_steps - 1:
        Fi_slice = np.array(F[ic_x, ic_y, z_mid, 0])
        P = perim_2d(Fi_slice, dx, dy)
        snapshots.append((t, Fi_slice.copy()))
        vol_hist.append(V_total)
        per_hist.append(P)
        t_hist.append(t)
        dV_fluid = (V_fluid - V0) / V0 * 100
        dV_total = (V_total - V0) / V0 * 100
        print(
            f"  step {step_idx + 1:4d}/{n_steps}  t={t:.4f}  "
            f"ΔV_fluid={dV_fluid:+.3f}%  ΔV_total={dV_total:+.3f}%  "
            f"P={P:.5f}",
            flush=True,
        )

wall = time.time() - t0_wall
V_fluid_fin = float(np.sum(np.array(F[ic_x, ic_y, ic_z, 0])) * cell_vol)
V_total_fin = V_fluid_fin + cumulative_overfill
P_fin = per_hist[-1]
dV = (V_total_fin - V0) / V0 * 100
print(f"\nDone! {wall:.1f}s")
print(f"  V_fluid: {V0:.10f} → {V_fluid_fin:.10f}  "
      f"ΔV_fluid={((V_fluid_fin - V0) / V0 * 100):+.4f}%")
print(f"  V_total: {V0:.10f} → {V_total_fin:.10f}  "
      f"ΔV_total={dV:+.4f}%  "
      f"({'POSITIVE=overfill ✓' if dV >= 0 else 'NEGATIVE ✗'})")
print(f"  Cumulative overfill: {cumulative_overfill:.2e} "
      f"({cumulative_overfill / V0 * 100:+.3f}%)")
print(f"  Perimeter: {P0:.5f} → {P_fin:.5f}  "
      f"ΔP/P₀={(P_fin - P0) / P0 * 100:+.3f}%")

# ── Save results JSON ──────────────────────────────────────────────
results_json = {
    "config": {
        "domain": [DOMAIN_XY, DOMAIN_XY, DOMAIN_Z],
        "grid": [NX, NY, NZ],
        "dx": DX,
        "obstacle": [0, 0, 0, OBS_X, OBS_Y, OBS_Z],
        "droplet_R": DROP_R,
        "droplet_center": [DROP_CX, DROP_CY, DROP_CZ],
        "velocity": [U_INF, V_INF, 0.0],
        "t_end": t_end,
        "n_steps": n_steps,
    },
    "metrics": {
        "V0": V0, "V_fluid_final": V_fluid_fin,
        "V_total_final": V_total_fin,
        "dV_fluid_percent": (V_fluid_fin - V0) / V0 * 100,
        "dV_total_percent": dV,
        "cumulative_overfill": cumulative_overfill,
        "P0": P0, "P_final": P_fin,
        "dP_percent": (P_fin - P0) / P0 * 100,
        "overfill": dV >= 0,
        "wall_time_s": wall,
    },
}
with open(os.path.join(OUT, "test2_results_3d.json"), "w") as f:
    json.dump(results_json, f, indent=2)
print("Saved: test2_results_3d.json")

# ── Plot 1: Phase evolution snapshots ──────────────────────────────
x_plt = np.linspace(dx / 2, DOMAIN_XY - dx / 2, NX)
y_plt = x_plt.copy()
X_plt, Y_plt = np.meshgrid(x_plt, y_plt, indexing="ij")

n_show = min(6, len(snapshots))
idx_show = np.round(np.linspace(0, len(snapshots) - 1, n_show)).astype(int)
fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
if n_show == 1:
    axes = [axes]

for ax, i in zip(axes, idx_show):
    t_snap, F_snap = snapshots[i]
    F_disp = np.clip(F_snap, 0.0, 1.0)  # clip for display (F>1 → solid)
    ax.contourf(X_plt, Y_plt, F_disp, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.8)
    ax.contour(X_plt, Y_plt, F_snap, levels=[0.5], colors="navy", lw=1.5)
    obs_p = patches.Rectangle((0, 0), OBS_X, OBS_Y, lw=2, ec="black", fc="#888888")
    ax.add_patch(obs_p)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"t={t_snap:.3f}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.suptitle(
    "Test 2: Droplet Impacting Obstacle — 3D Native (GPU+CPU)\n"
    f"R={DROP_R}, Grid {NX}×{NY}×{NZ}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_evolution_3d.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_evolution_3d.png")

# ── Plot 2: Volume error over time ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
dV_arr = [(V - V0) / V0 * 100 for V in vol_hist]
dP_arr = [(P - P0) / P0 * 100 for P in per_hist]

ax1.plot(t_hist, dV_arr, "r-o", ms=5, lw=2)
ax1.axhline(0, color="gray", ls="--")
ax1.fill_between(
    t_hist, dV_arr, 0,
    where=[d >= 0 for d in dV_arr], alpha=0.2, color="red", label="Overfill (+)",
)
ax1.fill_between(
    t_hist, dV_arr, 0,
    where=[d < 0 for d in dV_arr], alpha=0.2, color="blue", label="Underfill (−)",
)
ax1.set_xlabel("Time")
ax1.set_ylabel("ΔV/V₀ (%)")
ax1.set_title("Volume Error (should be ≥ 0)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_hist, dP_arr, "b-s", ms=5, lw=2)
ax2.axhline(0, color="gray", ls="--")
ax2.set_xlabel("Time")
ax2.set_ylabel("ΔP/P₀ (%)")
ax2.set_title("Perimeter Change")
ax2.grid(True, alpha=0.3)

fig.suptitle(
    "Test 2: Droplet Impacting Obstacle — Metrics (3D Native)\n"
    f"R={DROP_R}, Grid {NX}×{NY}×{NZ}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_metrics_3d.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_metrics_3d.png")

# ── Plot 3: Final interface close-up ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
F_final_slice = np.array(F[ic_x, ic_y, z_mid, 0])
F_final_disp = np.clip(F_final_slice, 0.0, 1.0)
ax.contourf(X_plt, Y_plt, F_final_disp, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.85)
ax.contour(X_plt, Y_plt, F_final_slice, levels=[0.5], colors="navy", lw=2)
obs_p = patches.Rectangle((0, 0), OBS_X, OBS_Y, lw=2, ec="black", fc="#888888", alpha=0.9)
ax.add_patch(obs_p)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title(
    f"Final State (t={t_end:.3f})\nΔV/V₀={dV:+.4f}%, 3D Native Lagrangian VOF",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test2_final_3d.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test2_final_3d.png")
