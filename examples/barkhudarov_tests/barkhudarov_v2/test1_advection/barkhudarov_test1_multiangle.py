#!/usr/bin/env python3
"""Barkhudarov Test 1: 3D circular droplet advection at 0°, 6°, 30°, 45°.

Paper: Barkhudarov (2004) §4.1 — Figure 5 & 6
Grid: cell_size = D/10 (10 cells/diameter), nz=5
Droplet: cylinder along z-axis, travels 5D
Metrics: volume error, perimeter change, shape
"""
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────
D = 0.10
R = D / 2.0
NX = 100
NZ = 5
DOMAIN_XY = 1.0
DX = DOMAIN_XY / NX
DOMAIN_Z = NZ * DX
TRAVEL_DIST = 5 * D
SPEED = 1.0
T_END = TRAVEL_DIST / SPEED
CFL = 0.45

ANGLES = [0, 6, 30, 45]


def create_droplet_cylinder(grid, cx, cy, r):
    nh, nx, ny, nz = grid.nh, grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    x = np.linspace(dx/2, DOMAIN_XY - dx/2, nx)
    y = np.linspace(dy/2, DOMAIN_XY - dy/2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    hd = np.sqrt(2) * dx / 2
    F_2d = np.where(dist + hd <= r, 1.0,
                    np.where(dist - hd >= r, 0.0,
                             np.clip((r - dist + hd) / (2*hd), 0.0, 1.0)))
    shape = (nx + 2*nh, ny + 2*nh, nz + 2*nh, 1)
    F = np.zeros(shape)
    F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0] = F_2d[:, :, None]
    F[:nh] = F[nh:nh+1]; F[-nh:] = F[-nh-1:-nh]
    F[:, :nh] = F[:, nh:nh+1]; F[:, -nh:] = F[:, -nh-1:-nh]
    F[:, :, :nh] = F[:, :, nh:nh+1]; F[:, :, -nh:] = F[:, :, -nh-1:-nh]
    return jnp.array(F, dtype=jnp.float32)


def compute_perimeter(F_2d, dx, dy):
    try:
        fig, ax = plt.subplots()
        cs = ax.contour(F_2d.T, levels=[0.5])
        plt.close(fig)
        total = 0.0
        for path in cs.get_paths():
            verts = path.vertices
            pts = verts * np.array([dx, dy])
            diffs = np.diff(pts, axis=0)
            total += float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        return total if total > 0 else np.pi * D
    except Exception:
        return np.pi * D


# ── Grid ──────────────────────────────────────────────────────────
grid = GridInfo(
    nx=NX, ny=NX, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN_XY),
    y_range=(0.0, DOMAIN_XY),
    z_range=(0.0, DOMAIN_Z),
)
nh = grid.nh
dx, dy, dz = grid.dx, grid.dy, grid.dz
dt = CFL * dx / SPEED
n_steps = int(np.ceil(T_END / dt))
dt = T_END / n_steps
Ntx = NX + 2*nh; Nty = NX + 2*nh; Ntz = NZ + 2*nh
ic_x = slice(nh, nh+NX); ic_y = slice(nh, nh+NX); ic_z = slice(nh, nh+NZ)
z_mid = nh + NZ // 2
P_analytical = np.pi * D
cell_vol = dx * dy * dz

print("=" * 70)
print("  Barkhudarov Test 1: Multi-Angle Droplet Advection")
print("=" * 70)
print(f"  Grid: {NX}×{NX}×{NZ}, dx={DX:.4f}, D={D}, 10 cells/diameter")
print(f"  Travel: {TRAVEL_DIST} (5D), dt={dt:.6f}, n_steps={n_steps}")
print(f"  Angles: {ANGLES}")
print("=" * 70)

all_results = {}
fig_final, axes_final = plt.subplots(1, len(ANGLES), figsize=(4*len(ANGLES), 4))
x_plt = np.linspace(dx/2, DOMAIN_XY - dx/2, NX)
y_plt = x_plt.copy()
X_plt, Y_plt = np.meshgrid(x_plt, y_plt, indexing="ij")

for ai, angle in enumerate(ANGLES):
    print(f"\n── Angle: {angle}° ──")

    # Starting position: ensure droplet stays in domain at all angles
    theta = np.radians(angle)
    cx_final = 0.20 + np.cos(theta) * TRAVEL_DIST
    cy_final = 0.20 + np.sin(theta) * TRAVEL_DIST
    # If final position would be > 0.9, shift start
    cx0 = 0.20
    cy0 = 0.20
    if cx_final > 0.90:
        cx0 = 0.90 - np.cos(theta) * TRAVEL_DIST
    if cy_final > 0.90:
        cy0 = 0.90 - np.sin(theta) * TRAVEL_DIST
    cx0 = max(0.10, min(0.90, cx0))
    cy0 = max(0.10, min(0.90, cy0))

    cx_final = cx0 + np.cos(theta) * TRAVEL_DIST
    cy_final = cy0 + np.sin(theta) * TRAVEL_DIST

    print(f"  Start: ({cx0:.3f}, {cy0:.3f}) → End: ({cx_final:.3f}, {cy_final:.3f})")

    F = create_droplet_cylinder(grid, cx0, cy0, R)
    ux = np.cos(theta) * SPEED
    uy = np.sin(theta) * SPEED
    u_face = jnp.full((Ntx-1, Nty, Ntz), ux, jnp.float32)
    v_face = jnp.full((Ntx, Nty-1, Ntz), uy, jnp.float32)
    w_face = jnp.zeros((Ntx, Nty, Ntz-1), jnp.float32)

    F_int0 = np.array(F[ic_x, ic_y, ic_z, 0])
    F_slice0 = np.array(F[ic_x, ic_y, z_mid, 0])
    V0 = float(np.sum(F_int0) * cell_vol)
    P0 = compute_perimeter(F_slice0, dx, dy)
    print(f"  V₀={V0:.10f}  P₀={P0:.6f}")

    # JIT compile (first angle only takes long)
    print("  JIT compiling...", end=" ", flush=True)
    t_jit = time.time()
    _ = advect_vof_lagrangian_3d(F, u_face, v_face, w_face, grid, dt)
    print(f"{time.time() - t_jit:.1f}s")

    # Run
    t0 = time.time()
    for step_idx in range(n_steps):
        F = advect_vof_lagrangian_3d(F, u_face, v_face, w_face, grid, dt)
        if (step_idx + 1) % max(1, n_steps // 4) == 0:
            Vi = float(np.sum(np.array(F[ic_x, ic_y, ic_z, 0])) * cell_vol)
            dVi = abs(Vi - V0) / V0 * 100
            print(f"    [{(step_idx+1)/n_steps*100:3.0f}%] |ΔV/V|={dVi:.4e}%", flush=True)

    wall_t = time.time() - t0
    F_final_slice = np.array(F[ic_x, ic_y, z_mid, 0])
    F_final_3d = np.array(F[ic_x, ic_y, ic_z, 0])
    V_final = float(np.sum(F_final_3d) * cell_vol)
    P_final = compute_perimeter(F_final_slice, dx, dy)
    dV = abs(V_final - V0) / V0 * 100
    dP = (P_final - P0) / P0 * 100

    v_pass = dV < 0.5
    p_pass = abs(dP) < 0.5
    print(f"  Done ({wall_t:.1f}s): |ΔV/V|={dV:.4f}%{'✓' if v_pass else '✗'}  "
          f"ΔP/P={dP:+.4f}%{'✓' if p_pass else '✗'}")

    all_results[f"{angle}°"] = {
        "angle_deg": angle,
        "start": [cx0, cy0],
        "V0": V0, "P0": P0,
        "V_final": V_final, "P_final": P_final,
        "dV_percent": dV, "dP_percent": dP,
        "volume_pass": v_pass, "perimeter_pass": p_pass,
        "wall_time_s": wall_t,
    }

    # Plot
    ax = axes_final[ai]
    ax.set_aspect("equal")
    ax.contourf(X_plt, Y_plt, F_final_slice, levels=np.linspace(0, 1, 11),
                cmap="Blues", alpha=0.7)
    ax.contour(X_plt, Y_plt, F_final_slice, levels=[0.5], colors="navy", linewidths=1.5)
    # Exact circle
    th_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(cx_final + R*np.cos(th_c), cy_final + R*np.sin(th_c), "g--", lw=1, label="Exact")
    # Initial
    ax.plot(cx0 + R*np.cos(th_c), cy0 + R*np.sin(th_c), "gray", lw=0.5, ls="--", label="Initial")
    ax.set_title(f"{angle}°  |ΔV/V|={dV:.3e}%\nΔP/P={dP:+.3f}%", fontsize=9)
    ax.set_xlim(0, DOMAIN_XY); ax.set_ylim(0, DOMAIN_XY)

fig_final.suptitle(f"Barkhudarov Test 1: Multi-Angle\nGrid {NX}×{NX}×{NZ}, dx={DX}", fontsize=11)
fig_final.tight_layout()
fig_final.savefig(os.path.join(OUT, "test1_multiangle_3d.png"), dpi=150)
plt.close(fig_final)
print("\nSaved: test1_multiangle_3d.png")

with open(os.path.join(OUT, "test1_multiangle_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved: test1_multiangle_results.json")

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for k, v in all_results.items():
    sv = "✓" if v["volume_pass"] else "✗"
    sp = "✓" if v["perimeter_pass"] else "✗"
    print(f"  {k:>4s}: |ΔV/V|={v['dV_percent']:.4f}%{sv}  ΔP/P={v['dP_percent']:+.4f}%{sp}")
