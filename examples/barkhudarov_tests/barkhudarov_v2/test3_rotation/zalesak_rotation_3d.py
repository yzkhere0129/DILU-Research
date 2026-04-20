#!/usr/bin/env python3
"""Test 3: Zalesak's slotted disk rotation (3D extrusion).

Classic benchmark for interface advection under shear flow.
A slotted disk rotates around the domain center by solid-body rotation.
After one full rotation, the disk should return to its initial position
with its slot still intact.

Configuration (classic Zalesak parameters):
  Domain: [0, 1] × [0, 1] × [0, 0.05]
  Grid: 100 × 100 × 5
  Disk: radius R = 0.15, center (0.5, 0.75)
  Slot: width 0.05, depth 0.25 (oriented along -y direction)
  Rotation: solid body, ω = 2π/T around (0.5, 0.5)
  Full rotation: T = 6.28 s (ω = 1 rad/s)
  CFL: 0.45

Success metrics:
  - L1 error: ∫|F_final - F_initial| dV should be < a few percent
  - Volume conservation: |ΔV/V₀| < 1%
  - Slot preserved (not filled in)
  - Shape close to circular disk
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

# ── Configuration (classic Zalesak) ──────────────────────────────
NX, NY, NZ = 100, 100, 5
DOMAIN_XY = 1.0
DX = DOMAIN_XY / NX  # 0.01
DOMAIN_Z = NZ * DX   # 0.05
CFL = 0.45

# Disk parameters
R = 0.15
DISK_CX, DISK_CY = 0.50, 0.75
SLOT_WIDTH = 0.05
SLOT_DEPTH = 0.25   # depth of slot cut into disk (from top of disk downward)

# Rotation parameters
ROT_CX, ROT_CY = 0.5, 0.5
OMEGA = 1.0  # rad/s
T_FULL_ROTATION = 2.0 * np.pi / OMEGA  # 6.283 s

# Timestep: CFL based on max velocity in domain
# max velocity at domain corner: ω × √(0.5² + 0.5²) ≈ 0.707
MAX_VEL = OMEGA * np.sqrt(0.5**2 + 0.5**2)
DT_MAX = CFL * DX / MAX_VEL

N_STEPS = int(np.ceil(T_FULL_ROTATION / DT_MAX))
DT = T_FULL_ROTATION / N_STEPS

print("=" * 70)
print("  Test 3: Zalesak Slotted Disk Rotation (3D)")
print("=" * 70)
print(f"  Domain: [0,{DOMAIN_XY}]² × [0,{DOMAIN_Z:.2f}]")
print(f"  Grid: {NX}×{NY}×{NZ}, dx={DX}")
print(f"  Disk: R={R}, center=({DISK_CX},{DISK_CY})")
print(f"  Slot: width={SLOT_WIDTH}, depth={SLOT_DEPTH}")
print(f"  Rotation: ω={OMEGA} rad/s around ({ROT_CX},{ROT_CY})")
print(f"  Full rotation: T={T_FULL_ROTATION:.3f} s")
print(f"  CFL={CFL}, max_vel={MAX_VEL:.4f}, dt={DT:.6f}")
print(f"  n_steps={N_STEPS}")
print("=" * 70)


def create_slotted_disk(grid, cx, cy, r, slot_w, slot_d):
    """Initialize VOF with a slotted disk (extruded in z).

    The disk is a circle at (cx, cy). The slot is a rectangular cut
    through the disk, oriented along -y direction from the top.
    Slot extent: x in [cx - slot_w/2, cx + slot_w/2],
                 y in [cy + r - slot_d, cy + r]
    """
    nh, nx, ny, nz = grid.nh, grid.nx, grid.ny, grid.nz
    dx = grid.dx

    # Use sub-sampling for smooth initial condition (4x4 per cell)
    NSUB = 4
    x_edges = np.linspace(0, DOMAIN_XY, nx + 1)
    y_edges = np.linspace(0, DOMAIN_XY, ny + 1)

    F_2d = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            # Sample NSUB × NSUB points in cell (i,j)
            xs = np.linspace(x_edges[i], x_edges[i+1], NSUB + 1)[:-1] + (x_edges[i+1] - x_edges[i]) / (2*NSUB)
            ys = np.linspace(y_edges[j], y_edges[j+1], NSUB + 1)[:-1] + (y_edges[j+1] - y_edges[j]) / (2*NSUB)
            count = 0
            for xv in xs:
                for yv in ys:
                    dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
                    in_disk = dist <= r
                    # Slot cuts from top (y > cy) downward for depth slot_d
                    in_slot = (abs(xv - cx) <= slot_w / 2) and \
                              (yv >= cy + r - slot_d) and (yv <= cy + r)
                    if in_disk and not in_slot:
                        count += 1
            F_2d[i, j] = count / (NSUB * NSUB)

    shape = (nx + 2*nh, ny + 2*nh, nz + 2*nh, 1)
    F = np.zeros(shape, dtype=np.float32)
    F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0] = F_2d[:, :, None]
    # Halo: extrapolate
    F[:nh] = F[nh:nh+1]; F[-nh:] = F[-nh-1:-nh]
    F[:, :nh] = F[:, nh:nh+1]; F[:, -nh:] = F[:, -nh-1:-nh]
    F[:, :, :nh] = F[:, :, nh:nh+1]; F[:, :, -nh:] = F[:, :, -nh-1:-nh]
    return jnp.array(F, dtype=jnp.float32)


def make_rotation_velocity(grid, omega, rot_cx, rot_cy):
    """Solid-body rotation velocity field (2D in xy, w=0).

    u(x, y) = -ω (y - rot_cy)
    v(x, y) =  ω (x - rot_cx)
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    Ntx, Nty, Ntz = nx + 2*nh, ny + 2*nh, nz + 2*nh
    dx = grid.dx

    # u_face[i, j, k] is at x-face between cells i and i+1, at y-cell center j
    # x_face = (i + 1 - nh) * dx  (for i = 0 .. Ntx-2, since Ntx-1 faces)
    # y_cc = (j - nh + 0.5) * dx
    u_face = np.zeros((Ntx - 1, Nty, Ntz), dtype=np.float32)
    v_face = np.zeros((Ntx, Nty - 1, Ntz), dtype=np.float32)
    w_face = np.zeros((Ntx, Nty, Ntz - 1), dtype=np.float32)

    for i in range(Ntx - 1):
        for j in range(Nty):
            x_f = grid.x_range[0] + (i + 1 - nh) * dx
            y_c = grid.y_range[0] + (j - nh + 0.5) * dx
            u = -omega * (y_c - rot_cy)
            u_face[i, j, :] = u

    for i in range(Ntx):
        for j in range(Nty - 1):
            x_c = grid.x_range[0] + (i - nh + 0.5) * dx
            y_f = grid.y_range[0] + (j + 1 - nh) * dx
            v = omega * (x_c - rot_cx)
            v_face[i, j, :] = v

    return jnp.array(u_face), jnp.array(v_face), jnp.array(w_face)


def perim_2d(F_slice, dx):
    v = np.array(F_slice)
    hc = ((v[:-1, :] - 0.5) * (v[1:, :] - 0.5)) < 0
    vc = ((v[:, :-1] - 0.5) * (v[:, 1:] - 0.5)) < 0
    return float(np.sum(hc) * dx + np.sum(vc) * dx)


# ── Setup ──────────────────────────────────────────────────────
grid = GridInfo(
    nx=NX, ny=NY, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN_XY),
    y_range=(0.0, DOMAIN_XY),
    z_range=(0.0, DOMAIN_Z),
)
nh = grid.nh
dx = grid.dx
ic_x = slice(nh, nh + NX)
ic_y = slice(nh, nh + NY)
ic_z = slice(nh, nh + NZ)
z_mid = nh + NZ // 2

print("\n  Initializing slotted disk...")
F = create_slotted_disk(grid, DISK_CX, DISK_CY, R, SLOT_WIDTH, SLOT_DEPTH)
F_init = np.array(F[ic_x, ic_y, ic_z, 0]).copy()  # save for comparison

print("  Building rotation velocity field...")
u_face, v_face, w_face = make_rotation_velocity(grid, OMEGA, ROT_CX, ROT_CY)

V0 = float(np.sum(F_init) * DX**3)
F_slice0 = np.array(F[ic_x, ic_y, z_mid, 0])
P0 = perim_2d(F_slice0, DX)
print(f"  V₀={V0:.10f}, P₀={P0:.5f}")

# JIT warmup
print("\n  JIT compiling...", end=" ", flush=True)
t_jit = time.time()
_ = advect_vof_lagrangian_3d(F, u_face, v_face, w_face, grid, DT)
_.block_until_ready()
print(f"{time.time() - t_jit:.1f}s")

# ── Run one full rotation ───────────────────────────────────────
snap_every = max(1, N_STEPS // 8)
snapshots = [(0.0, F_slice0.copy())]
vol_hist = [V0]
per_hist = [P0]
t_hist = [0.0]

t0_wall = time.time()
print(f"\n  Running {N_STEPS} steps for one full rotation...")
for step_idx in range(N_STEPS):
    F = advect_vof_lagrangian_3d(F, u_face, v_face, w_face, grid, DT)
    t = (step_idx + 1) * DT

    if (step_idx + 1) % snap_every == 0 or step_idx == N_STEPS - 1:
        Fi_3d = np.array(F[ic_x, ic_y, ic_z, 0])
        Fi_slice = np.array(F[ic_x, ic_y, z_mid, 0])
        V = float(np.sum(Fi_3d) * DX**3)
        P = perim_2d(Fi_slice, DX)
        snapshots.append((t, Fi_slice.copy()))
        vol_hist.append(V)
        per_hist.append(P)
        t_hist.append(t)
        dV = (V - V0) / V0 * 100
        # Angle rotated so far
        angle_deg = np.degrees(OMEGA * t) % 360
        print(
            f"  step {step_idx + 1:4d}/{N_STEPS}  t={t:.4f}  "
            f"angle={angle_deg:.1f}°  "
            f"ΔV/V₀={dV:+.4f}%  P={P:.5f}",
            flush=True,
        )

wall = time.time() - t0_wall

# ── Compute metrics ──────────────────────────────────────────────
F_final_3d = np.array(F[ic_x, ic_y, ic_z, 0])
V_fin = float(np.sum(F_final_3d) * DX**3)
dV_pct = (V_fin - V0) / V0 * 100

# L1 error: ∫|F_final - F_init| dV
L1_error = float(np.sum(np.abs(F_final_3d - F_init)) * DX**3)
L1_rel_pct = L1_error / V0 * 100

# L1 error relative to interface area (2 * disk area)
L1_interface = float(np.sum(np.abs(F_final_3d - F_init)) * DX**3) / (2 * V0)

F_fin_slice = np.array(F[ic_x, ic_y, z_mid, 0])
P_fin = perim_2d(F_fin_slice, DX)

print(f"\nDone! {wall:.1f}s")
print("=" * 70)
print("  Metrics after one full rotation:")
print("=" * 70)
print(f"  Volume:        V₀={V0:.10f} → V_fin={V_fin:.10f}")
print(f"                 ΔV/V₀ = {dV_pct:+.5f}%")
print(f"  L1 error:      ∫|F_fin - F_init| dV = {L1_error:.6e}")
print(f"                 L1 / V₀ = {L1_rel_pct:.4f}%")
print(f"  Perimeter:     P₀={P0:.5f} → P_fin={P_fin:.5f}")
print(f"                 ΔP/P₀ = {((P_fin - P0) / P0 * 100):+.3f}%")
print("=" * 70)

# ── Save results ─────────────────────────────────────────────────
results = {
    "config": {
        "grid": [NX, NY, NZ], "dx": DX,
        "disk": {"R": R, "center": [DISK_CX, DISK_CY],
                 "slot_width": SLOT_WIDTH, "slot_depth": SLOT_DEPTH},
        "rotation": {"omega": OMEGA, "center": [ROT_CX, ROT_CY]},
        "cfl": CFL, "dt": DT, "n_steps": N_STEPS,
        "t_end": T_FULL_ROTATION,
    },
    "metrics": {
        "V0": V0, "V_final": V_fin, "dV_percent": dV_pct,
        "L1_error": L1_error, "L1_rel_percent": L1_rel_pct,
        "P0": P0, "P_final": P_fin,
        "dP_percent": (P_fin - P0) / P0 * 100,
        "wall_time_s": wall,
    },
}
with open(os.path.join(OUT, "test3_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: test3_results.json")

# ── Plots ────────────────────────────────────────────────────────
x_plt = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
y_plt = x_plt.copy()
X_plt, Y_plt = np.meshgrid(x_plt, y_plt, indexing="ij")

# Plot 1: evolution snapshots
n_show = min(6, len(snapshots))
idx_show = np.round(np.linspace(0, len(snapshots) - 1, n_show)).astype(int)
fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
if n_show == 1: axes = [axes]
for ax, i in zip(axes, idx_show):
    t_snap, F_snap = snapshots[i]
    F_disp = np.clip(F_snap, 0.0, 1.0)
    ax.contourf(X_plt, Y_plt, F_disp, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.8)
    ax.contour(X_plt, Y_plt, F_snap, levels=[0.5], colors="navy", linewidths=1.5)
    ax.plot(ROT_CX, ROT_CY, 'r+', markersize=12, markeredgewidth=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    angle = np.degrees(OMEGA * t_snap) % 360
    ax.set_title(f"t={t_snap:.2f} ({angle:.0f}°)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
fig.suptitle(
    f"Test 3: Zalesak Slotted Disk Rotation\n"
    f"Grid {NX}×{NY}×{NZ}, ω={OMEGA}, "
    f"L1 err={L1_rel_pct:.2f}%, ΔV={dV_pct:+.3f}%",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test3_evolution.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test3_evolution.png")

# Plot 2: initial vs final comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
F_init_slice = F_init[:, :, NZ // 2]
F_fin_slice_disp = np.clip(F_fin_slice, 0, 1)

# Left: initial
ax = axes[0]
ax.contourf(X_plt, Y_plt, F_init_slice, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.8)
ax.contour(X_plt, Y_plt, F_init_slice, levels=[0.5], colors="navy", linewidths=2)
ax.plot(ROT_CX, ROT_CY, 'r+', markersize=15, markeredgewidth=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title("Initial (t=0)", fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")

# Middle: final
ax = axes[1]
ax.contourf(X_plt, Y_plt, F_fin_slice_disp, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.8)
ax.contour(X_plt, Y_plt, F_fin_slice, levels=[0.5], colors="navy", linewidths=2)
ax.plot(ROT_CX, ROT_CY, 'r+', markersize=15, markeredgewidth=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title(f"Final (t={T_FULL_ROTATION:.2f}, one rotation)", fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")

# Right: overlay (both contours)
ax = axes[2]
ax.contour(X_plt, Y_plt, F_init_slice, levels=[0.5], colors="red",
           linewidths=2, linestyles="--")
ax.contour(X_plt, Y_plt, F_fin_slice, levels=[0.5], colors="navy",
           linewidths=2)
ax.plot(ROT_CX, ROT_CY, 'k+', markersize=15, markeredgewidth=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title("Overlay (red=initial, blue=final)", fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")

fig.suptitle(
    f"Zalesak Test: L1 error = {L1_rel_pct:.2f}%, ΔV/V₀ = {dV_pct:+.3f}%",
    fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test3_comparison.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test3_comparison.png")

# Plot 3: metrics over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
dV_arr = [(v - V0) / V0 * 100 for v in vol_hist]
dP_arr = [(p - P0) / P0 * 100 for p in per_hist]

ax1.plot(t_hist, dV_arr, 'r-o', ms=5, lw=2)
ax1.axhline(0, color='gray', ls='--')
ax1.set_xlabel("Time (one full rotation at t=6.28)")
ax1.set_ylabel("ΔV/V₀ (%)")
ax1.set_title("Volume Error")
ax1.grid(True, alpha=0.3)

ax2.plot(t_hist, dP_arr, 'b-s', ms=5, lw=2)
ax2.axhline(0, color='gray', ls='--')
ax2.set_xlabel("Time")
ax2.set_ylabel("ΔP/P₀ (%)")
ax2.set_title("Perimeter Change")
ax2.grid(True, alpha=0.3)

fig.suptitle("Zalesak Rotation Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test3_metrics.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved: test3_metrics.png")
