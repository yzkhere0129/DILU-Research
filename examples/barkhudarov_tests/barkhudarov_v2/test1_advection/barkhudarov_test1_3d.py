#!/usr/bin/env python3
"""Barkhudarov Test 1: 3D circular droplet advection at 45°.

Paper: Barkhudarov (2004) §4.1
Grid: cell_size = D/10 (10 cells/diameter), nz=5, dz=dx (uniform cubic cells)
Droplet: cylinder along z-axis, travels 5D at 45°
Pipeline: 3D native C/OpenMP (no PLIC in overlay — better volume conservation)
Metrics: volume error < 0.0002%, perimeter change < 1%
"""
import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Barkhudarov paper configuration ─────────────────────────────────
D = 0.10
R = D / 2.0
NX = 100   # 10 cells/diameter per paper
NZ = 5     # minimal 3D: cylinder uniform in z
DOMAIN_XY = 1.0
DX = DOMAIN_XY / NX   # 0.01
DOMAIN_Z = NZ * DX
TRAVEL_DIST = 5 * D
SPEED = 1.0
T_END = TRAVEL_DIST / SPEED
CFL = 0.45

CX0, CY0 = 0.20, 0.20
ANGLE_DEG = 45


def create_droplet_cylinder(grid, cx, cy, r):
    nh, nx, ny, nz = grid.nh, grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy

    x = np.linspace(dx / 2, DOMAIN_XY - dx / 2, nx)
    y = np.linspace(dy / 2, DOMAIN_XY - dy / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    hd = np.sqrt(2) * dx / 2
    F_2d = np.where(
        dist + hd <= r, 1.0,
        np.where(dist - hd >= r, 0.0,
                 np.clip((r - dist + hd) / (2 * hd), 0.0, 1.0)),
    )

    shape = (nx + 2 * nh, ny + 2 * nh, nz + 2 * nh, 1)
    F = np.zeros(shape)
    F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0] = F_2d[:, :, None]
    F[:nh] = F[nh:nh+1]; F[-nh:] = F[-nh-1:-nh]
    F[:, :nh] = F[:, nh:nh+1]; F[:, -nh:] = F[:, -nh-1:-nh]
    F[:, :, :nh] = F[:, :, nh:nh+1]; F[:, :, -nh:] = F[:, :, -nh-1:-nh]
    return jnp.array(F, dtype=jnp.float32)


def make_uniform_velocity(grid, angle_deg):
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    Ntx, Nty, Ntz = nx + 2*nh, ny + 2*nh, nz + 2*nh
    theta = np.radians(angle_deg)
    ux = np.cos(theta) * SPEED
    uy = np.sin(theta) * SPEED
    u_face = jnp.full((Ntx-1, Nty, Ntz), ux, jnp.float32)
    v_face = jnp.full((Ntx, Nty-1, Ntz), uy, jnp.float32)
    w_face = jnp.zeros((Ntx, Nty, Ntz-1), jnp.float32)
    return u_face, v_face, w_face


def compute_perimeter(F_2d, dx, dy):
    """Proper perimeter via contour arc-length (0.5 isoline).

    Uses matplotlib to extract the iso-contour then sums segment lengths.
    Falls back to gradient integral if matplotlib fails.
    """
    try:
        fig, ax = plt.subplots()
        cs = ax.contour(F_2d.T, levels=[0.5])
        plt.close(fig)
        total = 0.0
        for path in cs.get_paths():
            verts = path.vertices  # pixel coords (col, row)
            # Convert pixel → physical (verts are in data coords 0..NX-1)
            pts = verts * np.array([dx, dy])
            diffs = np.diff(pts, axis=0)
            total += float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        return total if total > 0 else _perimeter_gradient(F_2d, dx, dy)
    except Exception:
        return _perimeter_gradient(F_2d, dx, dy)


def _perimeter_gradient(F_2d, dx, dy):
    """Fallback: sum |∇F| * cell_area ≈ interface length."""
    gx = np.gradient(F_2d, dx, axis=0)
    gy = np.gradient(F_2d, dy, axis=1)
    return float(np.sum(np.hypot(gx, gy)) * dx * dy)


def compute_volume_3d(F_int, dx, dy, dz):
    return float(np.sum(F_int) * dx * dy * dz)


def _make_step_fn(grid, u_face, v_face, w_face, dt):
    def step(F):
        return advect_vof_lagrangian_3d(F, u_face, v_face, w_face, grid, dt)
    return step


# ── Main ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  Barkhudarov Test 1: 3D Droplet Advection at 45° (PLIC overlay, no-clip)")
print("=" * 70)
print(f"  Droplet: D={D}, R={R}, cell_size={DX:.4f} (10 cells/diameter)")
print(f"  Grid: {NX}×{NX}×{NZ}, dx=dy=dz={DX:.4f}")
print(f"  Domain: [{DOMAIN_XY}×{DOMAIN_XY}×{DOMAIN_Z:.3f}]")
print(f"  Travel: {TRAVEL_DIST} (5 diameters) at |U|={SPEED}")
print(f"  Angle: {ANGLE_DEG}°")
print(f"  Targets: |ΔV/V| < 0.0002%, |ΔP/P| < 1%")
print("=" * 70)

grid = GridInfo(
    nx=NX, ny=NX, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN_XY),
    y_range=(0.0, DOMAIN_XY),
    z_range=(0.0, DOMAIN_Z),
)
dx, dy, dz = grid.dx, grid.dy, grid.dz
nh = grid.nh

F = create_droplet_cylinder(grid, CX0, CY0, R)
u_face, v_face, w_face = make_uniform_velocity(grid, ANGLE_DEG)

dt = CFL * dx / SPEED
n_steps = int(np.ceil(T_END / dt))
dt = T_END / n_steps

print(f"  dt={dt:.6f}, n_steps={n_steps}")

ic_x = slice(nh, nh + NX)
ic_y = slice(nh, nh + NX)
ic_z = slice(nh, nh + NZ)
z_mid = nh + NZ // 2

F_int0 = np.array(F[ic_x, ic_y, ic_z, 0])
F_slice0 = np.array(F[ic_x, ic_y, z_mid, 0])
V0 = compute_volume_3d(F_int0, dx, dy, dz)
P0 = compute_perimeter(F_slice0, dx, dy)
P_analytical = np.pi * D
print(f"  V₀={V0:.10f} (analytic={np.pi*R**2*DOMAIN_Z:.10f})")
print(f"  P₀={P0:.6f} (analytic πD={P_analytical:.6f})")

step_fn = _make_step_fn(grid, u_face, v_face, w_face, dt)

print("  JIT compiling...", end=" ", flush=True)
t_jit = time.time()
_ = step_fn(F)
print(f"{time.time() - t_jit:.1f}s")

t0_wall = time.time()
vol_history = [V0]
per_history = [P0]

for step_idx in range(n_steps):
    F = step_fn(F)

    if (step_idx + 1) % max(1, n_steps // 4) == 0 or step_idx == n_steps - 1:
        Fi_3d = np.array(F[ic_x, ic_y, ic_z, 0])
        Fi_slice = np.array(F[ic_x, ic_y, z_mid, 0])
        V = compute_volume_3d(Fi_3d, dx, dy, dz)
        P = compute_perimeter(Fi_slice, dx, dy)
        vol_history.append(V)
        per_history.append(P)
        dV = abs(V - V0) / V0 * 100
        dP = (P - P0) / P0 * 100
        frac = (step_idx + 1) / n_steps
        print(
            f"    [{frac*100:3.0f}%] step {step_idx+1}/{n_steps}  "
            f"|ΔV/V|={dV:.2e}%  ΔP/P={dP:+.3f}%",
            flush=True,
        )

wall_time = time.time() - t0_wall

F_final_3d = np.array(F[ic_x, ic_y, ic_z, 0])
F_final_slice = np.array(F[ic_x, ic_y, z_mid, 0])
V_final = compute_volume_3d(F_final_3d, dx, dy, dz)
P_final = compute_perimeter(F_final_slice, dx, dy)
dV_final = abs(V_final - V0) / V0 * 100
dP_final = (P_final - P0) / P0 * 100

theta = np.radians(ANGLE_DEG)
cx_final = CX0 + np.cos(theta) * TRAVEL_DIST
cy_final = CY0 + np.sin(theta) * TRAVEL_DIST

print(f"\n  Wall time: {wall_time:.1f}s")
print(f"  Final: V={V_final:.10f}  P={P_final:.6f}")
print(
    f"  |ΔV/V| = {dV_final:.6f}%  "
    f"{'✓ PASS' if dV_final < 0.0002 else '✗ FAIL'}"
)
print(
    f"  ΔP/P   = {dP_final:+.4f}%  "
    f"{'✓ PASS' if abs(dP_final) < 1.0 else '✗ FAIL'}"
)

# ── Save results JSON ──────────────────────────────────────────────
results_json = {
    "45°": {
        "angle_deg": ANGLE_DEG,
        "grid_nx_ny_nz": [NX, NX, NZ],
        "dx": DX,
        "n_steps": n_steps,
        "overlay_mode": "plic_centroid_clip",
        "V0": V0, "P0": P0, "P_analytical": P_analytical,
        "V_final": V_final, "P_final": P_final,
        "dV_percent": dV_final,
        "dP_percent": dP_final,
        "wall_time_s": wall_time,
        "volume_pass": dV_final < 0.0002,
        "perimeter_pass": abs(dP_final) < 1.0,
    }
}
with open(os.path.join(OUT, "test1_results_3d.json"), "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nSaved: test1_results_3d.json")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
x_plt = np.linspace(dx / 2, DOMAIN_XY - dx / 2, NX)
y_plt = x_plt.copy()
X_plt, Y_plt = np.meshgrid(x_plt, y_plt, indexing="ij")
theta_c = np.linspace(0, 2 * np.pi, 200)

ax.contourf(X_plt, Y_plt, F_final_slice,
            levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.7)
ax.contour(X_plt, Y_plt, F_final_slice, levels=[0.5], colors="navy", linewidths=2)
ax.plot(cx_final + R*np.cos(theta_c), cy_final + R*np.sin(theta_c),
        "g--", lw=1.5, alpha=0.8, label="Exact")
ax.plot(CX0 + R*np.cos(theta_c), CY0 + R*np.sin(theta_c),
        "r:", lw=1, alpha=0.5, label="Initial")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_title(
    f"45°  |ΔV/V|={dV_final:.2e}%  ΔP/P={dP_final:+.3f}%\n"
    f"(perimeter via contour arc-length)",
    fontsize=11,
)
ax.legend(fontsize=8, loc="upper left")
fig.suptitle(
    "Barkhudarov Test 1: 3D Droplet Advection — Native (no-PLIC overlay)\n"
    f"Grid {NX}×{NX}×{NZ}, dx={DX:.4f}",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "test1_final_interfaces_3d.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: test1_final_interfaces_3d.png")
