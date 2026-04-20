#!/usr/bin/env python3
"""Barkhudarov Test 1 (v3): 2D circular droplet advection using 3D native (GPU+CPU hybrid).

Uses jax.pure_callback → libhexboxclip.so (CPU/OpenMP) for overlay hot loop.
This is the "GPU主导大规模流场张量运算、CPU接管复杂分支几何裁剪" version.

Strictly follows Barkhudarov (2004) §4.1 specifications:
  - Droplet diameter D = 0.1, radius R = 0.05
  - Grid: cell_size = D/10 = 0.01 → 100×100 cells on [0,1]×[0,1]
  - Travel distance: 5D = 0.5 at unit speed (|U| = 1.0)
  - Angles: 0°, 6°, 30°, 45° with uniform velocity field
  - Validation targets: |ΔV/V| < 0.0002%, |ΔP/P| < 1%

Reference:
  Barkhudarov, M.R. (2004). "Lagrangian VOF advection Method for FLOW-3D®"
  Flow Science, Inc.

Usage:
    python barkhudarov_test1_advection_v3.py
"""

import sys, os, time, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d.move_3d import lagrangian_move_faces_3d
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native,
    is_native_available,
)

# ── Output directory ─────────────────────────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Barkhudarov paper configuration ──────────────────────────────
D = 0.10
R = D / 2.0
CELL_SIZE = D / 10
NX = 100
DOMAIN = 1.0
TRAVEL_DIST = 5 * D
SPEED = 1.0
T_END = TRAVEL_DIST / SPEED

# Use nz=1 for pseudo-2D
NZ = 1
DX = DOMAIN / NX
DY = DOMAIN / NX
# Use very small dz to create thin layer effect (pseudo-2D)
# This ensures z cells are much thinner than x/y cells
DZ = DX * 0.001  # dz = 0.00001, 1000:1 aspect ratio

ANGLES = [
    (0, "0°", (0.15, 0.50)),
    (6, "6°", (0.15, 0.40)),
    (30, "30°", (0.15, 0.35)),
    (45, "45°", (0.20, 0.20)),
]

CFL = 0.45


def create_droplet_vof(grid, cx, cy, r):
    """Initialize VOF field with a circular droplet in 3D array (nz=1)."""
    nh, nx, ny, nz = grid.nh, grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy

    x = np.linspace(dx / 2, DOMAIN - dx / 2, nx)
    y = np.linspace(dy / 2, DOMAIN - dy / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    hd = np.sqrt(2) * dx / 2
    F_int = np.where(
        dist + hd <= r,
        1.0,
        np.where(dist - hd >= r, 0.0, np.clip((r - dist + hd) / (2 * hd), 0.0, 1.0)),
    )

    shape = (nx + 2 * nh, ny + 2 * nh, nz + 2 * nh, 1)
    F = np.zeros(shape)
    F[nh : nh + nx, nh : nh + ny, nh : nh + nz, 0] = F_int[..., None]
    F = jnp.array(F).astype(jnp.float32)
    return F


def make_uniform_velocity(grid, angle_deg):
    """Create uniform velocity field at specified angle."""
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    Ntx, Nty, Ntz = nx + 2 * nh, ny + 2 * nh, nz + 2 * nh

    theta = np.radians(angle_deg)
    ux = np.cos(theta) * SPEED
    uy = np.sin(theta) * SPEED

    u_face = jnp.full((Ntx - 1, Nty, Ntz), ux, jnp.float32)
    v_face = jnp.full((Ntx, Nty - 1, Ntz), uy, jnp.float32)
    w_face = jnp.zeros((Ntx, Nty, Ntz - 1), jnp.float32)

    return u_face, v_face, w_face


def compute_perimeter(F_int, dx, dy):
    """Compute interface perimeter from 2D slice of VOF field."""
    v = np.array(F_int)
    hc = ((v[:-1, :] - 0.5) * (v[1:, :] - 0.5)) < 0
    vc = ((v[:, :-1] - 0.5) * (v[:, 1:] - 0.5)) < 0
    return float(np.sum(hc) * dy + np.sum(vc) * dx)


def compute_volume(F_int, dx, dy):
    """Compute total fluid volume from 2D slice of VOF field."""
    return float(np.sum(F_int) * dx * dy)


# ── Main test loop ────────────────────────────────────────────────
print("=" * 70)
print("  Barkhudarov Test 1: 2D Droplet Advection (Native 3D → GPU+CPU)")
print(f"  Native overlay available: {is_native_available()}")
print("=" * 70)
print(f"  Droplet: D={D}, R={R}, cell_size={CELL_SIZE} ({NX} cells/diameter)")
print(f"  Grid: {NX}×{NX}×{NZ} on [{DOMAIN}×{DOMAIN}×{DZ}]")
print(f"  Travel: {TRAVEL_DIST} ({5} diameters) at |U|={SPEED}")
print(f"  Targets: |ΔV/V| < 0.0002%, |ΔP/P| < 1%")
print("=" * 70)

results = {}

for angle_deg, label, (cx0, cy0) in ANGLES:
    print(f"\n{'─' * 60}")
    print(f"  Angle = {label}")

    grid = GridInfo(
        nx=NX,
        ny=NX,
        nz=NZ,
        nh=1,
        dx=DX,
        dy=DY,
        dz=DZ,
        x_range=(0.0, DOMAIN),
        y_range=(0.0, DOMAIN),
        z_range=(0.0, DZ),
    )
    dx, dy = grid.dx, grid.dy
    nh = grid.nh

    F = create_droplet_vof(grid, cx0, cy0, R)
    u_face, v_face, w_face = make_uniform_velocity(grid, angle_deg)

    dt = CFL * dx / SPEED
    n_steps = int(np.ceil(T_END / dt))
    dt = T_END / n_steps

    print(f"  Grid: {grid.nx}×{grid.ny}×{grid.nz}, dx={dx:.6f}")
    print(f"  dt={dt:.6f}, n_steps={n_steps}")

    ic = slice(nh, -nh)
    F_int0 = np.array(F[ic, ic, ic, 0])
    V0 = compute_volume(F_int0, dx, dy)
    P0 = compute_perimeter(F_int0, dx, dy)

    print(f"  V₀={V0:.8f}, P₀={P0:.5f}")

    # Step function using native overlay
    def step(F):
        xv, yv, zv = lagrangian_move_faces_3d(u_face, v_face, w_face, grid, dt)
        Fn = overlay_lagrangian_3d_native(F, xv, yv, zv, grid)
        Fo = jnp.zeros_like(F)
        Fo = Fo.at[nh : nh + NX, nh : nh + NX, nh : nh + NZ, 0].set(Fn)
        return Fo

    # JIT warmup
    _ = step(F)
    print("  JIT warmup done")

    t0_wall = time.time()
    snapshots = [(0.0, F_int0.copy())]
    vol_history = [V0]
    per_history = [P0]

    for step_idx in range(n_steps):
        F = step(F)

        if (step_idx + 1) % max(1, n_steps // 4) == 0 or step_idx == n_steps - 1:
            Fi = np.array(F[ic, ic, ic, 0])
            snapshots.append(((step_idx + 1) * dt, Fi.copy()))
            V = compute_volume(Fi, dx, dy)
            P = compute_perimeter(Fi, dx, dy)
            vol_history.append(V)
            per_history.append(P)
            dV = abs(V - V0) / V0
            dP = (P - P0) / P0 * 100
            frac = (step_idx + 1) / n_steps
            print(
                f"    [{frac * 100:3.0f}%] step {step_idx + 1}/{n_steps}  "
                f"|ΔV/V|={dV:.2e}  ΔP/P={dP:+.3f}%"
            )

    wall_time = time.time() - t0_wall

    F_final_int = np.array(F[ic, ic, ic, 0])
    V_final = compute_volume(F_final_int, dx, dy)
    P_final = compute_perimeter(F_final_int, dx, dy)
    dV_final = abs(V_final - V0) / V0
    dP_final = (P_final - P0) / P0 * 100

    theta = np.radians(angle_deg)
    cx_final = cx0 + np.cos(theta) * TRAVEL_DIST
    cy_final = cy0 + np.sin(theta) * TRAVEL_DIST

    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Final: V={V_final:.8f}  P={P_final:.5f}")
    print(
        f"  |ΔV/V| = {dV_final * 100:.6f}%  "
        f"{'✓ PASS' if dV_final * 100 < 0.0002 else '✗ FAIL'}"
    )
    print(
        f"  ΔP/P   = {dP_final:+.4f}%  {'✓ PASS' if abs(dP_final) < 1.0 else '✗ FAIL'}"
    )

    results[label] = {
        "angle": angle_deg,
        "cx0": cx0,
        "cy0": cy0,
        "cx_final": cx_final,
        "cy_final": cy_final,
        "V0": V0,
        "P0": P0,
        "V_final": V_final,
        "P_final": P_final,
        "dV": dV_final,
        "dP": dP_final,
        "wall_time": wall_time,
        "F_final": F_final_int,
    }

# ── Save results to JSON ─────────────────────────────────────────
results_json = {}
for label, r in results.items():
    results_json[label] = {
        "angle_deg": r["angle"],
        "initial_center": [r["cx0"], r["cy0"]],
        "final_center": [r["cx_final"], r["cy_final"]],
        "V0": r["V0"],
        "P0": r["P0"],
        "V_final": r["V_final"],
        "P_final": r["P_final"],
        "dV_percent": r["dV"] * 100,
        "dP_percent": r["dP"],
        "wall_time_s": r["wall_time"],
        "volume_pass": r["dV"] * 100 < 0.0002,
        "perimeter_pass": abs(r["dP"]) < 1.0,
    }

with open(os.path.join(OUT, "test1_results_v3.json"), "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nSaved: test1_results_v3.json")

# ── Plot final interface shapes ───────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
x_plt = np.linspace(dx / 2, DOMAIN - dx / 2, NX)
y_plt = x_plt.copy()
X_plt, Y_plt = np.meshgrid(x_plt, y_plt, indexing="ij")
theta_c = np.linspace(0, 2 * np.pi, 100)

for ax, (label, r) in zip(axes, results.items()):
    ax.contourf(
        X_plt,
        Y_plt,
        r["F_final"],
        levels=np.linspace(0, 1, 21),
        cmap="Blues",
        alpha=0.7,
    )
    ax.contour(X_plt, Y_plt, r["F_final"], levels=[0.5], colors="navy", lw=2)

    ax.plot(
        r["cx_final"] + R * np.cos(theta_c),
        r["cy_final"] + R * np.sin(theta_c),
        "g--",
        lw=1.5,
        alpha=0.8,
        label="Exact",
    )
    ax.plot(
        r["cx0"] + R * np.cos(theta_c),
        r["cy0"] + R * np.sin(theta_c),
        "r:",
        lw=1,
        alpha=0.5,
        label="Initial",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(
        f"Angle = {label}\n|ΔV/V| = {r['dV'] * 100:.2e}%  ΔP/P = {r['dP']:+.3f}%",
        fontsize=11,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(
    "Barkhudarov Test 1 (v3): 2D Droplet Advection — Native 3D (GPU+CPU)\n"
    "(Barkhudarov 2004, §4.1)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "test1_final_interfaces_v3.png"), dpi=200, bbox_inches="tight"
)
plt.close()
print("Saved: test1_final_interfaces_v3.png")

# ── Summary table ────────────────────────────────────────────────
print("\n" + "=" * 70)
print(
    f"  {'Angle':<8} {'|ΔV/V| (%)':>12} {'ΔP/P (%)':>12} "
    f"{'Vol Pass':>10} {'Peri Pass':>10}"
)
print("  " + "-" * 56)
all_vol_pass = True
all_peri_pass = True
for label, r in results.items():
    vp = "✓ PASS" if r["dV"] * 100 < 0.0002 else "✗ FAIL"
    pp = "✓ PASS" if abs(r["dP"]) < 1.0 else "✗ FAIL"
    if not vp.startswith("✓"):
        all_vol_pass = False
    if not pp.startswith("✓"):
        all_peri_pass = False
    print(f"  {label:<8} {r['dV'] * 100:>12.6f} {r['dP']:>12.4f} {vp:>10} {pp:>10}")
print("  " + "-" * 56)
print(f"  Target: |ΔV/V| < 0.0002%, |ΔP/P| < 1%")
print(
    f"  Overall: Volume = {'ALL PASS ✓' if all_vol_pass else 'SOME FAIL ✗'}, "
    f"Perimeter = {'ALL PASS ✓' if all_peri_pass else 'SOME FAIL ✗'}"
)
print("=" * 70)
