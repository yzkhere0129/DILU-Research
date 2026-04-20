#!/usr/bin/env python3
"""T2: Zalesak Slotted Disk Rotation — Eulerian PLIC benchmark.

Standard VOF validation: solid-body rotation of a slotted disk through
one full revolution (T = 2π s). Tests PLIC reconstruction under shear
(non-uniform velocity field).

Spec: vof-test-suite skill, T2 Standard Configuration (2D-extruded).
"""

from __future__ import annotations

import os
import sys
import time as timer

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x, sweep_flux_y, apply_flux_y,
)
from jax_laseram.vof.plic.diagnostics import compute_stats, format_stats

# ================================================================
# Parameters (T2 Standard Configuration)
# ================================================================
NX = NY = 100
NZ = 5
NH = 1
DX = DY = DZ = 1.0 / NX  # 0.01

DISK_CENTER = (0.50, 0.75)
DISK_R = 0.15
SLOT_WIDTH = 0.05  # 5 cells
SLOT_DEPTH = 0.25

OMEGA = 1.0  # rad/s
ROT_CENTER = (0.50, 0.50)
T_END = 2.0 * jnp.pi  # one full rotation
CFL = 0.45


# ================================================================
# Initialization: slotted disk with 4x4 sub-sampling
# ================================================================
def create_zalesak_disk():
    """Create slotted disk VOF field with 4x4 sub-sampling."""
    n_sub = 4
    x_c = np.linspace(DX / 2, 1 - DX / 2, NX)
    y_c = np.linspace(DY / 2, 1 - DY / 2, NY)

    sub_off = (np.arange(n_sub) + 0.5) / n_sub
    F_2d = np.zeros((NX, NY), dtype=np.float32)

    cx, cy = DISK_CENTER
    for i in range(NX):
        for j in range(NY):
            count = 0
            for si in range(n_sub):
                for sj in range(n_sub):
                    xs = x_c[i] - DX / 2 + DX * sub_off[si]
                    ys = y_c[j] - DY / 2 + DY * sub_off[sj]
                    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                    in_disk = r <= DISK_R
                    # Slot: centered at x=cx, width SLOT_WIDTH, extends
                    # from bottom of disk up by SLOT_DEPTH
                    in_slot = (
                        abs(xs - cx) < SLOT_WIDTH / 2
                        and ys < cy + DISK_R
                        and ys > cy + DISK_R - SLOT_DEPTH
                    )
                    if in_disk and not in_slot:
                        count += 1
            F_2d[i, j] = count / (n_sub * n_sub)

    # Embed into 3D with halos
    Nx_t = NX + 2 * NH
    Ny_t = NY + 2 * NH
    Nz_t = NZ + 2 * NH
    F = jnp.zeros((Nx_t, Ny_t, Nz_t), dtype=jnp.float32)
    for z in range(NH, NH + NZ):
        F = F.at[NH : NH + NX, NH : NH + NY, z].set(jnp.asarray(F_2d))
    return halo_update(F)


# ================================================================
# Halo: symmetry on all sides
# ================================================================
def halo_update(F):
    F = F.at[0].set(F[1])
    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1])
    F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1])
    F = F.at[:, :, -1].set(F[:, :, -2])
    return F


# ================================================================
# Velocity field: solid-body rotation at FACE centers
# ================================================================
def compute_rotation_velocity():
    """Return (u_face, v_face) for solid-body rotation.

    u_face[i, j, k] = velocity at x-face (i+1/2, j, k)
    v_face[i, j, k] = velocity at y-face (i, j+1/2, k)

    u = -ω(y - yc), v = +ω(x - xc)
    """
    Nx_t = NX + 2 * NH
    Ny_t = NY + 2 * NH
    Nz_t = NZ + 2 * NH
    xc, yc = ROT_CENTER

    # Cell centers
    x_cc = jnp.linspace(-NH * DX + DX / 2, 1 + NH * DX - DX / 2, Nx_t).astype(jnp.float32)
    y_cc = jnp.linspace(-NH * DY + DY / 2, 1 + NH * DY - DY / 2, Ny_t).astype(jnp.float32)

    # x-face centers: midpoints between cell i and i+1
    x_face_x = 0.5 * (x_cc[:-1] + x_cc[1:])  # (Nx_t-1,)
    y_face_x = y_cc  # (Ny_t,)
    _, Y_fx = jnp.meshgrid(x_face_x, y_face_x, indexing="ij")  # (Nx_t-1, Ny_t)
    u_face_2d = -OMEGA * (Y_fx - yc)

    # y-face centers
    x_face_y = x_cc  # (Nx_t,)
    y_face_y = 0.5 * (y_cc[:-1] + y_cc[1:])  # (Ny_t-1,)
    X_fy, _ = jnp.meshgrid(x_face_y, y_face_y, indexing="ij")  # (Nx_t, Ny_t-1)
    v_face_2d = OMEGA * (X_fy - xc)

    # Broadcast to 3D
    u_face = jnp.broadcast_to(u_face_2d[:, :, None], (Nx_t - 1, Ny_t, Nz_t)).astype(jnp.float32)
    v_face = jnp.broadcast_to(v_face_2d[:, :, None], (Nx_t, Ny_t - 1, Nz_t)).astype(jnp.float32)
    return u_face, v_face


# ================================================================
# One PLIC Strang step with array velocity
# ================================================================
def plic_step_rotation(F, u_face, v_face, dt):
    """Strang split x/2 → y → x/2 with spatially-varying velocity."""

    def subsweep_x(F, dt_sub):
        F = halo_update(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_x(F, nx, ny, nz, C, u_face, dt_sub, DX, DY, DZ)
        F = apply_flux_x(F, flux, DX, DY, DZ)
        return halo_update(F)

    def subsweep_y(F, dt_sub):
        F = halo_update(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_y(F, nx, ny, nz, C, v_face, dt_sub, DX, DY, DZ)
        F = apply_flux_y(F, flux, DX, DY, DZ)
        return halo_update(F)

    F = subsweep_x(F, dt / 2)
    F = subsweep_y(F, dt)
    F = subsweep_x(F, dt / 2)
    return F


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 60)
    print("T2: Zalesak Slotted Disk Rotation — Eulerian PLIC")
    print("=" * 60)
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX}, CFL={CFL}")
    print(f"Disk: R={DISK_R}, center={DISK_CENTER}, slot={SLOT_WIDTH}x{SLOT_DEPTH}")
    print(f"Rotation: ω={OMEGA} rad/s, center={ROT_CENTER}")
    print(f"T_end = {float(T_END):.4f} s (one rotation)", flush=True)

    # Init
    print("\nInitializing slotted disk (4x4 sub-sampling)...", end=" ", flush=True)
    t0 = timer.time()
    F_init = create_zalesak_disk()
    print(f"done in {timer.time()-t0:.1f}s", flush=True)

    # Velocity
    u_face, v_face = compute_rotation_velocity()
    max_vel = float(jnp.maximum(jnp.abs(u_face).max(), jnp.abs(v_face).max()))
    dt = CFL * DX / max_vel
    n_steps = int(np.ceil(float(T_END) / dt))
    dt = float(T_END) / n_steps
    print(f"max|u| = {max_vel:.4f}, dt = {dt:.6f}, n_steps = {n_steps}")

    cell_vol = DX * DY * DZ
    stats0 = compute_stats(F_init, NH, cell_vol)
    V0 = stats0.volume
    print(f"V0 = {V0:.10e}, n_interface = {stats0.n_interface}", flush=True)

    # JIT compile
    print("\nJIT compiling...", end=" ", flush=True)
    jit_step = jax.jit(lambda F: plic_step_rotation(F, u_face, v_face, dt))
    t0 = timer.time()
    F_warmup = jit_step(F_init)
    jax.block_until_ready(F_warmup)
    print(f"done in {timer.time()-t0:.1f}s", flush=True)

    # Time loop
    print(f"\nRunning {n_steps} steps...", flush=True)
    F = F_init
    t_wall0 = timer.time()
    save_every = max(1, n_steps // 10)

    for step in range(1, n_steps + 1):
        F = jit_step(F)
        if step % save_every == 0 or step == n_steps:
            jax.block_until_ready(F)
            stats = compute_stats(F, NH, cell_vol)
            elapsed = timer.time() - t_wall0
            print(
                f"  step {step:5d}/{n_steps} t={step*dt:.3f}  "
                f"{format_stats(stats, v_ref=V0)}  wall={elapsed:.1f}s",
                flush=True,
            )

    wall = timer.time() - t_wall0
    print(f"\nDone. {n_steps} steps in {wall:.1f}s ({wall/n_steps*1000:.1f} ms/step)")

    # ================================================================
    # Metrics
    # ================================================================
    F_fin = np.asarray(F)
    F_ini = np.asarray(F_init)
    z_mid = NH + NZ // 2

    F_fin_2d = F_fin[NH : NH + NX, NH : NH + NY, z_mid]
    F_ini_2d = F_ini[NH : NH + NX, NH : NH + NY, z_mid]

    # L1 error
    L1 = float(np.sum(np.abs(F_fin_2d - F_ini_2d)) * DX * DY)
    V_disk = float(np.sum(F_ini_2d) * DX * DY)
    L1_rel = L1 / V_disk * 100

    # Volume error
    V_fin = float(np.sum(F_fin_2d) * DX * DY)
    V_err = abs(V_fin - V_disk) / V_disk * 100

    # Perimeter (marching squares)
    def perimeter(vf):
        h = ((vf[:, :-1] - 0.5) * (vf[:, 1:] - 0.5)) < 0
        v = ((vf[:-1, :] - 0.5) * (vf[1:, :] - 0.5)) < 0
        return float(np.sum(h) * DY + np.sum(v) * DX)

    P_ini = perimeter(F_ini_2d)
    P_fin = perimeter(F_fin_2d)
    P_err = abs(P_fin - P_ini) / P_ini * 100 if P_ini > 0 else 0

    print("\n" + "=" * 60)
    print("  VOF Test Report: T2 Zalesak Slotted Disk")
    print("=" * 60)
    print(f"  Grid:       {NX} x {NY} x {NZ}")
    print(f"  Method:     Eulerian PLIC (Phase B analytic + Newton)")
    print(f"  Hardware:   {jax.devices()[0]}")
    print(f"  Wall time:  {wall:.1f}s ({wall/n_steps*1000:.1f} ms/step)")
    print(f"\n  METRICS:")
    print(f"    L1 error:         {L1_rel:.2f}%    {'PASS' if L1_rel < 10 else 'FAIL'} (<10%)")
    print(f"    Volume error:     {V_err:.4f}%    {'PASS' if V_err < 1 else 'FAIL'} (<1%)")
    print(f"    Perimeter error:  {P_err:.2f}%    {'PASS' if P_err < 100 else 'FAIL'} (<100%)")
    verdict = "EXCELLENT" if L1_rel < 3 else ("PASS" if L1_rel < 10 else "FAIL")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 60)

    # ================================================================
    # Plot: initial vs final overlay
    # ================================================================
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    x = np.linspace(DX / 2, 1 - DX / 2, NX)
    y = np.linspace(DY / 2, 1 - DY / 2, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    ax = axes[0]
    ax.contourf(X, Y, F_ini_2d, levels=np.linspace(0, 1, 21), cmap="Blues")
    ax.contour(X, Y, F_ini_2d, levels=[0.5], colors="k", linewidths=1.5)
    ax.set_title("Initial (t=0)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.contourf(X, Y, F_fin_2d, levels=np.linspace(0, 1, 21), cmap="Blues")
    ax.contour(X, Y, F_fin_2d, levels=[0.5], colors="r", linewidths=1.5)
    ax.set_title(f"Final (t={float(T_END):.2f}s, one rotation)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.contour(X, Y, F_ini_2d, levels=[0.5], colors="b", linewidths=2, label="Initial")
    ax.contour(X, Y, F_fin_2d, levels=[0.5], colors="r", linewidths=2, label="Final")
    ax.set_title(f"Overlay: L1={L1_rel:.2f}%, V_err={V_err:.4f}%", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Zalesak Slotted Disk — Eulerian PLIC ({NX}x{NY}, {n_steps} steps)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(_HERE, "results", "zalesak_result.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
