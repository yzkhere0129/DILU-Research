#!/usr/bin/env python3
"""End-to-end 45° droplet advection test using Barkhudarov Lagrangian VOF.

Chains Step 1 (PLIC) → Step 2 (Move) → Step 3 (Overlay) for each timestep.
Compares with the previous Strang-splitting + THINC result.

Target: perimeter distortion < 1% (vs 5.0% for Strang splitting).
"""

import sys
import time as timer
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

jax.config.update("jax_enable_x64", True)

from jax_laseram.grid import create_grid
from jax_laseram.vof.lagrangian import advect_vof_lagrangian
from jax_laseram.vof.lagrangian.reconstruction import (
    compute_plic_normals,
    compute_intercept_C,
    plic_line_segments_batch,
)


def create_droplet(grid, center=(0.2, 0.2), radius=0.05):
    """Create circular droplet VOF field with ghost cells."""
    nh = grid.nh
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    x = jnp.linspace(dx / 2, 1.0 - dx / 2, nx)
    y = jnp.linspace(dy / 2, 1.0 - dy / 2, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    cx, cy = center

    # Sharp geometric VOF: compute exact circle-rectangle intersection fraction.
    # Cells fully inside circle: F=1. Cells fully outside: F=0.
    # Interface cells: F = approximate area fraction.
    r_center = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # Distance from cell center to circle boundary
    # If cell center + half-diagonal is inside circle: F=1 (fully inside)
    # If cell center - half-diagonal is outside circle: F=0 (fully outside)
    half_diag = jnp.sqrt(dx**2 + dy**2) * 0.5
    F_int = jnp.where(
        r_center + half_diag <= radius, 1.0,  # fully inside
        jnp.where(
            r_center - half_diag >= radius, 0.0,  # fully outside
            jnp.clip((radius - r_center + half_diag) / (2 * half_diag), 0.0, 1.0)  # approximate interface
        )
    )

    shape = (nx + 2 * nh, ny + 2 * nh, 1)
    F = jnp.zeros(shape).at[nh : nh + nx, nh : nh + ny, :].set(F_int[:, :, None])
    # BCs
    F = F.at[:nh, :, :].set(F[nh : nh + 1, :, :])
    F = F.at[-nh:, :, :].set(F[-nh - 1 : -nh, :, :])
    F = F.at[:, :nh, :].set(F[:, nh : nh + 1, :])
    F = F.at[:, -nh:, :].set(F[:, -nh - 1 : -nh, :])
    return F


def constant_velocity_faces(grid):
    """Constant u=v=1.0 face velocities."""
    nh = grid.nh
    Ntot_x = grid.nx + 2 * nh
    Ntot_y = grid.ny + 2 * nh
    u_face = jnp.ones((Ntot_x - 1, Ntot_y, 1))
    v_face = jnp.ones((Ntot_x, Ntot_y - 1, 1))
    return u_face, v_face


def compute_perimeter(F_interior, dx, dy):
    """Interface perimeter via marching-squares edge counting."""
    vf = np.array(F_interior)
    h_cross = ((vf[:-1, :] - 0.5) * (vf[1:, :] - 0.5)) < 0
    v_cross = ((vf[:, :-1] - 0.5) * (vf[:, 1:] - 0.5)) < 0
    return float(np.sum(h_cross) * dy + np.sum(v_cross) * dx)


def run_test(nx=100, beta=2.0, cfl=0.5):
    """Run the 45° droplet advection with Lagrangian VOF."""
    grid = create_grid(nx, nx, nz=1, x_range=(0.0, 1.0), y_range=(0.0, 1.0), nh=1)
    dx, dy = grid.dx, grid.dy
    nh = grid.nh
    ic = slice(nh, -nh)

    F = create_droplet(grid)
    u_face, v_face = constant_velocity_faces(grid)

    dt = cfl * dx / 1.0  # max velocity component = 1.0
    t_end = 0.3536
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps

    dV = dx * dy
    F_int = np.array(F[ic, ic, 0])
    V0 = float(np.sum(F_int) * dV)
    P0 = compute_perimeter(F_int, dx, dy)

    print(f"45° Droplet — Lagrangian VOF: {nx}×{nx}, CFL={cfl}", flush=True)
    print(f"  dx={dx:.4f}, dt={dt:.6f}, n_steps={n_steps}", flush=True)
    print(f"  V₀={V0:.10f}, P₀={P0:.6f}", flush=True)

    # JIT warmup
    _ = advect_vof_lagrangian(F, u_face, v_face, grid, dt)

    t0 = timer.time()
    save_every = max(1, n_steps // 10)
    snapshots = [(0.0, np.array(F[ic, ic, 0]))]
    times, volumes, perimeters = [0.0], [V0], [P0]

    for step in range(n_steps):
        F = advect_vof_lagrangian(F, u_face, v_face, grid, dt)
        t = (step + 1) * dt

        F_int = np.array(F[ic, ic, 0])
        V = float(np.sum(F_int) * dV)
        P = compute_perimeter(F_int, dx, dy)
        times.append(t)
        volumes.append(V)
        perimeters.append(P)

        if (step + 1) % save_every == 0:
            snapshots.append((t, F_int.copy()))
            dV_rel = abs(V - V0) / V0
            elapsed = timer.time() - t0
            F_min, F_max = float(F_int.min()), float(F_int.max())
            print(
                f"  step {step + 1:5d}/{n_steps}, t={t:.4f}, "
                f"V={V:.10f}, dV/V={dV_rel:.2e}, "
                f"F∈[{F_min:.4f},{F_max:.4f}], "
                f"P={P:.4f}, {elapsed:.1f}s",
                flush=True,
            )

    wall = timer.time() - t0
    V_final = volumes[-1]
    P_final = perimeters[-1]
    dV_err = abs(V_final - V0) / V0
    dP_err = (P_final - P0) / P0 * 100

    print(f"\nDone in {wall:.1f}s", flush=True)
    print(f"Volume: {V0:.10f} → {V_final:.10f}, error={dV_err:.2e}", flush=True)
    print(f"Perimeter: {P0:.6f} → {P_final:.6f}, change={dP_err:+.4f}%", flush=True)

    if abs(dP_err) < 1.0:
        print(f"  WIN: Perimeter distortion < 1%!", flush=True)
    elif abs(dP_err) < 5.0:
        print(f"  GOOD: Better than Strang splitting (5%)", flush=True)
    else:
        print(f"  CHECK: Still above 5%", flush=True)

    return grid, F, snapshots, np.array(times), np.array(volumes), np.array(perimeters)


def plot_results(grid, F_final, snapshots, times, volumes, perimeters, nx):
    """Generate comparison plots."""
    nh = grid.nh
    ic = slice(nh, -nh)
    dx, dy = grid.dx, grid.dy
    x = np.linspace(dx / 2, 1 - dx / 2, nx)
    y = np.linspace(dy / 2, 1 - dy / 2, nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # ---- Plot 1: Phase evolution ----
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(2.8 * n_snaps, 3.0))

    for ax, (t, F_snap) in zip(axes, snapshots):
        ax.contourf(X, Y, F_snap, levels=np.linspace(0, 1, 21), cmap="Blues", alpha=0.8)
        ax.contour(X, Y, F_snap, levels=[0.5], colors="k", linewidths=1.0)
        # Exact position
        cx, cy = 0.2 + t, 0.2 + t
        theta = np.linspace(0, 2 * np.pi, 60)
        ax.plot(
            cx + 0.05 * np.cos(theta),
            cy + 0.05 * np.sin(theta),
            "g--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:.3f}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("y", fontsize=9)

    fig.suptitle(
        "45° Droplet — Lagrangian VOF (Barkhudarov)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    out = f"/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/e2e_45deg_phase_{nx}"
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}.png")

    # ---- Plot 2: Overlay comparison (initial vs final) ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_facecolor("#F5F5F5")

    F0 = snapshots[0][1]
    Ff = snapshots[-1][1]
    t_final = snapshots[-1][0]

    ax.contour(
        X, Y, F0, levels=[0.5], colors="blue", linewidths=2, label=f"t=0", zorder=2
    )
    ax.contour(
        X,
        Y,
        Ff,
        levels=[0.5],
        colors="red",
        linewidths=2,
        label=f"t={t_final:.3f} (Lagrangian)",
        zorder=3,
    )

    # Exact final circle
    theta = np.linspace(0, 2 * np.pi, 100)
    cx, cy = 0.2 + t_final, 0.2 + t_final
    ax.plot(
        cx + 0.05 * np.cos(theta),
        cy + 0.05 * np.sin(theta),
        "g--",
        linewidth=2,
        alpha=0.8,
        label="Exact",
        zorder=1,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.2, ls="--")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        "Interface: Initial vs Final (Lagrangian VOF)", fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    out2 = f"/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/e2e_45deg_overlay_{nx}"
    plt.savefig(f"{out2}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out2}.png")

    # ---- Plot 3: Metrics ----
    V0 = volumes[0]
    P0 = perimeters[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(times, volumes / V0 * 100, "r-o", ms=3, lw=1.5)
    ax1.axhline(100, color="gray", ls="--", alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Volume / V₀ (%)")
    ax1.set_title("Volume Conservation")
    ax1.grid(True, alpha=0.3, ls="--")

    ax2.plot(times, perimeters / P0 * 100, "b-s", ms=3, lw=1.5)
    ax2.axhline(100, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Perimeter / P₀ (%)")
    ax2.set_title("Interface Area Evolution")
    ax2.grid(True, alpha=0.3, ls="--")

    fig.suptitle("Lagrangian VOF Quality Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out3 = f"/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/e2e_45deg_metrics_{nx}"
    plt.savefig(f"{out3}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out3}.png")


if __name__ == "__main__":
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    grid, F, snaps, times, vols, peris = run_test(nx=nx)
    plot_results(grid, F, snaps, times, vols, peris, nx)
