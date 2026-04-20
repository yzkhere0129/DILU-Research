#!/usr/bin/env python3
"""Step 2 Lagrangian Move — Visual Unit Test.

Test A: uniform translation (u=v=1.0) → grid translates, stays square
Test B: shear flow (u=y, v=0) → grid deforms into parallelogram

Usage:
    python test_step2_move.py
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)

from jax_laseram.grid import create_grid
from jax_laseram.vof.lagrangian.move import lagrangian_move_faces


def create_uniform_velocity(grid, u_val=1.0, v_val=1.0):
    """Constant face velocities for uniform flow."""
    nh = grid.nh
    Ntot_x = grid.nx + 2 * nh
    Ntot_y = grid.ny + 2 * nh
    u_face = u_val * jnp.ones((Ntot_x - 1, Ntot_y, 1))
    v_face = v_val * jnp.ones((Ntot_x, Ntot_y - 1, 1))
    return u_face, v_face


def create_shear_velocity(grid):
    """Shear flow: u = y, v = 0.

    u_face[i,j] is at (x_{i+1/2}, y_j) → u = y_cc[j]
    """
    nh = grid.nh
    Ntot_x = grid.nx + 2 * nh
    Ntot_y = grid.ny + 2 * nh
    dx, dy = grid.dx, grid.dy

    # y at x-face locations (cell centers in y)
    y_cc = jnp.linspace(
        grid.y_range[0] + dy / 2 - (nh - 0.5) * dy,
        grid.y_range[1] - dy / 2 + (nh - 0.5) * dy,
        Ntot_y,
    )

    u_face = y_cc[None, :, None] * jnp.ones((Ntot_x - 1, Ntot_y, 1))
    v_face = jnp.zeros((Ntot_x, Ntot_y - 1, 1))
    return u_face, v_face


def draw_grid(ax, x_verts, y_verts, color="b", lw=1.0, alpha=1.0, ls="-"):
    """Draw grid lines from vertex arrays."""
    xv = np.array(x_verts[:, :, 0])
    yv = np.array(y_verts[:, :, 0])
    N_x, N_y = xv.shape

    # Horizontal lines
    for i in range(N_x):
        ax.plot(xv[i, :], yv[i, :], color=color, lw=lw, alpha=alpha, ls=ls)
    # Vertical lines
    for j in range(N_y):
        ax.plot(xv[:, j], yv[:, j], color=color, lw=lw, alpha=alpha, ls=ls)


def test_move(name, u_face, v_face, grid, dt, expected_desc):
    """Run a move test and generate visualization."""
    x_verts_0, y_verts_0 = lagrangian_move_faces(
        u_face * 0.0, v_face * 0.0, grid, dt
    )  # reference (no move)

    x_verts, y_verts = lagrangian_move_faces(u_face, v_face, grid, dt)

    # Check for topology inversion (area < 0)
    xv = np.array(x_verts[:, :, 0])
    yv = np.array(y_verts[:, :, 0])
    N_x, N_y = xv.shape

    min_area = np.inf
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            # Shoelace for quad (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            x0, y0 = xv[i, j], yv[i, j]
            x1, y1 = xv[i + 1, j], yv[i + 1, j]
            x2, y2 = xv[i + 1, j + 1], yv[i + 1, j + 1]
            x3, y3 = xv[i, j + 1], yv[i, j + 1]
            area = 0.5 * abs(
                (x0 * y1 - x1 * y0)
                + (x1 * y2 - x2 * y1)
                + (x2 * y3 - x3 * y2)
                + (x3 * y0 - x0 * y3)
            )
            min_area = min(min_area, area)

    original_area = grid.dx * grid.dy
    print(
        f"  {name}: min_area = {min_area:.6f} (original = {original_area:.6f}), "
        f"ratio = {min_area / original_area:.4f}"
    )
    if min_area <= 0:
        print(f"  WARNING: TOPOLOGY INVERSION DETECTED!")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title_suffix in zip(axes, ["Full domain", "Zoom (bottom-left)"]):
        ax.set_facecolor("#FAFAFA")
        draw_grid(ax, x_verts_0, y_verts_0, color="#CCCCCC", lw=0.5, ls="--")
        draw_grid(ax, x_verts, y_verts, color="#2171B5", lw=1.5, alpha=0.9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="gray")
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)

    axes[0].set_title(f"{name} — Full domain", fontsize=12, fontweight="bold")
    axes[1].set_title(
        f"{name} — Zoom (bottom-left 4×4)", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlim(grid.x_range[0] - 0.05, grid.x_range[0] + 4.5 * grid.dx)
    axes[1].set_ylim(grid.y_range[0] - 0.05, grid.y_range[0] + 4.5 * grid.dy)

    fig.suptitle(
        f"Step 2: Lagrangian Move — {name}\n{expected_desc}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = f"/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/step2_move_{name.lower().replace(' ', '_')}"
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}.png")


def main():
    # 10x10 grid on [0,1]^2
    grid = create_grid(10, 10, nz=1, x_range=(0, 1), y_range=(0, 1), nh=1)

    print("Step 2: Lagrangian Move Test", flush=True)
    print(f"  Grid: {grid.nx}x{grid.ny}, dx={grid.dx:.3f}, nh={grid.nh}", flush=True)

    # Test A: uniform translation
    dt_a = 0.3
    u_a, v_a = create_uniform_velocity(grid, u_val=1.0, v_val=0.5)
    test_move(
        "Test A Uniform",
        u_a,
        v_a,
        grid,
        dt_a,
        f"u=1.0, v=0.5, dt={dt_a} → translate by ({1.0 * dt_a}, {0.5 * dt_a})",
    )

    # Test B: shear flow
    dt_b = 0.3
    u_b, v_b = create_shear_velocity(grid)
    test_move(
        "Test B Shear",
        u_b,
        v_b,
        grid,
        dt_b,
        f"u=y, v=0, dt={dt_b} → parallelogram shear",
    )


if __name__ == "__main__":
    main()
