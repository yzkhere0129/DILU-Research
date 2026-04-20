#!/usr/bin/env python3
"""Step 3 Overlay — Volume Conservation Test.

Moves a 4×4 solid block diagonally by (0.3*dx, 0.7*dy) on a 20×20 grid.
Checks that total F is conserved to machine precision.

Usage:
    python test_step3_overlay.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from jax_laseram.grid import create_grid
from jax_laseram.vof.lagrangian.move import lagrangian_move_faces
from jax_laseram.vof.lagrangian.overlay import overlay_lagrangian


def test_overlay(nx=20, ny=20, block_size=4, shift_x=0.3, shift_y=0.7):
    """Test volume conservation with a solid block displacement."""
    grid = create_grid(nx, ny, nz=1, x_range=(0, 1), y_range=(0, 1), nh=1)
    dx, dy = grid.dx, grid.dy
    Ntot_x = nx + 2 * grid.nh
    Ntot_y = ny + 2 * grid.nh

    print(f"Step 3 Overlay Test: {nx}×{ny} grid, {block_size}×{block_size} block")
    print(f"  dx={dx:.4f}, dy={dy:.4f}")
    print(
        f"  Shift: ({shift_x}·dx, {shift_y}·dy) = ({shift_x * dx:.4f}, {shift_y * dy:.4f})"
    )

    # Initialize F: solid block in center
    F = jnp.zeros((Ntot_x, Ntot_y, 1))
    i_center = nx // 2
    j_center = ny // 2
    half = block_size // 2
    for i in range(i_center - half, i_center + half):
        for j in range(j_center - half, j_center + half):
            F = F.at[i + grid.nh, j + grid.nh, 0].set(1.0)

    F0_sum = float(jnp.sum(F))

    # Constant velocity: displacement = (shift_x*dx, shift_y*dy)
    dt = 1.0  # arbitrary, displacement = u*dt
    u_val = shift_x * dx / dt
    v_val = shift_y * dy / dt

    u_face = u_val * jnp.ones((Ntot_x - 1, Ntot_y, 1))
    v_face = v_val * jnp.ones((Ntot_x, Ntot_y - 1, 1))

    # Lagrangian move
    t0 = time.time()
    x_verts, y_verts = lagrangian_move_faces(u_face, v_face, grid, dt)
    print(f"  Move: {time.time() - t0:.2f}s")

    # Overlay
    t0 = time.time()
    F_new = overlay_lagrangian(F, x_verts, y_verts, grid)
    print(f"  Overlay: {time.time() - t0:.2f}s")

    # Volume conservation check
    F_new_sum = float(jnp.sum(F_new))
    error = abs(F_new_sum - F0_sum) / F0_sum if F0_sum > 0 else abs(F_new_sum)

    print(f"\n  F0 total:  {F0_sum:.15f}")
    print(f"  F_new total: {F_new_sum:.15f}")
    print(f"  Relative error: {error:.2e}")

    if error < 1e-12:
        print(f"  PASS: Machine precision conservation!")
    elif error < 1e-6:
        print(f"  WARN: Good but not machine precision")
    else:
        print(f"  FAIL: Volume NOT conserved!")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_cc = np.linspace(dx / 2, 1 - dx / 2, nx)
    y_cc = np.linspace(dy / 2, 1 - dy / 2, ny)
    X, Y = np.meshgrid(x_cc, y_cc, indexing="ij")

    F0_int = np.array(F[grid.nh : -grid.nh, grid.nh : -grid.nh, 0])
    Fnew_int = np.array(F_new)

    for ax, data, title in [
        (axes[0], F0_int, "Initial F"),
        (axes[1], Fnew_int, "After Overlay F_new"),
        (axes[2], Fnew_int - F0_int, "F_new - F0"),
    ]:
        im = ax.pcolormesh(X, Y, data, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[2].collections[0].set_cmap("RdBu_r")
    axes[2].collections[0].set_clim(-1, 1)

    fig.suptitle(
        f"Step 3: Overlay — Volume error = {error:.2e}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    out = "/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/step3_overlay"
    plt.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}.png")

    return error


if __name__ == "__main__":
    error = test_overlay()
