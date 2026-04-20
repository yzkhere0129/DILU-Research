#!/usr/bin/env python3
"""Step 1 PLIC Reconstruction — Visual Unit Test (v2, fixed quadrants).

Usage:
    python test_step1_plic.py [nx]
"""

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

jax.config.update("jax_enable_x64", True)

from jax_laseram.vof.advection import _apply_vof_bcs
from jax_laseram.vof.lagrangian.reconstruction import (
    compute_plic_normals,
    compute_intercept_C,
    plic_line_segments_batch,
)


def create_circle_vof(nx, dx, nh, center=(0.5, 0.5), radius=0.2):
    x = jnp.linspace(dx / 2, 1.0 - dx / 2, nx)
    y = jnp.linspace(dx / 2, 1.0 - dx / 2, nx)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    cx, cy = center
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    eps = 1.5 * dx
    F_int = jnp.clip(0.5 * (1.0 - jnp.tanh((r - radius) / eps)), 0.0, 1.0)
    shape = (nx + 2 * nh, nx + 2 * nh, 1)
    F = jnp.zeros(shape).at[nh : nh + nx, nh : nh + nx, :].set(F_int[:, :, None])
    F = _apply_vof_bcs(F, nh)
    return F


def test_plic(nx=50, center=(0.5, 0.5), radius=0.2):
    dx = 1.0 / nx
    nh = 1

    print(f"PLIC Reconstruction Test: {nx}×{nx}, R={radius}", flush=True)

    F = create_circle_vof(nx, dx, nh, center, radius)

    t0 = time.time()
    nx_field, ny_field = compute_plic_normals(F, dx, dx)
    C_field = compute_intercept_C(F, nx_field, ny_field, dx, dx)
    print(f"  Computation: {time.time() - t0:.3f}s", flush=True)

    # Extract interior
    ic = slice(nh, -nh)
    F_i = np.array(F[ic, ic, 0])
    nx_i = np.array(nx_field[ic, ic, 0])
    ny_i = np.array(ny_field[ic, ic, 0])
    C_i = np.array(C_field[ic, ic, 0])

    x_cc = np.linspace(dx / 2, 1.0 - dx / 2, nx)
    y_cc = np.linspace(dx / 2, 1.0 - dx / 2, nx)

    # Compute segments using batch function
    segments = plic_line_segments_batch(nx_i, ny_i, C_i, F_i, x_cc, y_cc, dx, dx)

    n_interface = int(np.sum((F_i > 0.01) & (F_i < 0.99)))
    print(f"  Interface cells: {n_interface}", flush=True)
    print(f"  PLIC segments: {len(segments)}", flush=True)

    # Error analysis
    errors = []
    cx, cy = center
    for seg in segments:
        mx = 0.5 * (seg[0][0] + seg[1][0])
        my = 0.5 * (seg[0][1] + seg[1][1])
        d = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
        errors.append(abs(d - radius))
    errors = np.array(errors)

    if len(errors) > 0:
        print(
            f"  Error: mean={errors.mean() / dx:.3f} dx, "
            f"max={errors.max() / dx:.3f} dx, "
            f"rms={np.sqrt(np.mean(errors**2)) / dx:.3f} dx",
            flush=True,
        )

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    X, Y = np.meshgrid(x_cc, y_cc, indexing="ij")

    # Panel 1: VOF + normals
    ax = axes[0]
    cf = ax.contourf(X, Y, F_i, levels=np.linspace(0, 1, 21), cmap="Blues")
    skip = max(1, nx // 15)
    ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        nx_i[::skip, ::skip],
        ny_i[::skip, ::skip],
        color="red",
        scale=20,
        width=0.003,
        alpha=0.7,
    )
    ax.set_title("VOF + Normals", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(cf, ax=ax, shrink=0.8)

    # Panel 2: PLIC segments (KEY PLOT)
    ax = axes[1]
    ax.set_facecolor("#F5F5F5")
    ax.contour(X, Y, F_i, levels=[0.5], colors="gray", linewidths=0.5, linestyles="--")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        "g-",
        lw=2,
        alpha=0.7,
        label="Exact circle",
        zorder=1,
    )
    if segments:
        lc = LineCollection(
            segments,
            colors="red",
            linewidths=1.8,
            alpha=0.9,
            label=f"PLIC ({len(segments)} seg)",
        )
        ax.add_collection(lc)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("PLIC vs Exact", fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Panel 3: Error heatmap
    ax = axes[2]
    err_map = np.full((nx, nx), np.nan)
    for seg in segments:
        mx = 0.5 * (seg[0][0] + seg[1][0])
        my = 0.5 * (seg[0][1] + seg[1][1])
        d = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
        # Find nearest cell
        ii = int(mx / dx)
        jj = int(my / dx)
        if 0 <= ii < nx and 0 <= jj < nx:
            err_map[ii, jj] = abs(d - radius)
    cf = ax.pcolormesh(X, Y, err_map, cmap="hot_r", vmin=0, vmax=2 * dx)
    ax.contour(X, Y, F_i, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_title("PLIC midpoint error", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(cf, ax=ax, shrink=0.8, label="|d - R|")

    fig.suptitle(
        f"Step 1: PLIC Reconstruction (v2) — {nx}×{nx}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out = f"/home/yzk/JAX-LaserAM/examples/lagrangian_vof_tests/step1_plic_v2_{nx}x{nx}"
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{out}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}.png / .pdf", flush=True)


if __name__ == "__main__":
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    test_plic(nx=nx)
