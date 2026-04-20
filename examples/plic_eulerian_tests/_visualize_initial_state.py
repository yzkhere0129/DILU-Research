#!/usr/bin/env python3
"""Generate initial-state visualizations for both PLIC benchmark cases.

This is a one-shot helper: it rebuilds the initial volume-fraction field
for each case, runs the Stage 1 PLIC primitives on it (Youngs normal,
intercept solver), and produces 2-panel figures showing:

    (a) Full domain F field with the analytic reference geometry
    (b) Zoomed interface region with Youngs normal arrows

Output is written to the respective ``results/`` subdirectories. These
images are intentionally not tracked in git (``*.png`` is gitignored).
To regenerate, run ``python _visualize_initial_state.py`` from this
directory.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import jax

jax.config.update("jax_enable_x64", True)

# Prefer the local checkout's jax_laseram over any editable install
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d  # noqa: E402
from jax_laseram.vof.plic.intercept_solver import solve_intercept  # noqa: E402

VOF_CMAP = LinearSegmentedColormap.from_list(
    "vof", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=256
)


def _interior_slice(F, grid):
    """Extract interior xy slice at the middle z layer."""
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    F_np = np.asarray(F)
    z_mid = nh + nz // 2 if nz > 1 else 0
    return F_np[nh : nh + nx, nh : nh + ny, z_mid]


def _normal_interior_slice(nx_arr, ny_arr, grid):
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    z_mid = nh + nz // 2 if nz > 1 else 0
    return (
        np.asarray(nx_arr)[nh : nh + nx, nh : nh + ny, z_mid],
        np.asarray(ny_arr)[nh : nh + nx, nh : nh + ny, z_mid],
    )


# ---------------------------------------------------------------------------
# Droplet 45-deg
# ---------------------------------------------------------------------------


def _load_by_path(mod_name: str, file_path: str):
    """Explicit file-based module loader to bypass ``sys.path`` collisions.

    Both ``droplet_45deg/`` and ``rt_instability/`` define a
    ``case_config.py`` module; a plain ``import_module("case_config")``
    resolves whichever one is first on ``sys.path`` and caches it in
    ``sys.modules``. We sidestep the cache by registering the loaded
    module under a *unique* alias tied to the file path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def plot_droplet():
    droplet_dir = os.path.join(_HERE, "droplet_45deg")
    case_config = _load_by_path(
        "droplet_case_config", os.path.join(droplet_dir, "case_config.py")
    )
    # init_droplet imports ``case_config`` itself, so temporarily expose
    # the droplet-side module under the bare name as well.
    sys.modules["case_config"] = case_config
    if droplet_dir not in sys.path:
        sys.path.insert(0, droplet_dir)
    init_droplet = _load_by_path(
        "droplet_init", os.path.join(droplet_dir, "init_droplet.py")
    )

    grid = init_droplet.build_grid()
    F = init_droplet.create_droplet_vof(grid)

    CASE = case_config.CASE
    cx0, cy0 = CASE["droplet"]["center"]
    R = CASE["droplet"]["radius"]
    cx_f, cy_f = CASE["analytic_final_center"]

    # Stage 1 smoke test: compute normals + intercepts
    nx_a, ny_a, nz_a = compute_youngs_normal_3d(F, grid.dx, grid.dy, grid.dz)
    C = solve_intercept(nx_a, ny_a, nz_a, F, grid.dx, grid.dy, grid.dz)

    F_slice = _interior_slice(F, grid)
    nx_slice, ny_slice = _normal_interior_slice(nx_a, ny_a, grid)
    C_slice = _interior_slice(C[..., None], grid)  # trick: reuse slicer

    # x, y coordinates (interior cell centers)
    x = np.linspace(grid.dx / 2, 1.0 - grid.dx / 2, grid.nx)
    y = np.linspace(grid.dy / 2, 1.0 - grid.dy / 2, grid.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel (a): Full domain
    ax1.contourf(X, Y, F_slice, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
    ax1.contour(X, Y, F_slice, levels=[0.5], colors="blue", linewidths=1.5)
    theta = np.linspace(0, 2 * np.pi, 120)
    ax1.plot(
        cx0 + R * np.cos(theta),
        cy0 + R * np.sin(theta),
        "b-",
        linewidth=1.0,
        alpha=0.6,
        label="analytic t=0",
    )
    ax1.plot(
        cx_f + R * np.cos(theta),
        cy_f + R * np.sin(theta),
        "g--",
        linewidth=1.5,
        label="analytic t=0.3536",
    )
    # 45-deg trajectory line
    ax1.plot([cx0, cx_f], [cy0, cy_f], "k:", linewidth=0.8, alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("(a) Initial droplet, full 1x1 domain", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3, ls="--")

    # Panel (b): Zoomed droplet with normal arrows
    margin = 3 * R
    xmin, xmax = cx0 - margin, cx0 + margin
    ymin, ymax = cy0 - margin, cy0 + margin
    # Find cells within zoom region
    i_in = (x >= xmin) & (x <= xmax)
    j_in = (y >= ymin) & (y <= ymax)
    i_idx = np.where(i_in)[0]
    j_idx = np.where(j_in)[0]
    X_z = X[np.ix_(i_idx, j_idx)]
    Y_z = Y[np.ix_(i_idx, j_idx)]
    F_z = F_slice[np.ix_(i_idx, j_idx)]
    nx_z = nx_slice[np.ix_(i_idx, j_idx)]
    ny_z = ny_slice[np.ix_(i_idx, j_idx)]

    ax2.contourf(X_z, Y_z, F_z, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
    ax2.contour(X_z, Y_z, F_z, levels=[0.5], colors="blue", linewidths=2.0)
    # Interface arrows
    mask_arrow = (F_z > 0.05) & (F_z < 0.95)
    ax2.quiver(
        X_z[mask_arrow],
        Y_z[mask_arrow],
        nx_z[mask_arrow],
        ny_z[mask_arrow],
        color="red",
        scale=25,
        scale_units="width",
        width=0.005,
        alpha=0.9,
    )
    ax2.plot(
        cx0 + R * np.cos(theta),
        cy0 + R * np.sin(theta),
        "k--",
        linewidth=1.5,
        alpha=0.7,
        label="exact circle",
    )
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title(
        f"(b) Zoom around droplet, {int(mask_arrow.sum())} interface cells with Youngs normals",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3, ls="--")

    fig.suptitle(
        "PLIC droplet_45deg initial state — Stage 1 smoke test",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_dir = os.path.join(_HERE, "droplet_45deg", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "initial_state.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    # Print smoke-test stats
    F_np = np.asarray(F)
    interface_mask = (F_np > 1e-6) & (F_np < 1.0 - 1e-6)
    n_if = int(interface_mask.sum())
    ni = np.asarray(nx_a)
    nj = np.asarray(ny_a)
    nk = np.asarray(nz_a)
    if n_if > 0:
        mag = np.sqrt(ni[interface_mask] ** 2 + nj[interface_mask] ** 2 + nk[interface_mask] ** 2)
        print(
            f"  droplet interface cells: {n_if}, mean|n|={mag.mean():.4f}, "
            f"mean|nx|={np.abs(ni[interface_mask]).mean():.4f}, "
            f"mean|ny|={np.abs(nj[interface_mask]).mean():.4f}"
        )
        C_vals = np.asarray(C)[interface_mask]
        print(f"  intercept C range: [{C_vals.min():.3e}, {C_vals.max():.3e}]")


# ---------------------------------------------------------------------------
# RT instability
# ---------------------------------------------------------------------------


def plot_rt():
    rt_dir = os.path.join(_HERE, "rt_instability")
    case_config_rt = _load_by_path(
        "rt_case_config", os.path.join(rt_dir, "case_config.py")
    )
    # Re-point bare ``case_config`` to the RT version before loading init
    sys.modules["case_config"] = case_config_rt
    if rt_dir not in sys.path:
        sys.path.insert(0, rt_dir)
    # Drop any stale init_alpha from a prior run
    sys.modules.pop("init_alpha", None)
    init_alpha = _load_by_path(
        "rt_init_alpha", os.path.join(rt_dir, "init_alpha.py")
    )

    grid = init_alpha.build_grid()
    F = init_alpha.create_rt_vof(grid)

    CASE = case_config_rt.CASE

    # Stage 1 smoke test
    nx_a, ny_a, nz_a = compute_youngs_normal_3d(F, grid.dx, grid.dy, grid.dz)
    C = solve_intercept(nx_a, ny_a, nz_a, F, grid.dx, grid.dy, grid.dz)

    F_slice = _interior_slice(F, grid)
    nx_slice, ny_slice = _normal_interior_slice(nx_a, ny_a, grid)

    x = np.linspace(grid.dx / 2, 1.0 - grid.dx / 2, grid.nx)
    y = np.linspace(grid.dy / 2, 4.0 - grid.dy / 2, grid.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Analytic cosine interface
    x_cos = np.linspace(0, 1, 200)
    y_cos = CASE["rt"]["interface_y0"] + CASE["rt"]["amplitude"] * np.cos(2 * np.pi * x_cos)

    # --- Plot: 3 panels ---
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.8, 1.8])

    # Panel (a): Full 1x4 domain
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.contourf(X, Y, F_slice, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
    ax1.contour(X, Y, F_slice, levels=[0.5], colors="blue", linewidths=1.0)
    ax1.plot(x_cos, y_cos, "r--", linewidth=1.2, alpha=0.7, label="analytic y=2+0.05cos(2πx)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 4)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("(a) Full domain 1x4", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7, loc="lower left")

    # Panel (b): Zoomed interface band (y in [1.85, 2.15])
    ax2 = fig.add_subplot(gs[0, 1])
    j_mask = (y >= 1.85) & (y <= 2.15)
    j_idx = np.where(j_mask)[0]
    X_z = X[:, j_idx]
    Y_z = Y[:, j_idx]
    F_z = F_slice[:, j_idx]
    ax2.contourf(X_z, Y_z, F_z, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
    ax2.contour(X_z, Y_z, F_z, levels=[0.5], colors="blue", linewidths=2.0)
    ax2.plot(x_cos, y_cos, "r--", linewidth=1.5, alpha=0.8, label="analytic interface")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1.85, 2.15)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("(b) Interface band, zoom y∈[1.85,2.15]", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")

    # Panel (c): Normals at interface cells
    ax3 = fig.add_subplot(gs[0, 2])
    nx_z = nx_slice[:, j_idx]
    ny_z = ny_slice[:, j_idx]
    ax3.contourf(X_z, Y_z, F_z, levels=np.linspace(0, 1, 21), cmap=VOF_CMAP)
    mask_arrow = (F_z > 0.05) & (F_z < 0.95)
    # sparse quiver for clarity: every 4th cell along x
    sub = np.zeros_like(mask_arrow, dtype=bool)
    for i in range(0, sub.shape[0], 4):
        sub[i, :] = mask_arrow[i, :]
    ax3.quiver(
        X_z[sub],
        Y_z[sub],
        nx_z[sub],
        ny_z[sub],
        color="red",
        scale=15,
        scale_units="width",
        width=0.004,
        alpha=0.9,
    )
    ax3.plot(x_cos, y_cos, "k--", linewidth=1.2, alpha=0.6)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(1.85, 2.15)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title(
        f"(c) Youngs normals ({int(mask_arrow.sum())} interface cells)",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "PLIC rt_instability initial state — Stage 1 smoke test",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_dir = os.path.join(_HERE, "rt_instability", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "initial_state.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    # Print smoke-test stats
    F_np = np.asarray(F)
    interface_mask = (F_np > 1e-6) & (F_np < 1.0 - 1e-6)
    n_if = int(interface_mask.sum())
    if n_if > 0:
        ni = np.asarray(nx_a)
        nj = np.asarray(ny_a)
        nk = np.asarray(nz_a)
        mag = np.sqrt(ni[interface_mask] ** 2 + nj[interface_mask] ** 2 + nk[interface_mask] ** 2)
        print(
            f"  RT interface cells: {n_if}, mean|n|={mag.mean():.4f}, "
            f"mean|nx|={np.abs(ni[interface_mask]).mean():.4f}, "
            f"mean|ny|={np.abs(nj[interface_mask]).mean():.4f}"
        )
        C_vals = np.asarray(C)[interface_mask]
        print(f"  intercept C range: [{C_vals.min():.3e}, {C_vals.max():.3e}]")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating initial-state visualizations for PLIC test cases")
    print("=" * 60)
    print("\n[1/2] droplet_45deg:")
    plot_droplet()
    print("\n[2/2] rt_instability:")
    plot_rt()
    print("\nDone.")
