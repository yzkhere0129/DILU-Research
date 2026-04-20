#!/usr/bin/env python3
"""Barkhudarov Test 1 — Multi-Angle Droplet Advection with Eulerian PLIC.

Ported from examples/barkhudarov_tests/barkhudarov_v2/test1_advection/
barkhudarov_test1_multiangle.py. Same geometry, same metrics, same plot
format — only the solver is swapped from Lagrangian VOF to the Eulerian
PLIC pipeline (Phase B analytic intercept + padded gather + F clip).

Paper: Barkhudarov (2004) §4.1 — Figure 5 & 6
Angles: 0°, 6°, 30°, 45°
Grid:   100×100×5, cell_size = D/10 (10 cells/diameter)
Travel: 5 diameters
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
sys.path.insert(0, _SRC)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x,
    apply_flux_x,
    sweep_flux_y,
    apply_flux_y,
)

# ── Configuration (identical to Lagrangian reference) ─────────────
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
NH = 1


def create_droplet_cylinder(cx, cy, r):
    """Sub-cell linear ramp init (same formula as reference, no tanh)."""
    x = np.linspace(DX / 2, DOMAIN_XY - DX / 2, NX)
    y = np.linspace(DX / 2, DOMAIN_XY - DX / 2, NX)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    hd = np.sqrt(2) * DX / 2
    F_2d = np.where(
        dist + hd <= r, 1.0,
        np.where(
            dist - hd >= r, 0.0,
            np.clip((r - dist + hd) / (2 * hd), 0.0, 1.0),
        ),
    )
    Ntx = NX + 2 * NH
    Ntz = NZ + 2 * NH
    F = np.zeros((Ntx, Ntx, Ntz), dtype=np.float32)
    F[NH : NH + NX, NH : NH + NX, NH : NH + NZ] = F_2d[:, :, None]
    return halo_symmetry(jnp.asarray(F))


def halo_symmetry(F):
    F = F.at[0].set(F[1])
    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1])
    F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1])
    F = F.at[:, :, -1].set(F[:, :, -2])
    return F


def compute_perimeter(F_2d, dx, dy):
    """Marching-squares contour perimeter at F=0.5 (matches reference)."""
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


def plic_step(F, u, v, dt):
    """Strang split x/2 → y → x/2 (constant scalar velocities)."""
    def sub_x(F, dt_s):
        F = halo_symmetry(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        C = analytic_intercept(nx, ny, nz, F, DX, DX, DX)
        flux = sweep_flux_x(F, nx, ny, nz, C, u, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo_symmetry(F)

    def sub_y(F, dt_s):
        F = halo_symmetry(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
        C = analytic_intercept(nx, ny, nz, F, DX, DX, DX)
        flux = sweep_flux_y(F, nx, ny, nz, C, v, dt_s, DX, DX, DX)
        F = jnp.clip(apply_flux_y(F, flux, DX, DX, DX), 0.0, 1.0)
        return halo_symmetry(F)

    F = sub_x(F, dt / 2)
    F = sub_y(F, dt)
    F = sub_x(F, dt / 2)
    return F


def main():
    dt = CFL * DX / SPEED
    n_steps = int(np.ceil(T_END / dt))
    dt = T_END / n_steps
    z_mid = NH + NZ // 2
    cell_vol = DX * DX * DX

    print("=" * 70)
    print("  Barkhudarov Test 1: Multi-Angle Droplet Advection — Eulerian PLIC")
    print("=" * 70)
    print(f"  Grid: {NX}×{NX}×{NZ}, dx={DX:.4f}, D={D}, 10 cells/diameter")
    print(f"  Travel: {TRAVEL_DIST} (5D), dt={dt:.6f}, n_steps={n_steps}")
    print(f"  Solver: Phase B analytic + padded gather + F clip")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Angles: {ANGLES}")
    print("=" * 70)

    all_results = {}

    fig_final, axes_final = plt.subplots(1, len(ANGLES), figsize=(4 * len(ANGLES), 4))
    x_plt = np.linspace(DX / 2, DOMAIN_XY - DX / 2, NX)
    X_plt, Y_plt = np.meshgrid(x_plt, x_plt, indexing="ij")

    for ai, angle in enumerate(ANGLES):
        print(f"\n── Angle: {angle}° ──")

        # Same starting-position logic as reference
        theta = np.radians(angle)
        cx_final = 0.20 + np.cos(theta) * TRAVEL_DIST
        cy_final = 0.20 + np.sin(theta) * TRAVEL_DIST
        cx0, cy0 = 0.20, 0.20
        if cx_final > 0.90:
            cx0 = 0.90 - np.cos(theta) * TRAVEL_DIST
        if cy_final > 0.90:
            cy0 = 0.90 - np.sin(theta) * TRAVEL_DIST
        cx0 = max(0.10, min(0.90, cx0))
        cy0 = max(0.10, min(0.90, cy0))
        cx_final = cx0 + np.cos(theta) * TRAVEL_DIST
        cy_final = cy0 + np.sin(theta) * TRAVEL_DIST
        print(f"  Start: ({cx0:.3f}, {cy0:.3f}) → End: ({cx_final:.3f}, {cy_final:.3f})")

        # Init
        F = create_droplet_cylinder(cx0, cy0, R)
        ux = float(np.cos(theta) * SPEED)
        uy = float(np.sin(theta) * SPEED)

        ic = slice(NH, NH + NX)
        icz = slice(NH, NH + NZ)
        F_np0 = np.asarray(F[ic, ic, icz])
        F_slice0 = np.asarray(F[ic, ic, z_mid])
        V0 = float(F_np0.sum() * cell_vol)
        P0 = compute_perimeter(F_slice0, DX, DX)
        print(f"  V₀={V0:.10f}  P₀={P0:.6f}")

        jit_step = jax.jit(lambda F: plic_step(F, ux, uy, dt))
        print("  JIT compiling...", end=" ", flush=True)
        t_jit = time.time()
        _ = jit_step(F)
        jax.block_until_ready(_)
        print(f"{time.time() - t_jit:.1f}s")

        t0 = time.time()
        for step_idx in range(n_steps):
            F = jit_step(F)
            if (step_idx + 1) % max(1, n_steps // 4) == 0:
                jax.block_until_ready(F)
                Vi = float(np.asarray(F[ic, ic, icz]).sum() * cell_vol)
                dVi = abs(Vi - V0) / V0 * 100
                print(
                    f"    [{(step_idx + 1) / n_steps * 100:3.0f}%] |ΔV/V|={dVi:.4e}%",
                    flush=True,
                )
        jax.block_until_ready(F)
        wall_t = time.time() - t0

        F_final_slice = np.asarray(F[ic, ic, z_mid])
        F_final_3d = np.asarray(F[ic, ic, icz])
        V_final = float(F_final_3d.sum() * cell_vol)
        P_final = compute_perimeter(F_final_slice, DX, DX)
        dV = abs(V_final - V0) / V0 * 100
        dP = (P_final - P0) / P0 * 100

        v_pass = dV < 0.5
        p_pass = abs(dP) < 0.5
        print(
            f"  Done ({wall_t:.1f}s): |ΔV/V|={dV:.4f}%{'✓' if v_pass else '✗'}  "
            f"ΔP/P={dP:+.4f}%{'✓' if p_pass else '✗'}"
        )

        all_results[f"{angle}°"] = {
            "angle_deg": angle,
            "start": [cx0, cy0],
            "V0": V0,
            "P0": P0,
            "V_final": V_final,
            "P_final": P_final,
            "dV_percent": dV,
            "dP_percent": dP,
            "volume_pass": v_pass,
            "perimeter_pass": p_pass,
            "wall_time_s": wall_t,
        }

        # Plot (same layout as reference)
        ax = axes_final[ai]
        ax.set_aspect("equal")
        ax.contourf(
            X_plt, Y_plt, F_final_slice, levels=np.linspace(0, 1, 11),
            cmap="Blues", alpha=0.7,
        )
        ax.contour(X_plt, Y_plt, F_final_slice, levels=[0.5], colors="navy", linewidths=1.5)
        th_c = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx_final + R * np.cos(th_c), cy_final + R * np.sin(th_c),
                "g--", lw=1, label="Exact")
        ax.plot(cx0 + R * np.cos(th_c), cy0 + R * np.sin(th_c),
                "gray", lw=0.5, ls="--", label="Initial")
        ax.set_title(
            f"{angle}°  |ΔV/V|={dV:.3e}%\nΔP/P={dP:+.3f}%",
            fontsize=9,
        )
        ax.set_xlim(0, DOMAIN_XY)
        ax.set_ylim(0, DOMAIN_XY)

    fig_final.suptitle(
        f"Barkhudarov Test 1: Multi-Angle (Eulerian PLIC)\n"
        f"Grid {NX}×{NX}×{NZ}, dx={DX}",
        fontsize=11,
    )
    fig_final.tight_layout()
    out_png = os.path.join(_HERE, "test1_multiangle_3d.png")
    fig_final.savefig(out_png, dpi=150)
    plt.close(fig_final)
    print(f"\nSaved: {out_png}")

    out_json = os.path.join(_HERE, "test1_multiangle_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_json}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for k, v in all_results.items():
        sv = "✓" if v["volume_pass"] else "✗"
        sp = "✓" if v["perimeter_pass"] else "✗"
        print(
            f"  {k:>4s}: |ΔV/V|={v['dV_percent']:.4f}%{sv}  "
            f"ΔP/P={v['dP_percent']:+.4f}%{sp}  "
            f"wall={v['wall_time_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
