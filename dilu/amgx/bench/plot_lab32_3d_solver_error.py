"""3D spatial error plot consuming lab 5060 prepared npz (melting + evap phases).

Input: compact npz from prepare_lab32_plot_data.py — one per phase.
For each phase, plots:
  - 3D oblique scatter of |x_OF      - x_truth| / ‖x_truth‖∞
  - 3D oblique scatter of |x_AMGx_e8 - x_truth| / ‖x_truth‖∞
  - histogram of both error distributions on same axes (log-x)

+ bottom panel: timing summary table from each npz's meta JSON.

Truth = AMGx tol=1e-12 actual residual (printed in title).

Usage:
    python -m dilu.amgx.bench.plot_lab32_3d_solver_error \\
        --melting dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz \\
        --evaporation dilu/amgx/bench/lab32_plot_evaporation_pd_corr0_9.0e-07.npz

Output: docs/benchmark/figures/amgx_3d_solver_error_lab32.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_phase(npz_path: Path, mesh_override: tuple | None = None) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    n = int(z["n"][0])
    if mesh_override is not None:
        nx, ny, nz, dx = mesh_override
        if nx * ny * nz != n:
            raise ValueError(
                f"mesh-shape {nx}*{ny}*{nz}={nx*ny*nz} != N={n} from {npz_path}")
        cid = np.arange(n, dtype=np.int64)
        i = (cid % nx).astype(np.int32)
        j = ((cid // nx) % ny).astype(np.int32)
        k = (cid // (nx * ny)).astype(np.int32)
    else:
        nx = int(z["nx"][0]); ny = int(z["ny"][0]); nz = int(z["nz"][0])
        dx = float(z["dx"][0])
        i = z["i"]; j = z["j"]; k = z["k"]
    return dict(
        x_OF      = z["x_OF"],
        x_AMGx_e8 = z["x_AMGx_e8"],
        x_truth   = z["x_truth"],
        b         = z["b"],
        i = i, j = j, k = k,
        nx = nx, ny = ny, nz = nz, dx = dx, n = n,
        meta = meta,
    )


def scatter3d_err(ax, i, j, k, dx, err, *, title, vmax,
                   max_points=40000, elev=22, azim=-55,
                   cmap="viridis", z_window=None):
    """3D scatter with linear-scale color = relative error."""
    if z_window is None:
        z_lo, z_hi = 0, k.max() * dx * 1e6
    else:
        z_lo, z_hi = z_window
    kmin = int(z_lo / (dx * 1e6))
    kmax = int(z_hi / (dx * 1e6))
    z_mask = (k >= kmin) & (k <= kmax)
    idx = np.where(z_mask)[0]
    if idx.size > max_points:
        rng = np.random.default_rng(0xC0FFEE)
        idx = rng.choice(idx, size=max_points, replace=False)

    xs = i[idx] * dx * 1e6
    ys = j[idx] * dx * 1e6
    zs = k[idx] * dx * 1e6
    es = err[idx]

    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-12))
    colors = plt.get_cmap(cmap)(norm(es))
    alpha = np.clip(es / max(vmax, 1e-12), 0.10, 0.95)
    colors[:, 3] = alpha

    ax.scatter(xs, ys, zs, c=colors, s=6, marker="o",
                linewidth=0, depthshade=True)
    nx_um = i.max() * dx * 1e6
    ny_um = j.max() * dx * 1e6
    ax.set_xlim(0, nx_um); ax.set_ylim(0, ny_um); ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((nx_um, ny_um, z_hi - z_lo))
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def render_phase(fig, gs_row, phase_data, *, phase_label, sm_handle):
    x_truth = phase_data["x_truth"]
    x_OF    = phase_data["x_OF"]
    x_e8    = phase_data["x_AMGx_e8"]
    denom = max(float(np.abs(x_truth).max()), 1e-300)
    err_OF = np.abs(x_OF - x_truth) / denom
    err_e8 = np.abs(x_e8 - x_truth) / denom
    err_max = max(err_OF.max(), err_e8.max())

    meta = phase_data["meta"]
    truth_resid = meta["amgx_truth"]["rel_resid_actual"]
    e8_resid    = meta["amgx_e8"]["rel_resid_actual"]
    truth_iters = meta["amgx_truth"]["iters"]
    e8_iters    = meta["amgx_e8"]["iters"]
    truth_solve = meta["amgx_truth"]["t_solve_s"] * 1e3
    e8_solve    = meta["amgx_e8"]["t_solve_s"] * 1e3

    # Phase-localized z window (full domain by default; melt pool is ~80-160μm
    # in LPBF_crosscheck-style cases).
    nz_um = phase_data["nz"] * phase_data["dx"] * 1e6
    z_window = (0, nz_um) if nz_um <= 250 else (80, 160)

    ax1 = fig.add_subplot(gs_row[0], projection='3d')
    sm = scatter3d_err(
        ax1, phase_data["i"], phase_data["j"], phase_data["k"],
        phase_data["dx"], err_OF,
        title=f"[{phase_label}]  OF DICPCG @ tol=1e-8\n"
              f"max rel err = {err_OF.max():.2e}  "
              f"(= {err_OF.max()*denom:.2g} Pa)",
        vmax=err_max, z_window=z_window)
    sm_handle["sm"] = sm

    ax2 = fig.add_subplot(gs_row[1], projection='3d')
    scatter3d_err(
        ax2, phase_data["i"], phase_data["j"], phase_data["k"],
        phase_data["dx"], err_e8,
        title=f"[{phase_label}]  AMGx @ tol=1e-8  "
              f"(iter={e8_iters}, solve={e8_solve:.1f}ms)\n"
              f"max rel err = {err_e8.max():.2e}  "
              f"(= {err_e8.max()*denom:.2g} Pa)",
        vmax=err_max, z_window=z_window)

    ax3 = fig.add_subplot(gs_row[2])
    bins = np.logspace(-13, np.log10(err_max * 1.5 + 1e-12), 50)
    e_OF = err_OF[err_OF > 1e-14]
    e_e8 = err_e8[err_e8 > 1e-14]
    ax3.hist(e_OF, bins=bins, alpha=0.55, color="tab:orange",
              edgecolor="black", linewidth=0.3,
              label=f"OF (med {np.median(e_OF):.1e}, max {e_OF.max():.1e})")
    ax3.hist(e_e8, bins=bins, alpha=0.55, color="tab:blue",
              edgecolor="black", linewidth=0.3,
              label=f"AMGx 1e-8 (med {np.median(e_e8):.1e}, max {e_e8.max():.1e})")
    ax3.set_xscale("log")
    ax3.set_xlim(1e-13, 1e-3)
    ax3.axvline(1e-3, color="red", linestyle="-", linewidth=0.8, alpha=0.6)
    ax3.text(0.95 * 1e-3, 0.95, "1e-3", transform=ax3.get_xaxis_transform(),
              fontsize=8, ha="right", va="top", color="red")
    ax3.set_xlabel("relative error  |x - x_truth| / ‖x_truth‖∞", fontsize=9)
    ax3.set_ylabel("# cells", fontsize=9)
    ax3.set_title(f"[{phase_label}]  truth = AMGx tol=1e-12  "
                    f"(iter={truth_iters}, solve={truth_solve:.1f}ms, "
                    f"rel_resid={truth_resid:.1e})", fontsize=9)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(alpha=0.3, which="both")
    return err_max


def render_timing_table(ax, phase_npzs, phase_labels):
    """Bottom row table: per-phase timing breakdown."""
    ax.axis("off")
    lines = [
        "TIMING BREAKDOWN (per phase) — Lab 5060 (RTX 5060)",
        "─" * 92,
        f"{'phase':<14s} {'reconstruct':>12s} {'AMGx_e12 setup':>15s} "
        f"{'AMGx_e12 solve':>15s} {'AMGx_e8 setup':>14s} "
        f"{'AMGx_e8 solve':>14s}",
        "─" * 92,
    ]
    for d, label in zip(phase_npzs, phase_labels):
        m = d["meta"]
        t = m["timings_s"]
        lines.append(
            f"{label:<14s} "
            f"{t['reconstruct_s']*1e3:>10.1f} ms "
            f"{m['amgx_truth']['t_setup_s']*1e3:>13.1f} ms "
            f"{m['amgx_truth']['t_solve_s']*1e3:>13.1f} ms "
            f"{m['amgx_e8']['t_setup_s']*1e3:>12.1f} ms "
            f"{m['amgx_e8']['t_solve_s']*1e3:>12.1f} ms"
        )
    lines += [
        "─" * 92,
        "ITERATIONS + ACTUAL RESIDUAL",
    ]
    for d, label in zip(phase_npzs, phase_labels):
        m = d["meta"]
        lines.append(
            f"  {label:<12s}  AMGx tol=1e-12 → "
            f"iter {m['amgx_truth']['iters']:>3d}, "
            f"rel_resid {m['amgx_truth']['rel_resid_actual']:.2e}    │    "
            f"AMGx tol=1e-8  → iter {m['amgx_e8']['iters']:>3d}, "
            f"rel_resid {m['amgx_e8']['rel_resid_actual']:.2e}"
        )
    lines.append(
        f"\n  OF dump consistency check  ‖A·x_OF - b‖/‖b‖:")
    for d, label in zip(phase_npzs, phase_labels):
        m = d["meta"]
        lines.append(
            f"    {label:<12s}  rel = {m['rel_OF_consistency']:.2e}  "
            f"(should be ≤ ~1e-8 for clean dump)")
    ax.text(0.0, 1.0, "\n".join(lines), fontfamily="monospace",
              fontsize=9.5, verticalalignment="top",
              transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--melting", required=False)
    ap.add_argument("--evaporation", required=False)
    ap.add_argument("--out-name", default="amgx_3d_solver_error_lab32")
    ap.add_argument("--mesh-shape", default=None,
                     help="override (nx,ny,nz) e.g. '50,200,50'")
    ap.add_argument("--dx", type=float, default=None,
                     help="override cell size in meters")
    args = ap.parse_args()

    mesh_override = None
    if args.mesh_shape:
        nx, ny, nz = [int(x) for x in args.mesh_shape.split(",")]
        if args.dx is None:
            raise SystemExit("--mesh-shape requires --dx")
        mesh_override = (nx, ny, nz, args.dx)

    phases = []
    labels = []
    if args.melting:
        phases.append(load_phase(Path(args.melting), mesh_override))
        labels.append("melting")
    if args.evaporation:
        phases.append(load_phase(Path(args.evaporation), mesh_override))
        labels.append("evaporation")
    if not phases:
        raise SystemExit("Provide at least --melting or --evaporation")

    n_phases = len(phases)
    fig = plt.figure(figsize=(20, 5 * n_phases + 4))
    # n_phases rows of (3D, 3D, hist) + 1 timing-table row at bottom
    gs = fig.add_gridspec(n_phases + 1, 3,
                            height_ratios=[1.0] * n_phases + [0.5],
                            hspace=0.35, wspace=0.3)

    err_max_overall = 0.0
    sm_handle = {}
    for row, (d, label) in enumerate(zip(phases, labels)):
        em = render_phase(fig, [gs[row, 0], gs[row, 1], gs[row, 2]],
                            d, phase_label=label, sm_handle=sm_handle)
        err_max_overall = max(err_max_overall, em)

    # Bottom row: spans all 3 cols — single timing/stats block
    ax_table = fig.add_subplot(gs[n_phases, :])
    render_timing_table(ax_table, phases, labels)

    # Title
    truth_resids = [p["meta"]["amgx_truth"]["rel_resid_actual"] for p in phases]
    title_lines = [
        f"Per-cell relative error — lab Xeon 32-rank dump  ({n_phases} phases)",
        f"Truth = AMGx tol=1e-12  |  rel_resid = "
        f"{', '.join(f'{l}: {r:.1e}' for l, r in zip(labels, truth_resids))}  "
        f"|  worst rel error = {err_max_overall:.2e}",
    ]
    fig.suptitle("\n".join(title_lines), fontsize=12, y=0.995)

    out = OUTDIR / f"{args.out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
