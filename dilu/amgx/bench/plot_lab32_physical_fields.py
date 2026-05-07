"""3D physical-field cloud plot consuming lab 5060 prepared npz.

Visualizes the actual pd (pressure, Pa) and T (temperature, K) fields from
melting + evaporation phases — NOT errors. Uses x_truth (AMGx tol=1e-12)
as the displayed value.

Layout (2 phases × 2 quantities):
    [melting pd cloud]      [melting T cloud]
    [evaporation pd cloud]  [evaporation T cloud]

Each panel: 3D scatter, color = physical value, alpha proportional to
deviation from background (so gas-phase fades, melt pool / vapor zone pops).

Usage:
    python -m dilu.amgx.bench.plot_lab32_physical_fields \\
        --melting-pd     dilu/amgx/bench/lab32_plot_melting_pd_corr0_3.8e-07.npz \\
        --melting-T      dilu/amgx/bench/lab32_plot_melting_T_corr0_3.8e-07.npz \\
        --evaporation-pd dilu/amgx/bench/lab32_plot_evaporation_pd_corr0_9.0e-07.npz \\
        --evaporation-T  dilu/amgx/bench/lab32_plot_evaporation_T_corr0_9.0e-07.npz

Any subset is fine — missing panels are skipped. Output:
    docs/benchmark/figures/lab32_physical_fields.png
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


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    return dict(
        x_truth = z["x_truth"],
        x_OF    = z["x_OF"],
        i = z["i"], j = z["j"], k = z["k"],
        nx = int(z["nx"][0]), ny = int(z["ny"][0]), nz = int(z["nz"][0]),
        dx = float(z["dx"][0]),
        meta = meta,
    )


def scatter3d_field(ax, d, *, title, cmap, units, max_points=40000,
                      elev=22, azim=-55, z_window=None, gamma=1.0,
                      vmin=None, vmax=None, value_kind="pressure"):
    """3D scatter of physical field with alpha=deviation-from-background.

    value_kind: "pressure" → background = median(x_truth) (gas-phase pressure)
                "temperature" → background = min(x_truth) (cold ambient)
    """
    val = d["x_truth"]
    nx, ny, nz, dx = d["nx"], d["ny"], d["nz"], d["dx"]
    i, j, k = d["i"], d["j"], d["k"]

    if z_window is None:
        z_lo, z_hi = 0, nz * dx * 1e6
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
    vs = val[idx]

    # Color
    if vmin is None: vmin = float(np.percentile(val, 1))
    if vmax is None: vmax = float(np.percentile(val, 99))
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap(cmap)(norm(vs))

    # Alpha = how far from background, normalized then γ-shaped
    if value_kind == "pressure":
        bg = float(np.median(val))
    else:  # temperature
        bg = float(np.percentile(val, 5))   # ambient cold tail
    dev = np.abs(vs - bg)
    dev_max = max(dev.max(), 1e-12)
    alpha = np.clip((dev / dev_max) ** gamma, 0.05, 0.95)
    colors[:, 3] = alpha

    ax.scatter(xs, ys, zs, c=colors, s=6, marker="o",
                linewidth=0, depthshade=True)
    nx_um = nx * dx * 1e6
    ny_um = ny * dx * 1e6
    ax.set_xlim(0, nx_um); ax.set_ylim(0, ny_um); ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((nx_um, ny_um, z_hi - z_lo))
    ax.set_xlabel("x (μm)", fontsize=8)
    ax.set_ylabel("y (μm)", fontsize=8)
    ax.set_zlabel("z (μm)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(vs)
    return sm, vmin, vmax


def render_panel(fig, gs_cell, d, *, title, cmap, units, value_kind):
    ax = fig.add_subplot(gs_cell, projection='3d')
    nz_um = d["nz"] * d["dx"] * 1e6
    z_window = (0, nz_um) if nz_um <= 250 else (60, 180)

    sm, vmin, vmax = scatter3d_field(
        ax, d, title=title, cmap=cmap, units=units,
        value_kind=value_kind, z_window=z_window)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, fraction=0.025, pad=0.03)
    cb.set_label(units, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--melting-pd", required=False)
    ap.add_argument("--melting-T",  required=False)
    ap.add_argument("--evaporation-pd", required=False)
    ap.add_argument("--evaporation-T",  required=False)
    ap.add_argument("--out-name", default="lab32_physical_fields")
    args = ap.parse_args()

    # 4 cells: rows = phases, cols = (pd, T)
    cells = [
        ("melting",    "pd", args.melting_pd,     "viridis", "pd  (Pa)",     "pressure"),
        ("melting",    "T",  args.melting_T,      "inferno", "T  (K)",       "temperature"),
        ("evaporation","pd", args.evaporation_pd, "viridis", "pd  (Pa)",     "pressure"),
        ("evaporation","T",  args.evaporation_T,  "inferno", "T  (K)",       "temperature"),
    ]

    n_avail = sum(1 for c in cells if c[2])
    if n_avail == 0:
        raise SystemExit("Provide at least one of "
                          "--melting-pd / --melting-T / "
                          "--evaporation-pd / --evaporation-T")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)

    rendered_meta = []
    for cell in cells:
        phase, kind, path, cmap, units, value_kind = cell
        if not path:
            continue
        d = load_npz(Path(path))
        m = d["meta"]
        # Sanity: file's eq should match expected kind
        eq_in_file = m.get("eq", "?")
        truth_resid = m["amgx_truth"]["rel_resid_actual"]
        truth_iters = m["amgx_truth"]["iters"]
        # Position in 2x2 grid
        row = 0 if phase == "melting" else 1
        col = 0 if kind == "pd" else 1
        title = (f"[{phase}, {kind}]   t = {m['time']}\n"
                 f"x_truth = AMGx tol=1e-12  (iter={truth_iters}, "
                 f"rel_resid={truth_resid:.1e})\n"
                 f"min={d['x_truth'].min():.2e}, max={d['x_truth'].max():.2e}")
        render_panel(fig, gs[row, col], d, title=title, cmap=cmap,
                       units=units, value_kind=value_kind)
        rendered_meta.append((phase, kind, m))

    fig.suptitle(
        "Lab Xeon 32-rank dump — physical fields  (truth = AMGx tol=1e-12)\n"
        "color = physical value | alpha = deviation from background",
        fontsize=12, y=0.995)

    out = OUTDIR / f"{args.out_name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")
    print(f"\nRendered {len(rendered_meta)} panels:")
    for phase, kind, m in rendered_meta:
        print(f"  [{phase:<11s} {kind:<2s}]  N={m['n_global']}, "
              f"truth iter={m['amgx_truth']['iters']}, "
              f"rel_resid={m['amgx_truth']['rel_resid_actual']:.1e}, "
              f"t_solve={m['amgx_truth']['t_solve_s']*1e3:.1f} ms")


if __name__ == "__main__":
    main()
