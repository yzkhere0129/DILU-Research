"""Generate distribution plots for OpenFOAM cross-check benchmark results.

Outputs PNGs into docs/benchmark/figures/distribution/ relative to the
repository root.  For each (solver, equation) pair that has rel_vs_OF data,
produces:
    1. Histogram of rel_vs_OF (log10 x-axis)
    2. Empirical CDF (log x-axis, vertical lines at P50/P95/P99/max)
    3. Scatter rel_vs_OF vs timestep (log y)
    4. Boxplot comparing all solvers side-by-side per equation

Usage:
    python3 plot_distributions.py <root>
    python3 plot_distributions.py <root> --eq pd
    python3 plot_distributions.py <root> --solver-config amgx_classical_v_diagscaled cusparse
    python3 plot_distributions.py <root> --outdir /path/to/figs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Re-use data collection from distribution_analysis
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
from distribution_analysis import collect  # noqa: E402


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

SOLVER_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]


def _solver_short(name: str) -> str:
    """Shorten long solver names for axis labels."""
    mapping = {
        "amgx_classical_v_diagscaled": "amgx_diagscaled",
        "amgx_classical_v_diagscaled_amortized": "amgx_diagscaled_amort",
        "amgx_classical_v_diagscaled_bicgstab_amortized": "amgx_bicgstab_amort",
        "amgx_classical_v_reg1e-02": "amgx_reg1e-02",
        "cusparse_bicgstab": "cusparse_bicgs",
        "multicolor_bicgstab": "multicolor_bicgs",
    }
    return mapping.get(name, name)


def _log_bins(vmin: float, vmax: float, n_bins: int = 30):
    """Return n_bins logarithmically spaced bin edges from vmin to vmax."""
    lo = np.floor(np.log10(max(vmin, 1e-20)))
    hi = np.ceil(np.log10(max(vmax, 1e-20)))
    return np.logspace(lo, hi, n_bins + 1)


def _percentile_lines(ax, arr, qs=(50, 95, 99), styles=None):
    """Draw vertical percentile lines on ax."""
    styles = styles or [
        ("P50", "k", "--", 0.8),
        ("P95", "#FF9800", "--", 0.9),
        ("P99", "#F44336", "--", 0.9),
    ]
    for q, (label, color, ls, alpha) in zip(qs, styles):
        val = np.percentile(arr, q)
        ax.axvline(val, color=color, linestyle=ls, alpha=alpha,
                   label=f"{label}={val:.1e}")
    val_max = arr.max()
    ax.axvline(val_max, color="#9C27B0", linestyle=":", alpha=0.9,
               label=f"max={val_max:.1e}")


# ---------------------------------------------------------------------------
# Individual (solver, eq) plots
# ---------------------------------------------------------------------------

def plot_histogram(arr: np.ndarray, solver: str, eq: str,
                   outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = _log_bins(arr.min(), arr.max(), 30)
    ax.hist(arr, bins=bins, color=SOLVER_COLORS[0], alpha=0.75, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("rel_vs_OF")
    ax.set_ylabel("count")
    ax.set_title(f"Histogram — {_solver_short(solver)} | eq={eq}\n"
                 f"N={len(arr)} | median={np.median(arr):.2e} | max={arr.max():.2e}")
    _percentile_lines(ax, arr)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = outdir / f"hist_{solver}_{eq}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


def plot_ecdf(arr: np.ndarray, solver: str, eq: str,
              outdir: Path) -> Path:
    sorted_arr = np.sort(arr)
    cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(sorted_arr, cdf, where="post", color=SOLVER_COLORS[1], linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("rel_vs_OF")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"ECDF — {_solver_short(solver)} | eq={eq}\n"
                 f"N={len(arr)} | median={np.median(arr):.2e} | max={arr.max():.2e}")
    for q, color, label in [(50, "k", "P50"), (95, "#FF9800", "P95"),
                             (99, "#F44336", "P99")]:
        val = np.percentile(arr, q)
        ax.axvline(val, color=color, linestyle="--", alpha=0.85,
                   label=f"{label}={val:.1e}")
        ax.axhline(q / 100.0, color=color, linestyle=":", alpha=0.3)
    ax.axvline(arr.max(), color="#9C27B0", linestyle=":", alpha=0.9,
               label=f"max={arr.max():.1e}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    fname = outdir / f"ecdf_{solver}_{eq}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


def plot_scatter_vs_ts(arr: np.ndarray, ts_labels: list[str],
                       solver: str, eq: str, outdir: Path) -> Path:
    """Scatter rel_vs_OF vs timestep index (sorted)."""
    ts_floats = []
    for ts in ts_labels:
        try:
            ts_floats.append(float(ts))
        except ValueError:
            ts_floats.append(float("nan"))
    ts_arr = np.array(ts_floats)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ts_arr, arr, color=SOLVER_COLORS[2], alpha=0.75, s=40, zorder=3)
    # Annotate worst 3 points
    top3_idx = np.argsort(arr)[-3:]
    for idx in top3_idx:
        ax.annotate(f"{arr[idx]:.1e}",
                    (ts_arr[idx], arr[idx]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="#F44336")
    ax.set_yscale("log")
    ax.set_xlabel("timestep (s)")
    ax.set_ylabel("rel_vs_OF")
    ax.set_title(f"rel_vs_OF vs timestep — {_solver_short(solver)} | eq={eq}")
    ax.axhline(np.percentile(arr, 95), color="#FF9800", linestyle="--",
               alpha=0.7, label=f"P95={np.percentile(arr,95):.1e}")
    ax.axhline(np.percentile(arr, 99), color="#F44336", linestyle="--",
               alpha=0.7, label=f"P99={np.percentile(arr,99):.1e}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = outdir / f"scatter_{solver}_{eq}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Per-equation comparison boxplot
# ---------------------------------------------------------------------------

def plot_boxplot_eq(groups: dict[tuple[str, str], list[dict]],
                    eq: str, outdir: Path) -> Path:
    """Side-by-side boxplot of rel_vs_OF for all solvers for a given equation."""
    solver_data = {}
    for (solver, grp_eq), rows in sorted(groups.items()):
        if grp_eq != eq:
            continue
        rv = [r["rel_vs_OF"] for r in rows if r.get("rel_vs_OF") is not None]
        if rv:
            solver_data[solver] = np.array(rv)

    if not solver_data:
        return None

    labels = list(solver_data.keys())
    data = [solver_data[s] for s in labels]
    short_labels = [_solver_short(s) for s in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2 + 2), 6))
    bp = ax.boxplot(data, labels=short_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(SOLVER_COLORS[i % len(SOLVER_COLORS)])
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_ylabel("rel_vs_OF")
    ax.set_title(f"rel_vs_OF comparison — eq={eq}")
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    fname = outdir / f"boxplot_eq_{eq}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate distribution plots for OpenFOAM cross-check results"
    )
    ap.add_argument("root", type=Path,
                    help="Root directory containing <TS>/<eq>_corr*/results/")
    ap.add_argument("--solver-config", nargs="+", dest="solver_config",
                    metavar="SOLVER",
                    help="Include only these solver labels")
    ap.add_argument("--eq", choices=["pd", "T"],
                    help="Filter to a single equation")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory for PNG files "
                         "(default: <repo_root>/docs/benchmark/figures/distribution/)")
    args = ap.parse_args()

    # Resolve output directory relative to repo root (3 dirs up from this file)
    if args.outdir is None:
        repo_root = Path(__file__).resolve().parents[3]
        outdir = repo_root / "docs" / "benchmark" / "figures" / "distribution"
    else:
        outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    groups = collect(args.root,
                     solver_filter=args.solver_config,
                     eq_filter=args.eq)
    if not groups:
        print("No results found under", args.root)
        sys.exit(1)

    generated = []

    # Per-solver-eq plots
    for (solver, eq), rows in sorted(groups.items()):
        rv = [r["rel_vs_OF"] for r in rows if r.get("rel_vs_OF") is not None]
        ts = [r["ts"] for r in rows if r.get("rel_vs_OF") is not None]
        if len(rv) < 2:
            continue
        arr = np.array(rv)
        print(f"Plotting {solver} / {eq} (N={len(arr)})")

        p = plot_histogram(arr, solver, eq, outdir)
        generated.append(p)
        print(f"  histogram -> {p}")

        p = plot_ecdf(arr, solver, eq, outdir)
        generated.append(p)
        print(f"  ecdf      -> {p}")

        p = plot_scatter_vs_ts(arr, ts, solver, eq, outdir)
        generated.append(p)
        print(f"  scatter   -> {p}")

    # Per-equation boxplots
    for eq in ["pd", "T"]:
        if args.eq and args.eq != eq:
            continue
        p = plot_boxplot_eq(groups, eq, outdir)
        if p is not None:
            generated.append(p)
            print(f"Boxplot eq={eq} -> {p}")

    print(f"\nGenerated {len(generated)} PNG files in {outdir}")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
