"""4-panel statistical view of E11: OF vs LU truth on 100 LPBF pd matrices.

Reads:
  audit_overnight_20260509/xeon_validation/results/E11/per_matrix/<ts>_pd_corr0.json
  audit_overnight_20260509/xeon_validation/results/E11/aggregate.json

Output:
  docs/benchmark/figures/E11_OF_vs_LU_100matrix.png

Panels:
  (1) err_max histogram (log x)
  (2) err_max vs timestep scatter (physical-stage colored)
  (3) rel_max CDF
  (4) cells_above_100Pa histogram
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


E11_DIR = Path("/home/yzk/DILU-Research/audit_overnight_20260509/xeon_validation/results/E11")
PER_DIR = E11_DIR / "per_matrix"
OUT = Path("/home/yzk/DILU-Research/docs/benchmark/figures/E11_OF_vs_LU_100matrix.png")


def phase_of_t(t_s: float) -> str:
    """Physical stage labels for LPBF single-track simulation.
    Based on prior audit: melt forms ~3-5e-7s, evap-onset ~7e-7s, evap ~1e-6s."""
    if t_s < 3e-7:
        return "pre-melt"
    elif t_s < 6e-7:
        return "melt"
    elif t_s < 9e-7:
        return "evap-early"
    else:
        return "evap-late"


PHASE_COLOR = {
    "pre-melt":  "#888",
    "melt":      "#1f77b4",
    "evap-early":"#ff7f0e",
    "evap-late": "#d62728",
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Load all per-matrix
    rows = []
    for p in sorted(PER_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        cmp = d["comparison_OF_vs_LU"]
        t = float(d["timestep"])
        rows.append({
            "t": t,
            "phase": phase_of_t(t),
            "err_max": float(cmp["max_abs_diff_Pa"]),
            "rel_max": float(cmp["rel_max"]),
            "cells_100": int(cmp["cells_above_100Pa"]),
            "cells_1k":  int(cmp["cells_above_1kPa"]),
            "median":    float(cmp["median_abs_diff"]),
            "of_resid":  float(cmp["x_OF_rel_resid"]),
        })
    rows.sort(key=lambda r: r["t"])
    print(f"loaded {len(rows)} per-matrix results")

    err_max = np.array([r["err_max"] for r in rows])
    rel_max = np.array([r["rel_max"] for r in rows])
    cells_100 = np.array([r["cells_100"] for r in rows])
    cells_1k = np.array([r["cells_1k"] for r in rows])
    times = np.array([r["t"] for r in rows])
    phases = [r["phase"] for r in rows]
    of_resid = np.array([r["of_resid"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax_hist, ax_scat, ax_cdf, ax_cell = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # (1) err_max histogram
    bins = np.logspace(0, 3.5, 30)
    ax_hist.hist(err_max, bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax_hist.axvline(np.mean(err_max), color="red", linestyle="--",
                     label=f"mean {np.mean(err_max):.0f} Pa")
    ax_hist.axvline(np.median(err_max), color="green", linestyle="--",
                     label=f"median {np.median(err_max):.0f} Pa")
    ax_hist.axvline(100, color="orange", linestyle=":", alpha=0.6, label="100 Pa")
    ax_hist.axvline(1000, color="purple", linestyle=":", alpha=0.6, label="1 kPa")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("max |x_OF − x_LU| (Pa) — log")
    ax_hist.set_ylabel("# matrices")
    ax_hist.set_title(f"OF vs LU truth: err_max distribution (n=100)\n"
                       f"mean {np.mean(err_max):.0f} Pa, max {np.max(err_max):.0f} Pa, "
                       f"p99 {np.percentile(err_max,99):.0f} Pa")
    ax_hist.legend(fontsize=8); ax_hist.grid(True, alpha=0.3)

    # (2) err_max vs timestep, colored by phase
    for phase in PHASE_COLOR:
        mask = np.array([p == phase for p in phases])
        if mask.sum() == 0: continue
        ax_scat.scatter(times[mask]*1e6, err_max[mask],
                         color=PHASE_COLOR[phase], label=f"{phase} (n={mask.sum()})",
                         alpha=0.7, s=25, edgecolor="black", linewidth=0.3)
    ax_scat.set_yscale("log")
    ax_scat.set_xlabel("timestep (μs)")
    ax_scat.set_ylabel("err_max (Pa) — log")
    ax_scat.set_title("OF err vs physical stage\n(melt formation 3-6e-7, evap 6e-7-1e-6)")
    ax_scat.legend(fontsize=8); ax_scat.grid(True, alpha=0.3)
    ax_scat.axhline(100,  color="orange", linestyle=":", alpha=0.5)
    ax_scat.axhline(1000, color="purple", linestyle=":", alpha=0.5)

    # (3) rel_max CDF
    sorted_rel = np.sort(rel_max)
    cdf = np.arange(1, len(sorted_rel)+1) / len(sorted_rel)
    ax_cdf.plot(sorted_rel, cdf, color="#1f77b4", linewidth=1.5)
    ax_cdf.set_xscale("log")
    for q, name in [(0.5, "median"), (0.95, "p95"), (0.99, "p99")]:
        v = np.percentile(rel_max, q*100)
        ax_cdf.axvline(v, color="red", linestyle=":", alpha=0.5)
        ax_cdf.annotate(f"{name}\n{v:.1e}", xy=(v, q),
                         xytext=(5, -15 if name=="median" else 5),
                         textcoords="offset points", fontsize=8, color="red")
    ax_cdf.set_xlabel("rel_max = max|Δ| / max|x_LU| — log")
    ax_cdf.set_ylabel("cumulative fraction of matrices")
    ax_cdf.set_title(f"OF vs LU rel_max CDF\n"
                      f"median {np.median(rel_max):.1e}, p95 {np.percentile(rel_max,95):.1e}, "
                      f"p99 {np.percentile(rel_max,99):.1e}")
    ax_cdf.grid(True, alpha=0.3)

    # (4) cells_above_100Pa histogram
    has_100 = (cells_100 > 0).sum()
    has_1k = (cells_1k > 0).sum()
    bins = np.concatenate([[0], np.logspace(0, 5, 25)])
    cells_100_for_plot = np.where(cells_100 == 0, 0.5, cells_100)
    ax_cell.hist(cells_100_for_plot, bins=bins, color="#ff7f0e",
                  edgecolor="black", linewidth=0.4)
    ax_cell.set_xscale("symlog", linthresh=1)
    ax_cell.set_xlabel("# cells where |Δ| > 100 Pa — log (0 at 0.5)")
    ax_cell.set_ylabel("# matrices")
    ax_cell.set_title(f"Cells exceeding 100 Pa per matrix\n"
                       f"{has_100}/100 matrices have any cell > 100Pa  "
                       f"({has_1k}/100 have cell > 1kPa)")
    ax_cell.grid(True, alpha=0.3)

    n_clean = (cells_100 == 0).sum()
    fig.suptitle(
        f"E11 — OF DICPCG @ tol=1e-8 vs CHOLMOD LU truth on 100 dense_track LPBF pd matrices  "
        f"(Xeon 32-thread CHOLMOD, 11.6h wall)\n"
        f"{n_clean}/100 matrices fully clean (no cell > 100Pa)  |  "
        f"x_OF resid max {np.max(of_resid):.2e} (sometimes exceeds tol=1e-8)  |  "
        f"x_LU resid max 1.6e-14 (truth confirmed)",
        fontsize=11, y=0.995
    )
    plt.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT}")

    # Print a stat summary
    print(f"\nSummary (n=100):")
    print(f"  err_max:     mean {np.mean(err_max):>7.1f} Pa  median {np.median(err_max):>7.1f}  "
          f"p95 {np.percentile(err_max,95):>7.1f}  p99 {np.percentile(err_max,99):>7.1f}  max {np.max(err_max):>7.1f}")
    print(f"  rel_max:     mean {np.mean(rel_max):.2e}  median {np.median(rel_max):.2e}  "
          f"p95 {np.percentile(rel_max,95):.2e}  p99 {np.percentile(rel_max,99):.2e}")
    print(f"  cells>100Pa: matrices_with_any={has_100}/100  max_per_matrix={np.max(cells_100)}")
    print(f"  cells>1kPa:  matrices_with_any={has_1k}/100   max_per_matrix={np.max(cells_1k)}")
    print(f"  x_OF resid:  max {np.max(of_resid):.2e}  median {np.median(of_resid):.2e}")
    print(f"\nPhase counts:")
    for phase in PHASE_COLOR:
        n = sum(1 for p in phases if p == phase)
        if n == 0: continue
        sub_err = err_max[[p == phase for p in phases]]
        print(f"  {phase:<12} n={n:3d}  err_max mean={np.mean(sub_err):>6.1f} max={np.max(sub_err):>7.1f}")


if __name__ == "__main__":
    main()
