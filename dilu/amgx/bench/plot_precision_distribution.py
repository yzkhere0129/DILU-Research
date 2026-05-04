"""Plot AMGx precision distribution vs OpenFOAM, from precision_results_ir1.json.

Outputs a 6-panel figure to dilu/amgx/bench/precision_distribution.png:
  Top row:  histogram of rel_vs_truth (AMGx, OF, AMGx-vs-OF), log-scale
  Mid row:  per-(case,eq) bars of AMGx-vs-truth + OF-vs-truth
  Bot row:  scatter (OF vs AMGx, on truth-relative axis) + iter histogram
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(results_path: Path, out_path: Path) -> None:
    with results_path.open() as fh:
        data = json.load(fh)
    rows = data["rows"]

    pd = [r for r in rows if r["eq"].lower().startswith("pd") and "amgx_vs_truth" in r]
    T  = [r for r in rows if r["eq"].lower().startswith("t")  and "amgx_vs_truth" in r]

    avt_pd = [r["amgx_vs_truth"] for r in pd]
    ovt_pd = [r["of_vs_truth"]   for r in pd if "of_vs_truth" in r]
    avo_pd = [r["amgx_vs_OF"]    for r in pd if "amgx_vs_OF" in r]
    avt_T  = [r["amgx_vs_truth"] for r in T]
    ovt_T  = [r["of_vs_truth"]   for r in T if "of_vs_truth" in r]
    avo_T  = [r["amgx_vs_OF"]    for r in T if "amgx_vs_OF" in r]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # === Row 1: histograms (log scale) ===
    bins = np.logspace(-16, -3, 27)
    ax = axes[0, 0]
    ax.hist(avt_pd, bins=bins, alpha=0.7, color="C0", label=f"AMGx vs truth (N={len(avt_pd)})")
    ax.hist(ovt_pd, bins=bins, alpha=0.7, color="C3", label=f"OF vs truth (N={len(ovt_pd)})")
    ax.set_xscale("log"); ax.set_xlim(1e-16, 1e-3)
    ax.set_xlabel("relative ∞-norm error vs scipy spsolve truth")
    ax.set_ylabel("count")
    ax.set_title(f"pd matrices ({len(pd)} dumps)")
    ax.axvline(1e-10, color="k", ls="--", alpha=0.5, label="1e-10 user spec")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(avt_T, bins=bins, alpha=0.7, color="C0", label=f"AMGx vs truth (N={len(avt_T)})")
    ax.hist(ovt_T, bins=bins, alpha=0.7, color="C3", label=f"OF vs truth (N={len(ovt_T)})")
    ax.set_xscale("log"); ax.set_xlim(1e-16, 1e-3)
    ax.set_xlabel("relative ∞-norm error vs scipy spsolve truth")
    ax.set_ylabel("count")
    ax.set_title(f"T matrices ({len(T)} dumps)")
    ax.axvline(1e-10, color="k", ls="--", alpha=0.5, label="1e-10 user spec")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # === Row 2: per-(case,ts,eq) bars ===
    ax = axes[1, 0]
    labels_pd = [f"{r['case'][:7]}/{r['ts'][:8]}/{r['eq'][-5:]}" for r in pd]
    x = np.arange(len(pd))
    w = 0.35
    ax.bar(x - w/2, avt_pd, w, color="C0", label="AMGx vs truth", log=True)
    ax.bar(x + w/2, ovt_pd, w, color="C3", label="OF vs truth", log=True)
    ax.axhline(1e-10, color="k", ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_pd, rotation=90, fontsize=6)
    ax.set_ylabel("rel error (log)")
    ax.set_title("pd: per-dump AMGx vs OF (vs truth)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    labels_T = [f"{r['case'][:7]}/{r['ts'][:8]}" for r in T]
    x = np.arange(len(T))
    ax.bar(x - w/2, avt_T, w, color="C0", label="AMGx vs truth", log=True)
    ax.bar(x + w/2, ovt_T, w, color="C3", label="OF vs truth", log=True)
    ax.axhline(1e-10, color="k", ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels_T, rotation=90, fontsize=6)
    ax.set_ylabel("rel error (log)")
    ax.set_title("T: per-dump AMGx vs OF (vs truth)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # === Row 3 left: scatter AMGx-vs-truth (x) vs OF-vs-truth (y) ===
    ax = axes[2, 0]
    ax.loglog(avt_pd, ovt_pd, "o", color="C0", alpha=0.7, label=f"pd (N={len(avt_pd)})")
    ax.loglog(avt_T,  ovt_T,  "s", color="C3", alpha=0.7, label=f"T  (N={len(avt_T)})")
    lo, hi = 1e-16, 1e-3
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.axvline(1e-10, color="gray", ls=":", alpha=0.5, label="user spec 1e-10")
    ax.set_xlabel("AMGx vs truth (rel ∞-norm)")
    ax.set_ylabel("OpenFOAM vs truth (rel ∞-norm)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title("Scatter: AMGx accuracy vs OpenFOAM accuracy\n(both measured against scipy direct LU)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # === Row 3 right: AMGx iters histogram ===
    ax = axes[2, 1]
    pd_iters = [r["amgx_iters"] for r in pd]
    T_iters  = [r["amgx_iters"] for r in T]
    bins_i = np.arange(0, max(max(pd_iters), max(T_iters)) + 2)
    ax.hist(pd_iters, bins=bins_i, alpha=0.6, color="C0", label=f"pd (median={int(np.median(pd_iters))})")
    ax.hist(T_iters,  bins=bins_i, alpha=0.6, color="C3", label=f"T (median={int(np.median(T_iters))})")
    ax.set_xlabel("AMGx primary outer iterations")
    ax.set_ylabel("count")
    ax.set_title("AMGx iteration count distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"AMGx + 1 IR step precision distribution (48 LPBF dumps, tol=1e-12)\n"
        f"AMGx vs truth: max={max(avt_pd+avt_T):.2e}, median={np.median(avt_pd+avt_T):.2e}  "
        f"|  OF vs truth: max={max(ovt_pd+ovt_T):.2e}, median={np.median(ovt_pd+ovt_T):.2e}",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")

    # Summary stats in text form too
    print(f"\nsummary:")
    print(f"  pd  AMGx vs truth: max={max(avt_pd):.2e}  median={np.median(avt_pd):.2e}  min={min(avt_pd):.2e}")
    print(f"  pd  OF   vs truth: max={max(ovt_pd):.2e}  median={np.median(ovt_pd):.2e}  min={min(ovt_pd):.2e}")
    print(f"  T   AMGx vs truth: max={max(avt_T):.2e}  median={np.median(avt_T):.2e}  min={min(avt_T):.2e}")
    print(f"  T   OF   vs truth: max={max(ovt_T):.2e}  median={np.median(ovt_T):.2e}  min={min(ovt_T):.2e}")


if __name__ == "__main__":
    base = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
    main(base / "precision_results_ir1.json",
         base / "precision_distribution.png")
