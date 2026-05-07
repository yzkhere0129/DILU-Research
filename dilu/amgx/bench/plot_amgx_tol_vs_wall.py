"""Plot AMGx wall vs tol — answer: how much speedup from relaxing tol?

Source: tol_sweep_results.json (21 bundles × 9 configs, lab 5060 RTX 5060
on senior initial-period 512K cells).

Includes OF DICPCG reference: 49.57 ms median (32-rank lab Xeon spot_melt,
historical baseline; same 500K-cell scale).

Key question: at OF's tol=1e-8, how does AMGx wall compare?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA = Path("/home/yzk/DILU-Research/dilu/amgx/bench/tol_sweep_results.json")

OF_REF_MS = 49.57   # OF DICPCG, 32-rank Xeon, spot_melt 500K cells
OF_TOL    = 1e-8


def main():
    d = json.load(DATA.open())
    cfgs = d["configs"]
    # Group by IR
    ir0 = sorted([c for c in cfgs if c["n_refine"] == 0],
                  key=lambda c: c["tol"])
    ir1 = sorted([c for c in cfgs if c["n_refine"] == 1],
                  key=lambda c: c["tol"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ====== Panel 1: solve_ms vs tol ======
    ax = axes[0, 0]
    if ir0:
        tols = [c["tol"] for c in ir0]
        solve = [c["solve_ms"] for c in ir0]
        ax.semilogx(tols, solve, "o-", color="tab:blue", markersize=8,
                     linewidth=2, label="AMGx (IR=0)")
    if ir1:
        tols = [c["tol"] for c in ir1]
        solve = [c["solve_ms"] for c in ir1]
        ax.semilogx(tols, solve, "s--", color="tab:cyan", markersize=8,
                     linewidth=2, label="AMGx + 1 IR")

    # OF reference
    ax.axhline(OF_REF_MS, color="tab:red", linestyle=":", linewidth=2,
                label=f"OF DICPCG (32-rank Xeon, {OF_REF_MS:.1f}ms)")
    # tol=1e-8 (OF's tol) vertical line
    ax.axvline(OF_TOL, color="tab:gray", linestyle=":", alpha=0.5,
                label="OF tol = 1e-8")

    ax.set_xlabel("AMGx tolerance", fontsize=11)
    ax.set_ylabel("median solve_ms (5060 RTX 5060)", fontsize=11)
    ax.set_title("AMGx wall vs tolerance\n(amortized solve, excludes setup)",
                  fontsize=11)
    ax.invert_xaxis()  # tighter tol on left? Actually default is tight on right,
                      # let's keep with tol from large to small
    ax.invert_xaxis()  # double invert = no-op, but for clarity want low tol→ left
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ====== Panel 2: iter vs tol ======
    ax = axes[0, 1]
    if ir0:
        tols = [c["tol"] for c in ir0]
        iters = [c["iter_med"] for c in ir0]
        ax.semilogx(tols, iters, "o-", color="tab:blue", markersize=8,
                     linewidth=2)
    ax.set_xlabel("AMGx tolerance", fontsize=11)
    ax.set_ylabel("median iter count", fontsize=11)
    ax.set_title("AMGx iter count vs tolerance", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    # ====== Panel 3: actual residual achieved vs tol ======
    ax = axes[1, 0]
    if ir0:
        tols = [c["tol"] for c in ir0]
        rels = [c["rel_resid_max"] for c in ir0]
        ax.loglog(tols, rels, "o-", color="tab:blue", markersize=8,
                   linewidth=2, label="IR=0 max actual residual")
    if ir1:
        tols = [c["tol"] for c in ir1]
        rels = [c["rel_resid_max"] for c in ir1]
        ax.loglog(tols, rels, "s--", color="tab:cyan", markersize=8,
                   linewidth=2, label="IR=1 max actual residual")
    # tol = actual diagonal (ideal)
    tol_range = [1e-15, 1e-3]
    ax.loglog(tol_range, tol_range, "k:", alpha=0.4,
               label="ideal: actual = tol")
    ax.axhline(1e-15, color="green", linestyle=":", alpha=0.5,
                label="machine ε")
    ax.set_xlabel("AMGx tolerance", fontsize=11)
    ax.set_ylabel("‖A·x - b‖₂/‖b‖₂  (max over 21 cases)", fontsize=11)
    ax.set_title("AMGx achieved residual vs requested tol", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ====== Panel 4: summary table ======
    ax = axes[1, 1]; ax.axis("off")
    lines = [
        "AMGx ON LAB 5060 (RTX 5060, sm_120, CUDA 13.2)",
        "Senior initial-period dump: 21 bundles, 512K cells, 3.5 M nnz",
        "",
        f"{'tol':>10s}  {'IR':>3s}  {'iter':>5s}  {'solve_ms':>9s}  {'rel_resid':>10s}",
        "─" * 50,
    ]
    for c in sorted(cfgs, key=lambda x: (-x["tol"], x["n_refine"])):
        lines.append(
            f"{c['tol']:>10.0e}  {c['n_refine']:>3d}  "
            f"{c['iter_med']:>5d}  {c['solve_ms']:>9.1f}  "
            f"{c['rel_resid_max']:>10.2e}"
        )
    lines += [
        "",
        f"OF DICPCG ref: {OF_REF_MS:.1f} ms (32-rank Xeon, spot_melt 500K)",
        "",
        "RELAXATION SPEEDUP (tol 1e-12 → 1e-8):",
    ]
    if any(c["tol"] == 1e-12 and c["n_refine"] == 0 for c in cfgs) and \
       any(c["tol"] == 1e-8 and c["n_refine"] == 0 for c in cfgs):
        c12 = next(c for c in cfgs if c["tol"] == 1e-12 and c["n_refine"] == 0)
        c8 = next(c for c in cfgs if c["tol"] == 1e-8 and c["n_refine"] == 0)
        lines += [
            f"  AMGx solve_ms: {c12['solve_ms']:.1f} → {c8['solve_ms']:.1f}  "
            f"({c12['solve_ms']/c8['solve_ms']:.2f}× faster)",
            f"  AMGx iter:    {c12['iter_med']} → {c8['iter_med']}  "
            f"({c12['iter_med']/c8['iter_med']:.2f}× fewer)",
        ]

    ax.text(0.0, 1.0, "\n".join(lines), fontfamily="monospace",
             fontsize=10, verticalalignment="top")

    fig.suptitle(
        "AMGx wall vs tolerance — RTX 5060 GPU\n"
        "(senior initial-period 512K cells, 21 bundles)",
        fontsize=13, y=0.995)
    fig.tight_layout()

    out = OUTDIR / "amgx_wall_vs_tol.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
