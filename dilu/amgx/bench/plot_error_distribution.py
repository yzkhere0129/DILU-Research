"""Plot AMGx error-magnitude × frequency histograms, per data slice.

Reads three pre-existing result JSONs and emits one PNG per "slice":
  - LPBF T matrices (48 dumps from Apr-May tests, scipy truth available)
  - Senior pd matrices (21 Initial_Period dumps, OF-replica + AMGx)
  - Tol-sweep summary (9 AMGx configs × 21 senior pd cases)

Each PNG: histogram of log10(error) on x-axis, count on y-axis,
multiple metrics overlaid as side-by-side panels.

Output: docs/benchmark/figures/amgx_error_dist_<slice>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


BENCH = Path("/home/yzk/DILU-Research/dilu/amgx/bench")
OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Common bin range covering ~1e-17 to ~1e-1, log10 scale
LOG_BINS = np.linspace(-17, -1, 33)


def _hist(ax, data, label, color):
    """Draw a single histogram of log10(|data|), ignoring NaN/zero."""
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return
    log_arr = np.log10(arr)
    ax.hist(log_arr, bins=LOG_BINS, alpha=0.55, label=label,
            color=color, edgecolor="black", linewidth=0.4)
    med = float(np.median(log_arr))
    ax.axvline(med, color=color, linestyle="--", linewidth=1.2)
    ax.text(med + 0.1, ax.get_ylim()[1] * 0.92,
            f"med={10**med:.1e}", color=color, fontsize=8)


def _decorate(ax, title, xlabel="log10(error magnitude)"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_xlim(-17, -1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)


# ---------------------------------------------------------------------
# Slice 1: LPBF 48 T-matrices — AMGx vs scipy truth
# ---------------------------------------------------------------------
def plot_lpbf():
    p = BENCH / "precision_results_ir1_v2.json"
    d = json.load(p.open())
    rows = d["rows"]
    print(f"[LPBF]  {len(rows)} matrices, tol={d['tol']}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # Panel A: residuals (||A·x - b||/||b||)
    ax = axes[0, 0]
    _hist(ax, [r["amgx_resid_rel2"] for r in rows], "AMGx (1 IR)", "tab:blue")
    _hist(ax, [r["of_final_res"]    for r in rows], "OpenFOAM final r",  "tab:orange")
    _hist(ax, [r["truth_resid_rel2"] for r in rows], "scipy truth",      "tab:gray")
    _decorate(ax, "Residual ‖A·x - b‖₂ / ‖b‖₂")

    # Panel B: error vs scipy ground truth
    ax = axes[0, 1]
    _hist(ax, [r["amgx_vs_truth"] for r in rows], "AMGx vs truth",  "tab:blue")
    _hist(ax, [r["of_vs_truth"]   for r in rows], "OF vs truth",    "tab:orange")
    _decorate(ax, "Error vs scipy SuperLU truth (rel)")

    # Panel C: AMGx vs OF (cross-check)
    ax = axes[1, 0]
    _hist(ax, [r["amgx_vs_OF"] for r in rows], "AMGx vs OF", "tab:green")
    _decorate(ax, "AMGx vs OpenFOAM (rel ‖xA - xOF‖∞ / ‖xOF‖∞)")

    # Panel D: AMGx setup + solve wall
    ax = axes[1, 1]
    walls = [(r.get("amgx_solve_s") or 0)*1e3 for r in rows]
    walls = [w for w in walls if w > 0]
    ax.hist(walls, bins=20, color="tab:purple", alpha=0.7,
            edgecolor="black", linewidth=0.4)
    ax.set_title(f"AMGx solve wall (ms) — n={len(walls)}", fontsize=10)
    ax.set_xlabel("wall (ms)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"AMGx error distribution — LPBF T-matrices (48 dumps, tol={d['tol']:.0e}, +1 IR)",
        fontsize=12, y=0.995)
    fig.tight_layout()
    out = OUTDIR / "amgx_error_dist_lpbf.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


# ---------------------------------------------------------------------
# Slice 2: Senior pd — 21 Initial_Period matrices
# ---------------------------------------------------------------------
def plot_senior_pd():
    p = BENCH / "byte_match_proof_results.json"
    d = json.load(p.open())
    rows = d["rows"]
    print(f"[senior pd]  {len(rows)} matrices")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    # Panel A: residuals — AMGx is 4-5 orders of magnitude tighter than replica
    ax = axes[0, 0]
    _hist(ax, [r["amgx_resid"]    for r in rows], "AMGx (1 IR, 1e-12)", "tab:blue")
    _hist(ax, [r["replica_resid"] for r in rows], "OF-replica (1e-12)", "tab:orange")
    _decorate(ax, "Residual ‖A·x - b‖₂ / ‖b‖₂")

    # Panel B: AMGx vs OF replica
    ax = axes[0, 1]
    _hist(ax, [r["rel_amgx_vs_replica"] for r in rows],
           "AMGx vs OF-replica", "tab:green")
    _decorate(ax, "rel ‖xA - xR‖∞ / ‖xR‖∞ (cross-validation)")

    # Panel C: vs senior xref — both AMGx and replica differ by SAME ~1.6%
    # (proof: senior xref has tol=1e-8 truncation × κ amplification, NOT solver error)
    ax = axes[1, 0]
    _hist(ax, [r["rel_amgx_vs_xref"]    for r in rows],
           "AMGx vs senior xref", "tab:blue")
    _hist(ax, [r["rel_replica_vs_xref"] for r in rows],
           "Replica vs senior xref", "tab:orange")
    _decorate(ax, "rel vs senior OF (tol=1e-8) xref — IDENTICAL gap = xref noise")

    # Panel D: replica iter count + solve wall
    ax = axes[1, 1]
    walls = np.array([r["replica_solve_s"]*1e3 for r in rows])
    iters = np.array([r["replica_iter"]        for r in rows])
    ax.hist(walls, bins=15, color="tab:orange", alpha=0.6,
            edgecolor="black", linewidth=0.4, label=f"replica wall (med {np.median(walls):.0f}ms)")
    ax.set_title(f"OF-replica solve wall (ms)\niter median={int(np.median(iters))}",
                 fontsize=10)
    ax.set_xlabel("wall (ms)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("AMGx error distribution — senior pd Initial_Period (21 dumps, κ≈10⁸, 512K cells)",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = OUTDIR / "amgx_error_dist_senior_pd.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


# ---------------------------------------------------------------------
# Slice 3: Tol sweep — 9 configs (tol × refine), each with 21 cases
# ---------------------------------------------------------------------
def plot_tol_sweep():
    p = BENCH / "tol_sweep_results.json"
    d = json.load(p.open())
    cfgs = d["configs"]
    print(f"[tol sweep]  {len(cfgs)} configs × {d['n_bundles']} cases")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Panel A: per-case rel_resid distribution per config (overlaid)
    ax = axes[0]
    cmap = plt.get_cmap("viridis", len(cfgs))
    for i, c in enumerate(cfgs):
        label = f"tol={c['tol']:.0e}, IR={c['n_refine']}"
        rels = [row["rel_resid"] for row in c["rows"]
                if isinstance(row.get("rel_resid"), (int, float)) and row["rel_resid"] > 0]
        if rels:
            log_rels = np.log10(rels)
            ax.hist(log_rels, bins=LOG_BINS, alpha=0.45,
                    label=label, color=cmap(i),
                    edgecolor="black", linewidth=0.3)
    _decorate(ax, "Per-case ‖A·x - b‖₂/‖b‖₂ across tol × IR sweep")

    # Panel B: per-config wall vs precision (scatter, gives the tradeoff curve)
    ax = axes[1]
    walls = [c["total_ms"]      for c in cfgs]
    rels  = [c["rel_resid_max"] for c in cfgs]
    iters = [c["iter_med"]      for c in cfgs]
    labels = [f"tol={c['tol']:.0e}/IR={c['n_refine']}" for c in cfgs]
    sc = ax.scatter(walls, rels, c=iters, cmap="plasma",
                    s=80, edgecolor="black", linewidth=0.7)
    for w, r, lb in zip(walls, rels, labels):
        ax.annotate(lb, (w, r), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("median total wall (ms)")
    ax.set_ylabel("max ‖A·x - b‖₂/‖b‖₂ across 21 cases")
    ax.set_title("AMGx tol/IR tradeoff — wall vs worst-case residual",
                 fontsize=10)
    ax.grid(alpha=0.3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("median iter count")

    fig.suptitle("AMGx tolerance + IR sweep — senior pd (21 matrices, 512K cells)",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = OUTDIR / "amgx_error_dist_tol_sweep.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


# ---------------------------------------------------------------------
# Slice 4: Tight match (loose vs tight) — proves OF tol=1e-8 is the noise
# ---------------------------------------------------------------------
def plot_loose_match():
    p = BENCH / "loose_match_results.json"
    if not p.exists():
        print("[loose match] skipped (file missing)")
        return
    d = json.load(p.open())
    rows = d["rows"]
    print(f"[loose match]  {len(rows)} matrices")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    _hist(ax, [r["amgx_loose_resid"]
               for r in rows if r.get("amgx_loose_resid", 0) > 0],
           "AMGx@1e-8 loose", "tab:blue")
    keys = [k for k in rows[0].keys() if "replica" in k and "resid" in k]
    if keys and any(r.get(keys[0], 0) > 0 for r in rows):
        _hist(ax, [r[keys[0]] for r in rows if r.get(keys[0], 0) > 0],
               f"OF-replica@1e-8 loose", "tab:orange")
    _decorate(ax, "Loose-tol residual (matches OF native tol=1e-8)")

    ax = axes[1]
    iters = [r.get("amgx_loose_iter") for r in rows
             if isinstance(r.get("amgx_loose_iter"), int)]
    if iters:
        ax.hist(iters, bins=range(min(iters), max(iters)+2),
                color="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.4)
        ax.set_title(f"AMGx loose iter count (med {int(np.median(iters))})",
                     fontsize=10)
        ax.set_xlabel("iters")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    fig.suptitle("AMGx loose match (tol=1e-8) — same precision as OF native",
                 fontsize=12)
    fig.tight_layout()
    out = OUTDIR / "amgx_error_dist_loose_match.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  → {out}")


def main():
    plot_lpbf()
    plot_senior_pd()
    plot_tol_sweep()
    plot_loose_match()
    print()
    print(f"All figures in: {OUTDIR}")


if __name__ == "__main__":
    main()
