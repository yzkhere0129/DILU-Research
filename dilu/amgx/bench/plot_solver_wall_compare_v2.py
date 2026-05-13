"""v2: 6-solver wall comparison adding 5060 amortized series.

Reads two sources:
  - dilu/amgx/bench/solver_comparison_500K.json  (4 original solvers, dev 3050 fresh)
  - <5060-results>/replay_amortized_e8.npz + replay_amortized_e12_IR.npz  (lab 5060)

6 solver series on 6 single_track timesteps:
  1. OF DICPCG @ 1e-8                    (Xeon 32-core MPI; wall_estimated)
  2. AMGx PCG @ 1e-8        — fresh      (dev RTX 3050)
  3. AMGx PCG @ 1e-8        — amortized  (lab RTX 5060)        ← NEW
  4. AMGx PCG @ 1e-12 + IR  — fresh      (dev RTX 3050)
  5. AMGx PCG @ 1e-12 + IR  — amortized  (lab RTX 5060)        ← NEW
  6. CHOLMOD LU             — direct     (dev 16-thread BLAS)

Usage:
    PYTHONPATH=. python dilu/amgx/bench/plot_solver_wall_compare_v2.py \\
        --replay-dir audit_overnight_20260509/lab_5060_replay/results_smoke \\
        --output     docs/benchmark/figures/solver_wall_compare_500K_with_5060.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JSON_PATH = Path("/home/yzk/DILU-Research/dilu/amgx/bench/solver_comparison_500K.json")

# Map JSON timestep → replay_npz timestep string (already aligned)
TS_ORDER = ["3.2e-07", "3.8e-07", "4.1e-07", "7e-07", "9e-07", "1.06e-06"]
LABEL_BY_TS = {
    "3.2e-07":  "320 ns\n(melt)",
    "3.8e-07":  "380 ns\n(melt)",
    "4.1e-07":  "410 ns\n(melt)",
    "7e-07":    "700 ns\n(evap_e)",
    "9e-07":    "900 ns\n(evap)",
    "1.06e-06": "1060 ns\n(evap_l)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True,
                    help="dir containing replay_amortized_e8.npz, replay_amortized_e12_IR.npz")
    ap.add_argument("--output",     required=True)
    args = ap.parse_args()

    replay_dir = Path(args.replay_dir).expanduser()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load JSON (4 original solvers) ---
    with open(JSON_PATH) as f:
        rows = json.load(f)
    # index by ts
    rows_by_ts = {r["timestep"]: r for r in rows}

    # --- 2. Load 5060 replay npz ---
    def load_npz(name):
        p = replay_dir / f"replay_{name}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        return np.load(p, allow_pickle=False)
    z_am_e8  = load_npz("amortized_e8")
    z_am_e12 = load_npz("amortized_e12_IR")
    # Map ts→index
    ts_5060 = list(z_am_e8["ts"])
    idx_5060 = {t: i for i, t in enumerate(ts_5060)}

    # --- 3. Build 6-series arrays aligned to TS_ORDER ---
    n_ts = len(TS_ORDER)
    def get_5060(z, ts):
        i = idx_5060.get(ts)
        if i is None:
            return np.nan, np.nan, 0
        # total = setup + update + solve + ir; wall in seconds
        total = (float(z["setup_s"][i]) + float(z["update_s"][i])
                 + float(z["solve_s"][i]) + float(z["ir_s"][i]))
        resid = float(z["rel_resid"][i])
        iters = int(z["iters"][i])
        return total, resid, iters

    series = []  # (name, color, wall_s[n_ts], iter_or_dash[n_ts], err_max_Pa[n_ts])

    # (1) OF DICPCG
    of_w = [rows_by_ts[t]["OF"]["wall_ms"]/1000.0 for t in TS_ORDER]
    of_i = [rows_by_ts[t]["OF"]["iter"]            for t in TS_ORDER]
    of_e = [rows_by_ts[t]["OF"]["err_max_Pa"]      for t in TS_ORDER]
    series.append(("OF DICPCG @ 1e-8\n(Xeon 32-core MPI)", "#1f77b4", of_w, of_i, of_e))

    # (2) AMGx@e8 fresh dev
    w = [rows_by_ts[t]["AMGx_e8"]["wall_ms"]/1000.0 for t in TS_ORDER]
    i = [rows_by_ts[t]["AMGx_e8"]["iter"]            for t in TS_ORDER]
    e = [rows_by_ts[t]["AMGx_e8"]["err_max_Pa"]      for t in TS_ORDER]
    series.append(("AMGx@1e-8 fresh\n(dev 3050)", "#aec7e8", w, i, e))

    # (3) AMGx@e8 amortized 5060   ← NEW
    w, i, e = [], [], []
    for t in TS_ORDER:
        ww, rr, it = get_5060(z_am_e8, t)
        w.append(ww); i.append(it)
        # err_max not stored — proxy = wall ratio * fresh-3050 err is invalid.
        # Use rel_resid * |x|_inf as a rough Pa proxy or just leave nan.
        e.append(np.nan)
    series.append(("AMGx@1e-8 amortized\n(lab 5060)", "#ff7f0e", w, i, e))

    # (4) AMGx@e12+IR fresh dev
    w = [rows_by_ts[t]["AMGx_e12_IR"]["wall_ms"]/1000.0 for t in TS_ORDER]
    i = [rows_by_ts[t]["AMGx_e12_IR"]["iter"]            for t in TS_ORDER]
    e = [rows_by_ts[t]["AMGx_e12_IR"]["err_max_Pa"]      for t in TS_ORDER]
    series.append(("AMGx@1e-12+IR fresh\n(dev 3050)", "#98df8a", w, i, e))

    # (5) AMGx@e12+IR amortized 5060   ← NEW
    w, i, e = [], [], []
    for t in TS_ORDER:
        ww, rr, it = get_5060(z_am_e12, t)
        w.append(ww); i.append(it)
        e.append(np.nan)
    series.append(("AMGx@1e-12+IR amortized\n(lab 5060)", "#2ca02c", w, i, e))

    # (6) LU CHOLMOD
    w = [rows_by_ts[t]["LU_CHOLMOD"]["wall_ms"]/1000.0 for t in TS_ORDER]
    i = ["LU"] * n_ts
    e = [0.0] * n_ts
    series.append(("CHOLMOD LU direct\n(dev 16-thread BLAS)", "#d62728", w, i, e))

    # Sort series by mean wall (ascending: fastest left → slowest right)
    series_with_keys = [(s, float(np.nanmean(np.asarray(s[2], dtype=float)))) for s in series]
    series_with_keys.sort(key=lambda t: t[1])
    series = [s for s, _ in series_with_keys]
    print("\nSeries ordered by mean wall (asc):")
    for s, mw in series_with_keys:
        print(f"  {mw:7.2f}s  {s[0].replace(chr(10), ' ')}")

    n_solvers = len(series)
    # --- 4. Plot ---
    fig, (ax_wall, ax_err) = plt.subplots(1, 2, figsize=(20, 7))

    # Panel 1: wall log-scale grouped bars
    x = np.arange(n_ts)
    bar_w = 0.13
    for s_idx, (name, color, walls, iters, _) in enumerate(series):
        offset = (s_idx - (n_solvers - 1) / 2) * bar_w
        bars = ax_wall.bar(x + offset, walls, bar_w, label=name,
                            color=color, edgecolor="black", linewidth=0.4)
        for b, v, it in zip(bars, walls, iters):
            if np.isnan(v) or v <= 0: continue
            txt = f"{v:.1f}s\n({it})" if it != "LU" else f"{v:.1f}s\n(LU)"
            ax_wall.text(b.get_x() + b.get_width()/2, v * 1.05, txt,
                          ha="center", va="bottom", fontsize=6.5)

    ax_wall.set_yscale("log")
    ax_wall.set_xticks(x)
    ax_wall.set_xticklabels([LABEL_BY_TS[t] for t in TS_ORDER], fontsize=8)
    ax_wall.set_ylabel("Wall time (s) — log scale", fontsize=10)
    ax_wall.set_title("Single-solve wall on 500K LPBF pd matrix\n"
                       "(top number = wall s, bottom = iter count)", fontsize=11)
    ax_wall.legend(loc="upper center", fontsize=7.5, framealpha=0.92, ncol=2)
    ax_wall.grid(True, axis="y", which="both", alpha=0.3)
    ax_wall.set_ylim(0.3, 300)

    # Panel 2: err vs LU truth (log)
    err_floor = 1e-7
    for s_idx, (name, color, _, _, errs) in enumerate(series):
        offset = (s_idx - (n_solvers - 1) / 2) * bar_w
        plot_e = [err_floor if (np.isnan(e) or e == 0) else max(e, err_floor) for e in errs]
        bars = ax_err.bar(x + offset, plot_e, bar_w, label=name,
                           color=color, edgecolor="black", linewidth=0.4)
        for b, e in zip(bars, errs):
            if np.isnan(e):
                txt = "n/a"
                y = err_floor * 1.4
            elif e == 0:
                txt = "0"
                y = err_floor * 1.4
            elif e < 1e-3:
                txt = f"{e:.1e}"; y = e * 1.4
            elif e < 1:
                txt = f"{e:.3g}"; y = e * 1.4
            else:
                txt = f"{e:.1f}"; y = e * 1.4
            ax_err.text(b.get_x() + b.get_width()/2, y, txt,
                         ha="center", va="bottom", fontsize=6.5)
    ax_err.set_yscale("log")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels([LABEL_BY_TS[t] for t in TS_ORDER], fontsize=8)
    ax_err.set_ylabel("max |Δ| vs CHOLMOD LU truth (Pa) — log", fontsize=10)
    ax_err.set_title("Solution error vs LU truth\n"
                      "(LU at floor 1e-7; 5060 amortized err not measured — n/a)",
                      fontsize=11)
    ax_err.legend(loc="upper center", fontsize=7.5, framealpha=0.92, ncol=2)
    ax_err.grid(True, axis="y", which="both", alpha=0.3)
    ax_err.set_ylim(err_floor * 0.5, 200)
    ax_err.axhline(1.0,  color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_err.text(n_ts - 0.5, 1.0,  " 1 Pa",  fontsize=7, va="center", color="gray")
    ax_err.axhline(10.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_err.text(n_ts - 0.5, 10.0, " 10 Pa", fontsize=7, va="center", color="gray")

    # Stats footer
    means = {s[0].split("\n")[0]: float(np.nanmean(s[2])) for s in series}
    foot = "  |  ".join([f"{k}: {v:.2f}s" for k, v in means.items()])
    fig.suptitle(
        f"6-solver comparison on 500K LPBF pd, 6 single_track timesteps  "
        f"(adding lab RTX 5060 amortized measurements)\n"
        f"mean wall: {foot}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # Print quantitative table
    print("\nWall summary (s, mean over 6 timesteps):")
    print(f"  {'solver':<48} {'mean':>8} {'median':>8} {'min':>8} {'max':>8}")
    for name, _, walls, _, _ in series:
        ws = np.array([w for w in walls if not np.isnan(w)])
        if len(ws) == 0:
            continue
        n = name.replace("\n", " ")
        print(f"  {n:<48} {np.mean(ws):>8.2f} {np.median(ws):>8.2f} "
              f"{np.min(ws):>8.2f} {np.max(ws):>8.2f}")

    # Speedups
    print("\nKey speedups:")
    of_mean    = means["OF DICPCG @ 1e-8"]
    am8_5060   = means["AMGx@1e-8 amortized"]
    am12_5060  = means["AMGx@1e-12+IR amortized"]
    lu_mean    = means["CHOLMOD LU direct"]
    fresh_e8   = means["AMGx@1e-8 fresh"]
    fresh_e12  = means["AMGx@1e-12+IR fresh"]
    print(f"  5060 amortized_e8 vs LU CHOLMOD:        {lu_mean/am8_5060:.1f}× faster")
    print(f"  5060 amortized_e8 vs OF wall_estimate:  {of_mean/am8_5060:.2f}× faster")
    print(f"  5060 amortized_e8 vs dev fresh_e8:      {fresh_e8/am8_5060:.2f}× faster (GPU + amort)")
    print(f"  5060 amortized_e12_IR vs LU CHOLMOD:    {lu_mean/am12_5060:.2f}× faster")
    print(f"  5060 amortized_e12_IR vs dev fresh_e12: {fresh_e12/am12_5060:.2f}× faster")


if __name__ == "__main__":
    main()
