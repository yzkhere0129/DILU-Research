"""6-solver wall + error comparison for T_corr0 on 6 single_track timesteps.

Same standardized format as plot_solver_wall_compare_v2.py (pd version) but
for T equation:
  - No CHOLMOD LU column (T not LU'd; SuperLU was too slow on Xeon)
  - Truth column uses AMGx@1e-12+IR 5060 (rel_resid 1e-16 = sub ε-machine)
  - Error y-axis in K instead of Pa

Sources (all 6 single_track timesteps):
  - dev 3050: /tmp/dev_T_smoke or similar (fresh_e8 + amortized_e8 + fresh_e12_IR + amortized_e12_IR)
  - 5060: audit_overnight_20260509/lab_5060_replay/results_smoke_T (same 4 protocols on 5060)
  - OF: estimate 50ms/step (Xeon 32-core MPI DILUPBiCG, 2-3 iter × ~25ms/iter)

Output:
  docs/benchmark/figures/T_solver_wall_compare_6.png

Usage:
  PYTHONPATH=. python dilu/amgx/bench/plot_solver_wall_compare_T.py \\
      --dev-results  /tmp/dev_T_smoke \\
      --5060-results audit_overnight_20260509/lab_5060_replay/results_smoke_T \\
      --output       docs/benchmark/figures/T_solver_wall_compare_6.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TS_ORDER = ["3.2e-07", "3.8e-07", "4.1e-07", "7e-07", "9e-07", "1.06e-06"]
LABEL_BY_TS = {
    "3.2e-07":  "320 ns\n(melt)",
    "3.8e-07":  "380 ns\n(melt)",
    "4.1e-07":  "410 ns\n(melt)",
    "7e-07":    "700 ns\n(evap_e)",
    "9e-07":    "900 ns\n(evap)",
    "1.06e-06": "1060 ns\n(evap_l)",
}
OF_T_WALL_S = 0.050  # OF DILUPBiCG single T solve on Xeon 32-core (estimate)


def per_step_wall_from_npz(npz_path: Path):
    """Return per-step wall in seconds (setup + update + solve + ir)."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    total = d["total_s"]  # already setup+update+solve+ir
    ts_list = [str(t) for t in d["ts"]]
    iters = d["iters"]
    return ts_list, total, iters


TARGETS_S = {
    "3.2e-07":  3.2e-7,
    "3.8e-07":  3.8e-7,
    "4.1e-07":  4.1e-7,
    "7e-07":    7.0e-7,
    "9e-07":    9.0e-7,
    "1.06e-06": 1.06e-6,
}


def sample_closest_6(ts_list, walls, iters):
    """From a larger ts_list (e.g. dense_track 384), pick the entries closest to
    the 6 single_track target timesteps. Returns walls[6], iters[6] aligned
    to TS_ORDER (NaN if no match within 5 ns)."""
    walls_out = []
    iters_out = []
    for t_key in TS_ORDER:
        target = TARGETS_S[t_key]
        diffs = [abs(float(s) - target) for s in ts_list]
        i = int(np.argmin(diffs))
        if diffs[i] > 5e-9:  # > 5 ns drift
            walls_out.append(np.nan); iters_out.append(0)
        else:
            walls_out.append(float(walls[i]))
            iters_out.append(int(iters[i]))
    return walls_out, iters_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-results", required=True,
                    help="dev 3050 results dir with replay_*.npz")
    ap.add_argument("--5060-results", dest="lab_results", required=True,
                    help="lab 5060 results dir with replay_*.npz")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dev_dir = Path(args.dev_results).expanduser()
    lab_dir = Path(args.lab_results).expanduser()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build series (name, color, walls[6], iters[6], err_K[6])
    series = []

    def collect_protocol(src_dir, proto, label, color):
        npz = src_dir / f"replay_{proto}.npz"
        if not npz.exists():
            print(f"  skip {label}: {npz} not found")
            return None
        ts_list, total, iters = per_step_wall_from_npz(npz)
        if len(ts_list) >= 50:
            # Likely dense_track 384 — sample closest to single_track 6
            walls, iter_vals = sample_closest_6(ts_list, total, iters)
        else:
            # Exact single_track 6
            walls = []; iter_vals = []
            for t in TS_ORDER:
                try:
                    i = ts_list.index(t)
                    walls.append(float(total[i]))
                    iter_vals.append(int(iters[i]))
                except ValueError:
                    walls.append(np.nan); iter_vals.append(0)
        return (label, color, walls, iter_vals)

    # 5060 amortized_e8 (橙)
    r = collect_protocol(lab_dir, "amortized_e8", "AMGx@1e-8 amortized\n(lab 5060)", "#ff7f0e")
    if r: series.append(r + ([np.nan]*6,))

    # 5060 amortized_e12+IR (深绿) — also serves as truth for error column
    r = collect_protocol(lab_dir, "amortized_e12_IR", "AMGx@1e-12+IR amortized\n(lab 5060) [TRUTH]", "#2ca02c")
    if r: series.append(r + ([0.0]*6,))   # truth, error = 0

    # OF DILUPBiCG estimate
    of_walls = [OF_T_WALL_S]*6
    of_iters = ["~2"]*6
    series.append(("OF DILUPBiCG @ tol_T\n(Xeon 32-core MPI est.)", "#1f77b4",
                    of_walls, of_iters, [np.nan]*6))

    # dev 3050 fresh_e8 (浅蓝)
    r = collect_protocol(dev_dir, "fresh_e8", "AMGx@1e-8 fresh\n(dev 3050)", "#aec7e8")
    if r: series.append(r + ([np.nan]*6,))

    # dev 3050 fresh_e12+IR (浅绿)
    r = collect_protocol(dev_dir, "fresh_e12_IR", "AMGx@1e-12+IR fresh\n(dev 3050)", "#98df8a")
    if r: series.append(r + ([np.nan]*6,))

    # Sort by mean wall
    series.sort(key=lambda s: float(np.nanmean(np.asarray(s[2], dtype=float))))
    print("\nSeries ordered by mean wall (asc):")
    for s in series:
        mw = float(np.nanmean(np.asarray(s[2], dtype=float)))
        print(f"  {mw*1000:7.1f}ms  {s[0].replace(chr(10), ' ')}")

    n_solvers = len(series)
    n_ts = len(TS_ORDER)

    fig, (ax_wall, ax_err) = plt.subplots(1, 2, figsize=(20, 7))

    # Panel 1: wall log-scale grouped bars
    x = np.arange(n_ts)
    bar_w = 0.14
    for s_idx, (name, color, walls, iters, _) in enumerate(series):
        offset = (s_idx - (n_solvers - 1) / 2) * bar_w
        bars = ax_wall.bar(x + offset, walls, bar_w, label=name,
                            color=color, edgecolor="black", linewidth=0.4)
        for b, v, it in zip(bars, walls, iters):
            if np.isnan(v) or v <= 0: continue
            it_str = f"{it}" if isinstance(it, int) else it
            # ms or s formatting
            if v < 1:
                wall_str = f"{v*1000:.0f}ms"
            else:
                wall_str = f"{v:.1f}s"
            txt = f"{wall_str}\n({it_str})"
            ax_wall.text(b.get_x() + b.get_width()/2, v * 1.05, txt,
                          ha="center", va="bottom", fontsize=6.5)
    ax_wall.set_yscale("log")
    ax_wall.set_xticks(x)
    ax_wall.set_xticklabels([LABEL_BY_TS[t] for t in TS_ORDER], fontsize=8)
    ax_wall.set_ylabel("Per-step GPU work (s) — log [excludes I/O]", fontsize=10)
    ax_wall.set_title("T_corr0 single-solve GPU work on 500K LPBF T matrix\n"
                       "(top number = wall, bottom = iter count)", fontsize=11)
    ax_wall.legend(loc="upper center", fontsize=7.5, framealpha=0.92, ncol=3)
    ax_wall.grid(True, axis="y", which="both", alpha=0.3)
    # Auto-fit y range
    all_walls = [v for _, _, w, _, _ in series for v in w if not np.isnan(v) and v > 0]
    if all_walls:
        ax_wall.set_ylim(min(all_walls)*0.3, max(all_walls)*5)

    # Panel 2: err vs AMGx@e12+IR 5060 truth (K)
    # Use truth (which is 0 by definition) at floor 1e-10 K for log display
    err_floor = 1e-10
    for s_idx, (name, color, _, _, errs) in enumerate(series):
        offset = (s_idx - (n_solvers - 1) / 2) * bar_w
        plot_e = [err_floor if (np.isnan(e) or e == 0) else max(e, err_floor) for e in errs]
        bars = ax_err.bar(x + offset, plot_e, bar_w, label=name,
                           color=color, edgecolor="black", linewidth=0.4)
        for b, e in zip(bars, errs):
            if np.isnan(e):
                txt, y = "n/a", err_floor * 1.4
            elif e == 0:
                txt, y = "0", err_floor * 1.4
            elif e < 1e-3:
                txt, y = f"{e:.1e}", e * 1.4
            else:
                txt, y = f"{e:.3g}", e * 1.4
            ax_err.text(b.get_x() + b.get_width()/2, y, txt,
                         ha="center", va="bottom", fontsize=6.5)
    ax_err.set_yscale("log")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels([LABEL_BY_TS[t] for t in TS_ORDER], fontsize=8)
    ax_err.set_ylabel("max |Δ| vs AMGx@1e-12+IR truth (K) — log", fontsize=10)
    ax_err.set_title("Solution error vs AMGx@1e-12+IR truth\n"
                      "(NOT MEASURED — would need post-process diff)", fontsize=11)
    ax_err.legend(loc="upper center", fontsize=7.5, framealpha=0.92, ncol=3)
    ax_err.grid(True, axis="y", which="both", alpha=0.3)
    ax_err.set_ylim(err_floor * 0.5, 1.0)
    ax_err.axhline(1e-3, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_err.text(n_ts - 0.5, 1e-3, " 1 mK", fontsize=7, va="center", color="gray")
    ax_err.axhline(1e-6, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_err.text(n_ts - 0.5, 1e-6, " 1 μK", fontsize=7, va="center", color="gray")

    # Title with means
    means_str = []
    for name, _, walls, _, _ in series:
        mw = float(np.nanmean(np.asarray(walls, dtype=float)))
        short = name.split("\n")[0]
        if mw < 1:
            means_str.append(f"{short}: {mw*1000:.0f}ms")
        else:
            means_str.append(f"{short}: {mw:.2f}s")
    fig.suptitle(
        f"T_corr0 5-solver comparison on 500K LPBF T (6 single_track timesteps)\n"
        f"mean GPU work per step: {'  |  '.join(means_str)}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out}")

    # Stats
    print("\nWall summary (s/step, mean over 6 timesteps):")
    print(f"  {'solver':<50} {'mean_ms':>10} {'median_ms':>10}")
    for name, _, walls, _, _ in series:
        ws = np.array([w for w in walls if not np.isnan(w)])
        if len(ws) == 0: continue
        n = name.replace("\n", " ")
        print(f"  {n:<50} {np.mean(ws)*1000:>10.1f} {np.median(ws)*1000:>10.1f}")

    # Speedup analysis
    by_name = {s[0].split("\n")[0]: float(np.nanmean(np.asarray(s[2], dtype=float))) for s in series}
    print(f"\nKey speedups (T_corr0):")
    if "AMGx@1e-8 amortized" in by_name and "OF DILUPBiCG @ tol_T" in by_name:
        ratio = by_name["OF DILUPBiCG @ tol_T"] / by_name["AMGx@1e-8 amortized"]
        print(f"  5060 amortized_e8 vs OF DILUPBiCG: {ratio:.2f}×")
    if "AMGx@1e-12+IR amortized" in by_name and "OF DILUPBiCG @ tol_T" in by_name:
        ratio = by_name["OF DILUPBiCG @ tol_T"] / by_name["AMGx@1e-12+IR amortized"]
        print(f"  5060 amortized_e12+IR vs OF DILUPBiCG: {ratio:.2f}×")


if __name__ == "__main__":
    main()
