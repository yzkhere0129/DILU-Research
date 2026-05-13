"""Plot 4 protocols × N steps from replay_5060_amgx_amortized output.

Reads replay_<protocol>.npz files in --input-dir and produces 4 panels:
    (1) Wall per step
    (2) Iter per step
    (3) Cumulative wall vs step
    (4) Breakdown stacked bar (setup / update / solve / ir)

Usage:
    ~/jax-env/bin/python3 dilu/amgx/bench/plot_replay_5060.py \\
        --input-dir audit_overnight_20260509/lab_5060_replay \\
        --output    docs/benchmark/figures/amortized_5060_replay.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTO_COLOR = {
    "fresh_e8":          "#1f77b4",
    "amortized_e8":      "#ff7f0e",
    "fresh_e12_IR":      "#2ca02c",
    "amortized_e12_IR":  "#d62728",
}
PROTO_ORDER = ["fresh_e8", "amortized_e8", "fresh_e12_IR", "amortized_e12_IR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--lu-wall-per-step", type=float, default=None,
                    help="LU CHOLMOD wall per step in seconds (for comparison line); "
                         "default = None (omit line)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load all protocols that exist
    recs = {}
    for proto in PROTO_ORDER:
        p = in_dir / f"replay_{proto}.npz"
        if not p.exists():
            print(f"  skip {proto}: {p} not found")
            continue
        d = np.load(p, allow_pickle=True)
        recs[proto] = {k: d[k] for k in d.files}
    if not recs:
        raise FileNotFoundError(f"No replay_*.npz under {in_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_wall, ax_iter, ax_cum, ax_bd = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- (1) Wall per step ----
    for proto, rec in recs.items():
        step = rec["step"]
        ax_wall.plot(step, rec["total_s"] * 1000, label=proto,
                     color=PROTO_COLOR[proto], linewidth=1.2, alpha=0.85)
    if args.lu_wall_per_step:
        ax_wall.axhline(args.lu_wall_per_step * 1000, color="black",
                        linestyle="--", linewidth=1, label=f"CHOLMOD LU ({args.lu_wall_per_step:.0f}s/step)")
    ax_wall.set_yscale("log")
    ax_wall.set_xlabel("Step index"); ax_wall.set_ylabel("Wall per step (ms) — log")
    ax_wall.set_title("Per-step wall time"); ax_wall.legend(fontsize=8)
    ax_wall.grid(True, alpha=0.3, which="both")

    # ---- (2) Iter per step ----
    for proto, rec in recs.items():
        ax_iter.plot(rec["step"], rec["iters"], label=proto,
                     color=PROTO_COLOR[proto], linewidth=1.2, alpha=0.85)
    ax_iter.set_xlabel("Step index"); ax_iter.set_ylabel("PCG iter")
    ax_iter.set_title("Per-step iteration count (warm-start savings)")
    ax_iter.legend(fontsize=8); ax_iter.grid(True, alpha=0.3)

    # ---- (3) Cumulative wall ----
    for proto, rec in recs.items():
        cum = np.cumsum(rec["total_s"])
        ax_cum.plot(rec["step"], cum, label=proto,
                    color=PROTO_COLOR[proto], linewidth=1.5)
    if args.lu_wall_per_step:
        # constant per step → linear cumulative
        any_rec = next(iter(recs.values()))
        n = len(any_rec["step"])
        ax_cum.plot(np.arange(n), np.arange(1, n+1) * args.lu_wall_per_step,
                    color="black", linestyle="--", linewidth=1.2,
                    label=f"CHOLMOD LU @{args.lu_wall_per_step:.0f}s/step")
    ax_cum.set_xlabel("Step index"); ax_cum.set_ylabel("Cumulative wall (s)")
    ax_cum.set_title("Cumulative wall vs step (the bottom-line plot)")
    ax_cum.legend(fontsize=8); ax_cum.grid(True, alpha=0.3)

    # ---- (4) Breakdown stacked bar ----
    protos = list(recs.keys())
    setup_t = [float(np.sum(r["setup_s"]))  for r in recs.values()]
    update_t= [float(np.sum(r["update_s"])) for r in recs.values()]
    solve_t = [float(np.sum(r["solve_s"]))  for r in recs.values()]
    ir_t    = [float(np.sum(r["ir_s"]))     for r in recs.values()]

    x = np.arange(len(protos))
    bot = np.zeros(len(protos))
    for arr, lbl, color in zip(
        [setup_t, update_t, solve_t, ir_t],
        ["setup", "update", "solve", "IR"],
        ["#888", "#aaaaff", "#ffaa66", "#66cc66"],
    ):
        ax_bd.bar(x, arr, bottom=bot, label=lbl, color=color, edgecolor="black", linewidth=0.4)
        for i, (b, v) in enumerate(zip(bot, arr)):
            if v > max(sum([setup_t, update_t, solve_t, ir_t], [])) * 0.02:
                ax_bd.text(i, b + v/2, f"{v:.0f}s", ha="center", va="center",
                           fontsize=7.5, color="black")
        bot = bot + np.array(arr)
    for i, total in enumerate(bot):
        ax_bd.text(i, total * 1.02, f"{total:.0f}s total", ha="center",
                   va="bottom", fontsize=8, fontweight="bold")
    ax_bd.set_xticks(x); ax_bd.set_xticklabels(protos, rotation=15, fontsize=8)
    ax_bd.set_ylabel("Cumulative time (s)")
    ax_bd.set_title("Wall breakdown (sum over all steps)")
    ax_bd.legend(fontsize=8, loc="upper left")
    ax_bd.grid(True, axis="y", alpha=0.3)

    # ---- Title with verdicts ----
    speedup_lines = []
    if "fresh_e8" in recs and "amortized_e8" in recs:
        sa = float(np.sum(recs["fresh_e8"]["total_s"])) / max(float(np.sum(recs["amortized_e8"]["total_s"])), 1e-12)
        speedup_lines.append(f"e8: amortized {sa:.2f}× faster than fresh")
    if "fresh_e12_IR" in recs and "amortized_e12_IR" in recs:
        sa = float(np.sum(recs["fresh_e12_IR"]["total_s"])) / max(float(np.sum(recs["amortized_e12_IR"]["total_s"])), 1e-12)
        speedup_lines.append(f"e12+IR: amortized {sa:.2f}× faster than fresh")
    # vs LU
    if args.lu_wall_per_step and "amortized_e8" in recs:
        n = len(recs["amortized_e8"]["step"])
        lu_total = n * args.lu_wall_per_step
        ratio = lu_total / float(np.sum(recs["amortized_e8"]["total_s"]))
        speedup_lines.append(f"amortized_e8 vs LU: {ratio:.2f}× faster than LU")
    sub = "  |  ".join(speedup_lines) if speedup_lines else ""

    n_any = len(next(iter(recs.values()))["step"])
    fig.suptitle(f"AMGx amortized vs fresh on {n_any} sane LPBF pd matrices  (lab RTX 5060 / dev RTX 3050)\n{sub}",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")

    # Also print the verdict table
    print("\nWall-totals (seconds):")
    print(f"  {'protocol':<22} {'setup':>8} {'update':>8} {'solve':>8} {'IR':>8} {'TOTAL':>10} {'iter_med':>10} {'resid_max':>12}")
    for proto in PROTO_ORDER:
        if proto not in recs: continue
        r = recs[proto]
        print(f"  {proto:<22} {float(np.sum(r['setup_s'])):>8.1f} "
              f"{float(np.sum(r['update_s'])):>8.1f} "
              f"{float(np.sum(r['solve_s'])):>8.1f} "
              f"{float(np.sum(r['ir_s'])):>8.1f} "
              f"{float(np.sum(r['total_s'])):>10.1f} "
              f"{float(np.median(r['iters'])):>10.0f} "
              f"{float(np.max(r['rel_resid'])):>12.2e}")


if __name__ == "__main__":
    main()
