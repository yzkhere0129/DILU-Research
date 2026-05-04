"""Aggregate scaling-test timings from existing timing_*cores.csv files.

Recovers usable medians from scaling_results_lab/ even though the original
scaling_test.sh wrote summary.csv with all-zero rows (split(',')[3] vs
3-field rows bug).

Each timing_<N>cores.csv has rows:
    step,time_s,eq,wall_ms          <- header (4 cols, misleading)
    <N>,<eq>,<wall_ms>              <- data (3 cols)

A laserMeltFoam step contains 3 pd correctors + 1 T solve (4 timing rows).
The first pd corrector each step is the "outer" pd_corr0 with full setup
cost (DIC analyze + RHS assemble); the next two are inner correctors that
re-use cached factors and are 10-100x cheaper. We report both:

  - pd_outer_med : median of pd_corr0 wall (matches v3.2 report's 2.57 s
                   single-solve metric, since v3.2 sampled only outer
                   correctors)
  - pd_step_med  : median of (pd_corr0 + pd_corr1 + pd_corr2) per step,
                   i.e. total pd cost per timestep (the "what does this
                   step actually cost" metric)
  - T_med        : median T solve wall

We drop the first WARMUP_STEPS steps to ignore the dt=1e-12 cold-start
where the matrix conditioning is unrepresentative.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median

WARMUP_STEPS = 5  # drop steps 1..5 (dt < 1e-11, non-stationary)
ROWS_PER_STEP = 4  # 3 pd correctors + 1 T solve, in source order


def parse_timing_csv(path: Path, n_cores: int) -> list[tuple[str, float]]:
    """Return [(eq, wall_ms), ...] in source order, skipping header."""
    rows: list[tuple[str, float]] = []
    with path.open() as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                _ = int(parts[0])  # cores field
                eq = parts[1]
                ms = float(parts[2])
            except ValueError:
                continue
            if eq not in ("pd", "T"):
                continue
            rows.append((eq, ms))
    return rows


def group_into_steps(rows: list[tuple[str, float]]) -> list[dict]:
    """Walk rows and group consecutive [pd, pd, pd, T] into one step.

    Returns list of {'pd': [w0, w1, w2], 'T': [wT, ...]}. Tolerant of
    irregularities: starts a new step every time we see a 'T' row.
    """
    steps: list[dict] = []
    cur: dict = {"pd": [], "T": []}
    for eq, ms in rows:
        cur[eq].append(ms)
        if eq == "T":
            steps.append(cur)
            cur = {"pd": [], "T": []}
    if cur["pd"] or cur["T"]:
        steps.append(cur)
    return steps


def summarize(rows: list[tuple[str, float]], n_cores: int) -> dict:
    steps = group_into_steps(rows)
    n_steps_total = len(steps)
    stationary = steps[WARMUP_STEPS:]
    if not stationary:
        return {
            "cores": n_cores,
            "n_steps_total": n_steps_total,
            "n_steps_stationary": 0,
            "pd_outer_med_ms": None,
            "pd_step_med_ms": None,
            "T_med_ms": None,
        }

    pd_outer = [s["pd"][0] for s in stationary if s["pd"]]
    pd_step = [sum(s["pd"]) for s in stationary if s["pd"]]
    t_solve = [w for s in stationary for w in s["T"]]

    return {
        "cores": n_cores,
        "n_steps_total": n_steps_total,
        "n_steps_stationary": len(stationary),
        "pd_outer_med_ms": median(pd_outer) if pd_outer else None,
        "pd_outer_min_ms": min(pd_outer) if pd_outer else None,
        "pd_outer_max_ms": max(pd_outer) if pd_outer else None,
        "pd_step_med_ms": median(pd_step) if pd_step else None,
        "T_med_ms": median(t_solve) if t_solve else None,
        "T_min_ms": min(t_solve) if t_solve else None,
        "T_max_ms": max(t_solve) if t_solve else None,
    }


def main(results_dir: Path) -> None:
    csv_files = sorted(
        results_dir.glob("timing_*cores.csv"),
        key=lambda p: int(p.stem.replace("timing_", "").replace("cores", "")),
    )
    if not csv_files:
        print(f"No timing_*cores.csv found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    for csv_path in csv_files:
        n = int(csv_path.stem.replace("timing_", "").replace("cores", ""))
        rows = parse_timing_csv(csv_path, n)
        if not rows:
            print(f"  N={n}: no data (likely failed run)")
            summaries.append({"cores": n, "n_steps_total": 0,
                              "pd_outer_med_ms": None, "pd_step_med_ms": None,
                              "T_med_ms": None})
            continue
        s = summarize(rows, n)
        summaries.append(s)
        po = s["pd_outer_med_ms"]
        ps = s["pd_step_med_ms"]
        tm = s["T_med_ms"]
        if po is None:
            print(f"  N={n:3d}: NO DATA")
        else:
            print(f"  N={n:3d}: {s['n_steps_total']} steps "
                  f"(stationary {s['n_steps_stationary']}) | "
                  f"pd_corr0={po:.1f}ms  pd_step={ps:.1f}ms  T={tm:.1f}ms")

    # Write new summary.csv
    out_csv = results_dir / "summary_clean.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cores", "n_steps", "pd_outer_med_ms",
                    "pd_step_med_ms", "T_med_ms",
                    "pd_outer_min_ms", "pd_outer_max_ms",
                    "T_min_ms", "T_max_ms"])
        for s in summaries:
            w.writerow([
                s["cores"],
                s.get("n_steps_total", 0),
                f"{s['pd_outer_med_ms']:.2f}" if s.get("pd_outer_med_ms") else "",
                f"{s['pd_step_med_ms']:.2f}" if s.get("pd_step_med_ms") else "",
                f"{s['T_med_ms']:.2f}" if s.get("T_med_ms") else "",
                f"{s.get('pd_outer_min_ms'):.2f}" if s.get("pd_outer_min_ms") else "",
                f"{s.get('pd_outer_max_ms'):.2f}" if s.get("pd_outer_max_ms") else "",
                f"{s.get('T_min_ms'):.2f}" if s.get("T_min_ms") else "",
                f"{s.get('T_max_ms'):.2f}" if s.get("T_max_ms") else "",
            ])
    print(f"\nWrote {out_csv}")

    # Write JSON for programmatic use
    out_json = results_dir / "summary_clean.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out_json}")

    # Plot
    try:
        plot(summaries, results_dir / "scaling_clean.png")
    except Exception as e:
        print(f"Plot failed: {e}", file=sys.stderr)


def plot(summaries: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A run is "stationary-comparable" only if dt actually reached the maxCo
    # cap. Heuristic: need >= 20 steps captured. Runs with fewer steps are
    # cold-start-only (dt < 1e-10) — their solve cost is artificially low
    # because the matrix is dt-dominated and very well-conditioned. Plotting
    # them on the same axis as stationary runs is misleading.
    MIN_STATIONARY_STEPS = 20
    valid = [s for s in summaries
             if s.get("pd_outer_med_ms")
             and s.get("n_steps_total", 0) >= MIN_STATIONARY_STEPS]
    cold = [s for s in summaries
            if s.get("pd_outer_med_ms")
            and 0 < s.get("n_steps_total", 0) < MIN_STATIONARY_STEPS]
    if cold:
        print(f"\nNote: dropping {len(cold)} cold-start-only point(s) from plot:")
        for s in cold:
            print(f"  N={s['cores']}: only {s['n_steps_total']} steps "
                  f"(dt never reached stationary), pd_corr0={s['pd_outer_med_ms']:.1f}ms "
                  f"is artificially low — NOT comparable")
    if not valid:
        print("No valid data to plot")
        return

    cores = [s["cores"] for s in valid]
    pd_outer = [s["pd_outer_med_ms"] for s in valid]
    pd_step = [s["pd_step_med_ms"] for s in valid]
    t_med = [s["T_med_ms"] for s in valid]

    base_pd_outer = pd_outer[0]
    base_pd_step = pd_step[0]
    base_t = t_med[0]
    su_pd_outer = [base_pd_outer / v for v in pd_outer]
    su_pd_step = [base_pd_step / v for v in pd_step]
    su_t = [base_t / v for v in t_med]
    ideal = [c / cores[0] for c in cores]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.plot(cores, ideal, "k--", label="ideal", alpha=0.4)
    ax1.plot(cores, su_pd_outer, "bo-", label="pd_corr0 speedup")
    ax1.plot(cores, su_pd_step, "g^-", label="pd_step (sum corr0+1+2)")
    ax1.plot(cores, su_t, "rs-", label="T speedup")
    ax1.set_xlabel("Processors")
    ax1.set_ylabel(f"Speedup vs N={cores[0]}")
    ax1.set_title("Strong scaling — laserMeltFoam (after warmup, "
                  f"≥{WARMUP_STEPS+1}-th step)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)

    ax2 = axes[1]
    ax2.plot(cores, pd_outer, "bo-", label="pd_corr0 (median, ms)")
    ax2.plot(cores, pd_step, "g^-", label="pd step total (median, ms)")
    ax2.plot(cores, t_med, "rs-", label="T solve (median, ms)")
    ax2.set_xlabel("Processors")
    ax2.set_ylabel("Wall time (ms)")
    ax2.set_title("Per-solve / per-step wall time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Plot saved: {out_png}")

    # Optimal core counts
    best_outer = min(valid, key=lambda s: s["pd_outer_med_ms"])
    best_step = min(valid, key=lambda s: s["pd_step_med_ms"])
    best_t = min(valid, key=lambda s: s["T_med_ms"])
    print(f"\nOptimal pd_corr0:  N={best_outer['cores']} "
          f"({best_outer['pd_outer_med_ms']:.1f} ms)")
    print(f"Optimal pd_step:    N={best_step['cores']} "
          f"({best_step['pd_step_med_ms']:.1f} ms)")
    print(f"Optimal T:          N={best_t['cores']} "
          f"({best_t['T_med_ms']:.1f} ms)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rd = Path(sys.argv[1])
    else:
        rd = Path(__file__).parent / "scaling_results_lab"
    main(rd)
