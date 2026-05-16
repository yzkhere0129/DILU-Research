"""Plot benchmark suite v1 results: wall + precision + cross-HW.

Generates 4 figures from results_HOSTNAME_DATE/ directory:

  1. wall_compare.png       — 4 protocol × 50 matrix mean wall (per equation)
                                + OF estimate baseline
  2. wall_per_step.png       — per-step trajectory for 4 protocols, 50 matrices
                                colored by phase (pre_melt/melt/evap_early/evap_late)
  3. precision_compare.png   — max |x - x_truth| (K or Pa) per protocol per phase
                                from precision_err.json
  4. iter_distribution.png   — iter histogram per protocol per phase

If multiple results dirs given (--results-dirs HW1 HW2 ...), also generates:
  5. cross_hw_wall.png       — wall comparison across HW for AMGx amortized
  6. cross_hw_speedup.png    — speedup ratio HW2/HW1 etc.

Usage:
  PYTHONPATH=. python dilu/amgx/bench/suite/plot_results.py \\
      --results-dirs results_dev3050_2026-05-16 results_lab5060_2026-05-16 \\
      --output-dir docs/benchmark/figures/suite_v1/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTOCOLS = ["fresh_e8", "amortized_e8", "fresh_e12_IR", "amortized_e12_IR"]
PROTO_COLOR = {
    "fresh_e8":          "#aec7e8",
    "amortized_e8":      "#ff7f0e",
    "fresh_e12_IR":      "#98df8a",
    "amortized_e12_IR":  "#2ca02c",
}
PHASE_ORDER = ["pre_melt", "melt", "evap_early", "evap_late"]
PHASE_COLOR = {
    "pre_melt":   "#888",
    "melt":       "#1f77b4",
    "evap_early": "#ff7f0e",
    "evap_late":  "#d62728",
}
OF_WALL = {"pd": 3.80, "T": 0.050}   # seconds per step
UNIT = {"pd": "Pa", "T": "K"}


def load_results(results_dir: Path):
    """Returns dict: {eq: {proto: rec_dict}}"""
    out = {}
    for eq in ("pd", "T"):
        eq_dir = results_dir / eq
        if not eq_dir.is_dir(): continue
        out[eq] = {}
        for proto in PROTOCOLS:
            p = eq_dir / f"replay_{proto}.npz"
            if not p.exists(): continue
            d = np.load(p, allow_pickle=False)
            out[eq][proto] = {k: d[k] for k in d.files}
    return out


def load_precision(results_dir: Path, eq: str):
    p = results_dir / eq / "precision_err.json"
    if not p.exists(): return None
    return json.loads(p.read_text())


def hostname_from_path(p: Path) -> str:
    # results_HOSTNAME_DATE
    name = p.name
    if name.startswith("results_"):
        parts = name[len("results_"):].split("_")
        return parts[0] if parts else name
    return name


# ---------------- Figure 1: wall compare per equation ----------------

def fig_wall_compare(all_data, output, of_baseline=True):
    """Grouped bar chart: per equation, 4 protocols, mean wall + IQR."""
    # all_data = {hostname: {eq: {proto: rec}}}
    hosts = list(all_data.keys())
    n_eq = 2

    fig, axes = plt.subplots(1, n_eq, figsize=(16, 6))
    if n_eq == 1: axes = [axes]
    for ax, eq in zip(axes, ("pd", "T")):
        # Bars: x = protocols, hue = hostname
        x = np.arange(len(PROTOCOLS))
        n_hosts = len(hosts)
        bw = 0.8 / n_hosts
        for h_idx, host in enumerate(hosts):
            recs = all_data[host].get(eq, {})
            means = []
            for proto in PROTOCOLS:
                if proto in recs:
                    means.append(float(np.mean(recs[proto]["total_s"])) * 1000)
                else:
                    means.append(np.nan)
            offset = (h_idx - (n_hosts-1)/2) * bw
            bars = ax.bar(x + offset, means, bw, label=host,
                           edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, means):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, v * 1.05,
                             f"{v:.0f}ms", ha="center", va="bottom", fontsize=7.5)
        if of_baseline:
            of_ms = OF_WALL[eq] * 1000
            ax.axhline(of_ms, color="black", linestyle="--", linewidth=1.4,
                        label=f"OF ~{of_ms:.0f}ms")
            ax.text(len(PROTOCOLS) - 0.5, of_ms, f" OF {of_ms:.0f}ms",
                     fontsize=8, va="center", color="black")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOLS, rotation=15, fontsize=8)
        ax.set_ylabel("Mean GPU work per matrix (ms) — log [excl. I/O]", fontsize=9)
        ax.set_title(f"{eq.upper()}_corr0 (50 matrices)", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.92)
        ax.grid(True, axis="y", which="both", alpha=0.3)

    fig.suptitle(f"DILU Suite v1: AMGx wall comparison ({len(hosts)} HW × 2 equations × 4 protocols)",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {output}")


# ---------------- Figure 2: per-step trajectory ----------------

def fig_per_step(results_dir, all_data, output):
    """Per-step wall trajectory, colored by physical phase, 4 protocols."""
    host = hostname_from_path(results_dir)
    data = all_data[host]
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, eq in zip(axes, ("pd", "T")):
        recs = data.get(eq, {})
        for proto in PROTOCOLS:
            if proto not in recs: continue
            r = recs[proto]
            walls_ms = r["total_s"] * 1000
            ax.plot(np.arange(len(walls_ms)), walls_ms, label=proto,
                     color=PROTO_COLOR[proto], linewidth=1.0, alpha=0.85, marker="o", markersize=3)
        ax.axhline(OF_WALL[eq]*1000, color="black", linestyle="--", linewidth=1.4,
                    label=f"OF ~{OF_WALL[eq]*1000:.0f}ms/step")
        # phase background bands
        phases = recs[PROTOCOLS[0]]["phase"] if recs else []
        if len(phases):
            for i, ph in enumerate(phases):
                ax.axvspan(i-0.5, i+0.5, alpha=0.08, color=PHASE_COLOR.get(ph, "#aaa"))
        ax.set_yscale("log")
        ax.set_ylabel(f"{eq.upper()} wall (ms) — log")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{eq.upper()}_corr0 per-step ({host})")
    axes[-1].set_xlabel(f"Matrix index (0..49, ordered by phase: " +
                         " | ".join(f"{p}" for p in PHASE_ORDER) + ")")
    plt.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {output}")


# ---------------- Figure 3: precision compare ----------------

def fig_precision(results_dir, output):
    """Bar chart: max|x - x_truth| per protocol per phase, both eq."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, eq in zip(axes, ("pd", "T")):
        prec = load_precision(results_dir, eq)
        if prec is None:
            ax.text(0.5, 0.5, f"No precision_err.json for {eq}\nrun compute_truth_err.py",
                     ha="center", va="center", transform=ax.transAxes)
            continue
        per_m = prec["per_matrix"]
        unit = prec["unit"]

        # group by phase, mean across phase for each protocol
        # Plot: x = phase, y = mean err, color = protocol
        srcs = ["x_OF"] + [p for p in PROTOCOLS if p != prec["truth_protocol"]]
        x = np.arange(len(PHASE_ORDER))
        bw = 0.8 / len(srcs)
        for s_idx, src in enumerate(srcs):
            phase_means = []
            for ph in PHASE_ORDER:
                vals = [m[src]["max"] for m in per_m if m["phase"] == ph and src in m]
                phase_means.append(np.mean(vals) if vals else np.nan)
            offset = (s_idx - (len(srcs)-1)/2) * bw
            color = "#1f77b4" if src == "x_OF" else PROTO_COLOR.get(src, "#888")
            label = "OF" if src == "x_OF" else src
            bars = ax.bar(x + offset, phase_means, bw, label=label,
                           color=color, edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, phase_means):
                if not np.isnan(v) and v > 0:
                    if v < 1e-6:
                        txt = f"{v:.1e}"
                    elif v < 1e-3:
                        txt = f"{v*1e6:.1f}μ"
                    elif v < 1:
                        txt = f"{v*1e3:.1f}m"
                    else:
                        txt = f"{v:.1f}"
                    ax.text(b.get_x() + b.get_width()/2, v*1.1,
                             txt, ha="center", va="bottom", fontsize=6.5, rotation=0)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(PHASE_ORDER, fontsize=9)
        ax.set_ylabel(f"max |x − x_truth| per phase ({unit}) — log", fontsize=9)
        ax.set_title(f"{eq.upper()}_corr0 — truth = {prec['truth_protocol']}", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.92, ncol=2)
        ax.grid(True, axis="y", which="both", alpha=0.3)

    fig.suptitle("DILU Suite v1: solution precision vs ε-machine truth (50 matrices × 4 phases)",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {output}")


# ---------------- Figure 4: iter distribution ----------------

def fig_iter(results_dir, all_data, output):
    host = hostname_from_path(results_dir)
    data = all_data[host]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, eq in zip(axes, ("pd", "T")):
        recs = data.get(eq, {})
        positions = []; iters_per_proto = []; colors = []; labels = []
        for proto in PROTOCOLS:
            if proto not in recs: continue
            r = recs[proto]
            iters_per_proto.append(r["iters"])
            colors.append(PROTO_COLOR[proto])
            labels.append(proto)
        bp = ax.boxplot(iters_per_proto, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_ylabel("AMGx iter count")
        ax.set_title(f"{eq.upper()}_corr0 iter distribution (50 matrices, {host})")
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dirs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for rd_str in args.results_dirs:
        rd = Path(rd_str).expanduser()
        host = hostname_from_path(rd)
        all_data[host] = load_results(rd)
        print(f"# {host}: {sum(len(v) for v in all_data[host].values())} proto-eq combos loaded")

    # Single-HW figures (use first results-dir)
    primary = Path(args.results_dirs[0]).expanduser()
    fig_per_step(primary, all_data, out_dir / "per_step_trajectory.png")
    fig_precision(primary, out_dir / "precision_compare.png")
    fig_iter(primary, all_data, out_dir / "iter_distribution.png")

    # Multi-HW figure: wall compare (works for 1 HW too)
    fig_wall_compare(all_data, out_dir / "wall_compare.png")

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
