"""Detect physical phases in a LaserMeltFoam case for matrix-dump planning.

Phases (LPBF physics):
  P0  cold-start         max(T) < T_solidus           (no melting yet)
  P1  melt-pool onset    max(T) ≥ T_solidus           (first melting)
  P2  melt-pool mature   max(gT) ≥ 0.5                (significant liquid volume)
  P3  recoil onset       max(T) ≥ T_vap               (Anisimov pressure starts)
  P4  keyhole forming    free-surface depression > beam_radius/4
  P5  steady-state       |Δ max(T)| / max(T) per Δt < 1e-3 over 5 consecutive samples

Inputs:
  case_dir/<time>/T              volScalarField (Kelvin)
  case_dir/<time>/gT             volScalarField (liquid fraction 0–1)
  case_dir/<time>/alpha.metal    volScalarField (metal phase indicator 0–1)
  case_dir/<time>/U              volVectorField (m/s)
  case_dir/constant/transportProperties for liquidus/solidus/vap temperatures

Outputs:
  case_dir/postProcessing/phase_timeline.csv
  case_dir/postProcessing/phase_timeline.png
  case_dir/postProcessing/recommended_dump_times.json    ← feed to matrixDumperDict

Usage:
  python -m dilu.amgx.bench.phase_detector <case_dir> [--T-sol K] [--T-vap K]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# OpenFOAM ASCII field parser (minimal, internalField only)
# ---------------------------------------------------------------------------

_LIST_HDR_RE = re.compile(r"internalField\s+nonuniform\s+List<(scalar|vector|tensor)>")
_UNIFORM_RE  = re.compile(r"internalField\s+uniform\s+([\(\)\-+\d.eE\s]+)")


def parse_internal_field(path: Path) -> np.ndarray | None:
    """Return internalField as flat ndarray (scalar) or (N,3) for vectors.
    Returns None if file missing or unparseable.
    """
    if not path.exists():
        return None
    text = path.read_text()

    # Uniform shortcut
    m = _UNIFORM_RE.search(text)
    if m:
        token = m.group(1).strip()
        if token.startswith("("):  # vector
            vals = [float(v) for v in token.strip("() ").split()]
            return np.array(vals, dtype=np.float64)
        else:
            return np.array([float(token)], dtype=np.float64)

    m = _LIST_HDR_RE.search(text)
    if not m:
        return None
    kind = m.group(1)
    after = text[m.end():]

    # Find size N then '('
    m_n = re.search(r"\s*(\d+)\s*\(", after)
    if not m_n:
        return None
    N = int(m_n.group(1))
    body_start = m_n.end()
    # Body ends at first ')' followed (after optional whitespace) by ';'
    m_end = re.search(r"\)\s*;", after[body_start:])
    if not m_end:
        return None
    body_end = body_start + m_end.start()
    body = after[body_start:body_end]

    if kind == "scalar":
        arr = np.fromstring(body, sep=" ", dtype=np.float64)
        if arr.size != N:
            arr = np.array([float(t) for t in body.split() if t], dtype=np.float64)
        return arr
    elif kind == "vector":
        # "(x y z)\n(x y z)\n..."  strip parens, split
        clean = body.replace("(", " ").replace(")", " ")
        arr = np.fromstring(clean, sep=" ", dtype=np.float64)
        return arr.reshape(-1, 3)[:N]
    else:
        return None


# ---------------------------------------------------------------------------
# Per-time stats
# ---------------------------------------------------------------------------

def list_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    out = []
    for d in case_dir.iterdir():
        if not d.is_dir(): continue
        name = d.name
        try:
            t = float(name)
        except ValueError:
            continue
        out.append((t, d))
    out.sort()
    return out


def probe_time_dir(time_dir: Path) -> dict:
    """Return per-time scalar summary."""
    T = parse_internal_field(time_dir / "T")
    gT = parse_internal_field(time_dir / "gT")
    alpha = parse_internal_field(time_dir / "alpha.metal")
    U = parse_internal_field(time_dir / "U")

    out = {}
    if T is not None:
        out["T_max"] = float(T.max())
        out["T_min"] = float(T.min())
        out["T_mean"] = float(T.mean())
    if gT is not None:
        out["gT_max"]    = float(gT.max())
        out["gT_volfrac"] = float((gT > 0.5).sum() / gT.size)
    if alpha is not None:
        out["alpha_min"] = float(alpha.min())
        out["alpha_max"] = float(alpha.max())
        # surface-cell deformation proxy: cells with 0.05 < α < 0.95 = interface
        interface_count = int(((alpha > 0.05) & (alpha < 0.95)).sum())
        out["interface_cells"] = interface_count
        # gas-volume fraction (cells with α<0.05). Keyhole/depression grows this.
        out["gas_volfrac"] = float((alpha < 0.05).sum() / alpha.size)
    if U is not None and U.ndim == 2:
        speed = np.linalg.norm(U, axis=1)
        out["U_max"] = float(speed.max())
        out["U_mean"] = float(speed.mean())
    return out


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def classify_phases(timeline: list[dict],
                    T_sol: float, T_liq: float, T_vap: float,
                    keyhole_gas_growth_factor: float = 1.5,
                    steady_window: int = 5,
                    steady_rel_tol: float = 1e-3) -> dict:
    """Return dict mapping phase name → first time index it occurs (or None).

    P4 keyhole detection: gas volume fraction grows by `keyhole_gas_growth_factor`
    relative to the initial value. Initial gas region (above powder bed) is
    baseline; keyhole formation pushes more cells into gas via surface cavitation.
    """
    phases = dict(P0_cold_start=None, P1_melt_onset=None,
                  P2_melt_mature=None, P3_recoil_onset=None,
                  P4_keyhole=None, P5_steady=None)
    times = np.array([row["t"] for row in timeline])
    Tmax  = np.array([row.get("T_max", 0.0) for row in timeline])
    gTmax = np.array([row.get("gT_max", 0.0) for row in timeline])
    gas_vf = np.array([row.get("gas_volfrac", 0.0) for row in timeline])

    if Tmax.size == 0:
        return phases

    phases["P0_cold_start"] = float(times[0])
    idx = np.where(Tmax >= T_sol)[0]
    if idx.size > 0: phases["P1_melt_onset"] = float(times[idx[0]])
    idx = np.where(gTmax >= 0.5)[0]
    if idx.size > 0: phases["P2_melt_mature"] = float(times[idx[0]])
    idx = np.where(Tmax >= T_vap)[0]
    if idx.size > 0: phases["P3_recoil_onset"] = float(times[idx[0]])

    # P4 keyhole: gas volume fraction grows. Baseline = initial.
    if gas_vf.size > 0 and gas_vf[0] > 0:
        threshold = gas_vf[0] * keyhole_gas_growth_factor
        idx = np.where(gas_vf >= threshold)[0]
        if idx.size > 0: phases["P4_keyhole"] = float(times[idx[0]])

    if Tmax.size >= steady_window + 1:
        for i in range(steady_window, Tmax.size):
            window = Tmax[i - steady_window:i + 1]
            ref = max(window.max(), 1.0)
            if (window.max() - window.min()) / ref < steady_rel_tol and window.max() > T_sol:
                phases["P5_steady"] = float(times[i])
                break
    return phases


def recommend_dump_times(phases: dict, timeline: list[dict],
                         per_phase: int = 100) -> list[float]:
    """For each detected phase boundary, sample `per_phase` times from that phase
    onwards (until next phase). Cap at total `len(timeline)` budget.
    """
    times = [row["t"] for row in timeline]
    boundaries = [v for v in phases.values() if v is not None]
    boundaries = sorted(set(boundaries))
    if not boundaries:
        return list(times)
    boundaries.append(times[-1] + 1)  # sentinel

    samples = []
    for k in range(len(boundaries) - 1):
        t0, t1 = boundaries[k], boundaries[k + 1]
        in_phase = [t for t in times if t0 <= t < t1]
        if not in_phase: continue
        # Sample `per_phase` times log-uniformly from in_phase
        if len(in_phase) <= per_phase:
            samples.extend(in_phase)
        else:
            # uniform sub-sampling
            stride = len(in_phase) // per_phase
            samples.extend(in_phase[::stride][:per_phase])
    return sorted(set(samples))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timeline(timeline: list[dict], phases: dict,
                  T_sol: float, T_liq: float, T_vap: float,
                  out_png: Path) -> None:
    if not timeline:
        print("(empty timeline; nothing to plot)")
        return
    t = np.array([r["t"] for r in timeline])
    Tmax = np.array([r.get("T_max", np.nan) for r in timeline])
    gTmax = np.array([r.get("gT_max", np.nan) for r in timeline])
    alphaMin = np.array([r.get("alpha_min", np.nan) for r in timeline])
    Umax = np.array([r.get("U_max", np.nan) for r in timeline])

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    ax = axes[0]
    ax.plot(t, Tmax, "r-o", ms=3, label="max(T)")
    ax.axhline(T_sol, color="orange", ls="--", alpha=0.6, label=f"T_solidus={T_sol:.0f} K")
    ax.axhline(T_liq, color="red",    ls="--", alpha=0.6, label=f"T_liquidus={T_liq:.0f} K")
    ax.axhline(T_vap, color="purple", ls="--", alpha=0.6, label=f"T_vap={T_vap:.0f} K")
    ax.set_ylabel("Temperature (K)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, gTmax, "b-o", ms=3, label="max(gT) (liquid fraction)")
    ax.axhline(0.5, color="k", ls="--", alpha=0.4, label="0.5 threshold")
    ax.set_ylabel("Liquid fraction")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, alphaMin, "g-o", ms=3, label="min(α_metal) (keyhole proxy)")
    ax.axhline(0.5, color="k", ls="--", alpha=0.4, label="0.5 threshold")
    ax.set_ylabel("min α_metal")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, Umax, "k-o", ms=3, label="max|U|")
    ax.set_ylabel("max |U| (m/s)")
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # annotate phase boundaries on top axis
    colors = ["#888", "#fa0", "#f00", "#a0f", "#0a0", "#0aa"]
    for i, (name, val) in enumerate(phases.items()):
        if val is None: continue
        for axx in axes:
            axx.axvline(val, color=colors[i], ls=":", alpha=0.5)
        axes[0].text(val, axes[0].get_ylim()[1] * 0.95, name.replace("_", "\n"),
                     fontsize=7, ha="center", va="top",
                     bbox=dict(facecolor="white", alpha=0.7, edgecolor=colors[i]))

    fig.suptitle(f"Phase timeline — {out_png.parent.parent.name}\n"
                 f"detected: {sum(1 for v in phases.values() if v is not None)}/6 phases")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Default thermal points for 316L stainless steel (LaserMeltFoam tutorial default)
TI_SOL_316L = 1685.0   # K
TI_LIQ_316L = 1727.0
TI_VAP_316L = 3133.0


def main(case_dir: Path, T_sol: float, T_liq: float, T_vap: float,
         per_phase_samples: int = 100):
    print(f"Scanning {case_dir} ...")
    times_dirs = list_time_dirs(case_dir)
    print(f"  found {len(times_dirs)} time directories")
    if not times_dirs:
        print("  (no time dirs — case has not been run yet, or VTK-only output)")
        return

    timeline = []
    for t, d in times_dirs:
        row = {"t": t}
        row.update(probe_time_dir(d))
        timeline.append(row)
    # First/last digest
    print(f"  t range: [{timeline[0]['t']:.3e}, {timeline[-1]['t']:.3e}] s")
    if "T_max" in timeline[-1]:
        print(f"  final T_max = {timeline[-1]['T_max']:.1f} K")
    if "gT_max" in timeline[-1]:
        print(f"  final gT_max = {timeline[-1]['gT_max']:.3f}")
    if "alpha_min" in timeline[-1]:
        print(f"  final α_min = {timeline[-1]['alpha_min']:.3f}")
    if "U_max" in timeline[-1]:
        print(f"  final |U|_max = {timeline[-1]['U_max']:.2e} m/s")

    phases = classify_phases(timeline, T_sol, T_liq, T_vap)
    print("\nPhase boundaries detected (None = phase not reached in this run):")
    for k, v in phases.items():
        s = f"{v:.4e} s" if v is not None else "—"
        print(f"  {k:24s} t = {s}")

    rec = recommend_dump_times(phases, timeline, per_phase=per_phase_samples)
    print(f"\nRecommended dump times: {len(rec)}")

    out_dir = case_dir / "postProcessing"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "phase_timeline.csv"
    with csv_path.open("w") as fh:
        keys = sorted(set(k for r in timeline for k in r.keys()))
        fh.write(",".join(keys) + "\n")
        for r in timeline:
            fh.write(",".join(f"{r.get(k,'')}" for k in keys) + "\n")
    print(f"Wrote {csv_path}")

    json_path = out_dir / "phases.json"
    json_path.write_text(json.dumps(dict(
        case=str(case_dir),
        T_sol=T_sol, T_liq=T_liq, T_vap=T_vap,
        timeline=timeline,
        phases=phases,
        recommended_dump_times=rec,
    ), indent=2))
    print(f"Wrote {json_path}")

    plot_timeline(timeline, phases, T_sol, T_liq, T_vap,
                  out_dir / "phase_timeline.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path,
                    help="path to a LaserMeltFoam case (parent of [0-9]+ time dirs)")
    ap.add_argument("--T-sol", type=float, default=TI_SOL_316L)
    ap.add_argument("--T-liq", type=float, default=TI_LIQ_316L)
    ap.add_argument("--T-vap", type=float, default=TI_VAP_316L)
    ap.add_argument("--per-phase", type=int, default=100,
                    help="recommended dump samples per phase")
    args = ap.parse_args()
    main(args.case_dir, args.T_sol, args.T_liq, args.T_vap, args.per_phase)
