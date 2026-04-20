#!/usr/bin/env python3
"""Post-processing for the PLIC RT benchmark.

Computes ``h1`` (bubble tip at x=0) and ``h2`` (spike tip at x=Lx/2)
from saved alpha snapshots and compares against the reference
OpenFOAM data bundled as ``of_h1_h2.npz``.

The alpha-0.5 crossing routine (``find_alpha05_y``) is copied
VERBATIM from ``examples/RT_air_helium/postProcess.py`` lines 70-80
to ensure our numerical reading of the interface matches the
reference post-processor bit-for-bit.
"""

from __future__ import annotations

import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case_config import CASE  # noqa: E402


def find_alpha05_y(alpha_col: np.ndarray, yc_arr: np.ndarray) -> list[float]:
    """Find all y-coordinates where alpha crosses 0.5 (linear interp).

    Copied from ``examples/RT_air_helium/postProcess.py`` lines 70-80.
    """
    results: list[float] = []
    for j in range(len(alpha_col) - 1):
        a0, a1 = alpha_col[j], alpha_col[j + 1]
        if (a0 - 0.5) * (a1 - 0.5) <= 0 and a0 != a1:
            frac = (0.5 - a0) / (a1 - a0)
            y_interp = yc_arr[j] + frac * (yc_arr[j + 1] - yc_arr[j])
            results.append(float(y_interp))
    return results


def _grid_metadata():
    nx = CASE["grid"]["nx"]
    ny = CASE["grid"]["ny"]
    Lx = CASE["domain"]["x"][1] - CASE["domain"]["x"][0]
    Ly = CASE["domain"]["y"][1] - CASE["domain"]["y"][0]
    dx = Lx / nx
    dy = Ly / ny
    xc = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    yc = np.linspace(dy / 2.0, Ly - dy / 2.0, ny)
    return nx, ny, Lx, Ly, dx, dy, xc, yc


def extract_h1_h2(alpha: np.ndarray) -> tuple[float | None, float | None]:
    """Return (h1, h2) for a single snapshot.

    ``alpha`` is expected shape ``(nx, ny)`` (x-fast ordering matching
    create_rt_vof's interior slice). The OpenFOAM reference uses
    ``(ny, nx)`` so we transpose on read if needed.
    """
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    # Expected layout: alpha[i, j], i along x, j along y.
    if alpha.shape[0] != CASE["grid"]["nx"]:
        alpha = alpha.T

    nx, _, _, _, _, _, _, yc = _grid_metadata()
    i_x0 = CASE["diagnostics"]["h1_probe_x_index"]
    i_x05 = CASE["diagnostics"]["h2_probe_x_index"]

    col_x0 = alpha[i_x0, :]
    col_x05 = alpha[i_x05, :]
    y05_x0 = find_alpha05_y(col_x0, yc)
    y05_x05 = find_alpha05_y(col_x05, yc)
    h1 = max(y05_x0) if y05_x0 else None
    h2 = min(y05_x05) if y05_x05 else None
    return h1, h2


def _load_of_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, CASE["of_reference"])
    if not os.path.exists(path):
        print(f"WARNING: OpenFOAM reference {path} not found.")
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main() -> int:
    print("=" * 60)
    print(f"Case: {CASE['name']}")
    print("=" * 60, flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    snaps = sorted(glob.glob(os.path.join(results_dir, "snapshot_*.npz")))

    of_ref = _load_of_reference()
    if of_ref is not None:
        print(f"Loaded OpenFOAM reference keys: {list(of_ref.keys())}")

    if not snaps:
        print(
            "No results yet - requires Stage 4 NS coupling to generate "
            "RT snapshots. Reference data above is available for later "
            "comparison.",
            flush=True,
        )
        return 0

    print(f"\n{'Time':>6s}  {'h1 (bubble)':>14s}  {'h2 (spike)':>14s}")
    print("-" * 40)
    times, h1s, h2s = [], [], []
    for p in snaps:
        data = np.load(p)
        t = float(data["time"])
        alpha = np.asarray(data["volume_fraction"])
        h1, h2 = extract_h1_h2(alpha)
        times.append(t)
        h1s.append(h1 if h1 is not None else np.nan)
        h2s.append(h2 if h2 is not None else np.nan)
        h1_str = f"{h1:.6f}" if h1 is not None else "N/A"
        h2_str = f"{h2:.6f}" if h2 is not None else "N/A"
        print(f"{t:6.3f}  {h1_str:>14s}  {h2_str:>14s}")

    out = os.path.join(results_dir, "plic_h1_h2.npz")
    np.savez(out,
             times=np.asarray(times),
             h1=np.asarray(h1s),
             h2=np.asarray(h2s))
    print(f"\nSaved: {out}")

    # Gate comparison vs OpenFOAM
    if of_ref is not None and "times" in of_ref:
        print("\nGate check vs OpenFOAM reference")
        gate = CASE["gate"]
        t_of = of_ref["times"]
        h1_of = of_ref.get("h1")
        h2_of = of_ref.get("h2")
        for t_targ in CASE["diagnostics"]["h_times"]:
            if t_targ not in times:
                continue
            idx_ours = times.index(t_targ)
            idx_of = int(np.argmin(np.abs(t_of - t_targ)))
            h1_rel = (
                abs(h1s[idx_ours] - h1_of[idx_of]) / abs(h1_of[idx_of])
                if h1_of is not None else float("nan")
            )
            h2_rel = (
                abs(h2s[idx_ours] - h2_of[idx_of]) / abs(h2_of[idx_of])
                if h2_of is not None else float("nan")
            )
            h1_pass = h1_rel < gate["h1_rel_err_max"]
            h2_pass = h2_rel < gate["h2_rel_err_max"]
            print(
                f"  t={t_targ:.2f}  "
                f"h1 rel err = {h1_rel:.2e} "
                f"({'PASS' if h1_pass else 'FAIL'})  "
                f"h2 rel err = {h2_rel:.2e} "
                f"({'PASS' if h2_pass else 'FAIL'})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
