#!/usr/bin/env python3
"""Extract volume / interface-perimeter metrics for the 45-deg droplet case.

This is the PLIC-research adaptation of
``examples/droplet_advection_45deg/extract_metrics.py``. It reuses the
vectorised marching-squares perimeter routine but reads our own output
format (``results/*.npz``) rather than JAX-Fluids HDF5.

Until Stage 2 generates actual simulation output, ``main()`` just
reports the analytic initial values (V0, P0) and early-returns with a
"no results yet" notice.
"""

from __future__ import annotations

import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case_config import CASE  # noqa: E402


def compute_interface_perimeter_vectorized(
    volume_fraction: np.ndarray, dx: float, dy: float
) -> float:
    """Marching-squares-style interface perimeter, fully vectorised.

    Copied from ``examples/droplet_advection_45deg/extract_metrics.py``
    lines 49-65. The input array is expected with shape ``(Nx, Ny)`` or
    ``(Ny, Nx)`` — the routine is symmetric in the two axes so either
    works as long as ``dx`` / ``dy`` are consistent with the ordering.
    """
    vf = np.asarray(volume_fraction)

    # Edges along axis 1 (each edge has length dy in the reference file;
    # we preserve that mapping exactly).
    h_left = vf[:, :-1]
    h_right = vf[:, 1:]
    h_cross = ((h_left - 0.5) * (h_right - 0.5)) < 0
    perimeter_h = float(np.sum(h_cross) * dy)

    # Edges along axis 0
    v_top = vf[:-1, :]
    v_bottom = vf[1:, :]
    v_cross = ((v_top - 0.5) * (v_bottom - 0.5)) < 0
    perimeter_v = float(np.sum(v_cross) * dx)

    return perimeter_h + perimeter_v


def _results_glob() -> list[str]:
    """Return the (possibly empty) list of snapshot files."""
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    if not os.path.isdir(results_dir):
        return []
    # Stage 2 will produce files like 'snapshot_000.npz' (convention TBD).
    return sorted(glob.glob(os.path.join(results_dir, "snapshot_*.npz")))


def main() -> int:
    # --- Grid / analytic initial values ---
    nx, ny, nz = CASE["grid"]["nx"], CASE["grid"]["ny"], CASE["grid"]["nz"]
    x_range = CASE["domain"]["x"]
    y_range = CASE["domain"]["y"]
    z_range = CASE["domain"]["z"]
    dx = (x_range[1] - x_range[0]) / nx
    dy = (y_range[1] - y_range[0]) / ny
    dz = (z_range[1] - z_range[0]) / max(nz, 1)

    V0 = float(CASE["analytic_V0"])
    P0 = float(CASE["analytic_P0"])

    print("=" * 60)
    print(f"Case     : {CASE['name']}")
    print(f"Grid     : {nx} x {ny}, dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
    print(f"Droplet  : R={CASE['droplet']['radius']}, "
          f"center={CASE['droplet']['center']}")
    print(f"Analytic : V0 = {V0:.10f}, P0 = {P0:.6f}")
    print("=" * 60, flush=True)

    snapshots = _results_glob()
    if not snapshots:
        print(
            "No results yet - Stage 2 required to generate snapshots. "
            "Analytic reference values printed above.",
            flush=True,
        )
        return 0

    # --- Parse snapshots and compute per-step metrics ---
    cx0, cy0 = CASE["droplet"]["center"]
    nh_guess = 0  # snapshots stored with halos already stripped

    # Interior cell-center coordinates (match init_droplet.py:79-81)
    x = np.linspace(dx / 2.0, x_range[1] - dx / 2.0, nx)
    y = np.linspace(dy / 2.0, y_range[1] - dy / 2.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    times, volumes, perimeters, cents_x, cents_y = [], [], [], [], []
    for path in snapshots:
        data = np.load(path)
        t = float(data["time"])
        vf = np.asarray(data["volume_fraction"])  # expected (Nx, Ny, Nz)

        # Defensive: if older snapshots still carry z-halos, strip them.
        if vf.ndim == 3 and vf.shape[2] > nz:
            nh = (vf.shape[2] - nz) // 2
            vf = vf[:, :, nh : nh + nz]

        if vf.ndim == 3:
            # Physical droplet volume = sum over all physical z layers.
            V = float(vf.sum() * dx * dy * dz)
            # Centroid + perimeter from the middle z slice (quasi-2D:
            # all layers are identical up to roundoff).
            z_mid = vf.shape[2] // 2
            vf_plane = vf[:, :, z_mid]
        else:
            V = float(vf.sum() * dx * dy * dz * max(nz, 1))
            vf_plane = vf

        P = compute_interface_perimeter_vectorized(vf_plane, dx, dy)

        total = float(vf_plane.sum()) + 1.0e-30
        cx = float((vf_plane * X).sum() / total)
        cy = float((vf_plane * Y).sum() / total)

        times.append(t)
        volumes.append(V)
        perimeters.append(P)
        cents_x.append(cx)
        cents_y.append(cy)

    times = np.asarray(times)
    volumes = np.asarray(volumes)
    perimeters = np.asarray(perimeters)
    cents_x = np.asarray(cents_x)
    cents_y = np.asarray(cents_y)

    V_init = float(volumes[0])
    P_init = float(perimeters[0])

    print(
        f"Initial  : V_numeric={V_init:.10e}  "
        f"(analytic {V0:.10e}, init bias {(V_init - V0) / V0 * 100:+.2f}% from tanh smoothing)"
    )
    print(
        f"           P_numeric={P_init:.6f}  "
        f"(analytic {P0:.6f}, init bias {(P_init - P0) / P0 * 100:+.2f}% from grid-aligned perimeter)"
    )

    print(
        f"\n{'Time':>10s}  {'V_drift(%)':>12s}  {'P_drift(%)':>12s}  "
        f"{'cx':>10s}  {'cy':>10s}  {'cx_err':>10s}  {'cy_err':>10s}"
    )
    print("-" * 86)
    for t, V, P, cx, cy in zip(times, volumes, perimeters, cents_x, cents_y):
        v_drift = (V - V_init) / V_init * 100
        p_drift = (P - P_init) / P_init * 100
        exp_cx = cx0 + t
        exp_cy = cy0 + t
        err_cx = cx - exp_cx
        err_cy = cy - exp_cy
        print(
            f"{t:10.5f}  {v_drift:+12.4e}  {p_drift:+12.4e}  "
            f"{cx:10.6f}  {cy:10.6f}  {err_cx:+10.6f}  {err_cy:+10.6f}"
        )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "advection_metrics.npz",
    )
    np.savez(
        out_path,
        times=times,
        volumes=volumes,
        perimeters=perimeters,
        centroids_x=cents_x,
        centroids_y=cents_y,
        V0=V0,
        P0=P0,
        V_init=V_init,
        P_init=P_init,
    )
    print(f"\nSaved: {out_path}")

    # ----- Gate check (from-initial advection drift + centroid error) -----
    gate = CASE["gate"]
    v_drift_rel = abs(volumes[-1] - V_init) / V_init
    p_drift_rel = abs(perimeters[-1] - P_init) / P_init
    v_pass = v_drift_rel < gate["volume_rel_err_max"]
    p_pass = p_drift_rel < gate["perimeter_rel_change_max"]
    # Centroid gate: final |cx - expected| and |cy - expected| must each be
    # under 2 * dx (plan "合格" threshold).
    exp_cx_final = cx0 + float(times[-1])
    exp_cy_final = cy0 + float(times[-1])
    cx_err_final = abs(cents_x[-1] - exp_cx_final)
    cy_err_final = abs(cents_y[-1] - exp_cy_final)
    c_pass = (cx_err_final < 2 * dx) and (cy_err_final < 2 * dx)

    print("\nGate check (advection drift from initial numeric values)")
    print(
        f"  volume drift    : {v_drift_rel:.2e}  "
        f"(< {gate['volume_rel_err_max']:.1e}?)  "
        f"{'PASS' if v_pass else 'FAIL'}"
    )
    print(
        f"  perimeter drift : {p_drift_rel:.2e}  "
        f"(< {gate['perimeter_rel_change_max']:.1e}?)  "
        f"{'PASS' if p_pass else 'FAIL'}"
    )
    print(
        f"  centroid error  : ({cx_err_final:.2e}, {cy_err_final:.2e})  "
        f"(< 2*dx = {2 * dx:.2e}?)  "
        f"{'PASS' if c_pass else 'FAIL'}"
    )
    return 0 if (v_pass and p_pass and c_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
