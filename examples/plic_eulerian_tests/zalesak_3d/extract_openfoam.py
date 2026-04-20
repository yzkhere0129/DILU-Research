#!/usr/bin/env python3
"""Extract alpha.water from an OpenFOAM Zalesak case → npz for JAX comparison.

Pipeline:
  1. reconstructPar -fields '(alpha.water)' for the requested time dirs
  2. Parse reconstructed alpha.water binary scalarField
  3. Reshape (N^3,) → (Nx, Ny, Nz) in blockMesh lexicographic order (i fastest)
  4. Save npz (F_init, F_final, plus any intermediates) matching the JAX schema

Assumes uniform cubic mesh (128³ here). No external VTK lib required — parses
OpenFOAM binary directly.

Usage:
    python extract_openfoam.py [case_dir]
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import numpy as np

CASE_DIR_DEFAULT = "/home/yzk/OpenFOAM/zalesak_3d_of"
N = 128                      # expected cube side (must match blockMeshDict)
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"

# quarter-rotation checkpoints (see controlDict: writeInterval 1.5708)
TIME_LABELS = {
    "t0":     "0",
    "t_q1":   "1.5690825",
    "t_half": "3.138165",
    "t_q3":   "4.7142212",
    "t_end":  "6.2833037",
}


def run_reconstruct(case: str, times: list[str]):
    """Call reconstructPar for each target time if not already done."""
    for t in times:
        if os.path.isdir(os.path.join(case, t, "alpha.water")) or \
           os.path.isfile(os.path.join(case, t, "alpha.water")):
            print(f"  [skip] {t}/alpha.water already reconstructed")
            continue
        # Skip t=0 reconstruction if it's only in processor dirs
        print(f"  reconstructPar -time {t} -fields '(alpha.water)'")
        cmd = ["reconstructPar", "-case", case, "-time", t,
               "-fields", "(alpha.water)"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-500:]); print(r.stderr[-500:])
            raise RuntimeError(f"reconstructPar failed for t={t}")


def read_of_scalar(path: str) -> np.ndarray:
    """Parse OpenFOAM binary (or ASCII) internalField scalarField → 1D float64."""
    with open(path, "rb") as f:
        data = f.read()

    # Uniform shortcut
    u = re.search(rb"internalField\s+uniform\s+([-\d.eE+]+)", data)
    if u:
        val = float(u.group(1))
        return np.full(N**3, val, dtype=np.float64)

    # Nonuniform — match size + open paren
    m = re.search(
        rb"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(",
        data,
    )
    if not m:
        raise ValueError(f"Cannot parse internalField in {path}")

    n_read = int(m.group(1))
    start = m.end()
    # Binary: raw float64 LE
    bin_size = n_read * 8
    blob = data[start:start + bin_size]
    if len(blob) == bin_size:
        try:
            arr = np.frombuffer(blob, dtype="<f8").copy()
            if np.all(np.isfinite(arr)) and arr.min() >= -1e-6 and arr.max() <= 1 + 1e-6:
                return arr
        except Exception:
            pass
    # ASCII fallback — extract text block between '(' and ')'
    close = data.find(b")", start)
    text = data[start:close].decode("ascii", errors="ignore")
    arr = np.fromstring(text, sep="\n")
    if arr.size != n_read:
        raise ValueError(f"Expected {n_read} values, got {arr.size} in {path}")
    return arr


def field_to_grid(flat: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Reshape flat OF internalField → (nx, ny, nz) with i-fastest lexicographic order.

    blockMesh orders cells as idx = i + j*nx + k*nx*ny.
    """
    assert flat.size == nx * ny * nz, f"size mismatch {flat.size} vs {nx*ny*nz}"
    # reshape to (nz, ny, nx) then transpose to (nx, ny, nz)
    return flat.reshape(nz, ny, nx).transpose(2, 1, 0).astype(np.float32)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else CASE_DIR_DEFAULT
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Reconstruct all checkpoint times
    print(f"Case: {case}")
    print("Reconstructing alpha.water for all quarter-turn times...")
    run_reconstruct(case, list(TIME_LABELS.values()))

    # 2. Read each
    fields = {}
    for label, t_dir in TIME_LABELS.items():
        path = os.path.join(case, t_dir, "alpha.water")
        if not os.path.isfile(path):
            print(f"  [missing] {path}"); continue
        flat = read_of_scalar(path)
        F = field_to_grid(flat, N, N, N)
        V = float(F.sum() * (1.0 / N) ** 3)
        print(f"  {label:8s}  t={t_dir:>10s}  V={V:.6e}  min={F.min():.3f}  max={F.max():.3f}")
        fields[label] = F

    # 3. Metrics (L1 vs t=0, volume drift)
    F0 = fields["t0"]
    V0 = float(F0.sum() * (1.0 / N) ** 3)
    metrics = {"V0": V0, "N": N, "dx": 1.0 / N}
    for label, F in fields.items():
        V = float(F.sum() * (1.0 / N) ** 3)
        L1 = float(np.abs(F - F0).sum() * (1.0 / N) ** 3)
        metrics[f"V_{label}"] = V
        metrics[f"V_drift_pct_{label}"] = (V - V0) / V0 * 100
        metrics[f"L1_rel_pct_{label}"] = L1 / V0 * 100

    # 4. Save
    out_path = os.path.join(OUT_DIR, "zalesak_3d_openfoam.npz")
    np.savez_compressed(
        out_path,
        F_init=fields.get("t0"),
        F_q1=fields.get("t_q1"),
        F_half=fields.get("t_half"),
        F_q3=fields.get("t_q3"),
        F_final=fields.get("t_end"),
        N=N, DX=1.0 / N,
        **{k: v for k, v in metrics.items() if k not in ("N", "dx")},
    )
    print(f"\nSaved: {out_path}")
    print("\nMetrics:")
    for k, v in metrics.items():
        if "L1_rel_pct" in k or "V_drift_pct" in k:
            print(f"  {k:30s} {v:+.4f}%")


if __name__ == "__main__":
    main()
