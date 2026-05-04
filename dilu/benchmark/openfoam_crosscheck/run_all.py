"""End-to-end: scipy + cuSPARSE + AMGx across all matrices + compare.

Usage:
    python3 -m dilu.benchmark.openfoam_crosscheck.run_all <root>

The scripts are run as subprocesses so the JAX VRAM is released between
driver types. This also isolates OOMs / crashes to one solver.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--skip-scipy", action="store_true",
                    help="skip scipy reference (N>200K → GMRES is slow)")
    ap.add_argument("--skip-cusparse", action="store_true")
    ap.add_argument("--skip-amgx", action="store_true")
    ap.add_argument("--amgx-cfg", nargs="+",
                    default=["classical_v", "aggressive"])
    ap.add_argument("--pattern", default="*/*_corr*")
    ap.add_argument("--tol", type=float, default=1e-10)
    args = ap.parse_args()

    py = sys.executable
    base = ["-m", "dilu.benchmark.openfoam_crosscheck"]
    rc = 0

    if not args.skip_scipy:
        rc |= _run([py, *base[:-1], "dilu.benchmark.openfoam_crosscheck.driver_scipy",
                    str(args.root), "--pattern", args.pattern])
    if not args.skip_cusparse:
        rc |= _run([py, *base[:-1], "dilu.benchmark.openfoam_crosscheck.driver_cusparse",
                    str(args.root), "--pattern", args.pattern,
                    "--tol", str(args.tol)])
    if not args.skip_amgx:
        rc |= _run([py, *base[:-1], "dilu.benchmark.openfoam_crosscheck.driver_amgx",
                    str(args.root), "--pattern", args.pattern,
                    "--cfg", *args.amgx_cfg, "--tol", str(args.tol)])

    # Compare always
    rc |= _run([py, *base[:-1], "dilu.benchmark.openfoam_crosscheck.compare",
                str(args.root)])

    print(f"\nrun_all finished rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
