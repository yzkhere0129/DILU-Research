"""Cross-hardware benchmark runner for DILU-Research suite v1.

Reads binary-format suite (manifest.json + matrices_npz/) on ANY hardware
with CUDA GPU + jax + AMGx FFI, and runs the standard 4 protocols × N matrices
for both pd and T equations.

Output:
  <out>/{equation}/replay_{protocol}.npz  per-step records
  <out>/{equation}/x_results.npz          final x per matrix per protocol (for truth diff)
  <out>/run_meta.json                     hostname / GPU / jax / git sha
  <out>/summary.json                      aggregate stats

Usage:
  PYTHONPATH=. python dilu/amgx/bench/suite/run_benchmark.py \\
      --suite ~/benchmark_suite_v1 \\
      --out   results_$(hostname)_$(date +%Y-%m-%d)/ \\
      --equations pd,T \\
      --protocols fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR

The runner is intentionally minimal — same code on dev/5060/H100. AMGx config
auto-selected per equation (pd → PCG, T → BICGSTAB).
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

from dilu.amgx.python import (
    Plan, CLASSICAL_V_DIAGSCALED, CLASSICAL_V_DIAGSCALED_BICGSTAB,
    with_tolerance,
)


EQUATION_CONFIG = {
    "pd": ("PCG_DIAGSCALED",      CLASSICAL_V_DIAGSCALED),
    "T":  ("BICGSTAB_DIAGSCALED", CLASSICAL_V_DIAGSCALED_BICGSTAB),
}
PROTOCOL_CONFIGS = {
    "fresh_e8":          {"tol": 1e-8,  "n_ir": 0, "amortized": False},
    "amortized_e8":      {"tol": 1e-8,  "n_ir": 0, "amortized": True},
    "fresh_e12_IR":      {"tol": 1e-12, "n_ir": 1, "amortized": False},
    "amortized_e12_IR":  {"tol": 1e-12, "n_ir": 1, "amortized": True},
}


def load_matrix_npz(npz_path: Path):
    """Load A, b, x_OF from binary npz. Auto sign-flips negative-diagonal
    matrices (OF lduMatrix exports pd with negative diag — flip to SPD form
    for AMGx PCG/BICGSTAB, and flip x_OF too so x_OF ~ x_AMGx)."""
    d = np.load(npz_path)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    b = d["b"]
    x_OF = d["x_OF"]
    diag_mean = float(np.mean(A.diagonal()))
    sign_flipped = diag_mean < 0
    if sign_flipped:
        A = -A
        b = -b
        # x_OF satisfies A_orig x = b_orig, i.e. (-A) x = (-b), so x_OF same.
        # No flip needed for x_OF.
    return A, b, x_OF


def hw_info():
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jax_devices": str(jax.devices()),
        "wall_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], text=True).strip()
    except Exception as e:
        info["gpu"] = f"no-smi: {e}"
    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent.parent.parent.parent.parent),
             "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        info["git_sha"] = "unknown"
    return info


def list_matrices(suite_dir: Path, equation: str):
    """Return list of (phase, ts, npz_path) for the equation."""
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    short = {"pd": "pd", "T": "T"}[equation]
    out = []
    for m in manifest["matrices"]:
        npz = suite_dir / "matrices_npz" / m["phase"] / f"ts_{m['ts']}" / f"{short}.npz"
        if npz.exists():
            out.append((m["phase"], m["ts"], npz))
    return out


def run_one_protocol(name: str, matrices: list, base_config: str, max_iters: int,
                      save_x: bool):
    cfg = PROTOCOL_CONFIGS[name]
    cfg_json = with_tolerance(base_config, cfg["tol"], max_iters=max_iters)
    n = len(matrices)
    rec = {
        "ts":        np.array([m[1] for m in matrices], dtype="U16"),
        "phase":     np.array([m[0] for m in matrices], dtype="U12"),
        "setup_s":   np.zeros(n), "update_s": np.zeros(n),
        "solve_s":   np.zeros(n), "ir_s":     np.zeros(n),
        "total_s":   np.zeros(n),
        "iters":     np.zeros(n, dtype=np.int32),
        "status":    np.zeros(n, dtype=np.int32),
        "rel_resid": np.zeros(n),
    }
    x_all = [] if save_x else None

    plan = None
    x_prev = None
    print(f"\n=== {name}  tol={cfg['tol']}  n_ir={cfg['n_ir']}  amortized={cfg['amortized']} ===")
    t_proto = time.time()

    for k, (phase, ts, npz) in enumerate(matrices):
        A, b, _ = load_matrix_npz(npz)
        rp = jnp.asarray(A.indptr)
        ci = jnp.asarray(A.indices)
        vv = jnp.asarray(A.data)
        bd = jnp.asarray(b)

        if cfg["amortized"]:
            if plan is None:
                t0 = time.time()
                plan = Plan(rp, ci, vv, cfg_json)
                rec["setup_s"][k] = time.time() - t0
            else:
                t0 = time.time()
                plan.update_coefficients(vv)
                rec["update_s"][k] = time.time() - t0
            this_plan = plan
        else:
            if plan is not None: plan.release(); plan = None
            t0 = time.time()
            this_plan = Plan(rp, ci, vv, cfg_json)
            rec["setup_s"][k] = time.time() - t0

        if cfg["amortized"] and x_prev is not None and x_prev.shape == b.shape:
            x0 = jnp.asarray(x_prev)
        else:
            x0 = jnp.zeros_like(bd)

        t0 = time.time()
        x_jax, it_d, st_d = this_plan.solve(bd, x0)
        x_jax.block_until_ready()
        rec["solve_s"][k] = time.time() - t0
        rec["iters"][k] = int(np.asarray(it_d)[0])
        rec["status"][k] = int(np.asarray(st_d)[0])
        x = np.asarray(x_jax)

        t_ir = 0.0
        for _ in range(cfg["n_ir"]):
            r_np = b - A @ x
            t0 = time.time()
            d_jax, _, _ = this_plan.solve(jnp.asarray(r_np), jnp.zeros_like(bd))
            d_jax.block_until_ready()
            t_ir += time.time() - t0
            x = x + np.asarray(d_jax)
        rec["ir_s"][k] = t_ir

        rec["rel_resid"][k] = float(
            np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
        rec["total_s"][k] = (rec["setup_s"][k] + rec["update_s"][k]
                              + rec["solve_s"][k] + rec["ir_s"][k])
        if not cfg["amortized"]:
            this_plan.release(); this_plan = None

        x_prev = x
        if save_x:
            x_all.append(x.astype(np.float64))

        if (k+1) % 5 == 0 or k == n-1:
            print(f"  [{k+1}/{n}] {phase}/{ts}  "
                  f"setup={rec['setup_s'][k]*1000:5.0f}ms  "
                  f"upd={rec['update_s'][k]*1000:5.0f}ms  "
                  f"solve={rec['solve_s'][k]*1000:5.0f}ms  "
                  f"ir={rec['ir_s'][k]*1000:4.0f}ms  "
                  f"iter={rec['iters'][k]:3d}  "
                  f"resid={rec['rel_resid'][k]:.2e}")
        del rp, ci, vv, bd, x0, x_jax
        if k % 10 == 0: gc.collect()

    if plan is not None: plan.release()
    rec["protocol_wall_s"] = np.array([time.time() - t_proto])
    if save_x:
        rec["x_all"] = np.stack(x_all)  # shape (n, N)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="benchmark_suite_v1 root dir")
    ap.add_argument("--out",   required=True, help="results output dir")
    ap.add_argument("--equations", default="pd,T")
    ap.add_argument("--protocols", default="fresh_e8,amortized_e8,fresh_e12_IR,amortized_e12_IR")
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--save-x", action="store_true",
                     help="save final x per matrix (for truth-diff post-process). "
                          "Saves ~6MB × n_matrices per protocol. Default OFF.")
    args = ap.parse_args()

    suite = Path(args.suite).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    info = hw_info()
    info.update({
        "suite":    str(suite),
        "equations": args.equations.split(","),
        "protocols": args.protocols.split(","),
        "max_iters": args.max_iters,
        "save_x":    bool(args.save_x),
    })
    (out / "run_meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    summary = {}
    for eq in args.equations.split(","):
        eq = eq.strip()
        if eq not in EQUATION_CONFIG:
            print(f"  skip unknown equation: {eq}"); continue
        cfg_name, base_config = EQUATION_CONFIG[eq]
        matrices = list_matrices(suite, eq)
        if not matrices:
            print(f"  no {eq} matrices found in {suite}/matrices_npz/")
            continue
        eq_out = out / eq
        eq_out.mkdir(exist_ok=True)
        print(f"\n### Equation: {eq}  config: {cfg_name}  matrices: {len(matrices)}")

        summary[eq] = {}
        for proto in args.protocols.split(","):
            proto = proto.strip()
            if proto not in PROTOCOL_CONFIGS:
                print(f"  skip unknown protocol: {proto}"); continue
            rec = run_one_protocol(proto, matrices, base_config, args.max_iters, args.save_x)
            npz_path = eq_out / f"replay_{proto}.npz"
            np.savez_compressed(npz_path, **rec)
            print(f"  → {npz_path}")
            summary[eq][proto] = {
                "n":            int(len(rec["ts"])),
                "wall_total_s": float(rec["protocol_wall_s"][0]),
                "setup_s":  float(np.sum(rec["setup_s"])),
                "update_s": float(np.sum(rec["update_s"])),
                "solve_s":  float(np.sum(rec["solve_s"])),
                "ir_s":     float(np.sum(rec["ir_s"])),
                "iter_mean":   float(np.mean(rec["iters"])),
                "iter_median": float(np.median(rec["iters"])),
                "iter_max":    int(np.max(rec["iters"])),
                "rel_resid_max":    float(np.max(rec["rel_resid"])),
                "rel_resid_median": float(np.median(rec["rel_resid"])),
                "n_failed":    int(np.sum(rec["status"] != 0)),
            }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n### DONE → {out / 'summary.json'}")


if __name__ == "__main__":
    main()
