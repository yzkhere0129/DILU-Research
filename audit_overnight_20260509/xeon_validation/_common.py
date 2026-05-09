"""Common utilities for E0X runners.

- record_environment(): host info + git rev + library versions
- file_md5(): for input MD5 verification
- record_result(): atomic JSON write to result.json
- TimedSection: context manager for wall_seconds capture
- sanity_check_matrix(): assert ‖A·x_truth - b‖/‖b‖ < threshold
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data: break
            h.update(data)
    return h.hexdigest()


def pin_threads_to_one():
    """A008: prevent BLAS multi-threading from polluting wall measurements."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def record_environment() -> dict:
    out = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }
    # git
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "-C", "/home/yzk/DILU-Research", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        out["git_commit"] = "unknown"
    # library versions
    for libname in ("numpy", "scipy"):
        try:
            mod = __import__(libname)
            out[f"{libname}_version"] = mod.__version__
        except Exception:
            out[f"{libname}_version"] = "missing"
    try:
        import sksparse
        out["sksparse_version"] = getattr(sksparse, "__version__", "unknown")
    except Exception:
        out["sksparse_version"] = "missing"
    out["openblas_threads"] = os.environ.get("OPENBLAS_NUM_THREADS", "default")
    out["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "default")
    return out


@contextmanager
def TimedSection():
    """Wall-clock + RSS profiling. Returns dict via .data."""
    start_ns = time.perf_counter_ns()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    data = {}
    try:
        yield data
    finally:
        end_ns = time.perf_counter_ns()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        data["wall_seconds"] = (end_ns - start_ns) / 1e9
        data["wall_seconds_method"] = "MEASURED:perf_counter_ns"
        data["mem_peak_kb"] = max(rss_before, rss_after)
        data["mem_peak_method"] = "MEASURED:resource.getrusage"
        data["mem_growth_kb"] = rss_after - rss_before


def write_result_json(path: Path, payload: dict):
    """Atomic write to result.json (write to .tmp, fsync, rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sanity_check_matrix(A: csr_matrix, b: np.ndarray, x_truth: np.ndarray,
                         threshold: float = 1e-7, label: str = "input"):
    """Assert ‖A·x_truth - b‖/‖b‖ < threshold; raise SystemExit if not."""
    res = float(np.linalg.norm(A @ x_truth - b)
                / max(np.linalg.norm(b), 1e-300))
    if res > threshold:
        raise SystemExit(
            f"SANITY GATE FAILED on {label}: "
            f"‖A·x_truth - b‖/‖b‖ = {res:.3e} > {threshold:.0e}. "
            f"Refusing to run experiment with corrupt input.")
    return res


def normalize_sign(A: csr_matrix, b: np.ndarray):
    """OF Laplacian: negative diag → flip to positive."""
    diag = A.diagonal()
    if np.all(diag <= 0) and np.any(diag < 0):
        return -A, -b, True
    return A, b, False


def load_npz_with_solutions(npz_path: Path):
    """Load single_*pd_corr0_*.npz returning all fields."""
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    return dict(
        x_OF=z["x_OF"], x_AMGx_e8=z["x_AMGx_e8"],
        x_truth=z["x_truth"], b=z["b"],
        meta=meta,
    )


def load_raw_matrix(case_dir: Path, time_str: str, eq: str = "pd_corr0"):
    """Load A.mm, b.mm, x_final.mm directly from OF dump."""
    eq_dir = case_dir / "postProcessing" / "matrices" / time_str / eq
    A = sio.mmread(str(eq_dir / "A.mm")).tocsr()
    b = sio.mmread(str(eq_dir / "b.mm")).flatten()
    x_OF = sio.mmread(str(eq_dir / "x_final.mm")).flatten()
    return A, b, x_OF


def cold_cache():
    """Best-effort cache flush (no sudo). Sleep + sync."""
    try:
        subprocess.run(["sync"], check=False, timeout=10)
    except Exception:
        pass
    time.sleep(2)


def find_npz(npz_dir: Path):
    """Return ordered list of (time_str, phase, npz_path) for the 6 timesteps."""
    mapping = {
        "3.2e-07":  ("melting", "single_melting_pd_corr0_3.2e-07.npz"),
        "3.8e-07":  ("melting", "single_melting_pd_corr0_3.8e-07.npz"),
        "4.1e-07":  ("melting", "single_melting_pd_corr0_4.1e-07.npz"),
        "7e-07":    ("evap_early", "single_evap_early_pd_corr0_7e-07.npz"),
        "9e-07":    ("evap", "single_evap_pd_corr0_9e-07.npz"),
        "1.06e-06": ("evap_late", "single_evap_late_pd_corr0_1.06e-06.npz"),
    }
    out = []
    for t, (phase, fname) in mapping.items():
        p = npz_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        out.append((t, phase, p))
    return out
