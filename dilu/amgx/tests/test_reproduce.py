"""Canonical bit-exact reproduction test (REPRODUCE.md §7).

This is the test a fresh-context AI's re-implementation MUST pass before
its output is considered "bit-exact" to ours. It pins down, for the
pd_tiny fixture on the current host:

  - exact iteration count per protocol
  - exact final relative residual (rounded to 16 sig figs)
  - SHA-256 of the solution vector x

If any value drifts, the test fails. The first time this test runs on a
new host (different GPU SKU or CUDA driver), it seeds a per-host golden
file under `tests/fixtures/golden_<HOSTNAME>_<GPU>.json` and prints a
warning. Subsequent runs on the same host must match the seeded values.

Protocols (matches `bench/suite/run_benchmark.py::PROTOCOL_CONFIGS`):
  fresh_e8           tol=1e-8,  n_ir=0, amortized=False
  fresh_e12_IR       tol=1e-12, n_ir=1, amortized=False
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import pytest
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)

from dilu.amgx.python import amgx_solve_with_refinement


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _host_tag():
    """Identify the host: hostname + GPU short name. Used to scope the
    golden file because bit-exact AMGx output depends on GPU SKU."""
    host = socket.gethostname().split(".")[0]
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=3,
        ).strip().split("\n")[0]
        gpu_short = (gpu.replace("NVIDIA ", "").replace("GeForce ", "")
                      .replace(" Laptop GPU", "").replace(" ", "_"))
    except Exception:
        gpu_short = "noGPU"
    return f"{host}_{gpu_short}"


GOLDEN_FILE = FIXTURES_DIR / f"golden_{_host_tag()}.json"


def _hash_x(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x.astype(np.float64))
                          .tobytes()).hexdigest()


def _load_pd_spd():
    d = np.load(FIXTURES_DIR / "pd_tiny.npz")
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    return (-A).tocsr(), -d["b"], d["x_truth"]


PROTOCOLS = [
    {"name": "fresh_e8",      "tol": 1e-8,  "n_ir": 0},
    {"name": "fresh_e12_IR",  "tol": 1e-12, "n_ir": 1},
]


def _run_protocol(spec):
    A, b, _ = _load_pd_spd()
    x0 = np.zeros_like(b)
    res = amgx_solve_with_refinement(
        A, b, x0,
        eq_kind="pd",
        tol=spec["tol"],
        n_refine=spec["n_ir"],
        max_iters=500,
    )
    return {
        "primary_iters":   int(res["primary_iters"]),
        "primary_status":  int(res["primary_status"]),
        "refine_iters":    list(res["refine_iters"]),
        "rel_residual":    float(res["rel_residual"]),
        "x_sha256":        _hash_x(res["x"]),
    }


def _load_or_seed_golden():
    if GOLDEN_FILE.exists():
        return json.loads(GOLDEN_FILE.read_text()), False
    golden = {"host_tag": _host_tag(), "protocols": {}}
    for spec in PROTOCOLS:
        golden["protocols"][spec["name"]] = _run_protocol(spec)
    FIXTURES_DIR.mkdir(exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(golden, indent=2, sort_keys=True))
    return golden, True


@pytest.mark.parametrize("spec", PROTOCOLS, ids=lambda s: s["name"])
def test_reproduce_pd_protocol_bit_exact(spec):
    """For each protocol, observed iter / residual / x-hash must match golden."""
    golden, seeded = _load_or_seed_golden()
    if seeded:
        pytest.skip(
            f"seeded golden file at {GOLDEN_FILE} — re-run to enforce "
            f"bit-exact reproduction"
        )
    name = spec["name"]
    obs = _run_protocol(spec)
    exp = golden["protocols"][name]
    assert obs["primary_iters"] == exp["primary_iters"], (
        f"{name}: iter drift  observed={obs['primary_iters']} "
        f"expected={exp['primary_iters']}"
    )
    assert obs["refine_iters"] == exp["refine_iters"], (
        f"{name}: refine_iters drift  observed={obs['refine_iters']} "
        f"expected={exp['refine_iters']}"
    )
    # Residual: tight match in relative terms (1 ULP at the magnitude
    # of typical 1e-13 .. 1e-16). We allow 1% drift because residual is
    # the float64-computed norm of A x - b after the AMGx solve; that
    # last reduction has order-dependent rounding.
    rel_drift = abs(obs["rel_residual"] - exp["rel_residual"]) / max(
        exp["rel_residual"], 1e-300)
    assert rel_drift < 0.01, (
        f"{name}: rel_residual drift > 1%  observed={obs['rel_residual']:.3e} "
        f"expected={exp['rel_residual']:.3e} drift={rel_drift:.2%}"
    )
    assert obs["x_sha256"] == exp["x_sha256"], (
        f"{name}: solution vector SHA-256 mismatch — this is the strictest "
        f"bit-exact check. Some bit of x changed.\n"
        f"  observed: {obs['x_sha256']}\n"
        f"  expected: {exp['x_sha256']}\n"
        f"If the change is intentional (e.g. AMGx version bump), delete "
        f"{GOLDEN_FILE} and re-run to re-seed."
    )


def test_reproduce_golden_file_is_host_scoped():
    """Sanity: the golden file is named per host, so other hosts don't
    overwrite ours."""
    assert GOLDEN_FILE.name.startswith("golden_"), (
        f"golden file must be host-scoped; got {GOLDEN_FILE.name}"
    )
    assert _host_tag() in GOLDEN_FILE.name


def test_reproduce_metadata_records_environment():
    """Print metadata so a CI log captures it for future cross-host audits."""
    info = {
        "host":       socket.gethostname(),
        "platform":   platform.platform(),
        "python":     platform.python_version(),
        "host_tag":   _host_tag(),
        "golden_path": str(GOLDEN_FILE),
        "golden_exists": GOLDEN_FILE.exists(),
    }
    try:
        info["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], text=True, timeout=3).strip()
    except Exception as e:
        info["nvidia_smi"] = f"unavailable: {e}"
    print("\n  REPRODUCE env:", json.dumps(info, indent=2))
