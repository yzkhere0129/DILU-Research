"""amgx_update_coefficients error-path test (REPRODUCE.md §4).

`update_coefficients` is fingerprint-guarded: the (N, NNZ) of the new
values must match what was passed to `amgx_setup`. Passing a different
NNZ MUST fail synchronously with a clear error — silently accepting it
would produce a corrupted solve.

This test pins down the contract on the pd_tiny fixture.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _spd_pd():
    d = np.load(FIXTURES_DIR / "pd_tiny.npz")
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    return (-A).tocsr()


def test_update_coefficients_rejects_wrong_nnz():
    """Passing fewer values than the original NNZ must error, not silent."""
    A = _spd_pd()
    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-10, max_iters=10)

    with Plan(rp, ci, vv, cfg) as plan:
        # Same dtype, wrong length (drop the last entry):
        bad = vv[:-1]
        with pytest.raises(Exception) as exc_info:
            plan.update_coefficients(bad)
        msg = str(exc_info.value).lower()
        assert "nnz" in msg or "fingerprint" in msg or "mismatch" in msg, (
            f"update_coefficients must fail with a clear NNZ/fingerprint "
            f"error; got: {exc_info.value!r}"
        )


def test_update_coefficients_accepts_same_nnz_different_values():
    """Passing same NNZ with different values must succeed and return 0."""
    A = _spd_pd()
    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-10, max_iters=10)

    with Plan(rp, ci, vv, cfg) as plan:
        # Perturb values by 1%, keep NNZ:
        rng = np.random.default_rng(20260517)
        perturbed = vv * (1.0 + 0.01 * rng.standard_normal(vv.shape[0]))
        status = plan.update_coefficients(perturbed)
        status.block_until_ready()
        code = int(np.asarray(status)[0])
        assert code == 0, (
            f"update_coefficients on valid same-NNZ values should succeed "
            f"(status=0), got {code}"
        )
