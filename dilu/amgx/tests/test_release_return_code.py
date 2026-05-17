"""amgx_release return-code contract test (CODE_REVIEW.md C2).

Contract (REPRODUCE.md §4 amgx_release):
  status = 0  ==>  success (released or already-absent — idempotent)
  status != 0 reserved for a future "actually failed to tear down" case.

Pre-2026-05 the C++ inverted this (1 = OK, 0 = unknown token). This test
pins down the post-patch contract so a future revert is caught.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import (
    amgx_setup, amgx_release, CLASSICAL_V_DIAGSCALED, with_tolerance,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_pd_spd():
    """Load pd_tiny fixture in SPD form (sign-flipped)."""
    d = np.load(FIXTURES_DIR / "pd_tiny.npz")
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=tuple(d["shape"]))
    return (-A).tocsr()


def test_release_returns_zero_on_valid_token():
    """Releasing a freshly-setup plan returns status=0."""
    A = _load_pd_spd()
    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-10, max_iters=10)

    token = amgx_setup(rp, ci, vv, cfg)
    token.block_until_ready()

    status = amgx_release(token)
    status.block_until_ready()
    code = int(np.asarray(status)[0])
    assert code == 0, (
        f"amgx_release on valid token must return 0 (success), got {code}. "
        f"The C++ contract may have reverted to the pre-patch inversion."
    )


def test_release_returns_zero_on_unknown_token_idempotent():
    """Releasing an already-released token returns 0 (idempotent)."""
    A = _load_pd_spd()
    rp = jnp.asarray(A.indptr.astype(np.int32))
    ci = jnp.asarray(A.indices.astype(np.int32))
    vv = jnp.asarray(A.data.astype(np.float64))
    cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, 1e-10, max_iters=10)

    token = amgx_setup(rp, ci, vv, cfg)
    token.block_until_ready()

    # First release: removes the token from the registry.
    status1 = amgx_release(token)
    status1.block_until_ready()
    assert int(np.asarray(status1)[0]) == 0

    # Second release: token is gone; contract says still 0 (no-op).
    status2 = amgx_release(token)
    status2.block_until_ready()
    code = int(np.asarray(status2)[0])
    assert code == 0, (
        f"amgx_release on unknown/already-released token must return 0 "
        f"(idempotent), got {code}. The C++ may have reverted to "
        f"distinguishing 'released-now' from 'already-absent'."
    )
