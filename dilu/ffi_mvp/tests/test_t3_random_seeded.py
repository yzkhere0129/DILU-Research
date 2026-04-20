"""T3: diag-dominant random with 3-order diagonal spread. Seed fixed for regression."""
import conftest  # noqa: F401

import numpy as np
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual, jacobi_residual_reference
from _harness import (
    dense_to_csr,
    diag_dominant_random,
    tolerance_bound,
)


def test_t3_random_seeded():
    n = 500
    A = diag_dominant_random(n, seed=0, diag_spread=1e3)
    row_ptr, col_idx, values = dense_to_csr(A)
    nnz = values.shape[0]
    nnz_per_row_avg = nnz / n

    rng = np.random.default_rng(3)
    x_h = rng.standard_normal(n).astype(np.float64)
    b_h = rng.standard_normal(n).astype(np.float64)
    diag_h = 1.0 / np.diag(A)

    y_kernel = np.asarray(
        jacobi_residual(row_ptr, col_idx, values, diag_h, b_h, x_h)
    )
    y_ref = np.asarray(
        jacobi_residual_reference(
            jnp.asarray(row_ptr), jnp.asarray(col_idx), jnp.asarray(values),
            jnp.asarray(diag_h), jnp.asarray(b_h), jnp.asarray(x_h),
        )
    )

    # Max fan-in per row bounds the associativity error. Use actual max.
    max_nnz_per_row = int(np.max(np.diff(row_ptr)))
    tol = tolerance_bound(max_nnz_per_row, y_ref)
    err = float(np.max(np.abs(y_kernel - y_ref)))
    print(
        f"T3 random: n={n}, nnz={nnz}, nnz/row avg={nnz_per_row_avg:.1f}, "
        f"max={max_nnz_per_row}, max_err={err:.3e}, tol={tol:.3e}"
    )
    assert err <= tol, f"T3 failed: err {err:.3e} > tol {tol:.3e}"


if __name__ == "__main__":
    test_t3_random_seeded()
    print("T3 OK")
