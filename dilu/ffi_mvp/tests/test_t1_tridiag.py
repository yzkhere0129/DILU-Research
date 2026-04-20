"""T1: 1-D Laplacian tridiag, n=1000. Hand-verifiable via analytic Toeplitz SpMV."""
import conftest  # noqa: F401 — sets x64 + sys.path

import numpy as np
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual, jacobi_residual_reference
from _harness import dense_to_csr, laplacian_1d, tolerance_bound


def test_t1_tridiag():
    n = 1000
    A = laplacian_1d(n)
    row_ptr, col_idx, values = dense_to_csr(A)
    rng = np.random.default_rng(1)
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

    nnz_per_row = 3
    tol = tolerance_bound(nnz_per_row, y_ref)
    err = float(np.max(np.abs(y_kernel - y_ref)))
    print(f"T1 tridiag: n={n}, max_err={err:.3e}, tol={tol:.3e}")
    assert err <= tol, f"T1 failed: err {err:.3e} > tol {tol:.3e}"


if __name__ == "__main__":
    test_t1_tridiag()
    print("T1 OK")
