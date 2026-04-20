"""T2: 3-D 7-point Laplacian on 10x10x10. Exercises 3-to-6 nnz-per-row irregularity."""
import conftest  # noqa: F401

import numpy as np
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual, jacobi_residual_reference
from _harness import dense_to_csr, laplacian_3d_7point, tolerance_bound


def test_t2_laplacian3d():
    nx = ny = nz = 10
    n = nx * ny * nz
    A = laplacian_3d_7point(nx, ny, nz)
    row_ptr, col_idx, values = dense_to_csr(A)
    rng = np.random.default_rng(2)
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

    nnz_per_row = 7  # worst-case interior row
    tol = tolerance_bound(nnz_per_row, y_ref)
    err = float(np.max(np.abs(y_kernel - y_ref)))
    print(f"T2 lap3d: n={n}, max_err={err:.3e}, tol={tol:.3e}")
    assert err <= tol, f"T2 failed: err {err:.3e} > tol {tol:.3e}"


if __name__ == "__main__":
    test_t2_laplacian3d()
    print("T2 OK")
