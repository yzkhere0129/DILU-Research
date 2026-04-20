"""T4: DILU factorization (D_* only) correctness vs NumPy reference.

Three test matrices, all N ≤ 10k per the safety brief:
  - T4.1: 1-D Laplacian tridiag n=500
  - T4.2: 3-D 7-point Laplacian on 10^3 = 1000
  - T4.3: diagonally dominant random n=200, seed 0
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.cusparse.python import dilu_factor, build_diag_offset
from _harness import dilu_d_reference, dense_to_csr, laplacian_3d_7point


def _tridiag_csr(n):
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_idx = []
    vals = []
    for i in range(n):
        if i > 0:
            col_idx.append(i - 1); vals.append(-1.0)
        col_idx.append(i); vals.append(2.0)
        if i < n - 1:
            col_idx.append(i + 1); vals.append(-1.0)
        row_ptr[i + 1] = len(col_idx)
    return (np.ascontiguousarray(row_ptr),
            np.ascontiguousarray(np.asarray(col_idx, dtype=np.int32)),
            np.ascontiguousarray(np.asarray(vals, dtype=np.float64)))


def _random_diag_dom(n, seed):
    rng = np.random.default_rng(seed)
    density = min(5.0 / n, 1.0)
    A = np.zeros((n, n), dtype=np.float64)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    offd = rng.uniform(-0.1, 0.1, size=(n, n))
    A = np.where(mask, offd, 0.0)
    A = 0.5 * (A + A.T)
    diag = rng.uniform(1.0, 10.0, size=n)
    row_abs_sum = np.sum(np.abs(A), axis=1)
    diag = np.maximum(diag, row_abs_sum + 1.0)
    np.fill_diagonal(A, diag)
    return dense_to_csr(A)


def _check(name, csr, tol_prefactor=100.0):
    row_ptr, col_idx, values = csr
    n = int(row_ptr.shape[0]) - 1
    diag_offset = build_diag_offset(row_ptr, col_idx)

    # Reference (NumPy, serial).
    d_ref = dilu_d_reference(row_ptr, col_idx, values)

    # GPU.
    d_gpu = np.asarray(dilu_factor(
        jax.device_put(jnp.asarray(row_ptr)),
        jax.device_put(jnp.asarray(col_idx)),
        jax.device_put(jnp.asarray(values)),
        jax.device_put(jnp.asarray(diag_offset)),
    ))

    abs_err = np.max(np.abs(d_gpu - d_ref))
    rel = abs_err / max(np.max(np.abs(d_ref)), 1.0)
    eps = np.finfo(np.float64).eps
    tol = tol_prefactor * n * eps * max(np.max(np.abs(d_ref)), 1.0)
    print(f"{name}: n={n}, max|Δd|={abs_err:.3e}, rel={rel:.3e}, tol={tol:.3e}")
    assert abs_err <= tol, f"{name} failed: err {abs_err:.3e} > tol {tol:.3e}"


def test_t4_1_tridiag():
    _check("T4.1 tridiag", _tridiag_csr(500))


def test_t4_2_laplacian3d():
    _check("T4.2 laplacian3d", laplacian_3d_7point(10, 10, 10))


def test_t4_3_random_diag_dom():
    _check("T4.3 random DD", _random_diag_dom(200, seed=0))


if __name__ == "__main__":
    test_t4_1_tridiag()
    test_t4_2_laplacian3d()
    test_t4_3_random_diag_dom()
    print("T4 OK")
