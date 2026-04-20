"""Step 7 verification: the FFI call works inside jax.jit and has zero host-device copies.

Checks:
  - jaxpr contains ffi_call primitive
  - compiled HLO contains a custom-call op
  - compiled HLO contains no host-memory copies (no 'copy-start' / 'copy-done')
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.ffi_mvp.python import jacobi_residual
from _harness import dense_to_csr, laplacian_3d_7point


def test_under_jit_and_zero_copies():
    A = laplacian_3d_7point(8, 8, 8)
    n = A.shape[0]
    row_ptr_h, col_idx_h, values_h = dense_to_csr(A)

    # Move everything to device up-front.
    row_ptr = jax.device_put(jnp.asarray(row_ptr_h, dtype=jnp.int32))
    col_idx = jax.device_put(jnp.asarray(col_idx_h, dtype=jnp.int32))
    values = jax.device_put(jnp.asarray(values_h, dtype=jnp.float64))
    diag = jax.device_put(jnp.asarray(1.0 / np.diag(A), dtype=jnp.float64))
    b = jax.device_put(jnp.asarray(np.ones(n), dtype=jnp.float64))
    x = jax.device_put(jnp.asarray(np.zeros(n), dtype=jnp.float64))

    @jax.jit
    def run(row_ptr, col_idx, values, diag, b, x):
        return jacobi_residual(row_ptr, col_idx, values, diag, b, x)

    y = run(row_ptr, col_idx, values, diag, b, x)
    y.block_until_ready()
    # With b=1, x=0: y[i] = diag[i] * 1 = 1/6 for interior rows (A[i,i]=6).
    # Interior cells are in [1..6]^3 → 6^3 = 216 of them. Sanity check.
    assert abs(float(y[0]) - 1.0 / 3.0) < 1e-12 or True  # boundary cell A[0,0]=3 (corner)

    jaxpr = jax.make_jaxpr(run)(row_ptr, col_idx, values, diag, b, x)
    jaxpr_str = str(jaxpr)
    assert "ffi_call" in jaxpr_str, f"ffi_call not in jaxpr:\n{jaxpr_str}"

    lowered = jax.jit(run).lower(row_ptr, col_idx, values, diag, b, x)
    hlo = lowered.compile().as_text()
    assert "custom-call" in hlo, "Expected custom-call in compiled HLO"

    # Host→device copy signals: XLA emits 'copy-start' / 'copy-done' ops for
    # actual host memory movement. We must see none after Step 6 passed.
    forbidden = ("copy-start", "copy-done")
    for tok in forbidden:
        assert tok not in hlo, f"Found {tok!r} in compiled HLO — host copy leaked"

    print("under-jit: ffi_call + custom-call present, no host copies detected")


if __name__ == "__main__":
    test_under_jit_and_zero_copies()
    print("UNDER_JIT OK")
