"""T11 — HLO inspection: amgx_solve inside jax.jit.

Acceptance (design §6.5):
    - Under jax.jit, the compiled HLO for a function that wraps amgx_solve
      contains exactly ONE custom-call per invocation of amgx_solve.
    - ZERO HLO-visible `copy` ops on user-controlled data (i.e. no
      HLO-lowered host copies of A, b, x). The 8-byte D->H token copy per
      solve IS allowed (it happens inside the FFI handler, below the HLO
      level, and shows up in Nsight but not HLO).

We grep the compiled HLO text for `custom-call` and `copy-start`.
"""
import conftest  # noqa: F401

import numpy as np
import jax
import jax.numpy as jnp

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance
from dilu.amgx.tests._harness import laplacian_3d_7point
from dilu.amgx.python.wrapper import amgx_solve


def test_t11_under_jit_8cubed():
    nx = ny = nz = 8
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)
    n = nx * ny * nz

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))

    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=1e-8, max_iters=50)
    rng = np.random.default_rng(0)
    b = jax.device_put(jnp.asarray(rng.standard_normal(n)))
    x0 = jnp.zeros(n, dtype=jnp.float64)

    with AmgxPlan(rp, ci, vv, cfg) as plan:
        # Solve once eagerly to make sure the plan works before we JIT.
        x_eager, _, _ = plan.solve(b, x0)
        x_eager.block_until_ready()

        # Wrap amgx_solve in a jit. token is captured as a closure value.
        token = plan.token

        def run(b_in, x0_in):
            x, iters, status = amgx_solve(token, b_in, x0_in)
            return x, iters, status

        run_jit = jax.jit(run)
        x_jit, iters_jit, status_jit = run_jit(b, x0)
        x_jit.block_until_ready()

        hlo = jax.jit(run).lower(b, x0).compile().as_text()

    # Accept 1 custom-call per amgx_solve invocation.
    n_custom_call = hlo.count("custom-call")
    # Ignore copy-start/copy-done lines if any — they appear for
    # cross-partition copies; on single-GPU we should see none on user data.
    n_copy_start = hlo.count("copy-start")
    n_copy_done = hlo.count("copy-done")
    print(f"T11 HLO: custom-call={n_custom_call}, copy-start={n_copy_start}, "
          f"copy-done={n_copy_done}")
    # Print a compact view for the log.
    snip = "\n".join(l for l in hlo.splitlines() if "custom-call" in l or "copy-" in l)
    print("T11 HLO snippet:\n" + snip)

    # 1 custom-call (exactly).
    assert n_custom_call == 1, f"expected 1 custom-call, got {n_custom_call}"
    # 0 copy ops on the HLO level. (Per design, token is a JAX array;
    # JAX may generate 0 or 1 copy-start for the token itself being
    # broadcast to the jitted function — check: we captured it in closure
    # so it's baked in, shouldn't appear.)
    assert n_copy_start == 0, f"unexpected copy-start ops: {n_copy_start}"
    assert n_copy_done == 0, f"unexpected copy-done ops: {n_copy_done}"

    # Also verify JIT and eager produce the same answer (sanity).
    max_diff = float(jnp.max(jnp.abs(jnp.asarray(x_jit) - jnp.asarray(x_eager))))
    print(f"T11 eager vs jit max diff: {max_diff:.3e}")
    assert max_diff <= 1e-10, f"eager/jit mismatch {max_diff}"
    print("T11 PASS")


if __name__ == "__main__":
    test_t11_under_jit_8cubed()
