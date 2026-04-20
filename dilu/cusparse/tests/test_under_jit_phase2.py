"""Under-jit verification for Phase 2 apply primitive.

Checks (per brief):
  - ffi_call primitive appears in jaxpr.
  - Compiled HLO contains one custom-call per FFI primitive per jitted call.
  - The ONLY tolerated host↔device traffic is the 8-byte uint64 token read
    per apply. Specifically the HLO should have the token-copy pattern but
    nothing larger.
"""
import conftest  # noqa: F401

import re
import numpy as np
import jax
import jax.numpy as jnp

from dilu.cusparse.python import build_diag_offset, Plan, cusparse_dilu_apply
from _harness import laplacian_3d_7point


def test_under_jit_apply_hlo():
    row_ptr, col_idx, values = laplacian_3d_7point(6, 6, 6)
    n = 6 * 6 * 6
    diag_offset = build_diag_offset(row_ptr, col_idx)

    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))
    rng = np.random.default_rng(1)
    r_d = jax.device_put(jnp.asarray(rng.standard_normal(n).astype(np.float64)))

    with Plan(rp, ci, vv, do) as plan:
        d_star = plan.factor(vv)

        @jax.jit
        def run_apply(values, d_star, r):
            return cusparse_dilu_apply(plan.token, values, d_star, r)

        z = run_apply(vv, d_star, r_d)
        z.block_until_ready()

        jaxpr = jax.make_jaxpr(run_apply)(vv, d_star, r_d)
        jaxpr_str = str(jaxpr)
        assert "ffi_call" in jaxpr_str, f"ffi_call not in jaxpr:\n{jaxpr_str}"

        hlo = jax.jit(run_apply).lower(vv, d_star, r_d).compile().as_text()
        # Count custom-calls: one per FFI primitive in the jitted function
        # (here, only apply).
        n_custom = len(re.findall(r"custom-call", hlo))
        print(f"under-jit apply: custom-call count = {n_custom}")
        assert n_custom == 1, f"expected 1 custom-call, got {n_custom}:\n{hlo[:500]}"

        # copy-start / copy-done only appear in HLO for HOST↔DEVICE transfers
        # (not for D↔D moves). The apply handler does one 8-byte D→H copy for
        # the token, but that is INSIDE the C++ handler — XLA does not see it
        # and does not emit HLO copy-start/copy-done for it. So HLO should
        # still show ZERO copy-start/copy-done.
        assert "copy-start" not in hlo, (
            "Unexpected copy-start in HLO — host copy leaked:\n" + hlo[:1000])
        assert "copy-done" not in hlo, (
            "Unexpected copy-done in HLO — host copy leaked:\n" + hlo[:1000])

        print("under-jit apply: 1× custom-call, 0× HLO-visible host copy")


def test_under_jit_factor_hlo():
    # dilu_factor should also fit a single custom-call under jit.
    row_ptr, col_idx, values = laplacian_3d_7point(4, 4, 4)
    diag_offset = build_diag_offset(row_ptr, col_idx)
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    do = jax.device_put(jnp.asarray(diag_offset))

    from dilu.cusparse.python import dilu_factor

    @jax.jit
    def run_factor(vv):
        return dilu_factor(rp, ci, vv, do)

    d = run_factor(vv); d.block_until_ready()

    hlo = jax.jit(run_factor).lower(vv).compile().as_text()
    n_custom = len(re.findall(r"custom-call", hlo))
    print(f"under-jit factor: custom-call count = {n_custom}")
    assert n_custom == 1
    assert "copy-start" not in hlo and "copy-done" not in hlo


if __name__ == "__main__":
    test_under_jit_apply_hlo()
    test_under_jit_factor_hlo()
    print("under-jit phase2 OK")
