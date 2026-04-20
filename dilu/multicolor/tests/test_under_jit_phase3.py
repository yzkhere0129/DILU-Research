"""Under-jit HLO verification for Phase 3 apply.

Per brief §Testing: single apply inside `jax.jit`, inspect HLO.
  - Exactly 1 custom-call per Phase 3 apply invocation.
  - Zero HLO-visible copy-start/copy-done (the 8-byte D→H token copy happens
    inside the C++ handler and is NOT surfaced to HLO).
"""
import conftest  # noqa: F401

import re
import numpy as np
import jax
import jax.numpy as jnp

from dilu.multicolor.python import MulticolorPlan, multicolor_apply
from _harness import laplacian_3d_7point


def test_under_jit_apply_hlo():
    nx = ny = nz = 6
    N = nx * ny * nz
    row_ptr, col_idx, values = laplacian_3d_7point(nx, ny, nz)

    rng = np.random.default_rng(1)
    r_h = rng.standard_normal(N).astype(np.float64)
    r_d = jax.device_put(jnp.asarray(r_h))
    values_d = jax.device_put(jnp.asarray(values))

    with MulticolorPlan(row_ptr, col_idx, values,
                        grid_shape=(nx, ny, nz)) as plan:
        d_star = plan.factor(values_d)

        @jax.jit
        def run_apply(values, d_star, r):
            return multicolor_apply(plan.token, values, d_star, r)

        z = run_apply(values_d, d_star, r_d)
        z.block_until_ready()

        jaxpr = jax.make_jaxpr(run_apply)(values_d, d_star, r_d)
        jaxpr_str = str(jaxpr)
        assert "ffi_call" in jaxpr_str, f"ffi_call missing in jaxpr:\n{jaxpr_str}"

        hlo = jax.jit(run_apply).lower(values_d, d_star, r_d).compile().as_text()
        n_custom = len(re.findall(r"custom-call", hlo))
        print(f"under-jit apply: custom-call count = {n_custom}")
        assert n_custom == 1, (
            f"expected 1 custom-call (the apply handler), got {n_custom}"
        )

        # HLO should have zero visible host copies — the 8-byte token D→H and
        # the internal offsets-mirror access both happen inside the handler
        # and are NOT surfaced to the XLA graph.
        assert "copy-start" not in hlo, (
            "Unexpected copy-start in HLO — host copy leaked\n" + hlo[:1000])
        assert "copy-done" not in hlo, (
            "Unexpected copy-done in HLO — host copy leaked\n" + hlo[:1000])
        print("under-jit apply: 1x custom-call, 0x HLO-visible host copy")


if __name__ == "__main__":
    test_under_jit_apply_hlo()
    print("under-jit phase3 OK")
