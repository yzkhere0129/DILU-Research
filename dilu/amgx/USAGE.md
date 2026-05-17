# Using dilu-amgx

## 30-second example

```python
import jax
jax.config.update("jax_enable_x64", True)   # AMGx is float64-only
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from dilu_amgx import Plan, CLASSICAL_V_DIAGSCALED, with_tolerance

# Your CSR matrix (e.g. an OpenFOAM lduMatrix snapshot).
# CAVEAT: if diagonal is negative (OpenFOAM convention), flip BOTH A and b:
A = sp.csr_matrix(...)        # n x n SPD or nearly so
b = np.asarray(...)           # RHS, length n
if float(np.mean(A.diagonal())) < 0:
    A, b = (-A).tocsr(), -b

rp = jnp.asarray(A.indptr.astype(np.int32))
ci = jnp.asarray(A.indices.astype(np.int32))
vv = jnp.asarray(A.data.astype(np.float64))
b_d = jnp.asarray(b)
x0  = jnp.zeros_like(b_d)

cfg = with_tolerance(CLASSICAL_V_DIAGSCALED, tol=1e-12, max_iters=500)
with Plan(rp, ci, vv, cfg) as plan:
    x, iters, status = plan.solve(b_d, x0)
    x.block_until_ready()

print(f"iters={int(iters[0])}, status={int(status[0])}, "
      f"||r||/||b|| = {np.linalg.norm(A @ x - b) / np.linalg.norm(b):.3e}")
```

## Warm-start amortization (the headline feature)

Setup is expensive (~10-30× a single solve). When the matrix's sparsity
pattern is unchanged across timesteps — only its values change — reuse the
same `Plan` and call `update_coefficients` between solves:

```python
with Plan(rp, ci, vv_step_0, cfg) as plan:
    x_prev = jnp.zeros(n)
    for k in range(1, n_steps):
        vv_new = jnp.asarray(A_step_k.data.astype(np.float64))
        plan.update_coefficients(vv_new)
        b_d = jnp.asarray(b_step_k)
        x, iters, status = plan.solve(b_d, x_prev)    # warm-start with x_prev
        x.block_until_ready()
        x_prev = x
```

On dense temporal sampling (LPBF dump at ~3 ns intervals), warm-start
reduces iteration count by ~47% on average vs cold-start.

## Iterative refinement (machine precision)

For cases where AMGx's PCG residual stagnates at `~1e-13`:

```python
from dilu_amgx import amgx_solve_with_refinement

res = amgx_solve_with_refinement(
    A, b, x0,
    eq_kind="pd",     # "pd" → PCG, "T" → BiCGStab
    tol=1e-12,
    n_refine=1,       # 1 IR step is usually enough
    max_iters=500,
)
print(res["x"])                  # the solution
print(res["primary_iters"])      # outer Krylov iterations
print(res["refine_iters"])       # list of [iter counts per IR step]
print(res["rel_residual"])       # final ||r||/||b||
```

Typical result on LPBF dumps: `rel_residual ≈ 1e-15` (machine ε), matching
scipy `spsolve` to all displayed digits.

## Choosing an `eq_kind`

| eq_kind | Outer Krylov | Use when                                            |
| ------- | ------------ | --------------------------------------------------- |
| `"pd"`  | PCG          | A is SPD (after sign flip). LPBF pressure equation. |
| `"T"`   | BiCGStab     | A is asymmetric (asym ~ 1e-6). LPBF energy equation.|

`"pd"` will silently produce wrong answers if A is genuinely asymmetric; check
with `np.abs((A - A.T).data).max()` before choosing.

## Pitfalls

- **Float32**: AMGx FFI is float64-only. Make sure `jax_enable_x64` is on
  *before* you import `dilu_amgx` or any JAX. The package no longer mutates
  global JAX config at import time — that responsibility is yours.
- **Negative diagonal**: OpenFOAM exports its pressure-correction system with
  negative diagonal. AMGx PCG requires positive diagonal (SPD). The
  `load_matrix_npz` helper in `benchmarks/run_benchmark.py` shows the
  canonical sign-flip; do the same yourself.
- **GC leaks**: don't rely on `Plan.__del__`. Always use the context manager
  or call `plan.release()` explicitly before process exit. `Plan.__del__` is
  defensive but invokes JAX FFI from the GC thread, which is undefined
  behavior.
- **CUDA architecture**: `libdilu_amgx.so` is built for a specific SM. Using
  it on a different SM gives ABI errors. Rebuild with `DILU_AMGX_CUDA_ARCH=`
  set per GPU.

For the long list (15 real traps), see `KNOWN_GOTCHAS.md`.
