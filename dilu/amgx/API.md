# dilu-amgx API Reference

All public symbols are exported from the top-level `dilu_amgx` package
(or, in monorepo development, from `dilu.amgx.python`). The two import
paths are equivalent; pick whichever fits your project layout.

## FFI primitives

These are direct XLA FFI handlers. The C++ implementations live in `cpp/`
and are bound through `python/wrapper.py` + `python/registration.py`.

### `amgx_setup(row_ptr, col_idx, values, config_json) -> token`

| Parameter     | Type                          | Notes                                          |
| ------------- | ----------------------------- | ---------------------------------------------- |
| `row_ptr`     | `jnp.int32[n+1]` (device)     | CSR row offsets                                |
| `col_idx`     | `jnp.int32[nnz]` (device)     | CSR column indices                             |
| `values`      | `jnp.float64[nnz]` (device)   | CSR nonzero values                             |
| `config_json` | `str` (host, FFI attr)        | AMGx JSON; see `python/config.py` for canon    |
| **returns**   | `jnp.uint64[1]` (device)      | Opaque token registered in a process-global map|

Builds the full AMG hierarchy (the expensive operation, typically 10-30×
the cost of a solve). Call once per unique matrix sparsity pattern.

### `amgx_update_coefficients(token, values) -> status`

| Parameter   | Type                        | Notes                                              |
| ----------- | --------------------------- | -------------------------------------------------- |
| `token`     | `jnp.uint64[1]` (device)    | From a previous `amgx_setup`                       |
| `values`    | `jnp.float64[nnz]` (device) | New values for the SAME (row_ptr, col_idx) pattern |
| **returns** | `jnp.int32[1]` (device)     | `0` = OK; non-zero = AMGx error code               |

Calls `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup`. Coarse
grids and interpolators are preserved; only operator-dependent smoother
state is recomputed. Roughly 10-30× cheaper than a full setup.

NNZ is fingerprint-checked; passing a different NNZ fails synchronously
with `nnz fingerprint mismatch`.

### `amgx_solve(token, b, x0) -> (x, iters, status)`

| Parameter   | Type                        | Notes                                                  |
| ----------- | --------------------------- | ------------------------------------------------------ |
| `token`     | `jnp.uint64[1]` (device)    |                                                        |
| `b`         | `jnp.float64[n]` (device)   | RHS                                                    |
| `x0`        | `jnp.float64[n]` (device)   | Initial guess; use `jnp.zeros(n)` for cold start       |
| **returns** | tuple of 3 device arrays    | `x: float64[n]`, `iters: int32[1]`, `status: int32[1]` |

`status` decodes to `AMGX_SOLVE_STATUS`:
- `0` = SUCCESS
- `1` = FAILED
- `2` = DIVERGED
- `3` = NOT_CONVERGED (max_iters reached)

The handler synchronizes the XLA stream BEFORE the AMGx solve (so any
preceding JAX ops have committed) and AFTER (so subsequent JAX ops see
the finished solve). No pipelining; correctness over throughput at the
matrix sizes we target.

### `amgx_release(token) -> status`

| Parameter   | Type                        | Notes                                       |
| ----------- | --------------------------- | ------------------------------------------- |
| `token`     | `jnp.uint64[1]` (device)    |                                             |
| **returns** | `jnp.int32[1]` (device)     | Always `0` (idempotent — see contract note) |

Idempotent: releasing an unknown / already-removed token returns `0`.
Releases the AMGx solver, matrix, vectors, config, and registry slot.

**Contract note (post-2026-05 patch):** the C++ handler returns `0` for
success including the "token was already absent" case. This matches the
Python wrapper docstring `0 = OK`. Pre-patch the C++ inverted this — be
sure you are running the patched build.

## High-level wrappers

### `Plan(row_ptr, col_idx, values, config_json)`

Context manager wrapping the setup / solve / release lifecycle.

```python
with Plan(rp, ci, vv, cfg) as plan:
    x, iters, status = plan.solve(b, x0)
    # values-only update for the next timestep:
    plan.update_coefficients(vv_new)
    x, iters, status = plan.solve(b_new, x)   # warm-start
```

- Strong-refs `row_ptr` and `col_idx` for the plan's lifetime so JAX cannot
  recycle their device buffers.
- Calls `block_until_ready()` on the setup output so setup failures surface
  at `__enter__`, not at first solve.
- `__exit__` calls `release()`; `__del__` calls `release()` defensively but
  GC-thread FFI is undefined behavior — prefer the context manager.

Properties:
- `plan._released: bool` — set by `release()`. Useful for leak-check tests.

`Plan.__del__` calls `release()` defensively in a `try / except: pass`
block, so accidental drop-on-GC will not leak the AMGx handle in normal
cases. However, GC-thread invocation of JAX FFI is undefined behavior and
the AMGx warning `detected some memory leaks in the code: trying to free
non-empty temporary device pool` may appear at process exit if a `Plan`
was never explicitly released. The reliable pattern is the `with` block;
the destructor is a safety net, not the API contract.

### `amgx_solve_with_refinement(A, b, x0, eq_kind, tol, max_iters, n_refine, ...) -> dict`

Wilkinson/Higham §12.1 iterative refinement. Solves `A x_0 ≈ b`, then for
each IR step computes `r = b - A x` (in float64) and solves `A δ = r`,
updates `x ← x + δ`.

| Parameter   | Default | Notes                                                          |
| ----------- | ------- | -------------------------------------------------------------- |
| `A`         |         | `scipy.sparse.csr_matrix` (host)                               |
| `b`         |         | `np.ndarray[n]`                                                |
| `x0`        |         | `np.ndarray[n]` initial guess                                  |
| `eq_kind`   | `"pd"`  | `"pd"` → `CLASSICAL_V_DIAGSCALED` (PCG); `"T"` → BiCGStab      |
| `tol`       | `1e-12` | Primary + IR solve tolerance                                   |
| `max_iters` | `500`   | Primary solver cap                                             |
| `n_refine`  | `2`     | IR steps after primary; `1` is usually enough for `~1e-15`     |
| `verbose`   | `False` | Print per-step residual                                        |

`x0` defaults to `np.zeros(n)` if you pass `None`. For warm-start across
timesteps, pass the previous step's `res["x"]`.

`refinement_history` has length `n_refine + 1`: index 0 is the relative
residual after the primary solve (no IR yet), indices 1..n_refine are the
relative residuals after each IR step. The list `refine_iters` has length
exactly `n_refine` and records the inner Krylov iteration count for each
δ-solve.

Returns:

```python
{
    "x":                  np.ndarray[n],   # solution
    "primary_iters":      int,
    "primary_status":     int,
    "refine_iters":       list[int],       # one entry per IR step
    "abs_residual":       float,           # ||A x - b||_2
    "rel_residual":       float,           # ||A x - b||_2 / ||b||_2
    "refinement_history": list[float],     # rel residual after each step
    "total_s":            float,           # wall, incl. Plan setup
}
```

### `enable_x64()` — opt-in JAX float64

Equivalent to `jax.config.update("jax_enable_x64", True)`. Call once at
process start *before* any JAX usage, or set the env var
`JAX_ENABLE_X64=1`. AMGx is float64-only; without x64 the FFI buffers will
have the wrong dtype and the solve will fail at boundary check.

Importing `dilu_amgx` does NOT mutate JAX config (this changed in v1.0.0;
prior versions did, which broke downstream float32 pipelines silently).

## Config strings

Canonical AMGx JSON strings, character-exact:

| Name                                 | Outer       | Preconditioner             | Use case                                |
| ------------------------------------ | ----------- | -------------------------- | --------------------------------------- |
| `CLASSICAL_V_CYCLE`                  | (AMG only)  | —                          | Reference (single V-cycle, no Krylov)   |
| `CLASSICAL_V_DIAGSCALED`             | PCG         | Classical V + diag scaling | **Production pd (LPBF pressure)**       |
| `CLASSICAL_V_DIAGSCALED_TIGHT`       | PCG         | same, narrower coarsening  | Stress-test                             |
| `CLASSICAL_V_DIAGSCALED_BICGSTAB`    | BiCGStab    | same                       | **Production T (LPBF energy)**          |
| `CLASSICAL_GS_PCG`                   | PCG         | Classical V + Gauss-Seidel | Slower but more robust                  |
| `CLASSICAL_GS_BICGSTAB`              | BiCGStab    | same                       | Asymm + extra-robust                    |
| `AGGREGATION_PCG`                    | PCG         | Aggregation AMG            | Alternative coarsening                  |
| `AGGRESSIVE_COARSENING`              | (AMG only)  | —                          | Memory-tight cases                      |
| `MINI_AMG_TEST`                      | (smoke)     | —                          | Smoke test only                         |

### `with_tolerance(base_config, tol, max_iters=None) -> str`

Patches the `tolerance` and (optionally) `max_iters` fields of a base
config JSON string in-place. Returns a new JSON string suitable for
`Plan(..., config_json=cfg)`. Whitespace and key ordering of the base
string are preserved as much as possible (re-serialized via
`json.dumps`).

For a *bit-exact* iteration-count reproduction, copy the base config
strings character-for-character from `python/config.py` and patch only
via this helper — do not hand-edit the JSON.
