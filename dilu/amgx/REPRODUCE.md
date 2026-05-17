# AMGx FFI Module — Reproducibility Specification

## Purpose

This document describes the exact recipe to rebuild the `dilu/amgx` module
(a JAX-FFI shim over NVIDIA AMGx 2.5.0) such that on the same input matrix
it produces the same iteration count and the same solution residual to
ε-machine precision. The intended audience is a blank-context AI or
engineer who has access only to this document. Every "MUST" is load-bearing;
deviating from a "MUST" invalidates byte-equivalent reproduction. The four
primitives exposed are `amgx_setup`, `amgx_update_coefficients`,
`amgx_solve`, `amgx_release`, plus a `Plan` context manager and an
iterative-refinement helper. AMGx's classical AMG / PCG hierarchy is itself
deterministic for a fixed config and fixed device coarsening order, so
"byte-identical" is achievable on the same GPU SKU and CUDA driver pair.

## System Requirements

- Hardware: CUDA-capable GPU, compute capability sm_75 minimum. Verified on
  sm_86 (RTX 3050 Laptop) and sm_120 (RTX 5060). Single GPU, device 0.
- OS: Linux x86-64. Verified on Ubuntu 24.04 LTS.
- CUDA toolkit: 12.4 verified, 13.x verified. The CUDA that AMGx is built
  against MUST match the CUDA that JAX is using; mismatched majors produce
  `cudaErrorInvalidValue` at the first `cudaMemcpyAsync`.
- Python: ≥ 3.12. (3.14 also works but requires the dual-platform FFI
  registration shim documented below; the shim is already in
  `registration.py`.)
- C++: C++17 compiler (gcc 11 or 12 verified). CMake ≥ 3.24.
- Build flags: `-O2`, `-fno-fast-math` (FP-determinism repo policy). NEVER
  use `-O3 -ffast-math`; it will perturb AMGx residual reductions.
- MPI: not used. AMGx MUST be built with `-DMPI_FOUND=FALSE`.

## Step 1: Build AMGx 2.5.0

```bash
cd ~/src
git clone --branch v2.5.0 --depth 1 https://github.com/NVIDIA/AMGX.git amgx-2.5.0
cd amgx-2.5.0
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$HOME/local/amgx" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_FOUND=FALSE \
  -DCMAKE_CUDA_ARCHITECTURES="86"      # adjust for your GPU
make -j8
make install
```

After install, you MUST have:

- `$HOME/local/amgx/lib/libamgxsh.so`
- `$HOME/local/amgx/include/amgx_c.h`

The shim links against the SHARED library `libamgxsh.so`, not the static
`libamgx.a`. We do NOT call `AMGX_finalize()` at process exit — see
Appendix B.

Re-implementer override: set `AMGX_ROOT` env var to point elsewhere; the
CMake script honors `-DAMGX_ROOT=...`, `$AMGX_ROOT`, or defaults to
`$HOME/local/amgx` in that order.

## Step 2: Install JAX

Use a clean pip virtualenv (NOT conda):

```bash
python3.12 -m venv ~/jax-env
source ~/jax-env/bin/activate
pip install -U pip
pip install "jax[cuda12]==0.9.0"
```

WARNING: do NOT use a conda-forge JAX build. Conda's JAX has a known FFI
dispatch bug where the CUDA platform string is canonicalized to lower-case
`"cuda"` instead of the historical `"CUDA"`, and historical handlers
registered under `"CUDA"` are not found. Our `registration.py` registers
under BOTH names to defend against this, but the conda build also has
unrelated XLA-FFI ABI drift that we have not stabilized against.

After install, verify:

```bash
python -c "import jax; print(jax.devices()); print(jax.config.read('jax_enable_x64'))"
# Expect: [CudaDevice(id=0)] ; True  (x64 is forced True by refinement.py import)
```

## Step 3: Build FFI shim libdilu_amgx.so

Layout of the module:

```
dilu/amgx/
  CMakeLists.txt
  build.sh
  cpp/
    plan_registry.h
    plan_registry.cc
    amgx_setup.cc
    amgx_update_coefficients.cc
    amgx_solve.cc
    amgx_release.cc
  python/
    __init__.py
    registration.py
    wrapper.py
    plan.py
    config.py
    refinement.py
```

Build:

```bash
source ~/jax-env/bin/activate
cd dilu/amgx
AMGX_ROOT=$HOME/local/amgx ./build.sh
```

Output: `dilu/amgx/build/libdilu_amgx.so` with `RPATH=$AMGX_ROOT/lib`.

CUDA architecture override: `build.sh` reads `DILU_AMGX_CUDA_ARCH` from the
environment (default `86`). To target sm_120 (RTX 5060), sm_90 (H100), or a
fat binary, set it before invoking:

```bash
DILU_AMGX_CUDA_ARCH=120     ./build.sh    # RTX 5060 Blackwell
DILU_AMGX_CUDA_ARCH="86;90" ./build.sh    # 3050 + H100 fat
```

Or bypass `build.sh` and pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` directly
to cmake. CMakeLists.txt's `if(NOT DEFINED ...)` guard only fires when the
variable was not set by either the env wrapper or the cmake CLI.

`build.sh` invokes `ldd libdilu_amgx.so` after build; you MUST see
`libamgxsh.so => /home/USER/local/amgx/lib/libamgxsh.so` resolved.

## Step 4: Python API Reference

The four FFI primitives, declared in `python/wrapper.py`, registered by
`python/registration.py`:

### `amgx_setup(row_ptr, col_idx, values, config_json: str) -> uint64[1]`
- `row_ptr`: `jnp.int32[n+1]` device CSR row offsets
- `col_idx`: `jnp.int32[nnz]` device CSR column indices
- `values`:  `jnp.float64[nnz]` device CSR values
- `config_json`: Python `str`. Passed via XLA FFI Attr (host-side, not
  traced). Must be a valid AMGx JSON (see Step 5).
- Returns: `jnp.uint64[1]` token. The token is opaque; the underlying
  PlanEntry lives in a process-global registry guarded by a mutex.
- Side effects: builds the full AMG hierarchy (expensive). Call once per
  unique matrix sparsity pattern.

### `amgx_update_coefficients(token, values) -> int32[1]`
- `token`: `jnp.uint64[1]` from a previous `amgx_setup`.
- `values`: `jnp.float64[nnz]` — new values for the SAME (row_ptr, col_idx)
  pattern. NNZ is fingerprint-checked; passing a different NNZ fails.
- Returns: `jnp.int32[1]` AMGx return code (0 = OK).
- Side effects: calls `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup`.
  The C/F splitting is preserved; only operator-dependent smoother state is
  recomputed. Roughly 10-30x cheaper than a full setup.

### `amgx_solve(token, b, x0) -> (x, iters, status)`
- `token`: `jnp.uint64[1]`
- `b`: `jnp.float64[n]` device RHS
- `x0`: `jnp.float64[n]` device initial guess (use `jnp.zeros(n)` for cold)
- Returns:
  - `x`: `jnp.float64[n]` solution
  - `iters`: `jnp.int32[1]` outer Krylov iteration count
  - `status`: `jnp.int32[1]` AMGX_SOLVE_STATUS — 0=SUCCESS, 1=FAILED,
    2=DIVERGED, 3=NOT_CONVERGED
- Side effects: synchronous on the XLA stream (we `cudaStreamSynchronize`
  before AND after AMGx for correctness; we do not pipeline).

### `amgx_release(token) -> int32[1]`
- `token`: `jnp.uint64[1]`
- Returns: `jnp.int32[1]` — `0 = success`, including the idempotent case
  where the token was already absent. Non-zero is reserved for a future
  "actually failed to tear down" condition. Reproducer MUST keep the
  Python and C++ contracts aligned on `0 = OK`.
- Idempotent: releasing an already-removed token returns 0 silently.

### `Plan` context manager (`python/plan.py`)
- Strong-refs `row_ptr` and `col_idx` for the plan's lifetime so JAX
  cannot recycle their device buffers.
- Calls `block_until_ready()` after `amgx_setup` to surface setup failures
  immediately (not at first solve).
- `__exit__` calls `release()`; `__del__` calls `release()` defensively.

### `amgx_solve_with_refinement(A, b, x0, eq_kind, tol, max_iters, n_refine, ...)`
- Located in `python/refinement.py`. Wraps Plan + n iterative-refinement
  passes (Wilkinson/Higham §12.1 style).
- `eq_kind="pd"` selects `CLASSICAL_V_DIAGSCALED` (PCG).
- `eq_kind="T"` selects `CLASSICAL_V_DIAGSCALED_BICGSTAB` (BiCGStab).
- Primary solve runs at `tol`, then up to `n_refine` correction solves
  reusing the same Plan handle. Returns a dict with `x`, `primary_iters`,
  `refine_iters`, `abs_residual`, `rel_residual`, `refinement_history`,
  `total_s`.
- IMPORTANT: AMGx FFI is float64-only. The caller is responsible for
  enabling x64 (`jax.config.update("jax_enable_x64", True)` at process start,
  or call `dilu.amgx.refinement.enable_x64()`). Importing this module no
  longer mutates global JAX state.

## Step 5: Canonical Config JSON Strings

The following 9 strings live verbatim in `python/config.py`. A reproducer
MUST copy them character-for-character (whitespace included). AMGx parses
this via `AMGX_config_create`. Any reformatting that changes a numeric
literal, key name, or nesting structure WILL change observable iteration
counts.

### `CLASSICAL_V_CYCLE`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `AGGRESSIVE_COARSENING`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "aggressive_levels": 2,
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D1",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `CLASSICAL_V_DIAGSCALED`

```json
{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `CLASSICAL_V_DIAGSCALED_TIGHT`

```json
{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 500,
        "tolerance": 1e-14,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `CLASSICAL_V_DIAGSCALED_BICGSTAB`

```json
{
    "config_version": 2,
    "solver": {
        "scaling": "DIAGONAL_SYMMETRIC",
        "solver": "BICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `CLASSICAL_GS_PCG`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `CLASSICAL_GS_BICGSTAB`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "BICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `AGGREGATION_PCG`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_2",
            "smoother": {
                "scope": "smoother",
                "solver": "MULTICOLOR_GS",
                "relaxation_factor": 1.0,
                "symmetric_GS": 1,
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 2,
            "postsweeps": 2,
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

### `MINI_AMG_TEST`

```json
{
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "smoother": {
                "scope": "jacobi",
                "solver": "BLOCK_JACOBI",
                "monitor_residual": 0,
                "print_solve_stats": 0
            },
            "presweeps": 1,
            "postsweeps": 1,
            "interpolator": "D2",
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 5,
            "monitor_residual": 0,
            "store_res_history": 0,
            "scope": "amg",
            "print_grid_stats": 0,
            "print_solve_stats": 0
        },
        "max_iters": 200,
        "tolerance": 1e-10,
        "convergence": "RELATIVE_INI_CORE",
        "norm": "L2",
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "store_res_history": 0,
        "scope": "main"
    }
}
```

The `with_tolerance(base_config, tol, max_iters=200)` helper uses
`json.loads` / `json.dumps` to override `solver.tolerance` and
`solver.max_iters` while preserving the rest. The output is `json.dumps`
default formatting (no indent, ASCII-only keys); a reproducer MUST use
`json.dumps` with default args, NOT pretty-print, when emulating this
helper — AMGx is whitespace-insensitive so the wire format does not affect
solver numerics, but downstream tests that hash the config string would
diverge.

## Step 6: Sign Convention

Different upstream matrix sources have different diagonal-sign conventions.
AMGx's PCG requires SPD (positive diagonal). For incompatible inputs the
caller MUST pre-flip:

| Matrix source | diagonal sign | Action before AMGx |
| --- | --- | --- |
| OpenFOAM `pd` (lduMatrix → CSR via cuSPARSE path) | negative (e.g. diag_mean ≈ −1.4e-14) | `A = -A; b = -b;` |
| OpenFOAM `T` equation | positive | none |
| Generic SPD (Laplacian, mass matrix) | positive | none |

Skipping the flip on pd matrices causes AMGx PCG to return
`AMGX_RC_NOT_IMPLEMENTED` or to silently diverge. The flip is mathematically
a no-op (it changes the sign of both sides of A x = b so x is invariant),
but the bits MUST be flipped — AMGx does not auto-detect.

For non-symmetric matrices (such as T with `fvm::div(rhoCpPhi, T)` upwinding,
where `upper != lower`), use a BICGSTAB-flavored config
(`CLASSICAL_V_DIAGSCALED_BICGSTAB` or `CLASSICAL_GS_BICGSTAB`).

## Step 7: Reproduction Test

Save the following as `dilu/amgx/tests/test_reproduce.py` and run:

```python
"""Reproduction smoke test for the AMGx FFI module.

Asserts that on a small 1-D Laplacian, AMGx converges within the expected
iteration budget and to the expected residual. A reproducer matches this
spec iff this test passes verbatim.
"""
import numpy as np
import scipy.sparse as sp
from jax import config as _jc
_jc.update("jax_enable_x64", True)
import jax.numpy as jnp

from dilu.amgx.python import Plan, MINI_AMG_TEST

N = 8
# 1-D Laplacian with Dirichlet boundaries: tridiag(-1, 2, -1)
diag = 2.0 * np.ones(N)
off = -1.0 * np.ones(N - 1)
A = sp.diags([off, diag, off], offsets=[-1, 0, 1], format="csr").astype(np.float64)
b = np.ones(N, dtype=np.float64)

row_ptr = jnp.asarray(A.indptr.astype(np.int32))
col_idx = jnp.asarray(A.indices.astype(np.int32))
values  = jnp.asarray(A.data.astype(np.float64))

with Plan(row_ptr, col_idx, values, MINI_AMG_TEST) as plan:
    x_jax, iters, status = plan.solve(jnp.asarray(b), jnp.zeros(N))
    x_jax.block_until_ready()
    x = np.asarray(x_jax)
    n_it = int(iters[0])
    st = int(status[0])

r = b - A @ x
rel = np.linalg.norm(r) / np.linalg.norm(b)
print(f"status={st} iters={n_it} rel_resid={rel:.3e}")

assert st == 0, f"AMGx status != SUCCESS: {st}"
assert 3 <= n_it <= 12, f"iter count {n_it} outside [3, 12]"
assert rel < 1e-9, f"rel_resid {rel:.3e} > 1e-9"
print("OK")
```

Run:

```bash
source ~/jax-env/bin/activate
cd /path/to/DILU-Research
PYTHONPATH=. python dilu/amgx/tests/test_reproduce.py
```

Expected output:

```
status=0 iters=8 rel_resid=<value below 1e-9>
OK
```

Determinism note: AMGx classical-AMG coarsening (Ruge-Stüben) is
deterministic given fixed input bits and fixed CUDA stream ordering. PCG is
deterministic given a fixed preconditioner application. On the same GPU SKU
and same CUDA toolkit version, iter count and residual are bit-reproducible.
Across different GPU SKUs or different driver versions, iter count is
expected to match exactly but the residual may differ in the last 1-2 bits
due to non-IEEE-determinism in CUDA-cuBLAS reductions (a known cuBLAS
property, not AMGx's fault). If a reproducer sees iter count drift by ≥ 1,
that is a real divergence and indicates a config-string mismatch.

## Step 8: Empirical Reference Numbers

These numbers come from `dilu/amgx/bench/`. A reproducer can re-verify on
the same matrix suite (saved as `.npz` files in `dilu/amgx/bench/`):

### Matrix family: 500K-cell LPBF `pd` (pressure-density Poisson)
- Config: `CLASSICAL_V_DIAGSCALED` with `with_tolerance(..., 1e-8)`
- Sign flip applied (pd has negative diag)
- Cold start (x0 = 0): iters 290-757 on `single_evap_late` 50-matrix sweep
- Fresh-setup, no IR: rel_resid ≤ 4.83e-9 (median ~1.7e-9)
- Config: `CLASSICAL_V_DIAGSCALED` with `with_tolerance(..., 1e-12)` + 1 IR step
- Amortized iters ≈ 8 with `update_coefficients` reuse across timesteps
- rel_resid max 2.14e-15 (matches CHOLMOD LU to 1.13e-11 relative)

### Matrix family: 500K-cell LPBF `T` (heat equation, non-symmetric)
- Config: `CLASSICAL_V_DIAGSCALED_BICGSTAB` with `with_tolerance(..., 1e-12)` + 1 IR step
- No sign flip (T has positive diag)
- Iters: 2-4 (per timestep)
- rel_resid: ~1.34e-16 (machine epsilon for float64 in L2 norm)

### First-call warmup
- First `amgx_solve` after `register_once()` incurs ~700 ms of JAX FFI
  dispatch warmup. Always discard the first measurement when timing.
- AMGx setup cost (one-shot hierarchy build) on 500K cells: ~1.5 s.

## Appendix A: Public API symbol table

Exported by `dilu.amgx.python` (see `__init__.py`):

| Symbol | Kind | Signature / value |
| --- | --- | --- |
| `amgx_setup` | function | `(row_ptr, col_idx, values, config_json: str) -> uint64[1]` |
| `amgx_update_coefficients` | function | `(token, values) -> int32[1]` |
| `amgx_solve` | function | `(token, b, x0) -> (float64[n], int32[1], int32[1])` |
| `amgx_release` | function | `(token) -> int32[1]` (1=removed, 0=unknown) |
| `Plan` | class | `Plan(row_ptr, col_idx, values, config_json)` — context manager |
| `Plan.solve` | method | `(b, x0=None) -> (x, iters, status)` |
| `Plan.update_coefficients` | method | `(new_values) -> status` |
| `Plan.release` | method | `() -> None` (idempotent) |
| `Plan.token` / `.row_ptr` / `.col_idx` | property | read-only |
| `CLASSICAL_V_CYCLE` | str | config JSON (Step 5) |
| `AGGRESSIVE_COARSENING` | str | config JSON (Step 5) |
| `CLASSICAL_V_DIAGSCALED` | str | config JSON (Step 5) |
| `CLASSICAL_V_DIAGSCALED_TIGHT` | str | config JSON (Step 5) |
| `CLASSICAL_V_DIAGSCALED_BICGSTAB` | str | config JSON (Step 5) |
| `CLASSICAL_GS_PCG` | str | config JSON (Step 5) |
| `CLASSICAL_GS_BICGSTAB` | str | config JSON (Step 5) |
| `AGGREGATION_PCG` | str | config JSON (Step 5) |
| `MINI_AMG_TEST` | str | config JSON (Step 5) |
| `with_tolerance` | function | `(base_config, tol, max_iters=200) -> str` |
| `amgx_solve_with_refinement` | function | see Step 4 IR helper |

XLA-FFI custom-call target names (registered by `registration.py`):

| Python target name | C++ handler symbol (in `libdilu_amgx.so`) |
| --- | --- |
| `dilu_amgx_setup` | `AmgxSetup` |
| `dilu_amgx_update_coefficients` | `AmgxUpdateCoefficients` |
| `dilu_amgx_solve` | `AmgxSolve` |
| `dilu_amgx_release` | `AmgxRelease` |

Each is registered under BOTH `"cuda"` AND `"CUDA"` platform strings; at
least one MUST succeed or `register_once()` raises `RuntimeError`. All four
handlers use `api_version=1`.

## Appendix B: AMGx C API symbol table

The following AMGx 2.5.0 C-API functions are called by the shim. The
reference is the `amgx_c.h` header that ships with the install at
`$HOME/local/amgx/include/amgx_c.h`. NVIDIA documentation is at the AMGx
GitHub repo wiki.

| AMGx symbol | Where called | Purpose |
| --- | --- | --- |
| `AMGX_initialize` | `plan_registry.cc` once-flag | Process-global init |
| `AMGX_register_print_callback` | `plan_registry.cc` | Route AMGx stderr through our `dilu_amgx_print_callback` |
| `AMGX_install_signal_handler` | `plan_registry.cc` | Catch SIGSEGV/SIGFPE inside AMGx with traceback |
| `AMGX_config_create` | `plan_registry.cc` (bootstrap `"{}"`), `amgx_setup.cc` (per-plan) | Parse JSON config |
| `AMGX_config_destroy` | `plan_registry.cc` (`destroy_plan_entry`) | Release config — MUST be after `AMGX_solver_destroy` (solver holds raw `AMG_Config*` ptr) |
| `AMGX_resources_create_simple` | `plan_registry.cc` | Singleton resources on device 0 |
| `AMGX_matrix_create` / `AMGX_matrix_upload_all` / `AMGX_matrix_replace_coefficients` / `AMGX_matrix_destroy` | `amgx_setup.cc`, `amgx_update_coefficients.cc`, `plan_registry.cc` | Matrix lifecycle, mode `AMGX_mode_dDDI` (device, double, double, int) |
| `AMGX_vector_create` / `AMGX_vector_upload` / `AMGX_vector_download` / `AMGX_vector_destroy` | `amgx_setup.cc`, `amgx_solve.cc`, `plan_registry.cc` | b/x lifecycle, reused across solves |
| `AMGX_solver_create` / `AMGX_solver_setup` / `AMGX_solver_resetup` / `AMGX_solver_solve` / `AMGX_solver_destroy` | `amgx_setup.cc`, `amgx_update_coefficients.cc`, `amgx_solve.cc`, `plan_registry.cc` | Solver lifecycle |
| `AMGX_solver_get_iterations_number` / `AMGX_solver_get_status` | `amgx_solve.cc` | Post-solve introspection |

`AMGX_finalize` is intentionally NEVER called. At interpreter exit JAX may
have already torn down CUDA context; calling AMGx finalize after that
produces undefined behavior. Per-process leak of AMGx-internal pools is
accepted (typically <1 MB visible at exit). See gotchas #7 and #9.

The `dDDI` mode token decodes as: matrix lives on device (`d`), values are
double (`D`), vectors are double (`D`), indices are int32 (`I`). All four
handlers use this mode and only this mode; block sizes are forced to 1×1.
