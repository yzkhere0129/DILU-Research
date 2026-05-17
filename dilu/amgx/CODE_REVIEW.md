# AMGx FFI Module — Pre-Merge Code Review

**Reviewer:** Claude (code-review agent)
**Date:** 2026-05-17
**Scope:** `dilu/amgx/{python,cpp,CMakeLists.txt,tests}` — JAX FFI binding to AMGx 2.5.0
**Out of scope:** `dilu/amgx/bench/*` (research auxiliaries, per request)

---

## Executive Summary (verdict + reasoning)

**Verdict: must-fix before merge.**

The module is functionally solid — it has shipped 6 months of production research, the lifecycle contract is well thought through, and the test suite covers the obviously important paths (correctness vs DILU baseline, iteration count budget, JIT lowering shape, VRAM ceiling, IR precision floor). The C++ side correctly handles the AMGx config-handle lifetime bug, registers a print callback, installs the AMGx signal handler, and protects the plan registry with a mutex.

However, four classes of defect would embarrass a merge into a collaborator's repository:

1. **A latent use-after-stack-scope bug repeated in all four FFI handlers.** Stack-local scalars (`token`, `status`, `iters`) are passed to `cudaMemcpyAsync` and then go out of scope before the stream is guaranteed to consume them. This works today only because XLA happens to synchronize before the host frame unwinds; under future XLA scheduler changes (or under pipelined CUDA graphs) it will corrupt the output buffer.
2. **A contract inversion in `amgx_release`**: the Python docstring says "status=0 means OK", but C++ returns 1 for the success case and 0 for "unknown token". The Python `Plan.release()` does not look at the return code, so this is silent today — but the senior's code will read it and trip on the inversion.
3. **Import-time side effects in `refinement.py`** (`jax_enable_x64=True`). Importing the public `dilu.amgx` package mutates the user's JAX config. A senior's project that intentionally runs float32 will be silently flipped.
4. **Hardcoded RTX 3050 CUDA arch in both `CMakeLists.txt` and `build.sh`**, no documentation on how to override for the lab's RTX 5060 (Blackwell, sm_120).

Plus a long tail of medium/low issues (config-string duplication, hardcoded test paths, private-attribute access in tests, `Plan.__del__` calling JAX FFI from the GC thread). None of these block the merge functionally; all of them slow down a new contributor.

After the four critical fixes, this is mergeable.

---

## Critical

### C1. Use-after-scope in `cudaMemcpyAsync` of stack-local scalars (all 4 handlers)

**Files / lines:**
- `cpp/amgx_setup.cc:208-211` — `uint64_t token` is stack-local
- `cpp/amgx_solve.cc:138-145` — `int32_t iters_host` and `int32_t status_host` are stack-local
- `cpp/amgx_update_coefficients.cc:94-96, 106-109` — `int32_t status` is stack-local
- `cpp/amgx_release.cc:43-45` — `int32_t status` is stack-local

**Current code (representative, from `amgx_solve.cc`):**
```cpp
  int32_t iters_host  = static_cast<int32_t>(n_iter);
  int32_t status_host = static_cast<int32_t>(solve_status);
  CHECK_CUDA(cudaMemcpyAsync(iters_out->typed_data(),  &iters_host,
                             sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(status_out->typed_data(), &status_host,
                             sizeof(int32_t), cudaMemcpyHostToDevice, stream));
  return ffi::Error::Success();
```

**Why it matters:** `cudaMemcpyAsync(..., H→D, stream)` with a *pageable* host source is documented by NVIDIA as **synchronous** with respect to the host (it falls back to a blocking copy because the driver cannot DMA from non-pinned memory). Today's behavior is therefore correct *by accident*. Three forward-looking risks:
- If a future XLA upgrade ever uses pinned-host registration for these output slots, the copy becomes truly asynchronous and the stack frame can unwind before the DMA fires, writing garbage.
- If anyone later wraps these calls in a CUDA graph (NVIDIA's recommended pattern for low-overhead repeat solves), the captured graph re-issues the copy long after the function returns — guaranteed garbage.
- The pattern is mis-educational: it reads as async but relies on a sync side effect.

**Proposed fix:** either pin the source via `cudaHostAlloc`/`cudaMallocHost` once at process start (a 64-byte pinned arena is fine), or synchronize the stream immediately after the copy:
```cpp
  CHECK_CUDA(cudaMemcpyAsync(iters_out->typed_data(),  &iters_host, ...));
  CHECK_CUDA(cudaMemcpyAsync(status_out->typed_data(), &status_host, ...));
  CHECK_CUDA(cudaStreamSynchronize(stream));  // make the H→D copy ordering explicit
```
The sync is essentially free (microseconds) compared to the AMGx solve.

**Rationale:** correctness-by-accident in CUDA host-pointer copies is the #1 source of "works on my machine, segfaults in CI" bugs. Make the synchronous behavior explicit, or pin the host buffer.

---

### C2. `amgx_release` return-code contract inversion

**File / line:** `cpp/amgx_release.cc:42-45`; doc contract in `python/wrapper.py:82-92` and `python/plan.py:90-94`.

**Current code:**
```cpp
  bool removed = plan_cache_remove(token_host);
  int32_t status = removed ? 1 : 0;
```
Python docstring (`wrapper.py:86`):
> "Idempotent: releasing an unknown token returns 0 rather than erroring."

So Python contract: 0 = OK, non-zero = error. Many AMGx and POSIX conventions agree. But the C++ returns 1 on successful release and 0 on unknown-token. The two layers disagree.

**Why it matters:** today `Plan.release()` only stores the JAX array and blocks on it; nothing reads the numeric value. The senior's project will almost certainly do `if status != 0: raise`. They will get an exception on *every successful release*.

**Proposed fix:** invert the C++ semantics to match the documented Python contract — `0 = OK (released or already gone, idempotent)`, non-zero = real error:
```cpp
  bool removed = plan_cache_remove(token_host);
  // 0 = success (idempotent: not-present is also success). Non-zero would
  // be reserved for a future "actually failed to tear down" case.
  int32_t status = 0;
  (void)removed;
```
Or, if you want to surface "released a stale token" to the user, change the docstring and pick a non-zero sentinel like `2` (and document that 0 means "freed", 2 means "was already absent — possibly a double-release").

**Rationale:** a contract mismatch between C++ and Python is a latent boundary bug. Fix one side; document the other.

---

### C3. Module-import side effect: `refinement.py` flips `jax_enable_x64=True` globally

**File / line:** `python/refinement.py:17-19`
```python
from jax import config as _jc
_jc.update("jax_enable_x64", True)   # AMGx FFI expects float64
import jax.numpy as jnp
```

`python/__init__.py:30` re-exports `amgx_solve_with_refinement`, so the line above runs on `import dilu.amgx`.

**Why it matters:** the senior's project may run mixed-precision workloads or intentionally use float32 for unrelated arrays. Importing `dilu.amgx` silently flips a process-wide JAX flag and there is no way to opt out short of not importing the module. This is the single most surprising thing in the whole package.

**Proposed fix:** remove the import-time mutation. Move it into a function the *caller* opts into:
```python
def enable_x64():
    """Convenience: the AMGx FFI requires float64. Call before constructing
    a Plan if your project hasn't already set jax_enable_x64."""
    from jax import config as _jc
    _jc.update("jax_enable_x64", True)
```
And at the *entry* of `amgx_solve_with_refinement`, check the flag instead of setting it:
```python
from jax import config as _jc
if not _jc.read("jax_enable_x64"):
    raise RuntimeError(
        "dilu.amgx requires jax_enable_x64=True. "
        "Call jax.config.update('jax_enable_x64', True) before importing."
    )
```

**Rationale:** library code must not mutate global interpreter state on import. This is a basic library hygiene rule that the senior will immediately notice.

---

### C4. `CMakeLists.txt` and `build.sh` hardcode RTX 3050 arch; no documented override mechanism

**Files / lines:** `CMakeLists.txt:9-11`, `build.sh:21`.

**Current code (`CMakeLists.txt`):**
```cmake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3050 Laptop; override for server GPUs.
endif()
```
**Current code (`build.sh`):**
```bash
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=86
  ...
)
```
Note the `build.sh` flag *unconditionally* sets `-DCMAKE_CUDA_ARCHITECTURES=86`, defeating the `if(NOT DEFINED ...)` guard in CMake. The CMake comment says "override for server GPUs" but does not say *how*.

**Why it matters:** the lab machine has an RTX 5060 (Blackwell, `sm_120`). Building with `sm_86` produces SASS that the driver must JIT-recompile at first use (PTX → SASS), adding 5-30 s setup latency and burning extra VRAM. Worse, if future AMGx kernels use Blackwell-only intrinsics, the build silently misses them. On a Hopper H100 (`sm_90`) or Blackwell B200 (`sm_100`) the same problem.

**Proposed fix:** make `build.sh` honor an environment variable, and document a multi-arch fat binary as the senior-machine default:
```bash
CUDA_ARCH="${DILU_AMGX_CUDA_ARCH:-86}"   # honor user override
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
  "-DAMGX_ROOT=${AMGX_ROOT}"
)
```
Then add a one-line README block (or top-of-CMakeLists.txt comment) listing common archs: Ampere=86, Ada=89, Hopper=90, Blackwell=120. Recommend `DILU_AMGX_CUDA_ARCH="86;90;120"` for a portable build.

**Rationale:** a research repo can hardcode one arch; a library merged into a collaborator's CI cannot.

---

## High

### H1. `Plan.__del__` issues JAX FFI calls from the GC thread

**File / line:** `python/plan.py:102-106`
```python
def __del__(self):
    try:
        self.release()
    except Exception:
        pass
```
`release()` calls `amgx_release(self._token)`, which is a JAX FFI op. The GC can run in any thread, at any time, including during interpreter shutdown when JAX runtimes may have been torn down. The bare `except Exception` silently eats every failure (including `AttributeError` if `__init__` raised before `_token` was assigned).

**Why it matters:**
- Quietly leaks AMGx resources at interpreter shutdown (per-process leak is the documented policy — fine for `AMGX_initialize`, but plan leaks are *per-Plan*, not per-process).
- Hides bugs: if a user forgets `with`, the leak is silent.
- In multi-thread / async workflows, GC-driven release races with the user's other JAX ops on the same device.

**Proposed fix:** keep `__del__` as a tripwire only, not a real release:
```python
def __del__(self):
    if not self._released:
        import warnings
        warnings.warn(
            f"AMGx Plan leaked: release() was never called. "
            f"Use `with Plan(...) as p:` or call p.release() explicitly. "
            f"Token={self._token}.",
            ResourceWarning, stacklevel=2,
        )
    # Do NOT call self.release() here — JAX may be torn down.
```
And add a module-level docstring note: "Plan does not own a finalizer; you must use the context manager or call `release()`."

Also: guard `__init__` so that `self._released` and `self._token = None` are set *before* the `amgx_setup` call, so a half-constructed object cleans up sanely.

**Rationale:** running JAX ops from `__del__` is documented as unsafe in the JAX FFI docs. The bare `except` makes the unsafety invisible.

---

### H2. `registration.py` silently swallows per-platform registration failures

**File / line:** `python/registration.py:80-94`
```python
def _reg(name, fn):
    capsule = jax.ffi.pycapsule(fn)
    registered_any = False
    for plat in ("cuda", "CUDA"):
        try:
            jax.ffi.register_ffi_target(name, capsule,
                                          platform=plat, api_version=1)
            registered_any = True
        except Exception:
            pass
    if not registered_any:
        raise RuntimeError(...)
```

**Why it matters:**
- A *real* registration error (e.g. capsule type mismatch from a JAX version skew) is swallowed silently as long as one of the two platform names succeeds. The user then gets a confusing "FFI target not found on platform X" error at the first call site, with no trace back to the underlying cause.
- The "register on both 'cuda' and 'CUDA'" workaround is a load-bearing comment with no test asserting that it actually solves the problem on any specific JAX version — if a future JAX rejects both, you find out at first solve.

**Proposed fix:** catch *only* the specific exception class JAX raises for "already registered" / "unknown platform" (likely `ValueError` or `XlaRuntimeError`):
```python
for plat in ("cuda", "CUDA"):
    try:
        jax.ffi.register_ffi_target(name, capsule,
                                      platform=plat, api_version=1)
        registered_any = True
    except ValueError as e:
        # JAX raises ValueError for "already registered" / "unknown platform".
        # Other exception types are real bugs — let them propagate.
        last_err = e
```
And add a unit test that imports `dilu.amgx`, calls `register_once()`, and asserts the four target names are present in JAX's FFI registry. That test will catch JAX-version breakage at CI time.

**Rationale:** bare `except Exception` is the standard antipattern for hiding bugs; the senior will flag it on first review.

---

### H3. AMGx error semantics in `amgx_solve` mix "Python exception" and "status=non-zero"

**File / line:** `cpp/amgx_solve.cc:120` and `cpp/amgx_solve.cc:127`.

**Current behavior:**
- `AMGX_solver_solve` returning `AMGX_RC_NOT_IMPLEMENTED` (the canonical PCG-on-non-SPD failure mode) raises an `ffi::Error` → Python sees `XlaRuntimeError`, not a return-code status. The user has to parse the exception message string to know what happened.
- `AMGX_solver_get_status` returns `AMGX_SOLVE_NOT_CONVERGED` (status=3) → Python sees status=3 in the output, no exception.

So the *same kind* of "solve failed" can manifest as either an exception or a return code depending on which AMGx layer detected it. The Python docstring (`wrapper.py:62-64`) lists the status codes 0/1/2/3 as the contract, but does not mention that AMGx C-API errors bypass the status channel entirely.

**Why it matters:** a user who writes
```python
x, iters, status = plan.solve(b, x0)
if int(status[0]) != 0:
    fallback_path()
```
will be surprised by an uncaught `XlaRuntimeError` when they feed a non-SPD matrix to a PCG config (which is exactly the case the `_BICGSTAB` config exists for). The current `test_f2_acid_32cubed` *expects* status=3 (NOT_CONVERGED) at fixed iters=15, but never tests the "AMGx C-API error" branch.

**Proposed fix:** route AMGx solve errors into the status channel rather than as exceptions, and document the two-tier scheme in `wrapper.py`. Concretely, change `cpp/amgx_solve.cc:120`:
```cpp
  AMGX_RC solve_rc = AMGX_solver_solve(entry->solver, entry->b_vec, entry->x_vec);
  AMGX_SOLVE_STATUS solve_status = AMGX_SOLVE_FAILED;
  if (solve_rc == AMGX_RC_OK) {
    AMGX_solver_get_status(entry->solver, &solve_status);
  }
  // Map AMGx C-API errors into a reserved high status range (e.g. 100+rc)
  // so user code can distinguish "AMG didn't converge" from
  // "AMG library said NOT_IMPLEMENTED on this matrix".
  int32_t status_host = (solve_rc == AMGX_RC_OK)
                          ? static_cast<int32_t>(solve_status)
                          : 100 + static_cast<int32_t>(solve_rc);
```
Document the mapping in `wrapper.py`. Update tests to exercise the "feed asymmetric matrix to PCG" path and assert status >= 100 instead of relying on an exception.

**Rationale:** API ergonomics — the contract `(x, iters, status)` claims to fully describe what happened, but it doesn't. Either fix the implementation to match the contract, or fix the docstring to warn callers.

---

### H4. `_pcg_dilu` re-implements PCG inside a test file and reaches into private state

**File / line:** `tests/test_t8_correctness.py:32-55`
```python
def _pcg_dilu(plan, spmv, b, tol=1e-10, max_iter=1000):
    ...
    d_star = plan.factor(plan._values)  # use seeded values
```

**Why it matters:**
- `plan._values` is a private attribute of the Phase 2 cuSPARSE `Plan`; this test will break the moment anyone refactors that class.
- 24 lines of hand-rolled PCG inside a test means the test is now testing *two* implementations at once — if T8 fails, you don't know whether AMGx or this reference PCG is wrong.
- The test imports `dilu.cusparse.python` (`Plan, build_diag_offset`), creating an undocumented dependency between the AMGx test suite and the Phase 2 Plan API. If `dilu.amgx` is packaged for merge, Phase 2 cusparse needs to come along, or this test is dead on arrival.

**Proposed fix:** use `scipy.sparse.linalg.cg` (or `spsolve`) as the reference. `scipy` is already a hard dep of the repo (see `_harness.py`), and the test then asserts only against a single, well-known oracle:
```python
import scipy.sparse as sp, scipy.sparse.linalg as sla
A = sp.csr_matrix((values, col_idx, row_ptr))
x_ref = sla.spsolve(A, b_h)
```
This also removes the cross-phase coupling.

**Rationale:** tests that depend on sibling-module private attributes break invisibly across refactors and double the failure-mode surface.

---

### H5. Test paths are hardcoded to `/home/yzk/LaserbeamFoam/...`; tests silently skip on any other machine

**File / line:** `tests/test_precision_vs_truth.py:32-44, 144-151`

**Current code:**
```python
CASES = [
    ("pd",
     "/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/dumper_pipeline_test/"
     "postProcessing/matrices/2.64e-12/pd_corr0"),
    ...
]
```
And `pytest.skip(f"dump not present: {p}")` if the path doesn't exist.

**Why it matters:** on the senior's machine, these paths won't exist. Every precision-floor regression test will silently skip — green CI, zero coverage. The "achieved 1e-15 vs truth" claim that justifies the whole IR module would not be CI-verified.

**Proposed fix:** ship a *small* canonical test matrix (e.g. a serialized 32^3 LPBF-like pd matrix, <10 MB) under `tests/data/`, and read paths from an env var or a `tests/data_locations.py` table with a documented override:
```python
def _matrix_path(case_key):
    override = os.environ.get(f"DILU_AMGX_TEST_{case_key.upper()}")
    if override and Path(override).exists():
        return Path(override)
    return Path(__file__).parent / "data" / f"{case_key}.npz"
```
Then have the test ALWAYS find at least one matrix and assert hard, not skip.

**Rationale:** silent skips are the second-worst CI outcome (after silent passes). A merged module's tests must run on the merger's machine without manual data acquisition.

---

## Medium

### M1. Nine near-identical JSON config strings in `config.py`

**File:** `python/config.py:31-405`

The 9 configs differ in 1-5 fields each. Hand-maintaining 9 copies means a future tweak (e.g. "we discovered presweeps=3 is better") requires editing 9 strings and risks divergence (today they already differ in subtle ways — e.g. `CLASSICAL_GS_PCG` has `coarse_solver=DENSE_LU_SOLVER` while `CLASSICAL_V_CYCLE` does not; whether intentional is unclear).

**Proposed fix:** build them programmatically from a base dict:
```python
def _base_pcg(*, scaling=None, smoother="BLOCK_JACOBI",
              interpolator="D2", presweeps=1, postsweeps=1,
              tol=1e-10, max_iters=200, outer="PCG", **extra):
    smoother_block = {
        "BLOCK_JACOBI": {"scope": "jacobi", "solver": "BLOCK_JACOBI",
                          "monitor_residual": 0, "print_solve_stats": 0},
        "MULTICOLOR_GS": {"scope": "smoother", "solver": "MULTICOLOR_GS",
                           "relaxation_factor": 1.0, "symmetric_GS": 1,
                           "monitor_residual": 0, "print_solve_stats": 0},
    }[smoother]
    cfg = {"config_version": 2, "solver": {"solver": outer,
        "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL",
            "smoother": smoother_block,
            "presweeps": presweeps, "postsweeps": postsweeps,
            "interpolator": interpolator, "max_iters": 1, "cycle": "V",
            "max_levels": 50, "scope": "amg",
            "monitor_residual": 0, "store_res_history": 0,
            "print_grid_stats": 0, "print_solve_stats": 0, **extra},
        "max_iters": max_iters, "tolerance": tol,
        "convergence": "RELATIVE_INI_CORE", "norm": "L2",
        "monitor_residual": 1, "print_solve_stats": 0,
        "obtain_timings": 0, "store_res_history": 0, "scope": "main"}}
    if scaling:
        cfg["solver"]["scaling"] = scaling
    return json.dumps(cfg)

CLASSICAL_V_CYCLE             = _base_pcg()
CLASSICAL_V_DIAGSCALED        = _base_pcg(scaling="DIAGONAL_SYMMETRIC")
CLASSICAL_V_DIAGSCALED_TIGHT  = _base_pcg(scaling="DIAGONAL_SYMMETRIC",
                                           presweeps=2, postsweeps=2,
                                           tol=1e-14, max_iters=500)
# ...etc
```

**Rationale:** DRY. Also makes diff review of config changes obvious instead of a wall of JSON.

---

### M2. `refinement.py` allocates a Plan per call (no plan-reuse API)

**File:** `python/refinement.py:64`
```python
with Plan(rp, ci, vv, cfg_primary) as plan:
```
Every call to `amgx_solve_with_refinement` builds a fresh AMGx hierarchy. For per-timestep usage (the typical LPBF setting where matrix structure is reused), this defeats the whole point of `update_coefficients`.

**Proposed fix:** accept an optional `plan: Plan | None = None` parameter; if provided, use it and skip setup. Document that the caller is responsible for `update_coefficients` between calls if the matrix changes.

**Rationale:** the IR path is the *production* path per the precision tests. Production users will care about repeat-call cost.

---

### M3. Tests `test_t9` / `test_t10` / `test_t13` rely on `nvidia-smi` text parsing

**Files:** `tests/test_t9_iter_count_128.py:22-32`, `tests/test_t10_wall_time_128.py` (similar), `tests/test_t13_vram_budget.py:20-24`

Calling `nvidia-smi` via subprocess for memory accounting is fragile:
- Fails on machines without `nvidia-smi` in `$PATH` (some containerized envs).
- Output format has changed across driver versions.
- Reports global VRAM, not per-process — race with other GPU users.

**Proposed fix:** use `pynvml` (already installed alongside any CUDA Python env) or `cudaMemGetInfo` via cupy/torch. Or, more conservatively, gate the VRAM-assertion logic behind a `if nvidia_smi_available()` check that the test can skip when the tool is missing.

**Rationale:** the merge target may not be the same Linux + driver as the dev/lab boxes. A test that crashes with `FileNotFoundError: nvidia-smi` is noise.

---

### M4. Stream-handling in `amgx_setup.cc`: pre-sync without explicit post-sync

**File / line:** `cpp/amgx_setup.cc:123` and `:208-211`.

The handler `cudaStreamSynchronize(stream)` before AMGx upload (good), but the final `cudaMemcpyAsync` of the token to `token_out` (line 209) is *not* followed by a sync. The next FFI call (`amgx_solve`) will sync the stream when it reads the token, so the data races are masked end-to-end. But if a user reads `plan.token` in Python without going through `solve`, they get a stream-pending value.

**Why it matters:** the `Plan` constructor (`plan.py:53`) does `self._token.block_until_ready()`, which materializes the value. But `Plan.token` is a `@property` returning the JAX array — anyone reading `plan.token[0]` outside `solve` triggers a synchronization that wasn't documented.

**Proposed fix:** sync the stream at the end of `AmgxSetupImpl` (it's already a slow op; the extra ms doesn't matter). Or document that "the token output is stream-pending; use `block_until_ready` or pass through another FFI op".

**Rationale:** consistent stream ordering across handlers.

---

### M5. `AMGX_install_signal_handler()` may collide with JAX's own SIGSEGV handler

**File / line:** `cpp/plan_registry.cc:62`

`AMGX_install_signal_handler()` registers an AMGx-private handler for SIGSEGV/SIGFPE/SIGINT. JAX, XLA, and CUDA each install their own crash handlers. The last installer wins. On a real segfault, the user gets AMGx's traceback instead of XLA's — which may obscure a JAX-side bug.

**Proposed fix:** make the signal handler optional, gated by env var:
```cpp
if (std::getenv("DILU_AMGX_INSTALL_SIGNAL_HANDLER")) {
  AMGX_install_signal_handler();
}
```
Default off; document the env var for users who want AMGx's tracer.

**Rationale:** a library should not silently steal process-wide signal handlers from its host.

---

### M6. `test_t11_under_jit.py` brittle HLO grep

**File / line:** `tests/test_t11_under_jit.py:57-75`

The assertion `n_custom_call == 1` and `n_copy_start == 0` parses HLO text. JAX has changed HLO text format multiple times across 0.4 → 0.9; a future version might emit `custom-call.1` vs `custom-call`, or use `host-compute` for the token read.

**Proposed fix:** at least catch the failure mode where JAX's HLO format changes — assert `>= 1` custom-call, not `== 1`, and assert `<= 0` copy-start with a clear message pointing to the JAX upgrade path. Better: walk the `jax.jit(...).lower(...).compile().runtime_executable()` via the structured API rather than text grep.

**Rationale:** text-grep tests on compiler IR are CI-fragile.

---

### M7. `xeon` / 5060 / driver assumptions baked into harness env defaults

**File / line:** `tests/conftest.py:9-11`
```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
```
These are sensible defaults for *this* 4 GB dev box, but the senior's env may want different settings. Using `os.environ.setdefault` is the right pattern (no override of pre-existing values), so this is mild — but a comment explaining the rationale would help.

**Proposed fix:** add a comment block citing the 4 GB / Phase 2 VRAM-rail history.

**Rationale:** cosmetic, but the merger needs to know why these are set.

---

### M8. `Plan.solve` does not return iters/status as host-side scalars

**File / line:** `python/plan.py:76-88`

Every solve returns `(x: jax.Array, iters: jax.Array[1], status: jax.Array[1])`. To use them in a Python `if`, the user must call `int(iters[0])`, which triggers a D→H sync. This is fine inside `with Plan` but means *every* convergence-check loop pays a sync — defeating the JIT.

**Proposed fix:** offer a thin wrapper `Plan.solve_blocking(b, x0)` returning `(x, int, int)`. The plain `Plan.solve` stays JIT-friendly.

**Rationale:** ergonomics: today's API silently forces a sync on every status-check.

---

## Low

### L1. Public API has no type hints

**Files:** `python/wrapper.py:24, 42, 58, 82`, `python/plan.py:40, 68, 76, 90`.

`amgx_setup(row_ptr, col_idx, values, config_json: str)` has only `config_json` typed. The array types are unspecified. Modern Python libraries annotate with `jax.Array | np.ndarray | ArrayLike`. Helps IDE completion and mypy.

---

### L2. Inconsistent docstring style

`wrapper.py` and `plan.py` use one-line summaries; `config.py` uses long prose with `**bold**` markdown; `refinement.py` is mixed. Pick a convention (PEP 257 / numpydoc) and apply uniformly.

---

### L3. `__init__.py` re-exports `MINI_AMG_TEST` (test-only fixture) as public API

**File / line:** `python/__init__.py:27, 46`

`MINI_AMG_TEST` is documented in `config.py:370` as "small-problem smoke test". It probably shouldn't be in the public `__all__`.

**Proposed fix:** drop from `__all__`; keep importable from `dilu.amgx.python.config` for tests.

---

### L4. `_locate_library()` calls itself twice on first `register_once()`

**File / line:** `python/registration.py:66` and `:103`

`register_once()` calls `_locate_library()` once; `library_path()` calls it again. Cache the result.

---

### L5. `plan_registry.cc:g_init_status = AMGX_RC_UNKNOWN` as initial value

**File / line:** `cpp/plan_registry.cc:38`

If `AMGX_initialize` throws / aborts (it shouldn't, but…), `g_init_status` remains `AMGX_RC_UNKNOWN`. Callers see that as a failure, but the registry is in a partially-initialized state. Consider explicit `AMGX_RC_INTERNAL` to distinguish "never tried" from "tried and failed unknown".

---

### L6. Magic number `1e-300` divisor guard scattered across files

`wrapper.py` doesn't have it, but `refinement.py:57`, `tests/test_smoke.py:44`, `tests/test_t8_correctness.py:96`, `tests/test_t12_physical_64.py` all use `1e-300` as a small-denominator guard. Centralize as `SAFE_DENOM = np.finfo(np.float64).tiny` (or just import once).

---

### L7. `Plan.row_ptr` / `Plan.col_idx` exposed as `@property` but `_values` is not

**File / line:** `python/plan.py:60-66`

If the user wants to read back the current values (for diagnostics), they must reach into `self._values`. Add a `@property values` for symmetry.

---

### L8. `refinement.py` `total_s` measurement starts before Plan setup

**File / line:** `python/refinement.py:63-105`

`t0 = time.time()` is set before `with Plan(...)` (which builds the AMG hierarchy — the dominant cost). `total_s` therefore includes setup. This is *probably* what users want, but the docstring should say so explicitly. Today it just says "total_s".

---

### L9. `_BUILD_DIR` resolution assumes the package is unzipped, not installed via pip

**File / line:** `python/registration.py:35`

`_BUILD_DIR = os.path.normpath(os.path.join(_PACKAGE_DIR, "..", "build"))` works when the package is checked out from git but breaks when installed as a wheel. If the merge intends pip-installable packaging, this needs to become `pkg_resources.resource_filename` or use `importlib.resources`.

---

### L10. `test_smoke.py` is missing from any obvious "run-first" convention

There's no `tests/README.md` or `tests/__init__.py` documenting "run `test_smoke.py` first to verify your build". Per the user's question: "Is there a smoke test that any new contributor can run to verify their build?" — yes, but it's not advertised.

**Proposed fix:** add a 10-line `tests/README.md`:
```
First run: `pytest tests/test_smoke.py -v -s`
If that passes, your build + JAX + AMGx + driver stack is OK.
Then: `pytest tests/test_t8_correctness.py` for the correctness baseline.
For the precision floor: `pytest tests/test_precision_vs_truth.py`
(requires OpenFOAM matrix dumps — see test file for path overrides).
```

---

## Positive Observations

- **Lifecycle correctness:** the `PlanEntry` design properly handles AMGx's surprise lifetime rule (config handle held by solver as raw pointer — explicit BUGFIX comment at `plan_registry.h:48-53` documents this beautifully).
- **Plan-cache mutex discipline:** `plan_cache_remove` correctly drops the lock before destroying the entry (`plan_registry.cc:117-128`). That avoids holding a heavily-contended lock across a slow CUDA teardown.
- **Token strong-ref strategy:** `Plan.__init__` holds `_row_ptr`, `_col_idx` to prevent JAX's allocator from recycling them (`plan.py:30-47`). Defensive, well-commented, matches the predecessor's cusparse pattern.
- **Fingerprint check:** `(N, nnz)` fingerprint on `update_coefficients` and `solve` catches the most common misuse (caller passes a differently-sized buffer) at the FFI boundary. Cheap, focused.
- **The verbose-mode instrumentation** (`DILU_AMGX_VERBOSE`) is exemplary debugging UX — `cudaPointerGetAttributes` per buffer plus phase-by-phase log. Keep it.
- **Bootstrap config decoupled from per-plan config** (`plan_registry.cc:80-95`) — good separation of "AMGx wants any config to build resources" from "the user's config drives the actual solve".
- **The print callback** (`plan_registry.cc:26-29`) is the single most important debugging affordance — without it AMGx errors silently disappear. Keep.
- **IR module's structure** (primary solve + N delta-solves on the same Plan) is mathematically correct Higham §12.1 and the test (`test_precision_vs_truth.py`) demonstrates the 1e-15 floor.
- **Config fingerprint via fingerprint struct** (`PatternFingerprint`) is a clean abstraction; equality operator is correctly defined.
- **`block_until_ready()` in `Plan.__init__`** (`plan.py:53`) is the right call — surfaces setup errors at construction site, not at first solve.
- **Tests cover the right axes**: smoke (T-smoke), correctness vs reference (T8), iter-count budget (T9), wall-time budget (T10), JIT lowering (T11), physical (T12), VRAM ceiling (T13), update-coefficients refresh (test_update_coefficients), the F2 interface-halo acid test, and the precision-vs-truth regression. That is genuinely good coverage for a research module.

---

## Overall Assessment

**Requires revision (must-fix before merge), but the foundation is sound.**

Fix in order:
1. **C1** (use-after-scope) — 30 minutes, applies to 4 files, mechanical.
2. **C2** (release status inversion) — 5 minutes; pick a convention and document.
3. **C3** (refinement.py side effect) — 10 minutes; move flag into a function.
4. **C4** (CUDA arch) — 15 minutes; env var + comment.
5. **H1** (Plan.__del__ → warning) — 20 minutes; less code, safer behavior.
6. **H2** (registration silently swallows) — 30 minutes; narrow the except, add registry-presence test.
7. **H3** (error semantics) — 1 hour; decide whether to surface AMGx C-errors via status or exception; update docs and add a test for the failure path.
8. **H4** (test_t8 uses cusparse private state) — 30 minutes; swap to scipy reference.
9. **H5** (hardcoded test paths) — 1-2 hours; bundle a small canonical matrix.

Total triage time: roughly half a day. After that, the module is in the shape of a library a senior collaborator can actually depend on.

The Medium and Low items can ship as a follow-up commit or as TODOs in an issue tracker.
