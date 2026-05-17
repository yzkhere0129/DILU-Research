# AMGx FFI Module — Known Gotchas (Blood-and-Tears Manual)

## How to use this document

This is the companion to `REPRODUCE.md`. `REPRODUCE.md` tells you *what to
build*; this file tells you *what will go wrong while you build it*. Six
months of debugging condensed into 15 traps. Treat it as a **checklist, not
reference reading.**

Recommended workflow:

1. Open this file in a second pane. Read each gotcha's one-line title now;
   you do not need to understand them yet — you need to recognize *symptoms*
   when they appear.
2. Gotchas are ordered by reproduction stage (Build → Import → Load →
   Setup/Solve → IR → Benchmark → Cross-HW). Walk top-to-bottom alongside
   `REPRODUCE.md`.
3. When something breaks: search for the *symptom string* (e.g.
   `NOT_IMPLEMENTED`, `cudaErrorInvalidValue`, `NOT_FOUND on platform`).
4. If no G1–G15 matches, see "Gotchas we have NOT seen but you might" —
   *speculative* but lists the next-most-likely traps from architecture
   transitions we have not exercised.

Each entry uses a consistent template: **Symptom → Why it happens →
Why today vs tomorrow → Detection → Fix → Cross-references.**

---

## G2. CUDA major-version mismatch between AMGx build-time and JAX runtime

**Symptom.** Build and `import dilu.amgx` succeed. The first `Plan(...)`
raises at the H→D copy of `row_ptr`:

```
CUDA error: invalid argument (cudaErrorInvalidValue, code=1)
  in amgx_setup.cc, cudaMemcpyAsync(rp_device, ...)
```

Matrix is correct; pointer is non-null; `nvidia-smi` works. The error
message is misleading — two CUDA runtimes are loaded into one process.

**Why it happens.** Three CUDA runtimes can coexist: system
`/usr/local/cuda-XX.Y`, the bundled `pip install "jax[cuda12]"` runtime,
and the one in `LD_LIBRARY_PATH` when AMGx's `cmake` ran. If AMGx links
against 12.4 but JAX loads 13.0, the `cudaStream_t` opaque handle JAX
passes through the FFI boundary does not match AMGx's loaded runtime.

**Why today / tomorrow.** Bites every OS or JAX-wheel upgrade. No
compile-time check.

**Detection.**

```bash
python -c "import jax; jax.devices()" &
pid=$!; sleep 1
cat /proc/$pid/maps | grep -E "libcudart|libamgxsh" | awk '{print $NF}' | sort -u
kill $pid
```

You want exactly one `libcudart.so.MAJOR.MINOR` line.

**Fix.** Build AMGx in the same environment that runs JAX:

```bash
source ~/jax-pip-venv/bin/activate
export CUDA_HOME=$(python -c "import nvidia.cuda_runtime, os; \
print(os.path.dirname(nvidia.cuda_runtime.__file__))")
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
cd ~/src/amgx-2.5.0/build && rm -rf * && cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ...
```

Or install JAX for the CUDA AMGx was built against:
`pip install "jax[cuda12_local]"` uses system CUDA.

**Cross-references.** REPRODUCE.md §System Requirements; G3.

---

## G3. conda-forge JAX 0.9.0 has a broken FFI dispatch on Blackwell

**Symptom.** Build and import succeed. `register_once()` reports four
targets registered. First `Plan(...)` raises:

```
XlaRuntimeError: NOT_FOUND: No FFI target found with name
'amgx_setup' on platform CUDA (canonical cuda)
```

The parenthesised `(canonical cuda)` is the smoking gun.

**Why it happens.** Conda-forge `jax 0.9.0` canonicalises the platform
name from historical `"CUDA"` to lower-case `"cuda"`. `registration.py`
registers under both names — but conda's dispatch table only checks under
one. Separate Blackwell bug: conda CUDA wheels do not ship `sm_120` PTX,
so kernels would JIT-fail even if registration worked.

**Why today / tomorrow.** PyPI `jax[cuda12]==0.9.0` does not have this bug;
conda-forge does. A future conda-forge rebuild may fix it; do not bet on
it.

**Detection.**

```bash
python -c "import jax; print(jax.__file__)" | grep -q conda && \
    echo "DANGER: conda JAX detected — see KNOWN_GOTCHAS.md G3"
```

**Fix.** Abandon conda. Use a pip venv:

```bash
python3.12 -m venv ~/jax-pip-venv
source ~/jax-pip-venv/bin/activate
pip install -U pip
pip install "jax[cuda12]==0.9.0"
```

See `memory/python_env.md` (the older `~/jax-env` reference; `~/jax-pip-venv`
is its conda-free replacement).

**Cross-references.** `memory/python_env.md`; REPRODUCE.md §Step 2 warning;
G2.

---

## G4. Hard-coded `CUDA_ARCHITECTURES=86` silently breaks on H100 / 5060 / B200

**Symptom.** Build, import, and solve succeed. The solve returns a number
— sometimes correct, sometimes garbage (non-deterministic across runs on
the same input). No exception, no warning. On lucky days first solve is
5–30 s slower than expected (PTX→SASS JIT) and you blame setup overhead.

**Why it happens.** `CMakeLists.txt:9-11` and `build.sh:21` both hard-code
`-DCMAKE_CUDA_ARCHITECTURES=86` (Ampere). On Hopper sm_90 / Blackwell
sm_100 / sm_120, the driver must JIT PTX→SASS at first launch (5–30 s).
Worse: if AMGx uses any arch-specific intrinsic (cooperative-groups
variants, async copy, Hopper TMA), the PTX lacks it and JIT silently
produces wrong-but-non-erroring SASS. `build.sh` unconditionally re-sets
the value, defeating CMake's `if(NOT DEFINED ...)`.

**Why today / tomorrow.** RTX 3050 sm_86 dev box = no symptom. Every other
GPU = potential silent corruption.

**Detection.**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
cuobjdump --dump-sass ~/local/amgx/lib/libamgxsh.so | grep -c "sm_120"
# Expect > 0 on Blackwell; 0 means only PTX fallback.
```

**Fix.** Honour an env var in `build.sh`; bake a multi-arch fat binary:

```bash
# build.sh:
CUDA_ARCH="${DILU_AMGX_CUDA_ARCH:-86;90;120}"   # Ampere + Hopper + Blackwell
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
  "-DAMGX_ROOT=${AMGX_ROOT:-$HOME/local/amgx}"
)
```

Rebuild AMGx itself with the matching `CMAKE_CUDA_ARCHITECTURES` list —
the FFI shim's arch must match what `libamgxsh.so` was built for.

**Cross-references.** CODE_REVIEW.md §C4; REPRODUCE.md §Step 1.

---

## G6. Importing `dilu.amgx` flips `jax_enable_x64=True` process-wide

**Symptom.** A float32-intentional project suddenly produces float64
arrays. Memory doubles. JIT cache misses on every previously-cached
function. `jnp.asarray(x, dtype=jnp.float32)` still returns float32 — but
internal-promotion paths (e.g. `jnp.linalg.norm`) now go through float64.

**Why it happens.** `dilu/amgx/python/refinement.py` (top of file):

```python
from jax import config as _jc
_jc.update("jax_enable_x64", True)   # AMGx FFI expects float64
```

`python/__init__.py` re-exports `amgx_solve_with_refinement` from
`refinement`, so the mutation fires on `import dilu.amgx`, *before the user
gets a chance to opt out.* This is a process-wide flag, not module-scoped.

**Why today / tomorrow.** AMGx handlers require float64, so the mutation
is *correct* for AMGx callers and *wrong* as a library-level side effect.
As long as your code only uses `dilu.amgx`, you do not notice. The moment
a second library in the same process expects float32, you have a silent
data-type war.

**Detection.**

```python
import jax
before = jax.config.read("jax_enable_x64")
import dilu.amgx
assert jax.config.read("jax_enable_x64") == before, "G6: dilu.amgx mutated x64"
```

Add this to CI; today it fails.

**Fix.** Remove the import-time mutation; replace with an opt-in helper
plus a guard at every x64-requiring entry point:

```python
# refinement.py
from jax import config as _jc

def _require_x64():
    if not _jc.read("jax_enable_x64"):
        raise RuntimeError(
            "dilu.amgx requires jax_enable_x64=True. Call "
            "jax.config.update('jax_enable_x64', True) before constructing a Plan."
        )

def amgx_solve_with_refinement(...):
    _require_x64()
    ...
```

**Cross-references.** CODE_REVIEW.md §C3. REPRODUCE.md §Step 2 line "x64
is forced True by refinement.py import" is *describing* this bug, not
endorsing it.

---

## G1. pd matrix with negative diagonal makes AMGx PCG raise NOT_IMPLEMENTED

**Symptom.** A pd matrix dumped from OpenFOAM (`lduMatrix` export) fed
directly into AMGx PCG. AMGx raises:

```
AMGX ERROR: code = 8 (NOT_IMPLEMENTED)
... in AMGX_solver_solve
```

Bubbles up as `jaxlib.xla_extension.XlaRuntimeError`. Matrix has right
shape, right nnz, `b` is non-zero — every sanity check passes, but solve
still fails.

**Why it happens.** OpenFOAM's `lduMatrix` places the *negative* of the
standard FV Poisson operator on the diagonal (OF residual convention `r =
source − A x` with sign-negated assembly). The dumped diagonal is
negative, so the matrix is *negative-definite*, not SPD. AMGx's PCG
hard-fails `NOT_IMPLEMENTED` rather than returning a silent wrong answer.

**Why today / tomorrow.** Permanent — inherent to OF's `lduMatrix` dumps.
Switching to BICGSTAB hides the symptom but gives worse conditioning.

**Detection.**

```python
diag_mean = float(np.mean(A.diagonal()))
if diag_mean < 0:
    print(f"WARNING: negative-diagonal matrix ({diag_mean=}), "
          f"likely OF convention. Flip A and b.")
```

**Fix.** Sign-flip both `A` and `b` (the solution `x` is invariant since
`(-A) x = -b ⇔ A x = b`). Reference: `dilu/amgx/bench/suite/run_benchmark.py`
`load_matrix_npz` (lines 64–80). Note: `x_OF` does **not** need to be
flipped — it already satisfies both signs.

```python
if float(np.mean(A.diagonal())) < 0:
    A = -A
    b = -b
    # x_OF satisfies A x = b regardless of sign — do NOT flip it.
```

**Cross-references.** REPRODUCE.md §Step 6 "Sign Convention";
`bench/suite/run_benchmark.py:64-80`.

---

## G5. `Plan.__del__` calls JAX FFI from the GC thread → intermittent hang/segfault

**Symptom.** A long-running script (e.g. benchmark over 100 matrices)
intermittently hangs at end, or segfaults during interpreter shutdown with
a backtrace mentioning `PyEval_RestoreThread` or
`xla::PjRtClient::~PjRtClient()`. Sometimes a 30-second stall before clean
exit. Interactive `ipython` never reproduces.

**Why it happens.** `python/plan.py:__del__` calls `self.release()`, which
dispatches an AMGx FFI op. Three problems compound: (1) `__del__` runs on
the GC thread, which may not be the main thread; (2) JAX FFI ops require
the JAX runtime to be alive, but at interpreter shutdown module teardown
order is undefined — `dilu.amgx` may be GC'd *after* JAX's CUDA backend
is gone; (3) the bare `except Exception: pass` silently swallows the
error, so you only see a hang (FFI call blocks indefinitely on a dead
stream).

**Why today / tomorrow.** Latent — depends on GC ordering, which changes
between CPython versions. Biggest trigger: forgetting `with Plan(...) as
p:` (i.e. relying on `del plan` or end-of-scope). Today's test suite uses
context managers everywhere; production code that does not will hit this.

**Detection.**

```python
import dilu.amgx as ax, numpy as np, scipy.sparse as sp
A = sp.eye(100, format='csr'); b = np.ones(100)
for _ in range(1000):
    p = ax.Plan(A.indptr, A.indices, A.data, ax.MINI_AMG_TEST)
    p.solve(b, np.zeros(100))   # no `with`, no explicit release
# at process exit: hang or segfault means G5
```

**Fix.** Convert `__del__` into a tripwire warning, never an active
release:

```python
def __del__(self):
    if getattr(self, "_released", True):
        return
    import warnings
    warnings.warn(
        f"AMGx Plan leaked (token={self._token}). Use `with Plan(...) as p:` "
        f"or call p.release() explicitly. Do NOT rely on GC.",
        ResourceWarning, stacklevel=2,
    )
    # Do NOT call self.release() here.
```

Document loudly in the `Plan` docstring: "This object does not own a
finalizer. You **must** use the context manager or call `release()`."

**Cross-references.** CODE_REVIEW.md §H1; REPRODUCE.md §Step 4 `Plan`
section.

---

## G7. `cudaMemcpyAsync` with stack-local scalar source (all four handlers)

**Symptom.** Today: nothing. Tomorrow, after any of: XLA upgrade that pins
host outputs / wrapping the solver in a CUDA graph / switching H→D copies
to a pinned host arena — `iters`, `status`, or `token` output arrays
contain garbage (zeros, stale values from a previous iteration, or
non-deterministic bit patterns). Solve appears to "fail to converge"
(iters=0) or "succeed in 0 iterations" (status=0 and iters=0).

**Why it happens.** All four handlers (`amgx_setup.cc:208-211`,
`amgx_solve.cc:138-145`, `amgx_update_coefficients.cc:94-96`,
`amgx_release.cc:43-45`) do:

```cpp
int32_t status_host = ...;        // stack-local
cudaMemcpyAsync(out->typed_data(), &status_host,
                sizeof(int32_t), cudaMemcpyHostToDevice, stream);
return ffi::Error::Success();     // stack unwinds here
```

`cudaMemcpyAsync(H→D)` with a *pageable* host source falls back to a
synchronous blocking copy (driver cannot DMA from non-pinned memory) —
that is why it works today. The moment the source becomes pinned, the
copy is truly asynchronous, the stack frame unwinds, and the stream
consumes a stale stack address.

**Why today / tomorrow.** Three concrete triggers: (a) XLA upgrade that
pins host buffers for FFI outputs; (b) `jax.jit` with input donation
(more aggressive pre-pinning); (c) CUDA-graph capture (recommended for
per-timestep solves) — the captured copy node remembers the *address*
`&status_host`, long-gone when the graph is replayed.

**Detection.** Force the bug by replacing the async copy with the explicit
synchronous variant in a debug build; if the test suite still passes, you
have not been bitten *yet* but the latent risk remains.

**Fix.** Sync the stream immediately after each `cudaMemcpyAsync` of a
stack scalar:

```cpp
CHECK_CUDA(cudaMemcpyAsync(out->typed_data(), &status_host,
                            sizeof(int32_t), cudaMemcpyHostToDevice, stream));
CHECK_CUDA(cudaStreamSynchronize(stream));   // pin the ordering explicitly
```

Or allocate a small pinned-host arena once at process start. The sync adds
microseconds; the AMGx solve takes milliseconds.

**Cross-references.** CODE_REVIEW.md §C1 (full file:line list).

---

## G8. `amgx_release` return-code contract is inverted between C++ and Python

**Symptom.** User code that defensively checks `status` after `release()`:

```python
status = plan.release()
if int(status[0]) != 0:
    raise RuntimeError(f"release failed: status={status[0]}")
```

…throws on **every successful release**. C++ returns `1` to mean OK; Python
contract says non-zero means error.

**Why it happens.** `cpp/amgx_release.cc:42-45`:

```cpp
bool removed = plan_cache_remove(token_host);
int32_t status = removed ? 1 : 0;       // C++: 1 = removed, 0 = not-present
```

`python/wrapper.py:86` documents the opposite:

> "Idempotent: releasing an unknown token returns 0 rather than erroring."

So Python contract: `0 = OK`, non-zero = error. The two layers disagree.
Today nobody reads the value (test suite ignores it), so it is silent.

**Why today / tomorrow.** First user who follows the documented contract
throws on every successful release.

**Detection.**

```python
import jax.numpy as jnp
from dilu.amgx.python import amgx_release
status = amgx_release(jnp.array([0], dtype=jnp.uint64))
print(int(status[0]))   # known token release returns 1 today; should be 0
```

**Fix.** Invert the C++ side to match the documented Python contract:

```cpp
bool removed = plan_cache_remove(token_host);
int32_t status = 0;     // 0 = success (idempotent: also for not-present)
(void)removed;
```

Or, if you want to surface "released a stale token", pick a distinct
non-zero sentinel and document it: `removed ? 0 : 2`.

**Cross-references.** CODE_REVIEW.md §C2.

---

## G10. `update_coefficients` requires identical sparsity pattern

**Symptom.** Amortized-mode benchmark loop swaps to a new matrix mid-loop.
`plan.update_coefficients(new_values)` returns status=0. `plan.solve(...)`
runs, returns a number. The number is wrong — sometimes silently
(`‖Ax − b‖ ≈ ‖b‖`), sometimes obviously (NaN, inf, 1e20). No exception.

If new nnz differs, the FFI fingerprint check catches it. The trap is
when new `(N, nnz)` are equal but `indptr/indices` differ.

**Why it happens.** AMGx's `update_coefficients` only takes the values
array. The AMG hierarchy (coarsening, interpolation, smoother factorisations)
was built during `setup` against the *original* `indptr/indices`. New
values in the old structure → smoother applies wrong neighbours, coarse
correction interpolates across wrong edges.

**Why today / tomorrow.** For LPBF time-stepping where mesh topology is
fixed across timesteps, `indptr/indices` are stable and `update_coefficients`
is exactly right. The trap fires when:

- Looping over multiple cases / phases with different mesh partitions.
- A remesh / AMR step in a time-stepping code.
- Accidentally passing a `csr_matrix` constructed in a different order
  (scipy's `eliminate_zeros()` can shift indices without changing nnz).

**Detection.** Bytes-level fingerprint check, not just `(N, nnz)`:

```python
self._pattern_hash = hash((rp.tobytes(), ci.tobytes()))
# in update_coefficients:
if new_hash != self._pattern_hash:
    raise ValueError("update_coefficients: sparsity pattern changed. "
                      "Must release() and rebuild Plan.")
```

**Fix.** Either (1) use amortized mode only within a single mesh's
timesteps; release + rebuild when structure changes; or (2) add the
bytes-level check above. The benchmark runner releases per-matrix in
`fresh_*` and reuses within `amortized_*` (same matrix family) — but does
**not** assert pattern equality; a future heterogeneous suite would
silently corrupt.

**Cross-references.** REPRODUCE.md §Step 4 `amgx_update_coefficients`;
`bench/suite/run_benchmark.py` `run_one_protocol`.

---

## G9. Tolerance 1e-8 is not "truth"; you need 1e-12 + 1 iterative refinement

**Symptom.** Report claims "AMGx achieves precision 2.5e-6 vs OpenFOAM on
pd matrices." After eight overnight tuning sessions trying to push AMGx
tighter, the number does not improve. You conclude AMGx is intrinsically
limited.

**Why it happens.** The "truth" was `x_OF`, OpenFOAM's PCG-DIC output at
`tolerance 1e-8` (default in `fvSolution: pd { tolerance 1e-8; }`). At
that tolerance, `x_OF` is itself ~1e-7 to 1e-4 away from the true solution
`x* = A^{-1} b`. The 2.5e-6 you measured is `‖x_AMGx − x_OF‖∞ /
‖x_OF‖∞` — i.e. the *reference noise*, not AMGx's error.

Real AMGx error vs a direct solver (`scipy.sparse.linalg.spsolve`):
`‖x_AMGx − x_true‖∞ / ‖x_true‖∞ ≤ 2.89e-15` on 48 pd matrices, median
1.83e-15 — i.e. machine precision.

**Why today / tomorrow.** Permanent. Comparing two iterative solvers'
outputs measures their *agreement*, not either one's accuracy. Truth must
be a direct solver.

**Detection.** Compute truth once with `scipy.sparse.linalg.spsolve(A, b)`,
then compare *both* AMGx and OpenFOAM against it:

```python
import scipy.sparse.linalg as sla
x_true = sla.spsolve(A, b)
err_amgx = np.linalg.norm(x_amgx - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
err_of   = np.linalg.norm(x_of   - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
print(f"AMGx vs truth: {err_amgx:.2e}")  # ~1e-15 with IR
print(f"OF   vs truth: {err_of:.2e}")    # ~1e-7 at OF tol=1e-8
```

**Fix.** To reach the 1e-15 floor with AMGx, use IR (`n_refine=1` closes
the residual gap from 1e-12 to 1e-15; tolerance alone is not enough):

```python
from dilu.amgx.python import amgx_solve_with_refinement
x, info = amgx_solve_with_refinement(A, b, x0=None,
                                       eq_kind='pd', tol=1e-12, n_refine=1)
```

For T matrices, OF uses tolerance 1e-12 already, so direct `rel_vs_OF ≈
1e-15` is achievable without IR.

**Cross-references.** `memory/amgx_precision_truth.md`; REPRODUCE.md §Step
4 `amgx_solve_with_refinement`; `tests/test_precision_vs_truth.py`.

---

## G11. Lazy-load ASCII I/O pollutes wall-time measurement

**Symptom.** Headline: "AMGx is 5× slower than OpenFOAM PCG-DIC on the
same matrix." Profiler shows 80% of wall time is *not* in `AMGX_solver_solve`
— it is in `numpy.loadtxt` / MatrixMarket parsing.

**Why it happens.** Naive benchmark loop:

```python
t0 = time.time()
A, b = read_matrixmarket("matrix.mtx")   # 250 ms on a 500K-cell matrix
plan = Plan(...)
x, _, _ = plan.solve(b, x0)
total = time.time() - t0
```

`total` is reported as "solve time" but dominated by ASCII parsing. The OF
comparison is unfair: OF solves in-process from its own data structures,
no I/O.

**Why today / tomorrow.** Permanent. Anyone benchmarking with file-system
input must split I/O vs solve.

**Detection.** Split timers and print both:

```python
t0 = time.time(); A, b, _ = load_matrix(...); io_s = time.time() - t0
t0 = time.time(); plan = Plan(...); setup_s = time.time() - t0
t0 = time.time(); x, _, _ = plan.solve(b, x0); x.block_until_ready()
solve_s = time.time() - t0
print(f"io={io_s*1000:.0f}ms setup={setup_s*1000:.0f}ms solve={solve_s*1000:.0f}ms")
```

If `io_s > solve_s`, the headline number is misleading.

**Fix.** `bench/suite/run_benchmark.py` already splits into `setup_s`,
`update_s`, `solve_s`, `ir_s`. When reporting GPU-vs-CPU, quote **`solve_s`
only**, not `total_s`. Fair OF comparison: either include OF's matrix
assembly in the OF total, or exclude AMGx's I/O — the honest move is the
latter.

**Cross-references.** G13 (related format trap); REPRODUCE.md §Step 8.

---

## G12. Warm-start gain is dt-dependent, not a fixed multiplier

**Symptom.** Internal report claims "amortized + warm-start saves 47%
iters / 2× wall-time vs cold-start." New experiment with finer time-step
shows "95% iters / 8× wall-time." Coarser shows "0% gain." Headline
unreproducible.

**Why it happens.** Warm-start gain = PCG convergence from `x0 = x_{t-1}`
(warm) vs `x0 = 0` (cold), a function of `‖x_{t} − x_{t-1}‖/‖x_{t}‖`
which depends on `dt`. Measured: sparse (6 timesteps/track, large
`Δx_{t}`) → iters 290–484, warm-start modest; dense (50 of 384 timesteps,
small `Δx_{t}`) → iters 8 mean, warm-start dominant. No constant
multiplier. The 47% number was specific to one sampling and wrongly
extrapolated.

**Why today / tomorrow.** A *science* trap, not engineering. Permanent. Any
future report quoting a single warm-start multiplier without `dt` is wrong.

**Detection.** Sweep `dt`, plot iters vs `dt`:

```python
for n_steps in [6, 12, 50, 100, 384]:
    matrices = sample_uniform(all_dumps, n_steps)
    iters = run_amortized(matrices)
    print(f"{n_steps=}: mean_iter={np.mean(iters):.1f}")
```

If the curve is monotonic and steep (it is, for LPBF pd), no single
multiplier describes warm-start gain.

**Fix.** Report warm-start as a function: "amortized warm-start at
`dt = X` reduces mean iters from `Y_cold` to `Y_warm`." Never quote a
single ratio. Always include `dt` and the matrix family.

**Cross-references.** `dilu/amgx/bench/suite/plot_results.py` (figures
should be per-matrix-family, not aggregated).

---

## G13. ASCII MatrixMarket vs binary npz I/O differs by ~50×

**Symptom.** Benchmark over 100 matrices takes 4 minutes. Breakdown:
`solve_s` total = 50 s. `io_s` total = 250 s — 5× the solve cost. Switching
from `.mtx` to `.npz` brings wall time to ~55 s.

**Why it happens.** ASCII MatrixMarket: `numpy.loadtxt` on a 1M-line dump
~250 ms + `csr_matrix` construction ~50 ms. Binary `np.load(npz)`: mmap of
three contiguous arrays ~5 ms + `csr_matrix` ~50 ms. At 100-matrix scale:
ASCII = 30 s parsing; binary = 5.5 s; solver itself ~50 s. Under ASCII,
I/O dominates; under binary, it disappears.

**Why today / tomorrow.** Permanent. Any matrix-dump-for-replay should use
a binary format.

**Detection.**

```bash
time python -c "import scipy.io as sio; sio.mmread('matrix.mtx').tocsr()"
time python -c "import numpy as np, scipy.sparse as sp; \
d = np.load('matrix.npz'); \
sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])"
```

If npz is more than 10× faster, you have G13.

**Fix.** Convert all matrix dumps to npz once at suite-construction; never
re-parse ASCII in benchmark loops. Current suite stores in
`matrices_npz/<phase>/ts_<k>/{pd,T}.npz` — keep that convention.

**Cross-references.** `bench/suite/run_benchmark.py` `load_matrix_npz`;
G11.

---

## G14. MPI on Lab Xeon HR54WV2 requires `--oversubscribe`

**Symptom.** `mpirun -np 32 simpleFoam` on lab Xeon:

```
There are not enough slots available in the system to satisfy the 32
slots that were requested by the application
```

`-np 28` works; `-np 29` fails.

**Why it happens.** HR54WV2 = 1 socket × 28 physical × 2 SMT = 56 logical.
OpenMPI's default `slots` = physical cores = 28. Any `-np > 28` is refused
unless told to oversubscribe.

**Why today / tomorrow.** Permanent on this machine.

**Detection.**

```bash
hostname        # if HR54WV2, apply the fix
lscpu | grep -E "Socket|Core|Thread"
```

**Fix.**

```bash
mpirun --oversubscribe --bind-to none -np 32 simpleFoam -parallel
```

Both flags matter: `--oversubscribe` lifts the slot cap; `--bind-to none`
prevents OpenMPI from binding 56 ranks to 28 cores in a way that crashes.

**Cross-references.** `memory/lab_machine_HR54WV2.md`.

---

## G15. Sparse Cholesky on Xeon HR54WV2 is fastest at 32 threads, not 56

**Symptom.** `scipy.sparse.linalg.spsolve` (UMFPACK) or `sksparse.cholmod`
on a 500K-cell LPBF matrix:

- `OMP_NUM_THREADS=56`: 2.8 s.
- `OMP_NUM_THREADS=32`: 1.9 s.
- `OMP_NUM_THREADS=16`: 2.1 s.

"Use all your cores" gives the worst result.

**Why it happens.** HR54WV2 is NUMA. Sparse Cholesky is memory-bandwidth-bound.
Past 32 threads spills across the socket boundary; cross-socket memory
traffic and cache-coherency overhead dominate any FP throughput gain. 32 ≈
"single socket + a few spillover" — empirical sweet spot.

**Why today / tomorrow.** Permanent on this machine. Workload-specific
(sparse, memory-bound); dense BLAS might prefer 56.

**Detection.** Sweep thread count once when establishing the truth-solver
baseline:

```bash
for n in 8 16 24 28 32 40 48 56; do
    echo "=== threads=$n ==="
    OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n \
        python -c "import time, scipy.sparse as sp, scipy.sparse.linalg as sla, numpy as np; \
d = np.load('matrix.npz'); \
A = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape']); \
t = time.time(); sla.spsolve(A, d['b']); print(f'{time.time()-t:.2f}s')"
done
```

**Fix.** For scipy-truth baseline on lab Xeon, set:

```bash
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
```

Do **not** `unset` and let the library auto-detect — auto-detect gives 56
and is measurably slower.

**Cross-references.** `memory/lab_xeon_optimal_threads.md`.

---

## Gotchas we have NOT seen but you might (speculative)

These are *predictions*, not observations. Each is a plausible failure mode
based on an architecture transition we have not exercised. Treat as "look
here first if you see weird behaviour on the listed hardware."

### S1. H100 sm_90 + CLASSICAL_V_DIAGSCALED non-deterministic coarsening *speculative*

AMGx classical coarsening enumerates strong neighbours in an order
dependent on GPU atomic ordering. On Ampere sm_86, our testing shows
bit-identical iteration counts across 100 runs of the same matrix. On
Hopper sm_90, the changed cooperative-groups API may reorder atomics;
first-call iter count may drift ±1 between runs. Test: run the same matrix
10 times, assert `iters[0] == ... == iters[9]`.

### S2. CUDA 13.x stream-graph behaviour change *speculative*

CUDA 13 changed how `cudaMemcpyAsync` interacts with stream-capture mode.
G7's use-after-scope ("works today by accident" on CUDA 12) may fail
outright on CUDA 13 if JAX upgrades to stream graphs internally. Apply the
G7 `cudaStreamSynchronize` fix defensively before any CUDA-13 deployment.

### S3. Blackwell B200 / GB200 unified memory triggers AMGx host-fallback *speculative*

On Blackwell DC GPUs with unified memory, `cudaPointerGetAttributes` in
`amgx_setup.cc` may report `cudaMemoryTypeManaged` for buffers JAX
allocates via the platform allocator. AMGx 2.5.0 was not validated on
managed-memory inputs and may silently fall back to host-DMA. Detect via
`DILU_AMGX_VERBOSE`: `pointerType=2` on row_ptr/col_idx ⇒ expect 10×
slowdown. Mitigation: force copy to device-local buffer before AMGx.

### S4. Multi-GPU `CUDA_VISIBLE_DEVICES` leak *speculative*

Module assumes device 0. If `CUDA_VISIBLE_DEVICES=1` is set and JAX picks
up GPU 1, the AMGx FFI may still hard-code device 0 internally (we have
not audited every `cudaSetDevice` call). Symptom: H→D copy of row_ptr
succeeds (CUDA peer access), solve produces garbage because AMGx hierarchy
is on a different device than the inputs. Audit:
`grep -r "cudaSetDevice\|AMGX_resources_create" cpp/` and confirm all use
`jax::ffi::Ctx`'s device, not hard-coded 0.

### S5. Python 3.14 free-threading mode breaks `Plan` cache mutex *speculative*

Python 3.14 introduces optional free-threaded (no-GIL) builds. The
`plan_registry.cc` mutex is held during C++ work, but Python operations on
`Plan` (reading `plan._row_ptr`) currently rely on the GIL for atomicity.
Under free-threading, concurrent Plan access from two threads may corrupt
the registry. Detect: run the test suite under `python3.14t` with a
multi-threaded harness.

---

## End

Found a 16th gotcha? Add it here. The cost of re-debugging is real; the
cost of documenting is small.
