# Phase 2 — cuSPARSE SpSV FFI Integration Architecture

**Project**: DILU Solver Low-Level Optimization Research (JAX-GPU)
**Scope**: Phase 2 Part 2 — engineering architecture for wrapping cuSPARSE SpSV + DILU factorization into JAX-compatible FFI primitives
**Companion doc**: `phase2_dilu_math_foundation.md` (math expert, parallel track — not yet landed at time of writing)
**Predecessor docs**: `phase1_ffi_prototype_architecture.md`, `phase1_dilu_math_foundation.md`
**Predecessor report**: `phase1_mvp_report.md`
**Date**: 2026-04-20
**Author**: `jax-cfd-expert`
**Hardware target for Phase 2 dev**: RTX 3050 Laptop (4 GB, compute 8.6, CUDA 12.4 toolkit)
**Status**: DESIGN — awaiting user sign-off before any Phase 2 code is written

---

## 0. TL;DR

Phase 1 closed a clean FFI pipeline for a stateless kernel (`y = D⁻¹(b − Ax)`): ~270 µs median dispatch on 3050, HLO-verified zero host-copies, float64 ULP-floor correctness on T1/T2/T3. Phase 2's job is the first **stateful** integration: cuSPARSE SpSV, which has a multi-phase lifecycle (`bufferSize → analysis → solve`) that does not fit a pure-functional primitive naturally.

The tension: JAX's `ffi_call` is a pure op with no "setup" hook; cuSPARSE wants a handle, a descriptor, a plan, and a workspace that all outlive a single call. The architecture below resolves this with a **two-primitive split plus a process-global handle-and-plan cache**, keyed on the sparsity pattern's device pointers. Specifically:

- **`cusparse_dilu_analyze`** — one FFI call per matrix, returns an opaque `uint64` token. Expensive O(nnz) work runs here; it is kept out of the PCG hot loop.
- **`cusparse_dilu_apply`** — the hot-loop FFI call. Takes the token (as a traced `jnp.uint64` scalar), the modified diagonal $D_*$, the CSR triplet, and $r$. Executes two cuSPARSE `SpSV_solve` calls. Analysis is never re-run here.
- **Handle scope** = per-device, per-process; initialized lazily on first call, bound to the XLA stream at each invocation via `cusparseSetStream` (per the cuSPARSE thread-safety rule verified below).
- **Plan (descriptor) scope** = keyed in a C++ static `unordered_map<uint64, PlanEntry>`, lookup via the token; explicit invalidation endpoint provided for pattern changes.

Web-search reconnaissance (§1.4, §2.6) established three facts that shape this design:
1. cuSPARSE **requires** the handle to be private per thread — "set stream affects all threads sharing the handle". This forbids any cross-thread handle sharing, even with locks.
2. The legacy `csrilu02` / `csrsv2` family is **deprecated and scheduled for removal** (cuSPARSE 13.0+); we target the generic `cusparseSpSV_*` API from day 1.
3. NVIDIA does **not document** the internal algorithm of `CUSPARSE_SPSV_ALG_DEFAULT`. The roadmap phrasing "Level Scheduling 并行算法" is plausibly accurate for older releases but is **not guaranteed** for 12.4+. Phase 2 treats this as a black box to characterize empirically (§6), not a specification to depend on.

This doc does not specify any DILU factorization mathematics — that is the math expert's parallel deliverable. It only commits to calling shape/dtype contracts (§4) that the math doc will fill in. Similarly no code is written; this is design for sign-off.

---

## 1. The Stateful ↔ Stateless Tension

### 1.1 What cuSPARSE demands

From the cuSPARSE 12.4 Management API and cuSPARSE 13.2 reference (URLs in §10):

| State object | Creation | Destruction | Scope hint from NVIDIA |
|---|---|---|---|
| `cusparseHandle_t` | `cusparseCreate()` — non-trivial (driver init, workspace, stream binding) | `cusparseDestroy()` | "Handle must be **private per thread**" — forum-confirmed by NVIDIA engineer |
| `cusparseSpMatDescr_t` (CSR) | `cusparseCreateCsr()` — wraps device pointers, dims, index type, base, value type | `cusparseDestroySpMat()` | Lightweight; one per matrix |
| `cusparseDnVecDescr_t` | `cusparseCreateDnVec()` — wraps a dense vector pointer | `cusparseDestroyDnVec()` | Lightweight; per call or cached |
| `cusparseSpSVDescr_t` | `cusparseSpSV_createDescr()` — the "plan" that carries analysis state | `cusparseSpSV_destroyDescr()` | Holds the O(nnz) analysis result — **this is the expensive thing we amortize** |
| External workspace | `cudaMalloc(sz)` where `sz` comes from `cusparseSpSV_bufferSize()` | `cudaFree` | Matrix-dependent size; hundreds of KB to tens of MB in practice |

Additionally, `cusparseSetStream(handle, stream)` must be called **before** each solve to bind the handle's operations to XLA's stream. This is the cheap per-call mechanism that reconciles one-handle-per-thread with JAX's stream model.

### 1.2 What JAX demands

From `phase1_ffi_prototype_architecture.md` §1 (live-probed against JAX 0.9.0):

- `jax.ffi.ffi_call` produces a pure-functional custom-call primitive. No setup/teardown hooks in the JIT graph.
- Inputs are traced tensors (shape+dtype); JAX re-traces the function when those change. Static values must go through `static_argnums` on the *outer* `jax.jit` — they are **not** visible inside the `ffi_call` as traced args.
- The C++ handler is called on every traced execution. There is no persistent user-data slot in the handler contract (XLA FFI v0.2 does expose `UserData<T>` via `Ffi::Bind().Ctx<>`, but the register-time binding is process-global, not per-call-site — see §2).
- The CUDA stream is supplied by XLA through `PlatformStream<cudaStream_t>`. JAX may change streams across calls (multi-stream executors) but within one call the stream is fixed.

### 1.3 The conflict, stated precisely

A naive mapping of `cusparse_spsv_solve` into a single `ffi_call` forces one of two unworkable choices:

- **(a)** Create handle + descriptor + analyze + solve + destroy inside the handler. The analysis is O(nnz); for a 3D 7-pt Laplacian at $n=10^5$ this is ~1–5 ms per call, run hundreds of times per PCG. The Phase 1 ~270 µs dispatch baseline becomes irrelevant — analysis dominates by an order of magnitude.
- **(b)** Cache inside the handler keyed on... what? Device pointers? Those are stable across JIT invocations for the same JAX arrays (XLA allocates once, reuses), but *not* stable across recompilations (shape/dtype change evicts the compiled HLO and may reallocate). We also cannot hash the **values** buffer — even when pattern is stable, values change every PCG step.

The canonical pattern in the JAX ecosystem for handling this (verified via search — cuDNN via jaxlib internals, transformer-engine-jax, etc.) is the **two-primitive split**: one op materializes the plan, another consumes it. The plan is passed as an opaque token. That is what this doc recommends (§2, Option C), with a few refinements specific to our cuSPARSE constraints.

### 1.4 Ecosystem prior art (from web search — verified citations in §10)

- **`jax.experimental.sparse`** wraps some cuSPARSE SpMV/SpMM operations in jaxlib internals, but its support is considered **unstable between cuSPARSE releases** (jax-ml/jax#16248 discussion), and jaxlib 0.6.x had load-time issues finding cuSPARSE on some systems (jax-ml/jax#30050, #29843). It does **not** expose SpSV. Not a reuse target; only a cautionary tale about binding to cuSPARSE versions.
- **Transformer Engine JAX** (`docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html`) wraps cuDNN handles as opaque state behind FFI calls. They use nanobind capsules with a process-global handle pool — confirmation that "singleton handle + per-call stream bind" is industry-accepted. They do **not** publish an opaque-token pattern because cuDNN's per-call planning is cheap enough to eat inside the call.
- **jax-triton** solves a different (kernel-compilation) state problem by using `static_argnums` + Python-side caching of the compiled kernel. This pattern is compatible with ours for *shape-level* keying but not for *pointer-level* keying, which is what SpSV analysis needs.

None of these give us a turnkey pattern. We synthesize: singleton handle (from TE), opaque token (from cuDNN's kernel-plan model, generalized), and pointer-based plan cache (our own, forced by SpSV's analysis cost).

---

## 2. Descriptor Lifecycle Strategies — Options and Recommendation

Five candidate patterns, summarized before the detailed write-up:

| Option | Handle | SpSV plan (`SpSVDescr`) | PCG-loop analysis cost | Complexity | Recommendation |
|---|---|---|---|---|---|
| A. Handle-per-call | per-call | per-call | **fatal** (O(nnz) × N_iter) | trivial | **Reject** |
| B. Singleton handle, plan-per-call | process singleton | per-call | **fatal** | low | **Reject** |
| C. Opaque-token analyze+apply split | process singleton | cached, indexed by `uint64` token | **amortized** — 1× | medium | **Recommend** |
| D. Python-side descriptor capsule via `static_argnums` | process singleton | held by a Python `PyCapsule` object | amortized | medium-high (fragile under tracing) | Fallback |
| E. Register custom JAX type | process singleton | proper JAX value | amortized | high, speculative | **Out of scope** |

### 2.1 Option A — Handle-per-call (baseline for comparison only)

Create handle, descriptor, bufferSize, malloc workspace, analyze, solve, free, destroy — all inside one FFI call.

- **Pros**: trivially stateless; no cross-call coupling; every call is a complete transaction.
- **Cons**: **Analysis cost dominates.** `cusparseSpSV_analysis` is O(nnz) and empirically hundreds of microseconds to several milliseconds (depends on nnz and hardware). A PCG with 500 iterations and 2 SpSV calls per iteration runs analysis 1000 times. On a $10^5$-row Laplacian, this is 1–10 seconds of redundant work per solve. The Phase 1 baseline number (270 µs dispatch) becomes moot.
- **Also**: workspace `cudaMalloc` inside the handler **breaks CUDA graph capture** (Phase 1 design doc §1.3 landmine #3), which `jax.jit` may attempt. This alone disqualifies the option.

**Reject.** Keep as the measurement baseline in §6 — we want a number for "what would happen if we didn't amortize".

### 2.2 Option B — Singleton handle, plan-per-call

Handle is process-global (created lazily, reused). Plan (descriptor + workspace) is still per-call.

- **Pros**: saves `cusparseCreate` cost (~10 ms one-time). Cheaper than A in the wall-clock sense.
- **Cons**: still pays full analysis cost per call. Handle init was never the bottleneck. This option addresses the wrong problem.

**Reject.**

### 2.3 Option C — Opaque-token analyze+apply split (RECOMMENDED)

Two FFI primitives:

1. **`cusparse_dilu_analyze`** — takes the CSR triplet and returns an opaque `uint64` token (device-resident or host-scalar; see implementation note below). Internally it:
   1. Takes-or-allocates the singleton cuSPARSE handle for the current device.
   2. Creates `cusparseSpMatDescr_t` for the L part (unit-lower) and U part (non-unit-upper) using the CSR triplet.
   3. Creates two `cusparseSpSVDescr_t` instances (one forward, one backward).
   4. Calls `cusparseSpSV_bufferSize` twice; `cudaMalloc`s the two workspaces.
   5. Calls `cusparseSpSV_analysis` twice on *stand-in vectors* (or on the actual $D_*$ if the math allows — TBD with math expert; analysis is pattern-based so the vector values should not matter).
   6. Stashes `{handle*, L_descr, U_descr, L_spsv, U_spsv, L_buf, U_buf}` in a process-global `unordered_map<uint64, PlanEntry>` keyed by a fresh token integer.
   7. Returns the token as a 0-d `uint64` tensor.

2. **`cusparse_dilu_apply`** — takes the token, $D_*$, CSR values (for recomputing or referencing $A$ in the apply), and $r$. Internally:
   1. Looks up the token in the static map.
   2. `cusparseSetStream(handle, xla_stream)`.
   3. Runs forward SpSV_solve, then a diag scale, then backward SpSV_solve, producing $z$.

3. **`cusparse_dilu_release`** (helper, not in hot path) — takes a token and frees the plan's resources. Called from Python `finally` / `__del__` hooks.

**How the token survives JAX's tracing model**:
- `analyze` returns a 0-d `jnp.uint64` device scalar. JAX traces it through any `jit` boundary normally. The actual integer value is written by the FFI handler into the output buffer.
- `apply` receives the token as a 0-d `jnp.uint64` input, reads its value on the C++ side (one `cudaMemcpy` D→H of 8 bytes — this is the one tolerated host copy; see §2.3.2 below), and looks up the plan.
- Because the token is a regular JAX array, it participates in `jit` caching the same way other inputs do. Same input token → same compiled HLO → same handler invocations.

**Pros**:
- Analysis runs exactly once per `analyze` call. PCG hot loop pays only the solve cost.
- Fits JAX's "data flows through ffi_call" model; no out-of-band Python state.
- Extends cleanly to multi-matrix usage (different tokens, independent plans).
- The map can be size-bounded (LRU with user-controlled cap) to avoid unbounded growth.

**Cons**:
- We tolerate **one** 8-byte D→H copy per apply (the token read). This is visible in HLO as a `copy-start`/`copy-done` pair on a scalar, NOT on any solver state. It is ~1 µs on the 3050 and well below dispatch cost. Phase 1's "zero host copies" rule was about hot solver data; a scalar control word is a narrow, specified exception.
- The token scheme requires discipline on the Python side: a token from one `analyze` must not be fed to an `apply` that saw a different CSR pattern. We protect this via (a) an explicit `Plan` wrapper class in Python whose `.apply(r)` method is the only public entry point, and (b) a C++-side **pattern-fingerprint check** — `analyze` stores `(row_ptr_device_ptr, col_idx_device_ptr, n, nnz)` in the map; `apply` verifies the CSR triplet's pointers are identical (see §3.3 on pattern-change detection).
- Fork-safety: the map is process-local. A `fork()` child inherits the map but not the CUDA context. The map must be explicitly cleared on `atfork` if we ever multi-process JAX. Out of scope for Phase 2 (single-device only per brief), but flagged in §8.

**#### 2.3.1 Why token must be a device scalar, not a host attribute**

An alternative is to make the token a Python `int` fed via `static_argnums`. This is Option D — less clean because:
- The outer `jax.jit` would need the token in its static args, meaning the compiled HLO cache key changes every time we analyze a new matrix. This fragments the JIT cache unnecessarily.
- Functions returning or taking opaque integers don't AD-transform. For Phase 2 we don't need AD, but Phase 3+ might.

A traced uint64 scalar flows cleanly through pytrees, `lax.scan`, `lax.fori_loop`, and any outer control flow the PCG might be wrapped in.

**#### 2.3.2 The tolerated scalar D→H copy — concrete cost**

The token copy happens once per `apply` call, 8 bytes, overlapping with kernel launches. Measured on other systems (not ours yet) at ~1–2 µs. If Phase 2 profiling reveals this is actually blocking (e.g., forces pipeline drain before the SpSV kernels), the mitigation is to cache the token as a C++ local variable across `apply` calls that share the same host-resident JAX array identity — but that requires a handler-internal cache of `(xla_buffer_address → token_value)`, which is its own complexity. We log this as a **conditional follow-up** (§7, step 7) and do not preemptively optimize.

### 2.4 Option D — Python-side descriptor capsule via `static_argnums`

Wrap the `PlanEntry*` in a Python `PyCapsule` and pass it to a `jax.jit`-wrapped `apply` function via `static_argnums`. The capsule's hash is its pointer value; cache hits require the exact same capsule.

- **Pros**: no traced-uint64 scalar; no D→H copy per call.
- **Cons**:
  - `static_argnums` re-hashes on every `jit`-call; if the user creates many plans, the JIT cache fragments.
  - Capsules are non-hashable by default (JAX requires `__hash__` + `__eq__`); we'd write a wrapper class. Manageable but fiddly.
  - The `apply` function cannot then be used inside traced control flow (`lax.scan`, etc.) because `static_argnums` cannot vary across trace iterations.

This is the canonical fallback if Option C's token copy measures badly. We accept it for a single outer-loop PCG where static args are fine, but Option C composes better.

### 2.5 Option E — Custom JAX type / pytree

Register a `DILUPlan` pytree or custom dtype, route it through JAX's type system. Theoretical cleanest fit, but:

- Registering new dtypes requires jaxlib surgery, which violates our platform-compatible-development rule and also breaks JAX-version portability.
- Registering a pytree for an opaque pointer reintroduces all Option D's problems — pytrees are static at trace time.

**Out of scope.** Revisit if the JAX team ever publishes a documented stateful-resource pattern.

### 2.6 Recommendation and its failure modes

**Recommendation: Option C (opaque-token analyze+apply split), with Option D as a named fallback if §2.3.2's copy is measured as blocking.**

Failure modes the design must defend against:

| Failure | Detection | Recovery |
|---|---|---|
| Sparsity pattern changes (values' device ptr same but row_ptr/col_idx different) between `analyze` and `apply` | C++-side pointer fingerprint (`row_ptr_ptr + col_idx_ptr + n + nnz`) stored at `analyze` time, checked at `apply` time. Mismatch → `ffi::Error::InvalidArgument("token-pattern mismatch")` | User calls `release(old_token); new_token = analyze(new_A)` |
| Stream changes across calls (multi-stream JAX) | `cusparseSetStream(handle, xla_stream)` at every `apply`. The per-thread rule (§1.1) means handle is safe under stream changes as long as we do not share the handle across threads | Handled by unconditional `SetStream` per call |
| Fork-based multiprocess JAX | Plan map is inherited but CUDA context is not; any plan lookup will hit an invalid handle | `atfork` child-side handler clears the map. Phase 2 is single-process; add the handler but do not test until Phase 4 |
| Plan map grows unboundedly (e.g., a user creates tokens in a loop and forgets to release) | Size-bounded LRU with env-var-configurable cap (default 64) | `release` eviction on insert-overflow; log a warning |
| Handle ever used from a non-device-owning thread | cuSPARSE rule is "handle per thread", but our singleton is implicitly shared across the process. JAX 0.9 on single-device runs the FFI handler on a single XLA scheduler thread per device, so this is typically fine. | Enforce with a `std::once_flag` guarded by `thread_id` check; fail loudly on multi-thread entry with a diagnostic "handle was created on tid=X, called from tid=Y" (Phase 2 has no multi-device; tripwire only) |

---

## 3. Analysis-Amortization Strategy

### 3.1 What "amortization" means concretely

For a PCG iteration loop like:
```python
plan = cusparse_dilu_analyze(A_csr)       # once
D_star = dilu_factor(A_csr)               # once (or once-per-values-update; see §3.2)
for k in range(max_iter):
    z = cusparse_dilu_apply(plan, A_csr, D_star, r)    # many times
    # ... alpha, beta, inner products, etc.
```

the invariant is:
- `cusparse_dilu_analyze` runs once; its plan lives as long as `plan` is reachable in Python.
- `dilu_factor` runs once per matrix-values change; Phase 2 assumes the pattern is frozen for the whole PCG (reasonable for linear solvers).
- `cusparse_dilu_apply` runs 2×N_iter times; must not touch the handle map beyond a lookup.

### 3.2 Does JAX's JIT caching "just work" for us?

No. Here is why, explicitly:

- `jax.jit(f)` caches the compiled HLO keyed on input shapes/dtypes. If we call `apply(token, A_csr, D_star, r)` inside a jitted PCG, the HLO is built once. Subsequent calls hit the HLO cache.
- **But**: each HLO execution runs our handler afresh. The handler is C++ code invoked by XLA's custom-call dispatcher. There is no "JIT-cached handler state".
- The HLO-level cache is therefore about *Python-to-HLO compilation*, not about *handler-to-plan* caching. We need our own map, per §2.3.

### 3.3 Pattern-change detection — the correctness-critical question

When is it safe to reuse an analysis descriptor? NVIDIA's contract (`cusparseSpSV_analysis` docs — see §10): **analysis encodes the sparsity pattern only**; the values matrix can change between analysis and solve, provided the CSR structure is bit-identical.

Our pattern-fingerprint at analyze time stores:
```
(row_ptr_device_ptr, col_idx_device_ptr, n, nnz, index_base, index_type)
```

At apply time we verify these are identical. If any diverges → reject with `InvalidArgument`. The user must call `analyze` again.

**What this does NOT catch**: a caller holds the same CSR pointers but has overwritten the memory with a different pattern. This is a user-code bug we cannot defend against in general — the pattern fingerprint only detects *different buffers*, not *modified contents of the same buffer*. The documented contract is "don't mutate the CSR pattern buffers between analyze and apply". We surface this in the Python `Plan.apply` docstring.

An alternative (stronger) fingerprint hashes the first N bytes of `row_ptr` and `col_idx`. Cost: one GPU-side reduction kernel per apply. Rejected for Phase 2; may revisit if mis-use bugs show up.

### 3.4 Alternative paths considered and rejected

- **Stash the plan pointer on the CUDA stream object** (e.g., via `cudaStreamAttachMemAsync` or a user-data pointer on the stream). No such API in CUDA. Rejected.
- **Hash the pattern by sampling** (e.g., MurmurHash the first 64 nonzeros). Fragile; false positives if the user deliberately builds two matrices with identical prefixes. Rejected.
- **Re-run analysis lazily on first `apply` if plan is stale**. This would make `apply` sometimes-expensive — a usability trap and a hidden-performance antipattern. Rejected in favor of explicit `analyze` + clean error.

---

## 4. FFI Primitive Design — Concrete Spec

The math foundation doc (parallel track) will confirm exact recurrences. The interfaces below are the engineering contract regardless.

### 4.1 `dilu_factor` — compute $D_*$

**Purpose**: given CSR triplet of $A$, produce the modified diagonal $D_*$ per the DILU recurrence (math doc §1.2):

$$ d_i = a_{ii} - \sum_{k<i,\,(i,k)\in\mathcal S(A)} \frac{a_{ik} a_{ki}}{d_k} $$

**Character**: **serial sweep** along the DAG — for Phase 2, we implement this as a **single custom CUDA kernel** with natural ordering on the CPU side or on device with one-thread-per-row-wave (the wavefront argument from `phase1_dilu_math_foundation.md` §1.3 applies here symmetrically). This is not an SpSV; it is a scalar forward pass where row $i$'s result depends on $d_k$ for $k < i$ that index into $A$'s pattern. **We do not use cuSPARSE for this step** — there is no cuSPARSE primitive for DILU's specific recurrence. Our kernel is hand-written.

For Phase 2 we tolerate a **naive one-level-at-a-time launch** pattern (analogous to level-scheduling) since the factorization runs once per matrix-values update, not in the hot loop. Optimizing `dilu_factor` is a Phase 3 concern; Phase 2's acceptance criterion is correctness + "not embarrassingly slow" (< 10 ms at $n = 10^5$).

**Signature** (FFI):
```
Inputs  (device, C-contiguous):
  row_ptr : int32[n+1]
  col_idx : int32[nnz]
  values  : float64[nnz]
Output:
  d_star  : float64[n]
Stream: XLA-supplied.
Errors: InvalidArg on shape mismatch; Internal on CUDA launch failure.
```

### 4.2 `cusparse_dilu_analyze` — one-shot analysis

**Purpose**: run `cusparseSpSV_analysis` twice (L and U parts), return an opaque token.

**Signature**:
```
Inputs (device, C-contiguous):
  row_ptr : int32[n+1]
  col_idx : int32[nnz]
  values  : float64[nnz]   -- passed to SpSV_analysis; per cuSPARSE docs the
                              pattern is what matters, but the API takes the
                              descriptor which is constructed against values
Output:
  token   : uint64[1]       -- 0-d-ish; JAX-side shape is () after a squeeze
Stream: XLA-supplied.
Errors: InvalidArg on malformed CSR; Internal on any cuSPARSE failure.
Side effect (PROCESS-GLOBAL STATE): inserts (token → PlanEntry) into the static
  unordered_map.
```

**L vs U splitting**: cuSPARSE's SpSV takes the full CSR plus a fill-mode (`UPPER`/`LOWER`) and a diag type (`UNIT`/`NON_UNIT`). For the DILU forward solve $(D_* + L) y = r$:
- Fill mode = LOWER
- Diag = NON_UNIT (we store $D_*$ on the diagonal)

For the backward solve $(D_* + U) z = D_* y$:
- Fill mode = UPPER
- Diag = NON_UNIT

So we build **one** `cusparseSpMatDescr_t` on the full CSR of $A$, then pass two different `(fill_mode, diag_type)` attribute settings through the descriptor's attribute API. **BUT** — and this is a sharp edge — for DILU the on-diagonal values in the CSR are $a_{ii}$, not $d_i$. We need to either (a) overwrite the diagonal slot with $d_i$ before SpSV, or (b) pass a separate descriptor that logically says "use $d_i$ instead". cuSPARSE does not natively support (b). Option (a) requires a mutable view of the CSR values array.

**Decision for Phase 2**: the `dilu_factor` output $D_*$ is stored as a separate `float64[n]` array. Before each SpSV pair, we **scatter-write** $D_*$ onto a working copy of the CSR values at the diagonal positions. Cost: one O(n) kernel per apply. This is the cleanest way to preserve the user's original $A$ and feed cuSPARSE a valid L/U operator. The scatter kernel is trivial; the overhead is bounded and measured in §6.

Alternative considered: hold two separate CSR structures for L-with-$D_*$-diag and U-with-$D_*$-diag, each a copy of $A$'s indices plus a custom values array. Doubles index memory; rejected.

### 4.3 `cusparse_dilu_apply` — the hot path

**Signature**:
```
Inputs (device, C-contiguous):
  token    : uint64[1]      -- from cusparse_dilu_analyze
  row_ptr  : int32[n+1]     -- must match the pattern analyze saw
  col_idx  : int32[nnz]     -- must match the pattern analyze saw
  values   : float64[nnz]   -- A's current values (may have changed from analyze)
  d_star   : float64[n]     -- from dilu_factor on the current values
  r        : float64[n]     -- the vector to precondition
Output:
  z        : float64[n]     -- z = M^{-1} r
Stream: XLA-supplied.
Errors: InvalidArg on token-pattern mismatch; Internal on cuSPARSE failure.
```

**Handler flow**:
1. Read token value (8-byte D→H scalar copy; §2.3.2 caveat).
2. Lookup `PlanEntry` in the static map. Fail if missing.
3. Verify pattern fingerprint (§3.3). Fail if mismatch.
4. `cusparseSetStream(handle, xla_stream)`.
5. Launch scatter kernel: working_values = values with diag positions overwritten by $D_*$.
6. Bind the `cusparseSpMatDescr_t` to `working_values`, configure fill=LOWER, diag=NON_UNIT.
7. `cusparseSpSV_solve(forward)` → produces $y$ in a workspace vector.
8. Launch elementwise kernel: $\tilde y = D_* \odot y$ (the "$D_* y$" RHS of the backward solve). This matches the $M = (D_*+L) D_*^{-1} (D_*+U)$ factorization — the middle $D_*^{-1}$ cancels with one of the diagonals; confirm exact form with the math expert before implementation.
9. Reconfigure SpMat to fill=UPPER, diag=NON_UNIT.
10. `cusparseSpSV_solve(backward)` → produces $z$.
11. Return.

Total: 2 × cuSPARSE SpSV_solve, 1 × scatter kernel, 1 × elementwise scale. Zero `cudaMalloc` in the hot path (workspaces were allocated at analyze time). Zero host syncs beyond the token copy.

### 4.4 `cusparse_dilu_release` — teardown

**Signature**:
```
Input:
  token : uint64[1]
Output:
  (none — returns int32[1] status for JAX to thread as a dummy dep)
```

**Flow**: map lookup, destroy all descriptors (`cusparseSpSV_destroyDescr` × 2, `cusparseDestroySpMat`, `cusparseDestroyDnVec` if any cached), `cudaFree` workspaces, remove from map. Idempotent — releasing a gone token is a no-op warning, not an error (robust to double-`release`).

### 4.5 (Optional) `dilu_refactor` — values-only update

**When useful**: A's values change (e.g., solver re-enters after a physics update) but pattern doesn't. Then we want to recompute $D_*$ without re-running `cusparse_dilu_analyze`.

**Answer**: This is already what we have. `dilu_factor` is pattern-independent in cuSPARSE terms — it's a standalone kernel. `cusparse_dilu_analyze` is the pattern-dependent step and is not repeated. "Refactor" is just "call `dilu_factor` again on new values; reuse the existing token."

So `dilu_refactor` is **not a new FFI primitive**; it is a Python-level workflow: `D_star_new = dilu_factor(A_csr_new); plan stays valid`. We document this in the Python `Plan.apply` docstring. No extra CUDA code.

### 4.6 Stream semantics — the one explicit rule

Every FFI entry point **must**:
1. Receive the stream via `Ctx<PlatformStream<cudaStream_t>>` (same as Phase 1).
2. Before any cuSPARSE call, `cusparseSetStream(handle, xla_stream)`. This binds the handle to the XLA stream for the duration of the call.
3. Issue no `cudaDeviceSynchronize`. `cudaStreamSynchronize` only if absolutely forced (it is not forced in our flows).
4. Not use any CUDA default stream (0) or `cudaStreamPerThread`.

This satisfies the Phase 1 "no host-device copy in the steady loop" rule, modulo the §2.3.2 token copy which is explicitly a control word, not solver data.

---

## 5. Build System Changes

Phase 1's `CMakeLists.txt` (at `dilu/ffi_mvp/CMakeLists.txt`, lines 1–55) already uses `find_package(CUDAToolkit)` and links `CUDA::cudart`. Phase 2 builds **a new library** (not a modification of `libdilu_ffi_mvp.so`) at `dilu/cusparse/CMakeLists.txt` producing `libdilu_cusparse.so`. Phase 1 artifacts stay untouched per the sign-off rule.

### 5.1 New link target

```cmake
target_link_libraries(dilu_cusparse PRIVATE
    CUDA::cudart
    CUDA::cusparse)     # new
```

`CUDA::cusparse` is exported by `find_package(CUDAToolkit)` for all supported CUDA versions (verified against CUDA 12.4 on the dev box via `cmake --find-package -DNAME=CUDAToolkit ... MODE=EXIST`).

### 5.2 cmake_minimum_required

`3.24` already suffices (Phase 1). No bump.

### 5.3 CUDA 13.x forward-compat posture

Key fact (search-verified, §10): in cuSPARSE 13.0+, `csrilu02`, `csrsv2`, and the rest of the legacy (pre-generic) sparse triangular solve API are **removed** or flagged for imminent removal. **Our Phase 2 code uses only `cusparseSpSV_*` generic APIs**. This survives the 13.x transition cleanly.

The dev box is on CUDA 12.4 (`nvcc 12.4.131`). The driver is 580.97 which supports CUDA 13 runtime compat, so if we link against 13.x in the future, it runs. There is **no** Phase 2 dependency on any API deprecated in 12.x or 13.x.

**Tripwire**: Phase 2 build step 1 compiles a single `.cu` file that `#include <cusparse.h>` and calls `cusparseSpSV_createDescr`, `cusparseSpSV_bufferSize`, `cusparseSpSV_analysis`, `cusparseSpSV_solve`, `cusparseSpSV_destroyDescr`. If any of these is missing on the active toolkit, build fails loudly and we escalate (§8).

### 5.4 Python side

No changes to `dilu/ffi_mvp/python/`. A new `dilu/cusparse/python/` directory houses:
- `registration.py` (same pattern as Phase 1 — load `.so`, register 4 targets: `cusparse_dilu_analyze`, `cusparse_dilu_apply`, `cusparse_dilu_release`, `dilu_factor`)
- `wrapper.py` (the `Plan` class and high-level functions)
- `plan.py` (the `Plan` context manager — creates analyze, owns token, calls release in `__exit__`)

### 5.5 Directory layout

```
dilu/
├── ffi_mvp/              [Phase 1, unchanged]
└── cusparse/             [Phase 2, new]
    ├── CMakeLists.txt
    ├── build.sh
    ├── cpp/
    │   ├── handle_registry.cc       # singleton handle per device
    │   ├── plan_cache.cc            # the unordered_map and its lock
    │   ├── dilu_factor.cc           # FFI wrapper
    │   ├── cusparse_dilu_analyze.cc # FFI wrapper
    │   ├── cusparse_dilu_apply.cc   # FFI wrapper
    │   └── cusparse_dilu_release.cc # FFI wrapper
    ├── cuda/
    │   ├── dilu_factor_kernel.cu    # hand-written DILU factorization kernel
    │   ├── scatter_diag_kernel.cu   # overwrites CSR diag with D_*
    │   └── elem_scale_kernel.cu     # y' = D_* * y between forward/backward solves
    ├── python/
    │   ├── __init__.py
    │   ├── registration.py
    │   ├── plan.py
    │   └── wrapper.py
    ├── tests/
    │   ├── _harness.py              # reuses Phase 1 CSR helpers (import, don't copy)
    │   ├── test_t4_factor_tridiag.py
    │   ├── test_t5_spsv_diag_is_jacobi.py
    │   ├── test_t6_spsv_laplacian3d.py
    │   ├── test_t7_pcg_small.py
    │   └── test_under_jit.py
    └── bench/
        ├── bench_dilu_apply.py
        └── bench_pcg.py
```

Lexically separate from `ffi_mvp/` per CLAUDE.md's "keep reference and new research lexically separated".

---

## 6. Testing & Verification Plan

Extending Phase 1's T1/T2/T3 structure:

### T4 — DILU factorization correctness

- **Reference**: pure-Python scipy-based DILU. We write ~30 LoC:
  ```
  def dilu_reference(A_csr):
      n = A_csr.shape[0]
      d = np.zeros(n)
      for i in range(n):
          d[i] = A_csr[i, i]
          for k in range(i):
              if A_csr[i, k] != 0 and A_csr[k, i] != 0:
                  d[i] -= A_csr[i, k] * A_csr[k, i] / d[k]
      return d
  ```
- **Test matrices**: T1 (tridiag $n=100$), T2 (3D 7-pt Laplacian $8^3=512$), T3 (diag-dominant random $n=200$).
- **Acceptance**: `|d_kernel - d_ref|_∞ ≤ C * κ(A) * ε_mach` with $C = O(n)$ for DILU's error recurrence. For well-conditioned test matrices this is ~$10^{-11}$ on float64.

### T5 — SpSV end-to-end, diagonal matrix sanity check

- **Setup**: $A$ = diagonal matrix (only diagonal entries nonzero in CSR). Then $L = U = 0$, and DILU reduces to Jacobi: $M = D$, $M^{-1} r = D^{-1} r$.
- **Reference**: Phase 1's `jacobi_residual_reference` with zero $A$-off-diagonals is exactly this. We compare `cusparse_dilu_apply(plan, A_csr, D_star, r)` against `r / diag`.
- **Acceptance**: `|z_kernel - r/d|_∞ ≤ 10 * ε_mach * |r|_∞`. If this fails, the bug is in our scatter kernel or in the fill-mode configuration — it rules out any cuSPARSE-level issue, because SpSV on a diagonal is trivial.

### T6 — SpSV on 3D Laplacian, dense-LU reference

- **Setup**: $A$ = 7-pt 3D Laplacian on $6^3 = 216$. Small enough to densify and LU on CPU.
- **Reference**: `z_ref = scipy.linalg.solve(M_dense, r)` where $M_{\text{dense}} = (D_* + L) D_*^{-1} (D_* + U)$ is assembled densely from $A$ and the Phase 2 $D_*$.
- **Acceptance**: `|z_kernel - z_ref|_∞ ≤ 100 * κ(M) * ε_mach * |r|_∞`. $κ(M)$ for this problem is $O(n)$ so tolerance is ~$10^{-12}$. Large headroom.

### T7 — Full PCG on a small stiff problem

- **Setup**: 3D 7-pt Laplacian on $20^3 = 8000$ with synthetic density jump (coefficient ratio $10^2$ across $z = N_z/2$). This is our "mini-AM" test case — stiffness without size.
- **PCG implementation**: pure-JAX outer loop with `cusparse_dilu_apply` as the preconditioner. ~40 LoC.
- **Convergence bar**: reach $|r|_2 / |r_0|_2 < 10^{-8}$ in ≤ 200 iterations. The OpenFOAM reference for this exact problem is not available without building a separate test harness, so we compare against:
  - **Reference 1**: unpreconditioned CG iteration count (expect 5–10× more).
  - **Reference 2**: Phase 1 Jacobi-preconditioned iteration count (expect DILU to win by ≥ 2×).
- **Acceptance**: DILU-PCG iteration count is **strictly lower** than Jacobi-PCG on the same problem, and final residual is below the bar.
- **Note**: T7 is not a cuSPARSE-correctness test, it's an end-to-end sanity test. Use it as the "does the stack compose?" signal.

### T-under-jit

- Same as Phase 1: `jax.jit` the PCG loop, verify HLO shows:
  - Exactly one `custom-call` per primitive (factor, analyze, apply).
  - Zero `copy-start`/`copy-done` **except** the 8-byte token copy on `apply` (documented exception).
  - No `cudaMalloc`/`cudaFree` in the loop body.

### Performance targets (not pass/fail, but logged)

| Metric | Dev box (3050) target | Rationale |
|---|---|---|
| `dilu_factor` at $n=10^4$ | < 5 ms | One-time cost; not hot path |
| `cusparse_dilu_analyze` at $n=10^4$, 7-pt | < 50 ms | One-time per matrix; tolerable |
| `cusparse_dilu_apply` at $n=10^4$ | < 2 ms | Hot path; budget 500 calls/sec |
| `cusparse_dilu_apply` at $n=10^5$ | < 20 ms | Same, scaled |
| PCG iter count on T7 | < 100 | Sanity |

If `apply` exceeds target by > 3×, **STOP** — likely a wrong descriptor attribute (e.g., fill mode) forcing cuSPARSE into a slow path.

---

## 7. Phase 2 Execution Checklist

Ordered, blocking. Each step: **(a)** what, **(b)** acceptance, **(c)** likely failure.

### Step 1 — cuSPARSE link probe

- **(a)** Write `dilu/cusparse/python/_probe.py` that imports and prints `CUDAToolkit_cusparse_version` (from a tiny C stub) plus verifies `cusparse.h` exists under `CUDAToolkit_INCLUDE_DIRS`.
- **(b)** Prints `CUSPARSE 12.4 OK, SpSV generic API present`. Fails loudly if version < 11.4 (when generic SpSV landed).
- **(c)** CUDA toolkit drift on user's box; fix = re-source env.

### Step 2 — Minimum SpSV "Hello"

- **(a)** Standalone `.cu` (NO JAX, NO FFI) — build a 5×5 tridiagonal lower-triangular matrix on device, run `cusparseSpSV_analysis` + `cusparseSpSV_solve` on a constant-1 RHS, `cudaMemcpy` result to host, print. Compile via one-line `nvcc`.
- **(b)** Prints a result matching the hand-computed forward solve of a 5×5 tridiag.
- **(c)** Link errors on `-lcusparse` → check `LD_LIBRARY_PATH` has `$CUDA_HOME/lib64`. STOP if still failing after env fix.

### Step 3 — Handle registry + plan cache skeleton

- **(a)** Implement `handle_registry.cc` (singleton handle, lazy init, one per device) and `plan_cache.cc` (thread-safe `unordered_map<uint64, PlanEntry>`). No FFI yet.
- **(b)** Unit test (GoogleTest or a trivial `main()`): create handle, insert a dummy PlanEntry, lookup, remove, confirm map empty.
- **(c)** Concurrency bug surfacing later if we don't think through the lock now; see §2.6 failure modes.

### Step 4 — `dilu_factor` FFI primitive

- **(a)** Implement `dilu_factor_kernel.cu` (naive one-level-at-a-time DILU factorization) + FFI wrapper. Match Phase 1's handler boilerplate.
- **(b)** T4 passes on T1/T2/T3 matrices.
- **(c)** Correctness bug in the serial recurrence will show up as divergence from the scipy reference at 1st row-off-diagonal; walk through T1 by hand.

### Step 5 — `cusparse_dilu_analyze` FFI primitive

- **(a)** Implement the analyze handler: handle lookup, descriptor creation, bufferSize, cudaMalloc, analysis, insert into plan cache, return token.
- **(b)** Returns a unique token each call. Running analyze then release leaves the plan cache empty. `nvidia-smi` shows no VRAM leak after 100 analyze+release cycles on T2.
- **(c)** `cusparseSpSV_analysis` returns `CUSPARSE_STATUS_NOT_SUPPORTED` if fill mode / diag type attribute is unset on the SpMatDescr. Fix = explicit `cusparseSpMatSetAttribute` calls with `CUSPARSE_SPMAT_FILL_MODE` and `CUSPARSE_SPMAT_DIAG_TYPE`.

### Step 6 — `cusparse_dilu_apply` FFI primitive

- **(a)** Implement the apply handler: token read (D→H), plan lookup, pattern fingerprint check, `cusparseSetStream`, scatter-diag kernel, forward SpSV, element-scale, backward SpSV.
- **(b)** T5 (diagonal-matrix case) passes first. Then T6 (Laplacian) passes.
- **(c)** T5 failing but T4 passing = wrong fill mode, wrong diag type, or wrong pattern fingerprint construction. T6 failing but T5 passing = missing diag scatter, or the `D_* y` element-scale is in the wrong place.

### Step 7 — Bench token-copy cost (§2.3.2 follow-up)

- **(a)** Measure `cusparse_dilu_apply` latency twice: once with the token copy (baseline) and once with a bypass that reads the token from a handler-internal cache keyed on the input buffer's device address (hack; only for measurement). Compare.
- **(b)** If token copy adds > 10% to apply latency at $n = 10^4$, commit the cache-bypass as a proper implementation. Otherwise drop the cache hack.
- **(c)** N/A — this is a conditional optimization.

### Step 8 — `cusparse_dilu_release` + VRAM leak test

- **(a)** Implement release handler. Write a Python test that runs analyze→apply×10→release 1000 times; check VRAM delta is < 5 MB.
- **(b)** Test passes.
- **(c)** Plan map retains stale pointers → release called twice leaks. Fix = `map.erase` before destroying cuSPARSE objects; hardened against double-release.

### Step 9 — T7 (full PCG) and under-jit verification

- **(a)** Implement the Python PCG loop, run T7 against unpreconditioned CG and Jacobi-CG. Wrap in `jax.jit`, inspect HLO, verify zero hot-path allocations.
- **(b)** All T7 acceptance criteria (§6) met. HLO shows only the allowed `copy-start`/`copy-done` for the token.
- **(c)** PCG stagnates = most likely our $D_*$ is wrong (T4 was insufficient coverage) OR the fill-mode/diag-type attributes flipped between analyze and apply. Log the issue against §8 STOP rules.

### Step 10 — Bench + report

- **(a)** Run `bench_dilu_apply.py` at $n \in \{10^3, 10^4, 10^5\}$. Run `bench_pcg.py` at T7. Produce `docs/benchmark/phase2_cusparse_report.md` with median/p95 and a comparison to Phase 1 Jacobi.
- **(b)** Report committed; acceptance criteria from §6 documented with actual numbers.
- **(c)** Performance below target → investigate before Phase 3 sign-off.

### Step 11 — Phase 2 sign-off

- **(a)** Append "Phase 2 implementation results" section to this doc. User reviews.
- **(b)** User signs Phase 2 complete. Phase 3 (multi-coloring) gates open.
- **(c)** N/A.

---

## 8. Fail-Fast Escape Hatches (CUDA-Specific)

Master-plan rule #1: stop and escalate rather than silently patch over. Phase 2's top 5 STOP signals, ordered by likelihood:

### STOP #1 — cuSPARSE version skew between jaxlib bundle and system CUDA

**Symptom**: `cusparseSpSV_createDescr` returns a success status but `analysis` returns `CUSPARSE_STATUS_INTERNAL_ERROR` with no further info. Or: link succeeds, runtime fails with `undefined symbol: cusparseSpSV_solve`. This is the jax-ml/jax#30050 / #29843 class of failure — jaxlib bundles cuSPARSE, but our `.so` also links cuSPARSE, and the two mismatch. Phase 1 did not hit this because it didn't link cuSPARSE.

**Detection**: Step 1's probe + link test. Plus: at handle-init time, print `cusparseGetVersion(handle, &v)` and compare to the compile-time `CUSPARSE_VERSION` macro. If different by more than the patch version, STOP.

**Escalation**: user reviews both (a) jaxlib's vendored cuSPARSE version and (b) the CUDA toolkit we linked, decides a pinning strategy. Do not attempt LD_PRELOAD tricks without approval.

### STOP #2 — Stream-ordering violation

**Symptom**: `apply` returns success, but subsequent JAX ops see uninitialized or stale data in `z`. Or: occasional `CUSPARSE_STATUS_EXECUTION_FAILED` that goes away with a `cudaStreamSynchronize`.

**Detection**: Missing `cusparseSetStream` before a solve call, or calling `cusparseSetStream` with the wrong stream object (e.g., the default stream). NVIDIA's documentation is explicit: any change to the handle's stream via `SetStream` affects all subsequent calls on that handle globally.

**Escalation**: STOP. Inspect the Nsight Systems trace on the failing test, verify the SpSV kernels are on the expected XLA stream, not on stream 0.

### STOP #3 — Descriptor double-free or leak

**Symptom**: `cudaErrorIllegalAddress` or `CUSPARSE_STATUS_ALLOC_FAILED` after many analyze+release cycles. OR `nvidia-smi` shows growing VRAM without bound across test iterations.

**Detection**: Step 8's leak test. Also: compute3-san (`compute-sanitizer`) run under CI — our build should be binary-compatible with it.

**Escalation**: STOP. Do not ship until leak plugged. This is a correctness-critical class of bug that will blow up in Phase 4 long runs.

### STOP #4 — cuSPARSE reports `CUSPARSE_STATUS_NOT_SUPPORTED` for the matrix structure

**Symptom**: `analysis` returns not-supported on an apparently-valid CSR. Possibilities: (a) matrix has zero-length rows (empty row in CSR); (b) matrix is not strictly triangular (after fill-mode attribute, cuSPARSE rejects non-triangular); (c) index type mismatch (we pass int32, cuSPARSE wanted int64).

**Detection**: T4–T6 failures with specific NOT_SUPPORTED errors.

**Escalation**: STOP. Read the error code mapping in the cuSPARSE manual, confirm which of (a)/(b)/(c) hit, fix or document the restriction in our `Plan` class. Do not silently coerce.

### STOP #5 — PCG iteration count explodes on T7

**Symptom**: T7's DILU-PCG needs > 500 iterations to converge, or does not converge. Jacobi-PCG converges. This is the "preconditioner is worse than no preconditioning" signal.

**Detection**: Step 9's acceptance test.

**Escalation**: STOP. Either $D_*$ is wrong (revisit T4 with more severe matrices), or the forward/backward solve is ordered incorrectly (the triangular solves are computing $M' \neq M$), or the `D_* y` middle step is wrong. Do not "tune" the PCG tolerance to hide the bug.

### (Documented, not top-5) — Fork-based multiprocess JAX

Our plan cache is per-process. If a JAX program forks (rare but possible), the child inherits the map but not the CUDA context. Mitigation: install a `pthread_atfork` child-handler that clears the plan map. This does not trigger STOP — it's a defensive guard, not a current failure mode.

---

## 9. Hardware Budget Revisit

### 9.1 Phase 1 baseline

- Dev box: RTX 3050 Laptop, 4096 MiB.
- Baseline (no Python) GPU-resident: ~907 MiB (desktop compositor, browser GPU context).
- Phase 1 peak JAX usage: < 50 MiB.
- Headroom: ~3 GB before OOM.

### 9.2 Phase 2 additions

Per-matrix, for a 3D 7-pt Laplacian at grid $N^3$ ($n = N^3$, $\mathrm{nnz} \approx 7n$):

| Item | Size | Per-matrix |
|---|---|---|
| CSR `row_ptr` | 4 × (n+1) bytes | ~28 KB at $n=10^4$ |
| CSR `col_idx` | 4 × nnz bytes | ~280 KB at $n=10^4$ |
| CSR `values` | 8 × nnz bytes | ~560 KB at $n=10^4$ |
| $D_*$ | 8 × n bytes | ~80 KB at $n=10^4$ |
| working_values (diag scatter) | 8 × nnz bytes | ~560 KB at $n=10^4$ |
| intermediate $y$ (between solves) | 8 × n bytes | ~80 KB at $n=10^4$ |
| `cusparseSpSV_bufferSize` forward | **variable** — cuSPARSE does not document the upper bound | typically O(nnz) words, so ~1–10 MB |
| `cusparseSpSV_bufferSize` backward | same | same |

Empirical estimate (to be verified in Step 5): total per-matrix plan cost ≈ **2–3× nnz bytes** = **4–10 MB** at $n = 10^4$.

### 9.3 Max grid size on 4 GB dev box

Budget: 3 GB VRAM available after desktop overhead. Reserve 1 GB for JAX's own runtime (XLA compiler, Python interpreter, misc).

Usable for solver + matrix: ~2 GB.

At $n = 10^5$, 7-pt 3D:
- CSR total: ~8.4 MB
- Plan workspaces: ~20 MB (upper bound)
- $D_*$, $r$, $z$, scratch: ~3 MB
- Total: ~30 MB per matrix. Comfortable.

At $n = 10^6$, 7-pt 3D ($100^3$):
- CSR total: ~84 MB
- Plan workspaces: ~200 MB
- Total: ~300 MB per matrix. Still fits.

At $n = 5 \times 10^6$ (~$170^3$):
- CSR total: ~420 MB
- Plan workspaces: ~1 GB
- Total: ~1.5 GB per matrix. **At the edge** of our budget.

**Phase 2 recommended test grid size**: $n \leq 10^6$ on the 3050. This is enough to exercise non-trivial SpSV behavior without OOM risk. Phase 3 moves to A100-class hardware and lifts the cap.

### 9.4 Plan-cache growth risk

If the user creates many distinct tokens (e.g., analyzing a matrix inside a loop body by mistake) and does not release, the map grows. At 20 MB per plan, 50 plans = 1 GB gone.

**Mitigation**: LRU cap with default 64 plans (env-var override). Warning emitted on insert-eviction. User-visible `Plan.__del__` wired through `release`.

---

## 10. References and Web-Verified Facts

Master plan rule #4 ("版本警觉"): none of the cuSPARSE minutiae below are trusted from training data. All verified via web search on 2026-04-20.

- **cuSPARSE Generic API reference (12.6 archive)**: `cusparseSpSV_bufferSize`, `_analysis`, `_solve`, `_createDescr`, `_destroyDescr` surface confirmed. <https://docs.nvidia.com/cuda/archive/13.0.0/cusparse/generic-api/generic-api-functions.html>
- **cuSPARSE 13.2 current**: confirms `csrilu02` family deprecated/removed, generic SpSV is the supported path. <https://docs.nvidia.com/cuda/cusparse/index.html>
- **cuSPARSE 12.4 archive** (matches our dev box toolkit): generic SpSV available; behavior as documented. <https://docs.nvidia.com/cuda/archive/12.4.0/cusparse/index.html>
- **cuSPARSE Management API** (handle / stream model): "handle must be private per thread"; `cusparseSetStream` affects all threads sharing the handle. This is the source for §2.6's thread-safety rule. <https://docs.nvidia.com/cuda/cusparse/basic-api/management-reference.html>
- **NVIDIA/CUDALibrarySamples SpSV CSR example**: canonical `cusparseSpSV_*` call sequence. <https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spsv_csr>
- **JAX FFI docs**: `ffi_call`, capsule-based handler registration. <https://docs.jax.dev/en/latest/ffi.html> and <https://docs.jax.dev/en/latest/jax.ffi.html>
- **JAX FFI "stateful" issue** (#25185): confirms the community pattern for stateful FFI is active but not yet turnkey. <https://github.com/jax-ml/jax/issues/25185>
- **jaxlib cuSPARSE load issues** (#29843, #30050): confirms cuSPARSE version skew is a real failure mode; source for STOP #1. <https://github.com/jax-ml/jax/issues/29843>, <https://github.com/jax-ml/jax/issues/30050>
- **Transformer Engine JAX** (cuDNN via FFI, singleton-handle pattern): <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html>
- **jax.experimental.sparse** (note: does NOT expose SpSV): <https://docs.jax.dev/en/latest/jax.experimental.sparse.html>
- **Naumov 2011**, "Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the GPU", NVIDIA NVR-2011-001 — the historical level-scheduling paper cited in our math foundation. Relevant to Phase 2 only as context; **we cannot assume `CUSPARSE_SPSV_ALG_DEFAULT` still uses this internally**. Our empirical characterization (§6, bench + Nsight trace) will reveal what the actual algorithm looks like on 12.4.

**Declared uncertainty** (master plan rule: be honest):
- NVIDIA has not publicly documented the internal algorithm of `CUSPARSE_SPSV_ALG_DEFAULT` for recent releases. The roadmap's phrasing "Level Scheduling 并行算法" is plausibly correct but is **not** a guarantee we can build on. Phase 2 treats cuSPARSE as a black box measured by T7 and Nsight, not a known-algorithm source.
- Exact cost of `cusparseSpSV_analysis` at $n = 10^5$ on RTX 3050 Laptop: unmeasured on this hardware; we estimate 1–5 ms from comparable benchmarks but will record the real number at Step 5.
- Whether the 8-byte token D→H copy (§2.3.2) blocks the stream: an assumption; measured at Step 7.

---

## 11. Phase 2 Engineering Acceptance Checklist

User sign-off required on all 8 before implementation begins.

1. **Descriptor-lifecycle strategy committed**: Option C (opaque-uint64-token analyze+apply split), with the single documented 8-byte D→H copy per apply as the sole tolerated host-device exception. Singleton cuSPARSE handle per device, `cusparseSetStream` at every apply. Option D held as a named fallback for the Step 7 measurement.

2. **Four FFI primitives agreed**: `dilu_factor` (standalone DILU factorization kernel; not cuSPARSE), `cusparse_dilu_analyze` (returns token), `cusparse_dilu_apply` (hot-path SpSV × 2), `cusparse_dilu_release` (teardown). No `dilu_refactor` primitive — refactor is a Python workflow.

3. **Pattern-fingerprint contract locked**: `(row_ptr_ptr, col_idx_ptr, n, nnz, idx_base, idx_type)` stored at analyze, checked at apply. Mismatch → `InvalidArgument`. User responsible for not mutating pattern buffers between analyze and apply (documented in `Plan.apply`).

4. **cuSPARSE API choice**: generic `cusparseSpSV_*` only. Zero use of deprecated `csrsv2`/`csrilu02`. Survives CUDA 13.x transition without API change.

5. **Build**: new library `libdilu_cusparse.so` under `dilu/cusparse/`, lexically separate from `ffi_mvp/`. Links `CUDA::cudart` + `CUDA::cusparse`. Same CMake / ctypes / pycapsule pipeline as Phase 1 — no new Python build machinery.

6. **Verification suite T4/T5/T6/T7** per §6 is pre-agreed. T5 (diagonal = Jacobi) is the correctness tripwire between our scatter kernel and cuSPARSE. T7 requires DILU-PCG < Jacobi-PCG iteration count on the stiff mini-AM problem.

7. **Top 5 STOP signals committed** (§8): cuSPARSE-version skew, stream-ordering violation, descriptor double-free/leak, `NOT_SUPPORTED` on matrix structure, PCG iteration-count explosion. When any fires, implementer writes a diagnostic and escalates.

8. **Hardware budget**: Phase 2 tests capped at $n \leq 10^6$ on the 3050 (per §9.3). Plan-cache size capped at 64 entries default. VRAM-leak test (Step 8) required before sign-off. Phase 1's VRAM-safety env vars carried over verbatim.

When all 8 are approved, implementation proceeds step-by-step per §7, with the user free to halt between any two steps.

**End of Phase 2 design document.**
