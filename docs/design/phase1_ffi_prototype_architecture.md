# Phase 1 FFI-Prototype Engineering Architecture

**Project**: JAX-GPU AM-CFD Platform — Stiff Pressure-Poisson Solver Kernel Research
**Scope**: Phase 1 Part 2 (engineering architecture for the JAX ↔ C++/CUDA FFI MVP)
**Companion doc**: `phase1_dilu_math_foundation.md` (mathematical foundation by `cfd-math-expert`)
**Date**: 2026-04-19
**Author**: `jax-cfd-expert`
**Status**: DESIGN — awaiting user sign-off before any code is written

---

## 0. TL;DR

The goal of Phase 1 is to prove that **we can land a user-written CUDA kernel inside a `jax.jit`-compiled function with zero host/device copies**. Nothing in Phase 1 attempts to accelerate a real solver — this is purely an end-to-end toolchain smoke test. The MVP kernel is the Jacobi-preconditioned residual `y = D^{-1}(b - A x)` on CSR, agreed with the math doc §5.1 because it exercises every handshake Phase 2 (cuSPARSE level scheduling) will need: CSR pointer triplet, int32/float64 dtype mix, SpMV memory pattern, reduction, device-resident output, XLA-supplied CUDA stream.

Ground truth established in §1: **JAX 0.9.0 ships the typed XLA FFI under `jax.ffi` (not `jax.extend.ffi`), with headers at `jaxlib/include/xla/ffi/api/{api,ffi,c_api}.h`**. `custom_call_api_version=4` on the Python side pairs with `api_version=1` in `register_ffi_target` and `XLA_FFI_API_MAJOR=0, MINOR=2` in the C API. That is the canonical contract we build against.

We pick **CMake + pybind11-free pure C extension loaded via `ctypes`** as the build/load path (§4), with `jax.ffi.pycapsule(libfoo.handler)` as the registration bridge. No pybind11 module to build, no ABI wrestling. The kernel goes in `dilu/ffi_mvp/`, lexically distinct from `src/vof/` reference material per `CLAUDE.md`.

---

## 1. JAX FFI API-Version Reconnaissance

### 1.1 What I verified on this machine (not from memory)

Commands run against the live install (JAX 0.9.0 at `/home/yzk/jax-env`):

| Probe | Result |
|---|---|
| `import jax.extend.ffi` | **`ModuleNotFoundError`** — the `jax.extend.ffi` path does not exist in 0.9.0 |
| `import jax.ffi` → `dir()` | `ffi_call, ffi_lowering, include_dir, pycapsule, register_ffi_target, register_ffi_target_as_batch_partitionable, register_ffi_type, register_ffi_type_id, build_ffi_lowering_function` |
| `jax.ffi.include_dir()` | `/home/yzk/jax-env/lib/python3.12/site-packages/jaxlib/include` |
| Header files present | `xla/ffi/api/api.h` (2297 LoC), `xla/ffi/api/ffi.h` (1675 LoC), `xla/ffi/api/c_api.h` (786 LoC) |
| `c_api.h` version | `XLA_FFI_API_MAJOR 0`, `XLA_FFI_API_MINOR 2` |
| `ffi_call` signature | `custom_call_api_version: int = 4` default; `vmap_method` required going forward |
| `register_ffi_target` signature | `api_version: int = 1` (typed FFI) vs. `0` (legacy untyped) |
| CUDA stream access pattern | `Ffi::Bind().Ctx<PlatformStream<cudaStream_t>>().To([](cudaStream_t s, ...) {...})` — documented inline in `ffi.h:1281-1293` |
| CUDA toolkit | `nvcc 12.4.131` at `/usr/local/cuda-12.4` |
| Runtime GPU | RTX 3050 4 GB (compute 8.6), driver 580.97 (CUDA 13 runtime compat) |

**Conclusion**: on 0.9.0 the correct public API is `jax.ffi.*`. The `jax.extend.ffi` path the roadmap cites is **stale** (it was the 2023-era transitional name that migrated to `jax.ffi` before 0.5.x). The roadmap's warning about API churn is validated — but the churn has settled; 0.9 is stable on this surface.

### 1.2 The canonical MVP contract (JAX 0.9)

**C++ side** — define a typed handler:
```cpp
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

ffi::Error JacobiResidualImpl(cudaStream_t stream,
                              ffi::Buffer<ffi::S32> row_ptr,
                              ffi::Buffer<ffi::S32> col_idx,
                              ffi::Buffer<ffi::F64> values,
                              ffi::Buffer<ffi::F64> diag,
                              ffi::Buffer<ffi::F64> b,
                              ffi::Buffer<ffi::F64> x,
                              ffi::Result<ffi::Buffer<ffi::F64>> y);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JacobiResidual, JacobiResidualImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()   // row_ptr
        .Arg<ffi::Buffer<ffi::S32>>()   // col_idx
        .Arg<ffi::Buffer<ffi::F64>>()   // values
        .Arg<ffi::Buffer<ffi::F64>>()   // diag
        .Arg<ffi::Buffer<ffi::F64>>()   // b
        .Arg<ffi::Buffer<ffi::F64>>()   // x
        .Ret<ffi::Buffer<ffi::F64>>()); // y
```

**Python side** — register and call:
```python
import ctypes, jax, jax.numpy as jnp, jax.ffi
lib = ctypes.CDLL("./libdilu_ffi_mvp.so")
jax.ffi.register_ffi_target(
    "jacobi_residual",
    jax.ffi.pycapsule(lib.JacobiResidual),
    platform="CUDA",
    api_version=1,        # typed FFI
)

def jacobi_residual(row_ptr, col_idx, values, diag, b, x):
    n = b.shape[0]
    out_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    return jax.ffi.ffi_call(
        "jacobi_residual", out_type,
        vmap_method="sequential",
    )(row_ptr, col_idx, values, diag, b, x)
```

### 1.3 Known landmines (flagged from JAX 0.9 source + XLA release notes)

1. **`vmap_method` is becoming mandatory**: the default `"sequential"` already emits a `DeprecationWarning` on 0.9 and will `raise NotImplementedError` in a future release. **We pin it explicitly** in the prototype.
2. **Buffer layouts default to row-major**. For 1-D vectors this is moot, but if Phase 2 passes any 2-D array (e.g., block-Jacobi block storage), we must supply `input_layouts=[[0, 1], ...]` major-to-minor, **not** the minor-to-major ordering XLA uses internally. (Documented in `ffi_call` docstring.)
3. **CUDA graph capture**: `jax.jit` may capture the kernel into a CUDA graph. Any non-`cudaMallocAsync` allocation or host-side synchronization inside the handler will **silently break graph replay**. Phase 1 rule: zero `cudaMalloc`, zero `cudaMemcpy`, zero `cudaDeviceSynchronize` inside the handler. Output is written to the `Result<Buffer>` XLA has already allocated.
4. **Error-propagation**: return `ffi::Error::Success()` on OK, `ffi::Error::Internal("msg")` on failure. Do **not** throw C++ exceptions — XLA FFI is `noexcept`-on-boundary; an uncaught exception aborts the process.
5. **Shape checks are **your** responsibility**: XLA does not enforce `row_ptr.element_count() == n + 1`. The handler must validate or UB.
6. **`int32` vs `int64` on CSR indices**: cuSPARSE Phase 2 APIs accept both but 32-bit is the performant path. We commit to `int32` throughout. JAX default is `int32` under `jax_enable_x64=False`; with x64 on (we need it for `float64` values), `jnp.arange` returns `int64`. **Explicitly cast CSR index arrays to `jnp.int32`** before the FFI boundary.
7. **`np.ascontiguousarray` before FFI** (per `CLAUDE.md`): enforced on every host-side construction of the CSR arrays in tests, even though JAX arrays created via `jnp.*` are already contiguous. This is a guardrail for the inevitable day someone passes a sliced view.

### 1.4 What I am NOT certain about (declared)

- Whether `XLA_FFI_API_MINOR` will bump before Phase 2 lands (reading the header, the ABI policy is "minor bumps are backward-compatible"). Mitigation: re-run the probe in §1.1 at the start of each phase and diff.
- Whether `register_ffi_target_as_batch_partitionable` will become the preferred path for per-device calls under `pmap`/`shard_map` by Phase 4. Phase 1 does not touch sharding; we revisit.
- Whether there is any lurking interaction between `jax.ffi.ffi_call` and `jax.experimental.pjit` that matters for a single-device RTX 3050 target. Probably not, but we will not enable pjit in the prototype.

**Verification plan**: the first file the engineer writes is `dilu/ffi_mvp/python/_probe.py` that re-runs the §1.1 commands and asserts on expected values, so any silent API drift is caught on day 1.

---

## 2. MVP Kernel Specification

### 2.1 Agreement with the math doc

The math expert's §5.1 recommends `y = D^{-1}(b - A x)` on CSR. I concur and adopt it verbatim. Rationale (engineering side):

| Handshake exercised | How |
|---|---|
| Device-pointer handoff | Seven device buffers (5 inputs + 1 output + `diag`) passed through typed `Buffer<>` |
| CSR triplet layout | `row_ptr (int32, n+1)`, `col_idx (int32, nnz)`, `values (float64, nnz)` — exact shape Phase 2 cuSPARSE expects |
| GPU kernel launch | One kernel, one block-per-row-chunk; the launch itself is the thing we are proving works |
| Output buffer allocation | XLA pre-allocates via `Result<Buffer>`; kernel just writes to `y.typed_data()` |
| Stream awareness | `cudaStream_t` pulled from `PlatformStream<cudaStream_t>` context |
| Reduction | Per-row `sum_j A[i,j] * x[j]` — the SpMV inner loop, which is the next kernel we actually care about |
| Numerical verification | Trivially compared to `jnp.matmul(A_dense, x)` on a small n |

A "copy the diagonal" kernel would exercise only pointer plumbing and skip every interesting layout question. Not chosen.

### 2.2 Input/output contract (locked)

```
Inputs  (all device-resident, C-contiguous):
  row_ptr : int32[n+1]       (CSR row offsets)
  col_idx : int32[nnz]       (CSR column indices, within each row)
  values  : float64[nnz]     (CSR nonzeros, same order as col_idx)
  diag    : float64[n]       (precomputed; no division in kernel)
  b       : float64[n]
  x       : float64[n]

Output (device-resident, C-contiguous):
  y       : float64[n]       where y[i] = (b[i] - sum_j values[k]*x[col_idx[k]]) * diag[i]
                                          for k in [row_ptr[i], row_ptr[i+1])

Assumptions (caller responsibility, verified in Python test harness):
  - diag[i] = 1.0 / A[i,i], precomputed on host or via a previous JAX op
  - A is square, n >= 1, nnz >= n
  - All arrays are contiguous; row_ptr[0] == 0; row_ptr[n] == nnz
```

Why **precomputed `diag`** instead of computing `D^{-1}` inside the kernel: the point of Phase 1 is FFI mechanics, not arithmetic. Adding the "find the diagonal index in CSR" step inflates the kernel. We can do it in JAX with `jnp.where` in one line and pass it in. Phase 2's real DILU kernel will need `diag` as a separate device array anyway (it is what `D` becomes after the DILU factorization).

### 2.3 float64 everywhere in Phase 1

Per `CLAUDE.md`: "*When using float32, use ε ≥ 1e-6f and double-precision accumulators at reduction points.*" Phase 1 sidesteps this entirely by using `float64` for values and vectors. The MVP is about the toolchain, not the precision budget. float32 mixed-precision is a Phase 3+ concern.

Caveat: the RTX 3050 has FP64 throughput at 1/32 of FP32 (consumer silicon). Performance on the dev box will be miserable. This is **expected and acceptable** — Phase 1 perf is a baseline, not a target. When we move to an A100/H100 class card in Phase 3, FP64 is 1/2.

### 2.4 Numerical verification recipe (locked, per math doc §5.3)

Three test matrices, all float64, all assembled in Python with explicit small-integer entries so the reference is bitwise-predictable.

| ID | n | Structure | Purpose |
|---|---|---|---|
| T1 | 5 | Tridiag(−1, 2, −1) | Trivial correctness. Hand-computable. |
| T2 | 1000 | 3-D 7-point Laplacian on a 10×10×10 box | Exercises realistic CSR irregularity (6 nnz per interior row, 3–5 on boundary). |
| T3 | 10 | Diagonally dominant random, fixed `np.random.seed(0)` | Regression reproducibility across hardware. |

Acceptance: `|y_kernel - y_reference|_∞ <= 10 * nnz_per_row * machine_eps * |y_reference|_∞`. Reference is `D^{-1} * (b - A @ x)` computed as a dense matmul in JAX on the same device with `jax_enable_x64=True`.

### 2.5 Line-budget commitment

- C++/CUDA: **target 150 LoC, hard ceiling 200** including the handler binding macros.
- Python glue (registration + wrapper + tests): **target 80, hard ceiling 100** (test harness excluded from count but also <150).

If either ceiling is breached during implementation, that is a **STOP** signal (§6).

---

## 3. Directory Layout

```
/home/yzk/DILU-Research/
├── CLAUDE.md                              (unchanged)
├── dilu/
│   └── ffi_mvp/                           ← all Phase 1 artifacts live here
│       ├── README.md                      (1-page "how to build & run")
│       ├── CMakeLists.txt
│       ├── cpp/
│       │   └── jacobi_residual.cc         (handler + Ffi::Bind, ~40 LoC)
│       ├── cuda/
│       │   └── jacobi_residual_kernel.cu  (the __global__ kernel, ~60 LoC)
│       ├── python/
│       │   ├── __init__.py
│       │   ├── _probe.py                  (re-verifies §1.1 API surface on import)
│       │   ├── registration.py            (loads .so, calls register_ffi_target)
│       │   └── wrapper.py                 (the jax-friendly jacobi_residual() function)
│       ├── tests/
│       │   ├── test_t1_tridiag.py
│       │   ├── test_t2_laplacian3d.py
│       │   ├── test_t3_random_seeded.py
│       │   └── test_under_jit.py          (proves zero host-device copies via trace)
│       └── bench/
│           ├── bench_ffi_dispatch.py      (measures per-call FFI overhead)
│           └── nsys_capture.sh            (wraps the bench under Nsight Systems)
└── docs/design/
    ├── phase1_dilu_math_foundation.md     (math expert)
    └── phase1_ffi_prototype_architecture.md  (this doc)
```

Rationale for separating `cpp/` and `cuda/`: `nvcc` compiles `.cu` files with device-code pipeline; the handler binding in `.cc` is pure host C++. Keeping them separate lets us also compile the host `.cc` with plain `g++` if we ever need to verify it links without the CUDA toolchain (sanity check for the FFI surface). CMake handles both in one target.

Build artifacts go to `dilu/ffi_mvp/build/` (in `.gitignore`, separate from source). The loadable shared object is `build/libdilu_ffi_mvp.so`.

**Lexical separation from `src/vof/`**: reference-only VOF code under `src/vof/` is untouched. Phase 1 imports nothing from it. The VOF code does, however, remain a **reading reference** for parallel-kernel style (OMP atomic patterns → Phase 3 multi-coloring reference).

---

## 4. Build System Decision

**Choice: CMake + pure C-ABI shared library loaded by `ctypes`, wrapped in PyCapsule via `jax.ffi.pycapsule()`.**

### 4.1 Why not pybind11 / nanobind

- The FFI handler does not need Python bindings. It is called **by XLA**, not by Python. Python touches the handler pointer exactly once: `jax.ffi.pycapsule(lib.JacobiResidual)`.
- pybind11/nanobind add a Python-C++ glue layer whose only job would be to expose a function pointer we already have. That is negative value.
- `ctypes.CDLL(path).SymbolName` already resolves to a `ctypes` callable whose `.value` is a `void*` — which is exactly what `jax.ffi.pycapsule` wraps into an `XLA_FFI_Handler`. The `jax.ffi.pycapsule` docstring (verified live in §1) shows this exact pattern.

### 4.2 Why CMake over setuptools

- CMake natively understands `CUDA` as a first-class language (`enable_language(CUDA)`), handles `.cu` compilation with correct host/device flag separation, and discovers the toolkit via `find_package(CUDAToolkit)`. Setuptools + nvcc requires either `torch.utils.cpp_extension`-style hacks (unacceptable — new dependency) or a shell out to `nvcc` with hand-written flags (fragile).
- CMake scales cleanly to Phase 2 when we add `CUDAToolkit::cusparse` and `CUDAToolkit::cusolver` (`target_link_libraries`) and Phase 4 AMGx. Setuptools would become a mess.
- JAX FFI headers are auto-discovered via a trivial CMake helper: `execute_process(COMMAND python3 -c "import jax.ffi; print(jax.ffi.include_dir())" OUTPUT_VARIABLE JAX_FFI_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)`. One line.

### 4.3 Hermetic reproducibility

The build script pins:
- `cmake_minimum_required(VERSION 3.24)` (for first-class CUDA language support without legacy `FindCUDA`)
- `CMAKE_CUDA_STANDARD 17`, `CMAKE_CXX_STANDARD 17` (matches JAX FFI header requirements)
- `CMAKE_CUDA_ARCHITECTURES` parameterized via env var (default `86` for the dev box's RTX 3050; A100 = `80`, H100 = `90`)
- Explicitly **no `-ffast-math`** (CLAUDE.md convention; FP associativity matters for PCG convergence regression checks)
- `-O3 -lineinfo` (lineinfo for Nsight Compute kernel attribution; no performance cost)

A `build.sh` wraps the CMake invocation with the right env, so the engineer's daily loop is `cd dilu/ffi_mvp && ./build.sh && pytest tests/`.

### 4.4 What we explicitly do NOT do in Phase 1

- No conda/mamba env management (user already has a working `/home/yzk/jax-env`).
- No Docker. Phase 1 is a dev-box prototype. Reproducibility via Docker is a Phase 3 concern once we want CI.
- No `pip install -e .` / no `pyproject.toml` for now. The Python package lives at `dilu/ffi_mvp/python/` and is added to `sys.path` from the test files. Premature packaging is noise.

---

## 5. Execution Checklist

Ordered, blocking steps. Each has **(a) what, (b) acceptance, (c) likely failure**. The engineer works one at a time; no step starts until the previous passes its acceptance.

### Step 1 — API probe
- **(a)** Write `python/_probe.py` that asserts `jax.__version__` starts with `0.9`, `jax.ffi.include_dir()` returns an existing directory, `xla/ffi/api/ffi.h` exists under it, `jax.default_backend() == "gpu"`, and `jax.devices()[0].platform == "gpu"`.
- **(b)** `python3 python/_probe.py` exits 0 with a one-line "PROBE OK: jax 0.9.X, CUDA device=…" summary.
- **(c)** Fails if user's env drifts (likely: `jax-env` not sourced). Fix = source env. **Not** a STOP signal.

### Step 2 — Trivial CUDA "hello"
- **(a)** Write a standalone `.cu` with a `main()` that launches a 1-block kernel writing `1.0f` to a device buffer and `cudaMemcpy`s it back. Compile via a one-line `nvcc` call (not CMake yet). Run.
- **(b)** Binary prints `1.0`.
- **(c)** Fails if CUDA driver/toolkit mismatch. Driver 580 + toolkit 12.4 is known-good. **Not** a STOP if fixable in <1 hour.

### Step 3 — CMakeLists.txt for the real build
- **(a)** Write CMakeLists producing `libdilu_ffi_mvp.so` from an empty stub `jacobi_residual.cc` that defines and exports a no-op `extern "C" XLA_FFI_Error* JacobiResidual(XLA_FFI_CallFrame*)` symbol. Include the JAX FFI headers. **Do not implement the kernel yet.**
- **(b)** `./build.sh` produces `build/libdilu_ffi_mvp.so`; `nm -D` on it shows `JacobiResidual` as a global symbol; the file links against `libcudart.so`.
- **(c)** JAX FFI headers have a transitive C++17 feature usage that breaks under older compilers. If gcc < 11, STOP (**escalation signal**, see §6).

### Step 4 — Empty handler registration round-trip
- **(a)** In `registration.py`, `ctypes.CDLL` the `.so`, pass its `JacobiResidual` symbol through `jax.ffi.pycapsule`, call `jax.ffi.register_ffi_target("jacobi_residual", ..., platform="CUDA", api_version=1)`. In `wrapper.py`, call `jax.ffi.ffi_call("jacobi_residual", ...)` with dummy inputs. The C++ handler still does nothing (returns success, writes zeros).
- **(b)** `jnp.all(y == 0)` holds. The call does **not** raise. The same thing inside `jax.jit(...)` also does not raise.
- **(c)** Registration-time errors ("unknown custom call target") almost always mean `platform=` mismatch (must be `"CUDA"` literal, not `"gpu"` or `"cuda"`). Misspelling = hours of wasted time. If persists after spelling check, STOP.

### Step 5 — Implement the kernel
- **(a)** Write the CUDA kernel (`__global__ void jacobi_residual_kernel`) with **one thread per row**, `__ldg` loads on `row_ptr`/`col_idx`/`values`, accumulator in register, no shared memory, no warp-level primitives. Wire it into the handler. Launch with `<<<(n + 255) / 256, 256, 0, stream>>>`.
- **(b)** Returns correct result for a hard-coded `n=4` test case (tridiag) compared to hand-computed reference.
- **(c)** Likely: passing `stream=0` instead of the XLA-provided stream → works but breaks under `jit`. Fix = ensure `Ctx<PlatformStream<cudaStream_t>>()` is first in `Ffi::Bind()`.

### Step 6 — T1/T2/T3 verification
- **(a)** Run the three test matrices from §2.4.
- **(b)** All three pass the `10 * nnz_per_row * eps * |y|_∞` tolerance.
- **(c)** T1 fails but T2 passes → index type bug (int32 vs int64 at the FFI boundary). T2 fails but T1 passes → out-of-bounds read on `col_idx`; probably missing bounds check or wrong `row_ptr[i+1]` interpretation. T3 flaky → race condition (shouldn't happen with one-thread-per-row, but check anyway).

### Step 7 — Under `jax.jit`, verify zero copies
- **(a)** Wrap the call in `jax.jit` and use `jax.debug.visualize_array_sharding` or `jax.lax.stop_gradient` + `.block_until_ready()`. Use `jax.live_arrays()` before and after to assert no transient host copies.
- **(b)** `str(jax.make_jaxpr(f)(...))` shows `ffi_call` as a primitive. The compiled HLO (accessible via `jax.jit(f).lower(...).compile().as_text()`) contains one `custom-call` op and zero `copy-start`/`copy-done` to host memory.
- **(c)** If `copy-start` appears on CPU→GPU for any of the CSR arrays, something upstream is producing a host array (missing `jax.device_put` or `jnp.asarray`).

### Step 8 — Profile baseline
- **(a)** Run `bench/bench_ffi_dispatch.py` (§7): measure per-call latency at n=1k, 10k, 100k via `jax.profiler` trace and via `time.perf_counter` + `.block_until_ready()` 10k-iter loop.
- **(b)** Produce a Markdown report in `dilu/ffi_mvp/bench/baseline_rtx3050.md` with median/p95 latency at each size, plus a `nsys` screenshot showing the kernel + launch overhead breakdown.
- **(c)** If FFI dispatch overhead > 50 μs, something is wrong (expected range: 5–20 μs per call for the typed FFI on modern JAX). STOP and diagnose.

### Step 9 — Sign-off doc
- **(a)** Append a "Phase 1 implementation results" section to **this** architecture doc with the actual numbers, commit hashes, and a screenshot of one successful Nsight capture.
- **(b)** User reviews and marks Phase 1 complete.
- **(c)** n/a.

---

## 6. Fail-Fast Escape Hatches

Per the roadmap rule #1 ("如果你在尝试构建 JAX-FFI 时发现不可调和的编译错误...请立刻停止并报告"). The top 5 **STOP and escalate to the user** signals:

1. **JAX FFI header compilation fails with the project's host compiler** (e.g., gcc < 11 not supporting required C++17 features, or a clash between JAX's pinned `std::variant` implementation and the system STL). Symptom: template instantiation errors in `xla/ffi/api/api.h`. Do not "fix" by patching the headers.

2. **`register_ffi_target` succeeds but `ffi_call` raises `XLA_FFI_Error: unknown custom call target`**. This is the classic sign of an **XLA FFI ABI minor-version skew** between what `jaxlib` shipped and what our compiled `.so` sees. Do not try to "force" a match. Pin JAX version and re-probe.

3. **Correctness passes T1 but fails T2/T3** after multiple kernel rewrites. Indicates a deep misunderstanding of the buffer layout contract (possibly XLA's layout guarantees differ from our expectation for multi-dim inputs). Stop writing kernels; go read `xla/ffi/api/ffi.h` §Buffer contract with a domain expert.

4. **`jax.jit`-compiled call hangs, does not return, and no error**. Likely a CUDA stream-ordering violation: our handler synchronizes on `stream_A` while XLA issues the next op on `stream_B`, deadlocking the pipeline. Stop. Read Nsight Systems' stream timeline. Do not bisect by guessing.

5. **FFI dispatch overhead measured >100 μs per call consistently on n=1k**. Either the typed FFI has a regression we did not expect, or we are hitting a pathology (e.g., PyCapsule recreated every call, `jit` cache miss on every invocation). Phase 1's performance bar is "reasonable, not fast" — but 100 μs makes Phase 2 meaningless. Stop and investigate before adding kernels.

Every STOP signal means: **the engineer writes a 1-page diagnostic note, pastes the failing output, and hands back to the user**. No heroic debugging. The roadmap is explicit that Phase 1 is allowed to fail fast rather than burn weeks on a toolchain dead end.

---

## 7. Profiling Plan

Phase 1 produces a **baseline, not a target**. The numbers we report define what Phase 2 has to beat.

### 7.1 What to measure

| Metric | Tool | Why |
|---|---|---|
| Per-call end-to-end latency (Python `time.perf_counter` bracketing `block_until_ready()`) | `time` stdlib | User-visible wall-clock; includes jit-dispatch + FFI + kernel + stream sync |
| `ffi_call` primitive wallclock (inside a jit-compiled trace) | `jax.profiler.trace` → Perfetto UI | Isolates the FFI dispatch from Python overhead |
| CUDA kernel execution time (device-side) | Nsight Systems (`nsys profile --stats=true`) | The "real" kernel cost, independent of dispatch |
| Kernel-launch overhead | Nsight Systems stream view | The irreducible ~5 μs per launch we fight in Phase 2 |
| Host→device copies per call | Nsight Systems memcpy rows | Must be **zero** after Step 7 passes |

### 7.2 Bench harness shape (sketch, not implementation)

`bench/bench_ffi_dispatch.py` will:
1. Build T2 (10³ Laplacian, n=1000) on device via `jax.device_put`.
2. Warm up: 100 calls inside `jax.jit`, `.block_until_ready()`.
3. Measure: 10 000 calls in a tight Python loop, each `.block_until_ready()`, report median/p95/p99.
4. Repeat for n=10 000, n=100 000.
5. Emit `baseline_rtx3050.md` with a table. Commit it to `dilu/ffi_mvp/bench/`.

### 7.3 Nsight capture

`nsys_capture.sh` wraps the bench under `nsys profile -t cuda,nvtx,osrt -o baseline.qdrep ...` producing a .qdrep we open in Nsight Systems GUI. Screenshots go in the sign-off doc.

### 7.4 What we expect to see (prediction, for calibration)

- n=1000: FFI dispatch ~10 μs + kernel ~5 μs + stream sync ~5 μs = ~20 μs total per call.
- n=100000: kernel dominated (~100 μs for SpMV of that size on a 3050 in FP64), FFI dispatch now only ~10% of runtime.
- Zero host-device traffic on steady-state calls.

If the actual numbers are within 2× of these predictions, Phase 1 is a success. If 10× worse, §6 STOP #5 triggers.

### 7.5 What we deliberately do NOT measure

- PCG convergence. There is no PCG in Phase 1.
- FP32 throughput. float64 only in Phase 1.
- Multi-GPU scaling. Single-GPU only.

---

## 8. Risk Register (brief)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| JAX 0.9.X → 0.10 before Phase 2 bumps FFI API again | Medium | High | Pin `jax==0.9.*` in a `requirements.lock`; re-run §1.1 probe monthly |
| RTX 3050 FP64 throughput too poor for any meaningful kernel timing | High | Low | Declared expected; Phase 3 moves to A100/H100 |
| CMake + CUDA language setup breaks on user's box | Low | Medium | Step 2 (`nvcc` standalone) isolates the toolchain concern from CMake |
| XLA FFI minor ABI bump silently breaks our `.so` | Low | High | `_probe.py` checks `XLA_FFI_API_MINOR` header constant; the `.so` reports its compiled-against version at load time |
| Roadmap step ordering drift (math before engineering, vs. both parallel) | n/a | Low | Math doc already landed; we align on MVP kernel choice §2.1 |

---

## 9. Phase 1 Engineering Acceptance Checklist

User sign-off required on all 8 before implementation begins.

1. **API surface committed**: `jax.ffi.ffi_call` (NOT `jax.extend.ffi`), `custom_call_api_version=4` Python-side / `api_version=1` registration-side, XLA FFI C API 0.2, `vmap_method="sequential"` explicit. Verified against live JAX 0.9.0 install (§1.1).
2. **MVP kernel is `y = D^{-1}(b - A x)` on CSR**, float64, `diag` precomputed on host, single kernel with one-thread-per-row launch. Aligned with math doc §5.1. (§2)
3. **Build: CMake only, no pybind11/nanobind**, `.so` loaded via `ctypes` and `jax.ffi.pycapsule`. `cmake_minimum_required 3.24`, C++17, CUDA architectures parameterized. (§4)
4. **Directory: `dilu/ffi_mvp/{cpp,cuda,python,tests,bench}`**, separate from `src/vof/` reference, separate from any future `dilu/` subdirectory for Phase 2+. (§3)
5. **Verification suite T1/T2/T3** per math doc §5.3 is pre-agreed; kernel is not "working" until all three pass the tolerance bound. (§2.4)
6. **Zero host-device copies** verified via HLO inspection (`jit.lower().compile().as_text()` shows no `copy-start`/`copy-done`) in Step 7. (§5, §7)
7. **Top 5 STOP signals agreed** (§6): hgcc/C++17 incompatibility, unknown-target on registered call, T2-but-not-T1 correctness, jit-hang, >100 μs dispatch. When any fires, engineer stops and writes a diagnostic note.
8. **Profiling deliverable: `bench/baseline_rtx3050.md`** with median/p95 latency at n=1k/10k/100k plus one Nsight Systems capture, committed. This is the baseline Phase 2 cuSPARSE work must beat. (§7)

When all 8 are approved, implementation proceeds one Execution-Checklist step at a time (§5), with the user free to halt between any two steps.

**End of design document.**
