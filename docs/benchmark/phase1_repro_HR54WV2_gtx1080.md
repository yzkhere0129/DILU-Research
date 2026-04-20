# Phase 1 FFI-dispatch reproduction on HR54WV2

Generated: 2026-04-20T09:23:59+00:00

## Environment

| key | value |
|---|---|
| hostname | `HR54WV2` |
| gpu_name | `NVIDIA GeForce GTX 1080` |
| driver | `581.57` |
| vram | `8192 MiB` |
| cuda | `12.9` |
| jax | `0.9.1` |
| python | `3.13.12` |
| cpu | `Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz` |
| os | `Linux 6.6.87.2-microsoft-standard-WSL2` |
| jax_devices | `[CudaDevice(id=0)]` |

## Protocol

- Matrix: 1-D Laplacian tridiag, n=1000, float64 (bit-identical to Phase 1 baseline)
- Random seed for b, x: 0 (numpy default_rng)
- Inputs pre-placed on device via `jax.device_put`
- Warmup: 200 jit-compiled calls (ignored)
- Measurement: 500 calls under `jax.jit`, each followed by `.block_until_ready()`
- Profiler trace: 100 calls inside `jax.profiler.trace`
- VRAM rails: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.5`, `ALLOCATOR=platform`
- Build artifact: `dilu/ffi_mvp/build/libdilu_ffi_mvp.so` (unmodified from Phase 1)

## Dispatch overhead (microseconds)

| mode | samples | min | p10 | p50 (median) | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bare (no profiler) | 500 | 410.67 | 434.82 | 455.26 | 513.98 | 715.61 | 957.69 | 471.96 |
| under profiler trace | 100 | 526.47 | 537.66 | 575.84 | 779.63 | 996.47 | 1551.45 | 621.96 |

Profiler trace directory: `dilu/ffi_mvp/bench/cross_machine_trace_HR54WV2`

## Comparison reference

Phase 1 RTX 3050 Laptop baseline (see `docs/benchmark/phase1_mvp_report.md` §4):
- bare, n=1000, 2000 iter: median 266.5 us, min 224.3 us, p95 369.8 us
- under profiler, 100 iter: median 413.8 us, min 272.9 us, p95 9147.2 us (flush-inflated)

Architecture-doc prediction (§7.4): ~20 us per call at n=1k.

## One-line summary

```
[REPRO] host=HR54WV2 gpu=NVIDIA GeForce GTX 1080 median=455.26us p99=715.61us n=1000 samples=500
```
