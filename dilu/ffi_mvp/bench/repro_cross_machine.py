"""Cross-machine FFI-dispatch overhead reproduction.

Phase 1 measured ~270 us median dispatch at n=1000 on RTX 3050 Laptop vs. the
architecture-doc prediction of ~20 us. That hypothesis (consumer-laptop
limits) needs a second data point on another GPU using the identical build
artifact and measurement protocol. This script captures env + runs the same
tridiag(n=1000,seed=0) jit-dispatch bench, writes a hostname-tagged report
under docs/benchmark/, and prints a one-line `[REPRO] ...` summary.
Run from repo root: `python dilu/ffi_mvp/bench/repro_cross_machine.py`
"""
from __future__ import annotations

import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

# VRAM safety rails --- must precede any JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)  # so `import bench_ffi_dispatch` resolves

# --- Phase 1 parameters (match profile_single_tiny.py exactly) -------------
N = 1000
SEED = 0
N_WARMUP = 200
N_MEASURE = 500        # >=100 per brief; 500 for tighter tail statistics
N_PROFILER_ITERS = 100
BUILD_ARTIFACT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "build", "libdilu_ffi_mvp.so")
)
# ---------------------------------------------------------------------------


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10, check=False)
        return (out.stdout or out.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"<unavailable: {e.__class__.__name__}>"


def _short_gpu(name):
    """'NVIDIA GeForce RTX 3050 Laptop GPU' -> 'rtx3050laptop'."""
    s = re.sub(r"nvidia|geforce|gpu", "", name.lower())
    return re.sub(r"[^a-z0-9]+", "", s) or "unknown"


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def collect_env():
    gpu_csv = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"])
    first = (gpu_csv.splitlines()[0] if gpu_csv and "unavailable" not in gpu_csv
             else "unknown, unknown, unknown")
    parts = [p.strip() for p in first.split(",")] + ["unknown"] * 3
    gpu_name, driver, vram = parts[0], parts[1], parts[2]
    m = re.search(r"release\s+([\d.]+)", _run(["nvcc", "--version"]))
    try:
        import jax  # deferred so env-capture can stand alone
        jax_version, jax_devices = jax.__version__, str(jax.devices())
    except Exception as e:  # noqa: BLE001
        jax_version, jax_devices = f"<import failed: {e}>", "unknown"
    return {"hostname": socket.gethostname(), "gpu_name": gpu_name,
            "gpu_short": _short_gpu(gpu_name), "driver": driver, "vram": vram,
            "cuda": m.group(1) if m else "unknown",
            "jax": jax_version, "jax_devices": jax_devices,
            "python": sys.version.split()[0], "cpu": _cpu_model(),
            "os": f"{platform.system()} {platform.release()}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


_ENV_KEYS = ("hostname", "gpu_name", "driver", "vram", "cuda", "jax",
             "jax_devices", "python", "cpu", "os", "timestamp_utc")


def verify_artifact():
    if not os.path.isfile(BUILD_ARTIFACT):
        sys.stderr.write(
            f"ERROR: build artifact missing: {BUILD_ARTIFACT}\nRun: cd "
            f"{os.path.dirname(os.path.dirname(BUILD_ARTIFACT))} && bash build.sh\n")
        sys.exit(2)


def _loop(f, args, n):
    """Dispatch f(*args).block_until_ready() n times; return per-call us list."""
    out = [0.0] * n
    for i in range(n):
        t0 = time.perf_counter()
        f(*args).block_until_ready()
        out[i] = (time.perf_counter() - t0) * 1e6
    return out


def measure(env):
    # Imports happen AFTER verify_artifact() so a missing .so surfaces cleanly.
    from jax import config as _cfg
    _cfg.update("jax_enable_x64", True)
    import numpy as np
    import jax
    import jax.numpy as jnp
    from dilu.ffi_mvp.python import jacobi_residual
    from bench_ffi_dispatch import build_tridiag

    (rp_h, ci_h, v_h), diag_h = build_tridiag(N)
    rng = np.random.default_rng(SEED)
    b_h = rng.standard_normal(N).astype(np.float64)
    x_h = rng.standard_normal(N).astype(np.float64)

    def put(h, dt):
        return jax.device_put(jnp.asarray(np.ascontiguousarray(h), dtype=dt))

    args = (put(rp_h, jnp.int32), put(ci_h, jnp.int32),
            put(v_h, jnp.float64), put(1.0 / diag_h, jnp.float64),
            put(b_h, jnp.float64), put(x_h, jnp.float64))

    @jax.jit
    def f(rp, ci, v, d, b, x):
        return jacobi_residual(rp, ci, v, d, b, x)

    for _ in range(N_WARMUP):
        f(*args).block_until_ready()
    bare = _loop(f, args, N_MEASURE)

    trace_dir = os.path.join(_THIS_DIR, f"cross_machine_trace_{env['hostname']}")
    os.makedirs(trace_dir, exist_ok=True)
    with jax.profiler.trace(trace_dir):
        traced = _loop(f, args, N_PROFILER_ITERS)
    return bare, traced, trace_dir


def summarize(samples):
    qs = statistics.quantiles(samples, n=100, method="inclusive")
    return {"n": len(samples), "min": min(samples), "p10": qs[9],
            "p50": statistics.median(samples), "p90": qs[89], "p99": qs[98],
            "max": max(samples), "mean": statistics.fmean(samples)}


def _fmt(s):
    return " ".join(f"{k}={s[k]:.2f}" for k in
                    ("min", "p10", "p50", "p90", "p99", "max")) + f" us (n={s['n']})"


def _row(label, s):
    return (f"| {label} | {s['n']} | " + " | ".join(
        f"{s[k]:.2f}" for k in ("min", "p10", "p50", "p90", "p99", "max", "mean")
    ) + " |")


def write_report(env, bare, traced, trace_dir):
    report_dir = os.path.join(_REPO_ROOT, "docs", "benchmark")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir,
                        f"phase1_repro_{env['hostname']}_{env['gpu_short']}.md")
    env_rows = "\n".join(
        f"| {k} | `{env[k]}` |"
        for k in ("hostname", "gpu_name", "driver", "vram", "cuda", "jax",
                  "python", "cpu", "os", "jax_devices"))
    body = f"""# Phase 1 FFI-dispatch reproduction on {env['hostname']}

Generated: {env['timestamp_utc']}

## Environment

| key | value |
|---|---|
{env_rows}

## Protocol

- Matrix: 1-D Laplacian tridiag, n={N}, float64 (bit-identical to Phase 1 baseline)
- Random seed for b, x: {SEED} (numpy default_rng)
- Inputs pre-placed on device via `jax.device_put`
- Warmup: {N_WARMUP} jit-compiled calls (ignored)
- Measurement: {N_MEASURE} calls under `jax.jit`, each followed by `.block_until_ready()`
- Profiler trace: {N_PROFILER_ITERS} calls inside `jax.profiler.trace`
- VRAM rails: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.5`, `ALLOCATOR=platform`
- Build artifact: `{os.path.relpath(BUILD_ARTIFACT, _REPO_ROOT)}` (unmodified from Phase 1)

## Dispatch overhead (microseconds)

| mode | samples | min | p10 | p50 (median) | p90 | p99 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{_row('bare (no profiler)', bare)}
{_row('under profiler trace', traced)}

Profiler trace directory: `{os.path.relpath(trace_dir, _REPO_ROOT)}`

## Comparison reference

Phase 1 RTX 3050 Laptop baseline (see `docs/benchmark/phase1_mvp_report.md` §4):
- bare, n=1000, 2000 iter: median 266.5 us, min 224.3 us, p95 369.8 us
- under profiler, 100 iter: median 413.8 us, min 272.9 us, p95 9147.2 us (flush-inflated)

Architecture-doc prediction (§7.4): ~20 us per call at n=1k.

## One-line summary

```
[REPRO] host={env['hostname']} gpu={env['gpu_name']} median={bare['p50']:.2f}us p99={bare['p99']:.2f}us n={N} samples={bare['n']}
```
"""
    with open(path, "w") as f:
        f.write(body)
    return path


def main():
    env = collect_env()
    print("=" * 72 + "\nPhase 1 FFI-dispatch cross-machine reproduction\n" + "=" * 72)
    for k in _ENV_KEYS:
        print(f"  {k:<14} {env[k]}")
    print("-" * 72)
    verify_artifact()
    print(f"Measuring: n={N}, warmup={N_WARMUP}, measure={N_MEASURE}, "
          f"profiler_iters={N_PROFILER_ITERS}")
    bare_raw, traced_raw, trace_dir = measure(env)
    bare, traced = summarize(bare_raw), summarize(traced_raw)
    print(f"\nbare  (no profiler):  {_fmt(bare)}")
    print(f"under profiler trace: {_fmt(traced)}")
    report = write_report(env, bare, traced, trace_dir)
    print(f"\nReport:    {report}\nTrace dir: {trace_dir}\n")
    print(f"[REPRO] host={env['hostname']} gpu={env['gpu_name']} "
          f"median={bare['p50']:.2f}us p99={bare['p99']:.2f}us "
          f"n={N} samples={bare['n']}")


if __name__ == "__main__":
    main()
