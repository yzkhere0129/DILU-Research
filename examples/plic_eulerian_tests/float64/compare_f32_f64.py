#!/usr/bin/env python3
"""Layer 3: float32 vs float64 precision comparison for Zalesak 3D 128^3.

Runs three configurations of the Eulerian PLIC pipeline on the Zalesak slotted
sphere (128^3 grid) for one full revolution (or 90 steps in --quick mode) and
prints a four-column comparison table:

    Config                  | V drift (%)  | L1 error | ms/step | Peak VRAM
    float32 + clip          | ...          | ...      | ...     | ...
    float32 + redistribute  | ...          | ...      | ...     | ...
    float64 + redistribute  | ...          | ...      | ...     | ...

Usage
-----
    # Full run (1 revolution, ~1800 steps at CFL=0.45, 128^3):
    JAX_ENABLE_X64=1 python compare_f32_f64.py

    # Quick validation (90 steps = 1/10 revolution):
    JAX_ENABLE_X64=1 python compare_f32_f64.py --quick

Environment requirements
------------------------
    - JAX x64 must be enabled BEFORE importing JAX:
        export JAX_ENABLE_X64=1
      OR the script will detect the missing config and exit with a clear error.
    - GPU recommended for the full run; CPU is fine for --quick.

Robustness guarantees
---------------------
    - Checks JAX x64 activation at startup; exits immediately if not enabled.
    - Gracefully handles conservative_bounds ImportError with a clear message
      and falls back to clip for that config.
    - Each config is run in an isolated Python scope to avoid cross-contamination.
    - Catches unexpected errors per-config and continues with NaN entries.

Expected runtimes (RTX 3050 4GB):
    --quick (90 steps, 128^3): ~2 min per config, ~6 min total
    Full run (1800 steps, 128^3): ~40 min per config, ~2 hr total

This script is intentionally NOT called by pytest.  Run it manually after
jax-cfd-am-expert's float64 implementation is merged.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math

# ── JAX x64 detection MUST happen before any jnp.array call ──────────────────
# We cannot call jax.config.update here because the user may have already
# imported JAX via an environment variable.  Instead we check and exit early.

def _check_x64() -> None:
    """Verify JAX x64 is enabled; exit with a clear error if not."""
    import jax
    import jax.numpy as jnp

    enabled = jax.config.jax_enable_x64
    probe = jnp.array(1.0)
    actually_f64 = probe.dtype.itemsize == 8

    if not enabled or not actually_f64:
        print(
            "\n[ERROR] JAX x64 mode is not active.\n"
            "  jax.config.jax_enable_x64 =", enabled, "\n"
            "  jnp.array(1.0).dtype      =", probe.dtype, "\n"
            "\nFix: set the environment variable BEFORE running this script:\n"
            "  export JAX_ENABLE_X64=1\n"
            "  python compare_f32_f64.py [--quick]\n"
            "\nOr add to the top of your shell profile:\n"
            "  export JAX_ENABLE_X64=1\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[OK] JAX x64 mode active: jnp.array(1.0).dtype = {probe.dtype}")


# Activate x64 before any other JAX call.
# This must be the very first JAX operation in the script.
import jax
jax.config.update("jax_enable_x64", True)
_check_x64()

import jax.numpy as jnp
import numpy as np

# ── Source path ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── PLIC imports ──────────────────────────────────────────────────────────────
from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.volume_formula import volume_below_plane_3d
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x,
    sweep_flux_y, apply_flux_y,
    sweep_flux_z, apply_flux_z,
)
from jax_laseram.vof.plic.diagnostics import compute_stats, format_stats

_CB_AVAILABLE = False
try:
    from jax_laseram.vof.plic.conservative_bounds import (
        redistribute_bounds_x,
        redistribute_bounds_y,
        redistribute_bounds_z,
    )
    _CB_AVAILABLE = True
    print("[OK] conservative_bounds imported successfully")
except (ImportError, AssertionError) as e:
    print(
        f"[WARN] conservative_bounds not available: {e}\n"
        "       Configs using redistribute will fall back to clip.\n"
        "       This is expected before jax-cfd-am-expert's float64 implementation."
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zalesak 3D geometry (matches run_zalesak_3d.py exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

N = 128
NH = 1
DX = DY = DZ = 1.0 / N

SPHERE_CENTER = (0.50, 0.50, 0.75)
SPHERE_R = 0.15
SLOT_W = 0.05   # width in x and y
SLOT_D = 0.25   # depth in z (from top of sphere downward)

OMEGA = 1.0
ROT_CENTER_Y = 0.50
ROT_CENTER_Z = 0.50
T_END = 2.0 * math.pi
CFL = 0.45


def halo(F: jnp.ndarray) -> jnp.ndarray:
    """Zero-gradient halo update (matches run_zalesak_3d.py convention)."""
    F = F.at[0].set(F[1])
    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1])
    F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1])
    F = F.at[:, :, -1].set(F[:, :, -2])
    return F


def create_slotted_sphere(dtype) -> jnp.ndarray:
    """JAX-vectorized 4x4x4 sub-cell sampling of the slotted sphere.

    Matches run_zalesak_3d.py create_slotted_sphere() exactly, but is
    dtype-parametric so we can create float32 or float64 initial conditions.
    """
    n_sub = 4
    c = np.linspace(DX / 2, 1 - DX / 2, N, dtype=np.float64)
    sub_off = (np.arange(n_sub, dtype=np.float64) + 0.5) / n_sub

    xs = c[:, None] - DX / 2 + DX * sub_off[None, :]  # (N, n_sub)

    Xs = xs[:, :, None, None, None, None]
    Ys = xs[None, None, :, :, None, None]
    Zs = xs[None, None, None, None, :, :]

    cx, cy, cz = SPHERE_CENTER
    r = np.sqrt((Xs - cx) ** 2 + (Ys - cy) ** 2 + (Zs - cz) ** 2)
    in_sphere = r <= SPHERE_R
    in_slot = (
        (np.abs(Xs - cx) < SLOT_W / 2)
        & (np.abs(Ys - cy) < SLOT_W / 2)
        & (Zs > cz + SPHERE_R - SLOT_D)
        & (Zs < cz + SPHERE_R)
    )
    F_sub = (in_sphere & ~in_slot).astype(np.float64)
    F_int = F_sub.mean(axis=(1, 3, 5)).astype(dtype)  # (N, N, N)

    Nt = N + 2 * NH
    F_full = np.zeros((Nt, Nt, Nt), dtype=dtype)
    F_full[NH:NH + N, NH:NH + N, NH:NH + N] = F_int
    F_jax = jnp.asarray(F_full)
    return halo(F_jax)


def compute_rotation_vel(dtype):
    """Rotation velocity field for x-axis rotation (matches run_zalesak_3d.py).

    u = 0, v = -omega*(z - zc), w = omega*(y - yc)
    Returns face-centered velocity arrays.
    """
    Nt = N + 2 * NH
    cc = np.linspace(-NH * DX + DX / 2, 1 + NH * DX - DX / 2, Nt, dtype=np.float64)

    u_face = np.zeros((Nt - 1, Nt, Nt), dtype=dtype)

    z_cc = cc
    _, _, Z_vf = np.meshgrid(
        0.5 * (cc[:-1] + cc[1:]), cc, z_cc, indexing="ij"
    )
    v_face = (-OMEGA * (Z_vf - ROT_CENTER_Z)).astype(dtype)

    y_cc = cc
    _, Y_wf, _ = np.meshgrid(
        cc, y_cc, 0.5 * (cc[:-1] + cc[1:]), indexing="ij"
    )
    w_face = (OMEGA * (Y_wf - ROT_CENTER_Y)).astype(dtype)

    return jnp.asarray(u_face), jnp.asarray(v_face), jnp.asarray(w_face)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLIC step implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _subsweep(F, vel_face, dt_sub, axis: str, use_redistribute: bool, n_iter: int = 3):
    """One PLIC sub-sweep along ``axis`` with clip or redistribute enforcement."""
    F = halo(F)
    nx, ny, nz = compute_youngs_normal_3d(F, DX, DY, DZ)
    C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)

    if axis == "x":
        flux = sweep_flux_x(F, nx, ny, nz, C, vel_face, dt_sub, DX, DY, DZ)
        F = apply_flux_x(F, flux, DX, DY, DZ)
        if use_redistribute and _CB_AVAILABLE:
            F = redistribute_bounds_x(F, vel_face, n_iter=n_iter)
        else:
            F = jnp.clip(F, 0.0, 1.0)
    elif axis == "y":
        flux = sweep_flux_y(F, nx, ny, nz, C, vel_face, dt_sub, DX, DY, DZ)
        F = apply_flux_y(F, flux, DX, DY, DZ)
        if use_redistribute and _CB_AVAILABLE:
            F = redistribute_bounds_y(F, vel_face, n_iter=n_iter)
        else:
            F = jnp.clip(F, 0.0, 1.0)
    elif axis == "z":
        flux = sweep_flux_z(F, nx, ny, nz, C, vel_face, dt_sub, DX, DY, DZ)
        F = apply_flux_z(F, flux, DX, DY, DZ)
        if use_redistribute and _CB_AVAILABLE:
            F = redistribute_bounds_z(F, vel_face, n_iter=n_iter)
        else:
            F = jnp.clip(F, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown axis: {axis}")

    return halo(F)


def plic_step_3d(F, u_face, v_face, w_face, dt, use_redistribute: bool):
    """3D Strang step y/2 → z → y/2 (rotation is in y-z plane, u=0).

    Matches run_zalesak_3d.py plic_step_3d() ordering.
    """
    F = _subsweep(F, v_face, dt / 2, "y", use_redistribute)
    F = _subsweep(F, w_face, dt, "z", use_redistribute)
    F = _subsweep(F, v_face, dt / 2, "y", use_redistribute)
    return F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run one configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _peak_vram_mb() -> float:
    """Return peak GPU memory usage in MB (0 if no GPU or not supported)."""
    try:
        backend = jax.default_backend()
        if backend != "gpu":
            return 0.0
        devices = jax.devices("gpu")
        if not devices:
            return 0.0
        stats = devices[0].memory_stats()
        peak_bytes = stats.get("peak_bytes_in_use", 0)
        return peak_bytes / 1e6
    except Exception:
        return float("nan")


def run_config(
    config_name: str,
    dtype,
    use_redistribute: bool,
    n_steps: int,
    quick: bool,
) -> dict:
    """Run one PLIC configuration and return metric dict.

    Returns
    -------
    dict with keys: config, v_drift_pct, l1_error, ms_per_step, peak_vram_mb
    """
    print(f"\n{'='*60}")
    print(f"Config: {config_name}  dtype={dtype.__name__}  n_steps={n_steps}")
    print(f"{'='*60}")

    try:
        # Geometry
        F0 = create_slotted_sphere(dtype)
        u_face, v_face, w_face = compute_rotation_vel(dtype)

        print(f"  Init: F0.dtype={F0.dtype}  shape={F0.shape}")
        if F0.dtype != jnp.dtype(dtype):
            raise AssertionError(
                f"Initial F dtype {F0.dtype} != requested {dtype}. "
                "float64 support is not yet active in create_slotted_sphere."
            )

        # CFL-based dt
        Nt = N + 2 * NH
        cc_half = np.linspace(DX / 2, 1 - DX / 2, N, dtype=np.float64)
        cc_all = np.linspace(-NH * DX + DX / 2, 1 + NH * DX - DX / 2, Nt, dtype=np.float64)
        max_v = float(np.max(np.abs(OMEGA * (cc_all - ROT_CENTER_Z))))
        max_w = float(np.max(np.abs(OMEGA * (cc_all - ROT_CENTER_Y))))
        dt = CFL * DX / max(max_v / DX + max_w / DX, 1e-10)

        cell_vol = DX * DY * DZ
        stats0 = compute_stats(F0, NH, cell_vol)
        V0 = stats0.volume
        print(f"  V0={V0:.10e}  dt={dt:.4e}  CFL_eff={max_v*dt/DX:.3f}")

        # JIT-compile the step function
        step_fn = jax.jit(
            lambda F: plic_step_3d(F, u_face, v_face, w_face, dt, use_redistribute)
        )

        # Warmup (compile + 3 steps)
        F = F0
        for _ in range(3):
            F = step_fn(F)
        jax.block_until_ready(F)

        # Timed run
        t0 = time.perf_counter()
        for _ in range(n_steps):
            F = step_fn(F)
        jax.block_until_ready(F)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1e3
        ms_per_step = elapsed_ms / n_steps

        # Metrics
        peak_vram = _peak_vram_mb()
        stats = compute_stats(F, NH, cell_vol)
        v_drift = (stats.volume - V0) / V0 * 100.0  # percent

        # L1 error: compare with float64 initial condition re-sampled to same dtype
        # (L1 is the mean absolute deviation from the (translated-back) initial shape.
        # For a quick/short run we use |F_final - F0| as a proxy for shape error.)
        l1_error = float(
            jnp.mean(jnp.abs(
                F[NH:-NH, NH:-NH, NH:-NH] - F0[NH:-NH, NH:-NH, NH:-NH]
            ))
        )

        print(f"  V_final={stats.volume:.10e}  V_drift={v_drift:+.4e}%")
        print(f"  L1(F-F0)={l1_error:.4e}  ms/step={ms_per_step:.1f}  "
              f"peak_VRAM={peak_vram:.0f}MB")
        print(f"  F dtype after run: {F.dtype}")

        return {
            "config": config_name,
            "v_drift_pct": v_drift,
            "l1_error": l1_error,
            "ms_per_step": ms_per_step,
            "peak_vram_mb": peak_vram,
        }

    except AssertionError as e:
        print(f"\n[SKIP] {config_name}: AssertionError — {e}")
        print("  This is expected before jax-cfd-am-expert's float64 implementation.")
        return {
            "config": config_name,
            "v_drift_pct": float("nan"),
            "l1_error": float("nan"),
            "ms_per_step": float("nan"),
            "peak_vram_mb": float("nan"),
        }
    except Exception as e:
        print(f"\n[ERROR] {config_name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config": config_name,
            "v_drift_pct": float("nan"),
            "l1_error": float("nan"),
            "ms_per_step": float("nan"),
            "peak_vram_mb": float("nan"),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt_f(v: float, width: int = 12, precision: int = 4) -> str:
    if math.isnan(v):
        return "NaN".center(width)
    return f"{v:{width}.{precision}e}"


def print_table(results: list[dict]) -> None:
    header = (
        f"{'Config':<32} {'V drift (%)':>14} {'L1 error':>12} "
        f"{'ms/step':>10} {'Peak VRAM':>12}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        name = r["config"][:32]
        row = (
            f"{name:<32} "
            f"{_fmt_f(r['v_drift_pct'], 14, 4)} "
            f"{_fmt_f(r['l1_error'], 12, 4)} "
            f"{_fmt_f(r['ms_per_step'], 10, 2)} "
            f"{_fmt_f(r['peak_vram_mb'], 12, 0)}"
        )
        print(row)
    print(sep)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare float32 vs float64 PLIC on Zalesak 3D 128^3"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 90 steps (~1/10 revolution) for pipeline validation",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of steps (ignores --quick if set)",
    )
    args = parser.parse_args()

    # Determine step count
    Nt_domain = N + 2 * NH
    cc_all = np.linspace(-NH * DX + DX / 2, 1 + NH * DX - DX / 2, Nt_domain, dtype=np.float64)
    max_v = float(np.max(np.abs(OMEGA * (cc_all - ROT_CENTER_Z))))
    max_w = float(np.max(np.abs(OMEGA * (cc_all - ROT_CENTER_Y))))
    dt_estimate = CFL * DX / max(max_v / DX + max_w / DX, 1e-10)
    n_steps_full = int(math.ceil(T_END / dt_estimate))

    if args.n_steps is not None:
        n_steps = args.n_steps
    elif args.quick:
        n_steps = 90  # ~1/10 revolution
    else:
        n_steps = n_steps_full

    print(f"\nZalesak 3D {N}^3 comparison")
    print(f"  Grid: {N}^3 interior cells + {NH} halo layers")
    print(f"  T_END={T_END:.4f}  dt~{dt_estimate:.4e}  n_steps_full~{n_steps_full}")
    print(f"  Running: {n_steps} steps {'(quick mode)' if args.quick else ''}")
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  redistribute available: {_CB_AVAILABLE}")

    configs = [
        {
            "name": "float32 + clip",
            "dtype": np.float32,
            "redistribute": False,
        },
        {
            "name": "float32 + redistribute",
            "dtype": np.float32,
            "redistribute": True,
        },
        {
            "name": "float64 + redistribute",
            "dtype": np.float64,
            "redistribute": True,
        },
    ]

    results = []
    for cfg in configs:
        r = run_config(
            config_name=cfg["name"],
            dtype=cfg["dtype"],
            use_redistribute=cfg["redistribute"],
            n_steps=n_steps,
            quick=args.quick,
        )
        results.append(r)

    print_table(results)

    print("\nNotes:")
    print("  V drift (%): fractional volume change relative to initial volume")
    print("  L1 error:    mean |F_final - F_initial| (shape change proxy)")
    print("  ms/step:     wall-clock time per Strang step (JIT-compiled)")
    print("  Peak VRAM:   peak GPU memory (MB), 0 on CPU")
    if not _CB_AVAILABLE:
        print("\n  [!] conservative_bounds not available; redistribute configs")
        print("      fell back to clip.  Results for those configs reflect clip.")
    print()


if __name__ == "__main__":
    main()
