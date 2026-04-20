#!/usr/bin/env python3
"""Zalesak 3D slotted-sphere, float32 vs float64 head-to-head.

Runs the SAME 128^3 case as `zalesak_3d/run_zalesak_3d.py` with a chosen
dtype and bounds scheme. Mirrors the reference script line-for-line so
numbers can be compared directly.

Usage:
    JAX_ENABLE_X64=1 python run_zalesak_3d_f64.py --dtype f64 --bounds redistribute
    python run_zalesak_3d_f64.py --dtype f32 --bounds clip   # matches baseline
"""
from __future__ import annotations
import argparse, os, sys, time as timer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))

# -------- CLI + JAX x64 must be set before any other JAX activity --------
parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=["f32", "f64"], default="f32")
parser.add_argument("--bounds", choices=["clip", "redistribute"], default="clip")
parser.add_argument("--quick", action="store_true", help="90 steps (1/10 rev)")
args = parser.parse_args()

import jax
if args.dtype == "f64":
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

DT = jnp.float64 if args.dtype == "f64" else jnp.float32
NP_DT = np.float64 if args.dtype == "f64" else np.float32

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_y, apply_flux_y, sweep_flux_z, apply_flux_z,
)
from jax_laseram.vof.plic.conservative_bounds import (
    apply_flux_y_conservative, apply_flux_z_conservative,
)

# ------------- Parameters (match run_zalesak_3d.py exactly) -------------
N = 128
NH = 1
DX = DY = DZ = 1.0 / N
SPHERE_CENTER = (0.50, 0.50, 0.75)
SPHERE_R = 0.15
SLOT_W = 0.05
SLOT_D = 0.25
OMEGA = 1.0
ROT_CENTER_Y = 0.50
ROT_CENTER_Z = 0.50
T_END = 2.0 * np.pi
CFL = 0.45


def halo(F):
    F = F.at[0].set(F[1]); F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F


def create_slotted_sphere():
    """CPU init with numpy (avoids float64 OOM at 128^3 × 4^3 subcells)."""
    n_sub = 4
    c = np.linspace(DX / 2, 1 - DX / 2, N, dtype=NP_DT)
    sub_off = (np.arange(n_sub, dtype=NP_DT) + 0.5) / n_sub
    xs = c[:, None] - DX / 2 + DX * sub_off[None, :]  # (N, n_sub)
    cx, cy, cz = SPHERE_CENTER

    # slice-by-slice in z to avoid 6D allocation (~4GB in f64)
    F_int = np.zeros((N, N, N), dtype=NP_DT)
    for zi in range(N):
        Zs = xs[zi, None, None, None, :]                    # (1,1,1,n_sub)
        Xs = xs[:, :, None, None, None]                     # (N,n_sub,1,1,1)
        Ys = xs[None, None, :, :, None]                     # (1,1,N,n_sub,1)
        r = np.sqrt((Xs - cx) ** 2 + (Ys - cy) ** 2 + (Zs - cz) ** 2)
        in_sphere = r <= SPHERE_R
        in_slot = (
            (np.abs(Xs - cx) < SLOT_W / 2)
            & (np.abs(Ys - cy) < SLOT_W / 2)
            & (Zs > cz + SPHERE_R - SLOT_D)
            & (Zs < cz + SPHERE_R)
        )
        F_sub = (in_sphere & ~in_slot).astype(NP_DT)
        F_int[:, :, zi] = F_sub.mean(axis=(1, 3, 4))

    Nt = N + 2 * NH
    F_full = np.zeros((Nt, Nt, Nt), dtype=NP_DT)
    F_full[NH : NH + N, NH : NH + N, NH : NH + N] = F_int
    F_full[0] = F_full[1]; F_full[-1] = F_full[-2]
    F_full[:, 0] = F_full[:, 1]; F_full[:, -1] = F_full[:, -2]
    F_full[:, :, 0] = F_full[:, :, 1]; F_full[:, :, -1] = F_full[:, :, -2]
    return jnp.asarray(F_full)


def compute_rotation_vel():
    Nt = N + 2 * NH
    cc = jnp.linspace(-NH * DX + DX / 2, 1 + NH * DX - DX / 2, Nt, dtype=DT)

    # u_face at x-faces: (Nt-1, Nt, Nt), identically zero
    u_face = jnp.zeros((Nt - 1, Nt, Nt), dtype=DT)

    # v_face at y-faces: (Nt, Nt-1, Nt), v = -ω(z - zc)
    cc_mid = 0.5 * (cc[:-1] + cc[1:])
    _, _, Z_vf = jnp.meshgrid(cc, cc_mid, cc, indexing="ij")
    v_face = (-OMEGA * (Z_vf - ROT_CENTER_Z)).astype(DT)

    # w_face at z-faces: (Nt, Nt, Nt-1), w = ω(y - yc)
    _, Y_wf, _ = jnp.meshgrid(cc, cc, cc_mid, indexing="ij")
    w_face = (OMEGA * (Y_wf - ROT_CENTER_Y)).astype(DT)

    return u_face, v_face, w_face


def make_step(u_face, v_face, w_face, dt):
    use_redist = args.bounds == "redistribute"

    def sub_y(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_y(F, nx, ny, nz, C, v_face, dt_s, DX, DY, DZ)
        F = apply_flux_y(F, flux, DX, DY, DZ)
        if use_redist:
            F = apply_flux_y_conservative(F, flux, dt_s, DY, v_face, n_iter=3)
        else:
            F = jnp.clip(F, 0.0, 1.0)
        return halo(F)

    def sub_z(F, dt_s):
        F = halo(F)
        nx, ny, nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_z(F, nx, ny, nz, C, w_face, dt_s, DX, DY, DZ)
        F = apply_flux_z(F, flux, DX, DY, DZ)
        if use_redist:
            F = apply_flux_z_conservative(F, flux, dt_s, DZ, w_face, n_iter=3)
        else:
            F = jnp.clip(F, 0.0, 1.0)
        return halo(F)

    def step(F):
        F = sub_y(F, dt / 2)
        F = sub_z(F, dt)
        F = sub_y(F, dt / 2)
        return F

    return jax.jit(step)


def main():
    print("=" * 68)
    print(f"Zalesak 3D  |  dtype={args.dtype}  |  bounds={args.bounds}  |  grid={N}³")
    print("=" * 68)
    print(f"  JAX x64 mode : {jax.config.jax_enable_x64}")
    print(f"  Device       : {jax.devices()[0]}")
    print(f"  DT constant  : {DT}")

    t0 = timer.time()
    F_init = create_slotted_sphere()
    jax.block_until_ready(F_init)
    print(f"  Init time    : {timer.time()-t0:.1f}s  F.dtype={F_init.dtype}")
    u_face, v_face, w_face = compute_rotation_vel()

    max_vel = float(jnp.maximum(jnp.abs(v_face).max(), jnp.abs(w_face).max()))
    dt = CFL * DX / max_vel
    n_steps_full = int(np.ceil(T_END / dt))
    dt = T_END / n_steps_full
    n_steps = max(1, n_steps_full // 10) if args.quick else n_steps_full
    print(f"  max|vel|={max_vel:.6f}  dt={dt:.4e}  n_steps={n_steps} "
          f"({'quick 1/10 rev' if args.quick else 'full rev'})")

    cell_vol = DX * DY * DZ
    F_ini_np = np.asarray(F_init).astype(NP_DT)
    V0 = float(np.sum(F_ini_np[NH:-NH, NH:-NH, NH:-NH]) * cell_vol)
    print(f"  V0 = {V0:.12e}")

    step = make_step(u_face, v_face, w_face, dt)

    # JIT compile
    tc0 = timer.time()
    Fw = step(F_init); jax.block_until_ready(Fw)
    print(f"  JIT compile : {timer.time()-tc0:.1f}s")

    F = F_init
    t_wall0 = timer.time()
    report_every = max(1, n_steps // 10)

    for s in range(1, n_steps + 1):
        F = step(F)
        if s % report_every == 0 or s == n_steps:
            jax.block_until_ready(F)
            F_np = np.asarray(F).astype(NP_DT)
            V = float(np.sum(F_np[NH:-NH, NH:-NH, NH:-NH]) * cell_vol)
            dV = (V - V0) / V0 * 100
            L1 = float(np.sum(np.abs(F_np[NH:-NH, NH:-NH, NH:-NH]
                                     - F_ini_np[NH:-NH, NH:-NH, NH:-NH])) * cell_vol)
            L1rel = L1 / V0 * 100
            fmin, fmax = float(F_np.min()), float(F_np.max())
            print(f"  step {s:5d}/{n_steps}  V_drift={dV:+.3e}%  L1={L1rel:6.3f}%  "
                  f"F∈[{fmin:+.3e},{fmax:.6f}]  wall={timer.time()-t_wall0:.1f}s", flush=True)

    wall = timer.time() - t_wall0
    F_np = np.asarray(F).astype(NP_DT)
    V = float(np.sum(F_np[NH:-NH, NH:-NH, NH:-NH]) * cell_vol)
    dV = (V - V0) / V0 * 100
    L1 = float(np.sum(np.abs(F_np[NH:-NH, NH:-NH, NH:-NH]
                              - F_ini_np[NH:-NH, NH:-NH, NH:-NH])) * cell_vol)
    L1rel = L1 / V0 * 100

    # Persist
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    tag = f"{args.dtype}_{args.bounds}{'_quick' if args.quick else ''}"
    np.savez_compressed(
        os.path.join(_HERE, "results", f"zalesak_3d_{tag}.npz"),
        F_final=F_np[NH:-NH, NH:-NH, NH:-NH],
        F_init=F_ini_np[NH:-NH, NH:-NH, NH:-NH],
        N=N, DX=DX, n_steps=n_steps, wall_time=wall,
        V0=V0, V_drift_pct=dV, L1_rel_pct=L1rel, dtype=args.dtype, bounds=args.bounds,
    )

    mem = jax.devices()[0].memory_stats() or {}
    peak_mb = mem.get("peak_bytes_in_use", 0) / 1024 / 1024

    print("=" * 68)
    print(f"FINAL  {tag}")
    print("=" * 68)
    print(f"  V drift      : {dV:+.6e}%")
    print(f"  L1 shape err : {L1rel:.4f}%")
    print(f"  ms/step      : {wall/n_steps*1000:.1f}")
    print(f"  total wall   : {wall:.1f}s  ({n_steps} steps)")
    print(f"  GPU peak VRAM: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
