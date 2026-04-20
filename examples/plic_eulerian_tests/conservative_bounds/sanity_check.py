#!/usr/bin/env python3
"""Sanity check for Weymouth-Zaleski conservative bounds redistribute.

Mirrors the 1M-cell 45-deg advection configuration of
``examples/plic_eulerian_tests/profile_1M/profile_1M_eulerian.py`` but
runs 10 Strang steps with both the original ``jnp.clip`` variant and
the new ``*_conservative`` variant, and compares:

  - V drift:      | V(T) / V(0) - 1 |
  - L1 error:     | F_conservative - F_clip |.sum() * dV   (should be tiny)
  - Hard bounds:  max(F - 1),  max(-F)   (should be < 1e-7)
  - Overhead:     ms/step vs clip
  - HLO size:     make_jaxpr equation counts for apply_flux_x vs
                  apply_flux_x_conservative
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))

import jax
import jax.numpy as jnp
import numpy as np

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x,
    sweep_flux_y, apply_flux_y,
    apply_flux_x_conservative, apply_flux_y_conservative,
)


# ---------- Config (matches profile_1M_eulerian.py) ---------------
NX, NY, NZ = 250, 250, 16
DOMAIN_XY = 2.5
DX = DOMAIN_XY / NX                      # 0.01
CFL = 0.45
SPEED = 1.0
ANGLE_DEG = 45
N_STEPS = 10
NH = 1
MAX_FRAC = 0.15

dt = CFL * DX / SPEED
theta = np.radians(ANGLE_DEG)
UX, UY = np.cos(theta) * SPEED, np.sin(theta) * SPEED

print("=" * 72)
print("  Weymouth-Zaleski conservative bounds  —  sanity check (1M cells)")
print("=" * 72)
print(f"  Grid: {NX}x{NY}x{NZ} = {NX*NY*NZ:,} cells")
print(f"  dx={DX:.4f}, dt={dt:.6f}, N_STEPS={N_STEPS}")
print(f"  Velocity: ({UX:.4f}, {UY:.4f})")
print(f"  Device: {jax.devices()[0]}")

# ---------- Init droplet -----------------------------------------
x_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
y_c = np.linspace(DX/2, DOMAIN_XY - DX/2, NY)
X2, Y2 = np.meshgrid(x_c, y_c, indexing="ij")
R = 0.25
cx, cy = 1.25, 1.25
dist = np.sqrt((X2 - cx)**2 + (Y2 - cy)**2)
hd = np.sqrt(2) * DX / 2
F2d = np.clip((R - dist + hd) / (2*hd), 0, 1).astype(np.float32)

Ntx, Nty, Ntz = NX + 2*NH, NY + 2*NH, NZ + 2*NH
F_np = np.zeros((Ntx, Nty, Ntz), dtype=np.float32)
F_np[NH:-NH, NH:-NH, NH:-NH] = F2d[:, :, None]
F_np[0] = F_np[1]; F_np[-1] = F_np[-2]
F_np[:, 0] = F_np[:, 1]; F_np[:, -1] = F_np[:, -2]
F_np[:, :, 0] = F_np[:, :, 1]; F_np[:, :, -1] = F_np[:, :, -2]
F_init = jnp.asarray(F_np)

cell_vol = DX**3
V0 = float(F_init[NH:-NH, NH:-NH, NH:-NH].sum()) * cell_vol
print(f"  V0 = {V0:.10f}")


# ---------- Halo closure -----------------------------------------
def halo(F):
    F = F.at[0].set(F[1]);    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F


TOTAL_CELLS = F_init.size
MAX_N = int(TOTAL_CELLS * MAX_FRAC)


def reconstruct(F):
    nx, ny, nz = compute_youngs_normal_3d(F, DX, DX, DX)
    is_if = (F > 1e-6) & (F < 1 - 1e-6)
    idx = jnp.where(is_if.ravel(), size=MAX_N, fill_value=0)[0]
    F_c = F.ravel()[idx]
    nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
    C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX, DX, DX)
    C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
    return nx, ny, nz, C


# ---------- CLIP variant ----------------------------------------
@jax.jit
def step_clip(F):
    # x/2
    F = halo(F)
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
    F = halo(F)
    # y
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt, DX, DX, DX)
    F = jnp.clip(apply_flux_y(F, flux, DX, DX, DX), 0.0, 1.0)
    F = halo(F)
    # x/2
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    F = jnp.clip(apply_flux_x(F, flux, DX, DX, DX), 0.0, 1.0)
    return halo(F)


# ---------- Diagnostic: raw step (no clip / no redistribute) ----
@jax.jit
def step_raw(F):
    """Same as step_clip but no clip — reveals the raw over/undershoot
    magnitude caused by the analytic intercept float32 residue.
    """
    F = halo(F)
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    F = halo(apply_flux_x(F, flux, DX, DX, DX))
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt, DX, DX, DX)
    F = halo(apply_flux_y(F, flux, DX, DX, DX))
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    return halo(apply_flux_x(F, flux, DX, DX, DX))


# ---------- CONSERVATIVE variant --------------------------------
# API (matches test suite):  apply_flux_x_conservative(F, flux, dt, dx, u_face)
# The caller must have already applied apply_flux_x; the conservative
# wrapper only enforces F in [0, 1] via redistribute.
@jax.jit
def step_cons(F):
    F = halo(F)
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    F = apply_flux_x(F, flux, DX, DX, DX)
    F = apply_flux_x_conservative(F, flux, dt/2, DX, UX, n_iter=3)
    F = halo(F)
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, UY, dt, DX, DX, DX)
    F = apply_flux_y(F, flux, DX, DX, DX)
    F = apply_flux_y_conservative(F, flux, dt, DX, UY, n_iter=3)
    F = halo(F)
    nx, ny, nz, C = reconstruct(F)
    flux = sweep_flux_x(F, nx, ny, nz, C, UX, dt/2, DX, DX, DX)
    F = apply_flux_x(F, flux, DX, DX, DX)
    F = apply_flux_x_conservative(F, flux, dt/2, DX, UX, n_iter=3)
    return halo(F)


# ---------- HLO equation counts ---------------------------------
print("\n" + "-" * 72)
print("  HLO / jaxpr equation counts")
print("-" * 72)


def _eqn_count(jaxpr):
    # A jaxpr has .eqns and sub-jaxprs; count recursively to catch
    # lax.fori_loop bodies too.
    total = len(jaxpr.eqns)
    for eqn in jaxpr.eqns:
        for p in eqn.params.values():
            # jax.core.Jaxpr or ClosedJaxpr in params (e.g. fori body)
            if hasattr(p, "jaxpr"):
                total += _eqn_count(p.jaxpr)
            elif hasattr(p, "eqns"):
                total += _eqn_count(p)
    return total


fake_F = F_init
fake_flux_x = jnp.zeros((Ntx - 1, Nty, Ntz), dtype=jnp.float32)
jaxpr_plain = jax.make_jaxpr(
    lambda F, flux: apply_flux_x(F, flux, DX, DX, DX)
)(fake_F, fake_flux_x)
jaxpr_cons = jax.make_jaxpr(
    lambda F, flux: apply_flux_x_conservative(F, flux, dt / 2, DX, UX, n_iter=3)
)(fake_F, fake_flux_x)

n_plain = _eqn_count(jaxpr_plain.jaxpr)
n_cons = _eqn_count(jaxpr_cons.jaxpr)
print(f"  apply_flux_x              : {n_plain:5d} eqns")
print(f"  apply_flux_x_conservative : {n_cons:5d} eqns  (+{n_cons-n_plain})")


# ---------- Warmup ----------------------------------------------
print("\n" + "-" * 72)
print("  JIT compile / warmup")
print("-" * 72)
t0 = time.time()
F_warm = step_clip(F_init); F_warm.block_until_ready()
print(f"  clip     jit: {time.time()-t0:.1f}s")
t0 = time.time()
F_warm = step_cons(F_init); F_warm.block_until_ready()
print(f"  cons     jit: {time.time()-t0:.1f}s")


# ---------- Run ------------------------------------------------
print("\n" + "-" * 72)
print(f"  Running {N_STEPS} Strang steps")
print("-" * 72)


def run(step_fn, label):
    F = F_init
    # Warm one more time for stable timing
    F_ = step_fn(F); F_.block_until_ready()
    t0 = time.time()
    for _ in range(N_STEPS):
        F = step_fn(F)
    F.block_until_ready()
    wall = time.time() - t0
    Fi = F[NH:-NH, NH:-NH, NH:-NH]
    V = float(Fi.sum()) * cell_vol
    overshoot = float(jnp.maximum(F - 1.0, 0.0).max())
    undershoot = float(jnp.maximum(-F, 0.0).max())
    fmin = float(F.min())
    fmax = float(F.max())
    print(
        f"  {label:12s}  "
        f"wall={wall*1000/N_STEPS:6.2f} ms/step  "
        f"V={V:.10f}  "
        f"dV/V={abs(V - V0)/V0:.3e}  "
        f"F in [{fmin:+.3e}, {fmax:.6f}]  "
        f"max(F-1)+={overshoot:.3e}  max(-F)+={undershoot:.3e}"
    )
    return F, wall


F_raw, _ = run(step_raw, "raw (no clip)")
F_clip, wall_clip = run(step_clip, "clip")
F_cons, wall_cons = run(step_cons, "conservative")

diff = float(jnp.abs(F_cons - F_clip).sum()) * cell_vol
max_diff = float(jnp.abs(F_cons - F_clip).max())
print(f"\n  L1(F_cons - F_clip) = {diff:.6e}   max|diff| = {max_diff:.3e}")
overhead_ms = (wall_cons - wall_clip) * 1000 / N_STEPS
print(f"  Overhead of conservative vs clip: {overhead_ms:+.3f} ms/step")

print("\n" + "=" * 72)
print("  SUMMARY — 45 deg translation")
print("=" * 72)
print(f"  Hard bound (conservative):  max(F-1) = "
      f"{float(jnp.maximum(F_cons - 1.0, 0.0).max()):.3e}")
print(f"                              max(-F)  = "
      f"{float(jnp.maximum(-F_cons, 0.0).max()):.3e}")
V_clip = float(F_clip[NH:-NH, NH:-NH, NH:-NH].sum()) * cell_vol
V_cons = float(F_cons[NH:-NH, NH:-NH, NH:-NH].sum()) * cell_vol
print(f"  V drift  (clip)         :  {abs(V_clip-V0)/V0:.3e}")
print(f"  V drift  (conservative) :  {abs(V_cons-V0)/V0:.3e}")
print("=" * 72)


# =================================================================
# Second test: mini Zalesak-like face-velocity case, 128^3 grid but
# short run.  This is the regime where the original jnp.clip produces
# measurable V drift (0.054% per full revolution @ 2pi rad).
# =================================================================
print("\n" + "=" * 72)
print("  Mini Zalesak rotation (face velocity)  —  64^3, 40 Strang steps")
print("=" * 72)

N2 = 64
DX2 = 1.0 / N2
OMEGA = 1.0
SPHERE_R = 0.15
Nt2 = N2 + 2 * NH

# Init: slotted sphere, 4x4x4 sub-sampling (slot axis = z, rotation in y-z)
c2 = jnp.linspace(DX2 / 2, 1 - DX2 / 2, N2, dtype=jnp.float32)
sub_off = (jnp.arange(4, dtype=jnp.float32) + 0.5) / 4
xs_ = c2[:, None] - DX2 / 2 + DX2 * sub_off[None, :]
Xs = xs_[:, :, None, None, None, None]
Ys = xs_[None, None, :, :, None, None]
Zs = xs_[None, None, None, None, :, :]
r_ = jnp.sqrt((Xs - 0.5)**2 + (Ys - 0.5)**2 + (Zs - 0.75)**2)
in_sphere = r_ <= SPHERE_R
in_slot = (
    (jnp.abs(Xs - 0.5) < 0.025)
    & (jnp.abs(Ys - 0.5) < 0.025)
    & (Zs > 0.75 + SPHERE_R - 0.25)
    & (Zs < 0.75 + SPHERE_R)
)
F2d = (in_sphere & ~in_slot).astype(jnp.float32).mean(axis=(1, 3, 5))
F_ini2 = jnp.zeros((Nt2, Nt2, Nt2), dtype=jnp.float32)
F_ini2 = F_ini2.at[NH:NH+N2, NH:NH+N2, NH:NH+N2].set(F2d)

def halo2(F):
    F = F.at[0].set(F[1]);    F = F.at[-1].set(F[-2])
    F = F.at[:, 0].set(F[:, 1]); F = F.at[:, -1].set(F[:, -2])
    F = F.at[:, :, 0].set(F[:, :, 1]); F = F.at[:, :, -1].set(F[:, :, -2])
    return F

F_ini2 = halo2(F_ini2)

# Rotation velocity around x-axis at (*, 0.5, 0.5)
cc2 = jnp.linspace(-NH * DX2 + DX2/2, 1 + NH * DX2 - DX2/2, Nt2, dtype=jnp.float32)
u_face2 = jnp.zeros((Nt2-1, Nt2, Nt2), dtype=jnp.float32)
_, _, Z_vf = jnp.meshgrid(cc2, 0.5 * (cc2[:-1] + cc2[1:]), cc2, indexing="ij")
v_face2 = (-OMEGA * (Z_vf - 0.5)).astype(jnp.float32)
_, Y_wf, _ = jnp.meshgrid(cc2, cc2, 0.5 * (cc2[:-1] + cc2[1:]), indexing="ij")
w_face2 = (OMEGA * (Y_wf - 0.5)).astype(jnp.float32)

max_vel2 = float(jnp.maximum(jnp.abs(v_face2).max(), jnp.abs(w_face2).max()))
dt2 = CFL * DX2 / max_vel2
N_STEPS2 = 200  # ~ half a revolution
print(f"  dt = {dt2:.5e}, max|vel| = {max_vel2:.3f}, N_STEPS = {N_STEPS2}")


TOTAL_2 = F_ini2.size
MAX_N_2 = int(TOTAL_2 * MAX_FRAC)

def reconstruct2(F):
    nx, ny, nz = compute_youngs_normal_3d(F, DX2, DX2, DX2)
    is_if = (F > 1e-6) & (F < 1 - 1e-6)
    idx = jnp.where(is_if.ravel(), size=MAX_N_2, fill_value=0)[0]
    F_c = F.ravel()[idx]
    nx_c = nx.ravel()[idx]; ny_c = ny.ravel()[idx]; nz_c = nz.ravel()[idx]
    C_c = analytic_intercept(nx_c, ny_c, nz_c, F_c, DX2, DX2, DX2)
    C = jnp.zeros(F.size, dtype=F.dtype).at[idx].set(C_c).reshape(F.shape)
    return nx, ny, nz, C


# Strang: y/2 -> z -> y/2 (rotation is in y-z plane)
@jax.jit
def step2_clip(F):
    F = halo2(F)
    nx, ny, nz, C = reconstruct2(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, v_face2, dt2 / 2, DX2, DX2, DX2)
    F = halo2(jnp.clip(apply_flux_y(F, flux, DX2, DX2, DX2), 0.0, 1.0))

    nx, ny, nz, C = reconstruct2(F)
    from jax_laseram.vof.plic.geometric_flux import sweep_flux_z, apply_flux_z
    flux = sweep_flux_z(F, nx, ny, nz, C, w_face2, dt2, DX2, DX2, DX2)
    F = halo2(jnp.clip(apply_flux_z(F, flux, DX2, DX2, DX2), 0.0, 1.0))

    nx, ny, nz, C = reconstruct2(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, v_face2, dt2 / 2, DX2, DX2, DX2)
    F = halo2(jnp.clip(apply_flux_y(F, flux, DX2, DX2, DX2), 0.0, 1.0))
    return F


@jax.jit
def step2_cons(F):
    from jax_laseram.vof.plic.geometric_flux import (
        sweep_flux_z, apply_flux_z, apply_flux_z_conservative,
    )
    F = halo2(F)
    nx, ny, nz, C = reconstruct2(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, v_face2, dt2 / 2, DX2, DX2, DX2)
    F = apply_flux_y(F, flux, DX2, DX2, DX2)
    F = halo2(apply_flux_y_conservative(F, flux, dt2 / 2, DX2, v_face2, n_iter=3))

    nx, ny, nz, C = reconstruct2(F)
    flux = sweep_flux_z(F, nx, ny, nz, C, w_face2, dt2, DX2, DX2, DX2)
    F = apply_flux_z(F, flux, DX2, DX2, DX2)
    F = halo2(apply_flux_z_conservative(F, flux, dt2, DX2, w_face2, n_iter=3))

    nx, ny, nz, C = reconstruct2(F)
    flux = sweep_flux_y(F, nx, ny, nz, C, v_face2, dt2 / 2, DX2, DX2, DX2)
    F = apply_flux_y(F, flux, DX2, DX2, DX2)
    F = halo2(apply_flux_y_conservative(F, flux, dt2 / 2, DX2, v_face2, n_iter=3))
    return F


V0_2 = float(F_ini2[NH:-NH, NH:-NH, NH:-NH].sum()) * (DX2 ** 3)
print(f"  V0 = {V0_2:.8e}")

print("\n  JIT warmup...")
step2_clip(F_ini2).block_until_ready()
step2_cons(F_ini2).block_until_ready()


def run2(step_fn, label):
    F = F_ini2
    # 3 warm-ups for stable cuBLAS autotune
    for _ in range(3):
        F = step_fn(F); F.block_until_ready()
    F = F_ini2
    step_fn(F).block_until_ready()
    t0 = time.time()
    for _ in range(N_STEPS2):
        F = step_fn(F)
    F.block_until_ready()
    wall = time.time() - t0
    V = float(F[NH:-NH, NH:-NH, NH:-NH].sum()) * (DX2 ** 3)
    print(
        f"  {label:12s}  "
        f"wall={wall * 1000 / N_STEPS2:.2f} ms/step  "
        f"V={V:.8e}  "
        f"|dV/V|={abs(V - V0_2)/V0_2:.3e}  "
        f"F in [{float(F.min()):+.3e}, {float(F.max()):.6f}]"
    )
    return F


F_c = run2(step2_clip, "clip")
F_k = run2(step2_cons, "conservative")
V_clip2 = float(F_c[NH:-NH, NH:-NH, NH:-NH].sum()) * (DX2 ** 3)
V_cons2 = float(F_k[NH:-NH, NH:-NH, NH:-NH].sum()) * (DX2 ** 3)
print(f"\n  V drift ratio (clip/cons): "
      f"{abs(V_clip2-V0_2)/max(abs(V_cons2-V0_2), 1e-20):.1f}x")
