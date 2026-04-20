#!/usr/bin/env python3
"""T2-3D: Zalesak Slotted Sphere — TRUE 3D Eulerian PLIC, 128^3 ≈ 2.1M cells.

Sphere R=0.15 centered at (0.5, 0.5, 0.75), slot width 0.05 in x&y,
depth 0.25 in z. Solid-body rotation around x-axis at (*, 0.5, 0.5).
One full revolution T = 2π s.

Per-stage GPU profiling of every PLIC sub-step.
"""

from __future__ import annotations
import os, sys, time as timer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src")))

import jax; import jax.numpy as jnp; import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from jax_laseram.vof.plic.normal_youngs import compute_youngs_normal_3d
from jax_laseram.vof.plic.analytic_intercept import analytic_intercept
from jax_laseram.vof.plic.geometric_flux import (
    sweep_flux_x, apply_flux_x, sweep_flux_y, apply_flux_y,
    sweep_flux_z, apply_flux_z,
)
from jax_laseram.vof.plic.diagnostics import compute_stats, format_stats

# ================================================================
# Parameters
# ================================================================
N = 128  # 128^3 ≈ 2.1M cells
NH = 1
DX = DY = DZ = 1.0 / N

SPHERE_CENTER = (0.50, 0.50, 0.75)
SPHERE_R = 0.15
SLOT_W = 0.05  # width in x and y
SLOT_D = 0.25  # depth in z (from top of sphere downward)

OMEGA = 1.0
ROT_CENTER_Y = 0.50
ROT_CENTER_Z = 0.50
T_END = 2.0 * np.pi
CFL = 0.45
N_PROFILE_STEPS = 5  # steps for per-stage profiling

# ================================================================
# Init: sphere with slot, 3x3x3 sub-sampling
# ================================================================
def create_slotted_sphere():
    """JAX-vectorized 4x4x4 sub-cell sampling (replaces 112s Python loop)."""
    n_sub = 4
    c = jnp.linspace(DX / 2, 1 - DX / 2, N, dtype=jnp.float32)
    sub_off = (jnp.arange(n_sub, dtype=jnp.float32) + 0.5) / n_sub

    # Sub-sample positions: xs[cell_i, sub_si] = cell_center - dx/2 + dx*sub_off
    xs = c[:, None] - DX / 2 + DX * sub_off[None, :]  # (N, n_sub)

    # Broadcast to 6D: (Nx, sx, Ny, sy, Nz, sz) — all sub-sample combos
    Xs = xs[:, :, None, None, None, None]   # x cell + sub
    Ys = xs[None, None, :, :, None, None]   # y cell + sub
    Zs = xs[None, None, None, None, :, :]   # z cell + sub

    cx, cy, cz = SPHERE_CENTER
    r = jnp.sqrt((Xs - cx) ** 2 + (Ys - cy) ** 2 + (Zs - cz) ** 2)
    in_sphere = r <= SPHERE_R
    in_slot = (
        (jnp.abs(Xs - cx) < SLOT_W / 2)
        & (jnp.abs(Ys - cy) < SLOT_W / 2)
        & (Zs > cz + SPHERE_R - SLOT_D)
        & (Zs < cz + SPHERE_R)
    )
    F_sub = (in_sphere & ~in_slot).astype(jnp.float32)
    F_int = F_sub.mean(axis=(1, 3, 5))  # average over sub axes → (N, N, N)

    Nt = N + 2 * NH
    F_full = jnp.zeros((Nt, Nt, Nt), dtype=jnp.float32)
    F_full = F_full.at[NH : NH + N, NH : NH + N, NH : NH + N].set(F_int)
    return halo(F_full)

def halo(F):
    F=F.at[0].set(F[1]);F=F.at[-1].set(F[-2])
    F=F.at[:,0].set(F[:,1]);F=F.at[:,-1].set(F[:,-2])
    F=F.at[:,:,0].set(F[:,:,1]);F=F.at[:,:,-1].set(F[:,:,-2])
    return F

# ================================================================
# Rotation velocity: around x-axis at (*, yc, zc)
# u=0, v=-ω(z-zc), w=ω(y-yc)
# ================================================================
def compute_rotation_vel():
    Nt = N + 2*NH
    cc = jnp.linspace(-NH*DX+DX/2, 1+NH*DX-DX/2, Nt, dtype=jnp.float32)

    # u_face = 0 everywhere
    u_face = jnp.zeros((Nt-1, Nt, Nt), dtype=jnp.float32)

    # v_face at y-faces: v = -ω(z - zc)
    z_cc = cc
    _, _, Z_vf = jnp.meshgrid(cc, 0.5*(cc[:-1]+cc[1:]), z_cc, indexing='ij')
    v_face = (-OMEGA * (Z_vf - ROT_CENTER_Z)).astype(jnp.float32)

    # w_face at z-faces: w = ω(y - yc)
    y_cc = cc
    _, Y_wf, _ = jnp.meshgrid(cc, y_cc, 0.5*(cc[:-1]+cc[1:]), indexing='ij')
    w_face = (OMEGA * (Y_wf - ROT_CENTER_Y)).astype(jnp.float32)

    return u_face, v_face, w_face

# ================================================================
# 3D Strang: y/2 → z → y/2 (rotation is in y-z plane)
# ================================================================
def plic_step_3d(F, u_face, v_face, w_face, dt):
    def sub_y(F, dt_s):
        F = halo(F)
        nx,ny,nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_y(F, nx, ny, nz, C, v_face, dt_s, DX, DY, DZ)
        F = jnp.clip(apply_flux_y(F, flux, DX, DY, DZ), 0.0, 1.0)
        return halo(F)

    def sub_z(F, dt_s):
        F = halo(F)
        nx,ny,nz = compute_youngs_normal_3d(F, DX, DY, DZ)
        C = analytic_intercept(nx, ny, nz, F, DX, DY, DZ)
        flux = sweep_flux_z(F, nx, ny, nz, C, w_face, dt_s, DX, DY, DZ)
        F = jnp.clip(apply_flux_z(F, flux, DX, DY, DZ), 0.0, 1.0)
        return halo(F)

    F = sub_y(F, dt/2)
    F = sub_z(F, dt)
    F = sub_y(F, dt/2)
    return F

# ================================================================
# Per-stage profiling
# ================================================================
def profile_stages(F, v_face, w_face, dt):
    """Profile each PLIC sub-step individually."""

    @jax.jit
    def jit_halo(F): return halo(F)
    @jax.jit
    def jit_normal(F): return compute_youngs_normal_3d(F, DX, DY, DZ)
    @jax.jit
    def jit_intercept(nx,ny,nz,F): return analytic_intercept(nx,ny,nz,F,DX,DY,DZ)
    @jax.jit
    def jit_flux_y(F,nx,ny,nz,C): return sweep_flux_y(F,nx,ny,nz,C,v_face,dt,DX,DY,DZ)
    @jax.jit
    def jit_flux_z(F,nx,ny,nz,C): return sweep_flux_z(F,nx,ny,nz,C,w_face,dt,DX,DY,DZ)
    @jax.jit
    def jit_apply_y(F,fl): return apply_flux_y(F,fl,DX,DY,DZ)
    @jax.jit
    def jit_apply_z(F,fl): return apply_flux_z(F,fl,DX,DY,DZ)
    @jax.jit
    def jit_full(F): return plic_step_3d(F, jnp.zeros_like(v_face), v_face, w_face, dt)

    # Warmup all
    Fh = jit_halo(F); jax.block_until_ready(Fh)
    nx,ny,nz = jit_normal(Fh); jax.block_until_ready(nx)
    C = jit_intercept(nx,ny,nz,Fh); jax.block_until_ready(C)
    fl = jit_flux_y(Fh,nx,ny,nz,C); jax.block_until_ready(fl)
    fl_z = jit_flux_z(Fh,nx,ny,nz,C); jax.block_until_ready(fl_z)
    Fa = jit_apply_y(Fh,fl); jax.block_until_ready(Fa)
    Ff = jit_full(F); jax.block_until_ready(Ff)

    def bench(name, fn, reps=N_PROFILE_STEPS):
        ts=[]
        for _ in range(reps):
            t0=timer.time(); r=fn(); jax.block_until_ready(r); ts.append((timer.time()-t0)*1000)
        t=np.array(ts)
        return name, np.median(t), t.min(), t.max(), t.std()

    results = [
        bench("halo_update", lambda: jit_halo(F)),
        bench("normal (Youngs 3x3x3)", lambda: jit_normal(Fh)),
        bench("intercept (analytic+Newton)", lambda: jit_intercept(nx,ny,nz,Fh)),
        bench("sweep_flux_y (array vel)", lambda: jit_flux_y(Fh,nx,ny,nz,C)),
        bench("sweep_flux_z (array vel)", lambda: jit_flux_z(Fh,nx,ny,nz,C)),
        bench("apply_flux_y", lambda: jit_apply_y(Fh,fl)),
        bench("apply_flux_z", lambda: jit_apply_z(Fh,fl_z)),
        bench("FULL STRANG STEP (y/2→z→y/2)", lambda: jit_full(F)),
    ]
    return results

# ================================================================
# Main
# ================================================================
def main():
    print("=" * 65)
    print("T2-3D: Zalesak Slotted Sphere — TRUE 3D Eulerian PLIC")
    print("=" * 65)
    n_cells = N**3
    print(f"Grid: {N}³ = {n_cells:,} cells ({n_cells/1e6:.1f}M), dx={DX:.6f}")
    print(f"Sphere: R={SPHERE_R}, center={SPHERE_CENTER}")
    print(f"Slot: {SLOT_W}x{SLOT_W}x{SLOT_D}")
    print(f"Rotation: ω={OMEGA} around x-axis at y={ROT_CENTER_Y}, z={ROT_CENTER_Z}")
    print(f"Device: {jax.devices()[0]}", flush=True)

    # Init
    print(f"\nInitializing slotted sphere (3x3x3 sub-sampling, {N}³)...", flush=True)
    t0 = timer.time()
    F_init = create_slotted_sphere()
    t_init = timer.time() - t0
    print(f"  Init time: {t_init:.1f}s")
    print(f"  F shape: {F_init.shape}, dtype: {F_init.dtype}")
    print(f"  F memory: {F_init.nbytes/1024/1024:.1f} MB")

    cell_vol = DX*DY*DZ
    stats0 = compute_stats(F_init, NH, cell_vol)
    print(f"  V0 = {stats0.volume:.6e}, interface cells = {stats0.n_interface:,}", flush=True)

    # Velocity
    u_face, v_face, w_face = compute_rotation_vel()
    max_vel = float(jnp.maximum(jnp.abs(v_face).max(), jnp.abs(w_face).max()))
    dt = CFL * DX / max_vel
    n_steps = int(np.ceil(T_END / dt))
    dt = T_END / n_steps
    print(f"\n  max|vel| = {max_vel:.4f}, dt = {dt:.6e}, n_steps = {n_steps}")

    # ================================================================
    # Per-stage profiling
    # ================================================================
    print(f"\n{'='*65}")
    print(f"PER-STAGE GPU PROFILING ({N_PROFILE_STEPS} reps each)")
    print(f"{'='*65}")

    results = profile_stages(F_init, v_face, w_face, dt)
    print(f"\n  {'Stage':40s}  {'median':>8s}  {'min':>8s}  {'max':>8s}  (ms)")
    print("  " + "-" * 70)
    for name, med, mn, mx, sd in results:
        print(f"  {name:40s}  {med:8.1f}  {mn:8.1f}  {mx:8.1f}")

    full_med = results[-1][1]
    print(f"\n  Throughput: {n_cells/(full_med/1000)/1e6:.2f} Mcells/s")

    # ================================================================
    # Full rotation (or subset for time)
    # ================================================================
    # For 2.1M cells at ~200ms/step, 900 steps = ~180s. Run full rotation.
    print(f"\n{'='*65}")
    print(f"FULL ROTATION ({n_steps} steps)")
    print(f"{'='*65}", flush=True)

    jit_step = jax.jit(lambda F: plic_step_3d(F, u_face, v_face, w_face, dt))
    # Warmup
    t0=timer.time(); Fw=jit_step(F_init); jax.block_until_ready(Fw)
    print(f"  JIT compile: {timer.time()-t0:.1f}s", flush=True)

    F = F_init
    t_wall0 = timer.time()
    save_every = max(1, n_steps // 5)
    V0 = stats0.volume

    for step in range(1, n_steps + 1):
        F = jit_step(F)
        if step % save_every == 0 or step == n_steps:
            jax.block_until_ready(F)
            stats = compute_stats(F, NH, cell_vol)
            elapsed = timer.time() - t_wall0
            print(f"  step {step:5d}/{n_steps}  {format_stats(stats, v_ref=V0)}  wall={elapsed:.1f}s", flush=True)

    wall = timer.time() - t_wall0

    # Save final F for post-processing
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez_compressed(
        os.path.join(_HERE, "results", "zalesak_3d_final.npz"),
        F_final=np.asarray(F[NH:NH+N, NH:NH+N, NH:NH+N]),
        F_init=np.asarray(F_init[NH:NH+N, NH:NH+N, NH:NH+N]),
        N=N, DX=DX, n_steps=n_steps, wall_time=wall,
    )

    # ================================================================
    # Metrics
    # ================================================================
    F_fin = np.asarray(F)
    F_ini = np.asarray(F_init)
    i_s = slice(NH, NH+N)
    F_fin_i = F_fin[i_s, i_s, i_s]
    F_ini_i = F_ini[i_s, i_s, i_s]

    L1 = float(np.sum(np.abs(F_fin_i - F_ini_i)) * cell_vol)
    V_disk = float(np.sum(F_ini_i) * cell_vol)
    L1_rel = L1 / V_disk * 100
    V_fin = float(np.sum(F_fin_i) * cell_vol)
    V_err = abs(V_fin - V_disk) / V_disk * 100

    mem = jax.devices()[0].memory_stats()
    peak_mb = mem.get('peak_bytes_in_use', 0)/1024/1024 if mem else 0

    print(f"\n{'='*65}")
    print(f"  VOF Test Report: T2-3D Zalesak Slotted Sphere")
    print(f"{'='*65}")
    print(f"  Grid:       {N}³ = {n_cells:,} cells ({n_cells/1e6:.1f}M)")
    print(f"  Method:     Eulerian PLIC (Phase B analytic + autodiff Newton)")
    print(f"  Hardware:   {jax.devices()[0]}")
    print(f"  Wall time:  {wall:.1f}s ({wall/n_steps*1000:.1f} ms/step)")
    print(f"  Throughput: {n_cells/(wall/n_steps)/1e6:.2f} Mcells/s")
    print(f"  GPU memory: {peak_mb:.0f} MB peak")
    print(f"\n  METRICS:")
    print(f"    L1 error:       {L1_rel:.2f}%    {'PASS' if L1_rel<10 else 'FAIL'} (<10%)")
    print(f"    Volume error:   {V_err:.6f}%  {'PASS' if V_err<1 else 'FAIL'} (<1%)")
    verdict = "EXCELLENT" if L1_rel<3 else ("PASS" if L1_rel<10 else "FAIL")
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*65}")

    # ================================================================
    # Plot: 3 slices × 2 rows (initial + final), binary F>0.5
    # ================================================================
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    x = np.linspace(DX / 2, 1 - DX / 2, N)
    X2, Y2 = np.meshgrid(x, x, indexing="ij")

    z_idx = int(SPHERE_CENTER[2] / DZ)  # z = 0.75
    x_idx = N // 2                       # x = 0.50

    # Coordinate grids for each slice
    xy_X, xy_Y = np.meshgrid(x, x, indexing="ij")  # for z-slice: axes = x, y
    yz_Y, yz_Z = np.meshgrid(x, x, indexing="ij")  # for x-slice: axes = y, z

    def plot_binary(ax, Xp, Yp, data, title):
        """Binary VOF rendering: F > 0.01 is blue, else white."""
        ax.pcolormesh(Xp, Yp, np.where(data > 0.01, 1.0, 0.0),
                      cmap="Blues", vmin=0, vmax=1, shading="auto")
        ax.contour(Xp, Yp, data, levels=[0.5], colors="k", linewidths=1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.15)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Initial
    plot_binary(axes[0, 0], xy_X, xy_Y, F_ini_i[:, :, z_idx],
                "Initial: z=0.75 (xy)")
    plot_binary(axes[0, 1], yz_Y, yz_Z, F_ini_i[x_idx, :, :],
                "Initial: x=0.50 (yz, rotation plane)")

    # Row 1: Final
    plot_binary(axes[1, 0], xy_X, xy_Y, F_fin_i[:, :, z_idx],
                "Final: z=0.75 (xy)")
    plot_binary(axes[1, 1], yz_Y, yz_Z, F_fin_i[x_idx, :, :],
                "Final: x=0.50 (yz, rotation plane)")

    # Overlays
    for row, (Xp, Yp, ini_s, fin_s, title) in enumerate([
        (xy_X, xy_Y, F_ini_i[:, :, z_idx], F_fin_i[:, :, z_idx],
         f"Overlay z=0.75: L1={L1_rel:.2f}%"),
        (yz_Y, yz_Z, F_ini_i[x_idx, :, :], F_fin_i[x_idx, :, :],
         "Overlay x=0.50 (rotation plane)"),
    ]):
        ax = axes[row, 2]
        ax.contour(Xp, Yp, ini_s, levels=[0.5], colors="b", linewidths=2)
        ax.contour(Xp, Yp, fin_s, levels=[0.5], colors="r", linewidths=2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Consistent limits for all panels
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0.2, 0.8)
            ax.set_ylim(0.4, 1.0)

    fig.suptitle(
        f"Zalesak 3D Slotted Sphere — {N}³ ({n_cells / 1e6:.1f}M), "
        f"L1={L1_rel:.2f}%, V_err={V_err:.6f}%, {wall / n_steps * 1000:.0f} ms/step",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(_HERE, "results", "zalesak_3d_result.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")
    return 0 if verdict != "FAIL" else 1

if __name__ == "__main__":
    sys.exit(main())
