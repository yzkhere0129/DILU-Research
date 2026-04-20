#!/usr/bin/env python3
"""Modular unit tests for 3D Lagrangian VOF overlay — find the actual bug.

Tests:
  T1: _poly_volume accuracy for known shapes at different positions
  T2: _hex_plic_box_volume: overlap sum == fluid_vol (per-donor conservation)
  T3: Single-step zero-velocity: F must be EXACTLY preserved
  T4: Single-step uniform velocity: total volume must be conserved
  T5: Per-step volume tracking for the full 112-step advection
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp

from jax_laseram.data_types import GridInfo
from jax_laseram.vof.lagrangian_3d.overlay_3d import (
    _poly_volume, _init_hex_poly, _hex_box_volume, _hex_plic_box_volume,
    _clip_poly_by_plane, _clip_poly_by_box,
)
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import (
    _volume_below_3d, compute_intercept_C_3d, compute_plic_normals_3d,
)
from jax_laseram.vof.lagrangian_3d.move_3d import build_deformed_hexahedra
from jax_laseram.vof.lagrangian_3d import advect_vof_lagrangian_3d


def make_cube_verts(x0, y0, z0, dx, dy, dz):
    """8 vertices of axis-aligned cube."""
    return jnp.array([
        [x0,    y0,    z0],
        [x0+dx, y0,    z0],
        [x0+dx, y0+dy, z0],
        [x0,    y0+dy, z0],
        [x0,    y0,    z0+dz],
        [x0+dx, y0,    z0+dz],
        [x0+dx, y0+dy, z0+dz],
        [x0,    y0+dy, z0+dz],
    ], dtype=jnp.float32)


print("=" * 70)
print("  MODULAR UNIT TESTS — Lagrangian VOF Overlay")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# T1: _poly_volume accuracy
# ═══════════════════════════════════════════════════════════════
print("\n── T1: _poly_volume accuracy ──")

for label, (x0, y0, z0) in [
    ("origin",  (0.0, 0.0, 0.0)),
    ("center",  (0.495, 0.495, 0.02)),
    ("far",     (0.895, 0.895, 0.04)),
]:
    dx = dy = dz = 0.01
    verts = make_cube_verts(x0, y0, z0, dx, dy, dz)
    poly = _init_hex_poly(verts)
    vol = float(_poly_volume(poly))
    exact = dx * dy * dz
    err = abs(vol - exact) / exact * 100
    status = "✓" if err < 0.01 else "✗"
    print(f"  {status} cube@{label}: vol={vol:.10e} exact={exact:.10e} err={err:.6f}%")

# Non-cube: stretched
verts_s = make_cube_verts(0.3, 0.4, 0.01, 0.02, 0.015, 0.01)
poly_s = _init_hex_poly(verts_s)
vol_s = float(_poly_volume(poly_s))
exact_s = 0.02 * 0.015 * 0.01
err_s = abs(vol_s - exact_s) / exact_s * 100
print(f"  {'✓' if err_s < 0.01 else '✗'} stretched: vol={vol_s:.10e} exact={exact_s:.10e} err={err_s:.6f}%")


# ═══════════════════════════════════════════════════════════════
# T2: PLIC clip + box clip conservation (per-donor)
# ═══════════════════════════════════════════════════════════════
print("\n── T2: PLIC donor conservation (sum overlap_i == fluid_vol) ──")

dx = dy = dz = 0.01

for F_val, n_vec, label in [
    (0.5, (0.0, 1.0, 0.0), "F=0.5 n=(0,1,0)"),
    (0.5, (0.707, 0.707, 0.0), "F=0.5 n=(1,1,0)/√2"),
    (0.3, (1.0, 0.0, 0.0), "F=0.3 n=(1,0,0)"),
    (0.8, (0.577, 0.577, 0.577), "F=0.8 n=(1,1,1)/√3"),
    (1.0, (0.0, 0.0, 1.0), "F=1.0 full"),
    (0.0, (0.0, 0.0, 1.0), "F=0.0 empty"),
]:
    cx, cy, cz = 0.505, 0.505, 0.025  # cell center (displaced slightly)
    x0, y0, z0 = cx - dx/2, cy - dy/2, cz - dz/2
    hex_v = make_cube_verts(x0, y0, z0, dx, dy, dz)

    nx, ny, nz = n_vec
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    if mag > 0:
        nx, ny, nz = nx/mag, ny/mag, nz/mag

    # Compute C via _volume_below_3d
    a = jnp.float32(nx * dx)
    b = jnp.float32(ny * dy)
    c = jnp.float32(nz * dz)
    F_t = jnp.clip(jnp.float32(F_val), 1e-10, 1 - 1e-10)
    C_half = 0.5 * (jnp.abs(a) + jnp.abs(b) + jnp.abs(c))
    C_lo, C_hi = -C_half, C_half
    C_mid = jnp.float32(0.0)
    for _ in range(30):
        vol_f = _volume_below_3d(C_mid, a, b, c)
        C_lo = jnp.where(vol_f < F_t, C_mid, C_lo)
        C_hi = jnp.where(vol_f < F_t, C_hi, C_mid)
        C_mid = 0.5 * (C_lo + C_hi)
    C_val = float(C_mid)

    # PLIC plane: d = C + n . centroid
    plic_d = C_val + nx * cx + ny * cy + nz * cz

    # Handle full/empty
    if F_val > 1 - 1e-6:
        plic_d = 1e6
    elif F_val < 1e-6:
        plic_d = -1e6

    plic_n = jnp.array([nx, ny, nz], dtype=jnp.float32)
    plic_d_j = jnp.float32(plic_d)
    F_j = jnp.float32(F_val)

    # Compute overlap with 3x3x3 neighborhood of acceptor boxes
    # The hex sits at cell (i,j,k). Acceptor boxes are the 3x3x3 cells around it.
    total_overlap = 0.0
    fluid_vol_val = None
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                bx0 = x0 - dx/2 + di * dx  # acceptor box lower corner
                by0 = y0 - dy/2 + dj * dy
                bz0 = z0 - dz/2 + dk * dz
                # Actually: acceptor box for cell at (i+di, j+dj, k+dk)
                # The acceptor box lower corner = (cx - dx/2 + di*dx - dx/2, ...)
                # Wait, let me think more carefully.
                # The hex center is at (cx, cy, cz).
                # The "home" acceptor box is centered at (cx_home, cy_home, cz_home)
                # But actually, in the overlay, the acceptor boxes are the EULERIAN grid cells.
                # The hex comes from the DEFORMED donor cell.
                # For this test, the hex IS a cube at (x0, y0, z0) to (x0+dx, y0+dy, z0+dz).
                # The acceptor grid is aligned to the original (undeformed) grid.
                # Let me use the original grid cell positions.

                # Original grid: cell (50,50,2) has box [0.50, 0.51] x [0.50, 0.51] x [0.02, 0.03]
                # The hex is displaced to [0.500, 0.510] x [0.500, 0.510] x [0.020, 0.030]
                # Acceptor for offset (di,dj,dk): cell (50+di, 50+dj, 2+dk)
                acc_x0 = 0.50 + di * dx
                acc_y0 = 0.50 + dj * dy
                acc_z0 = 0.02 + dk * dz
                box_min = jnp.array([acc_x0, acc_y0, acc_z0], dtype=jnp.float32)
                box_max = jnp.array([acc_x0 + dx, acc_y0 + dy, acc_z0 + dz], dtype=jnp.float32)

                ov, fv = _hex_plic_box_volume(hex_v, plic_n, plic_d_j, F_j, box_min, box_max)
                total_overlap += float(ov)
                if di == 0 and dj == 0 and dk == 0:
                    fluid_vol_val = float(fv)

    # fluid_vol should be same from any call (it's the PLIC-clipped hex volume)
    if fluid_vol_val is None:
        fluid_vol_val = 0.0

    if F_val < 1e-6:
        print(f"  ✓ {label}: skip (empty)")
        continue

    ratio = total_overlap / max(fluid_vol_val, 1e-30)
    err = abs(ratio - 1.0) * 100
    status = "✓" if err < 0.1 else "✗"
    print(f"  {status} {label}: Σoverlap={total_overlap:.10e} fluid_vol={fluid_vol_val:.10e} "
          f"ratio={ratio:.8f} err={err:.4f}%")
    # Also check F * overlap / fluid_vol sum
    transfer_sum = F_val * total_overlap / max(fluid_vol_val, 1e-30)
    terr = abs(transfer_sum - F_val) / max(F_val, 1e-30) * 100
    print(f"       F*Σovl/fv={transfer_sum:.8f} (expect {F_val:.1f}) err={terr:.4f}%")


# ═══════════════════════════════════════════════════════════════
# T3: Zero-velocity preservation
# ═══════════════════════════════════════════════════════════════
print("\n── T3: Zero-velocity preservation ──")

NX, NZ = 20, 3
DOMAIN = 0.2
DX = DOMAIN / NX
grid_t3 = GridInfo(
    nx=NX, ny=NX, nz=NZ, nh=1,
    dx=DX, dy=DX, dz=DX,
    x_range=(0.0, DOMAIN), y_range=(0.0, DOMAIN), z_range=(0.0, NZ * DX),
)
nh = grid_t3.nh
Ntx = NX + 2 * nh
Nty = NX + 2 * nh
Ntz = NZ + 2 * nh

# Simple circle droplet
x = np.linspace(DX/2, DOMAIN - DX/2, NX)
y = np.linspace(DX/2, DOMAIN - DX/2, NX)
X, Y = np.meshgrid(x, y, indexing="ij")
R = 0.03
cx0, cy0 = 0.10, 0.10
dist = np.sqrt((X - cx0)**2 + (Y - cy0)**2)
hd = np.sqrt(2) * DX / 2
F_2d = np.where(dist + hd <= R, 1.0, np.where(dist - hd >= R, 0.0,
               np.clip((R - dist + hd) / (2 * hd), 0.0, 1.0)))

F_t3 = np.zeros((Ntx, Nty, Ntz, 1))
F_t3[nh:nh+NX, nh:nh+NX, nh:nh+NZ, 0] = F_2d[:, :, None]
F_t3[:nh] = F_t3[nh:nh+1]; F_t3[-nh:] = F_t3[-nh-1:-nh]
F_t3[:, :nh] = F_t3[:, nh:nh+1]; F_t3[:, -nh:] = F_t3[:, -nh-1:-nh]
F_t3[:, :, :nh] = F_t3[:, :, nh:nh+1]; F_t3[:, :, -nh:] = F_t3[:, :, -nh-1:-nh]
F_t3 = jnp.array(F_t3, dtype=jnp.float32)

u_zero = jnp.zeros((Ntx-1, Nty, Ntz), jnp.float32)
v_zero = jnp.zeros((Ntx, Nty-1, Ntz), jnp.float32)
w_zero = jnp.zeros((Ntx, Nty, Ntz-1), jnp.float32)

V_before = float(jnp.sum(F_t3[nh:nh+NX, nh:nh+NX, nh:nh+NZ, 0]) * DX**3)

F_after = advect_vof_lagrangian_3d(F_t3, u_zero, v_zero, w_zero, grid_t3, dt=0.001)
V_after = float(jnp.sum(F_after[nh:nh+NX, nh:nh+NX, nh:nh+NZ, 0]) * DX**3)

F_diff = jnp.max(jnp.abs(F_after - F_t3))
dV_t3 = abs(V_after - V_before) / V_before * 100

print(f"  V_before={V_before:.10e}  V_after={V_after:.10e}")
print(f"  |ΔV/V| = {dV_t3:.6f}%  {'✓' if dV_t3 < 0.001 else '✗ FAIL'}")
print(f"  max|F_after - F_before| = {float(F_diff):.2e}  {'✓' if float(F_diff) < 1e-6 else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════
# T4: Single-step conservation with uniform velocity
# ═══════════════════════════════════════════════════════════════
print("\n── T4: Single-step conservation (uniform velocity, 1 step) ──")

D = 0.10; R_d = D/2; NX4 = 100; NZ4 = 5
DOMAIN4 = 1.0; DX4 = DOMAIN4/NX4; SPEED = 1.0
ANGLE = 45
grid_t4 = GridInfo(
    nx=NX4, ny=NX4, nz=NZ4, nh=1,
    dx=DX4, dy=DX4, dz=DX4,
    x_range=(0.0, DOMAIN4), y_range=(0.0, DOMAIN4), z_range=(0.0, NZ4*DX4),
)
nh4 = grid_t4.nh
Ntx4 = NX4+2*nh4; Nty4=NX4+2*nh4; Ntz4=NZ4+2*nh4

# Droplet
x4 = np.linspace(DX4/2, DOMAIN4-DX4/2, NX4)
y4 = np.linspace(DX4/2, DOMAIN4-DX4/2, NX4)
X4, Y4 = np.meshgrid(x4, y4, indexing="ij")
dist4 = np.sqrt((X4-0.20)**2 + (Y4-0.20)**2)
hd4 = np.sqrt(2)*DX4/2
F_2d4 = np.where(dist4+hd4<=R_d, 1.0, np.where(dist4-hd4>=R_d, 0.0,
                np.clip((R_d-dist4+hd4)/(2*hd4), 0.0, 1.0)))

F_t4 = np.zeros((Ntx4, Nty4, Ntz4, 1))
F_t4[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0] = F_2d4[:,:,None]
F_t4[:nh4]=F_t4[nh4:nh4+1]; F_t4[-nh4:]=F_t4[-nh4-1:-nh4]
F_t4[:,:nh4]=F_t4[:,nh4:nh4+1]; F_t4[:,-nh4:]=F_t4[:,-nh4-1:-nh4]
F_t4[:,:,:nh4]=F_t4[:,:,nh4:nh4+1]; F_t4[:,:,-nh4:]=F_t4[:,:,-nh4-1:-nh4]
F_t4 = jnp.array(F_t4, dtype=jnp.float32)

theta = np.radians(ANGLE)
ux = np.cos(theta)*SPEED; uy = np.sin(theta)*SPEED
u_face4 = jnp.full((Ntx4-1,Nty4,Ntz4), ux, jnp.float32)
v_face4 = jnp.full((Ntx4,Nty4-1,Ntz4), uy, jnp.float32)
w_face4 = jnp.zeros((Ntx4,Nty4,Ntz4-1), jnp.float32)

CFL = 0.45
dt4 = CFL * DX4 / SPEED

V0_t4 = float(jnp.sum(F_t4[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]) * DX4**3)

# Run 1 step
F_1step = advect_vof_lagrangian_3d(F_t4, u_face4, v_face4, w_face4, grid_t4, dt4)
V1_t4 = float(jnp.sum(F_1step[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]) * DX4**3)
dV1 = (V1_t4 - V0_t4) / V0_t4 * 100

# Run 5 more steps
F_cur = F_1step
for i in range(4):
    F_cur = advect_vof_lagrangian_3d(F_cur, u_face4, v_face4, w_face4, grid_t4, dt4)
V5_t4 = float(jnp.sum(F_cur[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]) * DX4**3)
dV5 = (V5_t4 - V0_t4) / V0_t4 * 100

print(f"  V₀={V0_t4:.10e}")
print(f"  After 1 step:  V={V1_t4:.10e}  ΔV/V={dV1:+.6f}%  {'✓' if abs(dV1)<0.01 else '✗'}")
print(f"  After 5 steps: V={V5_t4:.10e}  ΔV/V={dV5:+.6f}%  {'✓' if abs(dV5)<0.05 else '✗'}")

# Check F range
F_int = F_1step[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]
print(f"  F range after 1 step: [{float(jnp.min(F_int)):.6f}, {float(jnp.max(F_int)):.6f}]")
n_over1 = int(jnp.sum(F_int > 1.0 + 1e-6))
n_under0 = int(jnp.sum(F_int < -1e-6))
print(f"  Cells with F>1: {n_over1},  F<0: {n_under0}")

# Check how much the clip removed
# To do this, we need overlay WITHOUT clip. Let's compute pre-clip volume.
# Since clip only caps at [0,1], the difference tells us how much was clipped.
F_preclip = jnp.where(F_int > 1.0, F_int, 0.0)
clip_removed = float(jnp.sum(F_preclip - jnp.clip(F_preclip, 0.0, 1.0)) * DX4**3)
# This only captures the excess above 1 that was actually clipped
print(f"  Excess above 1 (per step): {float(jnp.sum(jnp.maximum(F_int-1.0, 0.0))*DX4**3):.2e}")


# ═══════════════════════════════════════════════════════════════
# T5: Per-step volume tracking (full 112 steps)
# ═══════════════════════════════════════════════════════════════
print("\n── T5: Per-step volume tracking (112 steps) ──")

F_run = F_t4
vol_history = [V0_t4]
cell_vol = DX4**3

for step in range(112):
    F_run = advect_vof_lagrangian_3d(F_run, u_face4, v_face4, w_face4, grid_t4, dt4)
    V_step = float(jnp.sum(F_run[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]) * cell_vol)
    vol_history.append(V_step)
    if step < 5 or step % 20 == 19 or step == 111:
        dV_s = (V_step - V0_t4) / V0_t4 * 100
        # Per-step change
        dV_ps = (V_step - vol_history[-2]) / V0_t4 * 100
        F_s = F_run[nh4:nh4+NX4, nh4:nh4+NX4, nh4:nh4+NZ4, 0]
        n_over = int(jnp.sum(F_s > 1.0 + 1e-6))
        fmax = float(jnp.max(F_s))
        print(f"  step {step+1:3d}: ΔV/V={dV_s:+.4f}% (Δ/step={dV_ps:+.6f}%) "
              f"Fmax={fmax:.4f} n(F>1)={n_over}")

V_final = vol_history[-1]
dV_final = (V_final - V0_t4) / V0_t4 * 100
print(f"\n  Final: |ΔV/V| = {abs(dV_final):.4f}%  {'✓ PASS' if abs(dV_final)<0.0002 else '✗ FAIL'}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
