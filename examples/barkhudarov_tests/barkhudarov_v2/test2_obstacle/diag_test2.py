#!/usr/bin/env python3
"""Diagnostic: trace volume at every pipeline stage for Test 2.

Runs 20 steps covering pre-contact + contact + post-contact.
Identifies exactly WHERE volume is lost during obstacle interaction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

import numpy as np
import jax
import jax.numpy as jnp
from jax_laseram.data_types import GridInfo

# ── Config (same as test2) ─────────────────────────────────────
NX, NY, NZ = 50, 50, 5
DX = 0.02
DOMAIN_XY, DOMAIN_Z = 1.0, NZ * DX
OBS_X, OBS_Y, OBS_Z = 0.5, 0.5, DOMAIN_Z
DROP_CX, DROP_CY, DROP_CZ = 0.70, 0.70, 0.05
DROP_R = 0.20
U_INF, V_INF = -1.0, -1.0
CFL = 0.45
dt = CFL * DX / abs(U_INF)

grid = GridInfo(nx=NX, ny=NY, nz=NZ, nh=1,
                dx=DX, dy=DX, dz=DX,
                x_range=(0.0, DOMAIN_XY),
                y_range=(0.0, DOMAIN_XY),
                z_range=(0.0, DOMAIN_Z))
nh = grid.nh
cell_vol = DX**3

# ── Init F ─────────────────────────────────────────────────────
x = np.linspace(DX/2, DOMAIN_XY - DX/2, NX)
y = np.linspace(DX/2, DOMAIN_XY - DX/2, NY)
z = np.linspace(DX/2, DOMAIN_Z - DX/2, NZ)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
dist = np.sqrt((X - DROP_CX)**2 + (Y - DROP_CY)**2 + (Z - DROP_CZ)**2)
hd = np.sqrt(3) * DX / 2
F_int = np.where(dist + hd <= DROP_R, 1.0,
         np.where(dist - hd >= DROP_R, 0.0,
                  np.clip((DROP_R - dist + hd) / (2*hd), 0, 1)))
obs_mask_np = (X < OBS_X) & (Y < OBS_Y)
F_int[obs_mask_np] = 0.0

shape = (NX + 2, NY + 2, NZ + 2, 1)
F = np.zeros(shape, dtype=np.float32)
F[1:-1, 1:-1, 1:-1, 0] = F_int
for ax in range(3):
    sl = [slice(None)]*4
    sl[ax] = 0; sl2 = list(sl); sl2[ax] = 1
    F[tuple(sl)] = F[tuple(sl2)]
    sl[ax] = -1; sl2[ax] = -2
    F[tuple(sl)] = F[tuple(sl2)]
F = jnp.array(F, dtype=jnp.float32)

# ── Velocity (uniform + zeroed obstacle faces) ─────────────────
Ntx, Nty, Ntz = NX + 2, NY + 2, NZ + 2
obs_nx_tot = int(round(OBS_X / DX)) + nh  # 26
obs_ny_tot = int(round(OBS_Y / DX)) + nh

u_np = np.full((Ntx - 1, Nty, Ntz), U_INF, dtype=np.float32)
v_np = np.full((Ntx, Nty - 1, Ntz), V_INF, dtype=np.float32)
w_np = np.zeros((Ntx, Nty, Ntz - 1), dtype=np.float32)

obs_2d = np.zeros((Ntx, Nty), dtype=bool)
obs_2d[:obs_nx_tot, :obs_ny_tot] = True
for k in range(Ntz):
    u_np[obs_2d[:-1, :] | obs_2d[1:, :], k] = 0.0
for k in range(Ntz):
    v_np[(obs_2d[:, :-1] | obs_2d[:, 1:]), k] = 0.0

u_face = jnp.array(u_np)
v_face = jnp.array(v_np)
w_face = jnp.array(w_np)

# ── Import pipeline components ─────────────────────────────────
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import (
    compute_plic_normals_3d, compute_intercept_C_3d)
from jax_laseram.vof.lagrangian_3d.move_3d import (
    lagrangian_move_faces_3d, build_deformed_hexahedra)
from jax_laseram.vof.lagrangian_3d.overlay_native import (
    overlay_lagrangian_3d_native, is_native_available)
from jax_laseram.vof.lagrangian_3d.overlay_batched import (
    overlay_lagrangian_3d_batched)

# ── Interior obs mask ──────────────────────────────────────────
x_cc = np.arange(NX) * DX + DX/2
y_cc = np.arange(NY) * DX + DX/2
obs_int = (x_cc[:, None, None] < OBS_X) & (y_cc[None, :, None] < OBS_Y)
obs_int_3d = np.broadcast_to(obs_int, (NX, NY, NZ))

V0 = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
print(f"V0 = {V0:.10f}")
print(f"Obstacle interior cells: {np.sum(obs_int_3d)}")
print(f"dt = {dt:.6f}, CFL = {CFL}")
print()

# ── Run 20 diagnostic steps (covers pre-contact + contact) ─────
for step in range(20):
    print(f"{'='*70}")
    print(f"  STEP {step + 1}")
    print(f"{'='*70}")

    F_int_before = np.array(F[1:-1, 1:-1, 1:-1, 0])
    V_before = float(np.sum(F_int_before) * cell_vol)
    print(f"  [INPUT] total V = {V_before:.10f}")
    print(f"          F in obs cells = {float(np.sum(F_int_before[obs_int_3d])) * cell_vol:.2e}")
    print(f"          max F = {float(np.max(F_int_before)):.6f}")
    print(f"          cells F>1: {int(np.sum(F_int_before > 1.0))}")
    print(f"          cells F>0: {int(np.sum(F_int_before > 1e-6))}")

    # Step 1: PLIC
    nx_f, ny_f, nz_f = compute_plic_normals_3d(F, DX, DX, DX)
    raw_gx = jnp.zeros_like(F)
    raw_gy = jnp.zeros_like(F)
    raw_gz = jnp.zeros_like(F)
    raw_gx = raw_gx.at[1:-1].set((F[2:] - F[:-2]) / (2*DX))
    raw_gy = raw_gy.at[:, 1:-1].set((F[:, 2:] - F[:, :-2]) / (2*DX))
    raw_gz = raw_gz.at[:, :, 1:-1].set((F[:, :, 2:] - F[:, :, :-2]) / (2*DX))
    raw_grad = jnp.sqrt(raw_gx**2 + raw_gy**2 + raw_gz**2)
    interface_mask = raw_grad > 0.5
    nx_f = jnp.where(interface_mask, nx_f, 0.0)
    ny_f = jnp.where(interface_mask, ny_f, 0.0)
    nz_f = jnp.where(interface_mask, nz_f, 0.0)
    C_f = compute_intercept_C_3d(F, nx_f, ny_f, nz_f, DX, DX, DX)

    # Step 2: Move faces
    x_v, y_v, z_v = lagrangian_move_faces_3d(u_face, v_face, w_face, grid, dt)

    # Step 3: Clamp
    xv = jnp.clip(x_v, 0.0, DOMAIN_XY)
    yv = jnp.clip(y_v, 0.0, DOMAIN_XY)
    zv = jnp.clip(z_v, 0.0, DOMAIN_Z)

    x0g = -DX + jnp.arange(Ntx + 1) * DX
    y0g = -DX + jnp.arange(Nty + 1) * DX
    z0g = -DX + jnp.arange(Ntz + 1) * DX
    xv0, yv0, zv0 = jnp.meshgrid(x0g, y0g, z0g, indexing='ij')
    eps = 1e-6

    push_x = ((xv0 >= OBS_X - eps) & (xv < OBS_X) &
              (yv < OBS_Y) & (zv < OBS_Z))
    xv = jnp.where(push_x, OBS_X, xv)
    push_y = ((yv0 >= OBS_Y - eps) & (yv < OBS_Y) &
              (xv <= OBS_X + eps) & (zv < OBS_Z))
    yv = jnp.where(push_y, OBS_Y, yv)

    n_push_x = int(jnp.sum(push_x))
    n_push_y = int(jnp.sum(push_y))
    print(f"  [CLAMP] vertices pushed: x={n_push_x}, y={n_push_y}")

    # Hex compression diagnostic
    hex_v = build_deformed_hexahedra(xv, yv, zv, grid)
    e1 = hex_v[..., 1, :] - hex_v[..., 0, :]
    e2 = hex_v[..., 3, :] - hex_v[..., 0, :]
    e3 = hex_v[..., 4, :] - hex_v[..., 0, :]
    V_hex = np.abs(np.array(
        e1[...,0]*(e2[...,1]*e3[...,2] - e2[...,2]*e3[...,1])
      - e1[...,1]*(e2[...,0]*e3[...,2] - e2[...,2]*e3[...,0])
      + e1[...,2]*(e2[...,0]*e3[...,1] - e2[...,1]*e3[...,0])))
    ratio = V_hex / cell_vol
    # Only check cells with F > 0
    F_int_cur = np.array(F[1:-1, 1:-1, 1:-1, 0])
    active = F_int_cur > 1e-6
    if np.any(active):
        min_ratio = float(np.min(ratio[active]))
        n_compressed = int(np.sum((ratio[active] < 0.5)))
        n_tiny = int(np.sum((ratio[active] < 0.01)))
        print(f"  [HEX]   min V_hex/V_cell = {min_ratio:.6f} "
              f"(<0.5: {n_compressed}, <0.01: {n_tiny})")

    # Step 4: Overlay
    F_new = overlay_lagrangian_3d_batched(
        F, xv, yv, zv, grid,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
        obs_mask=None,  # let fluid go to obstacle cells
    )
    # overlay_batched now returns RAW un-clipped values
    F_new_np = np.array(F_new)

    V_overlay = float(np.sum(F_new_np) * cell_vol)
    V_obs_overlay = float(np.sum(F_new_np[obs_int_3d]) * cell_vol)
    n_gt1 = int(np.sum(F_new_np > 1.0 + 1e-6))
    V_overfill = float(np.sum(np.maximum(F_new_np - 1.0, 0.0)) * cell_vol)
    max_F = float(np.max(F_new_np))
    min_F = float(np.min(F_new_np))

    print(f"  [OVERLAY] total V = {V_overlay:.10f}  ΔV = {(V_overlay - V_before)/V0*100:+.4f}%")
    print(f"            F in obs = {V_obs_overlay:.2e}")
    print(f"            max F = {max_F:.4f}, min F = {min_F:.6f}")
    print(f"            cells F>1 = {n_gt1}, V_overfill = {V_overfill:.2e}")
    print(f"            (RAW un-clipped values)")

    # Step 5: Redistribution
    obs_ix = int(round(OBS_X / DX))  # 25
    obs_iy = int(round(OBS_Y / DX))  # 25
    obs_iz = NZ

    F_redist = F_new_np.copy()
    V_obs_before_redist = float(np.sum(F_redist[obs_int_3d]) * cell_vol)
    if obs_ix < NX:
        F_redist[obs_ix, :obs_iy, :obs_iz] += F_redist[obs_ix-1, :obs_iy, :obs_iz]
    if obs_iy < NY:
        F_redist[:obs_ix, obs_iy, :obs_iz] += F_redist[:obs_ix, obs_iy-1, :obs_iz]

    V_after_redist = float(np.sum(F_redist) * cell_vol)
    V_obs_after_redist = float(np.sum(F_redist[obs_int_3d]) * cell_vol)
    print(f"  [REDIST] total V = {V_after_redist:.10f}")
    print(f"           obs fluid before redist = {V_obs_before_redist:.2e}")
    print(f"           obs fluid after redist  = {V_obs_after_redist:.2e}")

    # Step 6: Zero obstacle
    F_redist[obs_int_3d] = 0.0
    V_after_zero = float(np.sum(F_redist) * cell_vol)
    V_zeroed = V_after_redist - V_after_zero
    print(f"  [ZERO]   total V = {V_after_zero:.10f}  V zeroed = {V_zeroed:.2e}")

    # Step 7: Clip to [0,1] (in case redistribution caused F>1)
    n_gt1_final = int(np.sum(F_redist > 1.0 + 1e-6))
    V_clip = float(np.sum(np.maximum(F_redist - 1.0, 0.0)) * cell_vol)
    F_redist = np.maximum(F_redist, 0.0)  # floor at 0, allow F>1
    V_final = float(np.sum(F_redist) * cell_vol)
    print(f"  [CLIP]   cells F>1 = {n_gt1_final}, V clipped = {V_clip:.2e}")
    print(f"  [FINAL]  total V = {V_final:.10f}  ΔV/V0 = {(V_final - V0)/V0*100:+.4f}%")
    print()

    # Pack into F for next step
    F_out = np.zeros(shape, dtype=np.float32)
    F_out[1:-1, 1:-1, 1:-1, 0] = F_redist
    F_out[0] = F_out[1]; F_out[-1] = F_out[-2]
    F_out[:, 0] = F_out[:, 1]; F_out[:, -1] = F_out[:, -2]
    F_out[:, :, 0] = F_out[:, :, 1]; F_out[:, :, -1] = F_out[:, :, -2]
    F = jnp.array(F_out, dtype=jnp.float32)

print("\n" + "="*70)
print("  SUMMARY")
print("="*70)
V_end = float(np.sum(np.array(F[1:-1, 1:-1, 1:-1, 0])) * cell_vol)
print(f"  V0 = {V0:.10f}")
print(f"  V20 = {V_end:.10f}")
print(f"  ΔV/V0 = {(V_end - V0)/V0*100:+.4f}%")
