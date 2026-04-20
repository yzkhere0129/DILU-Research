#!/usr/bin/env python3
"""Diagonal Jet — uniform (1,1,1) velocity, no-PLIC (geometric overlay).

Clean conservative advection without PLIC to demonstrate the 3D pipeline.
PLIC integration requires higher resolution for accurate normals.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import jax, jax.numpy as jnp, numpy as np
from jax_laseram.vof.lagrangian_3d.move_3d import lagrangian_move_faces_3d
from jax_laseram.vof.lagrangian_3d.overlay_3d import overlay_lagrangian_3d
from jax_laseram.data_types import GridInfo

L = 2.0; N = 12; dx = L/N; nh = 1
R_jet = 0.2; cx, cy = 0.3, 0.3
vx = vy = vz = 1.0; v_mag = np.sqrt(3.)
CFL = 0.3; dt = CFL * dx / v_mag
t_end = 1.0; n_steps = int(t_end / dt) + 1
OUT = os.path.dirname(os.path.abspath(__file__))

grid = GridInfo(nx=N, ny=N, nz=N, nh=nh, dx=dx, dy=dx, dz=dx,
                x_range=(0.,L), y_range=(0.,L), z_range=(0.,L))
Nt = N + 2*nh
x_cc = jnp.arange(Nt)*dx + 0.5*dx - nh*dx
y_cc, z_cc = x_cc, x_cc
X, Y, Z = jnp.meshgrid(x_cc, y_cc, z_cc, indexing='ij')

u_face = jnp.full((Nt-1, Nt, Nt), vx, jnp.float32)
v_face = jnp.full((Nt, Nt-1, Nt), vy, jnp.float32)
w_face = jnp.full((Nt, Nt, Nt-1), vz, jnp.float32)
F = jnp.zeros((Nt, Nt, Nt, 1), jnp.float32)

d_hat = jnp.array([1., 1., 1.]) / jnp.sqrt(3.)
def _dist_axis(X, Y, Z):
    px, py, pz = X - cx, Y - cy, Z
    cx_ = py*d_hat[2] - pz*d_hat[1]
    cy_ = pz*d_hat[0] - px*d_hat[2]
    cz_ = px*d_hat[1] - py*d_hat[0]
    return jnp.sqrt(cx_**2 + cy_**2 + cz_**2)

r_cell = _dist_axis(X, Y, Z)
inject_mask = (r_cell < R_jet) & (Z < 1.5*dx)

def _step(F):
    xv, yv, zv = lagrangian_move_faces_3d(u_face, v_face, w_face, grid, dt)
    F_new = overlay_lagrangian_3d(F, xv, yv, zv, grid)
    F_out = jnp.zeros_like(F)
    F_out = F_out.at[nh:nh+N, nh:nh+N, nh:nh+N, 0].set(F_new)
    for ax in range(3):
        s1=[slice(None)]*4; s1[ax]=slice(0,nh)
        s2=[slice(None)]*4; s2[ax]=slice(nh,nh+1)
        F_out=F_out.at[tuple(s1)].set(F_out[tuple(s2)])
        s3=[slice(None)]*4; s3[ax]=slice(-nh,None)
        s4=[slice(None)]*4; s4[ax]=slice(-nh-1,-nh)
        F_out=F_out.at[tuple(s3)].set(F_out[tuple(s4)])
    return F_out

print(f"Diagonal jet: {N}³, v=(1,1,1), {n_steps} steps")
print("Compiling...", end=" ", flush=True)
t0 = time.time()
jit_step = jax.jit(_step)
jit_step(F).block_until_ready()
print(f"{time.time()-t0:.1f}s")

def write_vtk(fname, data, dx):
    nx_, ny_, nz_ = data.shape
    with open(fname, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nDiagonal Jet\nASCII\n')
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write(f'DIMENSIONS {nx_} {ny_} {nz_}\n')
        f.write(f'ORIGIN {0.5*dx} {0.5*dx} {0.5*dx}\n')
        f.write(f'SPACING {dx} {dx} {dx}\n')
        f.write(f'POINT_DATA {nx_*ny_*nz_}\n')
        f.write('SCALARS F float 1\nLOOKUP_TABLE default\n')
        for k in range(nz_):
            for j in range(ny_):
                for i in range(nx_):
                    f.write(f'{data[i,j,k]:.6f}\n')

snaps = sorted(set([0] + list(range(0, n_steps+1, max(1,n_steps//6))) + [n_steps]))
t_start = time.time()
for step in range(n_steps + 1):
    F = jnp.where(inject_mask[..., None], 1.0, F)
    if step in snaps:
        Fs = np.array(F[nh:nh+N, nh:nh+N, nh:nh+N, 0])
        vol = float(np.sum(Fs)) * dx**3
        fn = f'{OUT}/diag_jet_step{step:04d}.vtk'
        write_vtk(fn, Fs, dx)
        print(f"  step {step:3d}/{n_steps} t={step*dt:.3f} vol={vol:.5f} → {os.path.basename(fn)}")
    if step < n_steps:
        F = jit_step(F)
print(f"Done! {(time.time()-t_start)/60:.1f} min")
