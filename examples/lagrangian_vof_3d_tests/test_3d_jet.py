#!/usr/bin/env python3
"""Barkhudarov Fig.9 — 25D Axial Jet with PLIC, dx=0.4D.

PLIC-in-overlay should produce characteristic diagonal "rebar" bumps
at this ultra-coarse 2.5 cells/D resolution.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import jax, jax.numpy as jnp, numpy as np
from jax_laseram.vof.lagrangian_3d.move_3d import lagrangian_move_faces_3d
from jax_laseram.vof.lagrangian_3d.overlay_3d import overlay_lagrangian_3d
from jax_laseram.vof.lagrangian_3d.reconstruction_3d import compute_plic_normals_3d, compute_intercept_C_3d
from jax_laseram.data_types import GridInfo

dx = 1.0; Nx, Ny, Nz = 5, 5, 70
D_jet = 2.5; R_jet = D_jet / 2; nh = 1
Lx, Ly, Lz = Nx*dx, Ny*dx, Nz*dx
cx, cy = Lx/2, Ly/2
w_jet = 1.0; CFL = 0.3; dt = CFL * dx / w_jet
T_end = 25 * D_jet; n_steps = int(T_end / dt) + 1
OUT = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  Barkhudarov Fig.9 — 25D Jet WITH PLIC")
print("=" * 60)
print(f"  Grid: {Nx}×{Ny}×{Nz}, dx={dx}, D/dx={D_jet/dx}")
print(f"  w={w_jet}, CFL={CFL}, dt={dt}, steps={n_steps}")
print()

grid = GridInfo(nx=Nx, ny=Ny, nz=Nz, nh=nh, dx=dx, dy=dx, dz=dx,
                x_range=(0., Lx), y_range=(0., Ly), z_range=(0., Lz))
Nt_x, Nt_y, Nt_z = Nx+2*nh, Ny+2*nh, Nz+2*nh
x_cc = jnp.arange(Nt_x)*dx + 0.5*dx - nh*dx
y_cc = jnp.arange(Nt_y)*dx + 0.5*dx - nh*dx
z_cc = jnp.arange(Nt_z)*dx + 0.5*dx - nh*dx
X, Y, Z = jnp.meshgrid(x_cc, y_cc, z_cc, indexing="ij")

u_face = jnp.zeros((Nt_x-1, Nt_y, Nt_z), jnp.float32)
v_face = jnp.zeros((Nt_x, Nt_y-1, Nt_z), jnp.float32)
w_face = jnp.full((Nt_x, Nt_y, Nt_z-1), w_jet, jnp.float32)
F = jnp.zeros((Nt_x, Nt_y, Nt_z, 1), jnp.float32)
r_cell = jnp.sqrt((X - cx)**2 + (Y - cy)**2)
inject_mask = (r_cell < R_jet) & (Z < 1.5*dx)

def do_plic(F):
    nx,ny,nz = compute_plic_normals_3d(F, dx, dx, dx)
    C = compute_intercept_C_3d(F, nx, ny, nz, dx, dx, dx, n_iter=20)
    return nx, ny, nz, C

def do_step(F, nx_f, ny_f, nz_f, C_f):
    xv, yv, zv = lagrangian_move_faces_3d(u_face, v_face, w_face, grid, dt)
    F_new = overlay_lagrangian_3d(F, xv, yv, zv, grid,
                                   nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f)
    F_out = jnp.zeros_like(F)
    F_out = F_out.at[nh:nh+Nx, nh:nh+Ny, nh:nh+Nz, 0].set(F_new)
    for ax in range(3):
        s1=[slice(None)]*4; s1[ax]=slice(0,nh)
        s2=[slice(None)]*4; s2[ax]=slice(nh,nh+1)
        F_out=F_out.at[tuple(s1)].set(F_out[tuple(s2)])
        s3=[slice(None)]*4; s3[ax]=slice(-nh,None)
        s4=[slice(None)]*4; s4[ax]=slice(-nh-1,-nh)
        F_out=F_out.at[tuple(s3)].set(F_out[tuple(s4)])
    return F_out

print("Compiling...", end=" ", flush=True)
t0 = time.time()
jp = jax.jit(do_plic); jo = jax.jit(do_step)
r = jp(F); jo(F, *r).block_until_ready()
print(f"{time.time()-t0:.1f}s")

def write_vtk(fname, data, dx):
    nx_, ny_, nz_ = data.shape
    with open(fname, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nFig9 PLIC Jet\nASCII\n')
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

snap_Ds = [5, 10, 15, 20, 25]
snap_steps = sorted(set([0] + [int(d*D_jet/dt) for d in snap_Ds] + [n_steps]))

print(f"Running {n_steps} steps (25D)...")
t_start = time.time()
for step in range(n_steps + 1):
    F = jnp.where(inject_mask[..., None], 1.0, F)
    if step in snap_steps:
        Fs = np.array(F[nh:nh+Nx, nh:nh+Ny, nh:nh+Nz, 0])
        vol = float(np.sum(Fs)) * dx**3
        t_sim = step * dt
        D_travel = t_sim / D_jet
        fn = f'{OUT}/fig9_plic_{D_travel:.0f}D.vtk'
        write_vtk(fn, Fs, dx)
        print(f"  step {step:4d}/{n_steps}  {D_travel:5.1f}D  vol={vol:8.2f}  → {os.path.basename(fn)}")
    if step < n_steps:
        nx_f, ny_f, nz_f, C_f = jp(F)
        F = jo(F, nx_f, ny_f, nz_f, C_f)

print(f"\nDone! {(time.time()-t_start)/60:.1f} min")

# Isosurface
print("\nGenerating isosurface...")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes
    F_final = np.array(F[nh:nh+Nx, nh:nh+Ny, nh:nh+Nz, 0])
    if np.any(F_final > 0.3) and np.any(F_final < 0.3):
        verts, faces, _, _ = marching_cubes(F_final, level=0.3, spacing=(dx,dx,dx))
        fig = plt.figure(figsize=(6, 18))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2],
                        cmap='coolwarm', alpha=0.8, edgecolor='none')
        ax.set_xlim(0,Lx); ax.set_ylim(0,Ly); ax.set_zlim(0,Lz)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Fig.9 PLIC Jet — F=0.3, D/dx={D_jet/dx}')
        ax.view_init(elev=5, azim=-70)
        out_fig = os.path.join(OUT, 'fig9_plic_isosurface.png')
        plt.savefig(out_fig, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_fig}")
except ImportError as e:
    print(f"  Skip: {e}")
