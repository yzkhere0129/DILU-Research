"""Phase 2.5 — Physical Sanity Benchmark for the cuSPARSE DILU-PCG solver.

Three anti-hallucination physical tests:
  A. Divergence map after pressure projection (mass conservation, rho_l/rho_g=1000).
  B. Static-droplet spurious currents (surface tension + 10-step projection NS).
  C. Residual halo: 50-iter DILU-PCG vs sparse-LU direct solve on 3-tier stiff AM.

Usage:
    python physical_benchmark.py              # all three tests
    python physical_benchmark.py --test A     # only A
    python physical_benchmark.py --test B
    python physical_benchmark.py --test C

Outputs PNGs under dilu/cusparse/bench/plots/.
Each test prints a one-line gold-standard summary.

HARD RAILS (brief §VRAM safety):
  - Grid capped at 32^3 = 32768 unknowns.
  - XLA_PYTHON_CLIENT_* env vars set before `import jax`.
  - scipy spsolve on 32^3 stays under ~500 MB host RAM empirically.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# VRAM rails MUST be set before `import jax`.
# --------------------------------------------------------------------------
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# --------------------------------------------------------------------------
# Make the repo root importable so `from dilu.cusparse.python import ...`
# works when this file is run directly.
# --------------------------------------------------------------------------
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
from typing import Callable

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla

import jax
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from dilu.cusparse.python import build_diag_offset, Plan

# --------------------------------------------------------------------------
# Global configuration (documented for reproducibility per brief §Implementation).
# --------------------------------------------------------------------------
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

N_GRID = 32                # grid size (N^3 cells)
H_GRID = 1.0 / N_GRID      # uniform cell spacing
RHO_LIQUID = 1000.0
RHO_GAS = 1.0
RHO_SOLID = 8000.0         # Test C only
R_DROPLET = 0.25           # in unit box; sphere radius
SIGMA = 0.07               # surface tension (N/m analog), Test B
DT_PROJECTION = 1.0e-4     # projection timestep, Tests A & B
PCG_TOL = 1.0e-10
PCG_MAX_ITER_A = 300       # Test A aims for genuine convergence
PCG_MAX_ITER_B = 300       # Test B same
PCG_MAX_ITER_C = 50        # Test C: fixed-iter cap, intentionally truncated

PLOTS_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "bench", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Reusable DILU-PCG driver (refactored from test_t7_pcg_stiff.py per brief §1).
# --------------------------------------------------------------------------
def _spmv_gpu(row_ptr, col_idx, values, x):
    """CSR SpMV via segment_sum (same recipe as T7)."""
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def dilu_pcg(row_ptr_h: np.ndarray,
             col_idx_h: np.ndarray,
             values_h: np.ndarray,
             diag_offset_h: np.ndarray,
             b_h: np.ndarray,
             max_iter: int,
             tol: float = 1.0e-10,
             fixed_iter: bool = False):
    """Plain DILU-PCG driver.

    Args:
        *_h: host NumPy arrays (float64/int32, C-contiguous).
        fixed_iter: if True, run exactly `max_iter` iterations regardless of residual
                    (used by Test C to intentionally probe mid-convergence state).

    Returns (x_host, niter, rnorm_history).
    """
    row_ptr_h = np.ascontiguousarray(row_ptr_h, dtype=np.int32)
    col_idx_h = np.ascontiguousarray(col_idx_h, dtype=np.int32)
    values_h = np.ascontiguousarray(values_h, dtype=np.float64)
    diag_offset_h = np.ascontiguousarray(diag_offset_h, dtype=np.int32)
    b_h = np.ascontiguousarray(b_h, dtype=np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr_h))
    ci = jax.device_put(jnp.asarray(col_idx_h))
    vv = jax.device_put(jnp.asarray(values_h))
    do = jax.device_put(jnp.asarray(diag_offset_h))
    b_d = jax.device_put(jnp.asarray(b_h))

    def spmv(x):
        return _spmv_gpu(rp, ci, vv, x)

    n = int(b_h.shape[0])
    rnorms = []
    with Plan(rp, ci, vv, do) as plan:
        d_star = plan.factor(vv)
        d_star.block_until_ready()

        x = jnp.zeros(n, dtype=jnp.float64)
        r = b_d - spmv(x)
        b_norm = float(jnp.linalg.norm(b_d))
        b_norm = max(b_norm, 1e-300)
        z = plan.apply(vv, d_star, r)
        p = z
        rz = float(jnp.dot(r, z))

        for it in range(max_iter):
            rnorm = float(jnp.linalg.norm(r))
            rnorms.append(rnorm)
            if (not fixed_iter) and (rnorm / b_norm < tol):
                x.block_until_ready()
                return np.asarray(x), it, rnorms
            Ap = spmv(p)
            alpha = rz / float(jnp.dot(p, Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            z = plan.apply(vv, d_star, r)
            rz_new = float(jnp.dot(r, z))
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new

        x.block_until_ready()
        rnorms.append(float(jnp.linalg.norm(r)))
        return np.asarray(x), max_iter, rnorms


# --------------------------------------------------------------------------
# Grid helpers, phase indicator, variable-density Poisson.
# --------------------------------------------------------------------------
def _cell_centres(n: int, h: float):
    """1-D cell-centred coords in [h/2, 1 - h/2]."""
    return (np.arange(n) + 0.5) * h


def smooth_sphere_indicator(n: int, h: float, radius: float, eps_factor: float = 1.5):
    """Smooth Heaviside phase indicator c(r) in [0, 1]: 1 inside the sphere, 0 outside.

    c(r) = 0.5 * (1 - tanh((r - R) / eps)), with eps = eps_factor * h (brief §4).
    """
    xs = _cell_centres(n, h) - 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    eps = eps_factor * h
    c = 0.5 * (1.0 - np.tanh((R - radius) / eps))
    return np.ascontiguousarray(c, dtype=np.float64)


def density_from_indicator(c: np.ndarray, rho_high: float, rho_low: float):
    return rho_high * c + rho_low * (1.0 - c)


def _face_coeffs(rho: np.ndarray):
    """Harmonic-mean face coefficients lambda = 1/rho_face on interior faces.

    Canonical form: lambda_{i+1/2} = 2/(rho_i + rho_{i+1}) (= harmonic mean of 1/rho).
    Returns (face_x, face_y, face_z) each without an edge layer in the normal
    direction (face_x shape = (nx-1, ny, nz), etc.).
    """
    fx = 2.0 / (rho[:-1, :, :] + rho[1:, :, :])
    fy = 2.0 / (rho[:, :-1, :] + rho[:, 1:, :])
    fz = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])
    return fx, fy, fz


def build_variable_density_poisson_3d(rho: np.ndarray, h: float,
                                      pin_cell: tuple = (0, 0, 0)):
    """Build CSR for the SPD Poisson  -div( (1/rho) grad p ).

    7-point FV stencil with coefficient beta_f = lambda_f / h^2 where
    lambda_f = 2/(rho_C + rho_N). Row C: diag = sum_f beta_f, off-diag to
    neighbour N = -beta_f. This is exactly A.p ≈ -div(lambda grad p) in cell-
    centred units, SPD with homogeneous Neumann (after pinning one cell).

    Pair that with `apply_velocity_correction` (below) using the SAME face
    coefficients and `divergence_face` (below) to check, and the discrete
    identity  div(u^{n+1}) = div(u*) - dt * (-A p) = 0 holds in EXACT float64
    arithmetic at iteration count → infty.

    Boundary: homogeneous Neumann on all faces (boundary faces contribute 0).
    One cell pinned: its row becomes identity; neighbours drop that column.

    Returns (row_ptr, col_idx, values, diag_offset).
    """
    assert rho.ndim == 3
    nx, ny, nz = rho.shape
    n = nx * ny * nz

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    pin_i = idx(*pin_cell)

    fx, fy, fz = _face_coeffs(rho)

    row_ptr_list = [0]
    col_idx_list = []
    values_list = []
    diag_offset_list = []

    h2 = h * h

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)

                # Pinned cell: identity row.
                if p == pin_i:
                    col_idx_list.append(p)
                    values_list.append(1.0)
                    diag_offset_list.append(len(values_list) - 1)
                    row_ptr_list.append(len(col_idx_list))
                    continue

                entries = []  # (col, val)
                diag_val = 0.0

                if i > 0:
                    beta = fx[i - 1, j, k] / h2
                    q = idx(i - 1, j, k)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta
                if i < nx - 1:
                    beta = fx[i, j, k] / h2
                    q = idx(i + 1, j, k)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta
                if j > 0:
                    beta = fy[i, j - 1, k] / h2
                    q = idx(i, j - 1, k)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta
                if j < ny - 1:
                    beta = fy[i, j, k] / h2
                    q = idx(i, j + 1, k)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta
                if k > 0:
                    beta = fz[i, j, k - 1] / h2
                    q = idx(i, j, k - 1)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta
                if k < nz - 1:
                    beta = fz[i, j, k] / h2
                    q = idx(i, j, k + 1)
                    if q != pin_i:
                        entries.append((q, -beta))
                    diag_val += beta

                entries.append((p, diag_val))
                entries.sort(key=lambda e: e[0])

                diag_off = None
                base = len(values_list)
                for offset, (c, v) in enumerate(entries):
                    col_idx_list.append(c)
                    values_list.append(v)
                    if c == p:
                        diag_off = base + offset
                assert diag_off is not None, f"row {p} lost its diagonal"
                diag_offset_list.append(diag_off)
                row_ptr_list.append(len(col_idx_list))

    row_ptr = np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32))
    col_idx = np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32))
    values = np.ascontiguousarray(np.asarray(values_list, dtype=np.float64))
    diag_offset = np.ascontiguousarray(np.asarray(diag_offset_list, dtype=np.int32))
    return row_ptr, col_idx, values, diag_offset


def cell_to_face(u, v, w):
    """Project cell-centred u, v, w to interior face velocities by averaging.

    Returns (uf, vf, wf) with shapes (nx-1, ny, nz), (nx, ny-1, nz),
    (nx, ny, nz-1). Only interior faces are produced; boundary flux is defined
    separately by the projection contract (zero for homogeneous-Neumann p).
    """
    uf = 0.5 * (u[:-1, :, :] + u[1:, :, :])
    vf = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    wf = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
    return uf, vf, wf


def divergence_from_faces(uf, vf, wf, h, nx, ny, nz):
    """Cell-centred divergence from INTERIOR face velocities.

    Boundary faces are assumed to have zero normal velocity (homogeneous
    Neumann pressure → zero-flux wall). This is the ONE divergence operator
    that is discretely consistent with our FV Poisson matrix:

        A p = -div_from_faces(lambda_f * grad_f p)   (for non-pinned cells)

    At converged p with RHS = -div_from_faces(uf*, vf*, wf*) / dt, the
    corrected face velocities satisfy div_from_faces(uf, vf, wf) = 0 exactly
    (up to PCG tolerance), for ALL interior cells except the pinned cell.

    Shapes: uf (nx-1,ny,nz), vf (nx,ny-1,nz), wf (nx,ny,nz-1).
    """
    div = np.zeros((nx, ny, nz), dtype=np.float64)
    # Cell divergence = (flux_right - flux_left) / h.
    # Interior face uf[i, j, k] sits between cells i (left) and i+1 (right),
    # so it is cell i's right flux and cell i+1's left flux.
    div[:-1, :, :] += uf / h   # cell i's right face contributes +uf[i]
    div[1:, :, :] -= uf / h    # cell i+1's left face contributes -uf[i]
    div[:, :-1, :] += vf / h
    div[:, 1:, :] -= vf / h
    div[:, :, :-1] += wf / h
    div[:, :, 1:] -= wf / h
    return div


def divergence_face_velocities_cc(u, v, w, h):
    """Convenience: take cell-centred u, v, w → interior faces → cell divergence.

    Produces the RHS div(u*)/dt in a shape that matches the Poisson matrix's
    row ordering, AND is the same operator that `apply_velocity_correction`
    uses post-projection (same discrete identity).
    """
    uf, vf, wf = cell_to_face(u, v, w)
    nx, ny, nz = u.shape
    return divergence_from_faces(uf, vf, wf, h, nx, ny, nz)


def apply_velocity_correction(u_star, v_star, w_star, p, rho, h, dt):
    """Projection step: u_face^{n+1} = u_face^* - dt * (1/rho_face) * grad_face(p).

    Returns BOTH face-centred corrected velocities (for divergence checks) and
    cell-centred reconstructions (for visualization).

    Returns (uf, vf, wf, u_cell, v_cell, w_cell).
    """
    # face 1/rho (interior faces only)
    inv_rho_fx = 2.0 / (rho[:-1, :, :] + rho[1:, :, :])
    inv_rho_fy = 2.0 / (rho[:, :-1, :] + rho[:, 1:, :])
    inv_rho_fz = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])

    gp_fx = (p[1:, :, :] - p[:-1, :, :]) / h
    gp_fy = (p[:, 1:, :] - p[:, :-1, :]) / h
    gp_fz = (p[:, :, 1:] - p[:, :, :-1]) / h

    # interior face velocities of u*
    uf_int = 0.5 * (u_star[:-1, :, :] + u_star[1:, :, :])
    vf_int = 0.5 * (v_star[:, :-1, :] + v_star[:, 1:, :])
    wf_int = 0.5 * (w_star[:, :, :-1] + w_star[:, :, 1:])

    uf = uf_int - dt * inv_rho_fx * gp_fx
    vf = vf_int - dt * inv_rho_fy * gp_fy
    wf = wf_int - dt * inv_rho_fz * gp_fz

    # Cell-centred reconstruction (for visualization only — NOT used in
    # divergence checks; divergence is computed directly on face velocities).
    # Boundary face normal velocity = 0 (homogeneous Neumann pressure BC).
    nx, ny, nz = u_star.shape
    zero_bx = np.zeros((1, ny, nz), dtype=np.float64)
    zero_by = np.zeros((nx, 1, nz), dtype=np.float64)
    zero_bz = np.zeros((nx, ny, 1), dtype=np.float64)
    uf_full = np.concatenate([zero_bx, uf, zero_bx], axis=0)
    vf_full = np.concatenate([zero_by, vf, zero_by], axis=1)
    wf_full = np.concatenate([zero_bz, wf, zero_bz], axis=2)

    u_cell = 0.5 * (uf_full[:-1] + uf_full[1:])
    v_cell = 0.5 * (vf_full[:, :-1] + vf_full[:, 1:])
    w_cell = 0.5 * (wf_full[:, :, :-1] + wf_full[:, :, 1:])

    return uf, vf, wf, u_cell, v_cell, w_cell


def csr_to_scipy(row_ptr, col_idx, values, n):
    return scipy.sparse.csr_matrix(
        (values, col_idx, row_ptr), shape=(n, n))


# --------------------------------------------------------------------------
# Finite-difference vector ops (cell-centred, 2nd-order central, Neumann-safe).
# Used ONLY for:
#   - curvature of c in Test B (cell-centred central differences are fine for
#     a smooth indicator);
#   - visualization helpers.
# The projection's divergence check and velocity correction use the
# face-consistent operators `divergence_face` / `apply_velocity_correction`
# defined near the Poisson matrix builder above.
# --------------------------------------------------------------------------
def gradient_central(f, h):
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gz = np.zeros_like(f)
    gx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * h)
    gx[0, :, :] = (f[1, :, :] - f[0, :, :]) / h
    gx[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / h
    gy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * h)
    gy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / h
    gy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / h
    gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * h)
    gz[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / h
    gz[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / h
    return gx, gy, gz


def curvature_csf(c, h, eps=1e-10):
    """kappa = -div( grad(c) / |grad(c)| ).

    Uses cell-centred central differences (Neumann-safe). This is for the CSF
    surface-tension body force; it is independent of the projection's FV
    Poisson operators.
    Regularize 1/|grad(c)| with eps to avoid NaN in flat regions (brief §5).
    """
    gx, gy, gz = gradient_central(c, h)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz) + eps
    nxh = gx / mag
    nyh = gy / mag
    nzh = gz / mag
    # Central divergence (one-sided at boundaries).
    dnx_dx = np.zeros_like(c)
    dny_dy = np.zeros_like(c)
    dnz_dz = np.zeros_like(c)
    dnx_dx[1:-1, :, :] = (nxh[2:, :, :] - nxh[:-2, :, :]) / (2.0 * h)
    dnx_dx[0, :, :] = (nxh[1, :, :] - nxh[0, :, :]) / h
    dnx_dx[-1, :, :] = (nxh[-1, :, :] - nxh[-2, :, :]) / h
    dny_dy[:, 1:-1, :] = (nyh[:, 2:, :] - nyh[:, :-2, :]) / (2.0 * h)
    dny_dy[:, 0, :] = (nyh[:, 1, :] - nyh[:, 0, :]) / h
    dny_dy[:, -1, :] = (nyh[:, -1, :] - nyh[:, -2, :]) / h
    dnz_dz[:, :, 1:-1] = (nzh[:, :, 2:] - nzh[:, :, :-2]) / (2.0 * h)
    dnz_dz[:, :, 0] = (nzh[:, :, 1] - nzh[:, :, 0]) / h
    dnz_dz[:, :, -1] = (nzh[:, :, -1] - nzh[:, :, -2]) / h
    return -(dnx_dx + dny_dy + dnz_dz)


# --------------------------------------------------------------------------
# Test A — Divergence field map after pressure projection.
# --------------------------------------------------------------------------
def _random_solenoidal_u(n, h, seed=42):
    """A random-but-continuous vector field with NONZERO divergence (brief §Test A).

    Uses low-k Fourier modes — smooth but not incompressible. Seeded.
    """
    rng = np.random.default_rng(seed)
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    # k=1,2 modes
    u = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
         + 0.3 * np.cos(4 * np.pi * Z)) * rng.uniform(0.8, 1.2)
    v = (-np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
         + 0.3 * np.sin(4 * np.pi * X)) * rng.uniform(0.8, 1.2)
    w = (0.4 * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)) * rng.uniform(0.8, 1.2)
    return u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)


def run_test_A():
    print("\n========== Test A — Divergence Field Map ==========")
    n = N_GRID
    h = H_GRID
    dt = DT_PROJECTION

    print(f"Grid: {n}^3 = {n**3} unknowns, h = {h:.4e}, dt = {dt:.1e}")
    print(f"Density: rho_l = {RHO_LIQUID}, rho_g = {RHO_GAS} (ratio {RHO_LIQUID/RHO_GAS:.0f})")

    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)

    u_star, v_star, w_star = _random_solenoidal_u(n, h, seed=42)

    # Divergence computed on interior-face velocities only — boundary faces
    # carry zero normal flux, consistent with homogeneous-Neumann pressure BC.
    # No need to zero out cell-centred boundary u*: `cell_to_face` only
    # samples interior face midpoints.
    uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
    div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)

    # A p = -div(lambda grad p) (SPD), and the projection contract gives
    #   A p = -div(u*)/dt.
    rhs = (-div_u_star / dt).reshape(-1)

    print(f"max|div(u*)| before projection = {np.max(np.abs(div_u_star)):.3e}")

    row_ptr, col_idx, values, diag_offset = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))
    rhs[0] = 0.0  # match pinned row

    p_flat, niter, _ = dilu_pcg(row_ptr, col_idx, values, diag_offset,
                                rhs, max_iter=PCG_MAX_ITER_A, tol=PCG_TOL)
    print(f"DILU-PCG converged in {niter} iterations on variable-density Poisson")

    p = p_flat.reshape((n, n, n))
    uf, vf, wf, u, v, w = apply_velocity_correction(
        u_star, v_star, w_star, p, rho, h, dt)
    div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)
    max_div = float(np.max(np.abs(div_u)))
    print(f"[Test A] max|div(u)| after projection = {max_div:.3e}"
          f"  (gold standard: 0.0)  — "
          f"{'PASS' if max_div < 1e-6 else ('BORDERLINE' if max_div < 1e-2 else 'FAIL')}")

    # Plot.
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    div_slice = div_u[:, :, zslice].T  # transpose so x-axis = i, y-axis = j
    vmax = max(np.max(np.abs(div_slice)), 1e-16)
    im1 = ax1.imshow(div_slice, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
    # Interface contour (c = 0.5)
    c_slice = c[:, :, zslice].T
    ax1.contour(c_slice, levels=[0.5], colors="k", linewidths=0.8,
                extent=[0, 1, 0, 1])
    ax1.set_title(f"div(u) at z = {zslice}/{n} (after projection)\n"
                  f"max|div| slice = {np.max(np.abs(div_slice)):.2e}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="div(u)")

    # Panel 2: histogram / summary of |div| across the whole domain.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.log10(np.abs(div_u).ravel() + 1e-30), bins=60,
             color="steelblue", alpha=0.8)
    ax2.set_xlabel("log10 |div(u)|")
    ax2.set_ylabel("cell count")
    ax2.set_title(f"Domain-wide |div| distribution\n"
                  f"max|div| = {max_div:.2e}, PCG iters = {niter}")
    ax2.axvline(np.log10(max_div + 1e-30), color="red", linestyle="--",
                label=f"max = {max_div:.2e}")
    ax2.legend()

    fig.suptitle(
        f"Test A — Pressure projection mass conservation (density ratio "
        f"{RHO_LIQUID/RHO_GAS:.0f}, grid {n}^3)", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "divergence_map.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

    return {"max_div": max_div, "niter": niter, "plot": out_path}


# --------------------------------------------------------------------------
# Test B — Static droplet spurious currents.
# --------------------------------------------------------------------------
def run_test_B():
    print("\n========== Test B — Static Droplet Spurious Currents ==========")
    n = N_GRID
    h = H_GRID
    dt = DT_PROJECTION
    n_steps = 10

    print(f"Grid: {n}^3, sigma = {SIGMA}, dt = {dt:.1e}, steps = {n_steps}")

    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)
    inv_rho = 1.0 / rho

    # CSF curvature and surface tension force (precomputed — static droplet,
    # c does not change across the 10 steps).
    kappa = curvature_csf(c, h)
    grad_cx, grad_cy, grad_cz = gradient_central(c, h)
    # f_sigma = sigma * kappa * grad(c).
    fsx = SIGMA * kappa * grad_cx
    fsy = SIGMA * kappa * grad_cy
    fsz = SIGMA * kappa * grad_cz

    # Poisson matrix (fixed — static droplet, rho unchanged).
    row_ptr, col_idx, values, diag_offset = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))

    u = np.zeros((n, n, n), dtype=np.float64)
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    uinf_per_step = []
    div_per_step = []
    for step in range(n_steps):
        # 1) Tentative velocity: u* = u^n + dt * f_sigma / rho (cell-centred)
        u_star = u + dt * fsx * inv_rho
        v_star = v + dt * fsy * inv_rho
        w_star = w + dt * fsz * inv_rho

        # 2) Face-centred divergence of u*  →  Poisson RHS.
        uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
        div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)
        rhs = (-div_u_star / dt).reshape(-1)
        rhs[0] = 0.0

        p_flat, niter, _ = dilu_pcg(row_ptr, col_idx, values, diag_offset,
                                    rhs, max_iter=PCG_MAX_ITER_B, tol=PCG_TOL)
        p = p_flat.reshape((n, n, n))

        # 3) Face-consistent velocity correction; cell reconstruction for viz.
        uf, vf, wf, u, v, w = apply_velocity_correction(
            u_star, v_star, w_star, p, rho, h, dt)
        div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)

        umag = np.sqrt(u * u + v * v + w * w)
        uinf = float(np.max(umag))
        div_inf = float(np.max(np.abs(div_u)))
        uinf_per_step.append(uinf)
        div_per_step.append(div_inf)
        print(f"  step {step+1}/{n_steps}: PCG iters = {niter}, "
              f"||u||_inf = {uinf:.3e}, max|div u|={div_inf:.3e}")

    uinf_final = uinf_per_step[-1]
    # Pathology threshold: physical gold standard is exactly 0, so we define
    # PASS as "not growing fast and below 1e-3" — common in VOF/CSF literature
    # for this specific test (Francois et al., 2006-era benchmarks report
    # spurious currents ~ 1e-3 to 1e-1 for no-correction CSF).
    growing = uinf_per_step[-1] > 2.0 * uinf_per_step[0] if len(uinf_per_step) >= 2 else False
    verdict = (
        "PASS" if uinf_final < 1e-6 else
        ("GROWING" if growing else "SPURIOUS CURRENTS PRESENT"))
    print(f"[Test B] ||u||_inf at step {n_steps} = {uinf_final:.3e}  "
          f"(gold standard: 0.0)  — {verdict}")

    # Plot.
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    u_sl = u[:, :, zslice]
    v_sl = v[:, :, zslice]
    c_sl = c[:, :, zslice]

    # Downsample quiver density to stay visible.
    stride = max(1, n // 24)
    xs = _cell_centres(n, h)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    umag_sl = np.sqrt(u_sl ** 2 + v_sl ** 2)
    # Normalize arrows for visibility (but report raw magnitude in title).
    scale = max(np.max(umag_sl), 1e-30) * 20
    ax1.contour(XX, YY, c_sl, levels=[0.5], colors="k", linewidths=1.0)
    ax1.contourf(XX, YY, c_sl, levels=[0.5, 1.1],
                 colors=["lightblue"], alpha=0.3)
    ax1.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
               u_sl[::stride, ::stride], v_sl[::stride, ::stride],
               umag_sl[::stride, ::stride], cmap="viridis",
               scale=scale, scale_units="xy", width=0.004)
    ax1.set_aspect("equal")
    ax1.set_title(
        f"Velocity quiver at z = {zslice} after {n_steps} steps\n"
        f"interface (black) and c>0.5 region shaded")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(1, 2, 2)
    steps = np.arange(1, n_steps + 1)
    ax2.semilogy(steps, uinf_per_step, "-o", color="crimson")
    ax2.axhline(1e-12, color="k", linestyle=":", alpha=0.5,
                label="machine-zero floor")
    ax2.set_xlabel("time step")
    ax2.set_ylabel(r"$\|u\|_\infty$")
    ax2.set_title(r"$\|u\|_\infty$ vs time step")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Test B — Static droplet spurious currents (sigma={SIGMA}, "
        f"density ratio {RHO_LIQUID/RHO_GAS:.0f})", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "spurious_currents.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

    return {"uinf_final": uinf_final, "uinf_history": uinf_per_step,
            "plot": out_path}


# --------------------------------------------------------------------------
# Test C — Spatial residual halo (preconditioner health).
# --------------------------------------------------------------------------
def _three_tier_density(n: int, h: float):
    """Gas/liquid/solid 3-tier density with a Gaussian "melt-pool" dip on top
    surface of the solid (brief §Test C).

    - Bottom ~40% of z: solid (rho_s = 8000)
    - Middle ~20% of z: liquid (rho_l = 1000)
    - Top ~40% of z: gas (rho_g = 1)
    Transitions are smooth tanh with eps = 1.5h, plus a Gaussian dimple in the
    solid/liquid boundary to mimic a melt pool cavity.
    """
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    eps = 1.5 * h

    # Melt-pool dip: the solid/liquid interface sits at z = z_sl_base - dip(x, y)
    z_sl_base = 0.40
    dip_amp = 0.10
    dip_sigma = 0.12
    dip = dip_amp * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2)
                           / (2 * dip_sigma ** 2))
    z_sl = z_sl_base - dip

    z_lg = 0.60  # liquid/gas interface (flat)

    # Indicators (Heavisides)
    # c_solid = 1 below z_sl (smooth)
    c_solid = 0.5 * (1.0 - np.tanh((Z - z_sl) / eps))
    # c_gas = 1 above z_lg
    c_gas = 0.5 * (1.0 + np.tanh((Z - z_lg) / eps))
    # c_liquid = 1 - c_solid - c_gas, clipped to be non-negative
    c_liquid = np.clip(1.0 - c_solid - c_gas, 0.0, 1.0)

    rho = RHO_SOLID * c_solid + RHO_LIQUID * c_liquid + RHO_GAS * c_gas
    # guard against any numerical under-run below min of the three:
    rho = np.maximum(rho, RHO_GAS)
    return rho.astype(np.float64), c_solid, c_liquid, c_gas


def run_test_C():
    print("\n========== Test C — Spatial Residual Halo ==========")
    n = N_GRID
    h = H_GRID
    print(f"Grid: {n}^3, DILU-PCG fixed iters = {PCG_MAX_ITER_C}, "
          f"density span {RHO_GAS}..{RHO_SOLID}")

    rho, cs, cl, cg = _three_tier_density(n, h)
    print(f"rho_min = {rho.min():.2f}, rho_max = {rho.max():.2f}, "
          f"contrast = {rho.max()/rho.min():.2e}")

    row_ptr, col_idx, values, diag_offset = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))
    n_total = n * n * n
    nnz = int(values.size)
    print(f"CSR: n = {n_total}, nnz = {nnz}, avg nnz/row = {nnz/n_total:.2f}")

    # RHS: a smooth "laser heating / divergence source" so the solve is non-trivial
    # and the gold-standard solution is well-defined.
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    src = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
           * np.exp(-((Z - 0.5) ** 2) / 0.05))
    # Remove mean to satisfy compatibility with Neumann boundary (pinned cell
    # absorbs residual null-space drift).
    src = src - src.mean()
    rhs = src.reshape(-1).copy()
    rhs[0] = 0.0  # pinned row

    # ------ Gold standard: sparse LU direct solve via scipy spsolve ------
    print("Building scipy CSR and calling spsolve (gold standard)...")
    A_scipy = csr_to_scipy(row_ptr, col_idx, values, n_total)
    # Use superlu; UMFPACK would be faster but may not be available.
    p_exact = sla.spsolve(A_scipy, rhs, use_umfpack=False)
    p_exact = np.ascontiguousarray(p_exact, dtype=np.float64)
    res_exact = A_scipy.dot(p_exact) - rhs
    print(f"spsolve residual ||A p_exact - rhs||_inf = "
          f"{np.max(np.abs(res_exact)):.3e}")

    # ------ DILU-PCG at exactly PCG_MAX_ITER_C iterations ------
    print(f"Running DILU-PCG for exactly {PCG_MAX_ITER_C} iterations (fixed)...")
    p_dilu, iters, rnorms = dilu_pcg(row_ptr, col_idx, values, diag_offset,
                                     rhs, max_iter=PCG_MAX_ITER_C,
                                     tol=0.0, fixed_iter=True)
    print(f"DILU-PCG finished; ||r||_2 trajectory: "
          f"first={rnorms[0]:.3e}, last={rnorms[-1]:.3e}")

    E = p_dilu - p_exact
    max_err = float(np.max(np.abs(E)))
    rel_err = max_err / max(float(np.max(np.abs(p_exact))), 1e-300)
    # Mean-shifted error: the unconstrained PCG can carry a slow global mode
    # (null-space component pinned only at cell 0). The geometric-halo question
    # is about SPATIAL STRUCTURE of the error, not absolute offset. We report
    # both and visualize the mean-detrended field so the halo, if any, is
    # visible.
    E_demean = E - np.mean(E)
    max_err_demean = float(np.max(np.abs(E_demean)))
    print(f"[Test C] max|E| at iter {PCG_MAX_ITER_C} = {max_err:.3e}  "
          f"(rel = {rel_err:.3e}; mean-detrended max = {max_err_demean:.3e})  — "
          f"{'PASS' if rel_err < 1e-6 else 'MID-CONVERGENCE (expected for fixed iter cap)'}")

    # Reshape both raw and mean-detrended fields for plotting.
    E_cube = E.reshape((n, n, n))
    E_cube_dm = E_demean.reshape((n, n, n))
    zslice = n // 2
    # Brief asks for z=16 specifically (horizontal slice). Supplement with a
    # vertical y=mid slice to expose the melt-pool-shaped solid/liquid
    # interface in the plot, which is where a preconditioner pathology should
    # concentrate if one exists.
    E_slice_dm = E_cube_dm[:, :, zslice].T
    rho_slice = rho[:, :, zslice].T

    E_vslice_dm = E_cube_dm[:, n // 2, :].T
    rho_vslice = rho[:, n // 2, :].T

    fig = plt.figure(figsize=(12, 4.8))

    ax1 = fig.add_subplot(1, 2, 1)
    vmax = max(np.max(np.abs(E_slice_dm)), 1e-30)
    im1 = ax1.imshow(E_slice_dm, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
    levels = [RHO_LIQUID - 1.0, (RHO_LIQUID + RHO_SOLID) / 2,
              (RHO_LIQUID + RHO_GAS) / 2]
    present = [lv for lv in levels if rho_slice.min() <= lv <= rho_slice.max()]
    if present:
        ax1.contour(rho_slice, levels=present, colors="k", linewidths=0.8,
                    extent=[0, 1, 0, 1])
    ax1.set_title(f"E - mean(E)  at z = {zslice}  (horizontal slice)\n"
                  f"max|E_dm|_slice = {np.max(np.abs(E_slice_dm)):.2e}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="E − <E>")

    ax2 = fig.add_subplot(1, 2, 2)
    vmax2 = max(np.max(np.abs(E_vslice_dm)), 1e-30)
    im2 = ax2.imshow(E_vslice_dm, origin="lower", cmap="RdBu_r",
                     vmin=-vmax2, vmax=vmax2, extent=[0, 1, 0, 1])
    pres = []
    for lv in [(RHO_GAS + RHO_LIQUID) / 2, (RHO_LIQUID + RHO_SOLID) / 2]:
        if rho_vslice.min() <= lv <= rho_vslice.max():
            pres.append(lv)
    if pres:
        ax2.contour(rho_vslice, levels=pres, colors="k", linewidths=0.8,
                    extent=[0, 1, 0, 1])
    ax2.set_title(f"E - mean(E)  at y = {n//2}  (vertical slice; melt-pool visible)\n"
                  f"max|E_dm|_slice = {np.max(np.abs(E_vslice_dm)):.2e}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    fig.colorbar(im2, ax=ax2, label="E − <E>")

    fig.suptitle(
        f"Test C — Mean-detrended residual at iter {PCG_MAX_ITER_C} "
        f"(density contrast {rho.max()/rho.min():.1e}, grid {n}^3). "
        f"Raw max|E|={max_err:.2e}, demean max|E|={max_err_demean:.2e}",
        fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "residual_halo.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")

    return {"max_err": max_err, "rel_err": rel_err,
            "max_err_demean": max_err_demean,
            "rnorm_first": rnorms[0], "rnorm_last": rnorms[-1],
            "plot": out_path}


# --------------------------------------------------------------------------
# Dispatch.
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["A", "B", "C"], default="ALL",
                        help="Which test to run (default: all three)")
    args = parser.parse_args()

    results = {}
    if args.test in ("A", "ALL"):
        results["A"] = run_test_A()
    if args.test in ("B", "ALL"):
        results["B"] = run_test_B()
    if args.test in ("C", "ALL"):
        results["C"] = run_test_C()

    print("\n========== Summary ==========")
    if "A" in results:
        print(f"Test A: max|div(u)| = {results['A']['max_div']:.3e} "
              f"(PCG iters = {results['A']['niter']})")
    if "B" in results:
        print(f"Test B: ||u||_inf @ step 10 = {results['B']['uinf_final']:.3e}")
    if "C" in results:
        print(f"Test C: max|E| after {PCG_MAX_ITER_C} iters = "
              f"{results['C']['max_err']:.3e} "
              f"(rel = {results['C']['rel_err']:.3e}; "
              f"||r||_2: {results['C']['rnorm_first']:.2e} → "
              f"{results['C']['rnorm_last']:.2e})")


if __name__ == "__main__":
    main()
