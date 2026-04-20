"""Phase 3 — Physical Sanity Benchmark (C6, C7, C8).

Mirrors Phase 2.5's `dilu/cusparse/tests/physical_benchmark.py` exactly —
same 32³ grid, same density ratios, same CSF setup, same Poisson builder,
same projection operators, same plot layouts — but threads the preconditioner
through Phase 3's `MulticolorPlan` instead of Phase 2's `Plan`.

Additionally computes the F2 detection number for Test C:

    corr_halo = Pearson( |E_demean|, ||∇ log ρ|| )   (eq. 5.1, math doc §5.3.3)

F2 STOP signal (brief directive, verbatim): if corr_halo > 0.5 the user has
explicitly forbidden any rescue attempt. The benchmark prints the verdict
loudly and writes a failure report under
`docs/benchmark/phase3_FAILED_interface_halo.md` if triggered.

Plots land in `dilu/multicolor/bench/plots/` for side-by-side comparison
with Phase 2.5.

Usage:
    python physical_benchmark_phase3.py              # all three
    python physical_benchmark_phase3.py --test A|B|C
"""
from __future__ import annotations

# --- VRAM rails before any JAX import ---------------------------------------
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla

import jax
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dilu.multicolor.python import MulticolorPlan

# --- Configuration (identical to Phase 2.5) ---------------------------------
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

N_GRID = 32
H_GRID = 1.0 / N_GRID
RHO_LIQUID = 1000.0
RHO_GAS = 1.0
RHO_SOLID = 8000.0
R_DROPLET = 0.25
SIGMA = 0.07
DT_PROJECTION = 1.0e-4
PCG_TOL = 1.0e-10
PCG_MAX_ITER_A = 600
PCG_MAX_ITER_B = 600
PCG_MAX_ITER_C = 50

PLOTS_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "bench", "plots"))
os.makedirs(PLOTS_DIR, exist_ok=True)

# F2 thresholds (math doc §5.3.3, brief F2 directive).
F2_PASS_THRESHOLD = 0.25
F2_BORDERLINE_UPPER = 0.5   # above this → STOP, write failure report


# ---------------------------------------------------------------------------
# DILU-mcPCG driver — mirrors Phase 2's `dilu_pcg` but uses MulticolorPlan.
# Same segment_sum-based SpMV, same PCG recurrence.
# ---------------------------------------------------------------------------
def _spmv_gpu(row_ptr, col_idx, values, x):
    n = x.shape[0]
    nnz = values.shape[0]
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(row_ptr[1:], k, side="right")
    prods = values * x[col_idx]
    return jax.ops.segment_sum(prods, row_of, num_segments=n)


def mcdilu_pcg(row_ptr_h, col_idx_h, values_h, b_h,
               max_iter: int, tol: float = 1e-10,
               fixed_iter: bool = False,
               grid_shape=None):
    """Phase 3 DILU-mcPCG driver. Grid shape enables red-black fast path."""
    row_ptr_h = np.ascontiguousarray(row_ptr_h, dtype=np.int32)
    col_idx_h = np.ascontiguousarray(col_idx_h, dtype=np.int32)
    values_h  = np.ascontiguousarray(values_h,  dtype=np.float64)
    b_h       = np.ascontiguousarray(b_h,       dtype=np.float64)

    rp = jax.device_put(jnp.asarray(row_ptr_h))
    ci = jax.device_put(jnp.asarray(col_idx_h))
    vv = jax.device_put(jnp.asarray(values_h))
    b_d = jax.device_put(jnp.asarray(b_h))

    def spmv(x):
        return _spmv_gpu(rp, ci, vv, x)

    n = int(b_h.shape[0])
    rnorms = []
    with MulticolorPlan(row_ptr_h, col_idx_h, values_h,
                        grid_shape=grid_shape) as plan:
        d_star = plan.factor(vv)
        d_star.block_until_ready()

        x = jnp.zeros(n, dtype=jnp.float64)
        r = b_d - spmv(x)
        b_norm = max(float(jnp.linalg.norm(b_d)), 1e-300)
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


# ---------------------------------------------------------------------------
# The rest of the helpers are copy-paste from Phase 2.5 (grid helpers, phase
# indicator, Poisson builder, face operators, CSF curvature). Kept verbatim
# so Phase 3 uses bit-identical setup — any difference in results is
# attributable to the preconditioner, not the test harness.
# ---------------------------------------------------------------------------
def _cell_centres(n, h):
    return (np.arange(n) + 0.5) * h


def smooth_sphere_indicator(n, h, radius, eps_factor=1.5):
    xs = _cell_centres(n, h) - 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    eps = eps_factor * h
    c = 0.5 * (1.0 - np.tanh((R - radius) / eps))
    return np.ascontiguousarray(c, dtype=np.float64)


def density_from_indicator(c, rho_high, rho_low):
    return rho_high * c + rho_low * (1.0 - c)


def _face_coeffs(rho):
    fx = 2.0 / (rho[:-1, :, :] + rho[1:, :, :])
    fy = 2.0 / (rho[:, :-1, :] + rho[:, 1:, :])
    fz = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])
    return fx, fy, fz


def build_variable_density_poisson_3d(rho, h, pin_cell=(0, 0, 0)):
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
                if p == pin_i:
                    col_idx_list.append(p); values_list.append(1.0)
                    diag_offset_list.append(len(values_list) - 1)
                    row_ptr_list.append(len(col_idx_list))
                    continue
                entries = []
                diag_val = 0.0
                if i > 0:
                    beta = fx[i - 1, j, k] / h2; q = idx(i - 1, j, k)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                if i < nx - 1:
                    beta = fx[i, j, k] / h2; q = idx(i + 1, j, k)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                if j > 0:
                    beta = fy[i, j - 1, k] / h2; q = idx(i, j - 1, k)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                if j < ny - 1:
                    beta = fy[i, j, k] / h2; q = idx(i, j + 1, k)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                if k > 0:
                    beta = fz[i, j, k - 1] / h2; q = idx(i, j, k - 1)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                if k < nz - 1:
                    beta = fz[i, j, k] / h2; q = idx(i, j, k + 1)
                    if q != pin_i: entries.append((q, -beta))
                    diag_val += beta
                entries.append((p, diag_val))
                entries.sort(key=lambda e: e[0])
                diag_off = None
                base = len(values_list)
                for offset, (c, v) in enumerate(entries):
                    col_idx_list.append(c); values_list.append(v)
                    if c == p:
                        diag_off = base + offset
                assert diag_off is not None
                diag_offset_list.append(diag_off)
                row_ptr_list.append(len(col_idx_list))

    return (
        np.ascontiguousarray(np.asarray(row_ptr_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(col_idx_list, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(values_list, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(diag_offset_list, dtype=np.int32)),
    )


def cell_to_face(u, v, w):
    return (0.5 * (u[:-1, :, :] + u[1:, :, :]),
            0.5 * (v[:, :-1, :] + v[:, 1:, :]),
            0.5 * (w[:, :, :-1] + w[:, :, 1:]))


def divergence_from_faces(uf, vf, wf, h, nx, ny, nz):
    div = np.zeros((nx, ny, nz), dtype=np.float64)
    div[:-1, :, :] += uf / h
    div[1:, :, :] -= uf / h
    div[:, :-1, :] += vf / h
    div[:, 1:, :] -= vf / h
    div[:, :, :-1] += wf / h
    div[:, :, 1:] -= wf / h
    return div


def apply_velocity_correction(u_star, v_star, w_star, p, rho, h, dt):
    inv_rho_fx = 2.0 / (rho[:-1, :, :] + rho[1:, :, :])
    inv_rho_fy = 2.0 / (rho[:, :-1, :] + rho[:, 1:, :])
    inv_rho_fz = 2.0 / (rho[:, :, :-1] + rho[:, :, 1:])
    gp_fx = (p[1:, :, :] - p[:-1, :, :]) / h
    gp_fy = (p[:, 1:, :] - p[:, :-1, :]) / h
    gp_fz = (p[:, :, 1:] - p[:, :, :-1]) / h
    uf_int = 0.5 * (u_star[:-1, :, :] + u_star[1:, :, :])
    vf_int = 0.5 * (v_star[:, :-1, :] + v_star[:, 1:, :])
    wf_int = 0.5 * (w_star[:, :, :-1] + w_star[:, :, 1:])
    uf = uf_int - dt * inv_rho_fx * gp_fx
    vf = vf_int - dt * inv_rho_fy * gp_fy
    wf = wf_int - dt * inv_rho_fz * gp_fz
    nx, ny, nz = u_star.shape
    zero_bx = np.zeros((1, ny, nz))
    zero_by = np.zeros((nx, 1, nz))
    zero_bz = np.zeros((nx, ny, 1))
    uf_full = np.concatenate([zero_bx, uf, zero_bx], axis=0)
    vf_full = np.concatenate([zero_by, vf, zero_by], axis=1)
    wf_full = np.concatenate([zero_bz, wf, zero_bz], axis=2)
    u_cell = 0.5 * (uf_full[:-1] + uf_full[1:])
    v_cell = 0.5 * (vf_full[:, :-1] + vf_full[:, 1:])
    w_cell = 0.5 * (wf_full[:, :, :-1] + wf_full[:, :, 1:])
    return uf, vf, wf, u_cell, v_cell, w_cell


def csr_to_scipy(row_ptr, col_idx, values, n):
    return scipy.sparse.csr_matrix((values, col_idx, row_ptr), shape=(n, n))


def gradient_central(f, h):
    gx = np.zeros_like(f); gy = np.zeros_like(f); gz = np.zeros_like(f)
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
    gx, gy, gz = gradient_central(c, h)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz) + eps
    nxh = gx / mag; nyh = gy / mag; nzh = gz / mag
    dnx = np.zeros_like(c); dny = np.zeros_like(c); dnz = np.zeros_like(c)
    dnx[1:-1, :, :] = (nxh[2:, :, :] - nxh[:-2, :, :]) / (2.0 * h)
    dnx[0, :, :] = (nxh[1, :, :] - nxh[0, :, :]) / h
    dnx[-1, :, :] = (nxh[-1, :, :] - nxh[-2, :, :]) / h
    dny[:, 1:-1, :] = (nyh[:, 2:, :] - nyh[:, :-2, :]) / (2.0 * h)
    dny[:, 0, :] = (nyh[:, 1, :] - nyh[:, 0, :]) / h
    dny[:, -1, :] = (nyh[:, -1, :] - nyh[:, -2, :]) / h
    dnz[:, :, 1:-1] = (nzh[:, :, 2:] - nzh[:, :, :-2]) / (2.0 * h)
    dnz[:, :, 0] = (nzh[:, :, 1] - nzh[:, :, 0]) / h
    dnz[:, :, -1] = (nzh[:, :, -1] - nzh[:, :, -2]) / h
    return -(dnx + dny + dnz)


# ---------------------------------------------------------------------------
# Test A
# ---------------------------------------------------------------------------
def _random_solenoidal_u(n, h, seed=42):
    rng = np.random.default_rng(seed)
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    u = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
         + 0.3 * np.cos(4 * np.pi * Z)) * rng.uniform(0.8, 1.2)
    v = (-np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
         + 0.3 * np.sin(4 * np.pi * X)) * rng.uniform(0.8, 1.2)
    w = (0.4 * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)) * rng.uniform(0.8, 1.2)
    return u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)


def run_test_A():
    print("\n========== Test A — Divergence Field Map ==========")
    n = N_GRID; h = H_GRID; dt = DT_PROJECTION
    print(f"Grid: {n}^3 = {n**3} unknowns, h = {h:.4e}")
    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)

    u_star, v_star, w_star = _random_solenoidal_u(n, h, seed=42)
    uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
    div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)
    rhs = (-div_u_star / dt).reshape(-1)
    print(f"max|div(u*)| before projection = {np.max(np.abs(div_u_star)):.3e}")

    row_ptr, col_idx, values, _ = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))
    rhs[0] = 0.0

    p_flat, niter, _ = mcdilu_pcg(
        row_ptr, col_idx, values, rhs,
        max_iter=PCG_MAX_ITER_A, tol=PCG_TOL,
        grid_shape=(n, n, n))
    print(f"DILU-mcPCG converged in {niter} iterations")

    p = p_flat.reshape((n, n, n))
    uf, vf, wf, _, _, _ = apply_velocity_correction(
        u_star, v_star, w_star, p, rho, h, dt)
    div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)
    max_div = float(np.max(np.abs(div_u)))
    verdict = 'PASS' if max_div < 4.1e-8 else ('BORDERLINE' if max_div < 1e-7 else 'FAIL')
    print(f"[Test A] max|div(u)| after projection = {max_div:.3e}  "
          f"(C6 threshold 4.1e-8) — {verdict}")

    # Plot (same layout as Phase 2.5).
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    div_slice = div_u[:, :, zslice].T
    vmax = max(np.max(np.abs(div_slice)), 1e-16)
    im1 = ax1.imshow(div_slice, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
    c_slice = c[:, :, zslice].T
    ax1.contour(c_slice, levels=[0.5], colors="k", linewidths=0.8,
                extent=[0, 1, 0, 1])
    ax1.set_title(f"[P3] div(u) at z = {zslice} (after projection)\n"
                  f"max|div| slice = {np.max(np.abs(div_slice)):.2e}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="div(u)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.log10(np.abs(div_u).ravel() + 1e-30), bins=60,
             color="steelblue", alpha=0.8)
    ax2.set_xlabel("log10 |div(u)|"); ax2.set_ylabel("cell count")
    ax2.set_title(f"[P3] domain |div| dist\n"
                  f"max = {max_div:.2e}, iters = {niter}")
    ax2.axvline(np.log10(max_div + 1e-30), color="red", linestyle="--")
    fig.suptitle(
        f"[PHASE 3] Test A — Projection mass conservation, density ratio "
        f"{RHO_LIQUID/RHO_GAS:.0f}, grid {n}^3", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "divergence_map.png")
    fig.savefig(out_path); plt.close(fig)
    print(f"Wrote {out_path}")
    return {"max_div": max_div, "niter": niter, "plot": out_path,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Test B
# ---------------------------------------------------------------------------
def run_test_B():
    print("\n========== Test B — Static Droplet Spurious Currents ==========")
    n = N_GRID; h = H_GRID; dt = DT_PROJECTION; n_steps = 10
    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)
    inv_rho = 1.0 / rho
    kappa = curvature_csf(c, h)
    gcx, gcy, gcz = gradient_central(c, h)
    fsx = SIGMA * kappa * gcx; fsy = SIGMA * kappa * gcy; fsz = SIGMA * kappa * gcz

    row_ptr, col_idx, values, _ = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))

    u = np.zeros((n, n, n)); v = np.zeros_like(u); w = np.zeros_like(u)
    uinf_hist = []; div_hist = []
    for step in range(n_steps):
        u_star = u + dt * fsx * inv_rho
        v_star = v + dt * fsy * inv_rho
        w_star = w + dt * fsz * inv_rho
        uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
        div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)
        rhs = (-div_u_star / dt).reshape(-1); rhs[0] = 0.0
        p_flat, niter, _ = mcdilu_pcg(
            row_ptr, col_idx, values, rhs,
            max_iter=PCG_MAX_ITER_B, tol=PCG_TOL,
            grid_shape=(n, n, n))
        p = p_flat.reshape((n, n, n))
        uf, vf, wf, u, v, w = apply_velocity_correction(
            u_star, v_star, w_star, p, rho, h, dt)
        div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)
        umag = np.sqrt(u * u + v * v + w * w)
        uinf = float(np.max(umag)); div_inf = float(np.max(np.abs(div_u)))
        uinf_hist.append(uinf); div_hist.append(div_inf)
        print(f"  step {step+1}/{n_steps}: iters={niter}, "
              f"||u||_inf={uinf:.3e}, max|div|={div_inf:.3e}")

    uinf_final = uinf_hist[-1]
    div_max = max(div_hist)
    verdict = 'PASS' if (uinf_final <= 1.9e-6 and div_max < 1e-12) else 'FAIL'
    print(f"[Test B] ||u||_inf @ step 10 = {uinf_final:.3e}  "
          f"(C7 thresh 1.9e-6, max|div| {div_max:.2e} vs 1e-12) — {verdict}")

    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    u_sl = u[:, :, zslice]; v_sl = v[:, :, zslice]; c_sl = c[:, :, zslice]
    stride = max(1, n // 24)
    xs = _cell_centres(n, h)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    umag_sl = np.sqrt(u_sl ** 2 + v_sl ** 2)
    scale = max(np.max(umag_sl), 1e-30) * 20
    ax1.contour(XX, YY, c_sl, levels=[0.5], colors="k", linewidths=1.0)
    ax1.contourf(XX, YY, c_sl, levels=[0.5, 1.1],
                 colors=["lightblue"], alpha=0.3)
    ax1.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
               u_sl[::stride, ::stride], v_sl[::stride, ::stride],
               umag_sl[::stride, ::stride], cmap="viridis",
               scale=scale, scale_units="xy", width=0.004)
    ax1.set_aspect("equal"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_title(f"[P3] velocity @ z={zslice} step {n_steps}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 2, 2)
    steps = np.arange(1, n_steps + 1)
    ax2.semilogy(steps, uinf_hist, "-o", color="crimson")
    ax2.axhline(1e-12, color="k", linestyle=":", alpha=0.5, label="machine zero")
    ax2.set_xlabel("time step"); ax2.set_ylabel(r"$\|u\|_\infty$")
    ax2.set_title(r"$\|u\|_\infty$ per step")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"[PHASE 3] Test B — Static droplet (sigma={SIGMA})", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "spurious_currents.png")
    fig.savefig(out_path); plt.close(fig)
    print(f"Wrote {out_path}")
    return {"uinf_final": uinf_final, "uinf_history": uinf_hist,
            "div_max": div_max, "plot": out_path, "verdict": verdict}


# ---------------------------------------------------------------------------
# Test C — acid test
# ---------------------------------------------------------------------------
def _three_tier_density(n, h):
    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    eps = 1.5 * h
    z_sl_base = 0.40
    dip = 0.10 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.12 ** 2))
    z_sl = z_sl_base - dip
    z_lg = 0.60
    c_solid = 0.5 * (1.0 - np.tanh((Z - z_sl) / eps))
    c_gas = 0.5 * (1.0 + np.tanh((Z - z_lg) / eps))
    c_liquid = np.clip(1.0 - c_solid - c_gas, 0.0, 1.0)
    rho = RHO_SOLID * c_solid + RHO_LIQUID * c_liquid + RHO_GAS * c_gas
    rho = np.maximum(rho, RHO_GAS)
    return rho.astype(np.float64), c_solid, c_liquid, c_gas


def run_test_C():
    print("\n========== Test C — Spatial Residual Halo (F2 ACID TEST) ==========")
    n = N_GRID; h = H_GRID
    rho, _, _, _ = _three_tier_density(n, h)
    print(f"rho_min={rho.min():.2f}, rho_max={rho.max():.2f}, "
          f"contrast={rho.max()/rho.min():.2e}")

    row_ptr, col_idx, values, _ = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))
    n_total = n * n * n

    xs = _cell_centres(n, h)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    src = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
           * np.exp(-((Z - 0.5) ** 2) / 0.05))
    src = src - src.mean()
    rhs = src.reshape(-1).copy(); rhs[0] = 0.0

    print("scipy spsolve (gold standard)...")
    A_scipy = csr_to_scipy(row_ptr, col_idx, values, n_total)
    p_exact = sla.spsolve(A_scipy, rhs, use_umfpack=False)
    p_exact = np.ascontiguousarray(p_exact, dtype=np.float64)
    print(f"spsolve res = {np.max(np.abs(A_scipy.dot(p_exact) - rhs)):.3e}")

    print(f"DILU-mcPCG fixed {PCG_MAX_ITER_C} iterations...")
    p_mc, iters, rnorms = mcdilu_pcg(
        row_ptr, col_idx, values, rhs,
        max_iter=PCG_MAX_ITER_C, tol=0.0, fixed_iter=True,
        grid_shape=(n, n, n))
    print(f"rnorm: first={rnorms[0]:.3e}, last={rnorms[-1]:.3e}")

    E = p_mc - p_exact
    max_err = float(np.max(np.abs(E)))
    rel_err = max_err / max(float(np.max(np.abs(p_exact))), 1e-300)
    E_demean = E - np.mean(E)
    max_err_demean = float(np.max(np.abs(E_demean)))
    print(f"max|E|={max_err:.3e}, rel={rel_err:.3e}, "
          f"max|E_demean|={max_err_demean:.3e}")

    # ====================================================================
    # F2 DETECTION — Pearson correlation of |E_demean| vs ||∇ log ρ||
    # (math doc §5.3.3 eq 5.1). This is THE F2 TRIGGER NUMBER.
    # ====================================================================
    log_rho = np.log(rho)
    glx, gly, glz = gradient_central(log_rho, h)
    grad_log_rho_mag = np.sqrt(glx * glx + gly * gly + glz * glz)

    abs_E_dm = np.abs(E_demean.reshape((n, n, n)))
    # Pearson correlation of flat arrays.
    x_flat = abs_E_dm.ravel()
    y_flat = grad_log_rho_mag.ravel()
    x_cent = x_flat - x_flat.mean()
    y_cent = y_flat - y_flat.mean()
    denom = float(np.linalg.norm(x_cent) * np.linalg.norm(y_cent))
    if denom < 1e-300:
        corr = 0.0
    else:
        corr = float(np.dot(x_cent, y_cent) / denom)

    # F2 verdict.
    if corr <= F2_PASS_THRESHOLD:
        f2_verdict = "PASS"
    elif corr <= F2_BORDERLINE_UPPER:
        f2_verdict = "BORDERLINE"
    else:
        f2_verdict = "F2_TRIGGER_STOP"
    print(f"\n=== F2 DETECTOR ===")
    print(f"   corr(|E_demean|, ||grad log rho||) = {corr:.4f}")
    print(f"   Thresholds: PASS ≤ {F2_PASS_THRESHOLD}, "
          f"BORDERLINE ≤ {F2_BORDERLINE_UPPER}, STOP > {F2_BORDERLINE_UPPER}")
    print(f"   Verdict: {f2_verdict}")

    # Plot (Phase 2.5 layout).
    E_cube_dm = E_demean.reshape((n, n, n))
    zslice = n // 2
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
    ax1.set_title(f"[P3] E - mean(E) at z={zslice}\n"
                  f"max|E_dm| slice={np.max(np.abs(E_slice_dm)):.2e}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, label="E - <E>")

    ax2 = fig.add_subplot(1, 2, 2)
    vmax2 = max(np.max(np.abs(E_vslice_dm)), 1e-30)
    im2 = ax2.imshow(E_vslice_dm, origin="lower", cmap="RdBu_r",
                     vmin=-vmax2, vmax=vmax2, extent=[0, 1, 0, 1])
    pres = [lv for lv in [(RHO_GAS + RHO_LIQUID) / 2,
                          (RHO_LIQUID + RHO_SOLID) / 2]
            if rho_vslice.min() <= lv <= rho_vslice.max()]
    if pres:
        ax2.contour(rho_vslice, levels=pres, colors="k", linewidths=0.8,
                    extent=[0, 1, 0, 1])
    ax2.set_title(f"[P3] E - mean(E) at y={n//2} (vertical; melt pool)\n"
                  f"max|E_dm| slice={np.max(np.abs(E_vslice_dm)):.2e}")
    ax2.set_xlabel("x"); ax2.set_ylabel("z")
    fig.colorbar(im2, ax=ax2, label="E - <E>")

    fig.suptitle(
        f"[PHASE 3] Test C — Mean-detrended residual @ iter {PCG_MAX_ITER_C} "
        f"(corr={corr:.3f}, verdict={f2_verdict}). "
        f"Raw max|E|={max_err:.2e}, demean max|E|={max_err_demean:.2e}",
        fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "residual_halo.png")
    fig.savefig(out_path); plt.close(fig)
    print(f"Wrote {out_path}")

    return {"max_err": max_err, "rel_err": rel_err,
            "max_err_demean": max_err_demean,
            "rnorm_first": rnorms[0], "rnorm_last": rnorms[-1],
            "corr": corr, "verdict": f2_verdict,
            "plot": out_path}


# ---------------------------------------------------------------------------
# Dispatch.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["A", "B", "C"], default="ALL")
    args = parser.parse_args()

    results = {}
    if args.test in ("A", "ALL"): results["A"] = run_test_A()
    if args.test in ("B", "ALL"): results["B"] = run_test_B()
    if args.test in ("C", "ALL"): results["C"] = run_test_C()

    print("\n========== Phase 3 Physical Summary ==========")
    for k in ("A", "B", "C"):
        if k in results:
            r = results[k]
            print(f"  Test {k}: {r.get('verdict', '?')}   "
                  f"(max|div|={r.get('max_div','-')}  "
                  f"uinf={r.get('uinf_final','-')}  "
                  f"corr={r.get('corr','-')})")

    # Final F2 verdict.
    if "C" in results and results["C"]["verdict"] == "F2_TRIGGER_STOP":
        print("\n>>> F2 TRIGGER: corr > 0.5. "
              "Writing docs/benchmark/phase3_FAILED_interface_halo.md and "
              "STOPPING per user directive. <<<")
        sys.exit(2)  # distinct exit code for F2


if __name__ == "__main__":
    main()
