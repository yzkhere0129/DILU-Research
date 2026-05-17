"""T12 — Physical benchmarks A & B at 64^3 with AMG-PCG preconditioner.

Reuses Phase 2.5's problem builders. Swaps the DILU-PCG driver for an
AMG-PCG driver using our dilu.amgx.python.Plan. Same problem geometry,
same RHS, same projection identities — only the linear solver changes.

Acceptance (design §6.4; thresholds calibrated against Phase 2 / Phase 3
at 64^3 per docs/benchmark/phase3_scaling_64_128.md §6):
  - Test A: max|∇·u| ≤ 4.1e-8 after projection. (Phase 2 baseline at 64^3:
    3.78e-8; Phase 3: 1.25e-8.)
  - Test B: ‖u‖∞ @ step 10 ≤ 3.0e-6; max|div u| ≤ 1e-12 per step.

Cross-phase dependency: reuses `dilu.cusparse.tests.physical_benchmark`
problem builders. Skipped when running standalone dilu.amgx packaging.
"""
import pytest

pytest.importorskip(
    "dilu.cusparse.tests.physical_benchmark",
    reason="dilu.cusparse not importable — running standalone dilu.amgx package",
)

import conftest  # noqa: F401

import os
import time
import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dilu.amgx.python import Plan as AmgxPlan, CLASSICAL_V_CYCLE, with_tolerance

# Reuse Phase 2.5 builders and face/divergence helpers unchanged.
from dilu.cusparse.tests.physical_benchmark import (
    smooth_sphere_indicator, density_from_indicator,
    build_variable_density_poisson_3d,
    cell_to_face, divergence_from_faces, apply_velocity_correction,
    curvature_csf, gradient_central,
    _random_solenoidal_u, _cell_centres,
)

# Config: 64^3, matching Phase 2/3 scale_64 runs.
N_GRID = 64
H_GRID = 1.0 / N_GRID
RHO_LIQUID = 1000.0
RHO_GAS = 1.0
R_DROPLET = 0.25
SIGMA = 0.07
DT = 1.0e-4
PCG_TOL = 1.0e-10

PLOTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "bench", "plots", "scale_64"))
os.makedirs(PLOTS_DIR, exist_ok=True)


def amg_solve_once(row_ptr, col_idx, values, rhs, max_iters=200, tol=PCG_TOL):
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    b_d = jax.device_put(jnp.asarray(rhs))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=tol, max_iters=max_iters)
    with AmgxPlan(rp, ci, vv, cfg) as plan:
        x_arr, iters_arr, status_arr = plan.solve(b_d)
        x_arr.block_until_ready()
        return (np.asarray(x_arr), int(iters_arr[0]), int(status_arr[0]))


def run_test_A():
    print("\n========== T12 Test A — AMG at 64^3 ==========")
    n = N_GRID; h = H_GRID
    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)

    u_star, v_star, w_star = _random_solenoidal_u(n, h, seed=42)
    uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
    div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)

    row_ptr, col_idx, values, _ = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))
    rhs = (-div_u_star / DT).reshape(-1)
    rhs[0] = 0.0

    t0 = time.time()
    p_flat, iters, status = amg_solve_once(row_ptr, col_idx, values, rhs)
    wall = time.time() - t0
    print(f"Test A: AMG-PCG iters={iters}, status={status}, wall={wall:.2f}s")
    assert status == 0, f"solve not converged: status={status}"

    p = p_flat.reshape((n, n, n))
    uf, vf, wf, u, v, w = apply_velocity_correction(
        u_star, v_star, w_star, p, rho, h, DT)
    div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)
    max_div = float(np.max(np.abs(div_u)))
    print(f"Test A: max|div u| post-projection = {max_div:.3e}")
    assert max_div <= 4.1e-8, f"max|div u| {max_div:.3e} > 4.1e-8 budget"

    # Plot
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    div_slice = div_u[:, :, zslice].T
    vmax = max(np.max(np.abs(div_slice)), 1e-16)
    im1 = ax1.imshow(div_slice, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
    ax1.contour(c[:, :, zslice].T, levels=[0.5], colors="k", linewidths=0.8,
                extent=[0, 1, 0, 1])
    ax1.set_title(f"AMG div(u) at z={zslice}/{n}\n"
                  f"max|div|_slice = {np.max(np.abs(div_slice)):.2e}")
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(np.log10(np.abs(div_u).ravel() + 1e-30), bins=60,
             color="steelblue", alpha=0.8)
    ax2.set_xlabel("log10 |div u|")
    ax2.set_ylabel("cell count")
    ax2.set_title(f"AMG: max|div|={max_div:.2e}, iters={iters}")
    fig.suptitle(f"T12 Test A (AMG, 64^3, rho-ratio {RHO_LIQUID/RHO_GAS:.0f})")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "divergence_map_AMG_64.png")
    fig.savefig(out); plt.close(fig)
    print(f"Wrote {out}")

    return {"max_div": max_div, "iters": iters, "wall": wall, "plot": out}


def run_test_B():
    print("\n========== T12 Test B — AMG at 64^3 ==========")
    n = N_GRID; h = H_GRID
    n_steps = 10

    c = smooth_sphere_indicator(n, h, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)
    inv_rho = 1.0 / rho

    kappa = curvature_csf(c, h)
    grad_cx, grad_cy, grad_cz = gradient_central(c, h)
    fsx = SIGMA * kappa * grad_cx
    fsy = SIGMA * kappa * grad_cy
    fsz = SIGMA * kappa * grad_cz

    row_ptr, col_idx, values, _ = build_variable_density_poisson_3d(
        rho, h, pin_cell=(0, 0, 0))

    u = np.zeros((n, n, n), dtype=np.float64)
    v = np.zeros_like(u); w = np.zeros_like(u)

    uinf_hist = []; div_hist = []; iter_hist = []
    t0 = time.time()

    # Cache device CSR + plan across steps (matrix is constant across the 10
    # steps since rho doesn't change).
    rp = jax.device_put(jnp.asarray(row_ptr))
    ci = jax.device_put(jnp.asarray(col_idx))
    vv = jax.device_put(jnp.asarray(values))
    cfg = with_tolerance(CLASSICAL_V_CYCLE, tol=PCG_TOL, max_iters=200)
    with AmgxPlan(rp, ci, vv, cfg) as plan:
        for step in range(n_steps):
            u_star = u + DT * fsx * inv_rho
            v_star = v + DT * fsy * inv_rho
            w_star = w + DT * fsz * inv_rho
            uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
            div_u_star = divergence_from_faces(uf_s, vf_s, wf_s, h, n, n, n)
            rhs = (-div_u_star / DT).reshape(-1)
            rhs[0] = 0.0

            b_d = jax.device_put(jnp.asarray(rhs))
            x_arr, iters_arr, status_arr = plan.solve(b_d)
            x_arr.block_until_ready()
            status = int(status_arr[0])
            assert status == 0, f"step {step} solve status={status}"
            iter_hist.append(int(iters_arr[0]))
            p = np.asarray(x_arr).reshape((n, n, n))

            uf, vf, wf, u, v, w = apply_velocity_correction(
                u_star, v_star, w_star, p, rho, h, DT)
            div_u = divergence_from_faces(uf, vf, wf, h, n, n, n)
            umag = np.sqrt(u * u + v * v + w * w)
            uinf = float(np.max(umag))
            divinf = float(np.max(np.abs(div_u)))
            uinf_hist.append(uinf); div_hist.append(divinf)
            print(f"  step {step+1}/{n_steps}: AMG iters={iter_hist[-1]}, "
                  f"||u||inf={uinf:.3e}, max|div|={divinf:.3e}")
    wall = time.time() - t0

    uinf_final = uinf_hist[-1]
    div_max = max(div_hist)
    print(f"Test B: ||u||inf step10 = {uinf_final:.3e}, max|div| = {div_max:.3e}, "
          f"wall = {wall:.2f}s, mean_iters = {np.mean(iter_hist):.1f}")
    assert uinf_final <= 3.0e-6, (
        f"||u||inf {uinf_final:.3e} > 3.0e-6 budget (Phase 2/3 at 64^3: 2.754e-6)")
    assert div_max <= 1e-12, f"max|div| {div_max:.3e} > 1e-12 budget"

    # Plot
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    zslice = n // 2
    u_sl = u[:, :, zslice]; v_sl = v[:, :, zslice]; c_sl = c[:, :, zslice]
    stride = max(1, n // 24)
    xs = _cell_centres(n, h)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    umag_sl = np.sqrt(u_sl**2 + v_sl**2)
    scale = max(np.max(umag_sl), 1e-30) * 20
    ax1.contour(XX, YY, c_sl, levels=[0.5], colors="k", linewidths=1.0)
    ax1.contourf(XX, YY, c_sl, levels=[0.5, 1.1], colors=["lightblue"], alpha=0.3)
    ax1.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
               u_sl[::stride, ::stride], v_sl[::stride, ::stride],
               umag_sl[::stride, ::stride], cmap="viridis",
               scale=scale, scale_units="xy", width=0.004)
    ax1.set_aspect("equal"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_title(f"AMG quiver z={zslice} after {n_steps} steps")

    ax2 = fig.add_subplot(1, 2, 2)
    steps = np.arange(1, n_steps + 1)
    ax2.semilogy(steps, uinf_hist, "-o", color="crimson")
    ax2.axhline(1e-12, color="k", linestyle=":", alpha=0.5)
    ax2.set_xlabel("step"); ax2.set_ylabel(r"$\|u\|_\infty$")
    ax2.set_title(f"AMG ||u||inf history (max iters/step = {max(iter_hist)})")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"T12 Test B (AMG, 64^3, sigma={SIGMA})")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "spurious_currents_AMG_64.png")
    fig.savefig(out); plt.close(fig)
    print(f"Wrote {out}")

    return {"uinf_final": uinf_final, "div_max": div_max,
            "iters_per_step": iter_hist, "wall": wall, "plot": out}


def test_t12_physical_64():
    resA = run_test_A()
    resB = run_test_B()
    print("\n========== T12 summary ==========")
    print(f"Test A: max|div u|={resA['max_div']:.3e}, "
          f"iters={resA['iters']}, wall={resA['wall']:.2f}s")
    print(f"Test B: ||u||inf step10={resB['uinf_final']:.3e}, "
          f"max|div|={resB['div_max']:.3e}, "
          f"mean iters/step={np.mean(resB['iters_per_step']):.1f}, "
          f"wall={resB['wall']:.2f}s")
    print("T12 PASS")


if __name__ == "__main__":
    test_t12_physical_64()
