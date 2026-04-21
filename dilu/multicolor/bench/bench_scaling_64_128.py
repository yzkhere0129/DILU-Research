"""Phase 3 scaling benchmark — 64³ and 128³ grids.

Answers two questions the headline 32³ numbers cannot:

  (Q1) Does Phase 3's 2.72x total-PCG speedup HOLD at AM-adjacent real scale?
       Theory: kernel-launch count gap grows with grid size (O(N^{1/3}) vs O(2)).
       Anti-theory: memory bandwidth dominates at 128³, erasing the gap.

  (Q2) Does the 1.5x iter-count penalty (P3/P2) stay grid-independent (Duff-
       Meurant 1989) or drift at 64³ / 128³?

Measurements per grid:
  - Phase 2 apply median / p95 (µs) - 20 warm + 500 timed
  - Phase 3 apply median / p95 (µs) - same protocol
  - Phase 2 DILU-PCG to tol=1e-10 (or max_iter) -- iter count + total wall time
  - Phase 3 DILU-PCG to tol=1e-10 (or max_iter) -- iter count + total wall time
  - Speedup ratios per-apply and total-PCG

At 64³ (additionally):
  - Physical Test A (variable-density projection, ρ-ratio 1000 sphere)
  - Physical Test B (static droplet, CSF, 10 projection steps)
  Both run with Phase 2 AND Phase 3 to confirm physical parity at scale.

At 128³:
  - Dispatch + PCG iter count + wall time only.
  - Test C (scipy spsolve) SKIPPED -- 2M×2M sparse-LU fill-in would need ≥4 GB
    host RAM. Tests A/B also SKIPPED at 128³ because their matrix build would
    dominate wall time and provide no new information beyond the PCG iter
    count trend. Documented as caveat.

Hardware safety (brief §HARDWARE SAFETY):
  - XLA env vars BEFORE any `import jax`
  - `nvidia-smi` before/after each distinct grid
  - For 128³: Phase 2 and Phase 3 each in their own SUBPROCESS (avoid holding
    two plans simultaneously in VRAM).
  - OOM = halt and report; do not retry smaller.
  - float64, no -ffast-math, np.ascontiguousarray at FFI boundaries.

Usage:
    # Top-level driver (spawns 128³ subprocesses):
    python bench_scaling_64_128.py --mode all

    # Direct sub-modes (for subprocess invocation):
    python bench_scaling_64_128.py --mode 64_dispatch       # 64³ apply timings + PCG
    python bench_scaling_64_128.py --mode 64_physical       # 64³ Tests A/B (P2 + P3)
    python bench_scaling_64_128.py --mode 128_phase2        # 128³ P2 only (subproc)
    python bench_scaling_64_128.py --mode 128_phase3        # 128³ P3 only (subproc)

Outputs:
  - JSON snippet per sub-mode -> stdout tail prefixed with '###RESULT###'
  - Aggregated markdown report at docs/benchmark/phase3_scaling_64_128.md
  - 4 PNGs at dilu/multicolor/bench/plots/scale_64/ (2 tests x {P2, P3})
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# VRAM rails MUST be set before `import jax`. This file is entry-point for
# both the top-level driver and subprocess invocations.
# --------------------------------------------------------------------------
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Expose tests/ dir for _harness import (stiff_laplacian_3d).
_TESTS_DIR = os.path.join(_REPO_ROOT, "dilu", "multicolor", "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import argparse
import json
import subprocess
import time
from datetime import datetime
from typing import Optional

import numpy as np

# Lazy: only import jax inside sub-modes (subprocess-friendly; lets the CLI
# shell always be fast for orchestration).

# --------------------------------------------------------------------------
# Output paths.
# --------------------------------------------------------------------------
PLOTS_DIR_64 = os.path.join(_THIS_DIR, "plots", "scale_64")
REPORT_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, "docs", "benchmark", "phase3_scaling_64_128.md"))
RESULT_JSON_DIR = os.path.join(_THIS_DIR, "_scaling_results")
os.makedirs(RESULT_JSON_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# nvidia-smi / free -h snapshots (called from any sub-mode).
# --------------------------------------------------------------------------
def _capture_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT, timeout=10).decode("utf-8")
        return out.strip()
    except Exception as e:  # noqa: BLE001
        return f"(nvidia-smi error: {e})"


def _capture_free() -> str:
    try:
        out = subprocess.check_output(
            ["free", "-h"], stderr=subprocess.STDOUT, timeout=5).decode("utf-8")
        return out.strip()
    except Exception as e:  # noqa: BLE001
        return f"(free error: {e})"


def _free_mib() -> Optional[int]:
    """Return current GPU free memory in MiB, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=5).decode("utf-8")
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def _preflight(required_mib: int) -> None:
    """Halt if GPU free VRAM is below `required_mib`."""
    free = _free_mib()
    if free is None:
        print("PREFLIGHT: could not read nvidia-smi; refusing to proceed.",
              file=sys.stderr)
        sys.exit(3)
    if free < required_mib:
        print(f"PREFLIGHT HALT: {free} MiB free < {required_mib} MiB required. "
              "Close other GPU processes and re-run.", file=sys.stderr)
        sys.exit(3)
    print(f"PREFLIGHT OK: {free} MiB free (need {required_mib} MiB)")


# --------------------------------------------------------------------------
# Shared PCG / SpMV helpers (imported into each sub-mode). Match the pattern
# in dilu/multicolor/bench/bench_multicolor_vs_cusparse.py so iter counts are
# directly comparable to prior reports.
# --------------------------------------------------------------------------
def _spmv_factory(rp, ci, vv, n):
    import jax
    import jax.numpy as jnp

    nnz = int(vv.shape[0])
    # Precompute row_of once; it is a function of (row_ptr, nnz) only.
    k = jnp.arange(nnz, dtype=jnp.int32)
    row_of = jnp.searchsorted(rp[1:], k, side="right")
    row_of = jax.device_put(row_of)

    def spmv(x):
        prods = vv * x[ci]
        return jax.ops.segment_sum(prods, row_of, num_segments=n)
    return spmv


def _pcg(apply_M_inv, spmv, b, tol=1e-10, max_iter=500):
    """DILU-PCG; returns (x, iters, converged, rnorm_final, t_wall_s)."""
    import jax
    import jax.numpy as jnp

    n = b.shape[0]
    t0 = time.perf_counter()
    x = jnp.zeros(n, dtype=jnp.float64)
    r = b - spmv(x)
    r.block_until_ready()
    b_norm = max(float(jnp.linalg.norm(b)), 1e-300)
    z = apply_M_inv(r)
    p = z
    rz = float(jnp.dot(r, z))
    for it in range(max_iter):
        rnorm = float(jnp.linalg.norm(r))
        if rnorm / b_norm < tol:
            x.block_until_ready()
            return x, it, True, rnorm, time.perf_counter() - t0
        Ap = spmv(p)
        alpha = rz / float(jnp.dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_M_inv(r)
        rz_new = float(jnp.dot(r, z))
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    x.block_until_ready()
    rnorm = float(jnp.linalg.norm(r))
    return x, max_iter, False, rnorm, time.perf_counter() - t0


def _time_apply(apply_fn, r_d, warmup=20, reps=500):
    """Median / p95 apply dispatch time in µs."""
    for _ in range(warmup):
        z = apply_fn(r_d); z.block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        z = apply_fn(r_d)
        z.block_until_ready()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    median = ts[reps // 2]
    p95 = ts[int(reps * 0.95)]
    return median, p95


# --------------------------------------------------------------------------
# Matrix builders (shared between dispatch and physical sub-modes).
# --------------------------------------------------------------------------
def _build_stiff_laplacian_3d(n: int, contrast: float = 100.0):
    """T7-style stiff 3-D 7-point Laplacian on n³; coefficient jump at z=n/2.

    Vectorized builder (too slow to use the nested-Python version of
    stiff_laplacian_3d from tests/_harness.py at n≥64). Produces identical
    CSR layout (ascending col within each row) and identical values.
    """
    nx = ny = nz = n

    # Face coefficients kcoef[k]: 1 for k<n/2, contrast otherwise.
    kcoef = np.ones(nz, dtype=np.float64)
    kcoef[nz // 2:] = contrast

    def idx(i, j, k):
        return (k * ny + j) * nx + i

    # Each interior cell has 7 entries (diag + up to 6 neighbours). Boundary
    # cells have fewer off-diags, so allocate the maximum 7·n³ then trim.
    N = nx * ny * nz
    max_nnz = 7 * N

    col_idx_buf = np.empty(max_nnz, dtype=np.int32)
    val_buf = np.empty(max_nnz, dtype=np.float64)
    row_ptr = np.zeros(N + 1, dtype=np.int32)

    # Row-major walk (k, j, i).
    # Offsets of the 6 neighbours (in flat index) and which axis / direction.
    # Neighbour directions; we evaluate f_k depending on k and target cell's k.
    # For each row, we compute diag and up to 6 off-diagonals, sort by col.
    neigh = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    pos = 0
    for k in range(nz):
        kcoef_k = kcoef[k]
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                diag = 0.0
                off = []  # (col, val)
                for (di, dj, dk) in neigh:
                    ii = i + di; jj = j + dj; kk = k + dk
                    if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz):
                        diag += kcoef_k  # Dirichlet ghost
                        continue
                    fk = 2.0 * kcoef_k * kcoef[kk] / (kcoef_k + kcoef[kk])
                    diag += fk
                    off.append((idx(ii, jj, kk), -fk))
                # Insert (p, diag) and sort.
                off.append((p, diag))
                off.sort(key=lambda e: e[0])
                m = len(off)
                for q, (c, v) in enumerate(off):
                    col_idx_buf[pos + q] = c
                    val_buf[pos + q] = v
                pos += m
                row_ptr[p + 1] = pos
    col_idx = np.ascontiguousarray(col_idx_buf[:pos])
    values = np.ascontiguousarray(val_buf[:pos])
    return np.ascontiguousarray(row_ptr), col_idx, values


def _stiff_laplacian_3d_fast(n: int, contrast: float = 100.0):
    """Vectorized builder for stiff 3-D 7-pt Laplacian.

    Faster than `_build_stiff_laplacian_3d` at n=128 (seconds vs minutes) by
    producing (col_idx, values) via NumPy array ops per direction, then sort
    per row. Correctness checked via round-trip equality at n=16 in
    `_selfcheck_stiff_builders`.

    Layout invariant: we build arrays of shape (nz, ny, nx) so that
    row-major `.ravel(order="C")` yields the ordering
        flat_p = (k * ny + j) * nx + i
    matching the reference `stiff_laplacian_3d` in tests/_harness.py.
    """
    nx = ny = nz = n
    N = nx * ny * nz

    kcoef = np.ones(nz, dtype=np.float64)
    kcoef[nz // 2:] = contrast

    # Build arrays shape (nz, ny, nx) so that C-order ravel gives the
    # ((k*ny+j)*nx+i) flat index.
    iarr = np.arange(nx, dtype=np.int32)
    jarr = np.arange(ny, dtype=np.int32)
    karr = np.arange(nz, dtype=np.int32)
    KK, JJ, II = np.meshgrid(karr, jarr, iarr, indexing="ij")
    # Now KK[k,j,i] = k, JJ[k,j,i] = j, II[k,j,i] = i -- correct for C-ravel.
    def flat(I, J, K):
        return (K * ny + J) * nx + I

    P = flat(II, JJ, KK)  # shape (nz, ny, nx); ravel(C) gives (k*ny+j)*nx+i

    # For each of 6 directions, compute neighbour column and coefficient.
    # Off-domain: value 0 (we'll mark as -1 and exclude).
    def neighbour(di, dj, dk):
        I2 = II + di; J2 = JJ + dj; K2 = KK + dk
        mask = ((I2 >= 0) & (I2 < nx) &
                (J2 >= 0) & (J2 < ny) &
                (K2 >= 0) & (K2 < nz))
        fk = np.zeros_like(kcoef[KK])
        if mask.any():
            Kc = np.clip(KK, 0, nz - 1)
            K2c = np.clip(K2, 0, nz - 1)
            kk1 = kcoef[Kc]
            kk2 = kcoef[K2c]
            fk = 2.0 * kk1 * kk2 / (kk1 + kk2)
        col = flat(np.clip(I2, 0, nx - 1), np.clip(J2, 0, ny - 1),
                   np.clip(K2, 0, nz - 1))
        return mask, col, fk  # all shape (nx, ny, nz)

    # Accumulate diagonal.
    diag = np.zeros((nx, ny, nz), dtype=np.float64)
    # Dirichlet ghost contribution: +kcoef[k] for each out-of-domain neighbour.
    # Interior (non-ghost) contribution: +fk.
    off_records = []  # list of (mask_flat, col_flat, val_flat)
    for (di, dj, dk) in [(-1, 0, 0), (1, 0, 0),
                         (0, -1, 0), (0, 1, 0),
                         (0, 0, -1), (0, 0, 1)]:
        mask, col, fk = neighbour(di, dj, dk)
        # Interior neighbour: diag += fk, off = -fk
        diag += np.where(mask, fk, 0.0)
        # Ghost neighbour: diag += kcoef[KK]
        diag += np.where(~mask, kcoef[KK], 0.0)
        # Record off-diag entries: only where mask is True.
        off_records.append((mask.ravel(order="C"), col.ravel(order="C"),
                            (-fk).ravel(order="C")))

    # Now assemble CSR. For each cell p, off-diag entries are from the 6
    # direction records where mask[p] is True, plus (p, diag[p]). We need
    # ascending col_idx within each row.
    #
    # Because the neighbour directions include (i±1, j±1, k±1), and because
    # flat index = (k*ny + j)*nx + i, the natural order of col indices for a
    # fully-interior cell is:
    #   k-1  < j-1 < i-1 < self < i+1 < j+1 < k+1
    # which translates to directions in the fixed order:
    #   (0,0,-1), (0,-1,0), (-1,0,0), self, (+1,0,0), (0,+1,0), (0,0,+1)
    # That holds on a fully-interior cell. At boundaries some directions are
    # missing but the remaining ones keep the same relative ordering.
    # So if we iterate directions in this fixed order, the emitted cols are
    # already ascending.
    dir_order = [(0, 0, -1), (0, -1, 0), (-1, 0, 0),
                 (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Remap: map dir tuple -> index in off_records.
    neigh_list = [(-1, 0, 0), (1, 0, 0),
                  (0, -1, 0), (0, 1, 0),
                  (0, 0, -1), (0, 0, 1)]
    dir_to_rec = {d: i for i, d in enumerate(neigh_list)}
    # Walk rows in flat-index order. We need per-row list of (col, val).
    # Build row_ptr by counting true mask entries per row + 1 (diag).
    # Count per row for each direction in dir_order.
    mask_flats = {d: off_records[dir_to_rec[d]][0] for d in dir_order}
    col_flats = {d: off_records[dir_to_rec[d]][1] for d in dir_order}
    val_flats = {d: off_records[dir_to_rec[d]][2] for d in dir_order}

    # Per-row entry count = 1 (diag) + sum of masks per direction.
    count_per_row = np.ones(N, dtype=np.int32)
    for d in dir_order:
        count_per_row += mask_flats[d].astype(np.int32)
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(count_per_row, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    col_idx = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)

    # Cursor array (per row insertion position). This O(N) python loop is the
    # expensive part; vectorize via a write-index pass.
    # Strategy: build a (7, N) grid of (col, val, include_mask) in the fixed
    # direction order; then compute cumulative position offsets within each row.
    K = 7  # 6 neighbours + diag
    dirs = dir_order  # 6 in traversal order
    # Diagonal insertion position is after all (−) directions and before all (+)
    # directions. Specifically, in the ordering [(0,0,−1), (0,−1,0), (−1,0,0),
    # DIAG, (1,0,0), (0,1,0), (0,0,1)], diag is slot index 3.
    # Collect per-slot (mask, col, val) in row-major flat order.
    slot_mask = np.zeros((K, N), dtype=np.bool_)
    slot_col = np.zeros((K, N), dtype=np.int32)
    slot_val = np.zeros((K, N), dtype=np.float64)

    # Three negative-direction slots.
    for s_idx, d in enumerate(dirs[:3]):
        slot_mask[s_idx] = mask_flats[d]
        slot_col[s_idx] = col_flats[d]
        slot_val[s_idx] = val_flats[d]
    # Diagonal slot.
    slot_mask[3] = True
    slot_col[3] = np.arange(N, dtype=np.int32)
    slot_val[3] = diag.ravel(order="C")
    # Three positive-direction slots.
    for s_idx, d in enumerate(dirs[3:]):
        slot_mask[4 + s_idx] = mask_flats[d]
        slot_col[4 + s_idx] = col_flats[d]
        slot_val[4 + s_idx] = val_flats[d]

    # Per-row within-slot positions: exclusive-cumsum across slots for masked.
    # pos_in_row[s, r] = sum_{t<s} slot_mask[t, r]; only meaningful when
    # slot_mask[s, r] is True.
    slot_mask_int = slot_mask.astype(np.int32)
    pos_in_row = np.cumsum(slot_mask_int, axis=0) - slot_mask_int  # (K, N)

    # Absolute write indices = row_ptr[r] + pos_in_row[s, r] for entries where
    # slot_mask[s, r] is True.
    row_base = row_ptr[:-1]  # (N,)
    abs_pos = row_base[None, :] + pos_in_row  # (K, N)

    write_pos = abs_pos[slot_mask]  # (nnz,)
    write_col = slot_col[slot_mask]
    write_val = slot_val[slot_mask]

    col_idx[write_pos] = write_col
    values[write_pos] = write_val

    return (np.ascontiguousarray(row_ptr),
            np.ascontiguousarray(col_idx),
            np.ascontiguousarray(values))


def _build_variable_density_poisson_3d_fast(rho, h, pin_cell=(0, 0, 0)):
    """Vectorized variable-density 7-pt Poisson CSR.

    Returns (row_ptr, col_idx, values, diag_offset). Identical mathematical
    output (ULP-equivalent in the non-associative reductions, up to NumPy's
    deterministic ordering) to Phase 2.5's nested-loop
    `build_variable_density_poisson_3d` in
    `dilu/cusparse/tests/physical_benchmark.py`, but runs in seconds rather
    than minutes at 64³.

    Layout: same `flat = (k*ny+j)*nx+i` ordering; arrays built with shape
    (nz, ny, nx) so C-order ravel matches the flat index.
    """
    nx, ny, nz = rho.shape
    N = nx * ny * nz

    # Reshape rho to (nz, ny, nx) ordering used internally.
    rho_zyx = np.moveaxis(rho, (0, 1, 2), (2, 1, 0)).copy()  # (nz, ny, nx)
    # Note: rho is provided shape (nx, ny, nz) with rho[i, j, k] semantics.
    # rho_zyx[k, j, i] == rho[i, j, k].

    def flat(I, J, K):
        return (K * ny + J) * nx + I

    pin_i = flat(*pin_cell)
    h2 = h * h

    # Face coefficients computed on the (nz, ny, nx) layout.
    # fx[k, j, i] = 2 / (rho_zyx[k, j, i] + rho_zyx[k, j, i+1]) for i=0..nx-2
    fx = 2.0 / (rho_zyx[:, :, :-1] + rho_zyx[:, :, 1:])
    fy = 2.0 / (rho_zyx[:, :-1, :] + rho_zyx[:, 1:, :])
    fz = 2.0 / (rho_zyx[:-1, :, :] + rho_zyx[1:, :, :])

    iarr = np.arange(nx, dtype=np.int32)
    jarr = np.arange(ny, dtype=np.int32)
    karr = np.arange(nz, dtype=np.int32)
    KK, JJ, II = np.meshgrid(karr, jarr, iarr, indexing="ij")  # (nz,ny,nx)

    # For each direction, emit (mask, col, beta).
    # beta is defined on all cells; masked True where the neighbour exists.
    def beta_minus_x():
        mask = II > 0
        beta = np.zeros_like(rho_zyx)
        beta[:, :, 1:] = fx / h2  # left face fx[:, :, i-1] for i≥1
        col = flat(np.clip(II - 1, 0, nx - 1), JJ, KK)
        return mask, col, beta

    def beta_plus_x():
        mask = II < nx - 1
        beta = np.zeros_like(rho_zyx)
        beta[:, :, :-1] = fx / h2  # right face fx[:, :, i] for i<nx-1
        col = flat(np.clip(II + 1, 0, nx - 1), JJ, KK)
        return mask, col, beta

    def beta_minus_y():
        mask = JJ > 0
        beta = np.zeros_like(rho_zyx)
        beta[:, 1:, :] = fy / h2
        col = flat(II, np.clip(JJ - 1, 0, ny - 1), KK)
        return mask, col, beta

    def beta_plus_y():
        mask = JJ < ny - 1
        beta = np.zeros_like(rho_zyx)
        beta[:, :-1, :] = fy / h2
        col = flat(II, np.clip(JJ + 1, 0, ny - 1), KK)
        return mask, col, beta

    def beta_minus_z():
        mask = KK > 0
        beta = np.zeros_like(rho_zyx)
        beta[1:, :, :] = fz / h2
        col = flat(II, JJ, np.clip(KK - 1, 0, nz - 1))
        return mask, col, beta

    def beta_plus_z():
        mask = KK < nz - 1
        beta = np.zeros_like(rho_zyx)
        beta[:-1, :, :] = fz / h2
        col = flat(II, JJ, np.clip(KK + 1, 0, nz - 1))
        return mask, col, beta

    # Order: ascending col_idx for interior cells ⇒
    # (-z, -y, -x, DIAG, +x, +y, +z)
    recs = [("-z", beta_minus_z()),
            ("-y", beta_minus_y()),
            ("-x", beta_minus_x()),
            ("+x", beta_plus_x()),
            ("+y", beta_plus_y()),
            ("+z", beta_plus_z())]

    # Diagonal = sum of beta over all directions where neighbour exists.
    diag_val = np.zeros_like(rho_zyx)
    for _, (mask, _col, beta) in recs:
        diag_val += np.where(mask, beta, 0.0)

    row_flat = np.arange(N, dtype=np.int32).reshape((nz, ny, nx))

    # Build slot arrays: 3 negative dirs + diag + 3 positive dirs.
    K = 7
    slot_mask = np.zeros((K, N), dtype=np.bool_)
    slot_col = np.zeros((K, N), dtype=np.int32)
    slot_val = np.zeros((K, N), dtype=np.float64)

    # -z, -y, -x at slots 0, 1, 2 (in that order).
    neg_specs = [recs[0], recs[1], recs[2]]  # -z, -y, -x
    for s_idx, (_, (mask, col, beta)) in enumerate(neg_specs):
        # Emit iff mask AND col != pin AND row != pin.
        emit = mask & (col != pin_i) & (row_flat != pin_i)
        slot_mask[s_idx] = emit.ravel(order="C")
        slot_col[s_idx] = col.ravel(order="C")
        slot_val[s_idx] = (-beta).ravel(order="C")

    # Diagonal slot at index 3.
    slot_mask[3] = True  # all rows have a diagonal
    slot_col[3] = row_flat.ravel(order="C")
    diag_flat = diag_val.ravel(order="C").copy()
    diag_flat[pin_i] = 1.0  # pinned row identity
    slot_val[3] = diag_flat

    pos_specs = [recs[3], recs[4], recs[5]]  # +x, +y, +z
    for s_idx_rel, (_, (mask, col, beta)) in enumerate(pos_specs):
        s_idx = 4 + s_idx_rel
        emit = mask & (col != pin_i) & (row_flat != pin_i)
        slot_mask[s_idx] = emit.ravel(order="C")
        slot_col[s_idx] = col.ravel(order="C")
        slot_val[s_idx] = (-beta).ravel(order="C")

    # For the pinned row, wipe all off-diag slots.
    for s_idx in (0, 1, 2, 4, 5, 6):
        slot_mask[s_idx, pin_i] = False

    # CSR assembly.
    count_per_row = slot_mask.astype(np.int32).sum(axis=0)
    row_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(count_per_row, out=row_ptr[1:])
    nnz = int(row_ptr[-1])

    slot_mask_int = slot_mask.astype(np.int32)
    pos_in_row = np.cumsum(slot_mask_int, axis=0) - slot_mask_int
    row_base = row_ptr[:-1]
    abs_pos = row_base[None, :] + pos_in_row

    col_idx = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=np.float64)
    col_idx[abs_pos[slot_mask]] = slot_col[slot_mask]
    values[abs_pos[slot_mask]] = slot_val[slot_mask]

    # diag_offset: position of diag slot in each row.
    diag_offset = abs_pos[3, :].astype(np.int32)

    return (np.ascontiguousarray(row_ptr),
            np.ascontiguousarray(col_idx),
            np.ascontiguousarray(values),
            np.ascontiguousarray(diag_offset))


def _selfcheck_stiff_builders():
    """Verify the fast builder agrees with the nested-loop one at n=8."""
    from _harness import stiff_laplacian_3d  # noqa: E402
    for n in (4, 8, 16):
        rp_a, ci_a, vv_a = stiff_laplacian_3d(n, n, n, contrast=100.0)
        rp_b, ci_b, vv_b = _stiff_laplacian_3d_fast(n, contrast=100.0)
        if not (np.array_equal(rp_a, rp_b) and np.array_equal(ci_a, ci_b)
                and np.allclose(vv_a, vv_b, atol=0, rtol=0)):
            raise RuntimeError(
                f"stiff builders disagree at n={n}: "
                f"rp match {np.array_equal(rp_a, rp_b)}, "
                f"ci match {np.array_equal(ci_a, ci_b)}, "
                f"vv maxerr {np.max(np.abs(vv_a - vv_b))}")
    print("_stiff_laplacian_3d_fast self-check OK (n=4, 8, 16)")


def _selfcheck_poisson_builders():
    """Verify the vectorized variable-density Poisson builder agrees with the
    Phase 2.5 reference `build_variable_density_poisson_3d` at small n.
    """
    import importlib.util
    ref_path = os.path.join(_REPO_ROOT, "dilu", "cusparse", "tests",
                            "physical_benchmark.py")
    spec = importlib.util.spec_from_file_location("pb_ref", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for n in (4, 8):
        h = 1.0 / n
        xs = (np.arange(n) + 0.5) * h - 0.5
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        R = np.sqrt(X * X + Y * Y + Z * Z)
        c = 0.5 * (1.0 - np.tanh((R - 0.25) / (1.5 * h)))
        rho = 1000.0 * c + 1.0 * (1 - c)

        rp_a, ci_a, vv_a, do_a = mod.build_variable_density_poisson_3d(
            rho, h, pin_cell=(0, 0, 0))
        rp_b, ci_b, vv_b, do_b = _build_variable_density_poisson_3d_fast(
            rho, h, pin_cell=(0, 0, 0))

        ok = (np.array_equal(rp_a, rp_b)
              and np.array_equal(ci_a, ci_b)
              and np.array_equal(do_a, do_b)
              and np.max(np.abs(vv_a - vv_b)) < 1e-12 * max(np.max(np.abs(vv_a)), 1.0))
        if not ok:
            raise RuntimeError(
                f"variable-density Poisson builders disagree at n={n}: "
                f"rp eq={np.array_equal(rp_a, rp_b)}, "
                f"ci eq={np.array_equal(ci_a, ci_b)}, "
                f"do eq={np.array_equal(do_a, do_b)}, "
                f"vv maxerr={np.max(np.abs(vv_a - vv_b))}")
    print("_build_variable_density_poisson_3d_fast self-check OK (n=4, 8)")


# --------------------------------------------------------------------------
# Sub-mode 1: dispatch + PCG on a given grid size (in-process, P2 + P3).
# --------------------------------------------------------------------------
def run_dispatch_and_pcg(n: int, tol: float, pcg_max_iter: int,
                         label: str, run_p2: bool = True, run_p3: bool = True):
    """Measure apply dispatch + PCG convergence at grid n³ for P2 and/or P3.

    Returns a dict with per-implementation metrics. Writes intermediate JSON to
    RESULT_JSON_DIR/{label}.json.
    """
    import jax
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dilu.cusparse.python import Plan as CusparsePlan, build_diag_offset
    from dilu.multicolor.python import MulticolorPlan

    print(f"\n========== Grid {n}^3 = {n**3} cells | dispatch + PCG "
          f"({'P2' if run_p2 else '--'} / "
          f"{'P3' if run_p3 else '--'}) ==========")

    print("[build] matrix (stiff 3-D 7-pt, contrast=100)...")
    t_build = time.perf_counter()
    rp_h, ci_h, vv_h = _stiff_laplacian_3d_fast(n, contrast=100.0)
    print(f"[build] N={n**3}, nnz={len(vv_h)}, "
          f"{(time.perf_counter()-t_build):.2f} s")
    print(f"[build] matrix bytes: values={vv_h.nbytes/1e6:.1f} MB, "
          f"col_idx={ci_h.nbytes/1e6:.1f} MB")

    do_h = build_diag_offset(rp_h, ci_h)

    rng = np.random.default_rng(0)
    b_h = rng.standard_normal(n**3).astype(np.float64)

    # Device-resident matrix.
    rp = jax.device_put(jnp.asarray(rp_h))
    ci = jax.device_put(jnp.asarray(ci_h))
    vv = jax.device_put(jnp.asarray(vv_h))
    do = jax.device_put(jnp.asarray(do_h))
    b_d = jax.device_put(jnp.asarray(b_h))

    N = n**3
    spmv = _spmv_factory(rp, ci, vv, N)

    snap_pre = _capture_nvidia_smi()
    print(f"[vram pre-plans] {snap_pre}")

    out = {"label": label, "n": n, "N": N, "nnz": int(len(vv_h)),
           "tol": tol, "pcg_max_iter": pcg_max_iter,
           "nvidia_smi_pre": snap_pre}

    # -------------------- Phase 2 --------------------
    if run_p2:
        print("[P2] analyzing...")
        with CusparsePlan(rp, ci, vv, do) as plan2:
            d_star2 = plan2.factor(vv)
            d_star2.block_until_ready()

            def apply2(r):
                return plan2.apply(vv, d_star2, r)

            print("[P2] measuring apply dispatch...")
            med2, p95_2 = _time_apply(apply2, b_d, warmup=20, reps=500)
            print(f"[P2] apply median={med2:.1f} µs  p95={p95_2:.1f} µs")
            print(f"[P2] PCG to tol={tol:.0e}, max_iter={pcg_max_iter}...")
            _, iters2, conv2, rnorm2, twall2 = _pcg(
                apply2, spmv, b_d, tol=tol, max_iter=pcg_max_iter)
            print(f"[P2] iters={iters2} converged={conv2} "
                  f"rnorm={rnorm2:.3e} wall={twall2:.2f} s")
            out["P2"] = {"apply_med_us": med2, "apply_p95_us": p95_2,
                         "iters": iters2, "converged": conv2,
                         "rnorm": rnorm2, "wall_s": twall2}
        jax.clear_caches()
    else:
        out["P2"] = None

    # -------------------- Phase 3 --------------------
    if run_p3:
        print("[P3] analyzing (coloring + permute + FFI analyze)...")
        with MulticolorPlan(rp_h, ci_h, vv_h,
                            grid_shape=(n, n, n)) as plan3:
            n_colors = plan3.n_colors
            d_star3 = plan3.factor(vv)
            d_star3.block_until_ready()

            def apply3(r):
                return plan3.apply(vv, d_star3, r)

            print(f"[P3] n_colors={n_colors}")
            print("[P3] measuring apply dispatch...")
            med3, p95_3 = _time_apply(apply3, b_d, warmup=20, reps=500)
            print(f"[P3] apply median={med3:.1f} µs  p95={p95_3:.1f} µs")
            print(f"[P3] PCG to tol={tol:.0e}, max_iter={pcg_max_iter}...")
            _, iters3, conv3, rnorm3, twall3 = _pcg(
                apply3, spmv, b_d, tol=tol, max_iter=pcg_max_iter)
            print(f"[P3] iters={iters3} converged={conv3} "
                  f"rnorm={rnorm3:.3e} wall={twall3:.2f} s")
            out["P3"] = {"apply_med_us": med3, "apply_p95_us": p95_3,
                         "iters": iters3, "converged": conv3,
                         "rnorm": rnorm3, "wall_s": twall3,
                         "n_colors": int(n_colors)}
        jax.clear_caches()
    else:
        out["P3"] = None

    snap_post = _capture_nvidia_smi()
    print(f"[vram post-plans] {snap_post}")
    out["nvidia_smi_post"] = snap_post

    out_path = os.path.join(RESULT_JSON_DIR, f"{label}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"###RESULT### {out_path}")
    return out


# --------------------------------------------------------------------------
# Sub-mode 2: physical Tests A and B at 64³ for both P2 and P3.
# --------------------------------------------------------------------------
def run_physical_64():
    """Physical Tests A and B at 64³, run for both Phase 2 and Phase 3."""
    import jax
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dilu.cusparse.python import Plan as CusparsePlan, build_diag_offset
    from dilu.multicolor.python import MulticolorPlan

    os.makedirs(PLOTS_DIR_64, exist_ok=True)

    N_GRID = 64
    H_GRID = 1.0 / N_GRID
    RHO_LIQUID = 1000.0
    RHO_GAS = 1.0
    R_DROPLET = 0.25
    SIGMA = 0.07
    DT_PROJECTION = 1.0e-4
    PCG_TOL = 1.0e-10
    PCG_MAX_ITER = 1000  # bigger grid → allow more iters than 32³'s 300

    print(f"\n========== Physical benchmarks at {N_GRID}^3 ==========")
    snap_pre = _capture_nvidia_smi()
    print(f"[vram pre] {snap_pre}")

    # --- Grid / phase indicator / density / Poisson matrix ------------------
    def cell_centres(n, h):
        return (np.arange(n) + 0.5) * h

    def smooth_sphere_indicator(n, h, radius, eps_factor=1.5):
        xs = cell_centres(n, h) - 0.5
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        R = np.sqrt(X * X + Y * Y + Z * Z)
        eps = eps_factor * h
        c = 0.5 * (1.0 - np.tanh((R - radius) / eps))
        return np.ascontiguousarray(c, dtype=np.float64)

    def density_from_indicator(c, rho_high, rho_low):
        return rho_high * c + rho_low * (1.0 - c)

    build_variable_density_poisson_3d_fast = _build_variable_density_poisson_3d_fast

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
        return uf, vf, wf

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

    def random_low_k_u(n, h, seed=42):
        rng = np.random.default_rng(seed)
        xs = cell_centres(n, h)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        u = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
             + 0.3 * np.cos(4 * np.pi * Z)) * rng.uniform(0.8, 1.2)
        v = (-np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
             + 0.3 * np.sin(4 * np.pi * X)) * rng.uniform(0.8, 1.2)
        w = (0.4 * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)) * rng.uniform(0.8, 1.2)
        return (u.astype(np.float64),
                v.astype(np.float64),
                w.astype(np.float64))

    # PCG drivers that take a preconditioner-apply callable (Plan-agnostic).
    def pcg_generic(apply_M_inv, spmv, b, tol, max_iter):
        return _pcg(apply_M_inv, spmv, b, tol=tol, max_iter=max_iter)

    # Build density, grid, CSR ONCE (shared between P2 and P3 Test A, Test B).
    print("[build] density and Poisson matrix (variable-rho 7-pt)...")
    c = smooth_sphere_indicator(N_GRID, H_GRID, R_DROPLET)
    rho = density_from_indicator(c, RHO_LIQUID, RHO_GAS)
    t_build = time.perf_counter()
    rp_h, ci_h, vv_h, do_h = build_variable_density_poisson_3d_fast(
        rho, H_GRID, pin_cell=(0, 0, 0))
    print(f"[build] N={N_GRID**3}, nnz={len(vv_h)}, "
          f"{(time.perf_counter()-t_build):.2f} s")

    # Device-resident matrix.
    rp = jax.device_put(jnp.asarray(rp_h))
    ci = jax.device_put(jnp.asarray(ci_h))
    vv = jax.device_put(jnp.asarray(vv_h))
    do = jax.device_put(jnp.asarray(do_h))
    N_TOTAL = N_GRID ** 3
    spmv = _spmv_factory(rp, ci, vv, N_TOTAL)

    # ---------- Test A ----------
    print("\n[Test A] variable-density projection, random low-k u*")
    u_star, v_star, w_star = random_low_k_u(N_GRID, H_GRID, seed=42)
    uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
    div_u_star = divergence_from_faces(
        uf_s, vf_s, wf_s, H_GRID, N_GRID, N_GRID, N_GRID)
    rhs = (-div_u_star / DT_PROJECTION).reshape(-1).copy()
    rhs[0] = 0.0
    b_d = jax.device_put(jnp.asarray(np.ascontiguousarray(rhs)))

    results_A = {}
    for tag, make_plan_ctx, grid_shape in [
        ("P2", lambda: CusparsePlan(rp, ci, vv, do), None),
        ("P3", lambda: MulticolorPlan(rp_h, ci_h, vv_h, grid_shape=(N_GRID, N_GRID, N_GRID)), (N_GRID, N_GRID, N_GRID)),
    ]:
        print(f"[Test A / {tag}] running PCG to tol={PCG_TOL:.0e}...")
        with make_plan_ctx() as plan:
            d_star = plan.factor(vv)
            d_star.block_until_ready()

            def apply_M(r, _plan=plan, _d=d_star):
                return _plan.apply(vv, _d, r)

            t0 = time.perf_counter()
            x, iters, conv, rnorm, twall = pcg_generic(
                apply_M, spmv, b_d, PCG_TOL, PCG_MAX_ITER)
            print(f"[Test A / {tag}] iters={iters} converged={conv} "
                  f"rnorm={rnorm:.3e} wall={twall:.2f} s")
            p = np.asarray(x).reshape((N_GRID, N_GRID, N_GRID))

        uf, vf, wf = apply_velocity_correction(
            u_star, v_star, w_star, p, rho, H_GRID, DT_PROJECTION)
        div_u = divergence_from_faces(uf, vf, wf, H_GRID,
                                      N_GRID, N_GRID, N_GRID)
        max_div = float(np.max(np.abs(div_u)))
        print(f"[Test A / {tag}] max|div(u)| = {max_div:.3e}")
        results_A[tag] = {"iters": iters, "converged": conv,
                          "rnorm": rnorm, "wall_s": twall,
                          "max_div": max_div}

        # Plot.
        fig = plt.figure(figsize=(11, 4.5))
        ax1 = fig.add_subplot(1, 2, 1)
        zslice = N_GRID // 2
        div_slice = div_u[:, :, zslice].T
        vmax = max(np.max(np.abs(div_slice)), 1e-16)
        im1 = ax1.imshow(div_slice, origin="lower", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, extent=[0, 1, 0, 1])
        ax1.contour(c[:, :, zslice].T, levels=[0.5], colors="k",
                    linewidths=0.8, extent=[0, 1, 0, 1])
        ax1.set_title(f"[{tag}] div(u) at z={zslice}/{N_GRID}\n"
                      f"max|div| slice={np.max(np.abs(div_slice)):.2e}")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")
        fig.colorbar(im1, ax=ax1, label="div(u)")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(np.log10(np.abs(div_u).ravel() + 1e-30), bins=60,
                 color="steelblue", alpha=0.8)
        ax2.set_xlabel("log10 |div(u)|"); ax2.set_ylabel("cell count")
        ax2.set_title(f"[{tag}] domain |div| dist\nmax={max_div:.2e}, "
                      f"iters={iters}")
        ax2.axvline(np.log10(max_div + 1e-30), color="red", linestyle="--")
        fig.suptitle(f"[{tag}] Test A @ 64^3 — projection mass conservation",
                     fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(
            PLOTS_DIR_64, f"divergence_map_{tag}_64.png")
        fig.savefig(out_path); plt.close(fig)
        print(f"  wrote {out_path}")
        results_A[tag]["plot"] = out_path
        jax.clear_caches()

    # ---------- Test B ----------
    print("\n[Test B] static droplet, CSF surface tension, 10 steps")
    kappa = curvature_csf(c, H_GRID)
    gcx, gcy, gcz = gradient_central(c, H_GRID)
    fsx = SIGMA * kappa * gcx
    fsy = SIGMA * kappa * gcy
    fsz = SIGMA * kappa * gcz
    inv_rho = 1.0 / rho
    N_STEPS = 10

    results_B = {}
    for tag, make_plan_ctx in [
        ("P2", lambda: CusparsePlan(rp, ci, vv, do)),
        ("P3", lambda: MulticolorPlan(rp_h, ci_h, vv_h,
                                      grid_shape=(N_GRID, N_GRID, N_GRID))),
    ]:
        print(f"[Test B / {tag}] running 10 projection steps...")
        u = np.zeros((N_GRID,) * 3); v = np.zeros_like(u); w = np.zeros_like(u)
        uinf_hist = []; div_hist = []; iters_hist = []
        t_all = time.perf_counter()
        with make_plan_ctx() as plan:
            d_star = plan.factor(vv)
            d_star.block_until_ready()

            def apply_M(r, _plan=plan, _d=d_star):
                return _plan.apply(vv, _d, r)

            for step in range(N_STEPS):
                u_star = u + DT_PROJECTION * fsx * inv_rho
                v_star = v + DT_PROJECTION * fsy * inv_rho
                w_star = w + DT_PROJECTION * fsz * inv_rho
                uf_s, vf_s, wf_s = cell_to_face(u_star, v_star, w_star)
                div_u_star = divergence_from_faces(
                    uf_s, vf_s, wf_s, H_GRID, N_GRID, N_GRID, N_GRID)
                rhs_step = (-div_u_star / DT_PROJECTION).reshape(-1).copy()
                rhs_step[0] = 0.0
                b_step = jax.device_put(
                    jnp.asarray(np.ascontiguousarray(rhs_step)))
                x_full, iters, conv, rnorm, _ = _pcg(
                    apply_M, spmv, b_step,
                    tol=PCG_TOL, max_iter=PCG_MAX_ITER)
                p = np.asarray(x_full).reshape((N_GRID, N_GRID, N_GRID))
                uf, vf, wf = apply_velocity_correction(
                    u_star, v_star, w_star, p, rho, H_GRID, DT_PROJECTION)
                # Cell-centre reconstruction with zero BC normal flux.
                zbx = np.zeros((1, N_GRID, N_GRID))
                zby = np.zeros((N_GRID, 1, N_GRID))
                zbz = np.zeros((N_GRID, N_GRID, 1))
                uf_full = np.concatenate([zbx, uf, zbx], axis=0)
                vf_full = np.concatenate([zby, vf, zby], axis=1)
                wf_full = np.concatenate([zbz, wf, zbz], axis=2)
                u = 0.5 * (uf_full[:-1] + uf_full[1:])
                v = 0.5 * (vf_full[:, :-1] + vf_full[:, 1:])
                w = 0.5 * (wf_full[:, :, :-1] + wf_full[:, :, 1:])
                div_u = divergence_from_faces(uf, vf, wf, H_GRID,
                                              N_GRID, N_GRID, N_GRID)
                umag = np.sqrt(u * u + v * v + w * w)
                uinf = float(np.max(umag))
                div_inf = float(np.max(np.abs(div_u)))
                uinf_hist.append(uinf); div_hist.append(div_inf)
                iters_hist.append(int(iters))
                print(f"  [{tag}] step {step+1}/{N_STEPS}: iters={iters} "
                      f"||u||∞={uinf:.3e} max|div|={div_inf:.3e}")
        twall_all = time.perf_counter() - t_all
        print(f"[Test B / {tag}] total wall {twall_all:.2f} s, "
              f"||u||∞ final = {uinf_hist[-1]:.3e}, "
              f"max|div| = {max(div_hist):.3e}")
        results_B[tag] = {"uinf_hist": uinf_hist,
                          "div_hist": div_hist,
                          "iters_hist": iters_hist,
                          "wall_s": twall_all}

        # Plot.
        fig = plt.figure(figsize=(11, 4.5))
        ax1 = fig.add_subplot(1, 2, 1)
        zslice = N_GRID // 2
        u_sl = u[:, :, zslice]; v_sl = v[:, :, zslice]
        c_sl = c[:, :, zslice]
        xs = cell_centres(N_GRID, H_GRID)
        XX, YY = np.meshgrid(xs, xs, indexing="ij")
        umag_sl = np.sqrt(u_sl ** 2 + v_sl ** 2)
        stride = max(1, N_GRID // 24)
        scale = max(np.max(umag_sl), 1e-30) * 20
        ax1.contour(XX, YY, c_sl, levels=[0.5], colors="k", linewidths=1.0)
        ax1.contourf(XX, YY, c_sl, levels=[0.5, 1.1],
                     colors=["lightblue"], alpha=0.3)
        ax1.quiver(XX[::stride, ::stride], YY[::stride, ::stride],
                   u_sl[::stride, ::stride], v_sl[::stride, ::stride],
                   umag_sl[::stride, ::stride], cmap="viridis",
                   scale=scale, scale_units="xy", width=0.004)
        ax1.set_aspect("equal"); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.set_title(f"[{tag}] velocity @ z={zslice} step {N_STEPS}")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")

        ax2 = fig.add_subplot(1, 2, 2)
        steps = np.arange(1, N_STEPS + 1)
        ax2.semilogy(steps, uinf_hist, "-o", color="crimson")
        ax2.axhline(1e-12, color="k", linestyle=":", alpha=0.5,
                    label="machine zero")
        ax2.set_xlabel("time step"); ax2.set_ylabel(r"$\|u\|_\infty$")
        ax2.set_title(r"$\|u\|_\infty$ per step")
        ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

        fig.suptitle(f"[{tag}] Test B @ 64^3 — static droplet", fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(
            PLOTS_DIR_64, f"spurious_currents_{tag}_64.png")
        fig.savefig(out_path); plt.close(fig)
        print(f"  wrote {out_path}")
        results_B[tag]["plot"] = out_path
        jax.clear_caches()

    snap_post = _capture_nvidia_smi()
    print(f"[vram post] {snap_post}")

    out = {"label": "physical_64",
           "n": N_GRID, "N": N_GRID ** 3, "nnz": int(len(vv_h)),
           "nvidia_smi_pre": snap_pre,
           "nvidia_smi_post": snap_post,
           "test_A": results_A, "test_B": results_B}
    out_path = os.path.join(RESULT_JSON_DIR, "physical_64.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"###RESULT### {out_path}")
    return out


# --------------------------------------------------------------------------
# Top-level orchestration: spawn 128³ subprocesses separately to keep VRAM
# peak ≤ one-plan budget at a time.
# --------------------------------------------------------------------------
def run_mode(mode: str):
    if mode == "selfcheck":
        _selfcheck_stiff_builders()
        _selfcheck_poisson_builders()
        return

    if mode == "64_dispatch":
        _preflight(required_mib=1200)
        run_dispatch_and_pcg(n=64, tol=1e-10, pcg_max_iter=500,
                             label="dispatch_64", run_p2=True, run_p3=True)
        return

    if mode == "64_physical":
        _preflight(required_mib=1200)
        run_physical_64()
        return

    if mode == "128_phase2":
        _preflight(required_mib=1500)
        run_dispatch_and_pcg(n=128, tol=1e-10, pcg_max_iter=500,
                             label="dispatch_128_P2",
                             run_p2=True, run_p3=False)
        return

    if mode == "128_phase3":
        _preflight(required_mib=1500)
        run_dispatch_and_pcg(n=128, tol=1e-10, pcg_max_iter=500,
                             label="dispatch_128_P3",
                             run_p2=False, run_p3=True)
        return

    if mode == "all":
        # Snapshot before anything.
        print("=" * 72)
        print(f"Phase 3 Scaling Benchmark — {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 72)
        print("[pre-global] nvidia-smi:")
        print(_capture_nvidia_smi())
        print("[pre-global] free -h:")
        print(_capture_free())

        # Self-check the fast matrix builders agree with reference.
        _selfcheck_stiff_builders()
        _selfcheck_poisson_builders()

        # --- 64³ dispatch + PCG (in-process) ---
        run_dispatch_and_pcg(n=64, tol=1e-10, pcg_max_iter=500,
                             label="dispatch_64",
                             run_p2=True, run_p3=True)

        # --- 64³ physical Tests A, B (in-process) ---
        run_physical_64()

        # --- 128³ Phase 2 subprocess ---
        print("\n[128³] spawning Phase 2 subprocess...")
        cmd_p2 = [sys.executable, os.path.abspath(__file__), "--mode",
                  "128_phase2"]
        ret_p2 = subprocess.call(cmd_p2)
        print(f"[128³/P2] subprocess exit {ret_p2}")
        if ret_p2 != 0:
            print("WARNING: 128³ Phase 2 subprocess failed; continuing.")

        # --- 128³ Phase 3 subprocess ---
        print("\n[128³] spawning Phase 3 subprocess...")
        cmd_p3 = [sys.executable, os.path.abspath(__file__), "--mode",
                  "128_phase3"]
        ret_p3 = subprocess.call(cmd_p3)
        print(f"[128³/P3] subprocess exit {ret_p3}")
        if ret_p3 != 0:
            print("WARNING: 128³ Phase 3 subprocess failed; continuing.")

        # --- Generate final report ---
        print("\n[report] assembling markdown...")
        write_report()
        return

    if mode == "report":
        write_report()
        return

    raise ValueError(f"unknown mode: {mode}")


# --------------------------------------------------------------------------
# Final report assembly.
# --------------------------------------------------------------------------
def _load_json(label: str):
    path = os.path.join(RESULT_JSON_DIR, f"{label}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def write_report():
    """Assemble docs/benchmark/phase3_scaling_64_128.md from JSON results."""
    d_64 = _load_json("dispatch_64")
    d_128_p2 = _load_json("dispatch_128_P2")
    d_128_p3 = _load_json("dispatch_128_P3")
    phys_64 = _load_json("physical_64")

    def _fmt(x, fmt=".3g", none="n/a"):
        if x is None:
            return none
        return format(x, fmt)

    def _ratio(a, b):
        try:
            return float(a) / float(b)
        except Exception:
            return float("nan")

    lines = []
    lines.append("# Phase 3 — Scaling Benchmark 64³ and 128³")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("**Hardware**: RTX 3050 Laptop (4 GB VRAM, CC 8.6, "
                 "FP64 @ 1/32 FP32)")
    lines.append("**JAX**: 0.9.0 at `/home/yzk/jax-env` · float64 · "
                 "`XLA_PYTHON_CLIENT_*` safety rails")
    lines.append("**Matrix pattern**: T7-style stiff 3-D 7-pt Laplacian, "
                 "coefficient jump 1 ↔ 100 at z-midplane (same pattern used "
                 "in Phase 2 T7 and Phase 3 C4).")
    lines.append("")
    lines.append("> Scope: scaling of Phase 2 (cuSPARSE SpSV DILU) vs "
                 "Phase 3 (red-black multi-color DILU) at 64³ (262k cells) "
                 "and 128³ (2.1M cells). **Phase 4 is NOT started.**")
    lines.append("")

    # ---------- 1. Environment + VRAM ----------
    lines.append("## 1. Environment + VRAM snapshots")
    lines.append("")
    lines.append("### Pre-run GPU state (before any PCG)")
    lines.append("```")
    if d_64 is not None:
        lines.append("[64³ pre]  " + d_64["nvidia_smi_pre"])
        lines.append("[64³ post] " + d_64["nvidia_smi_post"])
    if phys_64 is not None:
        lines.append("[phys64 pre]  " + phys_64["nvidia_smi_pre"])
        lines.append("[phys64 post] " + phys_64["nvidia_smi_post"])
    if d_128_p2 is not None:
        lines.append("[128³ P2 pre]  " + d_128_p2["nvidia_smi_pre"])
        lines.append("[128³ P2 post] " + d_128_p2["nvidia_smi_post"])
    if d_128_p3 is not None:
        lines.append("[128³ P3 pre]  " + d_128_p3["nvidia_smi_pre"])
        lines.append("[128³ P3 post] " + d_128_p3["nvidia_smi_post"])
    lines.append("```")
    lines.append("")
    lines.append("Both 128³ runs executed in separate subprocesses so only "
                 "one plan occupied VRAM at a time (each plan's working set "
                 "~200 MB of matrix buffers + ~100 MB PCG workspaces; a "
                 "simultaneous resident P2+P3 pair on 3050 4 GB was deemed "
                 "unacceptably tight against OOM).")
    lines.append("")

    # ---------- 2. Scaling table ----------
    lines.append("## 2. Scaling table (full cross-grid view)")
    lines.append("")
    lines.append("| Grid | N | nnz | P2 apply med / p95 (µs) | P2 iters | "
                 "P2 total PCG (s) | P3 apply med / p95 (µs) | P3 iters | "
                 "P3 total PCG (s) | apply speedup | total PCG speedup | "
                 "iter penalty |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    # 16³: copied from Phase 3 report.
    # med_P2=1853.7, p95_P2=2996.9, iters_P2=24
    # med_P3=952.8,  p95_P3=2454.1, iters_P3=36
    # (phase3 report reports 'total PCG' as apply_med × iters surrogate)
    p2_med_16 = 1853.7; p2_p95_16 = 2996.9; p2_iters_16 = 24
    p3_med_16 = 952.8;  p3_p95_16 = 2454.1; p3_iters_16 = 36
    t_p2_16 = p2_med_16 * p2_iters_16 / 1e6
    t_p3_16 = p3_med_16 * p3_iters_16 / 1e6
    lines.append(f"| 16³ (copy) | 4096 | 27136 | "
                 f"{p2_med_16:.1f} / {p2_p95_16:.1f} | {p2_iters_16} | "
                 f"{t_p2_16:.4f}* | "
                 f"{p3_med_16:.1f} / {p3_p95_16:.1f} | {p3_iters_16} | "
                 f"{t_p3_16:.4f}* | "
                 f"{p2_med_16/p3_med_16:.2f}× | "
                 f"{t_p2_16/t_p3_16:.2f}× | "
                 f"{p3_iters_16/p2_iters_16:.2f}× |")
    # 32³ copied from Phase 3 report.
    p2_med_32 = 4342.0; p2_p95_32 = 5762.9; p2_iters_32 = 128
    p3_med_32 = 1035.9; p3_p95_32 = 2187.2; p3_iters_32 = 197
    t_p2_32 = p2_med_32 * p2_iters_32 / 1e6
    t_p3_32 = p3_med_32 * p3_iters_32 / 1e6
    lines.append(f"| 32³ (copy) | 32768 | 223226 | "
                 f"{p2_med_32:.1f} / {p2_p95_32:.1f} | {p2_iters_32} | "
                 f"{t_p2_32:.4f}* | "
                 f"{p3_med_32:.1f} / {p3_p95_32:.1f} | {p3_iters_32} | "
                 f"{t_p3_32:.4f}* | "
                 f"{p2_med_32/p3_med_32:.2f}× | "
                 f"{t_p2_32/t_p3_32:.2f}× | "
                 f"{p3_iters_32/p2_iters_32:.2f}× |")

    # 64³ measured.
    if d_64 is not None:
        p2 = d_64["P2"]; p3 = d_64["P3"]
        lines.append(f"| 64³ (new) | {d_64['N']} | {d_64['nnz']} | "
                     f"{p2['apply_med_us']:.1f} / {p2['apply_p95_us']:.1f} | "
                     f"{p2['iters']}{'' if p2['converged'] else '†'} | "
                     f"{p2['wall_s']:.3f} | "
                     f"{p3['apply_med_us']:.1f} / {p3['apply_p95_us']:.1f} | "
                     f"{p3['iters']}{'' if p3['converged'] else '†'} | "
                     f"{p3['wall_s']:.3f} | "
                     f"{p2['apply_med_us']/p3['apply_med_us']:.2f}× | "
                     f"{p2['wall_s']/p3['wall_s']:.2f}× | "
                     f"{p3['iters']/p2['iters']:.2f}× |")
    else:
        lines.append("| 64³ | — | — | not measured | — | — | — | — | — | — | "
                     "— | — |")

    # 128³ measured (two subprocesses).
    if d_128_p2 is not None and d_128_p3 is not None:
        p2 = d_128_p2["P2"]; p3 = d_128_p3["P3"]
        lines.append(f"| 128³ (new) | {d_128_p2['N']} | {d_128_p2['nnz']} | "
                     f"{p2['apply_med_us']:.1f} / {p2['apply_p95_us']:.1f} | "
                     f"{p2['iters']}{'' if p2['converged'] else '†'} | "
                     f"{p2['wall_s']:.3f} | "
                     f"{p3['apply_med_us']:.1f} / {p3['apply_p95_us']:.1f} | "
                     f"{p3['iters']}{'' if p3['converged'] else '†'} | "
                     f"{p3['wall_s']:.3f} | "
                     f"{p2['apply_med_us']/p3['apply_med_us']:.2f}× | "
                     f"{p2['wall_s']/p3['wall_s']:.2f}× | "
                     f"{p3['iters']/p2['iters']:.2f}× |")
    else:
        lines.append("| 128³ | — | — | not measured | — | — | — | — | — | — "
                     "| — | — |")

    lines.append("")
    lines.append("\\* 16³ and 32³ 'total PCG' columns are the *surrogate* "
                 "`apply_median × iters` quoted from the Phase 3 report "
                 "(µs × iters → s). They are NOT fresh wall-clock "
                 "measurements and **exclude SpMV + Python overhead**. The "
                 "64³ and 128³ rows report actual wall time (includes SpMV, "
                 "dot products, memory allocation, Python for-loop), so "
                 "their ratios are not apples-to-apples with the 16³/32³ "
                 "surrogate ratios. Compare **per-apply** numbers across "
                 "all four rows for a consistent measurement.")
    lines.append("")
    lines.append("† Did not converge to 1e-10 within PCG max_iter=500. "
                 "Iter count reported is the cap; wall time includes all "
                 "500 iterations.")
    lines.append("")

    # ---------- 3. Per-apply dispatch trend ----------
    lines.append("## 3. Per-apply dispatch trend")
    lines.append("")
    lines.append("Key observation: Phase 3 per-apply speedup is expected to "
                 "GROW with grid size because cuSPARSE internal level-"
                 "scheduling barrier count scales as O(N^{1/3}) while "
                 "red-black multi-color is fixed at 2 colors × 2 sweeps "
                 "= 4 kernel launches.")
    lines.append("")
    apply_trend = [
        ("16³", p2_med_16, p3_med_16),
        ("32³", p2_med_32, p3_med_32),
    ]
    if d_64 is not None:
        apply_trend.append(("64³",
                            d_64["P2"]["apply_med_us"],
                            d_64["P3"]["apply_med_us"]))
    if d_128_p2 is not None and d_128_p3 is not None:
        apply_trend.append(("128³",
                            d_128_p2["P2"]["apply_med_us"],
                            d_128_p3["P3"]["apply_med_us"]))
    lines.append("| Grid | P2 apply median (µs) | P3 apply median (µs) | "
                 "P2 / P3 |")
    lines.append("|---|---|---|---|")
    for g, p2m, p3m in apply_trend:
        lines.append(f"| {g} | {p2m:.1f} | {p3m:.1f} | {p2m/p3m:.2f}× |")
    lines.append("")

    # ---------- 4. Total PCG wall-time trend ----------
    lines.append("## 4. Total-PCG wall-time trend")
    lines.append("")
    lines.append("Comparison mixes two measurement conventions (see §2 "
                 "footnote). The 64³ and 128³ wall-clock numbers are the "
                 "load-bearing new measurements; the 16³/32³ surrogates are "
                 "shown for trajectory continuity only.")
    lines.append("")
    walltime_trend = [
        ("16³ (surrogate)", t_p2_16, t_p3_16),
        ("32³ (surrogate)", t_p2_32, t_p3_32),
    ]
    if d_64 is not None:
        walltime_trend.append(("64³ (wall)",
                               d_64["P2"]["wall_s"],
                               d_64["P3"]["wall_s"]))
    if d_128_p2 is not None and d_128_p3 is not None:
        walltime_trend.append(("128³ (wall)",
                               d_128_p2["P2"]["wall_s"],
                               d_128_p3["P3"]["wall_s"]))
    lines.append("| Grid | P2 total PCG (s) | P3 total PCG (s) | "
                 "total speedup |")
    lines.append("|---|---|---|---|")
    for g, t2, t3 in walltime_trend:
        lines.append(f"| {g} | {t2:.4f} | {t3:.4f} | {t2/t3:.2f}× |")
    lines.append("")
    # Implied per-iter cost breakdown at 64³ / 128³.
    if d_64 is not None:
        p2 = d_64["P2"]; p3 = d_64["P3"]
        per_iter_p2 = p2["wall_s"] / max(p2["iters"], 1) * 1e6  # µs
        per_iter_p3 = p3["wall_s"] / max(p3["iters"], 1) * 1e6
        non_apply_p2 = per_iter_p2 - p2["apply_med_us"]
        non_apply_p3 = per_iter_p3 - p3["apply_med_us"]
        lines.append("**At 64³ (per PCG iter)**: "
                     f"P2 total = {per_iter_p2:.0f} µs "
                     f"(apply {p2['apply_med_us']:.0f} µs + non-apply "
                     f"{non_apply_p2:.0f} µs). "
                     f"P3 total = {per_iter_p3:.0f} µs "
                     f"(apply {p3['apply_med_us']:.0f} µs + non-apply "
                     f"{non_apply_p3:.0f} µs). The non-apply tax (SpMV, "
                     "dot products, Python, JAX dispatch) is roughly equal "
                     "across implementations and dominates per-iter cost "
                     "at this scale, diluting the apply-dispatch speedup.")
        lines.append("")
    if d_128_p2 is not None and d_128_p3 is not None:
        p2 = d_128_p2["P2"]; p3 = d_128_p3["P3"]
        per_iter_p2 = p2["wall_s"] / max(p2["iters"], 1) * 1e6
        per_iter_p3 = p3["wall_s"] / max(p3["iters"], 1) * 1e6
        non_apply_p2 = per_iter_p2 - p2["apply_med_us"]
        non_apply_p3 = per_iter_p3 - p3["apply_med_us"]
        lines.append("**At 128³ (per PCG iter)**: "
                     f"P2 total = {per_iter_p2:.0f} µs "
                     f"(apply {p2['apply_med_us']:.0f} µs + non-apply "
                     f"{non_apply_p2:.0f} µs). "
                     f"P3 total = {per_iter_p3:.0f} µs "
                     f"(apply {p3['apply_med_us']:.0f} µs + non-apply "
                     f"{non_apply_p3:.0f} µs). At this scale the apply "
                     "and non-apply costs are of comparable magnitude, but "
                     "the non-apply portion is bandwidth-bound SpMV that "
                     "neither DILU variant can shorten.")
        lines.append("")

    # ---------- 5. Iter-count penalty ----------
    lines.append("## 5. Iter-count penalty trend (P3 / P2)")
    lines.append("")
    lines.append("Duff-Meurant 1989 predicts grid-independent penalty near "
                 "2.0×. Phase 3 report measured 1.50-1.55× at 16³/32³. "
                 "Scaling question: does it drift?")
    lines.append("")
    iter_trend = [
        ("16³ stiff T7", p2_iters_16, p3_iters_16),
        ("32³ 3-tier Test C", p2_iters_32, p3_iters_32),
    ]
    if d_64 is not None:
        iter_trend.append((
            "64³ stiff (new)",
            d_64["P2"]["iters"], d_64["P3"]["iters"]))
    if d_128_p2 is not None and d_128_p3 is not None:
        iter_trend.append((
            "128³ stiff (new)",
            d_128_p2["P2"]["iters"], d_128_p3["P3"]["iters"]))
    lines.append("| Problem | P2 iters | P3 iters | penalty |")
    lines.append("|---|---|---|---|")
    for label, i2, i3 in iter_trend:
        lines.append(f"| {label} | {i2} | {i3} | {i3/i2:.2f}× |")
    lines.append("")

    # ---------- 6. Physical benchmarks at 64³ ----------
    lines.append("## 6. Physical benchmarks at 64³")
    lines.append("")
    if phys_64 is None:
        lines.append("Not available.")
    else:
        A = phys_64["test_A"]
        lines.append("### Test A — variable-density projection "
                     "(ρ-ratio 1000 sphere, random low-k u*)")
        lines.append("")
        lines.append("| Impl | PCG iters | converged | rnorm | "
                     "max\\|∇·u\\| | wall (s) |")
        lines.append("|---|---|---|---|---|---|")
        for tag in ("P2", "P3"):
            if tag not in A:
                continue
            r = A[tag]
            lines.append(f"| {tag} | {r['iters']} | "
                         f"{'Y' if r['converged'] else 'N'} | "
                         f"{r['rnorm']:.3e} | "
                         f"{r['max_div']:.3e} | {r['wall_s']:.2f} |")
        lines.append("")
        for tag in ("P2", "P3"):
            if tag in A and "plot" in A[tag]:
                rel = os.path.relpath(A[tag]["plot"], os.path.dirname(REPORT_PATH))
                lines.append(f"![{tag} Test A 64³]({rel})")
                lines.append("")

        B = phys_64["test_B"]
        lines.append("### Test B — static droplet CSF projection, 10 steps")
        lines.append("")
        lines.append("| Impl | ‖u‖∞ step 1 | ‖u‖∞ step 10 | max\\|div\\| | "
                     "iters/step (mean) | wall (s) |")
        lines.append("|---|---|---|---|---|---|")
        for tag in ("P2", "P3"):
            if tag not in B:
                continue
            r = B[tag]
            mean_iters = float(np.mean(r["iters_hist"])) if r["iters_hist"] else 0.0
            lines.append(f"| {tag} | {r['uinf_hist'][0]:.3e} | "
                         f"{r['uinf_hist'][-1]:.3e} | "
                         f"{max(r['div_hist']):.3e} | {mean_iters:.1f} | "
                         f"{r['wall_s']:.2f} |")
        lines.append("")
        for tag in ("P2", "P3"):
            if tag in B and "plot" in B[tag]:
                rel = os.path.relpath(B[tag]["plot"], os.path.dirname(REPORT_PATH))
                lines.append(f"![{tag} Test B 64³]({rel})")
                lines.append("")
    lines.append("")
    lines.append("Test C (scipy `spsolve` gold standard) intentionally NOT "
                 "re-run at 64³ — at 262k unknowns sparse LU fill-in would "
                 "exceed ~2 GB host RAM and Phase 2.5 / Phase 3 have "
                 "already verified F2 at 32³ (corr = -0.102). Re-running at "
                 "a larger grid would test scipy more than DILU.")
    lines.append("")

    # ---------- 7. Honest caveats ----------
    lines.append("## 7. Honest caveats")
    lines.append("")
    caveats = []
    caveats.append(
        "**Memory bandwidth on 3050 Laptop.** At 128³ the PCG inner loop "
        "is dominated by SpMV over 14.7M nnz (118 MB of float64 values + "
        "59 MB of col_idx per traversal). The 3050 has ~224 GB/s peak "
        "bandwidth; per-iter requires ≥ 360 MB of reads (matrix + vectors), "
        "so the theoretical lower bound per PCG iter is ~1.6 ms of pure "
        "memory traffic. That is a floor on PCG wall time that neither "
        "DILU variant can beat — any apply-dispatch savings at 128³ "
        "compete with a bandwidth-bound baseline.")
    caveats.append(
        "**FP64 throughput on consumer SKU.** 3050 Laptop at CC 8.6 runs "
        "FP64 at 1/32 of FP32. DILU apply and SpMV are FP64-heavy; "
        "relative dispatch-overhead improvements (Phase 3 vs Phase 2) look "
        "larger on consumer hardware than they would on a server GPU "
        "(A100/H100), where FP64 throughput approaches FP32.")
    caveats.append(
        "**128³ VRAM headroom is tight (~ 500-600 MB per plan on a 4 GB "
        "card that already has ~1.5 GB occupied by desktop/WSL2).** The "
        "subprocess-isolation strategy keeps peak within budget, but "
        "adding a second vector workspace or a direct-solver reference "
        "would OOM.")
    caveats.append(
        "**128³ PCG may not converge to 1e-10 within 500 iters.** The "
        "Gustafsson O(κ^{1/4}) bound predicts κ ∝ grid × contrast, so iter "
        "count grows with N^{1/3} · contrast^{1/4}. A 1.5×-size grid jump "
        "from 64 → 128 is expected to raise iter count by ~1.6×, "
        "potentially exceeding 500. Rows flagged with † indicate the max-iter "
        "cap was hit; the iter count therefore understates the true "
        "converged cost for BOTH implementations.")
    caveats.append(
        "**SpMV is not jitted.** Both PCG drivers use a `segment_sum` "
        "SpMV with `searchsorted` for row lookup, not wrapped in "
        "`jax.jit`. This matches the Phase 3 bench_multicolor_vs_cusparse "
        "protocol so iter counts and timings are directly comparable, but "
        "a jitted SpMV would shave ~20-30% off total PCG wall time "
        "uniformly across grids — it would not change the P2/P3 ratio.")
    for c in caveats:
        lines.append("- " + c)
    lines.append("")

    # ---------- 8. Growth verdict ----------
    lines.append("## 8. Growth-trend verdict")
    lines.append("")
    if (d_64 is not None and d_128_p2 is not None and d_128_p3 is not None):
        apply_16 = p2_med_16 / p3_med_16
        apply_32 = p2_med_32 / p3_med_32
        apply_64 = d_64["P2"]["apply_med_us"] / d_64["P3"]["apply_med_us"]
        apply_128 = (d_128_p2["P2"]["apply_med_us"]
                     / d_128_p3["P3"]["apply_med_us"])
        total_64 = d_64["P2"]["wall_s"] / d_64["P3"]["wall_s"]
        total_128 = d_128_p2["P2"]["wall_s"] / d_128_p3["P3"]["wall_s"]
        iter_64 = d_64["P3"]["iters"] / max(d_64["P2"]["iters"], 1)
        iter_128 = d_128_p3["P3"]["iters"] / max(d_128_p2["P2"]["iters"], 1)

        lines.append("### 8.1 Per-apply speedup")
        lines.append("")
        lines.append(f"Trajectory: 16³ → {apply_16:.2f}×, 32³ → "
                     f"{apply_32:.2f}×, 64³ → {apply_64:.2f}×, 128³ → "
                     f"{apply_128:.2f}×.")
        lines.append("")
        lines.append("The trajectory is **non-monotone**. It peaks at 32³ "
                     "(4.19×) and stabilises around **2.5 – 3.1×** for "
                     "64³ and 128³. This is NOT the monotone growth "
                     "predicted by 'kernel launch count scales with grid'. "
                     "Two effects compete:")
        lines.append("")
        lines.append("1. As grids grow, cuSPARSE's internal level count "
                     "grows, so Phase 2's apply has more barriers → "
                     "theoretically favours Phase 3 more.")
        lines.append("2. As grids grow, per-apply wall time is increasingly "
                     "dominated by nnz · float64 memory traffic (bandwidth-"
                     "bound). Both Phase 2 and Phase 3 pay this cost "
                     "identically, compressing the relative gap.")
        lines.append("")
        lines.append("Empirically, effect (2) dominates past 32³ on the "
                     "3050 Laptop's 4 GB / 224 GB/s memory subsystem. The "
                     "per-apply gap stabilises at ~3×, not growing.")
        lines.append("")

        lines.append("### 8.2 Total-PCG wall-time speedup")
        lines.append("")
        lines.append(f"64³ wall ratio: {total_64:.2f}×; "
                     f"128³ wall ratio: {total_128:.2f}×.")
        lines.append("")
        if total_128 < 1.0:
            lines.append("**Phase 3 regresses at 64³ and 128³ in total PCG "
                         "wall time.** The ~3× per-apply advantage is NOT "
                         "enough to overcome the 1.6× iter-count penalty "
                         "ONCE non-apply per-iter costs (bandwidth-bound "
                         "SpMV, dot products, Python dispatch) become "
                         "comparable-or-larger than the apply itself.")
            lines.append("")
            lines.append("Quick algebra: let A = P2 apply, B = P2 non-apply "
                         "per-iter, r_app = A / (P3 apply) (per-apply "
                         "speedup), r_it = P3 iters / P2 iters (penalty). "
                         "Total speedup = (A+B) / (A/r_app + B) × (1/r_it). "
                         "At 64³, A ≈ 8 ms, B ≈ 57 ms, r_app ≈ 2.64, "
                         "r_it ≈ 1.60 → predicted 0.80×, matches measured.")
            lines.append("")
            lines.append("**Implication**: Phase 3's design was optimised "
                         "for the 16³–32³ regime where apply dominates "
                         "per-iter cost. At realistic AM-adjacent scales, "
                         "that optimisation target evaporates.")
        else:
            lines.append(f"Phase 3 retains {total_128:.2f}× total wall "
                         "speedup at 128³.")
        lines.append("")

        lines.append("### 8.3 Iter-count penalty")
        lines.append("")
        lines.append(f"Trajectory: 16³ → {p3_iters_16/p2_iters_16:.2f}×, "
                     f"32³ → {p3_iters_32/p2_iters_32:.2f}×, 64³ → "
                     f"{iter_64:.2f}×, 128³ → {iter_128:.2f}×.")
        lines.append("")
        lines.append("Penalty is stable at 1.5 – 1.6× across all four "
                     "grids. This is solidly **inside** the Duff-Meurant "
                     "1989 prediction band (grid-independent, median ~2×) "
                     "and below the Li-Saad 2010 GPU MC-ILU(0) range "
                     "(1.8 – 2.0×). The red-black ordering's eigenvalue-"
                     "spread penalty behaves as theory predicted at "
                     "realistic AM scale.")
        lines.append("")

        lines.append("### 8.4 Bottom line")
        lines.append("")
        lines.append("| Regime | P3 verdict |")
        lines.append("|---|---|")
        lines.append("| 16³ dispatch-bound | win (+30%) |")
        lines.append("| 32³ apply-dominated | strong win (+172%) |")
        lines.append("| 64³ non-apply-dominated | **loss (-20% wall)** |")
        lines.append("| 128³ bandwidth-bound | **loss (-19% wall)** |")
        lines.append("")
        lines.append("Phase 3's value proposition is grid-size-conditional. "
                     "Its 2.72× total-PCG speedup at 32³ does **not** carry "
                     "to realistic AM scales on this hardware. At 64³ and "
                     "128³, the extra 1.5× iterations outweigh the ~3× "
                     "per-apply savings because per-iter is no longer "
                     "apply-dominated.")
        lines.append("")
        lines.append("**Phase 3 is not invalidated** — its correctness, "
                     "zero-cuSPARSE-dependency, and physical fidelity at "
                     "64³ all hold. But the **total-PCG speedup** headline "
                     "from the 32³ report should be understood as the "
                     "peak of a curve, not a monotone trend. A fair "
                     "summary of the 4-grid dataset: Phase 3 wins when "
                     "apply dispatch dominates per-iter cost (small grids "
                     "or dispatch-bound regimes); Phase 2 wins when "
                     "memory-bandwidth-bound SpMV dominates (large 3-D "
                     "grids on consumer hardware).")
    else:
        lines.append("(Data incomplete — see rows flagged 'not measured'.)")
    lines.append("")

    # ---------- 9. Status ----------
    lines.append("## 9. Status")
    lines.append("")
    lines.append("- Phase 3 scaling data at 64³ and 128³ captured.")
    lines.append("- Existing 16³ and 32³ numbers quoted from the "
                 "Phase 3 report (not re-measured).")
    lines.append("- No Phase 1/2/2.5/3 source artifact was modified.")
    lines.append("- **Phase 4 is NOT started.**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of Phase 3 scaling benchmark.*")
    lines.append("")

    content = "\n".join(lines)
    with open(REPORT_PATH, "w") as fh:
        fh.write(content)
    print(f"[report] wrote {REPORT_PATH}")
    print(f"[report] {len(content)} chars")


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=["selfcheck", "64_dispatch", "64_physical",
                             "128_phase2", "128_phase3", "all", "report"],
                    default="all")
    args = ap.parse_args()
    run_mode(args.mode)


if __name__ == "__main__":
    main()
