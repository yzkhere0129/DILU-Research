"""Step 1: PLIC Interface Reconstruction — JAX-Native (v3, bisection C).

Computes piecewise linear interface normals (nx, ny) and intercepts C
for all cells using pure tensor operations (no Python loops).

Key change: intercept C computed via bisection (15 iterations) instead
of analytical area-inverse, which had persistent quadrant bugs.
"""

import jax
import jax.numpy as jnp
from ...data_types import Array


def compute_plic_normals(F: Array, dx: float, dy: float) -> tuple:
    """Compute PLIC interface normal via 3×3 Least-Squares gradient.

    Barkhudarov (2004) §4: "Here we adopted the Least Squares method
    using a 27-point stencil (9-point in 2D)."

    On a uniform grid, the LS gradient of F(x,y) = F₀ + a·Δx + b·Δy
    over the 3×3 neighborhood has an analytic closed-form solution.
    The normal equations (AᵀA)⁻¹Aᵀ reduce to a diagonal system because
    the stencil is symmetric, giving constant convolution kernels:

        K_x = 1/(6·dx) · [[-1,-1,-1],    K_y = 1/(6·dy) · [[-1, 0,+1],
                           [ 0, 0, 0],                       [-1, 0,+1],
                           [+1,+1,+1]]                       [-1, 0,+1]]

    This is the Prewitt operator — the arithmetic mean of 3 row-wise
    (or column-wise) central differences. Compared to simple centered
    differences:
      - 3× lower noise variance (6 data points vs 2)
      - Better isotropy (uses diagonal neighbors)
      - Same O(h²) truncation error on smooth fields

    Args:
        F: volume fraction, shape (Nx+2nh, Ny+2nh, 1)
        dx, dy: cell sizes

    Returns:
        nx, ny: normal components, same shape as F.
                Normal points from fluid (F=1) toward empty (F=0).
    """
    # ── LS gradient via 2D convolution (Prewitt kernels) ──
    Kx = jnp.array([[-1., -1., -1.],
                     [ 0.,  0.,  0.],
                     [+1., +1., +1.]]) / (6.0 * dx)

    Ky = jnp.array([[-1., 0., +1.],
                     [-1., 0., +1.],
                     [-1., 0., +1.]]) / (6.0 * dy)

    # Stack into (out_channels=2, in_channels=1, kH=3, kW=3)
    kernel = jnp.stack([Kx, Ky])[:, None, :, :]

    # Reshape F to (batch=1, channels=1, H, W) for conv
    F_2d = F[:, :, 0]
    lhs = F_2d[None, None, :, :]

    # VALID cross-correlation: output is (1, 2, H-2, W-2)
    grad = jax.lax.conv_general_dilated(
        lhs, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))

    # Pad back to full field size (zero gradient at boundaries)
    dFdx = jnp.zeros_like(F).at[1:-1, 1:-1, 0].set(grad[0, 0])
    dFdy = jnp.zeros_like(F).at[1:-1, 1:-1, 0].set(grad[0, 1])

    # Normalize to unit vector; normal points from fluid toward empty
    mag = jnp.sqrt(dFdx**2 + dFdy**2 + 1e-30)
    nx = -dFdx / mag
    ny = -dFdy / mag

    is_interface = (F > 1e-6) & (F < 1.0 - 1e-6)
    nx = jnp.where(is_interface, nx, 0.0)
    ny = jnp.where(is_interface, ny, 0.0)

    return nx, ny


def _area_below_line(C: Array, a: Array, b: Array) -> Array:
    """Compute area of {a*xi + b*eta < C} in unit square [0,1]^2.

    For ANY signs of a, b. Uses corner classification + polygon clipping.
    Pure JAX, no branches.

    Args:
        C: intercept, scalar or broadcastable
        a, b: line coefficients (signed), same shape

    Returns:
        area: fraction of unit square below the line, same shape
    """
    # Four corners of [0,1]^2
    #   (0,0): val = 0
    #   (1,0): val = a
    #   (0,1): val = b
    #   (1,1): val = a+b
    v00 = jnp.zeros_like(C)
    v10 = a
    v01 = b
    v11 = a + b

    # Below = (val < C)
    b00 = v00 < C
    b10 = v10 < C
    b01 = v01 < C
    b11 = v11 < C

    # Count below
    n_below = (
        b00.astype(jnp.int32)
        + b10.astype(jnp.int32)
        + b01.astype(jnp.int32)
        + b11.astype(jnp.int32)
    )

    # Intersection parameters on each edge (0 = start vertex, 1 = end vertex)
    # Edge 0-1 (bottom): from (0,0) to (1,0), param t, line at t = C/a
    eps = 1e-30
    t_bottom = jnp.where(jnp.abs(a) > eps, C / a, -1.0)
    # Edge 0-2 (left): from (0,0) to (0,1), line at t = C/b
    t_left = jnp.where(jnp.abs(b) > eps, C / b, -1.0)
    # Edge 1-3 (right): from (1,0) to (1,1), line at t = (C-a)/b
    t_right = jnp.where(jnp.abs(b) > eps, (C - a) / b, -1.0)
    # Edge 2-3 (top): from (0,1) to (1,1), line at t = (C-b)/a
    t_top = jnp.where(jnp.abs(a) > eps, (C - b) / a, -1.0)

    # Valid intersections: 0 < t < 1
    in_bottom = (t_bottom > 0) & (t_bottom < 1)
    in_left = (t_left > 0) & (t_left < 1)
    in_right = (t_right > 0) & (t_right < 1)
    in_top = (t_top > 0) & (t_top < 1)

    # --- Area by n_below case ---

    # n_below = 0: entire square above line, area = 0
    area_0 = jnp.zeros_like(C)

    # n_below = 4: entire square below line, area = 1
    area_4 = jnp.ones_like(C)

    # n_below = 1: one corner below → triangle with 2 intersection points
    # The corner below has value v, the line cuts two edges from that corner.
    # Triangle area = 0.5 * t1 * t2 where t1, t2 are intersection params on the two edges.
    # Corner (0,0): edges bottom (t_bottom) and left (t_left) → area = 0.5 * t_b * t_l
    area_1_00 = 0.5 * jnp.clip(t_bottom, 0, 1) * jnp.clip(t_left, 0, 1)
    # Corner (1,0): edges bottom (from 1: t=1-t_bottom) and right (t_right) → area = 0.5*(1-t_b)*t_r
    area_1_10 = 0.5 * (1 - jnp.clip(t_bottom, 0, 1)) * jnp.clip(t_right, 0, 1)
    # Corner (0,1): edges left (from 1: t=1-t_left) and top (t_top) → area = 0.5*(1-t_l)*t_t
    area_1_01 = 0.5 * (1 - jnp.clip(t_left, 0, 1)) * jnp.clip(t_top, 0, 1)
    # Corner (1,1): edges right (from 1: t=1-t_right) and top (from 1: t=1-t_top)
    area_1_11 = 0.5 * (1 - jnp.clip(t_right, 0, 1)) * (1 - jnp.clip(t_top, 0, 1))

    area_1 = jnp.where(
        b00, area_1_00, jnp.where(b10, area_1_10, jnp.where(b01, area_1_01, area_1_11))
    )

    # n_below = 3: three corners below → 1 - triangle of corner above
    area_3_00 = 1.0 - area_1_00  # corner (0,0) above
    area_3_10 = 1.0 - area_1_10
    area_3_01 = 1.0 - area_1_01
    area_3_11 = 1.0 - area_1_11

    area_3 = jnp.where(
        ~b00,
        area_3_00,
        jnp.where(~b10, area_3_10, jnp.where(~b01, area_3_01, area_3_11)),
    )

    # n_below = 2: two corners below → trapezoid
    # Adjacent pairs: (00,10), (00,01), (10,11), (01,11)
    # Diagonal pairs: (00,11), (10,01)

    # Adjacent (00,10): bottom edge fully below, trapezoid with left and right intersections
    # Area = 0.5 * (t_left + t_right) * 1  (width along xi, average height)
    # Wait, this needs careful geometry. Let me use a different approach.

    # For n_below=2 adjacent corners, the area is a trapezoid.
    # For n_below=2 diagonal corners, the area is 0.5 (the line cuts through the middle).

    # Detect diagonal: corners 00 and 11 below, OR 10 and 01 below
    is_diag_00_11 = b00 & b11 & ~b10 & ~b01
    is_diag_10_01 = b10 & b01 & ~b00 & ~b11
    is_diag = is_diag_00_11 | is_diag_10_01

    # Adjacent pairs (00,10): bottom edge
    is_adj_00_10 = b00 & b10 & ~b01 & ~b11
    # Adjacent pairs (00,01): left edge
    is_adj_00_01 = b00 & b01 & ~b10 & ~b11
    # Adjacent pairs (10,11): right edge
    is_adj_10_11 = b10 & b11 & ~b00 & ~b01
    # Adjacent pairs (01,11): top edge
    is_adj_01_11 = b01 & b11 & ~b00 & ~b10

    # Trapezoid areas
    # (00,10): area = 0.5*(t_left + t_right)
    trap_00_10 = 0.5 * (jnp.clip(t_left, 0, 1) + jnp.clip(t_right, 0, 1))
    # (00,01): area = 0.5*(t_bottom + t_top)
    trap_00_01 = 0.5 * (jnp.clip(t_bottom, 0, 1) + jnp.clip(t_top, 0, 1))
    # (10,11): area = 1 - 0.5*((1-t_left) + (1-t_top)) ... actually:
    # Corners 10 and 11 below, corners 00 and 01 above.
    # Trapezoid has right edge fully below. Intersections on bottom and top.
    # area = 0.5*((1-t_bottom) + (1-t_top))
    trap_10_11 = 0.5 * ((1 - jnp.clip(t_bottom, 0, 1)) + (1 - jnp.clip(t_top, 0, 1)))
    # (01,11): area = 0.5*((1-t_left) + (1-t_right))
    trap_01_11 = 0.5 * ((1 - jnp.clip(t_left, 0, 1)) + (1 - jnp.clip(t_right, 0, 1)))

    area_2 = jnp.where(
        is_diag,
        0.5,
        jnp.where(
            is_adj_00_10,
            trap_00_10,
            jnp.where(
                is_adj_00_01,
                trap_00_01,
                jnp.where(
                    is_adj_10_11, trap_10_11, jnp.where(is_adj_01_11, trap_01_11, 0.5)
                ),
            ),
        ),
    )

    # Select by n_below
    area = jnp.where(
        n_below == 0,
        area_0,
        jnp.where(
            n_below == 1,
            area_1,
            jnp.where(n_below == 2, area_2, jnp.where(n_below == 3, area_3, area_4)),
        ),
    )

    return jnp.clip(area, 0.0, 1.0)


def _physical_area_below(C_phys, nx, ny, dx, dy):
    """Compute area of {nx*(x-xc) + ny*(y-yc) < C_phys} in cell.

    Works in physical coordinates directly, no sign ambiguity.
    """
    a = nx * dx
    b = ny * dy
    # C' = C_phys + 0.5*(a+b)
    Cp = C_phys + 0.5 * (a + b)

    # Four corners: (xi, eta) in {0,1}^2
    # val(xi, eta) = a*xi + b*eta
    # below = val < Cp
    v00 = jnp.zeros_like(Cp)
    v10 = a
    v01 = b
    v11 = a + b

    b00 = v00 < Cp
    b10 = v10 < Cp
    b01 = v01 < Cp
    b11 = v11 < Cp

    n_below = (
        b00.astype(jnp.int32)
        + b10.astype(jnp.int32)
        + b01.astype(jnp.int32)
        + b11.astype(jnp.int32)
    )

    # Intersection params on edges
    eps = 1e-30
    ab_safe_a = jnp.where(jnp.abs(a) > eps, a, eps)
    ab_safe_b = jnp.where(jnp.abs(b) > eps, b, eps)

    t_bottom = Cp / ab_safe_a  # edge (0,0)→(1,0)
    t_left = Cp / ab_safe_b  # edge (0,0)→(0,1)
    t_right = (Cp - a) / ab_safe_b  # edge (1,0)→(1,1)
    t_top = (Cp - b) / ab_safe_a  # edge (0,1)→(1,1)

    clip = lambda t: jnp.clip(t, 0.0, 1.0)

    # Areas for each n_below
    area_0 = jnp.zeros_like(Cp)
    area_4 = jnp.ones_like(Cp)

    area_1 = jnp.where(
        b00,
        0.5 * clip(t_bottom) * clip(t_left),
        jnp.where(
            b10,
            0.5 * (1 - clip(t_bottom)) * clip(t_right),
            jnp.where(
                b01,
                0.5 * (1 - clip(t_left)) * clip(t_top),
                0.5 * (1 - clip(t_right)) * (1 - clip(t_top)),
            ),
        ),
    )

    area_3 = 1.0 - jnp.where(
        ~b00,
        0.5 * clip(t_bottom) * clip(t_left),
        jnp.where(
            ~b10,
            0.5 * (1 - clip(t_bottom)) * clip(t_right),
            jnp.where(
                ~b01,
                0.5 * (1 - clip(t_left)) * clip(t_top),
                0.5 * (1 - clip(t_right)) * (1 - clip(t_top)),
            ),
        ),
    )

    # n=2: adjacent or diagonal
    is_diag = (b00 & b11 & ~b10 & ~b01) | (b10 & b01 & ~b00 & ~b11)

    trap_bottom = 0.5 * (clip(t_left) + clip(t_right))  # (00,10) below
    trap_left = 0.5 * (clip(t_bottom) + clip(t_top))  # (00,01) below
    trap_right = 0.5 * ((1 - clip(t_bottom)) + (1 - clip(t_top)))  # (10,11) below
    trap_top = 0.5 * ((1 - clip(t_left)) + (1 - clip(t_right)))  # (01,11) below

    area_2 = jnp.where(
        is_diag,
        0.5,
        jnp.where(
            b00 & b10,
            trap_bottom,
            jnp.where(
                b00 & b01,
                trap_left,
                jnp.where(b10 & b11, trap_right, jnp.where(b01 & b11, trap_top, 0.5)),
            ),
        ),
    )

    area = jnp.where(
        n_below == 0,
        area_0,
        jnp.where(
            n_below == 1,
            area_1,
            jnp.where(n_below == 2, area_2, jnp.where(n_below == 3, area_3, area_4)),
        ),
    )

    return jnp.clip(area, 0.0, 1.0)


def compute_intercept_C(F: Array, nx: Array, ny: Array, dx: float, dy: float) -> Array:
    """Compute PLIC intercept C via bisection in physical C space (15 iterations).

    The PLIC line: nx*(x - xc) + ny*(y - yc) = C.
    Bisection finds C such that area_below(C) = F.

    Args:
        F: volume fraction, shape (Nx+2nh, Ny+2nh, 1)
        nx, ny: normal components (same shape)
        dx, dy: cell sizes

    Returns:
        C: intercept, same shape as F.
    """
    # Physical C bounds
    C_half = jnp.abs(nx) * dx * 0.5 + jnp.abs(ny) * dy * 0.5
    C_lo = -C_half
    C_hi = C_half

    F_target = jnp.clip(F, 1e-10, 1.0 - 1e-10)

    def body_fn(i, state):
        C_mid, lo, hi = state
        area = _physical_area_below(C_mid, nx, ny, dx, dy)
        below = area < F_target
        lo_new = jnp.where(below, C_mid, lo)
        hi_new = jnp.where(below, hi, C_mid)
        mid_new = 0.5 * (lo_new + hi_new)
        return mid_new, lo_new, hi_new

    C_init = jnp.zeros_like(F)
    C_final, _, _ = jax.lax.fori_loop(0, 20, body_fn, (C_init, C_lo, C_hi))

    # Mask non-interface cells
    is_interface = (F > 1e-6) & (F < 1.0 - 1e-6)
    C_final = jnp.where(is_interface, C_final, 0.0)

    return C_final


def plic_line_segments_batch(nx_f, ny_f, C_f, F_arr, x_cc, y_cc, dx, dy):
    """Compute PLIC line segments for all interface cells (Python loop, for viz).

    Args:
        nx_f, ny_f, C_f: (Nx, Ny) arrays of normal and intercept
        F_arr: (Nx, Ny) volume fraction array
        x_cc, y_cc: (Nx,) and (Ny,) cell center coordinates
        dx, dy: cell sizes

    Returns:
        segments: list of [(x1,y1), (x2,y2)] pairs
    """
    import numpy as np

    Nx, Ny = nx_f.shape
    segments = []
    tol = 1e-10

    for i in range(Nx):
        for j in range(Ny):
            nxi, nyi, Ci = float(nx_f[i, j]), float(ny_f[i, j]), float(C_f[i, j])
            if abs(nxi) < 1e-10 and abs(nyi) < 1e-10:
                continue  # pure cell (n=0), skip

            xc, yc = float(x_cc[i]), float(y_cc[j])
            pts = []

            # Left edge: x = xc - dx/2
            if abs(nyi) > 1e-14:
                y_int = yc + (Ci + nxi * dx / 2) / nyi
                if yc - dy / 2 - tol <= y_int <= yc + dy / 2 + tol:
                    pts.append((xc - dx / 2, y_int))

            # Right edge: x = xc + dx/2
            if abs(nyi) > 1e-14:
                y_int = yc + (Ci - nxi * dx / 2) / nyi
                if yc - dy / 2 - tol <= y_int <= yc + dy / 2 + tol:
                    pts.append((xc + dx / 2, y_int))

            # Bottom edge: y = yc - dy/2
            if abs(nxi) > 1e-14:
                x_int = xc + (Ci + nyi * dy / 2) / nxi
                if xc - dx / 2 - tol <= x_int <= xc + dx / 2 + tol:
                    pts.append((x_int, yc - dy / 2))

            # Top edge: y = yc + dy/2
            if abs(nxi) > 1e-14:
                x_int = xc + (Ci - nyi * dy / 2) / nxi
                if xc - dx / 2 - tol <= x_int <= xc + dx / 2 + tol:
                    pts.append((x_int, yc + dy / 2))

            # Deduplicate
            unique = []
            for p in pts:
                if not any(
                    abs(p[0] - q[0]) < 1e-8 and abs(p[1] - q[1]) < 1e-8 for q in unique
                ):
                    unique.append(p)

            if len(unique) >= 2:
                segments.append([unique[0], unique[1]])

    return segments
