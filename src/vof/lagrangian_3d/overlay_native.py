"""Native C/OpenMP overlay for Lagrangian VOF — jax.pure_callback wrapper.

Calls libhexboxclip.so for the overlay hot loop. Falls back to JAX batched
overlay if the shared library is not available or PLIC is requested.

Usage:
    from .overlay_native import overlay_lagrangian_3d_native
    F_new = overlay_lagrangian_3d_native(F, x_verts, y_verts, z_verts, grid)
"""

import os
import ctypes
import warnings
import numpy as np

import jax
import jax.numpy as jnp

# ══════════════════════════════════════════════════════════════
# Library loading (lazy, once)
# ══════════════════════════════════════════════════════════════

_LIB = None
_LOAD_ERR = None


def _load_lib():
    global _LIB, _LOAD_ERR
    if _LIB is not None:
        return _LIB
    if _LOAD_ERR is not None:
        return None

    lib_path = os.path.join(os.path.dirname(__file__), "csrc", "libhexboxclip.so")
    if not os.path.exists(lib_path):
        _LOAD_ERR = f"Not found: {lib_path}. Run 'make' in csrc/"
        return None

    try:
        lib = ctypes.CDLL(lib_path)

        lib.overlay_all_offsets.restype = None
        lib.overlay_all_offsets.argtypes = [
            ctypes.c_void_p,  # donor_hexes
            ctypes.c_void_p,  # donor_F
            ctypes.c_void_p,  # box_min_grid
            ctypes.c_void_p,  # box_max_grid
            ctypes.c_float,   # cell_vol
            ctypes.c_int32,   # nx
            ctypes.c_int32,   # ny
            ctypes.c_int32,   # nz
            ctypes.c_void_p,  # out_F
        ]

        lib.hex_box_vol_batch.restype = None
        lib.hex_box_vol_batch.argtypes = [
            ctypes.c_void_p,  # hex_verts
            ctypes.c_void_p,  # box_min
            ctypes.c_void_p,  # box_max
            ctypes.c_void_p,  # out_vols
            ctypes.c_int32,   # N
        ]

        lib.overlay_all_offsets_plic.restype = None
        lib.overlay_all_offsets_plic.argtypes = [
            ctypes.c_void_p,  # donor_hexes
            ctypes.c_void_p,  # donor_F
            ctypes.c_void_p,  # donor_plic_n
            ctypes.c_void_p,  # donor_plic_d
            ctypes.c_void_p,  # box_min_grid
            ctypes.c_void_p,  # box_max_grid
            ctypes.c_int32,   # nx
            ctypes.c_int32,   # ny
            ctypes.c_int32,   # nz
            ctypes.c_void_p,  # out_F
        ]

        _LIB = lib
        return lib
    except OSError as e:
        _LOAD_ERR = str(e)
        return None


def is_native_available():
    return _load_lib() is not None


def get_load_error():
    return _LOAD_ERR


# ══════════════════════════════════════════════════════════════
# Callback
# ══════════════════════════════════════════════════════════════

def _overlay_callback(hp_flat, Fp_flat, bmin_flat, bmax_flat,
                      cell_vol, nx, ny, nz):
    """CPU callback: np arrays in, np array out."""
    lib = _load_lib()

    # Ensure contiguous float32 (critical: no strides, no XLA layout issues)
    hp = np.ascontiguousarray(hp_flat, dtype=np.float32)
    Fp = np.ascontiguousarray(Fp_flat, dtype=np.float32)
    bmin = np.ascontiguousarray(bmin_flat, dtype=np.float32)
    bmax = np.ascontiguousarray(bmax_flat, dtype=np.float32)

    out = np.zeros(nx * ny * nz, dtype=np.float32)

    lib.overlay_all_offsets(
        hp.ctypes.data,
        Fp.ctypes.data,
        bmin.ctypes.data,
        bmax.ctypes.data,
        ctypes.c_float(cell_vol),
        ctypes.c_int32(nx),
        ctypes.c_int32(ny),
        ctypes.c_int32(nz),
        out.ctypes.data,
    )

    return out.reshape(nx, ny, nz)


def _overlay_plic_callback(hp_flat, Fp_flat, pn_flat, pd_flat,
                           bmin_flat, bmax_flat, nx, ny, nz):
    """CPU callback for PLIC overlay."""
    lib = _load_lib()

    hp   = np.ascontiguousarray(hp_flat, dtype=np.float32)
    Fp   = np.ascontiguousarray(Fp_flat, dtype=np.float32)
    pn   = np.ascontiguousarray(pn_flat, dtype=np.float32)
    pd   = np.ascontiguousarray(pd_flat, dtype=np.float32)
    bmin = np.ascontiguousarray(bmin_flat, dtype=np.float32)
    bmax = np.ascontiguousarray(bmax_flat, dtype=np.float32)

    out = np.zeros(nx * ny * nz, dtype=np.float32)

    lib.overlay_all_offsets_plic(
        hp.ctypes.data, Fp.ctypes.data,
        pn.ctypes.data, pd.ctypes.data,
        bmin.ctypes.data, bmax.ctypes.data,
        ctypes.c_int32(nx), ctypes.c_int32(ny), ctypes.c_int32(nz),
        out.ctypes.data,
    )

    return out.reshape(nx, ny, nz)


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def overlay_lagrangian_3d_native(
    F, x_verts, y_verts, z_verts, grid,
    nx_f=None, ny_f=None, nz_f=None, C_f=None,
    obs_mask=None,
):
    """Native C/OpenMP overlay with PLIC support.

    Falls back to JAX batched overlay only if libhexboxclip.so is not available.
    """
    if not is_native_available():
        warnings.warn(f"Native overlay unavailable ({get_load_error()}), "
                      "falling back to JAX batched overlay.")
        from .overlay_batched import overlay_lagrangian_3d_batched
        return overlay_lagrangian_3d_batched(
            F, x_verts, y_verts, z_verts, grid,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f, C_f=C_f,
            obs_mask=obs_mask)

    from .move_3d import build_deformed_hexahedra

    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    cell_vol = float(dx * dy * dz)

    hex_v = build_deformed_hexahedra(x_verts, y_verts, z_verts, grid)

    x_acc = jnp.linspace(grid.x_range[0], grid.x_range[1], nx + 1)
    y_acc = jnp.linspace(grid.y_range[0], grid.y_range[1], ny + 1)
    z_acc = jnp.linspace(grid.z_range[0], grid.z_range[1], nz + 1)

    F_int = F[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

    # Zero-padding = outflow boundary condition.
    # Fluid exiting the domain enters pad region with F=0 and leaves.
    # This is physically correct for open boundaries.
    # For wall/symmetry BCs, the CALLER should set F halos before calling.
    hp = jnp.pad(hex_v, ((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)))
    Fp = jnp.pad(F_int, ((1, 1), (1, 1), (1, 1)))

    # Acceptor box bounds
    ai, aj, ak = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    bmin_grid = jnp.stack([x_acc[ai], y_acc[aj], z_acc[ak]], axis=-1)
    bmax_grid = jnp.stack([x_acc[ai + 1], y_acc[aj + 1], z_acc[ak + 1]], axis=-1)

    # CRITICAL: flatten to contiguous 1D before passing to C
    hp_flat = jnp.array(hp).ravel()
    Fp_flat = jnp.array(Fp).ravel()
    bmin_flat = jnp.array(bmin_grid).ravel()
    bmax_flat = jnp.array(bmax_grid).ravel()

    result_shape = jax.ShapeDtypeStruct((nx, ny, nz), jnp.float32)

    if nx_f is not None:
        # PLIC path — use C SH overlay
        nx_int = nx_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        ny_int = ny_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        nz_int = nz_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]
        C_int  = C_f[nh:nh+nx, nh:nh+ny, nh:nh+nz, 0]

        hex_cx = jnp.mean(hex_v[..., 0], axis=-1)
        hex_cy = jnp.mean(hex_v[..., 1], axis=-1)
        hex_cz = jnp.mean(hex_v[..., 2], axis=-1)
        plic_d_int = C_int + nx_int * hex_cx + ny_int * hex_cy + nz_int * hex_cz

        is_full  = F_int > 1.0 - 1e-6
        is_empty = F_int < 1e-6
        plic_d_int = jnp.where(is_full, 1e6, jnp.where(is_empty, -1e6, plic_d_int))

        plic_n_int = jnp.stack([nx_int, ny_int, nz_int], axis=-1)
        has_normal = ~is_full & ~is_empty
        plic_n_int = jnp.where(has_normal[..., None], plic_n_int,
                               jnp.array([0.0, 0.0, 1.0]))

        pn_pad = jnp.pad(plic_n_int, ((1,1),(1,1),(1,1),(0,0)))
        pd_pad = jnp.pad(plic_d_int, ((1,1),(1,1),(1,1)))

        pn_flat = jnp.array(pn_pad).ravel()
        pd_flat = jnp.array(pd_pad).ravel()

        def _plic_cb(hp_a, Fp_a, pn_a, pd_a, bmin_a, bmax_a):
            return _overlay_plic_callback(
                np.asarray(hp_a), np.asarray(Fp_a),
                np.asarray(pn_a), np.asarray(pd_a),
                np.asarray(bmin_a), np.asarray(bmax_a),
                nx, ny, nz,
            )

        F_new = jax.pure_callback(
            _plic_cb, result_shape,
            hp_flat, Fp_flat, pn_flat, pd_flat, bmin_flat, bmax_flat,
        )
    else:
        # Non-PLIC path — use C tet-box overlay
        def _callback(hp_arr, Fp_arr, bmin_arr, bmax_arr):
            return _overlay_callback(
                np.asarray(hp_arr), np.asarray(Fp_arr),
                np.asarray(bmin_arr), np.asarray(bmax_arr),
                cell_vol, nx, ny, nz,
            )

        F_new = jax.pure_callback(
            _callback, result_shape,
            hp_flat, Fp_flat, bmin_flat, bmax_flat,
        )

    return F_new
