"""Step 2: 3D Lagrangian Face Move & Hexahedron Decomposition — JAX-Native.

Computes the displacement of all cell faces under a 3D velocity field,
following Barkhudarov (2004) Eq. 6-7 with second-order correction.

Each face moves by:
    dx_face = u · dt / (1 + 0.5 · du/dx · dt)

Vertex positions are computed as the average of adjacent face displacements.

Additionally provides the hexahedron → 5-tetrahedron decomposition used
in the overlay step for face-topology clipping.
"""

import jax.numpy as jnp
from ...data_types import GridInfo, Array


# ═══════════════════════════════════════════════════════════════
# 6-tetrahedron decomposition table (compile-time constant)
# ═══════════════════════════════════════════════════════════════
#
#   Hex vertex layout (right-hand convention):
#
#       v4 -------- v5
#      /|          /|
#     / |         / |
#    v7 -------- v6 |
#    |  |        |  |
#    |  v0 ------|-- v1
#    | /         | /
#    |/          |/
#    v3 -------- v2
#
#   Verified 6-tet decomposition.
#   Three internal face diagonals: (0,2), (0,4), (2,5).
#   Each tet has exactly 2 boundary faces + 2 internal faces.
#   Verified: Σ Vol(Tᵢ) = 1.0 for unit cube.
#
#   T1 = (0, 1, 2, 4)   Vol = 1/6
#   T2 = (1, 2, 4, 5)   Vol = 1/6
#   T3 = (2, 4, 5, 6)   Vol = 1/6
#   T4 = (2, 3, 4, 7)   Vol = 1/6
#   T5 = (2, 4, 6, 7)   Vol = 1/6
#   T6 = (0, 2, 3, 4)   Vol = 1/6

TET_INDICES = jnp.array(
    [
        [0, 1, 2, 4],  # T1
        [1, 2, 4, 5],  # T2
        [2, 4, 5, 6],  # T3
        [2, 3, 4, 7],  # T4
        [2, 4, 6, 7],  # T5
        [0, 2, 3, 4],  # T6
    ],
    dtype=jnp.int32,
)  # (6, 4)


def decompose_hex_to_tets(hex_verts: Array) -> Array:
    """Decompose a hexahedron into 6 tetrahedra.

    Correct decomposition: all 6 tetrahedra tile the hexahedron
    with zero overlap and zero gaps. Verified: Σ Vol(Tᵢ) = dx·dy·dz.

    Args:
        hex_verts: (..., 8, 3) — hexahedron vertex coordinates.

    Returns:
        tets_verts: (..., 6, 4, 3) — six tetrahedra vertex coordinates.
    """
    return hex_verts[..., TET_INDICES, :]  # (..., 6, 4, 3)


# ═══════════════════════════════════════════════════════════════
# 3D Lagrangian face displacement
# ═══════════════════════════════════════════════════════════════


def _vertex_displacement_from_faces(
    face_disp: Array,
    axis: int,
    Ntot_x: int,
    Ntot_y: int,
    Ntot_z: int,
) -> Array:
    """Convert face displacements to vertex displacements along one axis.

    Each vertex displacement = average of the two adjacent face displacements.
    Boundary faces are replicated (zero gradient BC).

    Args:
        face_disp: face displacements along `axis`.
            axis=0: (Ntot_x-1, Ntot_y, Ntot_z)
            axis=1: (Ntot_x, Ntot_y-1, Ntot_z)
            axis=2: (Ntot_x, Ntot_y, Ntot_z-1)
        axis: spatial axis (0=x, 1=y, 2=z).
        Ntot_x, Ntot_y, Ntot_z: total grid dimensions (incl. halos).

    Returns:
        vert_disp: (Ntot_x+1, Ntot_y+1, Ntot_z+1)
    """
    full = [Ntot_x, Ntot_y, Ntot_z]
    n_face = full[axis] - 1  # number of faces along this axis

    # Step 1: insert boundary face values → (Ntot+1) along axis
    shape_all = list(full)
    shape_all[axis] = full[axis] + 1  # Ntot+1 entries
    disp_all = jnp.zeros(tuple(shape_all))

    # Slice objects for indexing
    s0 = [slice(None)] * 3
    s1 = [slice(None)] * 3
    s2 = [slice(None)] * 3

    # disp_all[0] = face_disp[0] (copy boundary)
    s_bdy_lo = list(s0)
    s_bdy_lo[axis] = 0
    s_face_lo = list(s0)
    s_face_lo[axis] = 0
    disp_all = disp_all.at[tuple(s_bdy_lo)].set(face_disp[tuple(s_face_lo)])

    # disp_all[1:-1] = face_disp[:] (all interior faces)
    s_bdy_int = list(s0)
    s_bdy_int[axis] = slice(1, -1)
    disp_all = disp_all.at[tuple(s_bdy_int)].set(face_disp)

    # disp_all[-1] = face_disp[-1] (copy boundary)
    s_bdy_hi = list(s0)
    s_bdy_hi[axis] = -1
    s_face_hi = list(s0)
    s_face_hi[axis] = -1
    disp_all = disp_all.at[tuple(s_bdy_hi)].set(face_disp[tuple(s_face_hi)])

    # Step 2: vertex displacement = average of adjacent faces
    s_lo = list(s0)
    s_lo[axis] = slice(0, -1)
    s_hi = list(s0)
    s_hi[axis] = slice(1, None)
    vert_raw = 0.5 * (disp_all[tuple(s_lo)] + disp_all[tuple(s_hi)])

    # vert_raw shape: (Ntot_x, Ntot_y, Ntot_z)
    # Need to pad ALL three axes to (Ntot_x+1, Ntot_y+1, Ntot_z+1)
    # by copying the boundary vertex displacement.

    result = vert_raw
    for ax in [0, 1, 2]:
        pad_spec = [(0, 0), (0, 0), (0, 0)]
        pad_spec[ax] = (0, 1)  # pad 1 after along axis ax
        result = jnp.pad(result, pad_spec)

        # Copy second-to-last → last along this axis
        slc_set = [slice(None)] * 3
        slc_set[ax] = -1
        slc_get = [slice(None)] * 3
        slc_get[ax] = -2
        result = result.at[tuple(slc_set)].set(result[tuple(slc_get)])

    return result  # (Ntot_x+1, Ntot_y+1, Ntot_z+1)


def lagrangian_move_faces_3d(
    u_face: Array,
    v_face: Array,
    w_face: Array,
    grid: GridInfo,
    dt: float,
) -> tuple:
    """Compute deformed vertex positions after 3D Lagrangian face move.

    Args:
        u_face: x-face velocities, shape (Ntot_x-1, Ntot_y, Ntot_z).
        v_face: y-face velocities, shape (Ntot_x, Ntot_y-1, Ntot_z).
        w_face: z-face velocities, shape (Ntot_x, Ntot_y, Ntot_z-1).
        grid: GridInfo with nx, ny, nz, nh, dx, dy, dz, *_range.
        dt: timestep.

    Returns:
        x_verts, y_verts, z_verts: each (Ntot_x+1, Ntot_y+1, Ntot_z+1).
            Deformed vertex coordinates.
    """
    nh = grid.nh
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    Ntot_x = grid.nx + 2 * nh
    Ntot_y = grid.ny + 2 * nh
    Ntot_z = grid.nz + 2 * nh

    # ── x-face displacements ──
    dudx = jnp.zeros_like(u_face)
    dudx = dudx.at[1:-1, :, :].set((u_face[2:, :, :] - u_face[:-2, :, :]) / (2.0 * dx))
    dx_face = u_face * dt / (1.0 + 0.5 * dudx * dt)

    # ── y-face displacements ──
    dvdy = jnp.zeros_like(v_face)
    dvdy = dvdy.at[:, 1:-1, :].set((v_face[:, 2:, :] - v_face[:, :-2, :]) / (2.0 * dy))
    dy_face = v_face * dt / (1.0 + 0.5 * dvdy * dt)

    # ── z-face displacements ──
    dwdz = jnp.zeros_like(w_face)
    dwdz = dwdz.at[:, :, 1:-1].set((w_face[:, :, 2:] - w_face[:, :, :-2]) / (2.0 * dz))
    dz_face = w_face * dt / (1.0 + 0.5 * dwdz * dt)

    # ── Vertex displacements from adjacent face averages ──
    dx_verts = _vertex_displacement_from_faces(
        dx_face, axis=0, Ntot_x=Ntot_x, Ntot_y=Ntot_y, Ntot_z=Ntot_z
    )
    dy_verts = _vertex_displacement_from_faces(
        dy_face, axis=1, Ntot_x=Ntot_x, Ntot_y=Ntot_y, Ntot_z=Ntot_z
    )
    dz_verts = _vertex_displacement_from_faces(
        dz_face, axis=2, Ntot_x=Ntot_x, Ntot_y=Ntot_y, Ntot_z=Ntot_z
    )

    # ── Original vertex positions ──
    x_orig = grid.x_range[0] - nh * dx + jnp.arange(Ntot_x + 1) * dx
    y_orig = grid.y_range[0] - nh * dy + jnp.arange(Ntot_y + 1) * dy
    z_orig = grid.z_range[0] - nh * dz + jnp.arange(Ntot_z + 1) * dz

    x_grid, y_grid, z_grid = jnp.meshgrid(x_orig, y_orig, z_orig, indexing="ij")
    x_verts_0 = x_grid  # (Ntot_x+1, Ntot_y+1, Ntot_z+1)
    y_verts_0 = y_grid
    z_verts_0 = z_grid

    x_verts = x_verts_0 + dx_verts
    y_verts = y_verts_0 + dy_verts
    z_verts = z_verts_0 + dz_verts

    return x_verts, y_verts, z_verts


# ═══════════════════════════════════════════════════════════════
# Deformed hexahedra construction
# ═══════════════════════════════════════════════════════════════


def build_deformed_hexahedra(
    x_verts: Array,
    y_verts: Array,
    z_verts: Array,
    grid: GridInfo,
) -> Array:
    """Build deformed hexahedra from vertex displacement arrays.

    Vertex ordering (right-hand convention):

        v4 -------- v5          v0 = (0,0,0)  bottom-near-left
       /|          /|           v1 = (1,0,0)  bottom-near-right
      / |         / |           v2 = (1,1,0)  bottom-far-right
     v7 -------- v6 |          v3 = (0,1,0)  bottom-far-left
     |  |        |  |           v4 = (0,0,1)  top-near-left
     |  v0 ------|-- v1         v5 = (1,0,1)  top-near-right
     | /         | /            v6 = (1,1,1)  top-far-right
     |/          |/             v7 = (0,1,1)  top-far-left
     v3 -------- v2

    Args:
        x_verts, y_verts, z_verts: (Ntot_x+1, Ntot_y+1, Ntot_z+1).
        grid: GridInfo.

    Returns:
        hex_verts: (Nx, Ny, Nz, 8, 3) — deformed hexahedra for all cells.
    """
    nh = grid.nh
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Interior cell index ranges (with halo offset)
    ii = jnp.arange(nx)[:, None, None] + nh  # (Nx, 1, 1)
    jj = jnp.arange(ny)[None, :, None] + nh  # (1, Ny, 1)
    kk = jnp.arange(nz)[None, None, :] + nh  # (1, 1, Nz)

    def _vert(di, dj, dk):
        return jnp.stack(
            [
                x_verts[ii + di, jj + dj, kk + dk],
                y_verts[ii + di, jj + dj, kk + dk],
                z_verts[ii + di, jj + dj, kk + dk],
            ],
            axis=-1,
        )  # (Nx, Ny, Nz, 3)

    # Stack 8 vertices: (Nx, Ny, Nz, 8, 3)
    hex_verts = jnp.stack(
        [
            _vert(0, 0, 0),  # v0
            _vert(1, 0, 0),  # v1
            _vert(1, 1, 0),  # v2
            _vert(0, 1, 0),  # v3
            _vert(0, 0, 1),  # v4
            _vert(1, 0, 1),  # v5
            _vert(1, 1, 1),  # v6
            _vert(0, 1, 1),  # v7
        ],
        axis=3,
    )

    return hex_verts
