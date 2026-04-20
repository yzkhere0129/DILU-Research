/*
 * hex_box_clip.h — Native C/OpenMP kernel for Lagrangian VOF overlay.
 *
 * Computes vol(hex ∩ axis-aligned-box) via 6-tet decomposition + sequential
 * plane clipping with stack-allocated sub-tet pools.
 *
 * NOTE on 6-tet decomposition (mathematical caveat):
 *   Post-Lagrangian-deformation, hex faces are bilinear patches, not planes.
 *   The 6-tet decomposition assumes planar faces — a first-order geometric
 *   approximation. Volume error is O(CFL * du/dx * dx²). This is the same
 *   approximation used by FLOW-3D and similar production VOF codes.
 */

#ifndef HEX_BOX_CLIP_H
#define HEX_BOX_CLIP_H

#include <stdint.h>

/*
 * Compute vol(hex_i ∩ box_i) for N cell pairs.
 *
 * hex_verts: (N * 8 * 3) float32, row-major contiguous
 * box_min:   (N * 3)     float32, contiguous
 * box_max:   (N * 3)     float32, contiguous
 * out_vols:  (N)         float32, OUTPUT
 * N:         number of cells
 */
void hex_box_vol_batch(
    const float* hex_verts,
    const float* box_min,
    const float* box_max,
    float*       out_vols,
    int32_t      N
);

/*
 * Full overlay step: process all 27 neighbor offsets internally.
 *
 * donor_hexes:  ((nx+2)*(ny+2)*(nz+2)*8*3) float32, row-major contiguous
 *               1-padded deformed hex vertices
 * donor_F:      ((nx+2)*(ny+2)*(nz+2))     float32, contiguous
 *               1-padded volume fractions
 * box_min_grid: (nx*ny*nz*3)               float32, contiguous
 * box_max_grid: (nx*ny*nz*3)               float32, contiguous
 * cell_vol:     dx * dy * dz
 * nx, ny, nz:   interior grid dimensions
 * out_F:        (nx*ny*nz)                  float32, OUTPUT (clipped to [0,1])
 */
void overlay_all_offsets(
    const float* donor_hexes,
    const float* donor_F,
    const float* box_min_grid,
    const float* box_max_grid,
    float        cell_vol,
    int32_t      nx,
    int32_t      ny,
    int32_t      nz,
    float*       out_F
);

/*
 * PLIC overlay step: process all 27 neighbor offsets with PLIC clipping.
 *
 * Same padding convention as overlay_all_offsets. Additionally:
 * donor_plic_n: ((nx+2)*(ny+2)*(nz+2)*3) float32, PLIC normals (fluid→gas)
 * donor_plic_d: ((nx+2)*(ny+2)*(nz+2))   float32, PLIC intercepts
 *
 * out_F is RAW (not clipped to [0,1]) — caller handles clip.
 */
void overlay_all_offsets_plic(
    const float* donor_hexes,
    const float* donor_F,
    const float* donor_plic_n,
    const float* donor_plic_d,
    const float* box_min_grid,
    const float* box_max_grid,
    int32_t      nx,
    int32_t      ny,
    int32_t      nz,
    float*       out_F
);

#endif /* HEX_BOX_CLIP_H */
