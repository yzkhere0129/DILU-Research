"""1D strip decomposition of a structured-grid OF LDU matrix.

Splits an LDU matrix into N ranks by partitioning cells in 1D along an axis
of the structured mesh.  Faces fall into:

    internal_local  : both owner and neighbour are local cells (no halo)
    interface       : one endpoint local, one on another rank.  We send the
                      local-side psi to the other rank, receive their side,
                      and add face_coeff * received to our local Apsi.

OF's processorPolyPatch stores exactly this: each side has its own
coefficient.  The "owner" side uses upper[f]; the "neighbour" side uses
lower[f].  In a symmetric (pd) matrix upper==lower so the distinction
collapses, but we keep them separate for the asymmetric (T eqn) case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from ..ldu import LDU


@dataclass
class HaloPatch:
    """Communication descriptor for ONE neighbouring rank.

    For every interface face between this rank and `other_rank`:
      - send `psi[local_cell[i]]` to other rank
      - receive one value from other rank, add `coeff[i] * received` to
        `Apsi[local_cell[i]]`
    """
    other_rank: int
    local_cell: np.ndarray   # int32, the LOCAL cell on this rank's side
    coeff: np.ndarray        # float64; upper[f] if we own face, lower[f] if other owns


@dataclass
class LocalLDU:
    rank: int
    n_ranks: int
    local_to_global: np.ndarray
    global_to_local: Dict[int, int]
    diag: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    halos: List[HaloPatch] = field(default_factory=list)

    @property
    def n_local(self) -> int: return self.diag.size
    @property
    def n_local_faces(self) -> int: return self.upper.size


def decompose_1d(ldu: LDU, n_ranks: int, mesh_dim: tuple[int, int, int],
                  axis: int = 1) -> List[LocalLDU]:
    """Contiguous 1D strip decomp along `axis` (0=x, 1=y, 2=z)."""
    NX, NY, NZ = mesh_dim
    N = NX * NY * NZ
    assert ldu.n_cells == N, f"mesh_dim mismatch: {ldu.n_cells} != {N}"

    cid = np.arange(N, dtype=np.int64)
    if   axis == 0: idx = cid % NX;                    n_axis = NX
    elif axis == 1: idx = (cid // NX) % NY;            n_axis = NY
    elif axis == 2: idx = cid // (NX * NY);            n_axis = NZ
    else: raise ValueError(axis)

    bounds = np.linspace(0, n_axis, n_ranks + 1, dtype=np.int64)
    rank_of_cell = np.empty(N, dtype=np.int32)
    for r in range(n_ranks):
        m = (idx >= bounds[r]) & (idx < bounds[r + 1])
        rank_of_cell[m] = r

    own_rank = rank_of_cell[ldu.owner]
    nei_rank = rank_of_cell[ldu.neighbour]

    out: List[LocalLDU] = []
    for r in range(n_ranks):
        gids = np.where(rank_of_cell == r)[0].astype(np.int64)
        n_loc = gids.size
        # Build global->local lookup as a dense array for speed:
        g2l_arr = -np.ones(N, dtype=np.int64)
        g2l_arr[gids] = np.arange(n_loc)

        # Internal-local faces
        internal = (own_rank == r) & (nei_rank == r)
        own_int = ldu.owner[internal]; nei_int = ldu.neighbour[internal]
        upper_int = ldu.upper[internal]; lower_int = ldu.lower[internal]
        own_int_l = g2l_arr[own_int].astype(np.int32)
        nei_int_l = g2l_arr[nei_int].astype(np.int32)

        local = LocalLDU(
            rank=r, n_ranks=n_ranks,
            local_to_global=gids,
            global_to_local={int(g): int(g2l_arr[g]) for g in gids},
            diag=ldu.diag[gids].astype(np.float64).copy(),
            lower=lower_int.astype(np.float64).copy(),
            upper=upper_int.astype(np.float64).copy(),
            owner=own_int_l,
            neighbour=nei_int_l,
        )

        for s in range(n_ranks):
            if s == r: continue
            # Face "owner=r, nei=s": rank r is the OWNER side → uses upper, local cell = owner
            mask_a = (own_rank == r) & (nei_rank == s)
            # Face "owner=s, nei=r": rank r is the NEIGHBOUR side → uses lower, local cell = neighbour
            mask_b = (own_rank == s) & (nei_rank == r)
            if not (mask_a.any() or mask_b.any()): continue

            local_a = g2l_arr[ldu.owner[mask_a]].astype(np.int32)
            local_b = g2l_arr[ldu.neighbour[mask_b]].astype(np.int32)
            coef_a = ldu.upper[mask_a].astype(np.float64)
            coef_b = ldu.lower[mask_b].astype(np.float64)

            local.halos.append(HaloPatch(
                other_rank=s,
                local_cell=np.concatenate([local_a, local_b]),
                coeff=np.concatenate([coef_a, coef_b]),
            ))

        out.append(local)
    return out
