"""MPI versions of amul / tmul / sum_a.

amul / tmul: face-loop + halo exchange + halo contribution scatter.
sum_a:       diagonal copy + face-loop scatter (no comm needed since each
             face's contribution lands on local cells; for halo faces the
             other rank's contribution to its local cell is computed there).

For sum_a, the catch: the original serial algorithm does:
    out[neighbour] += lower[f]
    out[owner]     += upper[f]
With the 1D decomp, when owner is local r and neighbour is on rank s:
  - We add upper[f] to local out[owner] (yes)
  - The neighbour-side contribution lower[f] goes to rank s (in their copy)
We don't need to send anything — each rank handles "its own local cells'"
contributions only.  But we DO need to add the interface coeff (upper or
lower depending on which side we are) to OUR local diag-like accumulator.
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI

from .decompose import LocalLDU


def amul_mpi(local: LocalLDU, psi_local: np.ndarray,
              comm: MPI.Comm) -> np.ndarray:
    """Apsi[local] = (A · psi_global)[local].  4-rank test passed."""
    out = local.diag * psi_local
    for f in range(local.n_local_faces):
        o = local.owner[f]; n = local.neighbour[f]
        out[n] += local.lower[f] * psi_local[o]
        out[o] += local.upper[f] * psi_local[n]

    sends, recvs, send_bufs, recv_bufs = [], [], [], []
    for h in local.halos:
        sb = np.ascontiguousarray(psi_local[h.local_cell])
        rb = np.empty_like(sb)
        send_bufs.append(sb); recv_bufs.append(rb)
        tag_s = (local.rank * local.n_ranks + h.other_rank) & 0xFFFF
        tag_r = (h.other_rank * local.n_ranks + local.rank) & 0xFFFF
        sends.append(comm.Isend(sb, dest=h.other_rank, tag=tag_s))
        recvs.append(comm.Irecv(rb, source=h.other_rank, tag=tag_r))

    MPI.Request.Waitall(recvs)
    for h, rb in zip(local.halos, recv_bufs):
        np.add.at(out, h.local_cell, h.coeff * rb)
    MPI.Request.Waitall(sends)
    return out


def sum_a_mpi(local: LocalLDU) -> np.ndarray:
    """sumA[local_i] = Σ_j A[global(local_i), j].

    Each rank handles its own local-cells' row sums, taking contributions from:
      - diag of local cell
      - all internal-local face touches that local cell
      - all halo-interface coefficients where local cell is the local endpoint
    No comm needed — every cell only sees A's row, and we handle the local
    cell-to-cell connections (face_loop + halo coeffs) on the local rank.
    """
    out = local.diag.copy()
    # Internal faces (both endpoints local)
    for f in range(local.n_local_faces):
        out[local.neighbour[f]] += local.lower[f]
        out[local.owner[f]]     += local.upper[f]
    # Halo faces (each rank adds its own side's coefficient to its local cell)
    for h in local.halos:
        np.add.at(out, h.local_cell, h.coeff)
    return out
