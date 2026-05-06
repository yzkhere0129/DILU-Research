"""Three-way precision comparison: PCG vs LU vs OF reference.

For each pd dump on LPBF_sanity (2K cells) and dumper_pipeline_test
(8K cells), compute:
  rel(PCG, LU)  — our PCG vs SuperLU direct solve   ← machine-precision truth
  rel(PCG, OF)  — our PCG vs OpenFOAM x_final       ← OF tol=1e-8 reference
  rel(LU,  OF)  — SuperLU vs OpenFOAM x_final       ← shows OF's truncation

The story: rel(PCG,LU) ≪ rel(LU,OF) ≈ rel(PCG,OF), so OUR PCG matches the
true LU answer much better than OF's iterative reference does. Pretending
OF's x_final is "ground truth" makes us look bad, when actually it's OF's
1e-8 cut-off + κ amplification that produces the apparent gap.

Output: docs/benchmark/figures/amgx_precision_vs_LU.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dilu.openfoam_cpu.python.ldu import csr_to_ldu
from dilu.openfoam_cpu.python import pcg

OUTDIR = Path("/home/yzk/DILU-Research/docs/benchmark/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)
LMF = Path("/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam")


def collect(case_name: str) -> list[dict]:
    base = LMF / case_name / "postProcessing/matrices"
    out = []
    for ts in sorted(base.iterdir()):
        if not ts.is_dir(): continue
        for eq in sorted(ts.iterdir()):
            if not eq.is_dir(): continue
            try:
                A = sio.mmread(str(eq/"A.mm")).tocsc()
                b = sio.mmread(str(eq/"b.mm")).flatten()
                x_final = sio.mmread(str(eq/"x_final.mm")).flatten()
                x0 = sio.mmread(str(eq/"x0.mm")).flatten()
            except Exception:
                continue

            lu = splu(A, permc_spec="COLAMD")
            x_LU = lu.solve(b)

            ldu = csr_to_ldu(A.tocsr())
            res = pcg.solve(ldu, b, x0.copy(),
                            tolerance=1e-12, min_iter=1, max_iter=500)
            denom = max(float(np.abs(x_LU).max()), 1e-300)
            out.append(dict(
                case=case_name, ts=ts.name, eq=eq.name,
                n=A.shape[0],
                rel_PCG_LU=float(np.abs(res.x - x_LU).max() / denom),
                rel_PCG_OF=float(np.abs(res.x - x_final).max() / denom),
                rel_LU_OF=float(np.abs(x_LU - x_final).max() / denom),
                pcg_iter=res.n_iterations,
            ))
    return out


def main():
    cases = []
    for name in ["LPBF_sanity", "dumper_pipeline_test"]:
        print(f"Processing {name} ...")
        cases.extend(collect(name))
    print(f"  collected {len(cases)} dumps")

    # Split by equation type
    eq_types = ["T_corr0", "pd_corr0", "pd_corr1", "pd_corr2"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, eq in zip(axes, eq_types):
        rows = [r for r in cases if r["eq"] == eq]
        if not rows:
            ax.set_title(f"{eq}\n(no data)"); continue
        x = np.arange(len(rows))
        rel_pcg_lu = np.array([r["rel_PCG_LU"] for r in rows])
        rel_pcg_of = np.array([r["rel_PCG_OF"] for r in rows])
        rel_lu_of  = np.array([r["rel_LU_OF"]  for r in rows])

        ax.semilogy(x, np.maximum(rel_pcg_lu, 1e-17), "o-",
                    label="rel(PCG, LU)  ← truth match",
                    color="tab:blue", markersize=4)
        ax.semilogy(x, np.maximum(rel_pcg_of, 1e-17), "s-",
                    label="rel(PCG, OF)  ← OF tol noise",
                    color="tab:orange", markersize=4)
        ax.semilogy(x, np.maximum(rel_lu_of, 1e-17), "^--",
                    label="rel(LU, OF)   ← also OF noise",
                    color="tab:gray", markersize=4)
        ax.axhline(1e-15, color="green", linestyle=":", alpha=0.6,
                   label="machine ε")
        ax.axhline(1e-8, color="red", linestyle=":", alpha=0.6,
                   label="OF tol")
        ax.set_title(f"{eq} ({len(rows)} dumps)", fontsize=11)
        ax.set_xlabel("dump index", fontsize=9)
        ax.set_ylim(1e-17, 1e-2)
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0]:
            ax.set_ylabel("relative error (max-norm)")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Precision comparison: our PCG-DIC vs LU truth vs OpenFOAM reference\n"
        "On OUR OWN dumps (LPBF_sanity 2K cells + dumper_pipeline_test 8K cells)",
        fontsize=12)
    fig.tight_layout()
    out = OUTDIR / "amgx_precision_vs_LU.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out}")

    # Also print the summary medians
    print()
    print("=" * 72)
    print(f"{'eq':>10s}  {'N':>6s}  {'med rel(PCG,LU)':>15s}  {'med rel(PCG,OF)':>15s}  {'med rel(LU,OF)':>15s}")
    print("-" * 72)
    for eq in eq_types:
        rows = [r for r in cases if r["eq"] == eq]
        if not rows: continue
        N = rows[0]["n"]
        m_pcg_lu = np.median([r["rel_PCG_LU"] for r in rows])
        m_pcg_of = np.median([r["rel_PCG_OF"] for r in rows])
        m_lu_of  = np.median([r["rel_LU_OF"] for r in rows])
        print(f"{eq:>10s}  {N:>6d}  {m_pcg_lu:>15.2e}  {m_pcg_of:>15.2e}  {m_lu_of:>15.2e}")


if __name__ == "__main__":
    main()
