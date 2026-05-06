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

    eq_types = ["T_corr0", "pd_corr0", "pd_corr1", "pd_corr2"]

    # Same color/marker convention as plot_truth_vs_senior.py:
    #   blue  = our solver's distance to LU truth (= solver error)
    #   orange = our solver vs OF reference (= OF's noise)
    #   gray   = LU vs OF reference (also OF's noise; should overlap orange)
    style = {
        "PCG_LU": dict(color="tab:blue",   marker="o", linestyle="-"),
        "PCG_OF": dict(color="tab:orange", marker="s", linestyle="-"),
        "LU_OF":  dict(color="0.45",       marker="^", linestyle="--"),
    }

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0],
                           hspace=0.35, wspace=0.3)

    # Top row: per-equation precision lines
    for col, eq in enumerate(eq_types):
        ax = fig.add_subplot(gs[0, col])
        rows = [r for r in cases if r["eq"] == eq]
        if not rows:
            ax.set_title(f"{eq}  (no data)", fontsize=10); continue
        x = np.arange(len(rows))
        rel_pcg_lu = np.array([r["rel_PCG_LU"] for r in rows])
        rel_pcg_of = np.array([r["rel_PCG_OF"] for r in rows])
        rel_lu_of  = np.array([r["rel_LU_OF"]  for r in rows])

        ax.semilogy(x, np.maximum(rel_pcg_lu, 1e-17),
                     **style["PCG_LU"], markersize=5,
                     label="rel(PCG, LU)  ← solver error")
        ax.semilogy(x, np.maximum(rel_pcg_of, 1e-17),
                     **style["PCG_OF"], markersize=5,
                     label="rel(PCG, OF)  ← OF tol noise")
        ax.semilogy(x, np.maximum(rel_lu_of, 1e-17),
                     **style["LU_OF"], markersize=5,
                     label="rel(LU, OF)   ← OF tol noise")
        ax.axhline(1e-15, color="green", linestyle=":", alpha=0.6,
                    label="machine ε" if col == 0 else None)
        ax.axhline(1e-8,  color="red",   linestyle=":", alpha=0.6,
                    label="OF tol"     if col == 0 else None)

        N_str = f"N={rows[0]['n']}"
        ax.set_title(f"{eq} ({len(rows)} dumps, {N_str})", fontsize=10)
        ax.set_xlabel("dump index", fontsize=9)
        if col == 0:
            ax.set_ylabel("relative error (max-norm)", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper left")
        ax.set_ylim(1e-17, 1e-2)
        ax.grid(alpha=0.3, which="both")

    # Bottom-left: median bar chart per (eq × metric)
    ax_bar = fig.add_subplot(gs[1, 0:2])
    metrics = ["PCG_LU", "PCG_OF", "LU_OF"]
    metric_labels = ["rel(PCG, LU)", "rel(PCG, OF)", "rel(LU, OF)"]
    bar_x = np.arange(len(eq_types))
    bar_w = 0.27
    for i, m in enumerate(metrics):
        meds = []
        for eq in eq_types:
            rs = [r for r in cases if r["eq"] == eq]
            if rs:
                meds.append(np.median([r[f"rel_{m}"] for r in rs]))
            else:
                meds.append(np.nan)
        meds_safe = [max(v, 1e-17) for v in meds]
        ax_bar.bar(bar_x + (i-1)*bar_w, meds_safe, bar_w,
                    label=metric_labels[i],
                    color=style[m]["color"], alpha=0.8)
    ax_bar.set_yscale("log"); ax_bar.set_ylim(1e-17, 1e-2)
    ax_bar.set_xticks(bar_x); ax_bar.set_xticklabels(eq_types)
    ax_bar.set_ylabel("median relative error")
    ax_bar.set_title("Median relative error by equation type", fontsize=10)
    ax_bar.axhline(1e-15, color="green", linestyle=":", alpha=0.5)
    ax_bar.axhline(1e-8,  color="red",   linestyle=":", alpha=0.5)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(alpha=0.3, axis="y", which="both")

    # Bottom-right: summary text panel (same style as plot_truth_vs_senior)
    ax_text = fig.add_subplot(gs[1, 2:4]); ax_text.axis("off")
    lines = ["MEDIAN PRECISION ACROSS OUR OWN LPBF DUMPS",
              "="*55,
              f"{'equation':>14s} {'N':>6s} {'rel(PCG,LU)':>14s} "
              f"{'rel(PCG,OF)':>14s} {'rel(LU,OF)':>14s}",
              "-"*55]
    for eq in eq_types:
        rs = [r for r in cases if r["eq"] == eq]
        if not rs: continue
        N = rs[0]["n"]
        m1 = np.median([r["rel_PCG_LU"] for r in rs])
        m2 = np.median([r["rel_PCG_OF"] for r in rs])
        m3 = np.median([r["rel_LU_OF"] for r in rs])
        lines.append(f"{eq:>14s} {N:>6d} {m1:>14.2e} {m2:>14.2e} {m3:>14.2e}")
    lines += ["",
              "Reading the columns:",
              "  rel(PCG, LU) — our solver's distance from LU truth",
              "                 (≈ machine ε for T_corr0; ≈ 1e-9 for pd",
              "                  due to PCG tol=1e-12 + κ amplification)",
              "  rel(PCG, OF) — apparent gap to OF's reference x_final",
              "  rel(LU, OF)  — same gap. Identical to PCG's because both",
              "                 our PCG and LU agree to ~1e-9; OF reference",
              "                 has its own tol=1e-8 truncation noise."]
    ax_text.text(0.0, 1.0, "\n".join(lines),
                  fontfamily="monospace", fontsize=9,
                  verticalalignment="top")

    fig.suptitle(
        "Precision comparison: our PCG-DIC vs LU truth vs OpenFOAM reference\n"
        "OUR OWN consistent dumps (LPBF_sanity 2K cells + dumper_pipeline_test 8K cells)",
        fontsize=12, y=0.995)
    out = OUTDIR / "amgx_precision_vs_LU.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out}")

    # Console summary
    print()
    print("=" * 72)
    print(f"{'eq':>10s}  {'N':>6s}  {'med rel(PCG,LU)':>15s}  "
          f"{'med rel(PCG,OF)':>15s}  {'med rel(LU,OF)':>15s}")
    print("-" * 72)
    for eq in eq_types:
        rows = [r for r in cases if r["eq"] == eq]
        if not rows: continue
        N = rows[0]["n"]
        m_pcg_lu = np.median([r["rel_PCG_LU"] for r in rows])
        m_pcg_of = np.median([r["rel_PCG_OF"] for r in rows])
        m_lu_of  = np.median([r["rel_LU_OF"] for r in rows])
        print(f"{eq:>10s}  {N:>6d}  {m_pcg_lu:>15.2e}  "
              f"{m_pcg_of:>15.2e}  {m_lu_of:>15.2e}")


if __name__ == "__main__":
    main()
