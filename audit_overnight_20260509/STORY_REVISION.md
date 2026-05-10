LAST_REVIEWED: 2026-05-10T15:15+08:00
ITERATION: v1

# Story Revision — propagating H11 svds finding to derivative documents

Per closeout brief P7. The H11 finding (svds k=10 reveals near-null subspace cluster,
κ is method-dependent) was buried at the end of CLAIM_LEDGER.v3.md. It needs to be
propagated to 4 specific places. Below: the "before" text, the "after" text, the
specific line/section in each target file.

These are intended replacements. I am NOT writing them into the target files
directly (that would break the audit's "save .vN files don't overwrite" rule). The
user / next session should perform the actual replacements when promoting story to
public.

---

## REWRITE 1 — `docs/PROJECT_STATUS_REPORT.md` §1.2 row 2 about lab32

### BEFORE (current text in PROJECT_STATUS_REPORT.md §1.2)

> "lab32 5.9 kPa 偏离是 ill-posed null space artifact" → 用 SuperLU 直解证明 **system 良态**，5.9 kPa 是 OF tol=1e-8 + κ ~ 1e6 的标准放大

### AFTER (proposed replacement)

> "lab32 melt-380ns matrix's 5.9 kPa OF-vs-LU diff" — REVISED INTERPRETATION (per
> 2026-05-10 audit H11 svds k=10 finding):
>
> The matrix has a **10-dimensional near-null singular cluster** (smallest 10 σ
> ranging from 1.15e-16 to 9.29e-15, all below the fp64 noise floor of σ_max
> ≈ 1.45e-14). This means the matrix is **not strictly well-posed numerically** —
> SuperLU/CHOLMOD return *a* solution by pinning small pivots, but PCG iterative
> methods leave free a near-null component at their convergence tol.
>
> Therefore the 5.9 kPa diff between OF DICPCG (tol=1e-8) and LU is best understood
> as **the magnitude of the near-null-subspace component left free by OF's tol**;
> LU happens to pick a particular pinned member of that near-null family.
>
> The earlier "κ × tol with κ ~ 1e6" framing is INCORRECT in two ways:
> 1. κ is not a single number for this matrix — Lanczos shift-invert gives 3.2e+14,
>    svds gives 126, both at noise floor.
> 2. The bound κ × tol = 3.2e6 is theoretically loose; the observed 4.7e-3 is much
>    smaller because the near-null subspace is small-dim relative to the well-
>    conditioned subspace.
>
> Practical takeaway: this matrix should not be presented as a standard κ-amplification
> demonstration. It is a degenerate case (rays=0 → b≈0 → matrix-driven solution
> dominated by floor-level singular structure).

---

## REWRITE 2 — `docs/PROJECT_STATUS_REPORT.md` §1.1 table row "5.9 kPa story"

### BEFORE
| 5.9 kPa OF-vs-LU diff confirmed reproducible: 25 cells > 100 Pa, 4 cells > 1 kPa | re-cached LU + diff | Number stands |

### AFTER

| Lab32 5.9 kPa OF-vs-LU diff: 25 cells > 100 Pa, 4 cells > 1 kPa | re-cached LU (H2) + svds k=10 (H11) | Numbers stand; **interpretation revised**: matrix is numerically near-singular (10-dim near-null cluster σ ∈ [1e-16, 1e-14]); diff is OF leaving near-null component free, not classical κ × tol amplification |

---

## REWRITE 3 — `audit_overnight_20260509/CLAIM_LEDGER.v3.md` C012 verdict line

### BEFORE (line 31 of v3 master table)

| C012 | lab32 5.9 kPa = κ × tol; system well-posed (S4) | REFINED-VERIFIED | 75% | E09 (κ on more matrices) | A002, A010 |

### AFTER (proposed for v4)

| C012 | lab32 5.9 kPa diff explained by near-null subspace structure (svds k=10), NOT classical "κ × tol with single κ" | **VERIFIED-WITH-NUANCE** | 80% | E09 svds for 6 single_track | A002, A010, A021 |

Plus add to v4 paper-impact note:
> The "5.9 kPa is κ × tol amplification" one-line summary previously planned for paper
> abstract is **NO LONGER TENABLE**. Replacement summary: "On near-singular LPBF matrices,
> iterative solvers at engineering tol leave a near-null component free; LU picks one
> particular member. The cell-level diff (5.9 kPa peak in our case) is the magnitude
> of this near-null component, not the κ × tol error bound."

---

## REWRITE 4 — `audit_overnight_20260509/EVIDENCE_CHAIN.v2.md` C012 node explanation

### BEFORE (lines around C012 / C021 in v2)

```
[C012 lab32 5.9 kPa = κ × tol] REFINED-VERIFIED:
                |    - cache restored (H2)
                |    - κ measured at 3.2e+14 via Lanczos (H2)
                |    - σ_min = 4.5e-29 ⇒ matrix near-singular (new C021)
                |    - Predicted bound κ × tol = 3.2e6 (very loose); observed 4.7e-3 (much tighter)
                |    - Mechanism still κ × residual but EFFECTIVE κ in well-conditioned subspace 
                |       is much smaller than σ_max/σ_min.
```

### AFTER (proposed for v3)

```
[C012 lab32 5.9 kPa explained by near-null subspace structure] VERIFIED-WITH-NUANCE:
                |    - cache restored (H2 SuperLU rerun, x_LU available)
                |    - σ_min measurement is method-dependent and NEAR FP64 NOISE FLOOR:
                |        Lanczos shift-invert → 4.5e-29 (numerical noise convergence)
                |        svds(k=10, which='SM') → 1.15e-16 (still at noise but actually
                |                                  isolated by gap σ_2/σ_1 ≈ 13×)
                |    - σ_min "value" therefore not a single well-defined number; what IS
                |      well-defined is the EXISTENCE of near-null structure (10 σ all < 1e-13)
                |    - Implication: the formula "cell error ≤ κ × tol" with single-number κ
                |      does NOT apply. Observed diff bounded by:
                |        magnitude of x_OF's projection onto near-null subspace
                |    - This makes lab32 a DEGENERATE case (rays=0 → b ≈ 0 → matrix-driven
                |      solution dominated by near-null structure), not a typical LPBF pd matrix
                ↓
[C021 lab32 has 10-dim near-null singular cluster σ ∈ [1e-16, 1e-14]]   VERIFIED
        (svds k=10, multi-seed Lanczos confirms σ ≪ noise floor)
```

---

## Where these revisions should propagate (action items for next session)

| Target file | Current text | Replacement source |
|---|---|---|
| `docs/PROJECT_STATUS_REPORT.md` §1.2 row 2 | "ill-posed null space artifact retraction" | REWRITE 1 above |
| `docs/PROJECT_STATUS_REPORT.md` §1.1 table | "5.9 kPa story re-cached & verified" row | REWRITE 2 above |
| `audit_overnight_20260509/CLAIM_LEDGER.v3.md` C012 line | as quoted | REWRITE 3 above |
| `audit_overnight_20260509/EVIDENCE_CHAIN.v2.md` C012 node | as quoted | REWRITE 4 above |
| `docs/benchmark/data_inventory.md` DISCLAIMER section | "5.9 kPa = κ × tol with κ ~ 1e6" | adapt REWRITE 1 |

These are NOT done in this closeout (per R4: don't overwrite vN files). Promote
when next iteration session takes ownership of the public docs.

---

## Why this matters for paper

The previous narrative had a clean one-liner: "AMGx + IR matches LU to 1e-11; OF
tol=1e-8 deviates 5.9 kPa via standard κ amplification. AMGx in production beats LU
because of amortization."

The H11 finding **breaks the middle clause** of this one-liner. The 5.9 kPa is NOT a
standard amplification; it's a near-null artifact. This means:

1. **For publication**: cannot use lab32 as the "pretty κ × tol example". Must use a
   well-conditioned matrix (single_track? — pending E09 measurement) for that
   demonstration.

2. **For methodology**: the LU-as-truth strategy is fragile when matrix has near-null
   structure. LU's choice of pinned solution is implementation-dependent (CHOLMOD vs
   SuperLU vs MUMPS may pick different members of the near-null family). Verify with
   E07's SuperLU-vs-CHOLMOD cross-check.

3. **For the AMGx amortized story (S1/C009)**: unchanged — that hypothesis is
   independent of κ structure. E03/E04 still settle it.

The story revision is **NOT a defeat** — it's a substantive scientific clarification
that strengthens the paper's rigor when properly framed.
