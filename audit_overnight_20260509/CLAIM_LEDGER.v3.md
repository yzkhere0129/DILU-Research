LAST_REVIEWED: 2026-05-10T04:15+08:00
ITERATION: v3

# Claim Ledger v3 — final pre-Xeon snapshot

Changes from v2:
- Cleaner cross-references between claims and adversary attacks A001-A020.
- Confidence column added per claim (matches EVIDENCE_CHAIN.v2.md).
- Each claim has explicit "Xeon experiment ID" mapping.
- Verdict column normalized.

(C001-C021 unchanged in substance from v2; this is a v3 reorganization with adversary integration.)

---

## Master table of all claims (v3 status)

| ID | One-line claim | Verdict | Confidence | Xeon Expt | Adversary refs |
|---|---|---|---|---|---|
| C001 | AMGx + IR matches scipy LU on 8K cells max rel 2.89e-15 | VERIFIED | 95% | (locked) | A001 |
| C002 | OF DICPCG self-residual ≈ 1e-8 on 6 single_track timesteps | VERIFIED | 95% | (locked, recompute confirmed) | — |
| C003 | AMGx+IR algebraic residual ≈ 1.02e-15 on 6 timesteps | VERIFIED | 95% | (locked) | — |
| C004 | AMGx+IR vs LU truth max rel ≤ 1.13e-11 on 500K (S3) | HYPOTHESIS-CONDITIONAL | 50% | E07 | A001, A011 |
| C005 | OF DICPCG iter for pd_corr0 = {7,35,40,17,43,65} | VERIFIED | 95% | (locked) | — |
| C006 | OF wall = {770..7150} ms (S2) | REFUTED-AS-MEASUREMENT | 25% | E01 | A011, A018 |
| C007 | AMGx PCG iter @1e-8 = {497,535,539,709,435,773} | VERIFIED | 90% | (locked, interpretation needs A019) | A019 |
| C008 | AMGx wall 9-39s on RTX 3050 single-shot | VERIFIED-scoped | 80% | (re-verify on lab GPU via E02) | A005, A006 |
| C009 | AMGx amortized + warm-start beats LU 100-1000× (S1) | HYPOTHESIS | 30% | E03 + E04 | A012, A015 |
| C010 | OF beats AMGx 5-10× on 500K | REFUTED-conclusion | 25% | E01 + E02 + E03 | A018 |
| C011 | CHOLMOD wall 69-75s on lab Xeon | VERIFIED-CONDITIONAL | 75% | E05 | A007 |
| C012 | lab32 5.9 kPa = κ × tol; system well-posed (S4) | REFINED-VERIFIED | 75% | E09 (κ on more matrices) | A002, A010 |
| C013 | Phase 0 PID 466877 running (S6) | REFUTED | 5% | (no need; already settled) | — |
| C014 | matrixDumper 32-rank bug fixed (S5) | PARTIALLY-VERIFIED | 60% | E10 (optional) | — |
| C015 | AMGx amortized: setup 2-3s, update 50-100ms, solve 100-150ms | HYPOTHESIS | 30% | E03 | A003, A012 |
| C016 | warm-start saves 50-90% iter (S1) | HYPOTHESIS | 30% | E04 | A015 |
| C017 | CHOLMOD symbolic-reuse 2-5× speedup | HYPOTHESIS | 30% | E06 | — |
| C018 | AMGx wins at >5M cells / multi-GPU (S8) | REFUTED-as-validated | 20% | (out of resources) | — |
| C019 | senior melt/evap matrices unsolvable | VERIFIED-data-fact | 75% | (cannot settle without senior) | — |
| C020 | /tmp/bench_gold.log unknown provenance | UNKNOWN | n/a | (administrative) | — |
| C021 | lab32 σ_min ≈ 4.5e-29 — near-singular | VERIFIED | 80% | E09 (multi-seed verify) | A010 |

## SUSPECT_CLAIMS final mapping

| S | Maps to | v3 Verdict | Settle path |
|---|---|---|---|
| S1 (amortized 100-1000×) | C009 | HYPOTHESIS | E03 + E04 |
| S2 (OF wall estimated) | C006 | REFUTED-AS-MEASUREMENT | E01 |
| S3 (AMGx+IR vs LU 1.13e-11 6/6) | C004 | HYPOTHESIS-CONDITIONAL | E07 |
| S4 (5.9 kPa = κ × tol) | C012 | REFINED-VERIFIED | E09 (more matrices) |
| S5 (matrixDumper bug fix) | C014 | PARTIALLY-VERIFIED | E10 optional |
| S6 (Phase 0 running) | C013 | REFUTED | (settled tonight) |
| S7 (iter 17-65 vs 435-773) | C005 + C007 | VERIFIED-numbers | (interpretation OK with config caveat) |
| S8 (AMGx >5M) | C018 | REFUTED-as-validated | (out of resource range) |

## Confidence distribution

```
       95%: 4 claims (C001, C002, C003, C005)
       90%: 1 claim (C007)
       80%: 2 claims (C008, C021)
       75%: 3 claims (C011, C012, C019)
       60%: 1 claim (C014)
       50%: 1 claim (C004)
       30%: 4 claims (C009, C015, C016, C017)
       25%: 2 claims (C006, C010)
       20%: 1 claim (C018)
        5%: 1 claim (C013, REFUTED hard)
        n/a: 1 claim (C020)
       
Mean: ~58%; post-Xeon predicted ~85%.
```

## What MUST get settled tomorrow on Xeon

| Priority | Experiment | Settles |
|---|---|---|
| HIGH | E01 (OF wall) | C006 → 95% (or revise) |
| HIGH | E03 (AMGx amortized) | C009, C015 → 95% |
| HIGH | E04 (AMGx warm) | C016 → 95% |
| HIGH | E07 (LU vs AMGx_IR) | C004 → 95% |
| HIGH | E08 (headline ratios) | C010 ratio computed |
| MED | E02 (AMGx baseline lab GPU) | C008 re-scoped |
| MED | E05 (CHOLMOD baseline) | C011 → 95% |
| MED | E06 (CHOLMOD symbolic) | C017 → 95% |
| MED | E09 (Lanczos) | C012, C021 → 95% |
| LOW | E10 optional | C014 → 95% if budget |
| OUT | (n/a) | C018, C019, C020 not settle-able |
