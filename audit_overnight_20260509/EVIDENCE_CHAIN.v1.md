LAST_REVIEWED: 2026-05-10T01:45+08:00
ITERATION: v1

# Evidence DAG — Claim Dependencies

Edges labelled by strength: STRONG (file:line + math), MEDIUM (estimate + theory), WEAK (assertion).

## Claim DAG (ASCII)

```
                                   [C001 8K AMGx vs LU 2.89e-15]   STRONG (precision_results_ir1.json)
                                              |
                                              | (foundation for "AMGx algo correct")
                                              ↓ MEDIUM
                  +--------------- [AMGx algorithm implementation correct]
                  |                       (no separate C — implicit)
                  |
        STRONG    |                                [C019 senior melt/evap unsolvable]
        ↓         ↓                                       | VERIFIED-data
[C003 algebraic res 1e-15]    [C014 dumper bug fix]       | (cause unknown)
       |                              | code-correct     ↓
       |                              | NOT verified ON  [matrixDumper bug attribution]
       |                              ↓ senior data        UNRESOLVED
       |                       [C014 partial]
       |
       ↓
[C004 AMGx+IR vs LU 1.13e-11 (500K)] HYPOTHESIS-CONDITIONAL
       |
       | depends on lab Xeon CHOLMOD output (JSON not on dev)
       ↓
[C012 lab32 well-posed; 5.9 kPa = κ × tol]  PARTIALLY-REFUTED
       |
       | depends on x_LU cache (was deleted; re-caching now)
       ↓
[5.9 kPa attribution]   STORY UNDER REVIEW
       |
       | also depends on actual κ measurement (Lanczos)
       ↓ HYPOTHESIS
[κ_local ~ 1e6 estimate]   WEAK


[C005 OF iter 7-65]   STRONG (metadata.json)
                |
                ↓
[C002 OF self-residual ≈ 1e-8]   VERIFIED (recomputed tonight)
                |
                | × 110 ms/iter (estimated factor)
                ↓ WEAK
[C006 OF wall 770-7150 ms]  REFUTED-AS-MEASUREMENT
                |
                | combined with [C008 AMGx wall 9-39s on RTX 3050] (VERIFIED-scoped, dev-only)
                ↓ WEAK conclusion
[C010 "OF beats AMGx 5-10×"]  REFUTED-conclusion



[C007 AMGx iter 435-773]   STRONG (npz meta)
                |
                | speculation: AMGx config not tuned for this matrix
                ↓ MEDIUM
[C015 amortized cost setup 2-3s, update 50-100ms, solve 100-150ms]  HYPOTHESIS
                |
                | (predicting from old senior 21-bundle data, different matrix)
                ↓ WEAK
[C009 amortized + warm-start beats LU 100-1000×]  HYPOTHESIS, predicts 20-50× under measurement
                |
                | depends also on:
                ↓
[C016 warm-start saves 50-90% iter] HYPOTHESIS (zero data on this dataset)
                |
                ↓
[C017 CHOLMOD symbolic-reuse 2-5×]   HYPOTHESIS



[C011 CHOLMOD wall 69-75s]   VERIFIED-CONDITIONAL (conversation paste, JSON not on dev)
                |
                ↓ supporting
[C004]



[C018 AMGx wins at >5M] REFUTED-as-validated (no data, beyond resource range)
[C013 Phase 0 PID 466877 running] REFUTED (process dead, no output)
[C020 /tmp/bench_gold.log] UNKNOWN provenance
```

## Edge strength summary

| Edge | Strength | Notes |
|---|---|---|
| precision_results_ir1.json → C001 (8K AMGx vs LU = 2.89e-15) | STRONG | Direct re-verification by reading JSON |
| C001 → "AMGx algorithm correct on small case" | MEDIUM | Reasonable inference |
| "AMGx correct on small case" → C004 (AMGx correct on 500K) | WEAK | Generalization across mesh sizes is non-trivial; needs E07 |
| metadata.json → C002 (OF self-residual) | STRONG | Direct read + tonight's recompute |
| C002 + "5s/step assumption" + "45 iter/step assumption" + 110 ms/iter → C006 (OF wall) | WEAK | Layered estimation |
| C006 + C008 → C010 (OF beats AMGx 5-10×) | REFUTED | Cannot stand on REFUTED C006 |
| npz meta → C003 (algebraic res 1e-15) | STRONG | Direct read |
| Conversation paste → C004, C011 | MEDIUM | Real but JSON not stored locally; reproducibility weak |
| Theory + experience → C009, C015, C016, C017 | WEAK | Pure prediction, zero current measurement |
| matrixDumper.H source code → C014 code-correct | STRONG | Read lines 395-432 directly |
| C014 → "senior bug attributed to dumper bug" | UNRESOLVED | Confounded by case rays=0 |

## Most fragile sub-graph (highest priority for E-experiment validation)

```
[C002 OF self-resid 1e-8]  STRONG
       |
       | × 110 ms/iter (WEAK assumption)
       ↓
[C006 OF wall 770-7150 ms] REFUTED → must measure (E01)
       |
       | merged with C008 (RTX 3050, dev-only, scoped)
       ↓
[C010 OF vs AMGx ratio]  REFUTED conclusion → E01 + E02 + E03 to settle
       |
       | merged with HYPOTHETICAL C009 (amortized 100-1000×)
       ↓
[OVERALL "AMGx amortized beats LU" thesis]  HYPOTHESIS → E03+E04+E05+E06
```

This is the central thesis the project hangs on, and EVERY edge into it is currently WEAK. The Xeon validation plan must address this end-to-end.

## Isolated claims (no upstream evidence, frequently cited)

- **κ_local ~ 1e6** for lab32 broken matrix: cited in data_inventory.md as if measured, but actually inferred. Diag-spread proxy gives 7.1e+11 (very different). E09 (Lanczos) needed.
- **"AMGx wins at >5M cells / multi-GPU"**: cited in PROJECT_STATUS_REPORT 8.1 as background, but no data anywhere in this project. Pure literature claim.

## Floating claims (no clear DAG anchor)

- C019 senior melt/evap unsolvable — has data (lab_PCG_truth_results.json), but cause attribution disconnected from C014.
- C020 /tmp/bench_gold.log — unknown provenance, not in any DAG.

## Verdict on overall evidence quality

- 8K-cell AMGx-vs-LU validation: STRONG, locked.
- 500K-cell algebraic residuals: STRONG.
- 500K-cell OF iter counts: STRONG.
- 500K-cell wall times: WEAK (OF estimated, AMGx scoped).
- 500K-cell AMGx-vs-LU: HYPOTHESIS-CONDITIONAL, needs reproduction.
- Production-mode (amortized, warm-start) wall: ZERO data; pure HYPOTHESIS.
- Memory scaling, OOM threshold: ZERO data.

⇒ The Xeon validation plan must convert all WEAK edges into STRONG within tonight-+-tomorrow's budget.
