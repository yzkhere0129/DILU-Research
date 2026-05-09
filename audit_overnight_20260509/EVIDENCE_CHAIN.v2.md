LAST_REVIEWED: 2026-05-10T04:00+08:00
ITERATION: v2

# Evidence DAG v2 — Post-H2-measurements + Pass-6 adversary

Changes from v1:
- C012 / lab32 5.9 kPa story: now measured & reproducible (cache restored, κ measured at 3.2e+14).
- C002 upgraded VERIFIED.
- New C021 (matrix near-singular) entered DAG.
- A005, A009 robustness fixes integrated.

## Updated DAG (compact)

```
                                   [C001 8K AMGx vs LU 2.89e-15]   STRONG (precision_results_ir1.json)
                                              |
                                              | (foundation for "AMGx algo correct on small case")
                                              ↓ MEDIUM (generalization across N is non-trivial)
                                  [C004 AMGx+IR vs LU 1.13e-11 (500K)] HYPOTHESIS-CONDITIONAL → E07
                                              |
                                              | depends on lab Xeon CHOLMOD JSON not on dev
                                              ↓
                                              ※ E07 will lock this in JSON

[C003 algebraic res 1e-15]    STRONG (npz meta)
       |
       ↓ supports
[C004]


[C002 OF self-resid ~1e-8]  STRONG (now: H2 recompute confirms)
       |
       | × 110 ms/iter (assumed factor with 30% uncertainty)
       ↓ WEAK
[C006 OF wall 770-7150 ms]  REFUTED-AS-MEASUREMENT → E01
       |
       | merged with [C008 AMGx wall 9-39s on RTX 3050] (VERIFIED-scoped)
       ↓ WEAK conclusion
[C010 "OF beats AMGx 5-10×"]  REFUTED-conclusion → settle via E01 + E02


[C007 AMGx iter 435-773]   STRONG (npz meta)
                |
                | speculation: AMGx config not tuned
                ↓ MEDIUM
[C015 amortized cost: setup 2-3s, update 50-100ms, solve 100-150ms]  HYPOTHESIS → E03
                |
                ↓ HYPOTHESIS
[C009 amortized + warm-start beats LU 100-1000×]  HYPOTHESIS, predicts 1.5-3× under measurement
                |
                | depends also on:
                ↓
[C016 warm-start saves 50-90% iter] HYPOTHESIS → E04


[C011 CHOLMOD wall 69-75s]   VERIFIED-CONDITIONAL → E05 to lock JSON
                |
                ↓ supporting
[C004 + C012]


[C012 lab32 5.9 kPa = κ × tol] REFINED-VERIFIED:
                |    - cache restored (H2)
                |    - κ measured at 3.2e+14 via Lanczos (H2)
                |    - σ_min = 4.5e-29 ⇒ matrix near-singular (new C021)
                |    - Predicted bound κ × tol = 3.2e6 (very loose); observed 4.7e-3 (much tighter)
                |    - Mechanism still κ × residual but EFFECTIVE κ in well-conditioned subspace 
                |       is much smaller than σ_max/σ_min.
                ↓
[C021 lab32 σ_min ≈ 4.5e-29 — near-singular]   VERIFIED


[C014 matrixDumper bug fixed]   PARTIALLY-VERIFIED
                |    - code-correct (read source) STRONG
                |    - tested on lab32 only, which had separate physics bug → confounded
                ↓
[C019 senior melt/evap unsolvable] VERIFIED-data-fact
                                    | attribution: matrixDumper bug × case config UNRESOLVED


[C018 AMGx wins at >5M] REFUTED-as-validated (no data, beyond resource range) — out of scope
[C013 Phase 0 PID 466877 running] REFUTED — no output, dead PID
[C020 /tmp/bench_gold.log] UNKNOWN provenance — not in any DAG
```

## Edge updates

| Edge | v1 strength | v2 strength | Change |
|---|---|---|---|
| metadata.json → C002 | STRONG | STRONG | unchanged; now also recompute-confirmed |
| LU re-cache → C012 + 5.9 kPa | (was deleted) | STRONG | restored tonight |
| Lanczos → κ value | (asserted 1e6) | MEASURED 3.2e+14 | replaces baseless assertion |
| precision_results_ir1.json → C001 | STRONG | STRONG | unchanged |
| C001 → C004 | WEAK | WEAK (E07 will upgrade) | unchanged |
| Conversation paste → C004 | MEDIUM | MEDIUM (E07 will upgrade) | unchanged |
| Theory → C009/C015/C016/C017 | WEAK | WEAK | unchanged |

## Most fragile sub-graph (v2 status)

```
[C002 OF self-resid 1e-8]  STRONG ← unchanged
       |
       | × 110 ms/iter (WEAK assumption — A007 in adversary)
       ↓
[C006 OF wall] REFUTED-AS-MEASUREMENT — E01 fixes
       |
       ↓
[C010 OF vs AMGx ratio] REFUTED conclusion — E01 + E02 + E03 settle
       |
       ↓
[C009 thesis] HYPOTHESIS — E03+E04+E05+E06 settle
```

This central thesis chain is **fragile** in v1, **NEEDS XEON** in v2 (no upgrade possible tonight).

## Newly measured edges (this session)

```
H2 SuperLU rerun → x_LU cache restored → C012 5.9 kPa story now MEASURED-reproducibly
H2 Lanczos → σ_max=1.45e-14, σ_min=4.5e-29 → C021 new claim VERIFIED
```

These are the ONLY upgrade-deltas this audit produced; everything else awaits Xeon.

## Confidence per claim post-v2

| Claim | v1 confidence | v2 confidence | Path to upgrade |
|---|---|---|---|
| C001 | 95% (locked) | 95% | n/a |
| C002 | 80% | **95%** | recompute confirmed |
| C003 | 95% | 95% | n/a |
| C004 | 50% | 50% | E07 → 95% |
| C005 | 95% | 95% | n/a |
| C006 | 25% (REFUTED) | 25% | E01 → 95% (or revisit estimate) |
| C007 | 90% | 90% | n/a (interpretation tweak via E08 still possible) |
| C008 | 80% | 80% | n/a (scoped) |
| C009 | 30% (HYPOTHESIS) | 30% | E03+E04 → 95% one way or the other |
| C010 | 25% | 25% | E01+E02+E03 → 95% |
| C011 | 75% | 75% | E05 → 95% |
| C012 | 50% | **75%** | cache + κ measured; structure clear |
| C013 | 5% (REFUTED hard) | 5% | n/a |
| C014 | 60% | 60% | E10 optional → 95% |
| C015 | 30% | 30% | E03 → 95% |
| C016 | 30% | 30% | E04 → 95% |
| C017 | 30% | 30% | E06 → 95% |
| C018 | 20% (REFUTED) | 20% | not testable with our hardware |
| C019 | 75% | 75% | requires senior cooperation |
| C020 | n/a | n/a | provenance question |
| C021 (new) | n/a | **80%** | Lanczos seed check (running) → 95% |

Mean v2 confidence: ~58%. After Xeon: predicted ~85% (most claims locked or honestly refuted).
