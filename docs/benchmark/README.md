# benchmark/ — Performance, Profiling, Comparison Experiments

Quantitative performance and accuracy data for the PLIC / VOF pipeline. Updated
whenever any performance-affecting change lands in `src/` **and** the measured
numbers move by more than 10 %.

## Files

| File | Description |
|------|-------------|
| `PLIC_PERFORMANCE_PROGRESSION.md` | Stage-by-stage speedup journey: Lagrangian baseline (1622 ms/step) → no-JIT Eulerian (22 668 ms) → +JIT (378 ms) → +padded gather (74 ms) → +Phase B analytic (37 ms). 46× vs Lagrangian, 619× vs naive. |
| `PLIC_BENCHMARK_REPORT.md` | Detailed 1M-cell per-stage timing breakdown with all 4 variant comparison. |
| `PLIC_BENCHMARK_1M_FLOAT32.md` | 1M-cell float32 baseline (Mcells/s throughput, per-stage ms). |
| `PERFORMANCE_ANALYSIS.md` | Historical Lagrangian VOF bottleneck analysis (hex-box Sutherland-Hodgman overlay). |
| `ZALESAK_3D_PRECISION_COMPARISON.md` | float32 vs float64 × clip vs redistribute: V drift down to machine zero (±10⁻¹⁴ %), L1 unchanged across precision. |

## When to use

- **Reviewing performance regressions**: compare latest run to the number in the matching file.
- **Choosing dtype / bounds mode**: consult `ZALESAK_3D_PRECISION_COMPARISON.md` for the 4-config matrix.
- **Justifying architectural choices** (Eulerian vs Lagrangian, clip vs redistribute, f32 vs f64): cite the relevant doc instead of re-running experiments.

Benchmarks are historical snapshots, not specifications. When numbers disagree
with `specs/*.md`, either the code behavior changed (update spec) or a
regression landed (fix code). Never edit benchmark files to match new numbers
without also rerunning the underlying experiment.
