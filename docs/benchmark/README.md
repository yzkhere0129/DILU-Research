# benchmark/ — Performance, Profiling, Comparison Experiments

Quantitative performance and accuracy data for the project. Updated whenever
any performance-affecting change lands in `src/` or `dilu/` **and** the
measured numbers move by more than 10 %.

## DILU / AMGx work (current focus)

### OpenFOAM cross-validation (1M-cell LPBF spot melt, 50 pd + 100 T matrices)

| File | Status | Summary |
|------|--------|---------|
| `OPENFOAM_CROSSCHECK_20260427_v3.2.md` | ⭐ **CANONICAL** | 4 GPU paths × 150 matrices. AMGx pd 8.6× / T 2.8× faster than OpenFOAM single-thread |
| `OPENFOAM_CROSSCHECK_20260427_v3.1.md` | superseded | + amortized AMGx + AMGx-T 100 matrices |
| `OPENFOAM_CROSSCHECK_20260427_v3.md` | superseded | + std::chrono OpenFOAM 2.57 s/pd-solve |
| `OPENFOAM_CROSSCHECK_20260427.md` | superseded | + vs-OF precision metric |
| `OPENFOAM_CROSSCHECK_20260426.md` | superseded | initial 1-matrix pipeline validation |

When citing OpenFOAM cross-check results, **use v3.2 only**. Older files are
kept for traceability (showing how the analysis evolved); they are not
authoritative.

### Phase progression benchmarks

| File | Subject |
|------|---------|
| `CANONICAL_CASE.md` | Canonical 128³ stiff Poisson reference numbers (cuSPARSE/AMGx) — synthetic test |
| `phase1_mvp_report.md` | Phase 1 FFI prototype performance |
| `phase1_repro_HR54WV2_gtx1080.md` | Phase 1 cross-machine reproduction (HR54WV2 + GTX1080) |
| `phase2_cusparse_report.md` | Phase 2 cuSPARSE level-scheduling DILU report |
| `phase2.5_physical_report.md` | Phase 2.5 physical hallucination tests (real divergence / static droplet / residual halo) |
| `phase3_multicolor_report.md` | Phase 3 multicolor DILU report |
| `phase3_scaling_64_128.md` | Phase 3 scaling at 64³ and 128³ |
| `phase4_amgx_report.md` | Phase 4 AMGx integration |
| `canonical_cpu_dilu.md` | CPU baseline DILU (scipy.sparse.linalg + numpy) |
| `BLIND_REPRODUCTION_20260422.md` | Blind reproduction test (worktree-isolated AI) |

## Predecessor VOF / PLIC work

| File | Subject |
|------|---------|
| `PLIC_PERFORMANCE_PROGRESSION.md` | Stage-by-stage speedup: Lagrangian baseline → Eulerian + JIT + padded gather + Phase B analytic. 46× vs Lagrangian, 619× vs naive. |
| `PLIC_BENCHMARK_REPORT.md` | Detailed 1M-cell per-stage timing breakdown for 4 variant comparison |
| `PLIC_BENCHMARK_1M_FLOAT32.md` | 1M-cell float32 baseline throughput |
| `PERFORMANCE_ANALYSIS.md` | Historical Lagrangian VOF bottleneck analysis |
| `ZALESAK_3D_PRECISION_COMPARISON.md` | float32 vs float64 × clip vs redistribute matrix |
| `LAGRANGIAN_REPRODUCTION_TEST_DATA.md` | Lagrangian VOF reproduction test fixtures |
| `REPRODUCTION_TEST_DATA.md` | General reproduction test fixtures |

## When to use

- **Reviewing DILU/AMGx performance**: cite `OPENFOAM_CROSSCHECK_20260427_v3.2.md`
- **Reviewing PLIC performance**: cite `PLIC_PERFORMANCE_PROGRESSION.md` or `PLIC_BENCHMARK_REPORT.md`
- **Choosing dtype / bounds mode**: consult `ZALESAK_3D_PRECISION_COMPARISON.md`
- **Justifying architectural choices** (Eulerian vs Lagrangian, AMGx vs DILU, etc.): cite instead of re-running

## Authority

Benchmarks are **historical snapshots**, not specifications. When numbers
disagree with `specs/*.md`, either the code behavior changed (update spec) or
a regression landed (fix code). Never edit benchmark files to match new
numbers without also rerunning the underlying experiment.
