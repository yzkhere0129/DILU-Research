# Solver Production Comparison — Replay Experiment Report

**Status**: 🟡 In progress — Phase 0 pilot replay running, Phase 1 dense replay queued.

**Hypothesis** (from `solver_comparison_500K.md` Key Takeaway 4):
> "在真实 CFD 时间推进里 AMGx amortized + warm-start 应该比 LU 直解快 100-1000×"

**Method**: Replay dumped matrix sequence through 5 solver modes; compare total wall.

---

## Phase 0 — Pilot replay on 6 timesteps (single_track_dump)

*Will be filled in once `dilu/experiments/solver_production_comparison/replay_results.json` is produced.*

5 modes × 6 timesteps × pd_corr0:
- `amgx_fresh`        — new Plan() per step
- `amgx_amortized`    — Plan once, update_coefficients() per step
- `amgx_amortized_warm` — same + previous x as init guess
- `lu_fresh_CHOLMOD`  — full factor per step
- `lu_symbolic_reuse` — analyze() once, cholesky_inplace per step

### Pilot results (placeholder)

```
mode                       total wall (s)   mean/step (ms)   first step    rest avg   speedup
amgx_fresh                 ?                ?                ?             ?          ?
amgx_amortized             ?                ?                ?             ?          ?
amgx_amortized_warm        ?                ?                ?             ?          ?
lu_fresh_CHOLMOD           ?                ?                ?             ?          ?
lu_symbolic_reuse          ?                ?                ?             ?          ?
```

### Acceptance check
- [ ] All modes max rel_resid < 1e-7
- [ ] amortized 1st step ≈ fresh setup (确认 first 是 full setup)
- [ ] amortized 2..N step < 1/3 first
- [ ] symbolic 2..N step < CHOLMOD fresh

---

## Phase 1 — 384-timestep dense replay (planned)

Once `dense_track_dump_500K` finishes on lab Xeon (~7h), 384 timesteps gives:
- Statistical confidence on amortized speedup
- Quasi-production wall comparison
- Crossover analysis (when amortized AMGx beats fresh LU)

---

## Phase 2 — Synthetic 3D Poisson scaling

`dilu/experiments/solver_production_comparison/synthetic_scaling.py` runs:
- Cube sizes: 32, 50, 64, 80, 100 (32K, 125K, 256K, 500K, 1M cells)
- AMGx fresh, CHOLMOD direct, scipy CG (no precond)
- Records peak RSS for OOM threshold

### Goal
- Find LU OOM threshold on dev (9.7 GB) and lab Xeon (~64 GB)
- Estimate AMGx vs LU crossover mesh size for typical Laplacian

---

## Phase 3 — OF actual per-solve wall (planned)

Re-run single_track_dump with `solverInfo` function object to get OF DICPCG
per-solve wall directly (vs current 110 ms/iter estimate).

---

## Conclusions (TBD)

Will be written when Phase 0+1+2 complete. Expected key findings:
1. Quantify AMGx amortized speedup (predicted 5-50× over fresh)
2. Quantify warm-start speedup (predicted 1.5-3×)
3. Identify LU OOM threshold mesh size
4. Compute realistic AMGx-vs-LU crossover

---

## File index

```
dilu/experiments/solver_production_comparison/
├── EXPERIMENT_DESIGN.md           ← detailed protocol
├── replay_amortized.py            ← 5-mode replay implementation
├── synthetic_scaling.py           ← 3D Poisson scaling test
├── plot_replay_results.py         ← visualization
├── replay_results.json            ← Phase 0 + Phase 1 data
└── synthetic_scaling.json         ← Phase 2 data

docs/benchmark/figures/
├── replay_per_step.png            ← per-timestep wall lineplot
├── replay_cumulative.png          ← cumulative wall
├── replay_summary_bar.png         ← total wall bar chart
└── synthetic_scaling.png          ← mesh-vs-wall scaling
```
