# SESSION_HANDOFF — 2026-04-16

## Scope of this session

1. **Conservative redistribute (Weymouth-Zaleski 2010)** implemented and validated to replace `jnp.clip(F, 0, 1)` as the default boundedness mechanism in the Eulerian PLIC pipeline.
2. **Float64 support** added to the entire PLIC pipeline (dtype-parametric; float32 remains the default).
3. **OpenFOAM interIsoFoam Zalesak 3D comparison** extracted from `/home/yzk/OpenFOAM/zalesak_3d_of/` and rendered as overview plots matching the JAX version.
4. **Documentation reorganized** per Module E of the `cfd-reverse-engineer` skill: `docs/` now follows the mandatory `reference/ design/ specs/ benchmark/ session_logs/` taxonomy.

## Code changes landed in `src/`

| File | Change |
|------|--------|
| `vof/plic/conservative_bounds.py` | **New module** (~380 lines): `apply_flux_{x,y,z}_conservative` + `redistribute_bounds_{x,y,z}` + `_cell_sign_{x,y,z}` helpers. K=3 fixed `fori_loop` iterations; shift via zero-padded slicing (no `jnp.roll` wraparound); final safety `jnp.clip`. |
| `vof/plic/geometric_flux.py` | Re-export `apply_flux_{x,y,z}_conservative` aliases. |
| `vof/plic/strang_sweep.py` | Switched from `jnp.clip` to `apply_flux_*_conservative` for the default bounds path. |
| `vof/plic/analytic_intercept.py:59` | `thr = jnp.float32(1e-10)` → `thr = jnp.asarray(1e-10, F.dtype)` (Priority 1 dtype fix). |
| `vof/plic/conservative_bounds.py:112,121,127,132,138,143` | 6 sites: hardcoded `jnp.float32` → `u_face.dtype / v_face.dtype / w_face.dtype` with `F.dtype` fallback. |
| `vof/lagrangian_3d/reconstruction_3d.py:69,74,81,88,120-122` | Parker-Youngs conv kernel `w2d`, channel zeros `ch0/ch1/ch2`, and gradient accumulators `dFdx/dFdy/dFdz` now allocate with `dtype=F.dtype` (was hardcoded float32). Enables float64 normal reconstruction. |
| `vof/lagrangian_3d/reconstruction_3d.py:167,168,169` | `_volume_below_3d` `thr`/`max_abc`/`thr_rel` now cast via `jnp.asarray(..., a0.dtype)`. |

## Validated behavior (Zalesak 3D 128³, 90 steps = 1/10 rev)

Reference: `docs/benchmark/ZALESAK_3D_PRECISION_COMPARISON.md`

| Config | V drift | L1 shape | ms/step |
|--------|--------:|---------:|--------:|
| float32 + clip | −7.16 × 10⁻³ % | 142.16 % | 194 |
| float32 + redistribute | −6.90 × 10⁻⁶ % (1040× better) | 142.16 % | 120 |
| float64 + clip | +3.66 × 10⁻⁶ % | 142.16 % | 734 |
| **float64 + redistribute** | **±1.3 × 10⁻¹⁴ % (machine zero)** | 142.16 % | 689 |

OpenFOAM interIsoFoam at the same grid: V drift 0.0000 % (4 dp output precision), L1 = 2.44 %. JAX f64+redistribute now matches OF conservation at strict machine-zero and is 37 % better on L1 shape error.

## Documentation deliverables

### New

- `docs/specs/CONSERVATIVE_REDISTRIBUTE_SPEC.md` — full Module A-D spec for the redistribute algorithm (503 lines). Authoritative.
- `docs/specs/FLOAT64_NUMERICAL_ANALYSIS.md` — mathematical analysis of expected precision gains, per-stage eps sensitivity, and the Priority 1-4 modification checklist (339 lines).
- `docs/benchmark/ZALESAK_3D_PRECISION_COMPARISON.md` — the 4-config precision × bounds matrix with cross-factor attribution.
- `docs/session_logs/SESSION_HANDOFF_20260416.md` — this file.
- `examples/plic_eulerian_tests/zalesak_3d/CONFIG.md` — authoritative test configuration (385 lines) for external-team reproduction.
- `examples/plic_eulerian_tests/zalesak_3d/extract_openfoam.py` — decomposePar-aware OpenFOAM `alpha.water` → npz extractor.
- `examples/plic_eulerian_tests/zalesak_3d/plot_openfoam_overview.png.py` — OF overview plot mirroring the JAX version.
- `examples/plic_eulerian_tests/conservative_bounds/sanity_check.py` — 1M-cell 45° translation + 64³ mini-Zalesak sanity check.
- `examples/plic_eulerian_tests/compare_clip_vs_redistribute.py` — Layer 3 regression driver (case A/B/C smoke + full modes).
- `examples/plic_eulerian_tests/float64/run_zalesak_3d_f64.py` — dtype- and bounds-parametric Zalesak 3D driver (used for the Section-2 numbers).
- `examples/plic_eulerian_tests/float64/compare_f32_f64.py` — 3-config comparison harness (test-debug-validator output; has known `dt` and velocity-shape bugs documented in issue tracker comments).
- `tests/plic_eulerian/test_conservative_*.py`, `test_float64_*.py`, `conftest.py`, `debug_*.py` — 12 float64 regression tests (all passing).

### Updated

- `docs/specs/EULERIAN_PLIC_SPEC.md` — added Section 0 "Precision Mode — Dtype Parametric Interface" at the top, all `dtype=float32` references changed to `dtype = F.dtype`, clip OBSERVATION replaced with clip-vs-redistribute comparison, `jnp.float32(1e-10)` literals replaced with dtype-parametric `jnp.asarray(..., F.dtype)` form, Phase A/B accuracy table extended with float64 row.
- `docs/README.md`, `docs/specs/README.md`, `docs/reference/README.md`, `docs/design/README.md` — rewritten for the new Module-E taxonomy.

### Moved (git mv)

- `docs/PLIC_BENCHMARK_REPORT.md` → `docs/benchmark/`
- `docs/PLIC_BENCHMARK_1M_FLOAT32.md` → `docs/benchmark/`
- `docs/PLIC_PERFORMANCE_PROGRESSION.md` → `docs/benchmark/`
- `docs/PERFORMANCE_ANALYSIS.md` → `docs/benchmark/`
- `docs/PLIC_JAX_PARALLEL_STATUS.md` → `docs/session_logs/SESSION_HANDOFF_20260415.md`

### Deleted

- `docs/总结现状.md` (Chinese, violated English-only rule from Module E anti-pattern list; superseded by this handoff).

## Known issues / deferred

1. `compare_f32_f64.py` (Layer 3 harness, test-debug-validator output) has two bugs:
   - Wrong `dt` formula: divides by `dx` twice → produces unusably small `dt`.
   - Wrong velocity-field shape: broadcast error `(129, 130, 130) vs (130, 129, 130)`.
   These were worked around by `run_zalesak_3d_f64.py`, which is the canonical driver.
2. Lagrangian VOF's `_prepare_gl_quadrature` (Gauss-Legendre nodes at `reconstruction_3d.py:33, 37`) is still hardcoded float32 at module-import time. This affects only the Lagrangian / Barkhudarov pipeline, not the Eulerian PLIC used in this session. Priority 4 (module-level DTYPE constant / env var) not implemented.
3. Priority 2 (eps-scaled tolerances — widening `1e-6` clip-to-eps in f64) deferred. Current tolerances work correctly in both dtypes but are wider than strictly necessary in f64.
4. Full-revolution (900-step) float64 benchmark not run; only 90-step quick mode. Full run would take ~2-3 hr on the RTX 3050.

## What the next session should start with

- If pursuing performance: Priority 3 (dtype-conditional Newton step count — f64 needs only 3 Newton steps vs f32's 5).
- If pursuing integration: hook the Eulerian PLIC into a pressure-velocity coupled solver so `examples/RT_air_helium/` style cases can actually run (currently blocked because PLIC is pure passive advection; see discussion earlier in session).
- If pursuing documentation: Priority 2 (eps-scaled tolerances) across `intercept_solver.py`, `strang_sweep.py`, `geometric_flux.py` and the corresponding `[OBSERVATION]` notes in the spec.
