# specs/ — Reproduction-Grade Technical Specifications

Module A-D format specifications precise enough for an isolated AI to reproduce
the code without seeing the source.

**Authority**: when `design/` and `specs/` conflict, **`specs/` wins** —
`specs/` describes what the code actually does now; `design/` describes what
was once intended.

## Files

| File | Content | Code coverage | Status |
|------|---------|---------------|--------|
| `EULERIAN_PLIC_SPEC.md` | Eulerian PLIC VOF full-module spec, **dtype-parametric** (float32 default, float64 via `JAX_ENABLE_X64=1`) and **dual-bounds-mode** (`clip` or `redistribute`). Section 0 documents both precision modes with measured V drift numbers. | `src/jax_laseram/vof/plic/` — 8 modules | **Current** |
| `CONSERVATIVE_REDISTRIBUTE_SPEC.md` | Weymouth-Zaleski 2010-style overshoot/undershoot redistribute. K=3 fixed `fori_loop` iterations, sweep-direction-aware shift via zero-padded slicing, final safety `jnp.clip`. | `src/jax_laseram/vof/plic/conservative_bounds.py` | **Current** |
| `FLOAT64_NUMERICAL_ANALYSIS.md` | Dtype sensitivity analysis: expected V drift at clip/redistribute × f32/f64 (5 orders + 9 orders respectively), per-epsilon sensitivity audit, RTX 3050 performance forecast (~4× slowdown), Priority 1-4 modification checklist. Predicts L1 shape error is unaffected by dtype (dominated by O(Δx) Youngs truncation). | Cross-module | **Current** |
| `LAGRANGIAN_VOF_3D_SPEC.md` | Lagrangian VOF 3D pipeline (Barkhudarov 2004 hex-box Sutherland-Hodgman overlay). Kept as oracle / comparison baseline. | `src/jax_laseram/vof/lagrangian_3d/` | Legacy (current) |
| `MATRIX_EXTRACTION_SPEC.md` | OpenFOAM `laserMeltFoam` matrix extraction toolchain — solver-level hook + serialization to MatrixMarket + sanity verification + npz transport. Full file inventory, exact patches, step-by-step procedure, validation checklist. | `LaserbeamFoam/applications/solvers/laserMeltFoam/matrixDumper.H` + 3 patches; `dilu/benchmark/openfoam_crosscheck/` Python drivers | **Current** |

## When to use

- **Reproducing a module**: read the corresponding spec first. For the main
  pipeline, start at `EULERIAN_PLIC_SPEC.md` Section 0 (precision mode) then
  walk through Sections 1-8 in dependency order.
- **Adding a new bounds or intercept mode**: extend `EULERIAN_PLIC_SPEC.md`
  Section 5 + write a standalone spec like `CONSERVATIVE_REDISTRIBUTE_SPEC.md`.
- **Comparing Eulerian vs Lagrangian**: read `EULERIAN_PLIC_SPEC.md` and
  `LAGRANGIAN_VOF_3D_SPEC.md` side-by-side.
- **Verifying float64 correctness**: combine `FLOAT64_NUMERICAL_ANALYSIS.md`
  with `../benchmark/ZALESAK_3D_PRECISION_COMPARISON.md`.

## Naming conventions

- Overview specs: `UPPERCASE_SPEC.md` (top-level entry points)
- Algorithm-specific specs: `UPPERCASE_ALGORITHM_SPEC.md`
- Numerical analysis notes: `UPPERCASE_SUBJECT_ANALYSIS.md`
- No spaces in filenames; use underscores.
