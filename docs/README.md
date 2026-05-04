# DILU-Research Documentation Index

This directory contains all written knowledge for the DILU-Research project:
GPU sparse linear solver work (cuSPARSE DILU, multicolor DILU, AMGx) for
additive manufacturing CFD, with OpenFOAM cross-validation.

## Top-level meta documents

| File | Purpose |
|------|---------|
| `README.md` | This file — navigation index |
| `PROJECT_SUMMARY.md` | Master detailed reference: deterministic outputs, environment-sensitive ranges, critical implementation landmines |
| `PORTABILITY.md` | Cross-environment compatibility guide: tested version windows, breakage signal decoder, migration effort estimates |
| `AI_REPRODUCTION_PROMPT.md` | Self-contained prompt for blind AI reproduction of the project |

## Subdirectories (Module E taxonomy)

```
docs/
├── reference/      ← External read-only material (papers, third-party guides)
├── design/         ← Design intent, math derivations (historical, may be outdated)
├── specs/          ← Module A-D rigorous specs (authoritative for "what code does now")
├── benchmark/      ← Performance data + comparison experiments
└── session_logs/   ← Timestamped development handoff snapshots
```

### `reference/` — External material

| File | Source |
|------|--------|
| `Barkhudarov_Lagrangian_VOF.md` | Barkhudarov (2004) Lagrangian VOF paper, full text |
| `Matrix_Output_Guide_external.md` | Mei Yang's PCG-internal-hook method for OpenFOAM matrix extraction |

External truth, immutable. See `reference/README.md`.

### `design/` — Design intent

Phase 1-4 architecture design, math foundations, OpenFOAM cross-check plan.
Historical intent (pre-implementation); may be superseded by `specs/`.
See `design/README.md`.

### `specs/` — Code-level specifications

Module A-D rigorous specs: every formula, every index, every tolerance.

| File | Subject |
|------|---------|
| `LAGRANGIAN_VOF_3D_SPEC.md` | Lagrangian VOF 3-D advection (predecessor work) |
| `EULERIAN_PLIC_SPEC.md` | Eulerian PLIC reconstruction (predecessor work) |
| `CONSERVATIVE_REDISTRIBUTE_SPEC.md` | Conservative redistribute kernel (predecessor work) |
| `FLOAT64_NUMERICAL_ANALYSIS.md` | Float32 vs float64 precision analysis |
| `MATRIX_EXTRACTION_SPEC.md` | OpenFOAM `laserMeltFoam` matrix extraction toolchain (current work) |

Authority: when `design/` and `specs/` conflict, **`specs/` wins**.
See `specs/README.md`.

### `benchmark/` — Quantitative experiments

Predecessor VOF/PLIC benchmarks, plus current DILU/AMGx work:

| File | Subject |
|------|---------|
| `OPENFOAM_CROSSCHECK_20260427_v3.2.md` | **CANONICAL**: 150-matrix cross-check (50 pd + 100 T) — pd 8.6× faster than OpenFOAM |
| `OPENFOAM_CROSSCHECK_20260427_v3.1.md` | SUPERSEDED — pd + T BiCGStab w/o cuSPARSE/Multicolor T |
| `OPENFOAM_CROSSCHECK_20260427_v3.md` | SUPERSEDED — first full v3 with std::chrono OpenFOAM timing |
| `OPENFOAM_CROSSCHECK_20260427.md` | SUPERSEDED — v2.1 with vs-OF precision metric |
| `OPENFOAM_CROSSCHECK_20260426.md` | HISTORICAL — v1.0 single-matrix pipeline validation |
| `CANONICAL_CASE.md` | Canonical 128³ stiff Poisson reference numbers (cuSPARSE/AMGx) |
| `BLIND_REPRODUCTION_20260422.md` | Blind reproduction test report |
| `phase{1,2,3,4}_*_report.md` | Per-phase development benchmark reports |
| `phase2.5_physical_report.md` | Phase 2.5 physical hallucination tests |
| `phase3_scaling_64_128.md` | Phase 3 scaling at 64³ and 128³ |
| `canonical_cpu_dilu.md` | CPU baseline DILU (scipy reference) |
| `PERFORMANCE_ANALYSIS.md` | Predecessor PLIC performance breakdown |
| `PLIC_*.md`, `ZALESAK_3D_*.md` | Predecessor VOF/PLIC benchmarks |
| `LAGRANGIAN_REPRODUCTION_TEST_DATA.md`, `REPRODUCTION_TEST_DATA.md` | Predecessor test fixtures |

See `benchmark/README.md` for criteria of when to update.

### `session_logs/` — Development handoffs

| File | Topic |
|------|-------|
| `SESSION_HANDOFF_20260415.md` | PLIC parallel research status |
| `SESSION_HANDOFF_20260416.md` | Conservative redistribute + float64 + Zalesak 3D |
| `SESSION_HANDOFF_20260421.md` | DILU Phase 4 AMGx integration |

Historical snapshots. May reference numbers/decisions that no longer apply.
See `session_logs/README.md`.

## Authority hierarchy (E.5)

```
reference/  >  specs/  >  design/  >  session_logs/
```

When two sources disagree:
- Paper originals in `reference/` are immutable external truth
- `specs/` describes the current code (updated with code changes)
- `design/` describes pre-implementation intent (may be outdated)
- `session_logs/` are point-in-time snapshots (not maintained)

## See also

- Source code: `src/vof/` (predecessor reference) and `dilu/` (current DILU/AMGx work)
- Reproduction prompt: `AI_REPRODUCTION_PROMPT.md`
- Project meta: `PROJECT_SUMMARY.md`
