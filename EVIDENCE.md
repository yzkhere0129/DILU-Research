# Evidence Index

For reviewers who want to verify any claim made in `README.md` or
`HIGHLIGHTS.md`. Each row maps a claim → the exact file, figure, or
JSON key that supports it. All paths are repo-relative.

Conventions: numbers in `summary.json` are the ground truth; figures are
rendered from those JSONs by `dilu/amgx/bench/suite/plot_results.py`.

---

## 1. Core precision claims

| Claim | Source | Field to inspect |
|---|---|---|
| AMGx + 1-step IR `rel ≤ 2.14 × 10⁻¹⁵` median on 50 LPBF pd matrices | `results_dev3050_2026-05-16/summary.json` | `pd.fresh_e12_IR.rel_resid_median` |
| Same, max over 50 matrices: `2.14 × 10⁻¹⁵` | same | `pd.fresh_e12_IR.rel_resid_max` |
| Warm-start preserves precision (`2.14 × 10⁻¹⁵`) under amortized lifecycle | same | `pd.amortized_e12_IR.rel_resid_max` |
| T equation precision (BiCGStab + AMG + IR) | same | `T.fresh_e12_IR.*` |
| AMGx + IR vs CHOLMOD direct LU on 6 × 500K pd, **max rel 1.13 × 10⁻¹¹** | `docs/benchmark/AMGX_PRECISION_20260504.md` | "F1" claim row |
| Earlier scipy LU verification (8K cells, AMGx alg correctness) | `dilu/amgx/bench/precision_results_ir1.json` | top-level summary |

## 2. Iteration / wall-time claims

| Claim | Source |
|---|---|
| `pd` iter 12 → 8 (warm-start saves 47 %) | `summary.json` `pd.fresh_e8.iter_median` (12) vs `pd.amortized_e8.iter_median` (8) |
| `pd.fresh_e12_IR.iter_mean = 18`, `amortized_e12_IR.iter_mean = 8.42` (same 47 % savings at tighter tol) | same JSON |
| T-equation 6-solver wall-time chart | `docs/benchmark/figures/T_solver_wall_compare_6.png` |
| AMGx warm + amortized vs CHOLMOD fresh: **158× speedup** (hardware-mismatched: RTX 3050 vs Xeon 1-thread) | `audit_overnight_20260509/FINAL_SUMMARY.v3.md` §1 F11 |
| AMGx setup ~500 ms, not the 2-3 s a prior session estimated (C015 refuted) | `audit_overnight_20260509/FINAL_SUMMARY.v3.md` F8 |

## 3. Cross-hardware portability

| Claim | Source |
|---|---|
| 50 matrices × 4 protocols × 2 equations, 0 failures on RTX 3050 | `results_dev3050_2026-05-16/summary.json` (all `n_failed = 0`) |
| Same suite on RTX 5060 (sm_120) | `audit_overnight_20260509/lab_5060_replay/run_5060_replay.sh` + replay results under `lab_5060_replay/results_*` (gitignored, regenerable) |
| Run metadata (host, JAX version, GPU model, git SHA) | `results_dev3050_2026-05-16/run_meta.json` |
| FFI registered on both `'cuda'` and `'CUDA'` JAX platforms (the sm_120 portability fix) | commit `79291cf` |

## 4. OpenFOAM byte-level agreement claims

| Claim | Source |
|---|---|
| `pd` max diff ≈ 1.1 nPa vs OpenFOAM `DICPCG @ tol=1e-8` (single_track late-evap timestep) | `docs/benchmark/figures/single_track_evap_late_1.06e-06_OF_vs_AMGx_e8_diff_slices.png` (colorbar = abs diff in Pa) |
| Same plot family for 5 other timesteps + 2 melt phases | `docs/benchmark/figures/single_track_*_OF_vs_AMGx_e8_diff_slices.png` (18 total figures, 6 timesteps × 3 solver pairs) |
| `pd` ground-truth slice plots (`x_truth` from CHOLMOD multi-thread) | `docs/benchmark/figures/single_track_evap_late_1.06e-06_pd_field_x_truth_slices.png` |
| OF `DICPCG @ tol=1e-8` vs `x_truth` shows the ~44.6 Pa gap is condition-number × tol noise, *not* a bug in OF | `docs/benchmark/figures/single_track_*_OF_vs_truth_diff_slices.png` |
| T-field max diff vs OpenFOAM `PBiCG` ≈ 0.2 pK | `docs/benchmark/figures/single_track_*_T_*_solver_meltpool.png` |

## 5. Independent reproducibility (the "fresh AI" test)

| Claim | Source |
|---|---|
| `REPRODUCE.md` + `KNOWN_GOTCHAS.md` are sufficient for blind reimplementation | `dilu/amgx/REPRODUCE.md` § "Acceptance" |
| Byte-match proof harness used to grade the rewrite | `dilu/amgx/bench/byte_match_proof.py` |
| Result: SHA-256 of `x` vector matched to the byte | `dilu/amgx/bench/byte_match_proof_results.json` |
| Plan-registry / dispatch test on a sm_120 GPU we'd never touched | `audit_overnight_20260509/lab_5060_replay/test_ffi_register.py` |

## 6. OpenFOAM CPU baseline (byte-exact reimplementation)

| Claim | Source |
|---|---|
| Module purpose + scope statement | `dilu/openfoam_cpu/README.md` |
| Python ref impl: `dilu`, `pcg`, `pbicg`, `normfactor`, `ldu`, `spmv`, `io` | `dilu/openfoam_cpu/python/*.py` (~0.9 k LoC) |
| C++ port for performance bench | `dilu/openfoam_cpu/cpp/{include,src,bindings}/*` (~0.35 k LoC) |
| Byte-exact regression test (C++ kernel == Python kernel == OpenFOAM) | `dilu/openfoam_cpu/tests/test_cpp_byte_exact.py`, `tests/test_pbicg_cpp_byte_exact.py` |
| MPI smoke / perf benches (for future scale-out) | `dilu/openfoam_cpu/tests/{test_mpi_full,bench_mpi_amul_perf,bench_kernel_breakdown}.py` |
| Source audit doc the impl is built from | `docs/reference/openfoam_dilu_source_audit.md` (referenced by README) |

## 7. Audit campaign (the "did the project hold up?" trail)

| Claim | Source |
|---|---|
| 14 F-claims, all verified | `audit_overnight_20260509/FINAL_SUMMARY.v3.md` §2 |
| Confidence calibration over time (40 → 94 / 100) | same, §0 |
| Adversarial review notes | `audit_overnight_20260509/ADVERSARY_NOTES.v2.md` |
| Failure-mode registry | `audit_overnight_20260509/FAILURE_MODES.md` |
| Per-experiment runners (E01–E11, T01–T02) | `audit_overnight_20260509/xeon_validation/*_runner.py` |
| Earlier blind-reproduction report (a sibling effort, predecessor to this campaign) | `docs/benchmark/BLIND_REPRODUCTION_20260422.md` |

## 8. Project narrative / context

| Claim | Source |
|---|---|
| Full chronological status report (Mandarin + English) | `docs/PROJECT_STATUS_REPORT.md` |
| 4 dataset generations and why each was kept / dropped | same, §2.2 |
| Hardware matrix (dev / lab Xeon / lab 5060) | same, §3 |
| Phase reports (Phase 1 MVP through Phase 4 AMGx) | `docs/benchmark/phase{1,2,2.5,3,4}_*_report.md` |
| Cross-solver comparison reports | `docs/benchmark/solver_comparison*.md`, `OPENFOAM_CROSSCHECK_*.md` |

---

## What is NOT here, on purpose

- **The 1.2 GB of raw OpenFOAM matrix dumps** (`*.tgz`) — regenerable from
  the OpenFOAM cases under `single_track_dump.tgz` / `lab32_dump_minimal.tgz`
  on the dev box. Excluded from this branch to keep `git clone` fast.
- **Per-step replay npz outputs** — regenerable in 1-2 minutes via
  `run_benchmark.py`. The `summary.json` carrying the actual numbers is
  kept.
- **`dilu/amgx/build/` and other per-host compiled artifacts** — rebuilt
  via `dilu/amgx/build.sh` on each host (AMGx is built against the local
  CUDA architecture).
- **The `dilu/amgx/` package's actual GitHub PR** to the research partner's
  fork — that lives at `Shengfeng233/JAX-LaserAM` `feat/dilu-amgx-v1`
  (private repo, not accessible to reviewers outside the collaboration).
  Everything in that PR is also in this branch under `dilu/amgx/` — the
  branch is a public-facing equivalent.
- **`src/vof/`, `examples/`, `skills/`** — reference VOF / PLIC code
  inherited from the predecessor `JAX-LaserAM-plic-research` project. Kept
  in-tree as engineering-pattern reference (per `CLAUDE.md`); **not part of
  this work**, not a contribution made during this 6-month research.
