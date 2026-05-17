# Changelog

## 1.0.0 — 2026-05-17

Initial public release. Package extracted from the DILU-Research monorepo
and re-shaped for standalone distribution.

### Fixes (pre-merge audit, see CODE_REVIEW.md)

- **C1** Use-after-stack-scope in `cudaMemcpyAsync` of output scalars in all
  four FFI handlers. Added explicit `cudaStreamSynchronize` after each H→D
  scalar copy. Worked by accident before (pageable host source forced
  synchronous fall-back); now correct under CUDA graphs and pinned host
  memory.
- **C2** `amgx_release` return-code contract inversion. The C++ handler
  returned `1` for success and `0` for "unknown token", inverse of the
  Python docstring. Now returns `0` for both success and idempotent no-op.
- **C3** `refinement.py` no longer mutates `jax_enable_x64=True` at import
  time. Callers must opt in via the new `enable_x64()` helper or set
  `jax_enable_x64` themselves before any JAX import.
- **C4** `build.sh` and `CMakeLists.txt` no longer hardcode `sm_86`. Pass
  `DILU_AMGX_CUDA_ARCH=89` (Ada) / `90` (Hopper) / `120` (Blackwell) or a
  semicolon-separated list for a fat build.

### Test suite

- Added self-contained fixture-based precision tests
  (`tests/fixtures/{pd,T}_tiny.npz` + 4 `test_fixture_*` cases). No external
  data needed; CI gets real precision-vs-truth coverage instead of silent
  skips.
- LaserbeamFoam-backed precision tests now env-gated via
  `DILU_AMGX_LMF_ROOT`. Without it they cleanly skip; fixture tests still
  run.
- Cross-phase imports (`dilu.cusparse`, `dilu.benchmark`) gated via
  `pytest.importorskip`. Standalone install of `dilu_amgx` no longer trips
  collection errors; those tests skip with a clear reason.

### Documentation

- `REPRODUCE.md` — bit-exact reproduction spec for fresh-context engineers.
- `KNOWN_GOTCHAS.md` — 15 traps observed during 6 months of development.
- `INSTALL.md`, `USAGE.md`, `API.md`, `README.md`, `LICENSE`,
  `CHANGELOG.md`, `pyproject.toml` — packaging metadata.

### Benchmarks

- `bench/suite/` — cross-hardware 50-matrix benchmark suite. Portable
  runner produces summary.json + per-step trajectories + precision diffs on
  any CUDA + JAX + AMGx host (dev RTX 3050 verified; lab RTX 5060 +
  future H100 ready).

### Known limitations

- C++ shared library is built out-of-band via `./build.sh`; not yet wired
  through `scikit-build-core` for automatic wheel builds.
- Single-GPU only (device 0). Multi-GPU support requires AMGx-MPI build
  and is not in scope for v1.0.
