# dilu-amgx

JAX FFI bindings for [NVIDIA AMGx 2.5.0](https://github.com/NVIDIA/AMGX),
purpose-built for the LPBF (Laser Powder Bed Fusion) CFD hot path. Drop-in
sparse linear solver for AM-style pressure (PCG + Classical V-cycle) and
energy (BiCGStab + Classical V-cycle) systems, with warm-start amortization
across timesteps and iterative refinement to machine precision.

## What this is

- 4 FFI primitives (`amgx_setup`, `amgx_update_coefficients`, `amgx_solve`,
  `amgx_release`) that bind AMGx's setup-once / solve-many lifecycle to JAX
  device buffers without host round-trips.
- A `Plan` context manager wrapping the lifecycle and a registry of opaque
  uint64 tokens guarded by a process-global mutex.
- An iterative-refinement helper (`amgx_solve_with_refinement`) that reaches
  `1e-15` relative error vs scipy `spsolve` truth on LPBF pressure /
  temperature dumps.
- Canonical config JSON strings (`CLASSICAL_V_DIAGSCALED`,
  `CLASSICAL_V_DIAGSCALED_BICGSTAB`, etc.) tuned for OpenFOAM-style
  lduMatrix sparsity patterns.

## What this is not

- Not a general-purpose AMG framework (use AMGx directly for that).
- Not multi-GPU or distributed (device 0, single process).
- Not float32 (AMGx FFI is float64-only).

## Documentation map

| File              | Audience                                                |
| ----------------- | ------------------------------------------------------- |
| `INSTALL.md`      | Anyone trying to build the package on a new host        |
| `USAGE.md`        | Quickstart: solve `Ax = b` in 10 lines                  |
| `API.md`          | Reference: every public symbol with signature + meaning |
| `REPRODUCE.md`    | Bit-exact reproduction spec for fresh-context engineers |
| `KNOWN_GOTCHAS.md`| 15 real traps we hit; read before debugging anything    |
| `CODE_REVIEW.md`  | Pre-merge audit (historical; bugs cited there are fixed)|
| `CHANGELOG.md`    | Version history                                         |
| `benchmarks/`     | Cross-hardware 50-matrix benchmark suite (pd + T)       |

## Quickstart

```bash
# 1. Build AMGx 2.5.0 (one-time, ~10 min)
#    See INSTALL.md Step 1.

# 2. Build the FFI shim (one-time per CUDA arch)
DILU_AMGX_CUDA_ARCH=86 ./build.sh    # adjust 86 → 89/90/120 per GPU

# 3. Install the Python package
pip install -e .

# 4. Run the self-contained precision test (no external data needed)
pytest tests/test_precision_vs_truth.py::test_fixture_pd_amgx_ir_meets_user_spec -v
```

## Status

v1.0.0 — production-validated on ~50 LPBF matrices across dev RTX 3050
(sm_86) and lab RTX 5060 (sm_120). Achieves `1e-15` precision vs scipy
truth; warm-start saves ~47% iterations on dense temporal sampling.

## License

MIT (this package). NVIDIA AMGx itself is BSD-3-Clause, loaded as a
separately-installed shared library.
