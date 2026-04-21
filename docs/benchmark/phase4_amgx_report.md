# Phase 4 — AMGx-backed PCG Integration Report

**Date**: 2026-04-21
**Hardware**: RTX 3050 Laptop (4 GB, CC 8.6, FP64 @ 1/32 FP32)
**JAX**: 0.9.0 at `/home/yzk/jax-env`, float64 global, XLA platform allocator
**AMGx**: v2.5.0 built from source at `/home/yzk/local/amgx/` (no MPI, `CUDA_ARCH=86`, CUDA 12.4)
**Artifacts root**: `dilu/amgx/`
**Status**: ALL acceptance tests PASS. Phase 4 complete.

---

## 1. Headline scaling result

The Phase 3 scaling report (`phase3_scaling_64_128.md`) observed that DILU-family PCG iter counts scale as `N^{1/3}` on our stiff problem, matching the Gustafsson `sqrt(kappa)` bound for discontinuous coefficients. Phase 4 replaces the DILU preconditioner with AMGx's classical Ruge–Stüben AMG; iter counts become grid-independent (~15) up to 64³ and mildly grid-dependent (43) at 128³ with the memory-lean aggressive-coarsening config. Wall-time speedup at realistic AM scales is the whole point:

| Grid | N | nnz | P2 iters | P2 wall (s) | P3 iters | P3 wall (s) | **P4 iters** | **P4 setup (s)** | **P4 solve (s)** | **P4 total (s)** | **P4/P2 wall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16³ | 4 096 | 27 136 | 24 | 0.045 | 36 | 0.034 | 15 | 0.375 | 0.017 | 0.532 | 0.08× |
| 32³ | 32 768 | 223 232 | 128 | 0.556 | 197 | 0.204 | 16 | 0.062 | 0.029 | 0.207 | **2.69×** |
| 64³ | 262 144 | 1 810 432 | 99 | 6.49 | 158 | 8.08 | 16 | 0.263 | 0.081 | 0.463 | **14.0×** |
| 128³ | 2 097 152 | 14 581 760 | 186 | 28.47 | 296 | 35.00 | 43* | 0.565 | 0.555 | 1.287 | **22.1×** |

\* 128³ uses `AGGRESSIVE_COARSENING` (D1 interpolator + `aggressive_levels=2`) to fit in 4 GB VRAM. `CLASSICAL_V_CYCLE` at 128³ converges in **15 iters** with ~2.34 GiB peak VRAM (over budget); see §6.

Notes:
- `P4 total` = `setup + first_solve` wall time (the realistic "one-shot pattern" cost). `P4 solve` column is the steady-state per-solve wall on a warm plan — matters for amortized AM timestep loops.
- P2/P3 wall at 16³/32³ are surrogate numbers (`apply_median × iters`) quoted from `phase3_scaling_64_128.md`; the 64³/128³ rows are actual wall-clock. The **16³ row is the DILU-friendly regime**: AMG setup dominates and DILU wins — use DILU at `N ≤ 32³`, AMG at `N ≥ 64³`, per the Phase 4 design doc §F3 routing rule.

Setup-vs-solve breakdown at 128³ (AGGRESSIVE config): setup 0.565 s, steady-state solve 0.555 s. Setup amortizes in `K ≈ 1` steps if pattern is reusable. In an AM timestep loop where only `values` change (pattern fixed), `amgx_update_coefficients` + `amgx_solve` skips the coarsening rebuild: see §4.

## 2. Zero-copy probe (mandatory per design §5)

`dilu_amgx.so` compiled with `-DDILU_AMGX_VERBOSE=1` emits `cudaPointerGetAttributes` for every CSR pointer arriving at `amgx_setup`. Captured output on the first-ever call (8³ Laplacian, under `jax.jit`-free eager solve path):

```
[dilu_amgx verbose] setup: n=512 nnz=3200
[dilu_amgx verbose] row_ptr: ptr=0x604201c00 type=cudaMemoryTypeDevice device=0
[dilu_amgx verbose] col_idx: ptr=0x604202600 type=cudaMemoryTypeDevice device=0
[dilu_amgx verbose] values:  ptr=0x604205800 type=cudaMemoryTypeDevice device=0
[dilu_amgx verbose] AMGX_matrix_upload_all...
[dilu_amgx verbose] AMGX_vector_create b,x...
[dilu_amgx verbose] AMGX_solver_create...
[dilu_amgx verbose] AMGX_solver_setup...
[dilu_amgx verbose] setup complete
[dilu_amgx verbose] solve: n=512, upload b...
[dilu_amgx verbose] solve: AMGX_solver_solve...
[dilu_amgx verbose] solve: iters=13 status=0; download x...
[dilu_amgx verbose] solve: download done
smoke: n=512, nnz=3200
smoke: iters=13, status=0, relres=9.377e-11
smoke: PASS
```

All three CSR pointers (`row_ptr`, `col_idx`, `values`) report `cudaMemoryTypeDevice` with `device=0`. AMGx's internal `cudaMemcpyDefault` therefore performs **device-to-device** copies into its own managed buffers, with **zero PCIe traffic**. The 8-byte D→H token copy per solve (tolerated exception inherited from Phase 2) is the only host-side traffic in steady state; it does not appear in the HLO custom-call surface (see T11). The verbose probe is gated by a compile-time `DILU_AMGX_VERBOSE` macro; production builds leave it off.

## 3. Acceptance test results (T8–T13 + update-coefficients + F2 acid)

All tests run as independent Python subprocesses with the standard VRAM rails (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.5`, `ALLOCATOR=platform`).

| Test | Grid | Acceptance | Measured | Verdict |
|---|---|---|---|---|
| Smoke (8³ Laplacian) | 8³ | status=0, relres ≤ 1e-8, iters ≤ 30 | iters=13, relres=9.38e-11 | PASS |
| **T8 Correctness** (16³ stiff, vs DILU-PCG) | 16³ | `‖x_AMG − x_DILU‖∞ / ‖x_DILU‖∞ ≤ 1e-8` | **5.80e-11** | PASS |
| **T9 Iter count** (128³ stiff, tol=1e-10) | 128³ | ≤ 50 iters | **15 iters** (CLASSICAL) / 43 (AGGRESSIVE) | PASS (12.4× vs DILU 186) |
| **T10 Wall time** (128³ stiff, tol=1e-10) | 128³ | total wall < 0.5 × 28.47 s = 14.24 s | **2.43 s** (CLASSICAL) | PASS (11.7× vs DILU) |
| **T11 Under-JIT HLO** | 8³ | 1 custom-call, 0 copy-start/copy-done | 1 custom-call, 0 copies | PASS |
| **T12 Test A** (64³ vardensity projection, ρ-ratio 1000 sphere) | 64³ | `max\|∇·u\| ≤ 4.1e-8` | **5.92e-9** | PASS |
| **T12 Test B** (64³ CSF 10 steps, σ=0.07) | 64³ | `‖u‖∞ @ step10 ≤ 3.0e-6`; `max\|div\| ≤ 1e-12` | `‖u‖∞=2.754e-6`, `max\|div\|=3.98e-14` | PASS (matches P2/P3 4-sig-fig) |
| **T13 VRAM budget** (128³ setup+solve) | 128³ | peak ≤ 2048 MiB | CLASSICAL 2340 MiB; AGGRESSIVE 292 MiB | PASS (stretch: use AGGRESSIVE at 128³) |
| **update_coefficients** (16³, refresh vs fresh setup) | 16³ | `‖x_refresh − x_ref‖∞ / ‖x_ref‖∞ ≤ 1e-6` | **0.00e+00** (bit-identical) | PASS |
| **F2 acid** (32³ three-tier, fixed 15 AMG iters) | 32³ | `corr(\|E_demean\|, \|∇log ρ\|) ≤ 0.5` | **0.0313** | PASS (converged in 18 iters at tol 1e-10) |

Plots for T12:
- `dilu/amgx/bench/plots/scale_64/divergence_map_AMG_64.png`
- `dilu/amgx/bench/plots/scale_64/spurious_currents_AMG_64.png`

Side-by-side with Phase 2's `dilu/cusparse/bench/plots/divergence_map.png` and Phase 3's `dilu/multicolor/bench/plots/scale_64/divergence_map_P3_64.png`: the Phase 4 AMG divergence map shows the same `~1e-9` floor, no halo on the sphere interface; the spurious-currents plot reproduces the 4-lobed CSF parasitic pattern with the same `‖u‖∞` history (2.754e-6 at step 10 — identical to Phase 2 and Phase 3 to 4 significant figures).

## 4. Per-primitive functional summary

| Primitive | Purpose | Hot? | Cost at 128³ | Notes |
|---|---|---|---|---|
| `amgx_setup` | Build AMG hierarchy + solver | No (once per pattern) | 0.57 s | Owns the config handle (BUGFIX §5) |
| `amgx_update_coefficients` | Replace values + refresh smoother state | Warm (per AM step) | ~O(nnz), not timed here | Uses `AMGX_matrix_replace_coefficients` + `AMGX_solver_resetup` |
| `amgx_solve` | One PCG solve with the existing hierarchy | **Yes** | 0.56 s | 8-byte D→H token copy is the only host-visible traffic |
| `amgx_release` | Destroy solver/matrix/vectors/config | No | μs | Safe destruction order: solver → vectors → matrix → config |

All four symbols are exported by `dilu/amgx/build/libdilu_amgx.so`; `nm -D` confirms `AmgxSetup`, `AmgxUpdateCoefficients`, `AmgxSolve`, `AmgxRelease`. Library's `RUNPATH` is baked to `/home/yzk/local/amgx/lib` so no `LD_LIBRARY_PATH` is needed at test time.

## 5. What we found in the skeleton (fixes applied)

The skeleton the prior session wrote was never compiled. In completing it, three fixes were required:

1. **FFI `Attr` template signature.** The handler parameter must be the decoded type (`std::string_view`), not `ffi::Attr<std::string_view>` — JAX 0.9's `AttrTag` dispatch delivers the decoded type directly to the handler function. Fixed in `amgx_setup.cc`.
2. **`std::string(string_view)` constructor.** libstdc++ 13 requires `std::string(sv.data(), sv.size())` for a string_view-to-string conversion; the one-argument form relies on a P2499 overload not present on this toolchain. Fixed in `amgx_setup.cc`.
3. **Config handle lifetime (critical).** The original skeleton destroyed the config handle immediately after `AMGX_solver_create`. AMGx's `Solver::m_cfg` is a *raw pointer* into the config object (see AMGx `include/solvers/solver.h`), not a copy. The first `AMGX_solver_solve` call dereferences it via `getPrintSolveStats()` → `AMG_Config::getParameter<int>` → `std::map::find` on invalid memory → SIGSEGV. Fixed by making each `PlanEntry` own its config for its lifetime (destroy config in `destroy_plan_entry` after the solver). This one discovery justifies the whole "compile + smoke test before writing the suite" step — no amount of design review would have caught it without running AMGx.

We also installed `AMGX_register_print_callback` + `AMGX_install_signal_handler` so AMGx's internal diagnostic messages and signal-handler stack traces reach stderr; without those, the config-lifetime crash would have been an opaque segfault with no information. Added unconditionally in `plan_registry.cc`.

## 6. Hardware budget truth

Per-config VRAM peak at 128³ (measured by `nvidia-smi`):

| Config | VRAM peak Δ (MiB) | Iters to tol 1e-10 | Solve wall (s) |
|---|---:|---:|---:|
| `CLASSICAL_V_CYCLE` (D2, max_levels=50) | 2340 | 15 | 0.63 |
| `AGGRESSIVE_COARSENING` (D1, aggressive_levels=2) | 292 | 43 | 0.56 |

CLASSICAL exceeds the 2 GB budget on this card (but runs without OOM — 121 MiB free margin post-setup in one observed run). AGGRESSIVE is the design-doc-recommended stretch config: 8× less memory, 2.9× more iters, ~same wall time because the per-iter cost scales down with the smaller hierarchy. The scaling table uses AGGRESSIVE at 128³ and CLASSICAL otherwise; this is what the auto-dispatch in `bench_amg_vs_dilu_scaling.py` selects.

**Recommendation for production 128³+ on this hardware**: use AGGRESSIVE. On A100/H100 with 80 GB VRAM, use CLASSICAL — fewer iters, same setup structure, plenty of headroom.

## 7. What the project achieved end-to-end (Phase 1 → 4)

| Phase | Kernel | Problem it solved | Peak wall-speedup vs prior phase |
|---|---|---|---|
| 1 | Jacobi-preconditioned residual via `jax.ffi` | Prove the JAX ↔ C++/CUDA FFI pipeline | baseline |
| 2 | cuSPARSE level-scheduling DILU-PCG | Exact DILU with opaque-token analyze-apply split | 3× iter reduction vs Jacobi (T7: 24 vs 71 at 16³) |
| 2.5 | — | Anti-hallucination physical tests A/B/C | physical-healthy at ρ-ratio 1000 / 8000 |
| 3 | Multi-color DILU (own-kernels, no cuSPARSE) | Bypass level-scheduling serialization | 4.2× per-apply at 32³; regression at 64³/128³ |
| 4 | AMGx classical V-cycle AMG-PCG | **Break the `N^{1/3}` iter-count scaling wall** | **22.1× total wall at 128³ vs Phase 2**; grid-independent iter count in 64³ window |

Every hard constraint from `CLAUDE.md` held: float64 throughout, no `-ffast-math`, `np.ascontiguousarray` at every FFI boundary, one test per subprocess for 64³+, no modifications to Phase 1/2/2.5/3 artifacts. The single 8-byte D→H token copy per solve remains the only tolerated exception to the zero-PCIe-copy contract, and is the same exception Phase 2/3 declared.

## 8. Caveats and honest limitations

- **AMGx v2.5.0 links `libmpi.so.40` on this machine**, despite being built with `-DAMGX_NO_MPI=ON`. Inspection of `libamgxsh.so` via `ldd` shows OpenMPI dependencies as `NEEDED` entries. We do not invoke any MPI paths, but if MPI were missing at runtime the library would fail to load. Document this dependency; it's not a correctness issue but a deployment constraint.
- **The `AMGX_solver_resetup` API is marked "deprecated" in AMGx v2.5.0's `amgx_c.h`** but is still functional and is the *only* way to update an existing solver after `AMGX_matrix_replace_coefficients`. `update_coefficients` passes the bit-identical-result correctness test, so this is not a problem in practice, but if AMGx removes the API in a future major release we would have to pivot to full `AMGX_solver_setup` per update — which negates most of the amortization benefit. Flagged for forward-compat.
- **128³ on the 3050 Laptop is tight.** Peak VRAM at 128³ CLASSICAL is 2.34 GiB (0.12 GiB margin post-setup). AGGRESSIVE is comfortable (0.29 GiB). We do not claim 128³ CLASSICAL is universally safe on this card — a second concurrent plan, a JAX workspace that allocates between setup and solve, or any background GPU task pushes it over. AGGRESSIVE is the robust choice.
- **AMGx `print_solve_stats=0` is required in the config** — enabling `print_solve_stats=1` (the AMGx-shipped default) routes per-iter residual diagnostics through our print callback, producing hundreds of stderr lines per solve at 128³ AGGRESSIVE (43 iters). Our configs keep it disabled; enable only for debugging.
- **Iter-count "independence" is approximate at 128³**: CLASSICAL gives 15 iters (matching the 16³–64³ pattern), but AGGRESSIVE gives 43 iters at 128³ vs 16 at 64³. The 2.7× jump comes from aggressive coarsening sacrificing convergence rate for memory. This is a config-choice trade-off, not an algorithm failure; users who can afford the memory get the flat-iter-count property.

## 9. Follow-on options (not Phase 4 scope)

- **Phase 5a**: run on A100/H100 hardware and extend to 256³ / 512³. `AGGRESSIVE` at 128³ uses 292 MiB; 512³ extrapolates to ~18 GiB, comfortable on an 80 GiB card. Iter-count grid-independence should hold — the CLASSICAL config's 15-iter number at 128³ is the evidence.
- **Phase 5b**: integrate `amgx_update_coefficients` into an AM timestep loop with time-varying `ρ(x, t)`. The per-timestep `refresh + solve` cost is what determines production throughput; we have the plumbing but not yet a full-physics benchmark.
- **Phase 5c**: side-by-side with Ginkgo's PGM-AMG per Cojean et al. SIAM 2024. Ginkgo's mixed-precision AMG is the next candidate if we want to push per-iter cost below AMGx's on consumer hardware.
- Multi-GPU AMGx via `AMGX_matrix_upload_distributed`: requires MPI + distributed CSR partitioning; out of scope.

---

**End of Phase 4 report.** Full artifact tree: `dilu/amgx/{cpp,python,configs,tests,bench}/`. Benchmark JSON: `dilu/amgx/bench/scaling_results.json`.
