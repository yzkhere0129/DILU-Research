# SESSION_HANDOFF — 2026-04-21

**Scope**: end-of-roadmap handoff. Phase 1 → Phase 2 → Phase 2.5 → Phase 3 →
Phase 3 extended (64³/128³) → Phase 4 (AMGx). This session closes the
four-phase DILU-Research master plan. Next session is expected to be on
A100 / H100 hardware for scale-up and production AM integration.

Working tree at session end: `/home/yzk/DILU-Research` · `main` @ commit
`1e36584` (Phase 4 implementation) → will advance with this session's
final commits including `docs/PROJECT_SUMMARY.md` update and this handoff
file itself.

---

## 1. What this session produced

### 1.1 Full 4-phase implementation delivered

| Phase | Directory | Status | Commit |
|-------|-----------|--------|--------|
| Phase 1 (FFI MVP) | `dilu/ffi_mvp/` | PASS | prior sessions |
| Phase 2 (cuSPARSE DILU) | `dilu/cusparse/` | PASS | prior sessions |
| Phase 2.5 (physical benchmark) | `dilu/cusparse/tests/physical_benchmark.py` | PASS | prior sessions |
| Phase 3 (multi-color DILU) | `dilu/multicolor/` | PASS, F2 not triggered | prior sessions |
| Phase 3 extended (64³/128³) | `dilu/multicolor/bench/bench_scaling_64_128.py` | scaling regression documented | `9e4316d` |
| **Phase 4 (AMGx)** | `dilu/amgx/` | **PASS, 22.1× @ 128³** | `1e36584` |

### 1.2 Design docs added (8 total, paired math + engineering per phase)

`docs/design/`:
- `phase1_dilu_math_foundation.md`
- `phase1_ffi_prototype_architecture.md`
- `phase2_cusparse_level_scheduling_math.md`
- `phase2_cusparse_ffi_architecture.md`
- `phase3_multicoloring_math.md`
- `phase3_multicolor_ffi_architecture.md`
- `phase4_amg_math_foundation.md`
- `phase4_amgx_ffi_architecture.md`

### 1.3 Benchmark reports (7)

`docs/benchmark/`:
- `phase1_mvp_report.md`
- `phase1_repro_HR54WV2_gtx1080.md` (cross-machine repro: GTX 1080)
- `phase2_cusparse_report.md`
- `phase2.5_physical_report.md`
- `phase3_multicolor_report.md`
- `phase3_scaling_64_128.md` (CRITICAL: scaling regression finding)
- `phase4_amgx_report.md`

### 1.4 Result images (12 total, 3 sets of physical benchmarks + 64³ scaling)

- Phase 2.5 (32³): `dilu/cusparse/bench/plots/{divergence_map,spurious_currents,residual_halo}.png`
- Phase 3 (32³): `dilu/multicolor/bench/plots/{divergence_map,spurious_currents,residual_halo}.png`
- Phase 3 ext (64³ P2 vs P3): `dilu/multicolor/bench/plots/scale_64/*.png` (4 images)
- Phase 4 (64³ AMG): `dilu/amgx/bench/plots/scale_64/*.png` (2 images)

### 1.5 Project-level indexes

- `docs/PROJECT_SUMMARY.md` (updated this session to include Phase 3
  scaling + Phase 4 + critical implementation landmines + reproduction
  environment spec)
- `docs/session_logs/SESSION_HANDOFF_20260421.md` (this file)

---

## 2. Key results summary

### 2.1 T7 iteration count (16³ stiff ρ-contrast 100)

- Jacobi-PCG: 71 iters
- Phase 2 exact DILU-PCG: **24 iters** (3× reduction)
- Phase 3 multi-color DILU-PCG: **36 iters** (1.5× penalty vs P2)
- Phase 4 AMG-PCG: **15 iters** (4.7× reduction vs Jacobi)

### 2.2 Total PCG wall time (RTX 3050 Laptop)

| Grid | P2 wall (s) | P3 wall (s) | P4 wall (s) | **P4 / P2** |
|------|-------------|-------------|-------------|-------------|
| 16³ | 0.045 | 0.034 | 0.532 (setup-dominated) | 0.08× (DILU wins) |
| 32³ | 0.556 | 0.204 | 0.207 | 2.69× |
| 64³ | 6.49 | 8.08 | 0.463 | **14.0×** |
| **128³** | **28.47** | **35.00** | **1.287** | **22.1×** |

### 2.3 Physical correctness (acid test F2)

Test C correlation `corr(|E_demean|, |∇log ρ|)` at 32³ three-tier:
- Phase 2.5 (DILU 50 iters): ~0.1 magnitude (no interface correlation)
- Phase 3 (multi-color 50 iters): **-0.102**
- Phase 4 (AMG 15 iters): **0.0313**

F2 stop line is 0.5. **Never triggered** in any phase.

### 2.4 The Phase 3 scaling regression (critical project finding)

Phase 3's headline 2.72× total-PCG speedup at 32³ was the PEAK of a curve,
not a monotone trend. At 64³/128³ the total PCG speedup regresses to
0.80× / 0.81× because per-iter cost becomes bandwidth-bound SpMV (same for
both implementations), while the 1.6× iter penalty stays constant.

This is why Phase 4 exists: DILU-family preconditioners scale as $N^{1/3}$
in iter count on stiff 3-D Poisson. The only way to get grid-size-
independent convergence on this problem class is to switch preconditioner
class to AMG.

---

## 3. Environment at session end

### 3.1 Host

- Linux `6.6.87.2-microsoft-standard-WSL2` (WSL2 / Windows 11)
- Python 3.12 @ `/home/yzk/jax-env/`
- jax 0.9.0 / jaxlib 0.9.0 (cuda12, typed FFI)
- numpy ≥ 2.0 · scipy 1.17.0 · matplotlib 3.10.8
- gcc 13.3 / libstdc++ 13
- CUDA 12.4.131 · driver 580.97
- Host compiler for CMake: `gcc` / `g++` 13

### 3.2 GPU

- NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB total, CC 8.6
- FP64 @ 1/32 FP32 (consumer SKU trade-off)
- Baseline free VRAM when idle: ~3.0 GiB (varies with WSL2 + host desktop)

### 3.3 AMGx install (Phase 4 dependency)

- Source: `/home/yzk/src/amgx/` (tag `v2.5.0`, 2024-12-21)
- Install prefix: `/home/yzk/local/amgx/`
- Library: `/home/yzk/local/amgx/lib/libamgxsh.so` (~143 MiB)
- Headers: `/home/yzk/local/amgx/include/amgx_c.h`, `amgx_config.h`
- Build flags: `-DCUDA_ARCH=86 -DAMGX_NO_MPI=ON -DCMAKE_BUILD_TYPE=Release`
- Runtime dependency (soft): `libopenmpi-dev` installed on the system (even
  with `AMGX_NO_MPI=ON`, `ldd libamgxsh.so` lists `libmpi.so.40` as NEEDED)

### 3.4 Cross-machine data points (Phase 1 repro, not primary)

- GTX 1080 @ Xeon Gold 5120 (HR54WV2): CUDA 12.9, JAX 0.9.1, idle
- RTX 5060 @ Intel Ultra 7 265F (ME-6T8XHG4): CUDA 13.2, JAX 0.9.1,
  GPU was contended during repro (1003 µs median, not a clean baseline)

### 3.5 XLA runtime knobs (MUST be set in every Python entry point)

```
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Every `conftest.py` and bench script sets these before `import jax`.

---

## 4. Known limitations / honest caveats at session end

### 4.1 Hardware-bound

- 128³ `CLASSICAL_V_CYCLE` AMGx config peaks at 2340 MiB VRAM on the 3050 —
  over the 2 GiB design budget. `AGGRESSIVE_COARSENING` fits at 292 MiB but
  takes 43 iters instead of 15. Routing rule: AGGRESSIVE on 4 GB GPUs,
  CLASSICAL on ≥ 24 GiB.
- > 128³ not attempted on the 3050. Extrapolation in PROJECT_SUMMARY §10
  is theoretical, not measured.
- 128³ DILU-PCG (Phase 2 / Phase 3) may not converge to 1e-10 within 500
  iters. Max-iter cap is hit in some measurements (flagged with `†` in
  `phase3_scaling_64_128.md`). For A100 scaling, allow higher caps.

### 4.2 AMGx-specific

- `AMGX_solver_resetup` is marked `deprecated` in `amgx_c.h` v2.5.0 but
  still functional. If AMGx removes it in a future release, our
  K-amortization strategy breaks (have to do full setup per update).
- AMGx links `libmpi.so.40` at runtime even with `-DAMGX_NO_MPI=ON`.
  Deploy-time constraint only; not a correctness issue.
- `print_solve_stats=1` floods stderr with per-iter diagnostics at 128³.
  Our configs disable it; enable only for debugging.

### 4.3 Software environment

- JAX 0.9.0 → JAX 0.10.x is untested in this project. JAX's typed FFI
  API is stable, but the test coverage we have is 0.9.0-specific.
- CUDA 12.4 is the reference. CUDA 12.5+ should work (AMGx v2.5.0 claims
  12-13). CUDA 13.x: not tested in this session.
- Python 3.12 is the reference. Python 3.13+ introduces libstdc++
  C++ std::string_view overloads that we papered over (§8.3.2 in the
  project summary); on 3.13+ the skeleton might not need the two-arg
  `std::string(data, size)` form.

### 4.4 Test coverage gaps

- No CI; tests run locally by hand. A future A100 session should
  containerize the build + a GitHub Actions matrix on cuda12.4 / 12.6 / 12.8.
- No fuzz-testing of the pattern fingerprint (`(n, nnz)`). In principle
  two distinct patterns with the same `(n, nnz)` could collide. For AM
  we've only ever used 7-point and similar structured stencils; for
  unstructured / AMR meshes a stronger fingerprint is needed.
- Phase 4's T10 wall-time test compares against Phase 2's 28.47 s that
  was measured in a separate session. Not a deal-breaker (both use the
  same matrix generator with fixed seed) but could drift if Phase 2's
  setup changes.

---

## 5. Open items for the next session (A100 / H100 pivot)

### 5.1 Must-do on new hardware first

1. **Rebuild AMGx** for the server GPU compute capability:
   - A100: `-DCUDA_ARCH=80`
   - H100: `-DCUDA_ARCH=90`
   - B100/B200: `-DCUDA_ARCH=100` (Blackwell datacenter)
2. **Rebuild all four `libdilu_*.so`** with the matching `CUDA_ARCH`.
3. **Re-baseline Phase 1 dispatch overhead** on the new hardware. DO NOT
   compare 3050's 267 µs to server numbers; they are incomparable due to
   different JAX/CUDA/driver stacks.
4. **Re-run T7, T9, T10** to confirm deterministic iter counts still match
   the reference numbers in PROJECT_SUMMARY §6.
5. **Run Phase 2.5 / 3 / 4 physical benchmarks at 64³** to confirm the
   F2 acid test (corr ≤ 0.25) still holds on the new hardware.

### 5.2 Scale-up targets

- 256³ (16.8M cells) — AMG AGGRESSIVE expected ~2.3 GiB VRAM (scales with
  nnz ≈ 117M). CLASSICAL expected ~20 GiB. Both fit on A100 80 GiB.
- 512³ (134M cells) — AGGRESSIVE ~18 GiB. CLASSICAL may need tuning
  `max_levels` down to fit.
- Verify: does the 15-iter flatness hold at 256³+ with CLASSICAL? The
  Phase 4 math doc predicts yes (AMG is grid-independent by construction),
  but empirical confirmation is the point of the A100 run.

### 5.3 Production AM integration

- Drive `amgx_update_coefficients` from a ρ(x, t) update loop. Simulate
  a full melt-pool scan pass (thousands of PCG solves per physical second).
- Measure `K_amortize` in practice: how many timesteps between full
  `amgx_setup` calls? Design doc estimated K=20–50; measure the real
  iter-count drift over time in an AM trajectory.
- Couple to a minimal AM physics driver (momentum + phase advection +
  energy) to test the solver in the loop, not just in the unit test.

### 5.4 Follow-on solver comparisons

- **Ginkgo PGM-AMG** (SIAM 2024): mixed-precision AMG. On consumer SKU
  FP32 smoother + FP64 outer PCG could close the gap against AMGx-on-A100.
  Out-of-scope for this session but a natural next comparison.
- **Hypre BoomerAMG** (GPU): alternative AMG, more mature on CPU, newer
  on GPU. Could be a fallback if NVIDIA deprioritizes AMGx further.
- **Multi-GPU AMGx** (`AMGX_matrix_upload_distributed` + MPI): needed
  when AM problems cross single-GPU memory. Requires MPI build of AMGx
  and distributed CSR partitioning logic on the JAX side.

### 5.5 CI / reproducibility hardening

- Add GitHub Actions workflow: build Phase 1-4 on `ubuntu-22.04` + `cuda-12.4`
  container, run T1-T13 as CI gates.
- Add a "reproducibility smoke test": on the reference env, iter counts
  and correlation values from PROJECT_SUMMARY §6 must match exactly. One
  script, one pass/fail line.
- Pin JAX + jaxlib + AMGx + CUDA toolkit versions in a lockfile.

---

## 6. Recipe for "first day on A100 / H100"

```bash
# 1. Clone the repo (assuming already pushed to GitHub)
git clone git@github.com:yzkhere0129/DILU-Research.git
cd DILU-Research

# 2. Install AMGx from source for the target GPU
export AMGX_PREFIX=$HOME/local/amgx
git clone https://github.com/NVIDIA/AMGX $HOME/src/amgx
cd $HOME/src/amgx
git checkout v2.5.0
mkdir build && cd build

# A100:
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=80 -DAMGX_NO_MPI=ON \
         -DCMAKE_INSTALL_PREFIX=$AMGX_PREFIX
# H100:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=90 -DAMGX_NO_MPI=ON \
#          -DCMAKE_INSTALL_PREFIX=$AMGX_PREFIX

make -j8
make install

# 3. Install JAX on the target (match driver)
pip install --upgrade "jax[cuda12]==0.9.0" scipy matplotlib

# 4. Build all four phases with the new CUDA_ARCH
cd ~/DILU-Research
for phase in ffi_mvp cusparse multicolor amgx; do
    cd dilu/$phase
    CUDA_ARCH=80 bash build.sh    # 80 for A100, 90 for H100
    cd ../..
done

# 5. Re-baseline dispatch
python dilu/ffi_mvp/bench/repro_cross_machine.py

# 6. Confirm deterministic tests still pass
python dilu/cusparse/tests/test_t7_pcg_stiff.py                # expect 24 iters
python dilu/multicolor/tests/test_t7_pcg_iteration_count.py    # expect 36 iters
python dilu/amgx/tests/test_t9_iter_count_128.py               # expect 15 iters CLASSICAL
python dilu/amgx/tests/test_t_acid_32.py                       # expect corr=0.0313

# 7. Scale to 256³
# (write a new bench/bench_scaling_256.py by analogy with
#  dilu/multicolor/bench/bench_scaling_64_128.py; use AGGRESSIVE config
#  at first to be VRAM-safe, then move to CLASSICAL)
```

**Expected discrepancies vs 3050**:
- Wall times will be 5–50× faster on A100, 10–100× on H100.
- Iter counts MUST match: T7=24 (Phase 2), T7=36 (Phase 3), T9=15 (Phase 4
  CLASSICAL). Any deviation means environment drift, not hardware.
- Dispatch overhead will likely be much smaller (< 100 µs on server-class
  stacks) but may have its own floor.

---

## 7. Authoritative references at session close

- Master index: `docs/PROJECT_SUMMARY.md`
- Design (architecture + math): `docs/design/phase{1,2,3,4}_*.md` (8 files)
- Benchmark results: `docs/benchmark/phase{1,2,2.5,3,4}_*.md` (7 files)
- Hardware + environment spec: `docs/PROJECT_SUMMARY.md` §4
- Critical implementation landmines: `docs/PROJECT_SUMMARY.md` §8
- Deterministic outputs (reproduction checkpoints): `docs/PROJECT_SUMMARY.md` §6
- Environment-sensitive output ranges: `docs/PROJECT_SUMMARY.md` §7

If any conflict between these and the code, the CODE wins — specs follow
code, not the other way around. But within the spec hierarchy the order
is `reference/ > specs/ > design/ > session_logs/` (per Module E of the
cfd-reverse-engineer skill). This session_log is a point-in-time snapshot;
if in the future it conflicts with PROJECT_SUMMARY.md, the latter wins.

---

## 8. Closing notes

The 4-phase roadmap in CLAUDE.md is complete. The project has an
end-to-end stack from JAX tracing through FFI to CUDA kernels to an
AMG library, all with zero host-device copies beyond a single tolerated
8-byte token per solve. The headline win is 22.1× total PCG wall time
at 128³ — achieved by recognizing that DILU cannot escape its $N^{1/3}$
iter-count scaling and that grid-independent convergence requires the
AMG class of preconditioner.

Phase 3 (multi-color DILU) is preserved in the codebase even though it
regresses at real scale — it remains the demonstration that the bottleneck
is not kernel-launch count but preconditioner math class. Its presence
makes the argument for AMG concrete and measurable.

Phase 2 (cuSPARSE SpSV DILU) is preserved as the small-problem fallback
and as the DILU reference for correctness tests. Production routing rule
in PROJECT_SUMMARY §8.5 / §8.8.

Next session starts on A100 / H100. This handoff should be enough to get
from SSH login to first T9 PASS in under an hour.

*End of session handoff. Project status: 4-phase roadmap CLOSED.*
