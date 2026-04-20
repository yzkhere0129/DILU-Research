# PLIC on JAX/GPU: Research Status Briefing

> **Purpose**: This document is a handoff briefing for external experts.
> It describes our research goal, current progress, and the specific
> technical challenges we have encountered so far. It is written to be
> machine-readable and context-efficient for AI-assisted analysis.
>
> **Date**: 2026-04-12  
> **Branch**: `yzk/plic-research` on `/home/yzk/JAX-LaserAM-plic-research`  
> **Contact**: yzk (Yang Zikang)

---

## 1. Research Goal

Implement a traditional **Eulerian PLIC VOF** (Piecewise Linear Interface
Calculation, Volume-of-Fluid) method that runs efficiently on GPU via
**JAX** (Google's array computing framework with XLA compilation).

The long-term target is a multi-phase interface tracker for **laser
additive manufacturing (LPBF) simulation** — melt pool free surfaces,
solidification fronts, gas-metal interfaces. The immediate deliverable
is a standalone PLIC advection module with benchmark-quality accuracy
and measured GPU performance, whose design patterns can transfer to a
future team-internal AM CFD platform.

### Why JAX (not CUDA C++ or OpenFOAM)

- Automatic differentiation (AD) for inverse problems / ML coupling
- pytree-based immutable state for functional purity and composability
- Native multi-GPU via `jax.pmap` without MPI boilerplate
- Same source runs on CPU (dev/debug) and GPU (production)

### Why PLIC (not level-set or diffuse interface)

- Sharp interface (1-cell wide) with strict volume conservation
- No reinitialization or regularization needed
- Well-understood theory (Youngs 1982, Scardovelli-Zaleski 2000,
  Weymouth-Zaleski 2010)
- Industry standard in OpenFOAM (isoAdvector, interFoam) and FLOW-3D

---

## 2. What PLIC Requires (Algorithm Summary)

One time step of Eulerian PLIC advection on a Cartesian grid:

```
for each directional sweep (x, y, z — operator split):
    1. RECONSTRUCT interface in every mixed cell:
       a. Compute interface normal n from F field (e.g. Youngs 3x3x3 gradient)
       b. Find plane intercept d such that Vol(plane ∩ cell) = F  (root solve)
    2. ADVECT volume fraction:
       a. For each cell face, identify the upwind (donor) cell
       b. Compute the volume of fluid in the "sweep box" (the sub-region
          of the donor cell swept across the face in dt)
       c. Update F conservatively: F_new = F_old - div(flux)
    3. HALO UPDATE between sweeps (refresh boundary ghost cells)
```

Key computational characteristics relevant to GPU mapping:

| Property | Value | GPU implication |
|----------|-------|-----------------|
| Fraction of cells that are mixed (0 < F < 1) | typically 1-5% | 95-99% of cells need no PLIC work |
| Normal computation | 3x3x3 stencil convolution | Maps well to cuDNN / XLA conv |
| Intercept root solve | 1D monotone root on V(d) | Iterative; iteration count may vary per cell |
| Sweep-box volume | Plane ∩ axis-aligned box | Analytic formula exists (Scardovelli-Zaleski 2000) |
| Operator splitting | 3 directional sweeps per step | Sequential; halo update between sweeps |

---

## 3. Current Progress (Stages Completed)

### Stage 1 — Mathematical Primitives (commit `0d44abd`)

Delivered three standalone modules under `src/jax_laseram/vof/plic/`:

| Module | What it does | JAX idiom used |
|--------|-------------|----------------|
| `volume_formula.py` | Scardovelli-Zaleski analytic Vol(plane ∩ unit cube) | Nested `jnp.where` cascade (6 branches, branchless) |
| `normal_youngs.py` | Parker-Young 3x3x3 weighted gradient normal | `jax.lax.conv_general_dilated` (3D convolution) |
| `intercept_solver.py` | Regula Falsi + bisection fallback, 15 fixed steps | `jax.lax.fori_loop` (fixed iteration count) |

**Test gate** (39 pytest cases, all pass):

| Metric | Measured | Threshold |
|--------|----------|-----------|
| Youngs normal angular error (sphere R/h=16) | L2 = 0.0126 | < 0.02 |
| Youngs normal angular error (45° plane) | 0.011° | < 2° |
| Intercept solver 1000-sample fuzz max rel err | 2.19e-4 | < 5e-4 |
| HLO equation count: `volume_below_plane_3d` | 123 | < 500 |
| HLO equation count: `compute_youngs_normal_3d` | 98 | < 500 |
| HLO equation count: `solve_intercept` (15 steps) | **33** | < 2000 |

### Stage 2 — Geometric Flux + Strang Split (commit `11ad3fa`)

Delivered three more modules and wired them into a working benchmark:

| Module | What it does |
|--------|-------------|
| `geometric_flux.py` | Per-face flux via PLIC plane ∩ axis-aligned sweep box |
| `strang_sweep.py` | 2D Strang split (x/2 → y → x/2), per-sub-sweep reconstruction |
| `diagnostics.py` | Per-step volume / boundedness / interface-cell-count metrics |

**Benchmark result** — 45° droplet advection (100×100, CFL=0.5, 71 steps):

| Metric | Measured | Note |
|--------|----------|------|
| Volume drift | **0.000e+00** | Machine-precision conservation |
| F range | [-1.2e-8, 1.000] | No overshoot, no undershoot |
| Final centroid | (0.55406, 0.55360) | |
| Analytic final | (0.55360, 0.55360) | |
| Centroid error / dx | (0.046, 0.0003) | Sub-cell accuracy |
| Interface cells | 2196 → 123 (steady) | PLIC compresses tanh init to sharp interface |
| Wall time (CPU) | 145s / 71 steps = **2048 ms/step** | Not yet profiled on GPU |

---

## 4. Technical Challenges Encountered

The following are **observed difficulties** in mapping PLIC to JAX/XLA.
Each is documented with what we tried, what worked, and what remains
open. We present these as engineering constraints to be solved, not as
fundamental impossibilities.

### 4.1 Dense computation over sparse interface

**Problem**: Only ~1-5% of cells have interfaces, but JAX's JIT
compilation requires static array shapes. We cannot dynamically skip
pure cells or build a variable-length "active cell list".

**Current approach**: Compute PLIC operations on **all cells**, then
mask results with `jnp.where(is_interface, plic_result, upwind_fallback)`.
This means ~95% of intercept-solver iterations produce results that are
immediately discarded.

**Measured cost**: For the droplet benchmark (100×100×3 = 30k cells,
~600 interface cells), the intercept solver runs 15 iterations on all
30k cells instead of just 600. Estimated overhead factor: ~50x in
FLOPs, partially offset by GPU's ability to parallelize uniform work.

**Open question (Q2)**: JAX offers `jnp.nonzero(mask, size=MAX)` for
static-size gather. We have not yet measured whether gather → compute →
scatter on a compact array is faster than dense-over-all for typical
interface fractions. This is planned for Stage 4.

### 4.2 Fixed vs adaptive iteration count

**Problem**: `jax.lax.while_loop` on GPU executes until the **slowest
cell** converges, because all threads in a warp execute in lockstep.
For the intercept root solver, most cells converge in 3-4 iterations
but edge cases may need 15+.

**Current approach**: Fixed 15-step `jax.lax.fori_loop`. Every cell
runs exactly 15 iterations regardless of convergence. In float64 this
reaches machine precision; in float32 it saturates at ~1e-7 after
~10 steps (the last 5 are wasted).

**Measured cost**: The 15-step fori_loop compiles to **33 HLO equations**
(XLA wraps the body into a single scan primitive). This is compact and
does not cause JIT compilation blowup. The runtime cost is 15x the
single-evaluation cost per cell.

**Open question (Q3)**: The Scardovelli-Zaleski (2000) paper provides
an **analytic inverse** for the intercept (Cardano's formula for the
cubic). This would replace the 15-step iteration with a single O(1)
evaluation. We have not yet implemented this (planned for Stage 4,
"Phase B" of the intercept solver). The iterative solver serves as a
numerical oracle for validating the analytic formula.

### 4.3 HLO graph explosion from geometric clipping

**Problem**: Our earlier **Lagrangian VOF** implementation used
Sutherland-Hodgman polygon clipping (variable-length vertex lists,
nested loops over clip planes). When expressed in JAX via
`vmap(vmap(scan(...)))` with padded vertex arrays, the XLA HLO graph
grew to hundreds of thousands of operations. First-call JIT compilation
took **2656 seconds** — longer than the simulation itself.

**Current approach**: The Eulerian PLIC formulation avoids polygon
clipping entirely. The sweep box is always an axis-aligned rectangle,
and its intersection with the PLIC plane has a **closed-form analytic
volume** (Scardovelli-Zaleski 2000). The entire forward model compiles
to **123 HLO equations** with JIT time ~2 seconds.

**Key insight**: The choice of **Eulerian vs Lagrangian** framing has
a dramatic effect on JAX compilability — not because of the physics,
but because of the **data structure complexity** that each approach
requires. Eulerian PLIC needs only scalar fields and elementwise ops;
Lagrangian PLIC needs variable-length polygon representations that
stress XLA's static-shape requirement.

### 4.4 Halo update latency between directional sweeps

**Problem**: Strang operator splitting requires a halo (ghost cell)
update between each directional sub-sweep. For multi-GPU (`jax.pmap`),
this means an inter-device communication step per sub-sweep. In 3D
Strang (5 sub-sweeps per step), this is 5 halo exchanges.

**Current approach**: Single-device only (no pmap). Halo update is a
pure-JAX `F.at[...].set(...)` operation — effectively a memory copy.
On CPU this is cheap (~1% of step time).

**Open question (Q4)**: On multi-GPU, the halo exchange cost may become
significant relative to the compute (which is embarrassingly parallel).
We have not profiled this yet. The existing JAX-Fluids `HaloManager`
supports pmap-based halo exchange and could be reused.

### 4.5 No access to GPU shared memory or warp-level primitives

**Problem**: Optimal CUDA PLIC kernels use shared memory for halo
tiling (loading a 3x3x3 neighborhood tile into L1 for the Youngs
stencil) and warp-level ballot/shuffle for stream compaction. JAX
provides no mechanism to control shared memory allocation or use warp
intrinsics. All memory management is delegated to XLA.

**Current approach**: Rely on XLA's automatic fusion and cuDNN's
internal optimization for the convolution stencil. We have no direct
evidence of whether XLA tiles the data effectively.

**Open question**: This is fundamentally a "JAX as an abstraction layer"
limitation. The cost is unknown until we have GPU profiling data
(Stage 3-4). If the gap to CUDA is >10x, a `jax.ffi` custom kernel
for the Youngs stencil would be the escape hatch.

### 4.6 Quasi-2D constraint from 3D stencil

**Problem**: The Youngs normal uses a 3×3×3 convolution kernel, which
requires at least 3 cells in every spatial direction. Truly 2D
simulations (nz=1) produce zero normals everywhere because the kernel's
z-extent has no valid interior.

**Current approach**: Use `nz=3` with z-homogeneous data (quasi-2D).
The physics is identical to 2D but the code path exercises the full 3D
pipeline. All three z interior layers are valid, but the two z-boundary
layers (halo) have degraded normals.

**Note**: This is not a fundamental limitation — a 2D-specific Youngs
stencil (3×3 kernel) could be added. We chose to keep the code path
uniformly 3D for consistency with the target AM application.

---

## 5. Performance Context

### Theoretical bounds (RTX 5060, 256 GB/s bandwidth)

For 1M cells (100³), float32, one PLIC step with 3 Strang sub-sweeps:

- Memory traffic per step: ~92 MB (read F, n, C, u; write F; ×3 sweeps)
- Bandwidth-limited minimum: **0.36 ms/step**
- Compute-limited minimum (15-step bisection): ~0.5 ms/step

### Measured (CPU, 100×100×3 = 30k cells)

- **2048 ms/step** (145s / 71 steps)
- Breakdown (estimated from timing): ~680 ms per sub-sweep, dominated
  by `solve_intercept` (15 fori_loop iterations of `volume_below_plane_3d`)

### Reference (Lagrangian VOF, same codebase, 1M cells)

- **1622 ms/step** with C/OpenMP FFI for overlay (40% of time)
- Pure JAX overlay: JIT compile **2656 seconds**, abandoned

### Reference (OpenFOAM isoAdvector, CPU C++, 1M cells)

- ~0.5 ms/step (per-cell cost ~0.1 μs)

### GPU projection (not yet measured)

- Expected 20-50x speedup over CPU JAX → **40-100 ms/step** at 1M cells
- Stage 4 will run `profile_1M_plic.py` on RTX 5060 for actual numbers

---

## 6. Codebase Map

```
src/jax_laseram/vof/plic/            ← NEW (this research)
├── __init__.py                       module docstring + layout
├── volume_formula.py                 V(plane ∩ unit cube), SZ 2000
├── normal_youngs.py                  Parker-Young 3x3x3 conv normal
├── intercept_solver.py               15-step Regula Falsi + bisect fallback
├── geometric_flux.py                 per-face Eulerian donor sweep flux
├── strang_sweep.py                   2D Strang split orchestration
└── diagnostics.py                    per-step conservation metrics

tests/plic_eulerian/
├── test_primitives.py                39 cases: Stage 1 gate
└── (test_modules.py)                 planned: Stage 2 gate

examples/plic_eulerian_tests/
├── _visualize_initial_state.py       initial-state visualization helper
├── droplet_45deg/                    45° droplet advection benchmark
│   ├── case_config.py                all physical parameters
│   ├── init_droplet.py               tanh-smoothed circular droplet init
│   ├── run.py                        Stage 2 time loop (working)
│   ├── extract_metrics.py            volume + centroid + perimeter metrics
│   ├── make_summary_plot.py          3-row summary figure
│   └── plot_results.py               phase evolution plots
└── rt_instability/                   Rayleigh-Taylor benchmark (Stage 4+)
    ├── case_config.py                air/helium parameters
    ├── init_alpha.py                 cosine-perturbed interface init
    ├── run.py                        Stage 1 smoke test only (needs NS)
    ├── post_process.py               h1/h2 extraction + OF comparison
    └── of_h1_h2.npz                  OpenFOAM reference data
```

### Key dependencies from upstream JAX-Fluids

| What we use | Where | Why |
|-------------|-------|-----|
| `jax_laseram.grid.create_grid` | grid construction | Provides `GridInfo` with dx/dy/dz/nh |
| `jaxfluids.halos.halo_manager` | planned Stage 3+ | Multi-GPU halo exchange |
| `jax.lax.conv_general_dilated` | `normal_youngs.py` | 3D convolution via cuDNN |

---

## 7. Open Research Questions

These are the five questions from our research plan. Stage 1-2 have
partial answers; Stages 3-5 will complete them.

| ID | Question | Status |
|----|----------|--------|
| Q1 | Which JAX idioms approach bandwidth ceiling on PLIC, and which are performance black holes? | **Partial**: `jnp.where` cascade + `conv_general_dilated` are efficient; `fori_loop` with 15 steps is acceptable; polygon clipping via `vmap+scan` is a black hole (2656s JIT). GPU profiling needed. |
| Q2 | Can padded-gather sparse filtering help? | **Untested**: `jnp.nonzero(mask, size=MAX)` exists but we haven't measured the gather/scatter overhead vs dense compute. |
| Q3 | Fixed-step iteration vs analytic closed-form: which wins? | **Partial**: 15-step Regula Falsi works at float32 precision. Scardovelli-Zaleski analytic inverse (Cardano) not yet implemented. Expected to eliminate the ~37% bisection bottleneck entirely. |
| Q4 | Inter-sweep halo update cost: compute-bound or latency-bound? | **Untested**: Single-device only so far. |
| Q5 | How to avoid JIT compilation blowup? | **Answered for Stage 1-2**: `fori_loop` body compiles to 33 HLO equations (single scan primitive, no unrolling). Avoid nested `vmap` over variable-length structures. Axis-aligned geometry keeps HLO compact. |

---

## 8. What We Need Help With

We are specifically looking for expert input on:

1. **Analytic intercept solver (Scardovelli-Zaleski 2000 Appendix A)**:
   The 5-region piecewise cubic inverse with Cardano's trigonometric
   form. Has anyone implemented this branchlessly for GPU? What are the
   float32 precision pitfalls at region boundaries?

2. **Sparse-over-dense trade-off on XLA/GPU**: Is there empirical data
   on when `jnp.nonzero + gather + vmap(f) + scatter` beats
   `vmap(f_with_mask)` on modern GPUs (A100/H100/RTX 50xx)?

3. **3D Strang split symmetry**: For AM applications with anisotropic
   flow (deep keyhole melt pool), does the xyzyx Strang sequence
   introduce visible directional bias? Would a Lie-Trotter rotation
   (xyz one step, zyx next) be more robust in practice?

4. **Weymouth-Zaleski divergence correction**: For weakly compressible
   flow (JAX-Fluids context), the velocity field is not exactly
   div-free. The WZ 2010 correction term involves a per-cell flag
   `c^n ∈ {0, 1}` that alternates — is this flag compatible with
   JAX's functional/immutable state model, or does it need special
   treatment?

5. **Performance floor for JAX PLIC on GPU**: Given the ~20x
   dense-over-sparse overhead and the lack of shared-memory control,
   what is a realistic per-step target for 10^6 cells on RTX 5060?
   Is 10-20 ms/step achievable, or are we looking at 50-100 ms?
