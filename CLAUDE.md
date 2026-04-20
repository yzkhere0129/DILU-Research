# CLAUDE.md

This file guides Claude Code when working in `/home/yzk/DILU-Research`.

## Project Focus

**DILU (Diagonal-based Incomplete LU) solver low-level optimization research.**

The goal of this repository is to explore and optimize DILU preconditioner / solver implementations — algorithmic variants, memory layout, GPU/CUDA kernels, multi-threaded parallelism, and numerical precision trade-offs. Work here is *not* CFD-application development; it is solver-kernel research.

## Repository Layout

```
DILU-Research/
├── CLAUDE.md                  # this file
├── dilu/                      # (new) DILU solver implementations go here
├── docs/
│   ├── specs/                 # reference specs (Eulerian PLIC, Lagrangian VOF, conservative redistribute, float64 analysis)
│   ├── design/                # math design notes
│   ├── reference/             # reference paper notes (Barkhudarov)
│   ├── benchmark/             # performance reports and progression
│   └── session_logs/          # cross-session handoff notes
├── src/vof/                   # reference VOF/PLIC source (lagrangian, lagrangian_3d, plic)
├── examples/                  # reference VOF/PLIC test scripts (no output artifacts)
├── skills/                    # reference VOF-related skills
└── memory/                    # persistent memory index
```

## VOF/PLIC Material Is Reference Only

All `docs/`, `src/vof/`, `examples/`, and `skills/` content copied from the predecessor `JAX-LaserAM-plic-research` is **frozen reference material**. It is kept because:

- Parallel-kernel engineering patterns (OMP atomic, FFI boundaries, cache-friendly layouts) are directly reusable for DILU kernel work.
- Numerical-precision lessons (float32 ε bounds, double accumulators, `jnp.clip` semantics) generalize to sparse linear algebra.
- Volume-conservation debugging methodology (per-step diagnostic, tracing loss to a single operator) is a template for residual/convergence debugging in iterative solvers.

**Do not modify VOF/PLIC reference code or docs** unless the user explicitly asks. Treat it like an external library archived in-tree for reading.

## New DILU Work Goes Under `dilu/`

When the user begins implementing DILU solvers, new code should go under `dilu/` (or a clearly named sibling directory), never mixed into `src/vof/`. Keep the reference and the new research lexically separated so future readers can tell them apart at a glance.

## Related External Paths

- Full upstream history (with VTK/PNG artifacts, git log, branches): `/home/yzk/JAX-LaserAM-plic-research`
- Parallel JAX-LaserAM main-branch work: `/home/yzk/JAX-LaserAM`

## Conventions Carried Over From Predecessor

- Python ≥ 3.10, JAX + NumPy stack.
- No `-ffast-math`; keep FP associativity deterministic for regression comparability.
- When using float32, use ε ≥ 1e-6f and double-precision accumulators at reduction points.
- `np.ascontiguousarray` before any C/CUDA FFI boundary.

These are defaults from the reference material — override them intentionally once DILU's own numerical requirements are established.
