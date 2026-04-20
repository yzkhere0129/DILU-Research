# DILU-Research Memory

## Project Status
**Focus (2026-04-19)**: DILU (Diagonal-based Incomplete LU) solver low-level optimization research. VOF/PLIC materials under `../` are **reference only** — prior work kept for inspiration on parallelism, numerical precision, and performance engineering.

## VOF/PLIC Reference Memories
Carried over from `JAX-LaserAM-plic-research` for cross-domain inspiration (parallel strategy, numerical precision, volume-conservation debugging methodology).

- [Lagrangian VOF critical fixes](lagrangian_vof_fixes.md) — gradient gating + deformed quads + LS normals fixed hollow center and octagonal distortion
- [3D Lagrangian VOF status](lagrangian_3d_status.md) — full 3D pipeline, 5M cells proven, 700× slower than C++ (see PERFORMANCE_ANALYSIS.md)
- [Test 2 volume loss root cause](test2_volume_loss_root_cause.md) — -79% volume loss traced to overfill clipping at obstacle boundary
- [CFD numerics precision rules](feedback_cfd_numerics.md) — no -ffast-math, EPS≥1e-6f for float32, double accumulator, omp atomic, 6-tet approximation caveat, ascontiguousarray before FFI
- [VOF volume loss debugging](feedback_vof_volume_debugging.md) — per-step diagnostic; dV wrong in PLIC; F>1 clip kills volume

## Key Paths
- DILU-Research root: `/home/yzk/DILU-Research`
- VOF/PLIC reference docs: `docs/specs/`, `docs/design/`, `docs/benchmark/`, `docs/reference/`
- VOF/PLIC reference code: `src/vof/{lagrangian, lagrangian_3d, plic}`
- Upstream archive (full history): `/home/yzk/JAX-LaserAM-plic-research`
