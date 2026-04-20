---
name: CFD numerics precision rules
description: Critical numerical engineering rules for CFD/VOF code — float precision, compiler flags, thread safety
type: feedback
---

1. **NEVER use -ffast-math in CFD geometry code.** Breaks IEEE 754, causes coplanar detection errors, topo-judgment failures, negative tet volumes. Only acceptable for deep learning.
**Why:** Lagrangian VOF face clipping has near-degenerate distance comparisons that rely on IEEE 754 compliance.
**How to apply:** Any Makefile/build system for geometry kernels must explicitly omit -ffast-math.

2. **float32 EPS must be ≥ 1e-6f, not 1e-10f.** Machine epsilon for float32 is 1.19e-7. Anything below ~1e-7 is truncation noise.
**Why:** Testing `dist < 1e-10f` in float32 is a no-op — truncation error already exceeds this.
**How to apply:** Geometric decisions (inside/outside, coplanar) use EPS=1e-6f. Denominator guards (prevent div-by-zero) can use 1e-12f.

3. **Volume accumulation MUST use float64 (double).** When summing millions of tiny float32 volumes, "big eats small" catastrophically destroys conservation.
**Why:** 5M cells × micro-volumes → accumulated round-off error >> individual cell volumes.
**How to apply:** Accumulator arrays for F_new are double*, cast to float32 only at the final output.

4. **Use `#pragma omp atomic` for F_out writes as default.** Even if current grids are uniform, future non-uniform grids or large-CFL overlap may cause write conflicts.
**Why:** Production code must be robust to grid topology changes without silent data races.
**How to apply:** Always atomic on parallel writes to shared arrays. Measure overhead — if <10%, keep it.

5. **6-tet hex decomposition is a first-order approximation.** Post-Lagrangian deformation, hex faces are bilinear patches, not planes. Document this assumption.
**Why:** Strict scientific rigor requires acknowledging geometric approximation sources.
**How to apply:** Comment in code + mention in papers. The volume error is O(CFL × du/dx × dx²).

6. **Always `ascontiguousarray()` before C FFI.** XLA/JAX array strides may not match C row-major expectations.
**Why:** Prevents 90% of segfaults in JAX→C data transfer.
**How to apply:** Flatten to 1D in Python, reconstruct indices in C.
