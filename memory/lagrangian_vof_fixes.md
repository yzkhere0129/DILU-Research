---
name: Lagrangian VOF critical fixes
description: Three fixes that resolved hollow center + octagonal distortion in Barkhudarov Lagrangian VOF 45° droplet advection
type: project
---

Three critical fixes applied to `src/jax_laseram/vof/lagrangian/` (2026-04-01):

1. **Raw gradient gating** (`__init__.py`): `compute_plic_normals` returns NORMALIZED unit vectors (mag≈1), so comparing mag against 0.5/dx≈50 was always False. Fix: compute RAW gradient magnitude separately; threshold 0.5 cleanly separates real interface (|∇F|~50) from interior noise (|∇F|~0.05). Cells below threshold get zero normals → treated as full cells → no PLIC erosion cascade.

2. **Deformed quads** (`__init__.py`): Build donor polygons directly on deformed quads from `x_verts/y_verts` (output of `lagrangian_move_faces`), not original-grid quads + average translation. Correct for non-uniform flows.

3. **LS normals** (`reconstruction.py`): Replaced centered differences with 3×3 Least-Squares gradient (Prewitt convolution kernel) per Barkhudarov §4. Uses `jax.lax.conv_general_dilated` with VALID padding. 3× lower noise variance, better isotropy. Reduced perimeter distortion from +4.5% to +1.69%.

**Why:** Interior cells with F≈0.999 from numerical diffusion got random PLIC normals (since normalized mag≈1 always passes threshold), eroding the droplet core over time. The LS normals + gradient gating fix both the root cause (noisy normals) and the symptom (interior erosion).

**How to apply:** These fixes are specific to the Lagrangian VOF module. The gradient gating threshold of 0.5 is absolute and grid-independent for practical resolutions.
