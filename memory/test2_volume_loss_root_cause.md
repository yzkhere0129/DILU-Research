# Test 2 Volume Loss Root Cause Analysis

## Symptom
Test 2 (3D droplet impacting obstacle) shows -79% volume loss over 39 steps on 50×50×5 grid.

## Root Cause: Overfill Clipping at Obstacle Boundary

The volume loss is NOT from:
- F-zeroing in obstacle cells (verified: `in_obs = 0.0` always)
- Obstacle vertex clamping (verified: identical results with/without)
- Domain boundary clamping (verified: +0.17% without obstacle)
- Native vs batched overlay (verified: native never called, always falls back to batched due to PLIC)
- JAX trace differences (verified: single step identical, multi-step diverges from overfill)

The actual cause: **overfill clipping** in the overlay. The overlay does `jnp.clip(F_new, 0.0, 1.0)` which discards excess volume when acceptor cells receive F > 1.0. Near the obstacle boundary, fluid "piles up" against the wall, causing cells to overfill. The discarded excess is permanent volume loss.

Evidence:
- Even with `obs_x_max=-inf` (no obstacle params) but WITH obstacle velocity zeroing, volume loss is -12.9% at step 20
- Total volume (including obstacle cells) decreases even without F-zeroing
- The loss accelerates over time as more fluid accumulates near the obstacle

## Contributing Factors

1. **Grid resolution**: 50×50×5 (dx=0.02) is coarse. Barkhudarov paper uses 10 cells/diameter (dx=0.01 for D=0.1).
2. **Vertex clamping at obstacle**: 3-condition check fails for cells outside obstacle in one axis but with vertices on obstacle face. `eps=1e-10` is too tight for float32 (vertex at 0.49999997 fails `>= 0.5 - 1e-10`). Fixed by using `eps=1e-6`.
3. **PLIC overlay never used**: Native C doesn't support PLIC, batched fallback ignores PLIC data. Both use non-PLIC hex-box overlap.

## Code State (2026-04-08)
- `__init__.py`: obstacle clamping with 3-condition check, eps=1e-6, domain boundary clipping
- `overlay_native.py`: restored native callback (was accidentally removed)
- `overlay_batched.py`: ignores PLIC data (always uses `_hex_box_volume`)
