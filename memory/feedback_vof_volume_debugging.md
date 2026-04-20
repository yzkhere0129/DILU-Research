---
name: VOF volume loss debugging methodology
description: Per-step volume tracing identifies loss mechanisms; dV in PLIC is wrong; F>1 clip kills volume
type: feedback
originSessionId: 5e32a9bc-e0e1-4c93-90d7-d06a3fda388e
---
When debugging Lagrangian VOF volume loss, use a **per-step diagnostic** that traces volume at every pipeline stage (overlay, redistribution, zeroing, clip). This immediately identified the exact loss mechanism.

**Why:** Blind fix attempts (pressure projection, redistribution, uniform vs potential flow) wasted many iterations. The diagnostic script (diag_test2.py) showed in one run that:
1. Overlay preserves volume perfectly (raw ΔV ≈ 0%)
2. 100% of loss came from clipping F>1 → 1.0
3. V_clipped matched V_lost exactly — smoking gun

**How to apply:** For any VOF volume issue:
1. Print total V at each pipeline stage (raw overlay, redist, zeroing, clip)
2. Print max F, cells F>1, V_overfill BEFORE clip
3. Compare V_lost with V_clipped — if they match, clip is the culprit
4. Check F in obstacle cells — if 0, redistribution is a no-op

**Key formula insight:** In the PLIC transfer `F * overlap / fluid_vol`, the normalization Σ overlap/fluid_vol = 1 over all acceptors. So total = F. The dV factor (V_old/V_new) is WRONG — it over-deposits, causing F>1, which gets clipped. Barkhudarov Eq.8 dV applies to the non-PLIC case (overlap/cell_vol normalization), NOT the PLIC case.
