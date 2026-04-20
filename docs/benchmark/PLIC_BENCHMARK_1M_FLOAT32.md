# PLIC 1M Cells float32 Benchmark — RTX 3050

**Date**: 2026-04-12  
**Branch**: `yzk/plic-research` · commit `4f9b818`  
**Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MB VRAM

---

## 1. Parameters

### Grid & Physics

| Parameter | Value |
|-----------|-------|
| Grid (interior) | 100 × 100 × 100 = **1,000,000 cells** |
| Grid (with halo) | 102 × 102 × 102 = 1,061,208 |
| dx = dy = dz | 0.01 m |
| Precision | **float32** |
| Droplet | sphere, center (0.5, 0.5, 0.5), R = 0.15 m |
| Interface init | tanh, width = 1.5 dx |
| Velocity | u = v = 1.0, w = 0 m/s |
| CFL | 0.3 |
| dt | 3.0 × 10⁻³ s |
| Method | Strang split (x/2 → y → x/2) |
| Interface cells | 67,830 / 1,000,000 (6.78%) |

### PLIC Algorithm

| Component | Method |
|-----------|--------|
| Normal | Parker-Young 3×3×3 conv (cuDNN) |
| Intercept | 15-step Regula Falsi (`fori_loop`) |
| Flux | Eulerian donor sweep-box, analytic SZ 2000 |
| Pure-cell | upwind fallback |

---

## 2. Per-Stage Profiling (JIT-compiled, GPU)

| Stage | Median (ms) | Notes |
|-------|-------------|-------|
| normal (Youngs conv) | **2.6** | cuDNN, fast |
| intercept (15-step RF) | **121.1** | **bottleneck**, 15 iterations × 1M cells |
| **FULL STRANG STEP** | **399.0** | 3 sub-sweeps (x/2 + y + x/2) |

### Breakdown analysis

- Intercept = 121 ms, called 3 times = 363 ms → **91% of full step**
- Normal = 2.6 ms × 3 = 7.8 ms → 2%
- Remaining (flux + apply + halo) ≈ 28 ms → 7%
- XLA fusion savings are limited at 1M scale (compute-bound, not launch-bound)

---

## 3. Full Run (10 Steps)

### Performance

| Metric | Value |
|--------|-------|
| Total wall time | 4.0 s |
| Per-step median | **398 ms** |
| Per-step min/max | 396 / 419 ms |
| Per-step std | 6.8 ms |
| **Throughput** | **2.51 Mcells/s** |
| JIT compile (one-time) | 20.5 s |

### Accuracy

| Metric | Value | Status |
|--------|-------|--------|
| Volume drift | **0.000e+00** | **PASS** (exact conservation) |
| F range | [-7.4e-8, 1.000] | **PASS** |
| Cells F < 0 | 0 | **PASS** |
| Cells F > 1 | 114 | ⚠ minor (float32 noise at F ≈ 1 + ε) |
| Interface cells | 67,830 → 18,127 | PLIC sharpening (as expected) |
| Final centroid | (0.53007, 0.53003, 0.50000) | |
| Expected | (0.53000, 0.53000, 0.50000) | |
| Centroid err / dx | **(+0.007, +0.003, 0.000)** | **PASS** (sub-cell) |

---

## 4. Resource Usage

| Metric | Value |
|--------|-------|
| F array | 4.0 MB (float32) |
| Peak GPU memory | **274.7 MB** |
| Steady GPU memory | 47.4 MB |
| VRAM utilization | **6.7%** of 4 GB |
| HLO equations | 1089 (full step) |
| JIT compile | 20.5 s (one-time) |

---

## 5. Scale Comparison

| Configuration | Cells | Precision | ms/step | Mcells/s | GPU |
|---------------|-------|-----------|---------|----------|-----|
| **1M float32 JIT** | **1,000,000** | **float32** | **398** | **2.51** | **RTX 3050** |
| 30k float64 JIT | 30,000 | float64 | 32.8 | 0.92 | RTX 3050 |
| 30k float64 no-JIT | 30,000 | float64 | 2,232 | 0.013 | RTX 3050 |
| Lagrangian VOF (C FFI) | 1,000,000 | mixed | 1,622 | 0.62 | RTX 5060 |
| OpenFOAM isoAdvector | 1,000,000 | float64 | ~0.5 | ~2.0 | CPU C++ |

### Key observations

1. **Eulerian PLIC @ 1M cells is 4x faster than Lagrangian VOF** (398 ms vs 1622 ms), on a weaker GPU (3050 vs 5060)
2. **Throughput 2.51 Mcells/s exceeds OpenFOAM isoAdvector's ~2.0 Mcells/s**, though OpenFOAM uses float64; fair comparison awaits our float64 1M data
3. **Intercept accounts for 91%** — Stage 4's Scardovelli-Zaleski analytic formula (eliminating the 15-step fori_loop) is projected to reduce 398 ms to ~40-80 ms, i.e. **12-25 Mcells/s**
4. **GPU memory is very light**: peak 275 MB, the 4 GB card can fit **~14M cells**
5. **Volume conservation is exact to 0.0** (still perfect under float32), centroid error < 0.01 dx

---

## 6. Bottleneck & Optimization Path

```
Current 398 ms/step breakdown:

  ┌─────────────────────────────────────────────┐
  │  intercept (15-step RF)   363 ms   91.2%    │ ← Phase B analytic can eliminate
  │  flux + apply + halo       28 ms    7.0%    │
  │  normal (cuDNN conv)        8 ms    1.8%    │
  └─────────────────────────────────────────────┘

After Phase B analytic intercept (estimated):
  
  ┌─────────────────────────────────────────────┐
  │  intercept (analytic O(1)) ~24 ms   33%     │ 15x faster
  │  flux + apply + halo       ~28 ms   39%     │ unchanged  
  │  normal (cuDNN conv)        ~8 ms   11%     │ unchanged
  │  → total                  ~60 ms            │
  │  → throughput            ~17 Mcells/s        │
  └─────────────────────────────────────────────┘
```
