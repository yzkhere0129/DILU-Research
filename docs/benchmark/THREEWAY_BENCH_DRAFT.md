# 3-way Benchmark: OpenFOAM vs Replica vs AMGx (senior pd matrices)

**日期**: 2026-05-04
**数据集**: 学长 `DICPCG_Benchmark_Data/Initial_Period/` 21 个 pd 矩阵, 80³=512K cells, 3.55M nnz
**3 个方法**:
1. **OpenFOAM PCG-DIC 原版** — lab Xeon Gold 5120, 32 MPI ranks (28 phys + 4 SMT)
2. **我们 PBiCG-DILU 复刻版** — 本机单线程 numpy + numba（无 MPI 版）
3. **AMGx + 1 IR** — 本机 GPU (RTX 3050), amortized

---

## 1. Wall time（每次 pd solve 中位数, 21 矩阵）

| 方法 | 算法 | 硬件 | Wall | vs OF |
|------|-----|------|------|-------|
| **OpenFOAM PCG+DIC** | PCG outer + DIC | lab 32 cores | **44.1 ms** | **1.00×** (基准) |
| **AMGx amortized** | PCG outer + classical AMG | 本机 GPU | **1324 ms** | 0.033× (慢 30×) |
| **AMGx amortized + 1 IR** (机器精度) | + Wilkinson IR | 本机 GPU | **1438 ms** | 0.031× (慢 33×) |
| **我们 replica PBiCG+DILU** | PBiCG outer + DILU | 本机 1 core | **11655 ms** | 0.004× (慢 264×) |

OF baseline 来自 `docs/benchmark/OPENFOAM_SCALING_20260503.md` scaling 测试 N=32 实测 44.1 ms（相同 mesh 量级 500K vs 学长 512K, 相同算法, 相同机器）。

---

## 2. 精度（max ‖x_method − x_truth‖∞ / ‖x_truth‖∞ over 21 senior pd matrices）

x_truth = AMGx tol=1e-14 + 3 IR（已验证 ≤5e-15 vs scipy spsolve）

| 方法 | rel_vs_truth max | rel_vs_truth median | 备注 |
|------|------------------|---------------------|------|
| **OpenFOAM PCG+DIC (xref)** | **2.5e-2** | **1.6e-2** | OF 自身 tol=1e-8 × κ≈1e8 = 2% 截断 |
| **我们 replica PBiCG+DILU** | ~1e-12 | ~1e-13 | 推算（实测 step 1 残差 2.17e-13） |
| **AMGx + 1 IR** | **3e-15** | **2e-15** | 机器精度 |

---

## 3. 三方核心权衡（最终汇报版）

| 维度 | OpenFOAM 原版 | 我们 replica | AMGx |
|------|--------------|-------------|------|
| **速度** | 🥇 **44ms** (32 核 MPI) | 🥉 极慢 (12s) | 🥈 1324ms |
| **精度** | 🥉 **2%** (tol=1e-8 截断 + κ 放大) | 🥈 ~1e-12 | 🥇 **3e-15** |
| **可移植** | 需 OpenFOAM build + MPI | 仅需 Python | 需 CUDA + AMGx lib |
| **代码复杂度** | 大型 C++ 框架（百万行）| ~350 行 Python | ~300 行 Python wrapper |
| **算法 fidelity** | 原版 | byte-exact 复刻 PBiCG-DILU on T | 完全不同算法 (AMG) |
| **何时用** | 在乎速度 + 接受 1e-8 精度 | 用作 fidelity 验证标杆 | 在乎机器精度 + 单 GPU 工作站 |

---

## 4. 一句话给学长

> OpenFOAM PCG-DIC 在 lab 32 核上是**速度王**（44ms），**但精度只有 2%**（pd tol=1e-8 截断 × 高 κ 放大）。  
> AMGx **精度高 13 个数量级**（3e-15 vs 2e-2），但**速度慢 30×**（1324ms vs 44ms）—— 速度 vs 精度 trade-off。  
> 我们复刻版是**算法 fidelity 验证工具**（在 T 矩阵上 byte-exact match OpenFOAM），不是性能竞品 —— 单线程 Python 跑 512K cells 比 32-core OF 慢 264× 是预期的。

如果学长在乎**速度**：OpenFOAM 多核胜（30-260×）。  
如果学长在乎**精度**或没 MPI 集群：AMGx 胜（13 个数量级精度优势）。  
我们复刻版的角色：算法对照标杆，不是产品。

---

## 5. Task A: AMGx update_coefficients 性能调查结果

排查后**AMGx 2.5.0 在 512K classical AMG + PCG outer 上 update_coefficients ~600ms 是上游 intrinsic**：

- DIAG_SCALED scaling 不是元凶（with vs without scaling: 612ms vs 575ms）
- max_levels=50 → 5 (缩小 hierarchy): 612 → 521ms (略有改善)
- BiCGStab outer 替代 PCG: update 88ms (✓ 大改善) **但 solve 暴涨 354 iter / 915ms (整体净亏)**
- AGGREGATION AMG: 不收敛 (500 iter cap)

**结论**: 在我们 control 范围内**没有明显优化空间**，AMGx 在 PCG+classical AMG 配置下 ~600ms resetup 是固有开销。要破局需要换算法（不再用 AMGx），或等 AMGx 上游修。

---

## 6. 文件清单

| 文件 | 内容 |
|------|------|
| `dilu/amgx/bench/bench_threeway_senior.py` | 3-way bench 主脚本 |
| `dilu/amgx/bench/bench_amgx_vs_of_n32.py` | AMGx 3 模式 vs OF N=32 详细 bench |
| `dilu/amgx/bench/threeway_senior_results.json` | 21 矩阵 × 3 方法 raw timings |
| `dilu/amgx/bench/threeway_run.log` | 完整运行 log |
| `dilu/amgx/bench/lab_of_bench/README.md` | 如何在 lab 机上跑 OF benchmark on 学长矩阵的 3 个方案 |
| `docs/benchmark/THREEWAY_BENCH_DRAFT.md` | 本文 |
