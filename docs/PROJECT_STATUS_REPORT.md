# DILU-Research Project Status Report

**日期**: 2026-05-09
**目的**: 给研究伙伴看的综合状态报告 — 项目背景、当前状态、关键发现、开放问题

---

## 1. Executive Summary

本项目研究**线性求解器在 LPBF (laser powder bed fusion) CFD 仿真中的精度与性能**。核心问题：能否用 GPU AMG (AMGx) 替代 OpenFOAM 默认的 CPU DICPCG，加速大网格 LPBF 仿真？

### 1.1 已建立的事实（有数据支撑）

| Finding | 数据支撑 |
|---|---|
| AMGx CLASSICAL_V_DIAGSCALED + 1 IR 算法**实现正确** | 跟 scipy LU 在 8K cells 上一致到 max rel 2.89e-15 (`precision_results_ir1.json`) |
| AMGx + 1 IR 跟 CHOLMOD LU 在 500K LPBF 矩阵上一致到 rel **1.13e-11** | 6 timesteps 全 verified (`lu_truth_single_track_results.json`) |
| OF DICPCG @ tol=1e-8 跟 LU 真值差 max **44.6 Pa** (rel 3.5e-5) | 6 timesteps，符合 PCG 数值理论 κ × tol ≈ 1e-5 |
| AMGx PCG @ tol=1e-8 跟 LU 真值差 max **56.3 Pa** (rel 4.4e-5) | 6 timesteps，与 OF 同量级 |
| OF DICPCG 迭代次数 17-65 vs AMGx PCG 435-773（同 tol） | DIC preconditioner 对 Laplacian 类矩阵结构优势 |
| 单次求解 wall: OF~3s, AMGx 1e-8 ~25s, AMGx 1e-12+IR ~50s, CHOLMOD ~70s | 500K cells，数据来自 `solver_comparison_500K.json` |

### 1.2 已 settle 的混淆（之前误读）

- ❌ ~~"AMGx 比 OF 准"~~ → **OF 在 case 设的 tol=1e-8 下工作正确**，rel 1e-5 是 condition number 放大的预期
- ❌ ~~"lab32 5.9 kPa 偏离是 ill-posed null space artifact"~~ → 用 SuperLU 直解证明 **system 良态**，5.9 kPa 是 OF tol=1e-8 + κ ~ 1e6 的标准放大
- ❌ ~~"AMGx 在 lab32 broken case 上是对的，OF 错"~~ → **case 配置错误（rays=0 → b≈0）**，两个 solver 都按各自 tol 工作正常

### 1.3 待验证的核心假设

> **AMGx amortized + warm-start 在真实 CFD 时间推进下应该比 LU 直解快 100-1000×**（因为 LU 必须每步 refactor，而 AMGx `update_coefficients` 重用 AMG hierarchy）

正在做 **replay experiment**（Phase 0 6-step pilot 已跑，Phase 1 384-step dense 等 lab Xeon dump）。

---

## 2. 项目背景与历史

### 2.1 研究主线

研究 OF 的 DILU/DICPCG vs AMGx vs scipy LU 在 LPBF 矩阵上的：
- 数值精度（vs LU 真值）
- 求解速度（单次 + amortized）
- Memory scaling（N → 10M+）
- 适用场景

### 2.2 数据演化（4 版本数据集）

| 时期 | 数据集 | 状态 | 备注 |
|---|---|---|---|
| 早期 (2026-04) | LPBF_sanity 2K + dumper_pipeline 8K | ✅ archived | scipy LU 验证 AMGx algorithm 正确 |
| v3.2 (2026-04-27) | spot_melt_npz 1M (dev) | ❌ **数据丢失** | 报告了"AMGx 比 OF 快 8.6×"，但是 dev 单核 OF vs dev RTX 3050 AMGx 不公平比较 |
| 学长 (2026-05-04) | Initial_Period 21 bundles 512K | ✅ 21/21 收敛 | AMGx 精度验证 OK |
| 学长 (2026-05-06) | Melting + Evaporation 各 ~30-60 矩阵 512K | ❌ **不可解** | matrixDumper.H bug，‖A·x_OF - b‖/‖b‖ = 18 |
| 我们 (2026-05-07) | lab32_dump 500K (32-rank) | ⚠ **rays=0** | OF case 配置错（laser 没打中粉床），matrices 数学合法但物理空洞 |
| 我们 (2026-05-08) | **single_track_dump 500K (单核, 6 timesteps)** | ✅ **canonical** | rays>0 真实 LPBF 物理，T 1707K → 4067K，verified vs LU 1e-11 |
| 我们 (2026-05-09 跑中) | multi_track_dump_2M (单核, 24 timesteps) | 🟡 跑到 500ns/1200ns | 16/24 dump 已 trigger |
| 我们 (待启动) | dense_track_dump_500K (384 timesteps) | ⏳ template ready | 用于 amortized replay 实验 |

### 2.3 关键修复

- **matrixDumper.H 32-rank bug fixed** (commit `47d4863`): 之前漏了 processor patch 上的 boundary contributions，导致 32-rank dump 出来的矩阵不自洽
- **OF v2412 + laserMeltFoam 重新 build with patches** (commit `a62ffa0`): 集成我们的 matrixDumper hooks 到 fresh github tree，via apply_patches.py
- **Laser 配置**: 150W → 300W (commits in `single_track_dump/run_extraction.sh`)

---

## 3. 硬件资源

3 台机器，物理隔离：

| 机器 | 用户 | CPU | GPU | RAM | 角色 |
|---|---|---|---|---|---|
| **dev** (Yzk-laptop, WSL2) | yzk | 笔记本 4 核 | RTX 3050 (4 GB VRAM) | 9.7 GB + 8 GB swap | 开发、出图、AMGx GPU benchmark、500K LU |
| **lab Xeon** (HR54WV2) | yzk | Xeon Gold 5120, 28 phys × 2 SMT = 56 logical | 无 | ~64 GB | OF 真跑（laserMeltFoam）+ CHOLMOD LU + dump 矩阵 |
| **lab 5060** (ME-6T8XHG4) | manyxu | Xeon | RTX 5060 (sm_120), 8GB | 大 | AMGx GPU benchmark on Initial_Period |

---

## 4. 关键数据 — 500K canonical case

### 4.1 case 描述

- **物理**: LPBF 单 track，300W laser，t=0→1.2 μs (1200 timesteps)
- **网格**: 50 × 200 × 50 = 500,000 cells, dx = 4 μm
- **几何**: 200 × 800 × 200 μm domain, x-axis 100μm 处激光焦点 along +y 扫描
- **Solver**: OF v2412 laserMeltFoam (single-core)，DICPCG @ tol=1e-8 for pd, DILUPBiCG for T
- **物理验证**: T_max 演化 298 → 1707K (melt) → 3857K (keyhole)，都符合 LPBF 物理预期

### 4.2 矩阵 dumps

- **稀疏结构**: nnz = 3,455,000 = ~7 nnz/row (7-point stencil)
- **每个 timestep dump**: pd_corr0 + pd_corr1 + pd_corr2 + T_corr0
- **当前 6 timesteps**: 320, 380, 410, 700, 900, 1060 ns
- **Dense 384 timesteps** 待跑（Phase 1 实验）

### 4.3 4-solver comparison（已验证）

`docs/benchmark/solver_comparison_500K.md` 完整表格。Summary：

| solver | tol | iter | wall | max\|err\|∞ vs LU | rel max | ‖err‖₂ vs LU |
|---|---|---|---|---|---|---|
| OF DICPCG | 1e-8 | 17-65 | ~2-7 s* | 44.6 Pa | 3.5e-5 | 448 Pa |
| AMGx PCG | 1e-8 | 435-773 | 9-39 s | 56.3 Pa | 4.4e-5 | 333 Pa |
| AMGx + 1 IR | 1e-12 | 559-1123 | 22-101 s | 1.4e-5 Pa | 1.1e-11 | 1.4e-6 Pa |
| CHOLMOD LU | direct | — | 70-75 s | 0 (truth) | 0 | 0 |

\* OF wall estimated（log.run step-Δt averaging）not directly measured

### 4.4 物理量演化（验证 case 物理）

| timestep | T_max (K) | pd 范围 (Pa) | 物理状态 |
|---|---|---|---|
| 320 ns | 1676 | -2.4e5 ~ 1.28e6 | 熔池形成 |
| 380 ns | 1707 | 同 | 深熔 |
| 410 ns | 1749 | 同 | 高熔 |
| 700 ns | 2795 | 同 | 蒸发临界 |
| 900 ns | 3426 | -2.4e5 ~ 1.37e6 | 强蒸发 |
| 1060 ns | 3857 | -2.4e5 ~ 1.28e6 | keyhole 形成 |

---

## 5. 当前实验

### 5.1 Phase 0 — 6-timestep replay (running on dev)

5 个 solver mode 对比：
- AMGx fresh (worst)
- AMGx amortized (Plan once, update_coefficients)
- AMGx amortized + warm-start (predicted best)
- CHOLMOD fresh (worst LU)
- CHOLMOD symbolic reuse (analyze once, cholesky_inplace)

**Status**: 🟡 当前 dev 上跑（PID 466877，CPU 5+ min）

### 5.2 Phase 1 — 384-timestep dense replay (待 lab Xeon 启动)

已推 git commit `2fe3348`（template 在 `dilu/openfoam_patches/dense_track_dump_500K/`）。
启动后 7h 跑完，给 384 个 dense 矩阵供 Phase 0 replay 重跑（统计置信度更高）。

### 5.3 Phase 2 — Synthetic Poisson scaling (idle)

`synthetic_scaling.py` 写好（commit `fb2a6bb`），跑 32-100 cube sizes，找 LU OOM threshold。等 Phase 0 跑完 GPU 闲下来再做。

### 5.4 multi_track_dump_2M — 大网格 LPBF case (lab Xeon overnight)

- 80×320×80 = 2M cells，physics 同 single_track
- 24 dump points，已 dump 16/24（跑到 t=500 ns）
- 估计还要 18-21h 跑完，**或现在 kill 拿部分数据**
- 用途：scaling 测试 + 验证 single_track 结论在大网格上

---

## 6. 仓库结构

```
/home/yzk/DILU-Research/
├── CLAUDE.md                            ← project context
├── PROJECT_STATUS_REPORT.md             ← 本文
├── docs/
│   ├── PROJECT_SUMMARY.md
│   ├── benchmark/
│   │   ├── data_inventory.md            ← 全数据 inventory（重要）
│   │   ├── solver_comparison_500K.md    ← 4-solver 表格
│   │   ├── solver_production_comparison.md  ← Phase 0+1+2 报告
│   │   ├── AMGX_PRECISION_20260504.md
│   │   ├── THREEWAY_BENCH_FINAL.md
│   │   ├── OPENFOAM_CROSSCHECK_20260427_v3.2.md
│   │   └── figures/
│   │       ├── single_track_physical_fields.png         ← 4-panel pd/T 物理云图
│   │       ├── single_track_3d_solver_error.png         ← 3D 误差散点
│   │       ├── single_track_OF_vs_AMGx.png              ← OF vs AMGx 直接差
│   │       ├── scientific_solver_verification_*.png     ← 3 张科学图
│   │       ├── scientific_3d_physical_pd_6timesteps.png ← 6-timestep 物理
│   │       ├── scientific_3d_errors_6timesteps_3solvers.png ← 误差对比
│   │       └── ... 30+ 张其它图
│   ├── design/                          ← 算法设计文档
│   ├── reference/                       ← 论文阅读笔记
│   ├── session_logs/                    ← 跨 session 交接
│   └── partner_handoff/                 ← 老的 handoff 包（v3.2 时代）
├── dilu/
│   ├── amgx/                            ← AMGx Python wrapper + benchmarks
│   │   ├── python/                      ← Plan, update_coefficients, IR
│   │   └── bench/
│   │       ├── single_*pd_corr0_*.npz   ← 6 个 500K 处理后数据
│   │       ├── lu_truth_single_track.py ← LU 真值验证
│   │       ├── build_solver_comparison_table.py
│   │       ├── plot_3d_solver_overview.py
│   │       └── plot_scientific_verification.py
│   ├── openfoam_patches/                ← OF case templates
│   │   ├── laserMeltFoam/               ← matrixDumper.H + 集成 patches
│   │   ├── single_track_dump/           ← 500K case 模板
│   │   ├── dense_track_dump_500K/       ← 384-step case 模板
│   │   └── multi_track_dump_2M/         ← 2M case 模板
│   ├── experiments/
│   │   └── solver_production_comparison/  ← Phase 0+1+2 实验
│   │       ├── EXPERIMENT_DESIGN.md
│   │       ├── replay_amortized.py
│   │       ├── synthetic_scaling.py
│   │       └── plot_replay_results.py
│   └── benchmark/                       ← 历史 benchmark（学长数据 npz）
└── memory/                              ← persistent agent memory
```

---

## 7. 矩阵数据存放位置

| 数据 | 路径 | 大小 |
|---|---|---|
| **500K single_track 矩阵** (lab Xeon) | `~/cases/single_track_dump/postProcessing/matrices/` | ~250 MB |
| **500K single_track 矩阵** (dev, 解 tar) | `/home/yzk/single_track_dump/postProcessing/matrices/` | 同 |
| **6 个 npz** (含 x_OF + x_AMGx_e8 + x_truth) | `dilu/amgx/bench/single_*pd_corr0*.npz` | 6 × 13 MB = 78 MB（git） |
| **2M multi_track 矩阵** (lab Xeon, 跑中) | `~/cases/multi_track_dump_2M/postProcessing/matrices/` | ~10 GB（dump 中） |
| **学长 21 bundles** (Initial_Period) | `dilu/benchmark/DICPCG_Benchmark_Data_npz/` | 已 npz |
| **学长 Melting + Evaporation** (broken) | `dilu/benchmark/Melting/` + `Evaporation/` | CSV，未处理 |

---

## 8. 给研究伙伴的开放问题

### 8.1 战略选择
1. **要不要继续走 AMGx 路线？** 实测 500K 上 OF DICPCG 比 AMGx 快 5-10×。AMGx 优势主要在 (a) 大网格 (>5M)，(b) 多 GPU。我们的 LPBF case 是否会到 5M+ 量级？
2. **AMGx config 是否最优？** CLASSICAL_V_DIAGSCALED 在 LPBF pd 上需 435-773 iter。能否换 PCG-DIC 类前置（OF 用的方法）让 AMGx 收敛更快？
3. **OF DICPCG 真的不能并行？** OF 的 DICPCG 实测单核 ~110 ms/iter on 500K。多核 / GPU 实现是否可能 5-10× speedup？
4. **是否需要 P4 (replicate OF DICPCG in Python/C++)？** 当前 in_progress。值不值得继续投入？

### 8.2 实验设计选择
1. **Phase 1 dense replay（384 timesteps）够吗？** 还是要更密 (1200 timesteps × pd 全部)？
2. **Synthetic Poisson 用合成 vs LPBF**？前者可大网格找 OOM threshold，后者更代表性。
3. **OF 重跑 with solverInfo 拿真实 wall**？6.7h 单核成本，能拿到 per-solve 准确 wall（vs 当前估算）。值得吗？
4. **2M overnight kill 还是继续？** 已跑到 500ns（41%），再 18-21h 才到 1.2μs。500ns 数据 vs 1.2μs 数据差异有多大物理价值？

### 8.3 写论文方向
1. **核心 claim 是什么？** 选项:
   - A. "AMGx + 1 IR can match LU truth on real LPBF matrices"
   - B. "AMGx amortized + warm-start beats CPU CFD by N× in production"
   - C. "OF DICPCG on small mesh is hard to beat"
   - D. "matrixDumper.H + LU truth methodology for solver verification"
2. **目标 venue**? Journal of Computational Physics? Computers & Fluids? IJNMF?
3. **Novel contribution 是什么？** 算法上没有原创（AMGx 是现成）；novel 在**细致的 LPBF 矩阵 cross-check methodology + LU truth verification**。够 paper-worthy 吗？

### 8.4 技术问题
1. **lab32_dump (rays=0) 数据怎么处理？** 已发现是 case 配置错。要不要写诊断报告给学长？
2. **学长 Melting + Evaporation matrices 不可解** — 是 case 配置 + matrixDumper bug。要不要让他重 dump？
3. **2M LU 真值** — 单核 SuperLU 估计 ~10-30 min/case，跑得通但慢。要不要做？

---

## 9. 关键参考文档（detail 层）

| 文档 | 内容 |
|---|---|
| `docs/benchmark/data_inventory.md` | 全数据集 + 全实验全清单（最长，最 detail） |
| `docs/benchmark/solver_comparison_500K.md` | 4-solver 表格 + 结论 |
| `docs/benchmark/solver_production_comparison.md` | replay 实验报告（pending 数据） |
| `docs/benchmark/AMGX_PRECISION_20260504.md` | AMGx vs scipy LU 8K verify (canonical) |
| `dilu/experiments/solver_production_comparison/EXPERIMENT_DESIGN.md` | 实验协议详情 |
| `dilu/openfoam_patches/laserMeltFoam/` | OF source patches + apply script |

---

## 10. 时间线 (recent activity)

```
2026-05-04  AMGx vs scipy LU on 8K cells: 2.89e-15 max rel diff (canonical)
2026-05-05  21-bundle senior data lab 5060 AMGx benchmark: tol_sweep_results.json
2026-05-06  Senior melting/evap data lab Xeon LSMR truth attempt — FAIL (5/6 broken)
            Conclusion: senior matrixDumper.H bug漏 boundary patch contributions
2026-05-07  matrixDumper.H patched, lab32 32-rank rerun: 2.4e-9 self-consistency
            But case has rays=0 bug (laser misconfigured) → broken physics
2026-05-08  Single-core 500K case run on lab Xeon (single_track_dump):
              Real LPBF physics, 300W laser, 6.7h wall, 6 dumps × 4 eq
              T_max 4067K, pd up to 1.28 MPa (vapor recoil)
            User challenges 5.9 kPa diff hypothesis → CHOLMOD LU experiment
              proves system well-posed, 5.9 kPa = κ × tol amplification
2026-05-09  CHOLMOD LU verified AMGx + IR matches LU to rel 1e-11 (6/6 timesteps)
            4-solver comparison table + scientific 3D plots
            Replay experiment Phase 0 launched on dev (running)
            Dense_track_dump_500K case template pushed (Phase 1 ready)
            Multi_track_dump_2M overnight running on lab Xeon (16/24 dumps)
            ★ 本报告生成
```

---

## 附: Git history (最近 10 commits)

```
3b11e17  solver_production_comparison.md report 模板
fb2a6bb  EXPERIMENT_DESIGN.md + synthetic_scaling.py
d5d99cb  replay_amortized.py + plot_replay_results.py (Phase 0 框架)
2fe3348  dense_track_dump_500K case 384-step template
5d10ac9  solver_comparison_500K.md final + OF wall estimate
85dea74  500K 4-solver comparison table + 3D plots
cd5d3e3  scientific solver verification 3 figures
1d3a112  push 6 single_track pd npz + tolerate missing npz
1279832  lu_truth_single_track.py
1827f4c  LU direct solve settles AMGx vs OF on lab32 — null-space hypothesis WAS WRONG
```

---

**报告结束**。供研究伙伴策略参考。具体技术细节问题可深入到 §9 引用的 detail 文档。
