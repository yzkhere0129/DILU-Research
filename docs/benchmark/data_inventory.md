# DILU-Research 数据盘点 (Master Inventory)

**日期**: 2026-05-07
**目的**: 把分散在 12 个 JSON、20+ markdown、3 台机器上的 benchmark 数据全部集中到一张索引表里。

> ⚠ **最重要的一句话**：之前所有 "AMGx vs OF" 对比都不是同一矩阵 — OF 49.57 ms 是 lab Xeon 跑的 **`spot_melt_150W` 500K** mesh，AMGx 190 ms 是 lab 5060 跑的 **学长 Initial_Period 512K** bundles。**网格规模相近但不是同一个 case**。lab32_dump 是为了消除这个 mismatch 才跑的，目前刚跑完，数据还没整理成 JSON。

---

## ⚠ DISCLAIMER — OpenFOAM 是业界标答，本文档不质疑 OF 正确性

OpenFOAM 是 2004 年开源至今 20+ 年工业界 + 学术界使用的标准 CFD 工具，DICPCG/PBiCG/GAMG 等 solver 经过严格数值理论验证 + 千篇 peer-reviewed 论文检验。本仓库做的是**严格的、有限范围的、数值层面**的精度对比研究，不是 OF 的正确性审计。

**严格能讲的**：
1. AMGx 算法（CLASSICAL_V_DIAGSCALED + 1 IR）在 8K cells 上与 scipy LU 直解一致到机器精度（max rel diff 2.89e-15）— `precision_results_ir1.json`
2. AMGx 默认 + IR 可以把 ‖A·x - b‖/‖b‖ 解到 1e-15
3. OF DICPCG 默认 `tolerance 1e-8` 把残差停在 1e-8 — 这是 **case 工程参数**，不是 OF 算法限制
4. matrix + b 良态时（real LPBF physics），OF 和 AMGx 一致到 rel ~1e-5

**严格不能讲的**：
- ❌ "OF 不准" — OF 准度由 case `tolerance` 设定决定
- ❌ "AMGx 比 OF 准" — 不同 tolerance 选择
- ❌ "OF 在 null-space 偏离 truth" — ill-posed 系统没 truth
- ❌ "5.9 kPa 是 OF 的 artifact" — 是 ill-posed 系统的性质 (lab32 rays=0 case)

**lab32 5.9 kPa 解读** (2026-05-09 用 SuperLU 直解 settled，**之前的 ill-posed 解释错了**):

✓ System **是良态的** (LU 给唯一解，残差 2.5e-15)
✓ 5.9 kPa **不是 null-space artifact**

实际原因 — 经典 PCG 数值分析：
```
cell error  ≤  κ(A_local) × residual / ‖A‖

在 weakly diag-dominant cells (diag 1e-26 vs 1e-15) 上 κ_local ~ 1e6
OF tol=1e-8 → residual ~ 1e-8 → cell error ~ 1e-2 × ‖x‖ = 5kPa ★
AMGx tol=1e-12 + IR → residual ~ 1e-15 → cell error ~ 1e-9 × ‖x‖ = 0.004 Pa
```

实测 (`dilu/amgx/bench/lu_truth_lab32_settled.py`):

| solver | tol | max \|x - x_LU\| | 病态 cells > 100 Pa |
|---|---|---|---|
| OF DICPCG | 1e-8 | **5914 Pa** (rel 4.7e-3) | 25 |
| AMGx | 1e-8 | 28 Pa (rel 2.2e-5) | 0 |
| AMGx + 1 IR | 1e-12 | **0.004 Pa** (rel 2.9e-9) | 0 |

两个 solver **都没错** — 都满足各自 `tolerance` 设置。差异来自 condition number 把 residual 放大成 cell-level error，这是标准数值分析行为，不是 OF 或 AMGx 的 bug。**OF 设 `tolerance 1e-12` 也能到 AMGx 同等精度 — 默认 1e-8 是工程参数选择**。

---

## A. 三台机器、三个数据来源、三种 solver — 一张速查地图

### 三台机器

| 机器 | 角色 | CPU | GPU | RAM | 备注 |
|---|---|---|---|---|---|
| **dev** (本机, Yzk-laptop, WSL2) | 开发 + 单核 PCG + 部分 AMGx + 出图 | (4-core 笔记本) | RTX 3050 (4GB) | 9.7 GB + 8 GB swap | 9.7 GB 限制下 2M cell ASCII dump 卡 UI；不能跑 SuperLU on 2M |
| **lab Xeon** `manyxu@HR54WV2` | OF 32-rank 真实跑 + 32-rank 矩阵 dump | Xeon Gold 5120, 28 phys × 2 SMT = 56 logical | 无 GPU | 大 | MPI 必须 `--oversubscribe` 才能用 32 ranks；spot_melt_150W 是 500K cells (非学长 512K) |
| **lab 5060** `yzk@<5060>` | AMGx GPU benchmark | (Xeon 配套) | RTX 5060 (sm_120), CUDA 13.2 | 大 | AMGx 编译需 NVTX header workaround；本 jax-env 在 `/home/yzk/jax-env`（不是 .venv） |

### 三个数据来源

| 来源 | 数据集 | 网格 | 物理阶段 | 状态 |
|---|---|---|---|---|
| 学长 (manyxu) | **Initial_Period** | 80³ = 512K | cold-start (t < 100 ns) | ✅ 21 bundles 全部 AMGx 解到 ε-machine |
| 学长 | **Melting** | 80³ = 512K | t = 320-380 ns | ❌ **不可解** (LSMR istop=7, ‖A·x_OF - b‖/‖b‖ = 18) |
| 学长 | **Evaporation** | 80³ = 512K | t = 700-1060 ns | ❌ 2/3 不可解；step 84 部分收敛但 PISO 解偏差 31% |
| 我们 dev 单核 dump | **LPBF_crosscheck** | 80×320×80 = 2M | 单时间点 | ✅ AMGx tol=1e-13 当 truth；OF rel err 9e-6 |
| 我们 lab 32-rank dump | **lab32_dump** | 80×320×80 = 2M（同 mesh, 32 procs） | melting + evap 两阶段（计划） | 🟡 已 dump，正在整理中（本次工作） |
| 我们 lab Xeon OF 跑 | **spot_melt_150W** | 50×200×50 = 500K | 单 case 跑 | ✅ N=32 wall 49.57 ms；**没 dump 矩阵给 AMGx** |
| 我们旧 dev | **LPBF_sanity / dumper_pipeline** | 2K / 8K | 早期管道测试 | ✅ AMGx vs scipy truth 验证到 2.89e-15 |
| 我们旧 dev (v3.2) | **spot_melt_npz** (1M) | 80×160×80 = 1M | 历史数据 | ⚠ **不在当前 working tree**；产生过 8.6× 加速这个误传说法 |
| 我们 dev (合成) | **synthetic Poisson** | 16³-128³ | Phase 4 scaling | ✅ 历史档 (2026-04-21 closeout) |

### 三种 solver

| Solver | 跑在哪 | 用过的数据 | 主要文件 |
|---|---|---|---|
| **OF DICPCG** | lab Xeon (32 rank), dev (1 core 历史) | spot_melt 500K, lab32_dump, LPBF_crosscheck, v3.2 1M | `bench_amgx_vs_of_n32_results.json`, `of_lab_xeon_500K_pd_timings.txt`, `OPENFOAM_SCALING_20260503.md` |
| **AMGx** (CLASSICAL_V_DIAGSCALED) | lab 5060 (主力), dev RTX 3050 (历史) | 学长 Initial_Period 21 bundles, LPBF_sanity 2K, dumper 8K, LPBF_crosscheck 2M, lab32_dump 2M | `tol_sweep_results.json`, `precision_results*.json`, `bench_amgx_vs_of_n32_results.json`, `threeway_senior_results.json` |
| **AMGx + 1 IR** | 同上 | 同上 | 同上 (`n_refine=1` 行) |
| **scipy SuperLU / CHOLMOD / MUMPS** | dev (小 case), lab Xeon (大 case) | 学长 Initial_Period (CHOLMOD on lab), LPBF_sanity 8K (SuperLU truth), v3.2 1M (历史) | `lab_LU_robust.py`, `precision_results*.json`, recent commits a0b6ed5/b95f020/5207c3d |
| **LSMR** (least-squares) | lab Xeon (作为 broken 数据 truth proxy) | 学长 Melting + Evaporation (失败) | `lab_PCG_truth_results.json` |
| **我们的 PBiCG-DILU replica** (Python 单核) | dev | 学长 Initial_Period 21 bundles | `byte_match_proof_results.json`, `loose_match_results.json`, `threeway_senior_results.json` |
| **我们的 C++ DILU replica (P1)** | dev | 单元测试 | `dilu/cpp/` (P2 进行中) |

---

## B. 学长三组数据 — 详细盘点

### B.1 Initial_Period（唯一可解的一组）

| 字段 | 值 |
|---|---|
| mesh | 80×80×80 = **512,000** cells |
| nnz | 3,545,600 |
| 时间步 | steps {1, 2, 3, 4, 5, 10, 11} × correctors {1, 2, 3} = **21 bundles** |
| 物理阶段 | cold-start (laser 刚启动，t < 100 ns) |
| 文件位置 | `dilu/benchmark/DICPCG_Benchmark_Data_npz/bundle_pd_*.npz` |
| 符号约定 | A 存储为负对角 (OF Laplacian)，loader 翻转后给 AMGx |
| condition number κ(A) | ≈ 10⁸ (估算，来自 plot 注释) |
| 收敛性 | ✅ **AMGx + 1 IR 全部 21 bundles 跑到 ‖A·x - b‖/‖b‖ ≤ 2.17e-15** (即机器精度) |
| OF reference x_xref | 是 OF tol=1e-8 的解，与真解差 1.6%-2.5% (这是 OF 截断误差，被 v3.2 误读为 AMGx 误差) |

**所有 solver 在这组数据上的表现汇总**：

| Solver | 机器 | tol | iter | wall | rel_resid | rel_vs_xref | 数据源 |
|---|---|---|---|---|---|---|---|
| AMGx CLASSICAL_V_DIAGSCALED | lab 5060 | 1e-14 | 26 | 180.81 ms (solve) | 2.56e-13 | 2.5% | tol_sweep (config 1) |
| AMGx + 1 IR | lab 5060 | 1e-14 | 26 + IR | 182.61 + 96.50 = 279.11 ms | **2.17e-15** | 2.5% | tol_sweep (config 2) |
| AMGx | lab 5060 | 1e-12 | 26 | 190.20 ms | 2.56e-13 | 2.5% | tol_sweep (config 3) |
| AMGx + 1 IR | lab 5060 | 1e-12 | 26 + IR | 241.88 + 124.00 = 365.88 ms | **2.17e-15** | 2.5% | tol_sweep (config 4) |
| AMGx | lab 5060 | 1e-10 | 22 | 201.18 ms | 4.04e-11 | 2.5% | tol_sweep (config 5) |
| AMGx | lab 5060 | **1e-8** | 17 | **120.60 ms** | **8.10e-9** | 2.5% | tol_sweep (config 7) |
| AMGx | lab 5060 | 1e-6 | 13 | 123.42 ms | 4.11e-7 | 2.5% | tol_sweep (config 8) |
| AMGx | lab 5060 | 1e-4 | 9 | 86.37 ms | 7.96e-5 | 2.5% | tol_sweep (config 9) |
| AMGx (mode_amortized) | dev RTX 3050 | 1e-12 | 26 | ~200-240 ms (solve) / 540 ms (total) | n/a | 2.5% | bench_amgx_vs_of_n32 |
| AMGx + 1 IR (mode_ir) | dev RTX 3050 | 1e-12 | 26+IR | ~320 ms | n/a | 2.5% | bench_amgx_vs_of_n32 |
| AMGx (threeway report) | dev RTX 3050 | 1e-12 | 26 | **1324 ms total** ⚠ | n/a | n/a | threeway_senior — wall 与 bench 不一致 |
| 我们 PBiCG-DILU replica (tight) | dev 1 core | 1e-12 | 195-196 | 9-10.5 s/solve | 2.1e-13 | 2.5% | byte_match_proof |
| 我们 PBiCG-DILU replica (loose) | dev 1 core | 1e-8 | 137 | n/a | 1.8-1.9e-9 | 2.5% | loose_match |
| AMGx vs replica (loose) | dev | 1e-8 | n/a | n/a | rel_AMGx_vs_replica = 1.1e-4 ⚠ | n/a | loose_match (orphan finding) |
| AMGx vs replica (tight) | dev | 1e-12 | n/a | n/a | rel_AMGx_vs_replica = 6e-8 to 3e-7 | n/a | byte_match_proof (orphan finding) |
| OF DICPCG | **NEVER RUN** on this mesh — 学长跑过但记录没发我们 | n/a | n/a | n/a | n/a | n/a | (no JSON) |

### B.2 Melting（不可解）

| 字段 | 值 |
|---|---|
| mesh | 512K (80³) |
| 总矩阵数 | ~33 (steps 65-75, t=320-380 ns, 每步多 corrector) |
| 已抽样 | steps {65, 70, 75} × corr=1 = **3 个** |
| 文件位置 | `dilu/benchmark/Melting/Melting/` (CSV 原格式，未转 npz) |
| **状态** | ❌ **完全不可解** |

**LSMR truth attempt（lab Xeon, 2026-05-06）**：

| step | corr | iters | LSMR istop | LSMR actual_resid | rel(x_OF) | rel(x_PISO) | x_truth range | x_OF range |
|---|---|---|---|---|---|---|---|---|
| 65 | 1 | 5000 (cap) | 7 (无收敛) | 0.0976 | 18.10 | 18.10 | [-9.9e6, 7.2e6] | [99k, 119k] |
| 70 | 1 | 5000 (cap) | 7 | 0.0973 | 18.09 | 18.09 | [-1.3e7, 7.2e6] | [98k, 137k] |
| 75 | 1 | 5000 (cap) | 7 | 0.097 (类似) | 18 | 18 | 同样无意义 | 物理范围 ~1e5 Pa |

**诊断**：
- LSMR `norma = 1.9e-24` (operator norm 极小)
- `b_sum/b_abssum ≈ -1.0`（b 几乎是常向量）
- 推测：A 含有未消除的 null space（与 OF 的 reference cell pinning 有关），b 几乎在 null space 方向上 → A·x = b 没有解，OF 是靠迭代 + reference cell 把它"解"出来
- 来源：`lab_PCG_truth_results.json`
- **这个 finding 还没写进任何 markdown 报告** — 是孤儿数据

### B.3 Evaporation（2/3 不可解，1/3 部分收敛但 PISO 偏差大）

| 字段 | 值 |
|---|---|
| mesh | 512K (80³) |
| 总矩阵数 | ~57 (steps 84-102, t=700ns-1.06μs) |
| 已抽样 | steps {84, 95, 102} × corr=1 = **3 个** |

| step | corr | LSMR iters | istop | actual_resid | rel(x_OF) | rel(x_PISO_vs_truth) | 状态 |
|---|---|---|---|---|---|---|---|
| 84 | 1 | 1451 | 6 (收敛) | 2.6e-6 | 18.08 | **1.31** (31% 偏差) | 🟡 LSMR 收敛但 PISO 解≠真解 |
| 95 | 1 | 5000 (cap) | 7 | 0.097 | 18 | n/a | ❌ 同 melting |
| 102 | 1 | 5000 (cap) | 7 | 0.097 | 18 | n/a | ❌ |

来源：`lab_PCG_truth_results.json`

---

## C. 我们自己的数据 — 详细盘点

### C.1 LPBF_crosscheck (2M, dev 单核 dump) — 当前 3D 图用的数据

| 字段 | 值 |
|---|---|
| mesh | 80×320×80 = **2,048,000** cells |
| domain | 200 × 800 × 200 μm，2.5 μm cell |
| 跑在哪 | dev 单核 OF 跑（matrixDumper.H 串行版 dump） |
| 时间步 | 单时间点 t = 3.537e-09 s（cold-start 区间） |
| 文件位置 | `LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/postProcessing/matrices/3.537365257e-09/pd_corr0/` |
| 状态 | ✅ 自洽（dump 是干净的） |
| Truth 方法 | scipy SuperLU **OOM**（dev 9.7GB 装不下 2M），只能用 AMGx tol=1e-13 当 ε-machine truth proxy |

**solver 表现**：

| solver | tol | wall | iter | rel error vs truth | max err in Pa |
|---|---|---|---|---|---|
| AMGx tol=1e-13 (作 truth) | 1e-13 | n/a | n/a | 1.7e-12 | n/a (truth 自身) |
| AMGx tol=1e-12 | 1e-12 | n/a | 26 | 2.6e-13 | 0 (≈truth) |
| AMGx tol=1e-8 | 1e-8 | n/a | 17 | **2.6e-6 rel** | 5.9 Pa |
| OF DICPCG tol=1e-8 (x_final 来自 dump) | 1e-8 | n/a | ~5 | **9.0e-6 rel** | 20.9 Pa |

来源：`solver_comparison.md`, `figures/amgx_3d_solver_error.png`, `lab_LU_robust.py`（最近 commit a0b6ed5/b95f020/5207c3d 加的 LU 真解 fallback）

### C.2 lab32_dump (2M, lab Xeon 32-rank dump) — 本次重点

| 字段 | 值 |
|---|---|
| mesh | 同 LPBF_crosscheck (80×320×80 = 2M)，因为 case 是从 LPBF_crosscheck 拷贝的 |
| 跑在哪 | lab Xeon HR54WV2，32 MPI ranks `mpirun --oversubscribe` |
| matrixDumper 版本 | **patched (commit 47d4863)，能正确处理 processor patches** |
| 计划时间步 (per `run_xeon_32rank_dump.md`) | melting 320/380/410 ns + evaporation 700/900/1060 ns，共 6 步 |
| 实际 dump 状态 | 🟡 **未在本机；需 SSH 到 lab Xeon 确认实际跑到哪** |
| 自洽性（用户报） | median ‖A·x_final - b‖/‖b‖ = **2.4e-9** ✓ |
| **数据所在** | `manyxu@HR54WV2:~/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_lab32_dump/processor*/postProcessing/matrices/` |
| **JSON 结果文件** | ❌ 还没生成，是个 orphan — `prepare_lab32_plot_data.py`（本次新写的脚本）跑完后会生成 |

**待办**: 跑 `prepare_lab32_plot_data.py` 生成 melting + evaporation 各一个 timestep 的 npz（含 AMGx truth + AMGx tol=1e-8 + 计时统计）。

### C.3 spot_melt_150W (500K, lab Xeon OF baseline)

| 字段 | 值 |
|---|---|
| mesh | 50×200×50 = **500,000** cells（**注意**：和学长 512K 80³ 不一样） |
| 跑在哪 | lab Xeon HR54WV2 |
| solver 用过 | **只有 OF DICPCG**，没 dump 矩阵给 AMGx |

**OF MPI 32-rank wall (`of_lab_xeon_500K_pd_timings.txt`)**：

| 统计 | 值 |
|---|---|
| n_runs | 234 (78 时间步 × 3 PISO correctors, pd_corr0) |
| **median wall** | **49.57 ms** |
| min | 6.68 ms |
| max | 210.28 ms |
| mean | 17.83 ms |
| 注 | 分布右偏（warmup + PISO 内迭代交替） |

**OF MPI scaling (`OPENFOAM_SCALING_20260503.md`)**：

| N (ranks) | n_steps | pd_corr0 ms | speedup |
|---|---|---|---|
| 1 | 51 | 567.4 | 1× |
| 2 | 21 | 296.4 | 1.91× |
| 4 | 51 | 210.2 | 2.70× |
| 8 | 51 | 144.0 | 3.94× |
| 16 | 51 | 113.4 | 5.00× |
| 24 | 78 | 61.8 | 9.18× |
| 28 | 78 | 50.5 | 11.24× |
| **32** | 78 | **44.1** | **12.86×** (efficiency 40%) |
| 56 (SMT 全用) | 10 (cold-start crash) | 157.6 | n/a |

> ⚠ **44.1 vs 49.57** 都来自 lab Xeon N=32 spot_melt_500K，是两次不同跑的不同 timestep 窗口，差 11%。两个数都合法，但发表/对比时**必须明确指出来源** — 推荐统一用 49.57 ms（含 PISO inner correctors，更接近真实负载）。

### C.4 LPBF_sanity (2K) + dumper_pipeline (8K) — 早期 dev 管道测试

| 字段 | 值 |
|---|---|
| mesh | LPBF_sanity = 16×8×16 = 2048; dumper_pipeline = 16×32×16 = 8192 |
| 跑在哪 | dev RTX 3050 |
| 用途 | AMGx vs scipy SuperLU truth 精度验证（小 case 才能跑 SuperLU） |
| dump 数 | LPBF_sanity 12 + dumper_pipeline 36 = **48 总** |
| 状态 | ✅ 全部 |

| 配置 | iter | rel_resid | rel_vs_truth | source |
|---|---|---|---|---|
| AMGx tol=1e-12 (no IR) | 18-24 (pd) | 1e-13~1e-15 | n/a | precision_results.json |
| **AMGx tol=1e-12 + 1 IR** | 18-24 + IR | 3-4e-16 | **2.89e-15 max, 1.83e-15 median, 5.72e-16 min** | precision_results_ir1.json **(canonical)** |
| AMGx tol=1e-12 + 1 IR v2 | 同上 | n/a | n/a | precision_results_ir1_v2.json **(orphan, 重跑没新发现)** |
| AMGx tol=1e-14 | 同上 | 1e-15 | 比 1e-12 没改进 | precision_results_tol1e14.json **(orphan)** |

**这是 "AMGx 在机器精度量级" 这个论断的全部数据基础** — 但**只在 8K cell 验证过**，没有在 512K 或 2M 验证（因为 SuperLU OOM）。来源：`AMGX_PRECISION_20260504.md`，memory `amgx_precision_truth.md`。

### C.5 v3.2 spot_melt_npz (1M dev) — 历史，**数据不在当前 working tree**

| 字段 | 值 |
|---|---|
| mesh | 80×160×80 = 1,024,000 |
| 跑在哪 | dev (4-core OF + RTX 3050 AMGx) |
| 矩阵数 | 50 pd + 100 T = 150 |
| 文件位置 | `dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/`（**已不在 repo**，应该是清理 dev 磁盘时删了） |
| 状态 | ✅ 跑过，✅ 论文级数据，⚠ 重跑不出来（数据没了） |

**v3.2 报告的关键数字**：

| 项 | 值 | 来源 |
|---|---|---|
| OF wall (1 core dev) | 2.57 s | OPENFOAM_CROSSCHECK_20260427_v3.2.md |
| AMGx wall (RTX 3050) | 0.30 s | 同上 |
| 加速 | **8.6×** | 同上 |
| rel_AMGx_vs_OF | 2.5e-6 | 同上（**这个数字是 OF tol=1e-8 截断误差，不是 AMGx 误差**，2026-05-04 才看清） |

**这是大家口口相传的 "AMGx 比 OF 快 8.6×"，但是基于 dev 单核 OF + dev RTX 3050 AMGx，不是公平对比**。后续 lab benchmark（lab Xeon 32 ranks vs lab 5060 GPU）反而是 OF 快 2.4× — 因为 OF 用了 32 cores，AMGx 用了 1 GPU，且 mesh 是 500K 不是 1M。

### C.6 Phase 4 synthetic Poisson (16³-128³) — 历史档

| mesh | nnz | p2 (cuSPARSE DILU-PCG) | p3 (Multicolor) | p4 (AMGx) |
|---|---|---|---|---|
| 16³ = 4096 | 27k | n/a | n/a | n/a |
| 32³ = 32k | 223k | n/a | n/a | n/a |
| 64³ = 262k | 1.81M | n/a | n/a | n/a |
| **128³ = 2.1M** | 14.58M | 28.47 s (186 iter) | 35.0 s (296 iter) | **1.29 s (43 iter)** |

加速 p4/p2 = **22.1×**。这是 2026-04-21 的 Phase 4 closeout 数据，纯合成 Poisson，跟 LPBF 无关，frozen。来源：`scaling_results.json`, `docs/benchmark/phase4_amgx_report.md`。

---

## D. 主索引表 — 一张表查所有 solver run

> 时间排序 (从旧到新)。每行是一个独立的 solver run / benchmark 输出。

| # | 日期 | 数据 | mesh | 机器 | solver | tol | iter | wall | rel_resid | rel_vs_truth | converged | source 文件 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-21 | synthetic 128³ Poisson | 2.1M | dev RTX 3050 | cuSPARSE DILU-PCG | n/a | 186 | 28.47 s | n/a | n/a | ✅ | scaling_results.json |
| 2 | 2026-04-21 | 同 | 2.1M | dev RTX 3050 | Multicolor DILU | n/a | 296 | 35.0 s | n/a | n/a | ✅ | 同 |
| 3 | 2026-04-21 | 同 | 2.1M | dev RTX 3050 | AMGx CLASSICAL | n/a | 43 | **1.29 s** | n/a | n/a | ✅ | 同 |
| 4 | 2026-04-26 | spot_melt_npz | 1M | dev RTX 3050 | AMGx vs OF | 1e-12 | n/a | AMGx 0.30 s, OF 2.57 s | n/a | rel_OF=2.5e-6 (后被推翻) | ✅ | OPENFOAM_CROSSCHECK_20260427_v3.2.md |
| 5 | 2026-05-03 | spot_melt_150W | 500K | lab Xeon (N=1..32) | OF DICPCG | 1e-8 | n/a | 567ms→44.1ms | 1e-8 | n/a | ✅ | OPENFOAM_SCALING_20260503.md |
| 6 | 2026-05-04 | LPBF_sanity 2K + dumper 8K | 2K, 8K | dev RTX 3050 | AMGx (no IR) | 1e-12 | 18-24 | 30-100 ms | 1e-13 | vs scipy: 2.89e-15 max | ✅ | precision_results.json |
| 7 | 2026-05-04 | 同 | 同 | dev RTX 3050 | **AMGx + 1 IR** | 1e-12 | 18-24+IR | 30-100 ms | 3-4e-16 | **vs scipy: 2.89e-15** | ✅ | precision_results_ir1.json (canonical) |
| 8 | 2026-05-04 | 同 | 同 | dev RTX 3050 | AMGx + 1 IR v2 | 1e-12 | 同 | 同 | 同 | 同 | ✅ | precision_results_ir1_v2.json (**orphan**) |
| 9 | 2026-05-04 | 同 | 同 | dev RTX 3050 | AMGx tol=1e-14 | 1e-14 | 同 | 同 | 1e-15 | 没改进 | ✅ | precision_results_tol1e14.json (**orphan**) |
| 10 | 2026-05-04 | 学长 Initial_Period 21 bundles | 512K | dev RTX 3050 | AMGx tol=1e-12, 3 modes | 1e-12 | 26 | 540 ms total / 200 solve | n/a | n/a | ✅ | senior_precision_results_notruth.json |
| 11 | 2026-05-04 | 同 | 512K | dev RTX 3050 | AMGx fresh / amortized / +IR | 1e-12 | 26 | 540 ms / 540 ms / 320 ms | n/a | rel_xref=2.5% | ✅ | bench_amgx_vs_of_n32_results.json |
| 12 | 2026-05-04 | 同 | 512K | dev RTX 3050 | AMGx + replica 3-way | 1e-12 | AMGx=26, replica=195 | AMGx 1324 ms, replica 11.6 s, OF 44.1 ms | n/a | n/a | ✅ | threeway_senior_results.json (**superseded by lab numbers**) |
| 13 | 2026-05-04 | OF spot_melt_150W (32-rank, 全跑) | 500K | lab Xeon | OF DICPCG | 1e-8 | n/a | **49.57 ms median**, 234 timings | n/a | n/a | ✅ | of_lab_xeon_500K_pd_timings.txt |
| 14 | 2026-05-05 | 学长 Initial_Period 21 bundles | 512K | dev RTX 3050 | AMGx tight + replica tight | 1e-12 | n/a | n/a | AMGx 1.7e-15, replica 2.1e-13 | rel_AMGx_vs_replica = 6e-8 to 3e-7 | ✅ | byte_match_proof_results.json (**orphan finding**) |
| 15 | 2026-05-05 | 同 | 512K | dev RTX 3050 | AMGx loose + replica loose | 1e-8 | AMGx=17, replica=137 | n/a | AMGx 8e-9, replica 1.9e-9 | rel_AMGx_vs_replica = 1.1e-4 ⚠ | ✅ | loose_match_results.json (**orphan finding**) |
| 16 | 2026-05-05 | 同 | 512K | **lab 5060 RTX 5060** | **AMGx 9 tol-configs × 21 bundles** | 1e-14 ~ 1e-4, IR=0/1 | 9-26 | 86-454 ms | 7.96e-5 ~ 2.17e-15 | n/a | ✅ | **tol_sweep_results.json (CANONICAL lab GPU benchmark)** |
| 17 | 2026-05-06 | 学长 Melting 3 + Evaporation 3 | 512K | lab Xeon | LSMR truth attempt | atol/btol=1e-12 | 5000 (cap) | 49-239 s | 0.097 (broken) | rel_OF=18 ⚠ | ❌ 5/6 | **lab_PCG_truth_results.json (orphan, never reported)** |
| 18 | 2026-05-07 | LPBF_crosscheck | 2M | dev RTX 3050 | AMGx tol=1e-12 (truth) + tol=1e-8 + OF (from dump) | 1e-12 / 1e-8 | 26 / 17 | n/a | 2.6e-13 / 8e-9 | OF rel 9e-6, AMGx 1e-8 rel 2.6e-6 | ✅ | solver_comparison.md, figures/amgx_3d_solver_error.png |
| 19 | 2026-05-07 | 学长 Initial_Period | 512K | lab Xeon (CPU LU) | **CHOLMOD / SuperLU / MUMPS / LSMR** 4-tier fallback | direct | n/a | n/a | n/a | (作真解给 PCG 比) | (depends on tier) | lab_LU_robust.py recent commits a0b6ed5/b95f020/5207c3d (**没有 results JSON 在 repo**) |
| 20 | 2026-05-07 | lab32_dump (我们 32-rank dump) | 2M | lab Xeon | matrixDumper patched dump 自洽性 | n/a | n/a | n/a | **2.4e-9 median (用户报)** | n/a | ✅ | (no JSON; orphan; 待 prepare_lab32_plot_data.py 输出) |

---

## E. 矛盾与孤儿 — 需要清理

### E.1 矛盾的 wall time 数字

| 表述 | 出处 | 实际数据 | 备注 |
|---|---|---|---|
| "OF lab N=32 wall = 44.1 ms" | OPENFOAM_SCALING_20260503.md, bench_amgx_vs_of_n32_results.json | scaling 跑的 51 timesteps median | 较短 timestep window |
| "OF lab N=32 wall = 49.57 ms" | of_lab_xeon_500K_pd_timings.txt, solver_comparison.md (最新) | 78 timesteps × 3 corr = 234 pd_corr0 median | 包含 PISO inner correctors，更接近真实负载；**推荐用这个** |
| "AMGx dev wall = 540 ms" (mode_amortized solve) | bench_amgx_vs_of_n32_results.json | dev RTX 3050 | 只算 solve_ms |
| "AMGx dev wall = 1324 ms" (amortized total) | threeway_senior_results.json | dev RTX 3050, 同一天 | 把 setup 也算进去；**和上面口径不一致** |
| "AMGx lab 5060 = 190 ms" tol=1e-12 | tol_sweep_results.json | **canonical lab GPU 数字** | 只算 solve_ms |
| "AMGx 8.6× faster than OF" | OPENFOAM_CROSSCHECK_20260427_v3.2.md | dev 1-core OF vs dev RTX 3050 AMGx | **不公平，dev OF 单核**，不是 lab 32-core |

### E.2 矛盾的 mesh 规模 & 数据集

| 不严谨说法 | 实际 |
|---|---|
| "OF vs AMGx, 500K cells" | OF 是 50×200×50 = **500,000** spot_melt_150W；AMGx 是 80³ = **512,000** 学长 Initial_Period — **不是同一矩阵** |
| "AMGx 比 OF 准 4×" | 是 LPBF_crosscheck 2M dev dump 上的对比，**不在学长 512K 上验证过** |
| "AMGx 在机器精度" | 是 LPBF_sanity 2K + dumper 8K 验证的 (vs SuperLU)，**没在 512K / 2M 验证过** |

### E.3 孤儿数据（生成了但没写进任何报告）

| JSON | 价值 finding | 应该去哪 |
|---|---|---|
| `loose_match_results.json` | rel_AMGx_vs_replica @ tol=1e-8 = 1.1e-4 — AMGx 和 PCG-DILU 在松 tol 下分歧到 1e-4 量级 | 应该和 byte_match_proof 合并写一个 "AMGx vs DILU-PCG agreement" 报告 |
| `byte_match_proof_results.json` | rel_AMGx_vs_replica @ tol=1e-12 = 6e-8 to 3e-7 — 收敛后两个 solver 一致到 7-8 位 | 同上 |
| `lab_PCG_truth_results.json` | **学长 Melting + Evaporation 不可解**，6 行决定性证据 | **应该单独写一个文档**，目前是最大的 orphan |
| `precision_results_ir1_v2.json` | 重跑 ir1，没新发现 | 删除或 archive |
| `precision_results_tol1e14.json` | tol=1e-14 比 1e-12 没收益 | 在 AMGX_PRECISION 里加一段说明，然后删 |
| `lab_LU_robust.py` 输出 | 4-tier fallback 真解 | 应该跑完后存 JSON，目前只在 commit msg 里 |
| **lab32_dump** | matrixDumper.H 修好后的干净 dump，rel=2.4e-9 | **本次 prepare_lab32_plot_data.py 会修这个 orphan** |

### E.4 重复 / 旧版的 markdown

| 系列 | 哪个是 canonical |
|---|---|
| OPENFOAM_CROSSCHECK_20260426 / 20260427 / v3 / v3.1 / v3.2 | **v3.2 是最新但已经 historical**（数据丢了） |
| THREEWAY_BENCH_DRAFT vs THREEWAY_BENCH_FINAL | **FINAL** |
| solver_comparison.md (2026-05-07) | **当前最新**，应该取代 THREEWAY_BENCH_FINAL |

---

## F. 还在 lab 机器上、本地 repo 没有的数据

⚠ **下面这些需要 SSH 到 lab 上确认**：

### F.1 lab 5060 (`yzk@<5060>`)

- [ ] `~/DILU-Research/dilu/amgx/bench/lab_deploy/results/amgx_5060_bench.log` — THREEWAY_BENCH_FINAL 引用，但本地没这个文件
- [ ] `amgx_5060_precision.json`, `amgx_5060_summary.md` — 同上
- [ ] AMGx build artifacts under `~/local/amgx/` (sm_120 binary, NVTX header workaround)
- [ ] 可能有 `tol_sweep_results.json` 的原 stdout log

### F.2 lab Xeon `manyxu@HR54WV2`

- [ ] **整个 32-rank LPBF_lab32_dump 的 `processor*/postProcessing/matrices/` 原始数据** — 这是本次画图要用的，需要 rsync 到 5060
- [ ] `lab_LU_robust.py` 运行结果（CHOLMOD/MUMPS 真解 npy）— 最近三个 commit (a0b6ed5/b95f020/5207c3d) 加的 fallback 代码，但**结果文件没传回 dev**
- [ ] `~/cases/spot_melt_150W/` OF case 完整目录 + scaling 跑出的 N=1..56 timing CSVs (`scaling_results_lab/timing_*.csv`)
- [ ] 学长 DICPCG_Benchmark_Data 的原始 CSV（dev 只有 npz）

### F.3 dev 旧数据（已清理）

- [ ] v3.2 的 `dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/` 1M cell npz — **被清理了**，重跑不出来；如果要复盘 8.6× 那个数字，必须重跑 OF + AMGx

---

## G. 建议接下来的动作（按优先级）

1. **lab32_dump 跑 prepare 脚本** → 生成 melting + evap 两个 JSON + 3D 图（本次工作）。
2. **写一个独立的 "Senior 数据可解性诊断" 报告**，把 lab_PCG_truth_results.json 的 6 行证据 + matrixDumper.H 的修复（commit 47d4863）整合，发给学长。
3. **合并 byte_match_proof + loose_match 写一个 "AMGx vs DILU-PCG agreement" 报告**（一页就够），处理 1.1e-4 的孤儿 finding。
4. **archive** `OPENFOAM_CROSSCHECK_20260426..v3.1.md`、`THREEWAY_BENCH_DRAFT.md`、`precision_results_ir1_v2.json`、`precision_results_tol1e14.json` 到 `docs/archive/`。
5. **重命名/合并** `solver_comparison.md` → `THREEWAY_BENCH_FINAL_v2.md`（统一用新的 lab 5060 + 49.57 ms baseline）。

---

## H. 如何用这个文档

- 找 "某个数字哪来的" → 看 D 章主索引表
- 找 "学长某组数据" → B 章
- 找 "我们某次 dump" → C 章
- "为什么两个数字打架" → E.1 / E.2
- "lab 上有什么本地没的" → F 章
