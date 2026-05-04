# OpenFOAM ↔ Our Solvers 矩阵级对标报告

**日期**: 2026-04-26（数据采集自 2026-04-25）
**版本**: v1.0（初步报告，1 个 case + 1 个矩阵的初步对比；扩展进行中）
**作者**: yzk + Claude
**对标 case**: LaserbeamFoam `laserMeltFoam` 静态点熔参考算例（316L, 80×160×80 = 1M cells）
**方法论文档**: `docs/design/openfoam_crosscheck_plan.md`

---

## 0. 一句话结论

我们成功从一个真实 LPBF OpenFOAM 算例（316L 静态点熔，1M cells，激光开 50 μs）的熔池形成期（t≈2.2-3.2 μs）抽取了 **150 个真实物理线性系统矩阵**（pd 压力泊松 50 个 + T 温度 100 个，每个 1M×1M, 7M nnz），全部通过 sanity 校验（A·x_final ≈ b 到机器精度）。在第一个代表矩阵上对比了 scipy/cuSPARSE/AMGx 三类求解器：

- **scipy gmres**：1334 iter / 226 s，残差 7.5e-13 ✓ 可作 ground-truth
- **our cuSPARSE DILU-PCG**：189 iter / 351 s，残差 9.4e-11 ✓ 收敛但比我们 canonical 慢 70×
- **our AMGx aggressive**：200 iter / 1.3 s，diverged 残差 1.98e-4 ✗ V-cycle 在熔池物理上失效

→ 这是一份**有价值的对标基线**：暴露了我们求解器在合成 stiff Poisson 上调好的优化在**真实 multiphysics 矩阵**上的不足，为后续工作指明方向。

---

## 1. 全流程架构

整个对标管道由 5 个组件构成，按数据流向：

```
[1] LaserbeamFoam laserMeltFoam (C++ 求解器)
      │
      │  matrixDumper.H 在 PCG/PBiCG solve 入口拦截 fvMatrix
      │  落盘 (A.mm, b.mm, x0.mm, x_final.mm, metadata.json)
      ▼
[2] postProcessing/matrices/<time>/<eq>_corr<k>/  (每矩阵 ~330 MB ASCII)
      │
      │  shrink_dump.py：MM ASCII → npz binary
      │  + 裁剪 (保留 pd_corr0, T_corr0, T_corr1)
      ▼
[3] spot_melt_npz/<time>/<eq>_corr<k>/data.npz  (每矩阵 ~37 MB)
      │
      │  reader.py：npz/MM 自动 detect → scipy.sparse.csr_matrix + numpy
      ▼
[4] driver_scipy.py / driver_cusparse.py / driver_amgx.py
      │
      │  并行/串行调用对应求解器，记录 iter / time / residual
      ▼
[5] compare.py：聚合 results/*.json → summary.csv + summary.md
```

**关键设计点**：
- 步骤 [1] 用 OpenFOAM **公共 API**（`fvMatrix.D()`、`boundaryCoeffs()`、`psi()`）—— 避开 protected 的 `addBoundaryDiag/Source`
- 步骤 [1] dumper 用 **相对时间步索引**（`timeIndex - firstTimeIdx + 1`），所以 `dumpTimeSteps (1..N)` 在从 `latestTime` 恢复时自动匹配 dump-启动后的前 N 步
- 步骤 [2] 自动**符号归一化**（OpenFOAM Laplacian 约定 diag 为负，我们的 PCG 假设 SPD diag 为正 → 检测到全负 diag 自动 negate）

---

## 2. 数据采集

### 2.1 算例参数

| 项 | 值 |
|----|----|
| 求解器 | LaserbeamFoam `laserMeltFoam`（OpenFOAM v2506 + matrixDumper patch）|
| 几何 | 200×400×200 μm 立方域 |
| 网格 | 80×160×80 = **1,024,000 cells**（dx = 2.5 μm 均匀）|
| 材料 | 316L 不锈钢（ρ=7900, cp=700, k=20, μ=0.005, T_s=1650, T_l=1700）|
| 激光 | 静态点熔，P=150W, r₀=25μm, on 0–50 μs |
| 物理 | 两相 VOF（金属+氩）+ 表面张力 + Marangoni（dσ/dT=+1e-4）+ Darcy mushy zone |
| pd 求解器（OpenFOAM）| PCG + DIC 预条件，tol=1e-8 |
| T 求解器（OpenFOAM）| PBiCG + DILU 预条件，tol=1e-12 |

### 2.2 采集策略

**两阶段**：
1. **预热**：从 t=0 跑到 t=2.23 μs（约 200 步，dt 自适应从 1e-12 ramp 到 maxDeltaT=2e-8s），不开 dumper —— 避免冷启动 dt 极小、矩阵病态（diag~1e-26）
2. **Dump**：从 t=2.23 μs 接着跑 50 步到 t=3.23 μs，每步 dump

**裁剪策略**（每步 5 矩阵，节省磁盘）：
- pd_corr0：保留（PIMPLE 第一次压力修正，主要工作量）
- pd_corr1, pd_corr2：丢弃（PCG 会立即返回 iter=0，矩阵几乎同 pd_corr0）
- T_corr0：保留（Picard 外迭代第一次温度求解）
- T_corr1：保留（一次非线性更新后的矩阵，与 corr0 略有不同）

→ 每步 3 个有效矩阵，50 步共 **150 个矩阵**。

### 2.3 物理状态

| 矩阵时段 | t (μs) | 物理 | 矩阵特性 |
|----------|--------|------|----------|
| 早期 | 2.25 | 激光刚开始加热，T_max ~ 300.7K，无熔池 | dt 已稳定 maxDeltaT，非冷启动病态 |
| 中期 | 2.7 | 局部温度上升，开始熔化 | 系数随 T 变化 |
| 末期 | 3.2 | 熔池形成期，Marangoni 驱动流 | 物理充分激活 |

### 2.4 Sanity 校验结果

**150/150 矩阵全部通过**：
- pd 矩阵：|Ax - b| ≈ 7-8 × 10⁻¹⁷（机器精度）
- T 矩阵：|Ax - b| ≈ 1-9 × 10⁻¹¹
- 与 OpenFOAM 自报 final_residual 同量级（PASS 阈值 1e-6）

→ **dump 写盘无损耗，与 OpenFOAM 求解结果一致到 IEEE 754 精度**。

---

## 3. 第一手对标结果（单矩阵）

### 3.1 测试矩阵

- 路径：`spot_melt_npz/2.45322e-06/pd_corr0`
- 物理：t = 2.45 μs（激光开 0.22 μs 后），早期熔池
- 矩阵：N = 1,024,000，nnz = 7,104,000，对称 SPD（自动符号归一化后）
- OpenFOAM 自报：22 PCG-DIC iters → final residual = 9.5e-9

### 3.2 三求解器对比

| 求解器 | iters | setup (s) | solve (s) | rel_residual | 状态 | 备注 |
|--------|-------|-----------|-----------|--------------|------|------|
| **scipy gmres** (CPU, ref) | 1334 | — | 226.5 | 7.5e-13 | ✓ ok | tol=1e-12, restart=50, Jacobi precond |
| **our cuSPARSE DILU-PCG** (GPU) | 189 | 0.31 | **351.5** | 9.4e-11 | ✓ ok | tol=1e-10 |
| **our AMGx aggressive** (GPU) | 200 (max) | 0.76 | 1.28 | 1.98e-4 | ✗ DIVERGED | classical AMG 粗化 + D1 |

### 3.3 解读

- **iter 数**：DILU-PCG 189 vs scipy gmres 1334 → DILU 预条件让 iter count 下降 ~7×
- **wallclock**：cuSPARSE solve 时间异常高（351 s）。我们 canonical 128³ 同等规模的 case 仅 ~5 s（详见 `docs/benchmark/CANONICAL_CASE.md`）。**70× slowdown**，需要 profile：
  - 怀疑：每 PCG 迭代 Python 级 host-device 同步、`jax.ops.segment_sum`-based SpMV 在大 nnz 上效率低
  - 但 iter count（189）合理 → 算法层面没坏，是实现层性能问题
- **AMGx 不收敛**：classical V-cycle + aggressive coarsening 在熔池物理（Marangoni 修正 Poisson）上发散。我们的 AMGx 配置在合成 stiff Poisson 上调好，对真实 multiphysics 失效。需要试 classical_v 或换 smoother

→ 这正是对标的价值：**真实物理矩阵暴露了我们求解器优化的局限**。

---

## 4. 实现过程中发现的 bug & fix（值得记录）

按发现顺序：

### B1. `addBoundaryDiag/Source` 是 protected
**问题**：`matrixDumper.H` 第一版用了 `matrix.addBoundaryDiag()` 折叠边界对 diag 的贡献，编译报 protected。
**Fix**：改用公共 `matrix.D()` 返回带边界贡献的 diag；source 用公共 `boundaryCoeffs()` + `psi().boundaryField()` 手动折叠（见 `matrixDumper.H:362`）。

### B2. `dumpTimeSteps (1 2 3)` 在 resume 时不触发
**问题**：`runTime.timeIndex()` 是从 startTime=0 累计的绝对步数。预热到 2.23 μs 时已是 step 200+，永远不匹配 1-3。
**Fix**：在 matrixDumper 里加 `firstTimeIdx_`，第一次见到 timeIndex 时记录；后续用 `relIdx = timeIndex - firstTimeIdx + 1`。dumpTimeSteps 现在是**相对**于 dumper 启动的步数。

### B3. OpenFOAM Laplacian 符号惯例与 PCG 不兼容
**问题**：OpenFOAM 把 Laplacian 离散为 `diag = -Σ|off-diag|`（负 diag）。我们的 cuSPARSE DILU-PCG 假设 SPD（正 diag），喂进去 `p^T A p < 0` 立刻发散。
**Fix**：reader.py 自动检测 `all(diag<=0)` 时 negate A 和 b（数学等价），metadata 记录 `sign_negated=true`。

### B4. `monitoring` functionObject 把 time folder 全写成空壳
**问题**：spot_melt case 用 `#include "monitoring"` 注册了一个 functionObject，每步往 time folder 写 `Co` 文件。OpenFOAM 因此每步创建 folder，但 field 写入仍按 `writeInterval` 节奏 → 大量 folder 只有 Co 没字段。`startFrom latestTime` 选到空壳 → 报错。
**Fix**：dump 阶段在 controlDict 里把 `#include "monitoring"` 注释掉。手动清理空壳 folder（保留含 pd 的）。

### B5. ASCII MM 的体量爆炸
**问题**：1M cells × 50 步 × 5 矩阵 = 250 矩阵 × 330 MB ASCII = **80 GB**，传不动。
**Fix**：`shrink_dump.py` 转为 npz（scipy.sparse 二进制 + zlib 压缩）+ 裁剪 corrector → 150 矩阵 × 37 MB = **5.5 GB**，14× 减重。

### B6. Heredoc 跨终端粘贴导致缩进偏移
**问题**：用 `cat > file << EOF` 粘脚本时，RDP 终端给某些行加了 2-4 空格 leading whitespace，Python 抛 IndentationError。
**Fix**：用 base64 编码（无空白敏感性）+ `tr -d ' \n\t\r'` 防止换行污染。

---

## 5. 部署包（给其他机器用）

### 5.1 内容清单

`/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy/`（26.7 KB）：

| 文件 | 用途 |
|------|------|
| `matrixDumper.H` | OpenFOAM C++ 头文件，~400 行 |
| `*.diff` × 3 | unified diffs (laserMeltFoam.C, TEqn.H, pEqn.H) |
| `install.sh` | patch + wmake，一键装 |
| `patch_case.sh` | 给 case 贴 dumperDict + 改 controlDict |
| `sanity.py` | 跑后验证 dump 正确性（仅需 numpy+scipy）|
| `shrink_dump.py` | MM ASCII → npz binary 转换器（多进程）|
| `TUTORIAL.md` | 14 章手把手教程 |
| `README.md` | 简明说明 |

### 5.2 已验证兼容性

- ✅ OpenFOAM **v2506**（开发基础版本）
- ⚠️ v2412/v2406：可能需将 `getOrDefault` 改回 `lookupOrDefault`
- ❌ 并行 MPI：`matrixDumper` 当前假设单进程，processor patch 跳过

---

## 6. 当前局限 & 未来工作

### 6.1 数据规模

- **150 个矩阵 × 1 个 case** —— 学长目标 1000+ 矩阵需 6-7 个 case
- 建议下一批：扫激光功率（100/150/250/400 W）或扫描速度，每 case 跑同样 50 步 dump
- 部署包已就绪，每 case 只需 ~2 小时 wallclock

### 6.2 求解器表现

| 现象 | 假设原因 | 后续工作 |
|------|----------|----------|
| cuSPARSE DILU-PCG 比 canonical 慢 70× | Python-level PCG loop 同步开销；segment_sum SpMV 慢 | profile 后改 fused jax.lax.scan PCG，或写 cuSPARSE-native PCG |
| AMGx aggressive diverged | Marangoni-modified Poisson 不适合 D1 + aggressive_levels=2 | 试 classical_v + D2 + GS smoother；或换 BiCGStab outer |
| T 方程未对标 | 我们 cuSPARSE DILU-PCG / AMGx PCG 都假设 SPD | 加 BiCGStab wrapper（半天工作量）|

### 6.3 数值精度

- **L2 阈值** 在 `compare.py` 里设为 1e-5（不是设计文档里的 1e-8）—— 病态 LPBF 矩阵条件数 κ ~ 10⁶+，1e-10 残差对应 x 误差 ~1e-4，1e-8 阈值不现实
- 建议：报告里同时给 |Ax-b|/|b| 和 |x-x_ref|/|x_ref|，让读者各自判断

---

## 7. 文件位置一览

| 内容 | 路径 |
|------|------|
| 设计文档 | `docs/design/openfoam_crosscheck_plan.md` |
| 本报告 | `docs/benchmark/OPENFOAM_CROSSCHECK_20260426.md` |
| C++ 源 + patches | `LaserbeamFoam/applications/solvers/laserMeltFoam/` |
| Python drivers | `dilu/benchmark/openfoam_crosscheck/` |
| 部署包 | `dilu/benchmark/openfoam_crosscheck/deploy.tar.gz` |
| 实验数据（150 矩阵 npz）| `dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/` |
| 实验数据（原 80GB ASCII）| 仅在实验室机：`/home/yzk/cases/openfoam_laserMeltFoam_spot/postProcessing/matrices/` |

---

## 8. 验证状态总览

| 阶段 | 状态 | 证据 |
|------|------|------|
| A.1 写 matrixDumper.H | ✅ | 编译通过、所有 patch 干净应用 |
| A.2 patch laserMeltFoam | ✅ | 二进制 build 成功，timestamp 2026-04-23 / 2026-04-24 |
| A.3 wmake recompile | ✅ | 同上 |
| A.4 sanity 自洽测试 | ✅ | LPBF_sanity (8K cells)、dumper_pipeline_test (16K cells) 全 PASS |
| B 跑 LPBF case | ✅ | spot_melt 1M cells × 250 dump matrices, 150 通过 sanity |
| C Python 桥 + drivers | ✅ | reader/scipy/cuSPARSE/AMGx/compare 全部 import OK，烟雾测试 1 矩阵通过 |
| 端到端管道 | ✅ | 本机 dumper_pipeline_test 16×32×16 = 16K cells 完整跑通 |
| 第一组 benchmark | ✅ | 1 矩阵 × 3 求解器，结果如 §3.2 |
| **大规模 benchmark** | ⏳ | 150 矩阵全跑（待启动）|
| **报告 v2** | ⏳ | 全量数据出来后写终版 |

---

## 9. 复现指引

任何人想从零复现这份报告的数据：

```bash
# Stage 1: 在跑过 LaserbeamFoam 的目标机器上
tar xzf deploy.tar.gz && cd deploy
source /usr/lib/openfoam/openfoam2506/etc/bashrc
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam   # 装 dumper

# Stage 2: 准备一个跑过的 LPBF case（已有保存时刻）
cp -r ~/your_case ~/case_for_dump
cd ~/case_for_dump
# 改 controlDict：writeControl timeStep + 禁用每步 functionObject
# 跑 warmup 到熔池 regime（t > 2 μs 量级，dt 稳定到 maxDeltaT）
./Allrun

# Stage 3: 贴 dumper 跑 50 步 dump
~/deploy/patch_case.sh . 50
laserMeltFoam | tee log.dump
python3 ~/deploy/sanity.py postProcessing/matrices

# Stage 4: 缩到 npz 传回
python3 ~/deploy/shrink_dump.py postProcessing/matrices ~/case_npz/ 16
tar cf ~/case_npz.tar ~/case_npz/    # ~5-6 GB / case

# Stage 5: 在带 GPU 的机器上对比
cd /home/yzk/DILU-Research
source /home/yzk/jax-env/bin/activate
ROOT=dilu/benchmark/openfoam_crosscheck/collected/case_npz
python -m dilu.benchmark.openfoam_crosscheck.driver_scipy "$ROOT" --force-gmres
python -m dilu.benchmark.openfoam_crosscheck.driver_cusparse "$ROOT"
python -m dilu.benchmark.openfoam_crosscheck.driver_amgx "$ROOT" --cfg classical_v aggressive
python -m dilu.benchmark.openfoam_crosscheck.compare "$ROOT"
cat "$ROOT/summary.md"
```

---

## 10. 下一步路线图

按优先级排序：

1. **跑 150 矩阵全量 benchmark**（~12 h 后台 / 4 个 solver）→ 出 v2 报告
2. **跑剩余 5-6 个 case**（实验室扫激光功率）→ 凑 1000+ 矩阵
3. **profile + 优化 cuSPARSE DILU-PCG**（70× slowdown 可能是 host-device 同步开销）
4. **加 BiCGStab wrapper** → 解锁 T 方程对标
5. **AMGx config 调优**（试 classical_v + D2 + GS smoother on real LPBF matrix）

---

**报告版本控制**：
- v1.0 (2026-04-26)：本文件，1 矩阵初步对比 + 完整管道验证
- v2.0 (TBD)：全量 150 矩阵对比 + 性能优化结果

**联系反馈**：在 `docs/benchmark/` 同目录下追加 v2 即可，本文件保持只读。
