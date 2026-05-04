# OpenFOAM ↔ Our Solvers 矩阵级对标报告 v2.1

**日期**: 2026-04-27
**版本**: v2.1（修订版：用 vs-OpenFOAM 真精度替代自洽残差作主指标）
**作者**: yzk + Claude
**对标 case**: LaserbeamFoam `laserMeltFoam` 静态点熔参考算例（316L, 80×160×80 = 1M cells）
**前置版本**:
- `OPENFOAM_CROSSCHECK_20260426.md` v1.0（1 矩阵初步报告）
- v2.0（2026-04-27 早版，**已撤销**：rel_residual 仅自洽校验，被误读为 vs-OF 精度，造成 AMGx 结果误导）
**方法学文档**: `docs/Matrix_Output_via_Solver_Hook.md`

---

## 0. 一句话结论

在 50 个真实 LPBF 熔池形成期 pd 矩阵（1M cells, 7M nnz, t∈[2.25, 3.23] μs）上对比三种 SPD 求解器，**用 `‖x_solver − x_OpenFOAM‖∞ / ‖x_OpenFOAM‖∞` 作为主精度指标**：

| Solver | 收敛 | iter med | solve med (s) | **rel_vs_OF med** | 物理正确性 |
|--------|------|----------|---------------|-------------------|-----------|
| **our cuSPARSE DILU-PCG** | 50/50 | 189 | 15.34 | **4.38e-04** | ✅ 与 OpenFOAM 解一致到该量级 |
| **our Multicolor DILU-PCG** | 50/50 | 290 | 19.21 | **5.51e-04** | ✅ 同精度，多色 penalty 让 iter 多 1.5× |
| AMGx classical V-cycle + reg=1e-2 | 50/50（自洽） | 138 | 3.92 | **1.00e+00** | ❌ **物理错的**——解了不同的问题 |

**两条 DILU 路线（cuSPARSE / Multicolor）都能正确复现 OpenFOAM 的解，AMGx + 正则化得到的"收敛"是数学幻觉。**

---

## 1. v2.0 → v2.1 修订原因

v2.0 报告里把 `‖A·x − b‖ / ‖b‖`（自洽残差）当作精度指标。这个数对所有三家都是 ~1e-10，看起来 AMGx 也"正确收敛"了。

**问题**：AMGx 用了正则化（A + ε·I），自洽残差是相对**修改后的矩阵**算的。AMGx 实际解的是 `(A + ε·I) x' = b`，而 OpenFOAM 解的是 `A x = b`。两个方程的解可以差很多。

**v2.1 验证**：用 `‖x_solver − x_OF‖∞ / ‖x_OF‖∞` 直接对比三家解和 OpenFOAM 自报的 `x_final`，得到上面那张表。AMGx 的"收敛"实际是个 100% 偏差的解，和 OpenFOAM 完全无关。

---

## 2. 三家求解器全量对比（50 矩阵）

数据来源：`dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/aggregate.json`

### 2.1 主表

| Solver / Config | n | converged | iter med | iter max | solve med (s) | solve max (s) | rel_vs_OF med | 注 |
|-----------------|---|-----------|----------|----------|---------------|---------------|---------------|----|
| **cuSPARSE DILU-PCG** | 50 | **50/50** | 189 | 191 | **15.34** | 20.24 | **4.38e-04** | row-level precond, level scheduling |
| **Multicolor DILU-PCG** (red-black, 80×160×80) | 50 | **50/50** | 290 | 293 | 19.21 | 69.47 | **5.51e-04** | 多色 penalty: iter 数高 1.5× |
| **AMGx classical_v + reg=1e-2** | 50 | 50/50 (false) | 138 | 142 | **3.92** | 4.29 | **1.00e+00** | 解了 (A+εI)x=b 不是 Ax=b |
| AMGx classical_v (no reg) | 1 | 0/1 | 200 (cap) | — | 2.97 | — | — (DIVERGED) | reg=0 时 5 个 config 全 fail |
| AMGx aggressive (no reg) | 1 | 0/1 | 200 | — | 1.34 | — | — | |
| AMGx classical_gs_pcg (no reg) | 1 | 0/1 | 200 | — | 68.11 | — | — | |
| AMGx classical_gs_bicgstab (no reg) | 1 | 0/1 | 200 | — | 2.44 | — | — | |
| AMGx aggregation_pcg (no reg) | 1 | 0/1 | 200 | — | 54.15 | — | — | |
| AMGx classical_v + reg=1e-4 | 1 | 0/1 | 200 | — | 3.13 | — | — | reg 不够，仍发散 |
| AMGx classical_v + reg=1e-6 | 1 | 0/1 | 200 | — | 3.69 | — | — | reg 太小 |

### 2.2 怎么读这张表

**对于"我们求解器能不能用作 OpenFOAM 替代品"这个核心问题**：
- ✅ **cuSPARSE DILU-PCG**：精度合格（4e-4，被矩阵条件数限制），50/50 全收敛。**首选生产求解器**
- ✅ **Multicolor DILU-PCG**：精度合格（5e-4），但 iter 数多 1.5× 且 wallclock 多 25%。Phase 3 在合成 Poisson 上的 1.5× 速度优势在真实异构矩阵上**消失**
- ❌ **AMGx**：用任何 config / 正则化都不能给出正确答案。**这种 case 不能用 AMGx**

**这个真实测试暴露了我们求解器原来在合成 case 上调好的优化在真实物理上的局限**——研究意义恰恰在这里。

---

## 3. 关键技术发现

### 3.1 cuSPARSE DILU-PCG 性能 12.7× 提升（v1 → v2）

**问题**：v1.0 单矩阵 solve 351 s（合成同等规模仅 5 s）。

**根因**：原 PCG 主循环每 iter 用 `float(jnp.linalg.norm(r))` 强制 GPU→CPU 同步 3 次。

**修法**：整个 PCG 循环写成 `jax.lax.while_loop` + `@jax.jit`：

```python
@jax.jit
def pcg(...):
    def cond_fn(state):
        ...
    def body_fn(state):
        ...  # 全 GPU，无 Python sync
    return jax.lax.while_loop(cond_fn, body_fn, init)
```

**结果**：单矩阵 27.75 s（含 5s JIT 编译）；摊销后 ~15 s/矩阵。Canonical 32³ 回归 1.10 s 通过。

### 3.2 AMGx 在 LPBF 矩阵上发散的根因

**矩阵诊断**：
```
N = 1,024,000,  nnz = 7,104,000
|A − A.T|_F / |A|_F = 0.000e+00              ← 完美对称
diag range: [7.5e-27, 2.3e-13]                ← 跨 14 个数量级
diag dominance |diag|/Σ|off-diag|:
    61.6% rows: ratio == 1.0  ← Laplacian 标志
    18.4% rows: ratio < 1.0   ← 不严格对角占优
|A · 1|_∞ = 6.58e-14                          ← 几乎奇异（常向量是零空间）
```

**机理**：纯 Neumann Laplacian → A 奇异；OpenFOAM 用 `setReference()` 钉一个 cell 的 pd 值，把这个 cell 的 diag 翻倍，让 A 数学上非奇异，但 λ_min ≈ 1e-13、condition number ≈ 10¹².

**两个独立问题让 AMGx 失效**：
1. **奇异性**：AMG 经典粗化假设强对角占优，对几乎奇异矩阵层级会失稳
2. **diag 异质**：diag 跨 14 个数量级（设想 cells 在 mushy zone 边缘），均匀的 AMG smoother 在低 diag 行根本不工作

5 种 config（classical_v / aggressive / GS_PCG / GS_BICGSTAB / aggregation_pcg）全 diverge，rel_residual 卡在 1e-4 量级。

### 3.3 AMGx 正则化方案为什么"看似收敛但物理上错"

**正则化方法**：在 reader 里加 `regularize(bundle, eps_rel=1e-2)`，对 A 加 ε·I 让其严格 SPD：

```python
def regularize(bundle, eps_rel=1e-2):
    eps = eps_rel * np.abs(bundle.A.diagonal()).max()
    A_reg = bundle.A + eps * I
```

**self-consistency 上的"收敛"**：AMGx 现在能在 ~138 iter 把 `‖(A+εI)x' − b‖ / ‖b‖` 压到 1e-10。这是真的——它确实解了那个修改后的方程。

**vs OpenFOAM 的"100% 偏差"**：但 `(A+εI) x' = b` 的解 x' 跟 `A x = b` 的解 x 关系不简单：
- 原 A 几乎奇异，零空间近似是常向量
- OpenFOAM `setReference` 用一个 cell 的钉值锁住零空间
- 我们 reg 用均匀 ε·I 锁住零空间（每个 cell 都被同等约束）
- **两种锁法给出的 x 在零空间方向上完全不同**——差出整个解的尺度

实测 `‖x_amgx − x_OF‖∞ / ‖x_OF‖∞ = 1.00`，**说明 AMGx 的解物理上不是 OpenFOAM 的解**。

**结论**：对于 OpenFOAM 风格的 setReference Laplacian，**AMGx + 简单正则化无法用作 OpenFOAM 求解器替代**。需要更精细的预处理（比如显式找到 OpenFOAM 的 ref cell + 模拟 setReference 操作），但那会让 AMGx 失去通用性。

### 3.4 Multicolor DILU-PCG 的 1.5× iter penalty 在真实矩阵上是真的

我们 Phase 3 设计文档里预测了多色 vs level scheduling 的"1.5-1.6× iter count penalty"（Duff-Meurant 1989 理论）。

**v2.1 实测**（红黑色，80×160×80 网格）：
- cuSPARSE level scheduling: 189 iter
- Multicolor red-black: 290 iter
- ratio = **1.53**（在理论预测范围内）

而 wallclock 维度：cuSPARSE 15.34 s vs Multicolor 19.21 s，Multicolor 慢 25%。
- 理论预测：多色因为 color 间 barrier 少 → 每 iter 应该更快 → 总 wallclock 接近平手
- 实测：每 iter 19.21/290 ≈ 66 ms vs cuSPARSE 15.34/189 ≈ 81 ms，**Multicolor 单 iter 是更快**（1.23×）
- 但 iter 多了 1.53× → 总 wallclock 慢

**研究价值**：合成 stiff Poisson 上 Multicolor 显示的优势依赖于矩阵的"理想"结构。真实 LPBF 多相 + setReference 矩阵让 Multicolor 的 iter penalty 完全暴露，速度优势消失。

---

## 4. 求解器选择建议（更新版）

| 场景 | 推荐 | 原因 |
|------|------|------|
| 真实 LPBF / setReference Laplacian / 异质 diag 矩阵 | **cuSPARSE DILU-PCG** | 唯一能正确复现 OpenFOAM 解的，且最快 |
| 合成 stiff Poisson / 严格对角占优 / 单一物理 | AMGx classical V-cycle | 22.1× 加速（canonical case 测过） |
| 多 GPU / 大并行场景 | Multicolor 可能值得（理论） | 当前单卡上没优势 |
| 非对称矩阵（T 方程）| **暂不可用** | 需 BiCGStab wrapper（task #15）|

---

## 5. 当前局限

1. **只对 pd（SPD）方程做了对标**。100 个 T 矩阵已 dump 但所有 SPD-only 求解器都跳过了
2. **单 case 验证**。spot_melt 之外的 LPBF case 矩阵特性可能不同（不同激光功率、扫描模式、不同 setReference 处理）
3. **OpenFOAM wallclock 缺**。我们的 `solve_s` 是端到端 Python 时间（含 JIT 编译开销 + 数据传输）；OpenFOAM 自报的只有 iter 数，不知 wallclock。要凑 OpenFOAM 速度需要从 lab 机器的 log 提取
4. **AMGx 正则化只测了 1% 量级**。理论上更精细的 reg（比如只对 ref cell 加大 diag）可能保留物理意义，但需要先识别 OpenFOAM 的 ref cell——非平凡

---

## 6. 数据集 & 复现

### 数据集
- 50 个 pd 矩阵：`dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/<time>/pd_corr0/data.npz`（每个 ~37 MB，总 1.85 GB）
- 100 个 T 矩阵（暂存）：`<time>/T_corr0|T_corr1/data.npz`
- 每矩阵的 results/{cusparse, multicolor, amgx_classical_v_reg1e-02}.json 含完整 iter / time / rel_vs_OF
- 聚合：`collected/spot_melt_npz/aggregate.json`

### 一键重现命令

```bash
source /home/yzk/jax-env/bin/activate
ROOT=dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz

# cuSPARSE DILU-PCG
python -m dilu.benchmark.openfoam_crosscheck.driver_cusparse "$ROOT" \
    --pattern "*/pd_corr0" --no-save-x

# Multicolor DILU-PCG (red-black coloring with grid hint)
python -m dilu.benchmark.openfoam_crosscheck.driver_multicolor "$ROOT" \
    --pattern "*/pd_corr0" --grid 80 160 80 --no-save-x

# AMGx (with regularization — produces wrong answers, kept for documentation)
python -m dilu.benchmark.openfoam_crosscheck.driver_amgx "$ROOT" \
    --pattern "*/pd_corr0" --cfg classical_v --reg 1e-2 --no-save-x

# Aggregate
python -m dilu.benchmark.openfoam_crosscheck.aggregate_results "$ROOT"
```

---

## 7. 下一步路线图

1. **BiCGStab wrapper（task #15）** — 解锁 100 个 T 矩阵的对标
2. **多 case 扩展**（其他激光参数 / 扫描速度），凑 1000+ 矩阵
3. **OpenFOAM wallclock 对比** — 从 lab 机器 log 提取，加进对比表
4. **cuSPARSE 性能进一步优化** — 当前 15s/矩阵中位数比 v2.0 报告里说的 8.67 s 慢，因 driver 在每矩阵后 `jax.clear_caches()` 强制 JIT 重编译。修这个 bug 应该让 cuSPARSE 回到 8 s 量级
5. **AMGx 智能正则化**（可选） — 识别 OpenFOAM ref cell，模拟 setReference 而不是均匀 ε·I

---

## 8. v2.1 验证记录

| 阶段 | 状态 | 证据 |
|------|------|------|
| cuSPARSE 50 矩阵 + rel_vs_OF | ✅ | aggregate.json，rel_vs_OF=4.38e-4 |
| Multicolor 50 矩阵 + rel_vs_OF | ✅ | rel_vs_OF=5.51e-4，与 cuSPARSE 同量级 |
| AMGx + reg 50 矩阵 + rel_vs_OF | ✅ | rel_vs_OF=1.00（自洽收敛但物理错）|
| 三家正确性验证 | ✅ | cuSPARSE 与 Multicolor 互验：他们的解互差 ~1e-3 量级，与各自和 OF 的 4-5e-4 一致 |

---

**报告版本控制**：
- v1.0 (2026-04-26)：1 矩阵初步对比 + 完整管道验证
- v2.0 (2026-04-27 早)：50 矩阵 cuSPARSE+AMGx + 误用 rel_residual 作主指标 — **已撤销**
- **v2.1 (2026-04-27)：50 矩阵三家齐全 + 用 vs-OpenFOAM 真精度 + 修 v2.0 误读**
- v3.0 (TBD)：+ T 方程 BiCGStab + 多 case + OpenFOAM wallclock
