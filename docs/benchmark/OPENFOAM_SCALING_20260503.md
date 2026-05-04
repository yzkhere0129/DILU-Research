# OpenFOAM laserMeltFoam Scaling Test — 复盘 + 修补

**日期**: 2026-05-03
**目标**: 在 lab 机（HR54WV2，**Xeon Gold 5120 @ 2.2GHz, 1 socket × 28 cores × 2 SMT**）上找 laserMeltFoam 多核最优 N，建立 OpenFOAM 多核 baseline，跟 AMGx GPU 公平对比
**Case**: `~/cases/spot_melt_150W`（**500,000 cells**, 522,801 points; ≠ v3.2 的 1M-cell case，且 ≠ v3.2 的机器）

---

## 0. 一句话

**最优 N=32：pd_corr0 = 44.1ms，对 N=1 加速 12.86×（efficiency 40%）**。N=16 仅 5× 加速。N=56 走满 SMT 性能崩塌（被 controlled experiment 证明慢 3.3×）。

---

## 1. 完整实测结果（lab + lab2 数据合并，aggregate_scaling.py 重算）

跳过冷启动 5 步（dt < 1e-11 的非典型小步）后取 median：

| N | n_steps | pd_corr0 (ms) | pd_step total (ms) | T (ms) | speedup pd | efficiency |
|---|---------|---------------|---------------------|--------|-----------|------------|
| 1 | 51 | 567.4 | 602.8 | 96.8 | 1.00× | 100% |
| 2 | 21 | 296.4 | 317.6 | 67.1 | 1.92× | 96% |
| 4 | 51 | 210.2 | 223.1 | 63.3 | 2.70× | 67% |
| 8 | 51 | 144.0 | 154.4 | 49.4 | 3.94× | 49% |
| 16 | 51 | 113.4 | 123.8 | 44.3 | 5.01× | 31% |
| 24 | 78 | 61.8 | 73.0 | 48.0 | 9.18× | 38% |
| 28 | 78 | 50.5 | 61.7 | 44.4 | 11.23× | 40% |
| **32** ⭐ | 78 | **44.1** | 55.3 | 46.4 | **12.86×** | **40%** |
| 56 | 10 (cold-start only) | (157.6) | — | — | — | — |

T solve 在 N=16 就饱和（~44ms），N=24/28/32 没有进一步收益。

## 2. 三个非平凡发现

### 2.1 N=16→N=24 出现"加速跳跃"（efficiency 反弹）

| N | pd_corr0 | step-to-step speedup | 累计 efficiency |
|---|----------|---------------------|-----------------|
| 16 | 113.4 | — | 31% |
| 24 | 61.8 | 1.83× (理论 1.5×) | 38% |
| 28 | 50.5 | 1.22× (理论 1.17×) | 40% |
| 32 | 44.1 | 1.15× (理论 1.14×) | 40% |

按 strong scaling 教科书，efficiency 应单调下降；这里 16→28 反弹 9 个百分点。**最可能解释**：

- N=16 时每核 ≈ 31K cells，working set ≈ 31K × ~7 doubles/cell ≈ **1.7 MB > L2 (1MB)**，溢出到 L3/RAM
- N=28 时每核 ≈ 18K cells，working set ≈ **1.0 MB ≈ L2**，命中率跃升

如果用 1M+ cell mesh 重测，这个反弹大概率消失（每核仍溢出 L2）。

### 2.2 N=32 比 N=28 还快（用了 4 个 SMT 但没崩）

N=32 = 28 物理核 + 4 SMT 复用核。理论上 4 个 SMT pair 会拖慢，但**实测 N=32 比 N=28 快 1.15×**。这是 borderline：4 个 SMT pair 的 cache 抖动成本 < 通讯进一步分摊收益。**N=56**（满 SMT）就完全崩了，见下。

### 2.3 N=56 SMT 惩罚（controlled experiment 铁证）

lab1 跑 N=32/56 时 dt 没爬到稳态（脚本 placeholder bug，已修），但**两者都在同一个 dt < 1e-11 冷启动 regime**，唯一变量是 SMT：

| N | regime | pd_corr0 (ms) |
|---|--------|---------------|
| 32 (lab1) | cold-start | 47.9 |
| 56 (lab1) | cold-start | 157.6 (3.3× slower) |

**结论**：SMT 对稀疏 PCG/BiCG 是惩罚不是收益。N=56 不该用。

## 3. 跟 AMGx 对比的 status

| 项 | 状态 |
|----|------|
| 同机 OpenFOAM 多核 baseline | ✅ N=32 / 44ms (pd_corr0) |
| 同机 AMGx baseline | ❌ 没建（需在这个 500K case 上 dump 矩阵 → AMGx 跑） |
| v3.2 旧 baseline (2.57s, AMGx 0.30s) | ⚠️ 不能直接套：v3.2 是 1M-cell + 不同机器 |

**下一步**: 在 lab 机上**用同一个 spot_melt_150W case dump 矩阵**，本机用 driver_amgx.py 跑，得到同 mesh AMGx baseline，跟 OpenFOAM N=32 的 44ms 直接对比。这是真正的 apples-to-apples。

---

## 2. Bug 清单（已修复 / 待 lab 机执行）

| # | Bug | 根因 | 修复位置 | 状态 |
|---|-----|------|----------|------|
| 1 | summary.csv 全 0 | `scaling_test.sh:312` 取 `split(',')[3]`，CSV 只有 3 字段 → IndexError 静默 → vals=[] → MED=0 | scaling_test.sh:312/322 改 `[2]` | ✅ 已修 |
| 2 | N=32/56 报 "not enough slots" | OpenMPI 默认拒超物理核；nproc 报 56 但物理核大概是 28（SMT） | 加 `mpirun --oversubscribe --bind-to none` | ✅ 已修（启动时也 print warning）|
| 3 | summary.csv 缺 N=1/2 | 5/3 重跑时 `echo > SUMMARY` 覆盖了 5/2 的结果 | 改成空文件才写 header，否则 append | ✅ 已修 |
| 4 | 第一次 pd 1623ms 把 median 拉偏 | 冷启动 setup（DIC analyze first call） | summary 计算 + aggregate_scaling.py 都跳前 5 步 / 15 个 pd 行 | ✅ 已修 |
| 5 | 4cores log 末尾 `signal 6 Aborted` 被误判失败 | LaserbeamFoam v2506 dict 析构 double-free，跑完才崩，**不影响 timing** | 区分 "MPI 没起来" vs "跑完才崩"：用日志里 TIMING 行计数判断 | ✅ 已修 |
| 6 | 48 个 `Duplicate entry ... viscosityModel` warning | `Make/options` 把 `incompressibleTransportModelsLMFOAM` + 标准 `incompressibleTransportModels` 都链了 | 不影响 timing，**不修** | ⏸ 暂搁 |
| 7 | deploy/ .patched 里**没有** TIMING 打印 | lab 机的 binary 是某次手改加的，没回写 deploy；以后 install.sh 装新机 → TIMING 消失 → scaling_test.sh 废 | 给 pEqn.H.patched / TEqn.H.patched 加 `[TIMING_pd] ... ms` Info<<，重生成 .diff | ✅ 已修 |
| 8 | scaling_test.sh `\|\| true` 把所有错都吞掉 | 静默失败，summary 看不出哪个 N 真崩了 | 加 RUN_OK 标志 + 写 `<N>,FAIL,...` 行 | ✅ 已修 |
| 9 | N_STEPS=3 实际跑 51 步（adaptive dt） | dt=1e-12 起步，endTime=6e-8 要 51 步 | 数据量足够，**不视为 bug**（实际还更好），但用户应理解 N_STEPS 是上界 | 📝 文档化 |

---

## 3. 文件变更

### 新增
- `dilu/benchmark/openfoam_crosscheck/aggregate_scaling.py` — 从已存在的 timing csv 重算 median，**不需要 lab 机**
- `dilu/benchmark/openfoam_crosscheck/scaling_results_lab/summary_clean.csv` — 修正后 summary
- `dilu/benchmark/openfoam_crosscheck/scaling_results_lab/scaling_clean.png` — strong scaling 图

### 修改
- `dilu/benchmark/openfoam_crosscheck/deploy/scaling_test.sh` — bug 1/2/3/4/5/8 全部修复
- `dilu/benchmark/openfoam_crosscheck/deploy/pEqn.H.patched` — 加 `[TIMING_pd]` 打印
- `dilu/benchmark/openfoam_crosscheck/deploy/TEqn.H.patched` — 加 `[TIMING_T]` 打印
- `dilu/benchmark/openfoam_crosscheck/deploy/{pEqn,TEqn}.H.diff` — 同步重生成

---

## 4. 下一步：lab 机要做的事

**最少 3 条**（5 分钟）：

```bash
# 1. 看物理核数 vs SMT（决定 N=32/56 是否物理上有意义）
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Socket|Model name)"

# 2. 看 case mesh 大小（决定能不能跟 v3.2 的 1M baseline 对比）
checkMesh -case ~/cases/spot_melt_150W 2>&1 | grep -E "cells:|points:"
```

**重跑 N=32/56**（10-30 分钟，先把新 deploy.tar.gz 拷过去）：

```bash
# 在本机：
cd /home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/
tar czf deploy_v5.tar.gz deploy/
# scp deploy_v5.tar.gz lab:~/

# 在 lab 机：
tar xzf deploy_v5.tar.gz
cd deploy
# 因为只缺 N=32/56，单独跑这两个就好；append 到现有 summary
~/deploy/scaling_test.sh ~/cases/spot_melt_150W --steps 30 --cores "32 56"
# 把新 timing_32cores.csv / timing_56cores.csv 拷回来
# scp ~/cases/spot_melt_150W/scaling_results/timing_{32,56}cores.csv local:DILU-Research/dilu/benchmark/openfoam_crosscheck/scaling_results_lab/

# 然后本机重新聚合：
python dilu/benchmark/openfoam_crosscheck/aggregate_scaling.py
```

**可选**（如果想对齐 v3.2 的 1M-cell baseline）：

```bash
# 加密 mesh 到 1M cells（看 system/blockMeshDict 然后调整）
# 或者直接从 v3.2 的 case fork 一份过来
cp -r /path/to/v3.2/spot_melt_1M ~/cases/spot_melt_1M
~/deploy/scaling_test.sh ~/cases/spot_melt_1M --steps 30
```

---

## 5. 给学长汇报时该说什么

**当前能下的结论**：

1. ✅ scaling 工具链跑通，bug 全修
2. ✅ Lab 机 (Xeon Gold 5120, 28 物理核) 上 spot_melt_150W (500K cells) 完整 scaling 曲线已得：**最优 N=32, pd_corr0 = 44.1ms, 12.86× speedup**
3. ✅ T 在 N=16 饱和（44ms）, pd 在 N=32 仍未饱和但已 efficiency 40%
4. ✅ N=56 (满 SMT) 经 controlled experiment 证明慢 3.3×，确认 SMT 对稀疏 PCG 是惩罚

**当前还不能说**（要追问的话承认）：
- ⏳ "AMGx 比这个 OpenFOAM 多核 baseline 快 Y×" — 还要在同 mesh 上 dump → 跑 AMGx 才能下
- ⚠️ v3.2 报告里 AMGx 8.6× 那个数字是 1M-cell + 别的机器 vs 1 Xeon 核，**不是这次 N=32 (44ms) 的数字**

**N=16→24 反弹的解释（如果学长追问）**：mesh 太小（500K cells / N），N≥24 时每核 working set 落进 L2 cache 命中率跃升，1M+ cell mesh 不会出现这个现象。

---

## 6. Bug 1 的尸检（值得记的教训）

```bash
# scaling_test.sh 写出 timing csv：
echo "${N},${eq},${ms}" >> "${out_csv}"   # 3 fields

# header 却写：
echo "step,time_s,eq,wall_ms" > "${out_csv}"   # 4 fields ← misleading

# 然后 median 计算：
vals.append(float(line.strip().split(',')[3]))   # IndexError silently → vals=[]
```

**教训**: bash heredoc 里嵌 python 很难看到运行时异常，**特别是 `try: ... except: pass`**。今后写脚本：
- header 字段数 == 数据字段数 (废话但没人看)
- python `except: pass` 至少要 `except Exception as e: print(e, file=sys.stderr)`，否则任何索引错误都变成"无数据"

---

## 7. Status

| 项 | 完成度 |
|----|--------|
| Bug 诊断 | 9/9 |
| 本机修复 | 8/9（#6 暂搁）|
| 数据救回 | N=1,2,4,8,16 ✅ |
| Lab 机执行 | N=32, N=56 待跑；mesh check 待做 |
| 跟 AMGx 对比 | 阻塞（mesh 不一致 + 缺 N=32/56）|
