# 全流程教程：多核 Scaling + 500+ 矩阵采集 + 精度改进实验

## 目录

1. [环境准备](#1-环境准备)
2. [安装 patched solver](#2-安装-patched-solver)
3. [多核 Scaling 测试](#3-多核-scaling-测试)
4. [多时间点矩阵采集 (500+)](#4-多时间点矩阵采集-500)
5. [数据传输到 GPU 机器](#5-数据传输到-gpu-机器)
6. [精度改进实验](#6-精度改进实验)
7. [全局范数分析](#7-全局范数分析)
8. [分布图生成](#8-分布图生成)
9. [故障排除](#9-故障排除)

---

## 1. 环境准备

### Lab 机器 (CPU, 56 核)

```bash
# 确认 OpenFOAM 已安装
source /usr/lib/openfoam/openfoam2506/etc/bashrc  # 或你的路径
echo $WM_PROJECT_VERSION  # 应该是 v2506 或 2506

# 确认 MPI
mpirun --version

# 确认 Python3 + numpy + scipy + matplotlib
python3 -c "import numpy, scipy, matplotlib; print('OK')"
```

### GPU 机器

```bash
# 确认 CUDA + cuSPARSE
nvidia-smi
python3 -c "import jax; print(jax.devices())"

# 确认 AMGx 可用
python3 -c "from dilu.amgx.python import Plan; print('AMGx OK')"
```

---

## 2. 安装 patched solver

在 **lab 机器** 上操作。

```bash
# 2.1 解压 deploy bundle
tar xzf deploy_v4.tar.gz
cd deploy

# 2.2 安装 patched solver (编译 laserMeltFoam)
#      参数是 LaserbeamFoam 的 solver 源码目录
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam

# 2.3 验证安装
which laserMeltFoam
# 应该输出 $FOAM_USER_APPBIN/laserMeltFoam
```

`install.sh` 会：
- 备份原始 `laserMeltFoam.C`, `TEqn.H`, `pEqn.H` 为 `.orig`
- 打入 matrixDumper + timing 补丁
- `wmake` 编译

---

## 3. 多核 Scaling 测试

目的：找到 56 核机器上 pd 和 T 方程各自的最优核数。

### 3.1 准备 case

确保你的 spot melt case 已经设置好（跑过几步，有完整的时间步数据）：

```bash
CASE_DIR="$HOME/cases/spot_melt_150W"  # 改成你的 case 路径

# 确认 case 结构
ls $CASE_DIR/system/controlDict
ls $CASE_DIR/constant/
ls $CASE_DIR/[0-9]*/  # 应该有时间步文件夹
```

### 3.2 运行 scaling 测试

```bash
cd deploy/

# 测试所有核数: 1, 2, 4, 8, 16, 24, 32, 40, 48, 56
# 每个核数跑 10 步
./scaling_test.sh $CASE_DIR --steps 10 --cores "1 2 4 8 16 24 32 40 48 56"
```

### 3.3 查看结果

```bash
# 查看 summary
cat $CASE_DIR/scaling_results/summary.csv

# 查看 scaling 图
ls $CASE_DIR/scaling_results/scaling_plot.png
```

输出示例：
```
Optimal for pd: 24 cores (12.3 ms)
Optimal for T:  32 cores (8.7 ms)
```

**记录下最优核数**，下一步要用。

### 3.4 Scaling 测试输出文件

```
$CASE_DIR/scaling_results/
├── summary.csv              # 汇总: cores, pd_ms, t_ms
├── scaling_plot.png         # scaling 曲线图
├── scaling_1cores.log       # 完整 solver log (每个核数)
├── scaling_2cores.log
├── ...
├── timing_1cores.csv        # 每步 timing
├── timing_2cores.csv
├── ...
├── decompar_1.log           # decomposePar log
├── decompar_2.log
└── ...
```

---

## 4. 多时间点矩阵采集 (500+)

目的：在 6 个物理阶段采集 pd 和 T 矩阵，总计 500+ 组。

### 4.1 调整时间范围

先确认你的 case 的物理时间范围：

```bash
# 查看现有时间步
ls $CASE_DIR/[0-9]* | head -5
ls $CASE_DIR/[0-9]* | tail -5

# 查看 deltaT 和 endTime
grep -E 'deltaT|endTime' $CASE_DIR/system/controlDict
```

**重要**：编辑 `multi_timepoint_collect.sh` 中的 `PHASE_START` 和 `PHASE_END`，
使其匹配你的 case 的物理时间。

默认值是针对 spot melt case (deltaT=2e-8, 从 ~2.25e-6 开始)。如果你的 case
不同，需要修改。脚本里有注释说明怎么确定正确的物理时间点。

### 4.2 运行采集

```bash
# 假设 scaling 测试显示 24 核最快
NP=24

# 每 2 步 dump 一次 (500+ matrices)
./multi_timepoint_collect.sh $CASE_DIR --np $NP --every 2
```

### 4.3 验证采集结果

```bash
# 查看每个阶段的矩阵数
cat $CASE_DIR/collected_matrices/summary.json

# 期望输出类似:
# {
#   "phase1_initial_heating":     { "pd_matrices": 50, "T_matrices": 80, ... },
#   "phase2_melt_pool_formation": { "pd_matrices": 50, "T_matrices": 85, ... },
#   ...
#   "total": { "pd_matrices": 300, "T_matrices": 600, "total": 900 }
# }
```

### 4.4 采集输出结构

```
$CASE_DIR/collected_matrices/
├── phase1_initial_heating/
│   ├── 2.25322e-06/
│   │   ├── pd_corr0/   { A.mm, b.mm, x0.mm, x_final.mm, metadata.json }
│   │   ├── pd_corr1/
│   │   ├── T_corr0/
│   │   └── ...
│   ├── 2.25522e-06/
│   └── ...
├── phase2_melt_pool_formation/
│   └── ...
├── phase3_recoil_pressure/
├── phase4_keyhole_formation/
├── phase5_quasi_steady/
├── phase6_cooling/
├── timings.csv           # 每步 pd/T wall time
├── summary.json          # 矩阵数统计
└── solver.log            # 完整 solver log
```

---

## 5. 数据传输到 GPU 机器

```bash
# 5.1 打包采集的矩阵
cd $CASE_DIR
tar czf matrices_v4.tar.gz collected_matrices/ scaling_results/

# 5.2 传输到 GPU 机器
scp matrices_v4.tar.gz user@gpu-machine:~/DILU-Research/

# 5.3 在 GPU 机器上解压
cd ~/DILU-Research
tar xzf matrices_v4.tar.gz
```

数据量预估：
- 每个矩阵 (A.mm + b.mm + x0.mm + x_final.mm) ≈ 100-200 MB (1M cells)
- 500 矩阵 ≈ 50-100 GB
- npz 压缩后 ≈ 10-20 GB

**建议**：先用 `shrink_dump.py` 压缩再传输：

```bash
# 在 lab 机器上压缩
python3 deploy/shrink_dump.py $CASE_DIR/collected_matrices/ --delete-mm

# 传输压缩后的 npz 文件
tar czf matrices_npz_v4.tar.gz collected_matrices/
```

---

## 6. 精度改进实验

在 **GPU 机器** 上操作。

### 6.1 快速测试 (10 个矩阵)

```bash
cd ~/DILU-Research

python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \
    dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/ \
    --max-matrices 10
```

输出示例：
```
[1/10] 2.25322e-06: baseline=4.38e-04, tight=1.23e-06
[2/10] 2.27322e-06: baseline=5.51e-04, tight=2.34e-06
...

SUMMARY
Baseline (tol=1e-10, sweeps=1):
  N=10, median=4.38e-04, P95=6.03e-04, max=6.36e-04

Tight (tol=1e-14, sweeps=2):
  N=10, median=2.34e-06, P95=8.12e-06, max=1.28e-05

Improvement ratio (median): 187×
```

### 6.2 全量测试

```bash
# 测试所有 pd 矩阵
python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \
    dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/ \
    --eq pd

# 测试所有 T 矩阵
python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \
    dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/ \
    --eq T
```

### 6.3 查看结果

```bash
# JSON 格式
cat dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/precision_experiment.json

# CSV 格式 (方便 Excel/Python 处理)
cat dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/precision_experiment.csv
```

---

## 7. 全局范数分析

在 **GPU 机器** 上操作。需要先跑完步骤 6（保存了 x 向量）。

### 7.1 确认 x 向量已保存

```bash
# 检查 tight config 的 x 向量
find dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/ \
    -name "*tight_x.npy" | wc -l
# 应该等于矩阵数
```

### 7.2 运行全局范数分析

```bash
python3 dilu/benchmark/openfoam_crosscheck/global_norm_analysis.py \
    dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/ \
    --csv dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/global_norms.csv
```

输出示例：
```
| solver                                 | eq  |    N | global_L2  | global_L∞  | weighted_L2 | med_per_mat | max_per_mat |
|----------------------------------------|-----|------|------------|------------|-------------|-------------|-------------|
| amgx_classical_v_diagscaled_tight      | pd  |   50 |  3.456e-06 |  1.280e-05 |  2.891e-06  |  2.340e-06  |  1.280e-05  |
```

### 7.3 理解范数指标

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| `global_L2` | `√(Σ‖x_s−x_OF‖²) / √(Σ‖x_OF‖²)` | 整体能量误差占比，物理意义最强 |
| `global_L∞` | `max(‖x_s−x_OF‖∞) / max(‖x_OF‖∞)` | 最差单点误差，保守保证 |
| `weighted_L2` | `Σ‖x_s−x_OF‖₂ / Σ‖x_OF‖₂` | L1 of L2s，对异常值不敏感 |
| `med_per_mat` | 中位数（旧指标） | 只看"典型"情况，掩盖尾部 |
| `max_per_mat` | 最大值（旧指标） | 等价于 global_L∞ |

**组会汇报建议**：用 `global_L2` 作为主指标，`global_L∞` 作为 worst-case 保证。

---

## 8. 分布图生成

在 **GPU 机器** 上操作。

```bash
# 生成所有分布图 (26 张 PNG)
python3 dilu/benchmark/openfoam_crosscheck/plot_distributions.py \
    dilu/benchmark/openfoam_crosscheck/collected/spot_melt_npz/

# 查看生成的图
ls docs/benchmark/figures/distribution/
```

输出文件：
```
docs/benchmark/figures/distribution/
├── boxplot_eq_pd.png           # 所有求解器 pd 对比箱线图
├── boxplot_eq_T.png            # 所有求解器 T 对比箱线图
├── hist_amgx_*_pd.png          # AMGx pd 直方图
├── ecdf_amgx_*_pd.png          # AMGx pd ECDF (带 P50/P95/P99/max 标线)
├── scatter_amgx_*_pd.png       # AMGx pd rel_vs_OF vs 时间步
├── hist_amgx_*_T.png           # AMGx T 直方图
├── ecdf_amgx_*_T.png           # AMGx T ECDF
└── scatter_amgx_*_T.png        # AMGx T scatter
```

---

## 9. 故障排除

### Scaling 测试

**问题**: `decomposePar` 失败
```
原因: case 的 boundary 设置可能不兼容并行
解决: 检查 constant/polyMesh/boundary 文件，确保所有 patch 类型正确
      如果有 `processor` patch，先删除 processor* 目录再 decompose
```

**问题**: `mpirun` 报错 "unable to find"
```
解决: mpirun -np N --allow-run-as-root laserMeltFoam -parallel
      或设置 hostfile
```

**问题**: TIMING 行没有出现在 log 中
```
原因: timing 补丁没有正确应用
解决: 检查 TEqn.H 和 pEqn.H 中是否有 std::chrono 相关代码
      重新运行 install.sh
```

### 矩阵采集

**问题**: matrixDumperDict 中的 dumpTimeRanges 不生效
```
原因: matrixDumper.H 没有更新到最新版本
解决: 重新 install.sh (会覆盖 matrixDumper.H)
```

**问题**: 每个阶段只采集到很少的矩阵
```
原因: 时间范围设置不对
解决: 先跑一次不 dump 的完整仿真，查看时间步范围
      修改 multi_timepoint_collect.sh 中的 PHASE_START/END
```

### GPU 精度实验

**问题**: AMGx 报错 "ZERO_PIVOT"
```
原因: 矩阵有零对角元素
解决: 已有 normalize_sign() 处理，检查是否被调用
```

**问题**: tight config 迭代数太多 (>500)
```
原因: tol=1e-14 可能对某些矩阵太严格
解决: 改为 tol=1e-12，或增大 max_iters
```

---

## 快速参考命令

```bash
# === Lab 机器 ===

# 安装 solver
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam

# Scaling 测试
./scaling_test.sh $CASE --steps 10

# 采集 500+ 矩阵
./multi_timepoint_collect.sh $CASE --np 24 --every 2

# 打包
tar czf matrices.tar.gz $CASE/collected_matrices/

# === GPU 机器 ===

# 精度实验 (快速)
python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \
    data/spot_melt_npz/ --max-matrices 10

# 精度实验 (全量)
python3 -m dilu.benchmark.openfoam_crosscheck.precision_experiment \
    data/spot_melt_npz/

# 全局范数
python3 dilu/benchmark/openfoam_crosscheck/global_norm_analysis.py \
    data/spot_melt_npz/ --csv norms.csv

# 分布图
python3 dilu/benchmark/openfoam_crosscheck/plot_distributions.py \
    data/spot_melt_npz/
```
