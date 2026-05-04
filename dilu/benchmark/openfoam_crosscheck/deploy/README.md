# DILU-Research OpenFOAM 矩阵导出器 — 部署包

**目标**：把 `laserMeltFoam` 求解器打一个补丁，让它在每次 PCG/PBiCG solve 入口把
线性系统 (A, b, x₀, x_final) 落盘为 MatrixMarket 格式，用于后续和 cuSPARSE / AMGx 对标。

## 这个包做什么（简版）

1. 给你在**另一台电脑**上的 `laserMeltFoam` 源码打 3 个补丁（+ 增加一个头文件）
2. 重编译 `laserMeltFoam` → 新二进制落在 `$FOAM_USER_APPBIN/`
3. 给你的**已有 case** 贴一个 `matrixDumperDict` 并改 `controlDict`，
   让它从 `latestTime` 接着跑 N 步，每步 dump 矩阵
4. 跑完后用 `sanity.py` 校验 dump 是否正确

## 前置条件

- [x] OpenFOAM 已装好并能跑（任意版本，**推荐 v2506**，其他版本可能需要微调）
- [x] 原始 `laserMeltFoam` 源码目录存在（例如 `~/LaserbeamFoam/applications/solvers/laserMeltFoam/`）
- [x] Python 3 + numpy + scipy（用于 sanity 检查，不需要 GPU）
- [x] 磁盘预算：**每个矩阵 ~700 MB ASCII @ 2M cells**。50 步 × 4 矩阵 ≈ **140 GB**。
      如果磁盘紧张就减少 `N_STEPS_TO_DUMP` 或考虑用更小网格的 case

## 文件清单

```
deploy/
├── README.md                    这份文档
├── TUTORIAL_FULL.md             完整全流程教程 (v4)
├── install.sh                   一键安装到 laserMeltFoam 源码
├── patch_case.sh                给已有 case 打 matrixDumperDict
├── scaling_test.sh              多核 scaling 测试 [NEW v4]
├── multi_timepoint_collect.sh   多时间点矩阵采集 [NEW v4]
├── run_everything.sh            一键全流程脚本 [NEW v4]
├── sanity.py                    跑完后验证 dump 是否正确
│
├── matrixDumper.H               新增的头文件 (v2: 支持时间范围触发)
├── laserMeltFoam.C.diff         3 行 patch
├── TEqn.H.diff                  2 行 patch
├── pEqn.H.diff                  3 行 patch
│
├── *.pristine                   原始参考文件（diff 是基于这些生成的）
└── *.patched                    最终结果参考文件
```

## 标准流程（3 步）

### Step 1: 装补丁 + 重编译

```bash
# 把整个 deploy/ 目录 rsync 到目标机器（随便放哪）
# 然后：
cd ~/deploy
source /usr/lib/openfoam/openfoamXXXX/etc/bashrc   # 你实际的 OpenFOAM 版本
./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam
```

**预期输出末尾**：`[install] SUCCESS  New binary: .../bin/laserMeltFoam`

**如果 install.sh 报 patch 失败**：你的 laserMeltFoam 源码和我这边的基准版本不一致。
两个选择：
1. 看 `*.diff` 文件，手动把插入点对齐——patch 的内容就 ~3 行每文件
2. 如果你没魔改过 `laserMeltFoam.C / TEqn.H / pEqn.H`，直接拿 `*.patched` 覆盖

### Step 2: 给 case 打 matrixDumperDict

```bash
./patch_case.sh ~/my_lpbf_case 50
#                ↑               ↑
#                case 根目录     从 latestTime 再跑 50 步（dump 50×5 ≈ 250 矩阵）
```

这会做的事：
- 写 `<case>/system/matrixDumperDict`（dumpTimeSteps = 1..N）
- 备份并修改 `<case>/system/controlDict`：
  - `startFrom latestTime;`
  - `endTime = latestTime + N × maxDeltaT`（保守估算）

检查一下再跑：
```bash
cd ~/my_lpbf_case
cat system/matrixDumperDict      # 看 dumpTimeSteps 对不对
cat system/controlDict | grep -E '(startFrom|endTime)'
```

### Step 3: 跑 + sanity

```bash
cd ~/my_lpbf_case
laserMeltFoam | tee log.laserMeltFoam   # 运行
# （等完成——按 mesh 和步数估 10min-2h）

# 校验
python3 ~/deploy/sanity.py postProcessing/matrices
```

**期望结果**：每个矩阵的 `|Ax-b|` < OF 报告的 `final_residual` × 100。
全部 PASS → 大功告成，把 `postProcessing/matrices/` 整个拷回我这边做对比求解。

## 推荐的 N_STEPS_TO_DUMP 取值

| 你的目标 | N | 预计磁盘 | 预计运行时间 |
|----------|---|----------|--------------|
| 快速烟雾测试 | 5 | ~15 GB | ~5 min |
| 小规模对标 | 20 | ~55 GB | ~15 min |
| 中等样本（几百矩阵） | 50 | ~140 GB | ~30 min |
| 大样本（>1000 矩阵） | 250 | ~700 GB ⚠️ | ~2.5 h |

如果每个 case 跑 50 步 dump，**5 个不同 case × 50 步 ≈ 1000-1250 矩阵**，刚好够你学长的要求，且参数/物理状态有多样性。

## 多 case 场景

假设你有 3 个 case：`case_150W/`, `case_250W/`, `case_400W/`。推荐做法：

```bash
# 每个跑一遍
for c in case_150W case_250W case_400W; do
    cd ~/cases/$c
    ../../deploy/patch_case.sh . 50
    laserMeltFoam | tee log.laserMeltFoam
    python3 ../../deploy/sanity.py postProcessing/matrices \
        | tee sanity_report.txt
    cd ..
done
```

跑完后每个 case 都有自己的 `postProcessing/matrices/` —— **打包拷回我这边**：

```bash
tar czf all_matrices.tar.gz \
    case_*/postProcessing/matrices/ case_*/sanity_report.txt
```

## 回滚

想把 laserMeltFoam 还原：
```bash
cd ~/LaserbeamFoam/applications/solvers/laserMeltFoam
for f in laserMeltFoam.C TEqn.H pEqn.H; do
    mv $f.orig $f
done
rm matrixDumper.H
wmake
```

想把 controlDict 还原：
```bash
cd ~/my_lpbf_case
mv system/controlDict.orig system/controlDict
rm system/matrixDumperDict
```

## 遇到问题

贴给我：
1. 目标机 `WM_PROJECT_VERSION`
2. `install.sh` 或 `wmake` 末尾 30 行输出
3. `sanity.py` 全部输出
4. 你 case 的 `system/fvSolution` 里 `pd` 和 `T` 的求解器配置

## 已知兼容性

- ✅ OpenFOAM **v2506**（本包基于此开发）
- ⚠️ v2412 / v2406：`getOrDefault` 在 v2212 前叫 `lookupOrDefault`；如报编译错误把 matrixDumper.H 里的三处 `getOrDefault` 改 `lookupOrDefault`
- ⚠️ foam-extend：API 差异较大，**不支持**
- ✅ 并行 MPI (v4 新增)：`scaling_test.sh` 和 `multi_timepoint_collect.sh` 支持 `--np N` 并行运行
