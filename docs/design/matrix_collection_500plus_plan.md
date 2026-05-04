# 500+ Matrix Collection Plan — LPBF Phase-aware Dumping

**日期**: 2026-05-04
**目标**: 收 500+ AMGx vs OpenFOAM benchmark 矩阵，覆盖 5 个物理相位
**前置**: `dilu/amgx/bench/phase_detector.py`, `docs/benchmark/AMGX_PRECISION_20260504.md`

---

## 1. 物理相位定义（学长指定）

| Phase | 物理含义 | 触发判据（detector） |
|-------|---------|----------------------|
| **P0** cold start | 仿真刚开始，无熔池 | `max(T) < T_solidus`（默认 316L: 1685K）|
| **P1** melt-pool onset | 首次出现液相 | `max(T) ≥ T_solidus` 第一时刻 |
| **P2** melt-pool mature | 熔池稳定形成 | `max(gT) ≥ 0.5`（液相分数过半） |
| **P3** recoil-pressure onset | T 突破沸点，蒸气反冲压力建立 | `max(T) ≥ T_vap`（316L: 3133K） |
| **P4** keyhole forming | 自由表面深陷成空腔 | `gas_volfrac(t) / gas_volfrac(0) ≥ 1.5`（气相体积比初始增长 1.5×） |
| **P5** steady-state | 准稳态熔池 | `max(T)` 在 5 个连续采样窗口内变化 < 0.1% |

判据已写进 `dilu/amgx/bench/phase_detector.py:classify_phases()`。

---

## 2. 现有数据结论（已实测）

| Case | mesh | endTime | 实测 final T_max | 触发的 phase |
|------|------|---------|------------------|-------------|
| LPBF_sanity | 2K | 5e-10 s | 298 K | P0 only |
| LPBF_crosscheck | 2M | **1e-6 s** | **298 K** | **P0 only** ← 远未升温 |
| dumper_pipeline_test | 8K | 1.5e-7 s | 298 K | P0 only |

**结论：所有现有 dump 都在 P0 cold-start。**v3.2 报告 50 个矩阵 + 我们刚 dump 的 48 个矩阵全部都是同一阶段的"刚启动"矩阵，物理上**单一**。

### Why 没升温

LaserbeamFoam tutorial 配置 `laserPower = 150 W, beamRadius = 25 μm`（detected from `constant/LaserProperties` + `constant/timeVsLaserPower`）。

热传导 back-of-envelope：
- 功率密度 = 150 W / π(25e-6)² ≈ 7.6e10 W/m²
- 316L 热扩散率 α ≈ 4e-6 m²/s, ρCp ≈ 4e6 J/(m³·K)
- 升温时间常数 ≈ ρCp·d / (功率密度) ≈ 4e6 × 25e-6 / 7.6e10 ≈ **1.3e-9 s** 量级 setup
- 实际熔化（Δ T = 1400 K）：~10–50 μs（耦合相变潜热 + 传导）

→ **endTime=1e-6 s 比熔化所需短约 30–100 倍**。LPBF_crosscheck 跑了 40 min wall 还在 P0。

---

## 3. 收集 500+ 矩阵的两阶段策略

### Stage A：blind run（**无 matrixDumper**，只写 field 快照）

目的：找出每个 phase 的真实时间窗，不污染后续 dump（dumper 写 ASCII 一次 ~200MB）。

**配置变更**：
```cpp
// system/controlDict
endTime          200e-6;        // 200 μs，足够覆盖 P0~P5
deltaT           1e-12;          // unchanged，maxCo 自适应
writeInterval    1e-6;           // 每 1 μs 写一次 → 200 个 field 快照
writeControl     adjustableRunTime;

// system/matrixDumperDict
enabled          false;          // 关闭 dumper
```

**mesh 选择**：用 **16K cells** 版本（dumper_pipeline 那个）跑长 endTime。
- 2M cells × 200 μs ≈ 200 × 40 min = 5.5 天，**不可行**
- 16K cells × 200 μs ≈ 200/200K × 200 μs scale ≈ 1-2 小时，**可行**

### Stage B：phase 检测（用我们的 detector）

```bash
/home/yzk/jax-env/bin/python -m dilu.amgx.bench.phase_detector \
  /path/to/the/long/case/  --T-sol 1685 --T-vap 3133
```

输出：
- `postProcessing/phase_timeline.png` — 4 panel：T_max, gT_max, α_min, |U|_max vs time，phase 分界标线
- `postProcessing/phases.json` — `recommended_dump_times` 字段，已分相位均匀采样
- `postProcessing/phase_timeline.csv` — 原始数据

### Stage C：phase-aware dump 重跑

把 Stage B 给出的 `recommended_dump_times` 喂回 `system/matrixDumperDict`：

```cpp
enabled          true;
binaryMM         false;
dumpTimeRanges   ((t_P0_start t_P1_start) (t_P1 t_P2) ... );  // 5 区间
maxCorrectorsPerEq { pd 3; T 1; }                              // 每 step 3 pd + 1 T
equations        (pd T);
```

**预算**：每 phase ~100 step × (3 pd + 1 T) = 400 dump/phase × 5 phases = **2000 个 dump**（远超 500）。
如果磁盘紧张，每 phase 取 25 step → 100 dump/phase × 5 = **500** 正好达标。

### Stage D：跑完后验证

```bash
# 用 sweep_amgx_precision.py 在新数据集上跑（已支持任意 case_dir 通过环境变量扩展）
```

---

## 4. 推荐 case 配置（直接给 lab 机用）

### 4.1 16K cells, 200 μs, blind run

```cpp
// system/blockMeshDict — 用 dumper_pipeline_test 的 mesh (16×32×16=8192 cells 太小)
// 推荐改成 32×64×16 = 32K cells 平衡速度 vs 物理细节
blocks (
    hex (0 1 2 3 4 5 6 7) (32 64 16) simpleGrading (1 1 1)
);
```

### 4.2 system/controlDict

```cpp
startTime        0;
stopAt           endTime;
endTime          2e-4;            // 200 μs
deltaT           1e-12;
writeControl     adjustableRunTime;
writeInterval    2e-6;            // 100 个 field 快照
maxCo            0.5;
adjustTimeStep   yes;
maxDeltaT        2e-8;
```

### 4.3 constant/LaserProperties — 增大 power 加速触发 keyhole

```cpp
// 默认 150 W → P3 recoil 触发约需 50-100 μs
// 如要在 200 μs 内确切观察到 keyhole，建议 300 W
timeVsLaserPower
{
    file "$FOAM_CASE/constant/timeVsLaserPower";
}
// constant/timeVsLaserPower:
((0  300) (200e-6  300))
```

### 4.4 system/matrixDumperDict（**Stage A 关掉**）

```cpp
enabled  false;
```

---

## 5. 时序 + 工时估算

| Stage | 操作 | 谁跑 | wall time | 产出 |
|-------|------|------|-----------|------|
| A | 配置 32K case + 200 μs blind run | lab 机 | ~6-12 小时 | 100 个 field 快照 |
| B | 跑 phase_detector | 本机 | 1 分钟 | phases.json + png |
| C | 配置 dumper + 重跑 | lab 机 | ~12-24 小时 | 500-2000 矩阵 dump |
| D | 跑 sweep_amgx_precision 全集 | 本机 GPU | 5-30 分钟 | precision report |

**总挂壁时间 ~24-48 小时**（Stage A、C 是 lab 机长跑）。本机操作不到 1 小时。

---

## 6. 风险 + 缓解

| # | 风险 | 缓解 |
|---|------|------|
| R1 | 32K mesh 物理上不真实（mesh 太粗看不到 keyhole 细节）| 先跑 32K 快速 scout 找 phase 时间窗，**再用 1M+ mesh 在已知关键时间窗 dump 少量高分辨率矩阵** |
| R2 | adjustTimeStep 让 dt → 2e-8 太大，跨 phase 边界 | maxDeltaT 降到 1e-8 或更小；用 `dumpTimeRanges` 而非 `dumpTimeSteps` 做时间窗匹配 |
| R3 | 300 W 导致 mesh 不收敛（流场太剧烈）| fallback 到 150 W，但延长 endTime 到 500 μs |
| R4 | matrixDumper 写 ASCII 把 wall time 翻倍 | dumper 已支持 `binaryMM` 选项；改 dumper 加 npz 直写（已存在 `shrink_dump.py` 后处理） |
| R5 | 学长要的"500+ 矩阵"是物理 distinct 还是 PIMPLE corrector 算 | 我们 dump 每 step 3 pd + 1 T = 4 矩阵/step，125 step 即 500 矩阵；要小心**"distinct timesteps"** vs **"distinct correctors"** 两种数法 |

---

## 7. 立即可执行 deliverables

1. ✅ **`phase_detector.py`** — 已写好，已在 LPBF_crosscheck/dumper_pipeline_test 上跑通（确认两者都只在 P0）
2. ✅ **本文** — 给学长的 plan
3. ⏳ **scout case**: 等用户决定 mesh/laser 配置后，由 lab 机跑 Stage A
4. ⏳ **dumper config 生成器**: 等 Stage A/B 出 `phases.json` 后，写脚本自动生成 `dumpTimeRanges`

---

## 8. 给学长汇报的关键点

1. **现有 50+48 = ~100 个矩阵都是 P0 cold start**，物理单一，benchmark 不充分 — 学长的判断完全对
2. **Detector 工具已就绪**，可自动识别 P0–P5 5 个 phase
3. **需要新跑一个 endTime ≥ 100 μs 的 case** 才能采集到 P1–P5
4. **预计可拿 500–2000 个 phase-balanced 矩阵**（取决于 dump 密度）
5. **本次精度 benchmark 的发现**：AMGx 已达机器精度（≤ 2.89e-15 vs scipy truth），v3.2 报告里的"AMGx 2.5e-6"是 OF pd tol=1e-8 的 reference 噪声，不是 AMGx 误差
