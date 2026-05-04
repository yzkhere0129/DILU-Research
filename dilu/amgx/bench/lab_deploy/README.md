# Lab 部署：拿"真"的 3-way bench 数字

## 现状（不严谨）

之前 `docs/benchmark/THREEWAY_BENCH_DRAFT.md` 用了：
- AMGx 数字 = **本机 RTX 3050 实测**（不是 lab 5060）
- OF 44ms = **从 docs/benchmark/OPENFOAM_SCALING_20260503.md 旧测搬的**（spot_melt_150W 500K cells, 不是学长 512K, 不是同一组矩阵）

→ 这俩拼到一起说"3-way 对比"，**不算 fair**。

## 要真的 fair 需要做的

| 任务 | 在哪台机器 | 怎么做 | 预计时间 |
|------|----------|--------|----------|
| **A** OF on 80³ N=32 | lab 56 核 (HR54WV2) | `run_of_lab_xeon.md` —— 修 spot_melt 到 80³ + 跑 30 步 + grep TIMING_pd | 30 分钟 lab 时间 |
| **B** AMGx on 学长矩阵 | lab 5060 GPU 机 | `run_amgx_lab_5060.md` —— 复用现有 bench script | 10 分钟 lab 时间 |
| **C** 我们 replica | 本机 1 core 已测 | (no change) | — |

## 拿到 A、B 数字后

我会重写 `THREEWAY_BENCH.md`（去掉 DRAFT）—— 三方都是同 mesh 量级 / 真测数字 / 对应硬件，可以直接给学长。

## 决策点

- 你要立刻去 lab 机敲 A 和 B 吗？我整理好命令了
- 还是先拿目前数据（带"不严谨"caveat）给学长，等他要更精确再补？

我建议：**先去敲 A 和 B**（一起 ~40 分钟 lab 时间）。学长肯定问"AMGx 在 5060 上多快"，到时候没数。
