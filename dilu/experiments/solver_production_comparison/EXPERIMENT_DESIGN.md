# Solver Production Comparison — Experiment Design

**目的**：用受控实验严格验证以下假设（来自 `solver_comparison_500K.md` Key Takeaway 4）：

> "在真实 CFD 时间推进里 AMGx amortized + warm-start 应该比 LU 直解快 100-1000×"

之前没有公平对比 — 我们只测了 single-shot wall。该实验通过 **回放 dumped matrix 序列** 模拟 PISO 时间推进，公平比较各 solver mode。

---

## 方法学

### 不能直接做的事

OF 是闭合代码，**无法**把 AMGx 直接塞进 OF 的 PISO loop 替换 DICPCG。所以不能直接比 production end-to-end。

### 可以做的事 — Replay Test

利用 OF 已经 dump 的 N 个连续 timestep 的 (A, b)。每个 mode：
1. 按时间顺序消费 N 个矩阵
2. 模拟"上一步的解作为下一步初值"（warm-start）
3. 记录每步 wall + 总 wall + 残差

这虽然不是字面意义的 in-loop replacement，但**线性求解部分的公平比较**接近 production：
- 矩阵确实是 OF 时间推进里实际产生的
- warm-start 反映 PISO inner correctors 的现实
- AMGx amortized `update_coefficients` 重用 hierarchy 是真实工程模式

### 实验范围

| 维度 | 取值 |
|---|---|
| Mesh | 50×200×50 = 500K (现有 single_track) + 384-step dense (运行中) |
| 物理 | 300W laser, 1.2 μs, melting + early evap |
| Timesteps | 6 (single_track 现有) → 384 (dense, ETA 7h) |
| Tolerance | 1e-8 (= OF default), 1e-12 (extra precision check) |
| 矩阵 | pd_corr0 (PISO 第一压力修正, 主要数值瓶颈) |

### 5 个测试 modes

| # | Mode | 实现 | 预期 |
|---|------|------|------|
| 1 | **AMGx fresh** | 每步 new Plan() | 慢 — 每步全 setup ~2-3s |
| 2 | **AMGx amortized** | Plan 1 次, update_coefficients() per step | 中 — 重用 AMG hierarchy |
| 3 | **AMGx amortized warm** | 同 #2 + x_prev 作初值 | **快** — 减少 iter 数 |
| 4 | **CHOLMOD fresh** | cholesky() per step | 慢 — full factor 每步 |
| 5 | **CHOLMOD symbolic reuse** | analyze() 1 次, cholesky_inplace(A_new) per step | 中 — 重用 reordering |

### 假设要回答的问题

1. **AMGx amortized 比 fresh 快多少？** — 量化 setup overhead 节省
2. **warm-start 减少多少 iter？** — PISO 内 well-conditioned 矩阵下应有 2-5×
3. **CHOLMOD symbolic-reuse 比 fresh 快多少？** — 量化 reordering 节省（数值 factor 还得做）
4. **AMGx amortized vs CHOLMOD symbolic** 谁快？— 这是 production 真实选择
5. **Crossover 在哪？** — 多少 timesteps 后 amortized AMGx 超过 fresh AMGx 总成本？

### Acceptance criteria

实验有效需满足：
- ✅ 每个 mode 的 max rel_resid < 1e-7（验证求解收敛了）
- ✅ AMGx amortized 第 1 步 wall ≈ fresh setup（确认 setup 一次完整）
- ✅ AMGx amortized 第 2..N 步 wall < 1/3 of first（确认 update_coefficients 起作用）
- ✅ CHOLMOD symbolic 第 1 步 ≈ fresh CHOLMOD（确认 analyze 包含在 first）
- ✅ CHOLMOD symbolic 第 2..N 步 < first（确认 inplace 起作用）

如果 amortized 模式没快或没收敛，记录 + debug。

---

## 阶段划分

### Phase 0 — 基础设施（今天完成）
- [x] `replay_amortized.py` — 5 modes implementation
- [x] `plot_replay_results.py` — 3 figures
- [x] 现有 6 timesteps 预演，verify pipeline

### Phase 1 — 384-timestep dense replay
- [ ] dense_track_dump_500K 跑完（lab Xeon, 7h）
- [ ] tar 拉回 dev 解包
- [ ] 在 384 timesteps 上跑 5 modes
- [ ] 出 plots + 写 results.md

### Phase 2 — 大规模可解性 (synthetic Poisson)
- [ ] 写 synthetic Poisson generator (64K, 256K, 1M, 2M, 5M)
- [ ] 测 LU OOM 临界点
- [ ] 测 AMGx scaling
- [ ] 出 mesh-vs-wall log-log plot

### Phase 3 — 真实 OF wall 验证
- [ ] 改 single_track case 加 solverInfo function object
- [ ] 重跑 (6.7h)
- [ ] 解析 per-solve OF wall
- [ ] 跟 estimated 110 ms/iter 对比

---

## 输出 deliverables

```
dilu/experiments/solver_production_comparison/
├── EXPERIMENT_DESIGN.md           ← 本文
├── replay_amortized.py            ← 5 mode replay
├── plot_replay_results.py         ← 出图
├── replay_results.json            ← 实验数据
└── results/
    ├── phase0_pilot_6steps.json
    ├── phase1_dense_384steps.json
    ├── phase2_synthetic_scaling.json
    └── phase3_of_real_wall.json

docs/benchmark/
├── solver_production_comparison.md   ← 最终报告
└── figures/
    ├── replay_per_step.png
    ├── replay_cumulative.png
    ├── replay_summary_bar.png
    └── synthetic_scaling.png
```

---

## 已知 caveats

1. **Replay ≠ in-loop** — 我们没修改 OF source 把 AMGx 塞进去，只是 replay matrix sequence。但 PISO 内部 linear solve 占总 wall 70-80%，这部分 fair compare 已经够说明问题。
2. **Sign convention** — OF 存负对角 Laplacian，AMGx 习惯正对角；replay 需要 `normalize_sign` flip。已在 `replay_amortized.py` 处理。
3. **Memory pressure on dev** — 384 timesteps × 6 MB ASCII A.mm = 2.3 GB raw load。需要 streaming loader 或 lazy load. 当前 eager load OK for 6 但 384 需要 monitor。如果超限改 lazy。
4. **GPU contention** — RTX 3050 4GB 容易满。Replay 串行跑 modes，no contention 问题。
