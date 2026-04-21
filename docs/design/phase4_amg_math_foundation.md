# Phase 4 — 代数多重网格 (AMG / AMGx) 数学基础与 Plan B 调研

**Project**: JAX-GPU AM-CFD Platform — Stiff Pressure Poisson Solver Kernel Research
**Phase**: 4 / 4 — Part 1 (Math / Algorithm foundation). 零拷贝 FFI 集成架构由并行 jax-cfd-am-expert Part 2 交付。
**Author role**: CFD-math 专家 (本文)
**Date**: 2026-04-21
**Scope**: 纯数学/算法分析。为 Phase 4 AMGx 集成设定数值契约与验收门槛。**不写代码**。

---

## §0 重新 briefing 自查清单

本文档基于以下已读材料:

- `docs/design/phase1_dilu_math_foundation.md` — 特别是 §3.3 (Route C 的定性预览: AMG 在极端密度比下是 "right tool"), §4.1 (条件数 $\kappa_2(A) \sim \gamma/h^2$ 的推导), §4.4 (Gustafsson 尺度律在不连续系数下崩坏)。本文是对 Route C 的正式数学展开。
- `docs/design/phase2_cusparse_level_scheduling_math.md` — 特别是 §4.1 (Gustafsson $\kappa^{1/4}$ 迭代数预测), §4.2 (AMG 压 ILU 一个数量级, 5-30 iter), §4.3 (per-iter wall-time 构成)。Phase 2 已为 "AMG 压迭代数" 的论点打下线索。
- `docs/design/phase3_multicoloring_math.md` — §1 (重排 DILU 代数), §2 (着色算法), Phase 3 确认 DILU-家族预处理器对 3-D 网格的 per-iter 收益天花板。
- `docs/benchmark/phase2_cusparse_report.md` — T7 16³ 24-vs-71 iter baseline.
- `docs/benchmark/phase2.5_physical_report.md` — 物理健康基线 A/B/C; 32³ 上 115 iters (Test A) / 50+ iters (Test C) 已经 "可观但还可承受"。Phase 4 AMG 必须同样通过 A/B/C。
- `docs/benchmark/phase3_multicolor_report.md` — Phase 3 验收 8 条 PASS, T7 36 iters, F2 stop not triggered。
- `docs/benchmark/phase3_scaling_64_128.md` — **Phase 4 最核心触发数据**。 详见 §1。
- `docs/PROJECT_SUMMARY.md` — 跨 phase 总览。
- `CLAUDE.md` — 约束: `jax` 0.9.x + `jax.ffi` + C++/CUDA; no `-ffast-math`; float32 双精度累加器; FFI 前 `np.ascontiguousarray`.

**WebSearch 交叉确认完成 (2026-04)**:
- NVIDIA AMGx GitHub `NVIDIA/AMGX`: 仍在维护, 最新 tag v2.4.0 (Thrust 2.1.0 子模块; sm_70+ 同步 bug 修复; 新增 `AMGX_matrix_check_symmetry`)。平台支持 Linux x86-64, CUDA 11.2 / 11.8 / 12.2, A100/H100 验证。**较确定** AMGx 仍可用, **不确定** CUDA 12.4 以上 / CUDA 13.x 的官方兼容性 (NVIDIA 网站列表没列到 12.3 之上)。
- `shwina/pyamgx` (Python 绑定): 2025-05, 2026-03 均有 issue 提交, 表示项目仍有生命; 但 doc `pyamgx 0.1` 版本号很久未升, 代码实际活跃度 **较低**。
- hypre `BoomerAMG` (备选 AMG 库): 活跃维护, GPU 后端成熟, 支持 CUDA / HIP / SYCL, 分别提供 aggressive 与 classical 路径。
- `AMGX.jl` (Julia) 与 `pyamgx` 均确认 `replace_coefficients()` / `AMGX_matrix_replace_coefficients()` 语义: **稀疏模式不变时可复用 setup, 只更新 values**。这是 Phase 4 AM 工况 (每时间步 $\rho$ 变 → values 变, pattern 不变) 的关键契约。

**已知空白 (要在 §4 / §7 flag)**:
- AMGx 最新 release (v2.4.0) 的确切发布日期未从 WebSearch 明示。 **需验证** `git log --tags` 本地克隆后校对。
- pyamgx 是否支持 **device pointer 直喂** (vs 只支持 SciPy CSR / NumPy 数组从 host 上传) — doc 里说 "directly upload from SciPy sparse CSR, NumPy, Numba DeviceArrays"。对 **JAX DeviceArray 的指针形态** 没有明示, 这是 §5 的关键风险。

**版本警觉 (Master Plan 条)**: 本文基于 2026-04 所见。任何 API 断言 (函数签名、参数顺序) 在实现开始前 **必须** 以实际克隆版本的 `include/amgx_c.h` 为准。

---

## §1 为什么是 AMG, 以及为什么是现在

### 1.1 Phase 3 scaling 数据的结构性验证

Phase 3 scaling benchmark (`docs/benchmark/phase3_scaling_64_128.md`) 给出了以下 **关键迭代数表** (Phase 2 exact DILU, T7-类刚性矩阵, tol=1e-10):

| 网格 | $N$ | $\mathrm{nnz}$ | Phase 2 iters | 相对 16³ 的比 | $N^{1/3}$ 比 |
|---|---:|---:|---:|---:|---:|
| 16³ | 4 096 | 27 136 | 24 | 1.000 | 1.000 |
| 32³ | 32 768 | 223 226 | 128¹ | 5.33 | 2.000 |
| 64³ | 262 144 | 1 810 432 | 99 | 4.13 | 4.000 |
| 128³ | 2 097 152 | 14 581 760 | 186 | 7.75 | 8.000 |

¹ 32³ 这一行是 Test C (3-tier gas/liquid/solid, $\rho$-contrast 8000, geometry + melt-pool dip); 不是纯 T7 图案。它在这里作为"次严格"对照出现, 与其它三行的 T7 样式略有区别, 我在下文的 fit 里**只用 16³ / 64³ / 128³ 三点**。

对 16³, 64³, 128³ 三个纯 T7 数据点: 若假设 $N_{\text{pcg}} = C \cdot N^\alpha$:

- 16³ → 64³ 比 99/24 = 4.125; 若 $\alpha = 1/3$, 预测 $(262144/4096)^{1/3} = 4.000$。**实测 4.125, 预测 4.000, 误差 3%**。
- 16³ → 128³ 比 186/24 = 7.75; 若 $\alpha = 1/3$, 预测 $(2097152/4096)^{1/3} = 8.000$。**实测 7.75, 预测 8.000, 误差 3%**。

这是 **两位数精度的 $N^{1/3}$ 尺度律确认**。实测数据 (Phase 3 scaling report) 完美匹配 Gustafsson 1978 理论:

$$
\kappa(M_{\text{DILU}}^{-1} A) \sim \sqrt{\kappa(A)}, \qquad N_{\text{pcg}} \propto \sqrt[4]{\kappa(M_{\text{DILU}}^{-1} A)} \cdot \sqrt{\kappa(M_{\text{DILU}}^{-1} A)} \sim \kappa(A)^{1/4}.
$$

对我们的 T7 刚性问题 ($\gamma = 100$ 沿 z 中面), 条件数标度为 $\kappa(A) \sim \gamma / h^2 \sim \gamma \cdot N^{2/3}$. 所以

$$
N_{\text{pcg}} \sim \kappa(A)^{1/4} \sim \gamma^{1/4} \cdot N^{1/6} \quad \text{(Gustafsson 最优情形)}
$$

但 T7 实测是 $N^{1/3}$, 即 **比 Gustafsson 最优差一个方根**, 恰好对应 Phase 1 §4.4 所说的 "不连续系数下 Gustafsson 崩坏, 退化到 Jacobi 级别的 $\sqrt{\kappa(A)}$":

$$
N_{\text{pcg}} \sim \sqrt{\kappa(A)} \sim \gamma^{1/2} \cdot N^{1/3}.
$$

这就是我们观察到的那条 $N^{1/3}$ 曲线。这不是随机结果, 是 "DILU 在界面附近的谱失配" 在实测中的直接显形。 **Confidence: 确定** (两位数精度实测 + 理论闭合)。

### 1.2 外推到 256³ 与生产 AM 工况

典型 AM melt-pool 生产网格: 256³ ($N \sim 1.68 \times 10^7$) 或更高。由 $N^{1/3}$ 尺度律外推:

- **256³**: iter $\approx 24 \times 16 = 384$ (T7 刚性; tol=1e-10)。
- **512³**: iter $\approx 24 \times 32 = 768$。
- **真实 AM 密度比 1000 (非 T7 的 100)**: 由 (1.1) 的 $\gamma^{1/2}$ 系数, 额外 $\sqrt{10} \approx 3.16 \times$ 放大。256³ 真实 AM: $\approx 1200$ iters; 512³: $\approx 2400$ iters。

**每时间步 wall-time 估算** (3050 Laptop, 128³ Phase 2 实测 per-iter ~153 ms):
- 256³ per-iter ~1.2 s (bandwidth scales linearly with $N$); × 384 iter = **~460 s per timestep** (纯 T7)。
- 真实 AM: ~1800 s ≈ **30 min per timestep**。

**一个典型 LPBF 模拟需 $10^4 - 10^6$ 时间步**. 30 min × $10^4$ = $\sim 200$ days。 **不可接受**。

即使迁移到 A100 / H100 (假设 per-iter wall time 快 5-10×):
- 256³ 真实 AM per-timestep $\sim 3-6$ min。$10^4$ timesteps → 500-1000 小时。仍然 **不可接受**。

**结论**: DILU 家族 (Phase 2 exact + Phase 3 multicolor) 在 AM-realistic 分辨率下, 无论硬件如何进步, 都无法达到生产要求。**问题不在实现, 在算法本身的尺度律**。

### 1.3 AMG 的承诺: grid-independent convergence

AMG 的核心理论承诺 (Ruge-Stüben 1987; Stüben 2001 综述; Brandt 1986): 对 second-order elliptic PDE 的 AMG V-cycle, 若满足 standard regularity 假设 (SPD, 稀疏, 强对角占优或弱对角占优的 M-matrix 或 H-matrix), 则:

$$
\|e^{(k+1)}\|_A \leq \rho_{\text{AMG}} \cdot \|e^{(k)}\|_A, \qquad \rho_{\text{AMG}} < 1 \text{ 独立于 } h.
$$

即收敛率 $\rho_{\text{AMG}}$ 与网格分辨率无关。对应的 PCG 迭代数 **不随 $N$ 增长**, 而 **仅依赖物理问题的类型** (椭圆性常数、各向异性、系数跳跃特征)。

**典型实测范围** (文献综合, 见 §3):
- 常系数 Poisson: 5-15 iters (Poisson 定理极限)。
- 中等系数变化 (比如 anisotropy factor 10×): 10-20 iters。
- 不连续系数 (AM: $\gamma = 10^3$): 20-50 iters (文献数据稀薄, 存在较大不确定性)。

**关键观察**: 即便取最坏 50 iters 的上界, 对 128³ 这是 Phase 2 的 186 → 50, **压缩比 3.7×**; 对外推 256³ 真实 AM 是 1200 → 50, **压缩比 24×**。

**Phase 4 的命题**: 即使 AMG 的 per-iter cost 比 DILU 高 3×, 总 wall-time 仍可能快 10×:

$$
\text{speedup}_{\text{AMG over DILU}} = \frac{N_{\text{DILU}} \cdot T_{\text{iter,DILU}}}{N_{\text{AMG}} \cdot T_{\text{iter,AMG}}} = \frac{1200}{50} \cdot \frac{1}{3} = 8\times \quad (\text{256³ 真实 AM 预测})
$$

这里的 3× per-iter penalty 是合理的 AMG 开销上界 (见 §3)。 **较确定**。

### 1.4 为什么 "现在" 做

Phase 1 (FFI 打通), Phase 2 (exact DILU 正确性), Phase 2.5 (物理健康), Phase 3 (多色 DILU 并行度) — 全部落地后, **DILU-家族的技术路径被完整走通**。scaling benchmark 揭示出 per-iter dispatch 优化 (Phase 3) 只在 $N \lesssim 32^3$ 窗口有效; 在生产工况所在的 64³+ 窗口反而 **回归**。

换句话说: 我们已经 **穷尽** 了 DILU 的可榨取并行度。再往下一步必须换算法类。

Phase 4 不是 "又一个 DILU 优化", 而是 **降维打击** (master plan 语): 把 "迭代数作为 $N^{1/3}$ 增长函数" 这个尺度律本身打掉。

---

## §2 AMG 代数基础 (非几何, 非教学)

下文采用 Ruge-Stüben (classical) 路径的精确数学定义。Smoothed aggregation (SA) 是另一大类, 在 §2.6 简述差异。我们 Phase 4 的默认目标是 **classical AMG**, 对严格对角占优稀疏矩阵效果最稳健, 也是 AMGx 的默认选项之一。

### 2.1 记号

$A \in \mathbb{R}^{n \times n}$ SPD, 稀疏。邻接图 $G(A) = (V, E)$, $V = \{1, \ldots, n\}$, $E = \{(i,j) : i \neq j, A_{ij} \neq 0\}$。

目标: 构造 $A_0 = A, A_1, A_2, \ldots, A_L$ 的矩阵序列 (多重网格层次), 其中 $A_\ell \in \mathbb{R}^{n_\ell \times n_\ell}$ 且 $n_0 > n_1 > \cdots > n_L$, 最粗层 $n_L \leq n_{\text{direct}} \approx 100$ 可直接求解。

### 2.2 Coarsening: C/F 分裂与 strength of connection

**Strength of connection** (SOC). 对 SPD $A$, 定义 "节点 $i$ 强依赖节点 $j$" 当且仅当

$$
-A_{ij} \geq \theta \cdot \max_{k \neq i} (-A_{ik}), \qquad \theta \in (0, 1). \tag{2.1}
$$

默认 $\theta = 0.25$ (Ruge-Stüben 经典阈值)。负号对应 M-matrix 约定 (对角正、非对角负)。若 $A$ 不是 M-matrix, 需要预先取 $-\min(A_{ij}, 0)$ 或按绝对值定义 SOC — **这是我们 AM 工况潜在风险点**, 因为变系数离散化可能产生零星的正非对角项 (§7 F2)。

由 (2.1) 定义 **强连接图** $S \subseteq E$: $(i,j) \in S \Leftrightarrow i$ 强依赖 $j$。注意 $S$ 一般 **非对称** (方向敏感: $i$ 强依赖 $j$ 不意味 $j$ 强依赖 $i$)。

**C/F 分裂** (Ruge-Stüben 启发式): 将 $V$ 分为 coarse 节点集 $C$ 和 fine 节点集 $F = V \setminus C$。合法 C/F 分裂的经典 **强插值准则** (Strong Interpolation Condition, Ruge-Stüben):

- **(C1)** 每个 $i \in F$ 在 $S$ 中至少有一个强邻居在 $C$ 中。(这是 "F 节点要能被 C 插值出来" 的必要条件。)
- **(C2)** 对每个 $i \in F$ 和每个 $j \in F$ 使得 $(i,j) \in S$, 存在共同的 $k \in C$ 使得 $(i,k) \in S$ 和 $(j,k) \in S$。(C1 的加强: F-F 强边必须通过 C 节点 "桥接"。)

**算法 (启发式 — 不是 NP-hard optima)**:
1. 预先按 lambda-值 ($\lambda_i = |\{j : (j,i) \in S\}|$, 即 i 的入强度) 排序。
2. 贪婪: 取最大 $\lambda$ 节点加入 $C$; 标记其所有强邻居为 $F$; 更新剩余节点的 $\lambda$; 重复直到所有节点分类。
3. 二次扫描修正: 不满足 (C2) 的 $F$ 节点升级为 $C$。

**复杂度**: $O(n + |E|) = O(\mathrm{nnz}(A))$ (使用 bucket sort), 并行化可能性: 中等 (heuristic 在 Jones-Plassmann 风格下可并行, 但严格 Ruge-Stüben 偏序列化)。

**目标粗化比率**: 经典 Poisson 3D 是 $|C| / |V| \approx 1/8$ (体心立方几何, 每 $2^3$ cell 合 1)。**操作复杂度**:

$$
C_{\text{op}} = \frac{\sum_\ell \mathrm{nnz}(A_\ell)}{\mathrm{nnz}(A_0)} \leq 2 \text{ (理想)} \tag{2.2}
$$

对 Ruge-Stüben Poisson 3D 典型 $C_{\text{op}} \in [1.8, 2.5]$。SA Poisson 3D 可达 $C_{\text{op}} \in [1.3, 1.5]$ (更激进聚合, 代价是收敛率略差)。

**Confidence: 确定** (Ruge-Stüben 1987 论文有精确定义 + 复杂度证明)。

### 2.3 Interpolation 算子 $P$

$P \in \mathbb{R}^{n \times n_c}$, $n_c = |C|$。把粗网格误差提升回细网格:

$$
(Pe_c)_i = \begin{cases}
(e_c)_{\tilde i} & i \in C \text{ 且 } \tilde i \in \{1, \ldots, n_c\} \text{ 是 } i \text{ 在 } C \text{ 中的索引} \\
\sum_{j \in C_i} w_{ij} (e_c)_{\tilde j} & i \in F
\end{cases} \tag{2.3}
$$

其中 $C_i = \{k \in C : (i,k) \in S\}$ 是 $i$ 的强 C 邻居集。插值权重 $w_{ij}$ 按 **classical RS interpolation** (Ruge-Stüben 1987):

$$
w_{ij} = -\frac{1}{a_{ii}} \left( a_{ij} + \sum_{k \in F_i^s} \frac{a_{ik}\, a_{kj}}{\sum_{m \in C_i} a_{km}} \right), \qquad j \in C_i, \tag{2.4}
$$

其中 $F_i^s = \{k \in F : (i,k) \in S\}$ 是 $i$ 的强 F 邻居集。该公式的直觉:
- 第一项 $-a_{ij}/a_{ii}$: $i$ 到 $j$ 的直接 "影响因子", 类似 Jacobi 权。
- 第二项: **F-F 强连接的二次平均** — 把通过 F 节点 $k$ 传递到 $j$ 的影响也算进来。分母规范化确保行和 = 1 (零常模保持)。

$P$ 的稀疏模式: 每行 nnz $\leq |C_i| + 1$, 对 3D Poisson 典型 1-7。

**向量化写法**: $P$ 按 F-subset 与 C-subset 分块:

$$
P = \begin{pmatrix} I_{nc} \\ W \end{pmatrix}, \qquad W \in \mathbb{R}^{(n - n_c) \times n_c}, \tag{2.5}
$$

其中 $W_{ij} = w_{ij}$ 按 (2.4) 填充。行按 $C$ 排先, $F$ 排后 (AMG 的标准置换约定)。

**Confidence: 确定** (Ruge-Stüben 原文 + Stüben 2001 综述公式一致)。

### 2.4 Galerkin 三积: 粗操作 $A_c$

**定义**:

$$
A_c = P^\top A P. \tag{2.6}
$$

这是 AMG 算法的 **代数心脏**。解释:
- (2.6) 保证 $A_c$ **仍然 SPD** (若 $A$ SPD 且 $P$ 列满秩)。
- $A_c$ 的稀疏模式比 $A$ "致密": 每个 C 节点的邻居 in $A_c$ 是其原图中 2-hop 邻居 (经过 F 节点路径) 的并集。
- $\mathrm{nnz}(A_c)$ 通常为 $\mathrm{nnz}(A)$ 的 $1/4$ 到 $1/2$ (3D Poisson 典型), 形成 geometric decrease 序列; 累积 $C_{\text{op}}$ 见 (2.2)。

**计算成本**: 两次稀疏矩阵乘法 (SpGEMM):
- $B = A P$: 成本 $\sim \mathrm{nnz}(A) \times \mathrm{nnz}(P)_{\text{avg per col}}$
- $A_c = P^\top B$: 成本 $\sim \mathrm{nnz}(P) \times \mathrm{nnz}(B)_{\text{avg per row}}$

GPU SpGEMM 在 2020+ 已基本成熟 (cuSPARSE `cusparseSpGEMM_*` 系列, AMGx 内部实现); 成本通常占 **AMG setup 总成本的 40-60%**。

### 2.5 V-cycle 递归

**一次 V-cycle (抽象递归)**, 给定当前右端 $b$ 求解 $A u = b$:

```
function vcycle(A_0, A_1, ..., A_L; P_0, P_1, ..., P_{L-1}; S_0, S_1, ..., S_{L-1}; b_0, u_0, ell=0):
    if ell == L:
        u_0 = A_L^{-1} b_0    # direct solve at coarsest
        return u_0
    
    # Pre-smoothing: ν_1 sweeps of smoother S_ell on A_ell
    u_ell = smooth(A_ell, u_ell, b_ell, ν_1)
    
    # Residual restriction
    r_ell = b_ell - A_ell u_ell
    b_{ell+1} = P_ell^T r_ell      # restrict to next level
    u_{ell+1} = 0                   # zero initial guess on coarse
    
    # Recurse
    u_{ell+1} = vcycle(..., ell+1)
    
    # Prolong correction
    u_ell = u_ell + P_ell u_{ell+1}
    
    # Post-smoothing: ν_2 sweeps
    u_ell = smooth(A_ell, u_ell, b_ell, ν_2)
    
    return u_ell
```

其中:
- **Smoother $S_\ell$**: 常用 Gauss-Seidel / SOR / polynomial Chebyshev / Jacobi。Gauss-Seidel 在 CPU 是 de facto; **GPU 上 Chebyshev polynomial smoother 或 weighted Jacobi 更合适** (完全并行), AMGx 默认 Jacobi 或 polynomial。
- **Pre/post smooth sweep 数** $\nu_1, \nu_2$: 典型 $\nu_1 = \nu_2 = 1$ 或 $2$。
- **W-cycle** 变体: 在中间层调用 2 次递归; 更强但成本更高 (1.5-2×)。AMGx 默认 V-cycle。

**PCG 驱动**: 一次 V-cycle 作为 PCG 的预处理器应用。外层 Krylov 加速 + 内层 V-cycle 平滑分担模式 = $O(1)$-iter 收敛的关键组合。

### 2.6 Smoothed Aggregation (SA) 路径 — 对照

SA 不用 C/F 分裂; 而是把 $V$ 分为 **aggregates** $\{V_a\}_{a=1}^{n_c}$, 每个 aggregate 对应粗网格一个节点。

- 初始 tentative interpolation $\tilde P$: 分块指示矩阵 (0/1 entries)。
- **Smoothing step**: $P = (I - \omega D^{-1} A) \tilde P$, 其中 $\omega$ 阻尼因子 (典型 $\omega = 4/3 / \lambda_{\max}$)。这一步使 $P$ 的列扩散到邻居, 提升插值精度。
- Galerkin $A_c = P^\top A P$ 一样。

**SA vs RS**:
| 属性 | RS | SA |
|---|---|---|
| 稀疏模式 | $P$ 有 $n-1$ 非零 / C 列 (约束松) | $P$ 固定 4-8 非零 / 列 (可预测) |
| 收敛率 (Poisson) | 最优 | 稍差 (~1.3× 更多 iter) |
| 各向异性强度 | 一般 | 较强 (若选 smoothed $\tilde P$) |
| 并行 setup | 较难 (序列 C/F heuristic) | 较易 (聚合可并行) |
| GPU 实现 | 成熟 (AMGx default) | 成熟 (AMGx option) |

**Phase 4 建议**: 默认用 classical RS (`AMG_CLASSICAL`); 若 setup 过慢或 C/F 分裂崩坏, 回退 SA (`AMG_AGGREGATION`)。AMGx 两路都支持。

### 2.7 最粗层直接求解

当 $n_L \leq 100$ 左右, 用 dense LU 直接求解 $A_L u_L = b_L$。AMGx 内部用自己的 direct solve (小规模 cuSOLVER 调用)。成本 $O(n_L^3)$ — 因为 $n_L$ 小, 绝对 wall-time 可忽略 (~μs)。

---

## §3 AMG vs DILU: 为何迭代数网格无关

### 3.1 谱分析解释

PCG 在 SPD 系统 $Ax = b$ 上的收敛界 (Saad 2003, §6):

$$
\|e^{(k)}\|_A \leq 2 \left(\frac{\sqrt{\kappa(M^{-1}A)} - 1}{\sqrt{\kappa(M^{-1}A)} + 1}\right)^k \|e^{(0)}\|_A. \tag{3.1}
$$

达到 residual 压缩 $\epsilon$ 所需迭代:

$$
k \geq \frac{\sqrt{\kappa(M^{-1}A)}}{2} \ln\!\left(\frac{2}{\epsilon}\right). \tag{3.2}
$$

- **DILU**: $\kappa(M_{\text{DILU}}^{-1} A) \sim \sqrt{\kappa(A)}$ (Gustafsson 1978, 最优情形), **退化**到 $\kappa(A)$ (不连续系数下; Phase 1 §4.4)。Phase 2/3 实测显示我们属于后者: $k \sim \sqrt{\kappa(A)} \sim \gamma^{1/2} \cdot N^{1/3}$。
- **AMG**: $\kappa(M_{\text{AMG}}^{-1} A) \sim O(1)$, **独立于 $N$**。 Hackbusch 1985 / Ruge-Stüben 1987 的正式证明基于 two-grid convergence theorem: 若 smoother 满足 "smoothing property" 且 interpolation 满足 "approximation property", 则

$$
\|e^{(k+1)}\|_A \leq \rho \|e^{(k)}\|_A, \qquad \rho = \rho_{\text{smooth}} + \rho_{\text{coarse}} < 1, \tag{3.3}
$$

且 $\rho$ **独立于 $h$**。对 PCG 驱动的 AMG, (3.1) 中 $\kappa(M^{-1}A) \approx 1/(1 - \rho)^2 = O(1)$。

### 3.2 文献实测数据汇总

以下是我对 AMG 迭代数的文献综合 (各种 Poisson 变体, 2D/3D, 不同 stiffness):

| 问题 | AMG 方法 | 典型迭代数 (PCG tol 1e-8 到 1e-10) | 参考 |
|---|---|---|---|
| 常系数 3D Poisson, $n = 10^6$ | RS-AMG V-cycle | 6-10 | Stüben 2001 综述 |
| 常系数 3D Poisson, $n = 10^8$ | RS-AMG V-cycle | 6-10 | 同上, 网格无关 |
| 各向异性 ($\sigma_{\max}/\sigma_{\min} = 10^3$) Poisson | RS-AMG + GS smoother | 15-25 | Ruge-Stüben 1987 Table 5 |
| 多相流 Poisson, $\gamma = 10^3$ (petroleum) | AMGx classical | 5-15 | Naumov et al. 2015 Table 2 |
| ILU(0) (比较对照) | — | 数百 (不收敛/很慢) | 同上 |
| AM-like VOF $\gamma = 10^3$, cut-cell | Unknown (literature 空白) | 预测 20-50 | 本文 §3.3 外推 |

**核心观察**: AMG 迭代数对 $N$ 的依赖在各种刚性变形下都保持在 5-50 窗口。**没有 $N$ 指数的显式出现**。

### 3.3 AM 工况下的 AMG 预期数值

文献空白部分 (AM-like 不连续 $\gamma = 10^3$, 非平凡 cut-cell 界面) 的预测基于两个外推:

**外推路径 A (从 petroleum multiphase)**: Naumov 2015 Table 2 给 5-15 iters for $\gamma = 10^3$ 多相流 petroleum 问题 ($n \sim 10^6$)。我们的 AM 更复杂 (薄膜 + 曲率大界面), 但 $\gamma$ 相当, 估计 AM 要 **多 1.5-3×** iter: **8-45 iters 区间, 中值 20-30**。

**外推路径 B (从 anisotropic Poisson)**: Ruge-Stüben 1987 在 $10^3$-anisotropy 上给 15-25 iters。AM 的 discontinuity 是 "局部" 的 (仅沿界面), 比全局 anisotropy 更友好; 估计 AM 要 **类似或略少** iter: **15-25 iters**。

**综合预测**: Phase 4 在 128³ AM-realistic 工况下, AMG-PCG 迭代数 **应在 20-50 区间**。这是 §6 验收阈值的来源。 **较确定** (文献支持但有空白; 实测才能确认)。

### 3.4 Per-iter cost 对比: 打破 "AMG 胜过 DILU" 的直觉陷阱

**关键警告**: AMG 每个 PCG iter 成本比 DILU **更高**, 不是 "更低但网格无关"。这是两个不同的优化轴:

| 操作 | DILU per-PCG-iter | AMG per-PCG-iter (V-cycle) |
|---|---|---|
| SpMV $y = Ax$ | 1 次 | 1 次 (相同, PCG 共享) |
| Preconditioner apply | 1× forward + 1× backward SpSV | 1× V-cycle: $L$ 层, 每层 2 次 smoother (forward + reverse), 每层 restrict + prolong (2 次 SpMV-类) |
| 每层 smoother 成本 | N/A | $2 (\nu_1 + \nu_2)$ matvec + scalar ops |
| Restrict/prolong 成本 | N/A | $P^\top$ 和 $P$ SpMV 各一次, 每层 |
| **典型比例 (3D Poisson)** | 1.0 (baseline) | 2-5× baseline |

**典型数字** (AMGx 文献自述): $T_{\text{iter,AMG}} \approx 2-3 \times T_{\text{iter,DILU}}$ 在 well-setup 3D Poisson 上。

**break-even 分析**: AMG 总 wall time 胜过 DILU 当

$$
N_{\text{AMG}} \cdot T_{\text{iter,AMG}} < N_{\text{DILU}} \cdot T_{\text{iter,DILU}} \iff \frac{N_{\text{DILU}}}{N_{\text{AMG}}} > \frac{T_{\text{iter,AMG}}}{T_{\text{iter,DILU}}}.
$$

代入 §3.3 预测 (AMG 20-50 iters) 和 §1.3 DILU 实测 (128³ 186 iters, 256³ 外推 400-1200 iters):

| 网格 | $N_{\text{DILU}}$ | $N_{\text{AMG}}$ (pred) | $N_{\text{DILU}}/N_{\text{AMG}}$ | $T_{\text{iter,AMG}}/T_{\text{iter,DILU}}$ 盈亏点 |
|---|---:|---:|---:|---:|
| 128³ (T7 刚性) | 186 | 30 | 6.2× | AMG 只要 per-iter $< 6.2×$ DILU 就胜 |
| 256³ (真实 AM 外推) | 1200 | 35 | 34× | AMG 即使 per-iter $10×$ DILU 仍胜 3× |
| 512³ (未来) | 2400 | 40 | 60× | AMG 秒杀 |

**Phase 4 关键洞察**: 我们不是在 "优化 per-iter cost"; 我们是在 **换 iter-count 曲线**。DILU 的 $N^{1/3}$ 曲线和 AMG 的 $O(1)$ 曲线终将 (在 $N \gtrsim 64^3$) 相交且越来越远。

### 3.5 Setup cost 摊销问题

AMG 有一次性 setup 成本:
- C/F splitting 或 aggregation: $O(\mathrm{nnz}(A))$ flops, serial-leaning heuristic。
- 所有层 $P_\ell$ 构造: $O(\sum_\ell \mathrm{nnz}(A_\ell)) = O(C_{\text{op}} \cdot \mathrm{nnz}(A))$。
- 所有层 Galerkin 三积 $A_\ell = P^\top A_{\ell-1} P$: 2 次 SpGEMM per layer, 总 $O(C_{\text{op}} \cdot \mathrm{nnz}(A)^{1.1-1.5})$ (SpGEMM 非线性)。

**典型实测**: AMGx setup 在 $128^3$ Poisson 上 ~100-500 ms (NVIDIA benchmark on V100)。对比 DILU factor (Phase 2 实测 18 ms for $n = 1000$, 推测 $\sim 500$ ms for $n = 2 \times 10^6$), AMG setup 比 DILU 多 **$\sim 5-10×$**。

**关键**: 若稀疏模式不变, setup 只做一次, 后续时间步只需 `AMGX_matrix_replace_coefficients()` (§4 API) 更新 values + **重跑 setup (!) 因为 AMG 的 C/F 分裂依赖 values** (注意 2.1 SOC 定义中的 $A_{ij}$)。

**然而**: AMGx 文档指出, 如果只更新 values 而不重做 C/F, 可以 **保留现有层次** (称为 "coefficient refresh" 模式; AMGx 术语 "REFRESH"). 收敛率会略差 (因为 C/F 用的是旧 values 的 SOC), 但成本大幅降低。典型策略: 每 10-100 时间步做一次完整 setup, 其余步只 refresh。 **较确定** (AMGx 文档确认存在; 具体参数命名 **需验证**)。

**摊销公式**: 设完整 setup 成本 $T_{\text{setup}}$, coefficient refresh $T_{\text{refresh}}$, 每时间步 solve $T_{\text{solve}} = N_{\text{AMG}} \cdot T_{\text{iter,AMG}}$, 每 $K$ 步做一次完整 setup:

$$
T_{\text{effective per step}} = T_{\text{solve}} + T_{\text{refresh}} + \frac{T_{\text{setup}} - T_{\text{refresh}}}{K}.
$$

若 $T_{\text{setup}} = 10 T_{\text{solve}}$, $T_{\text{refresh}} = T_{\text{solve}}$, 且 $K = 20$:

$$
T_{\text{eff per step}} = T_{\text{solve}} + T_{\text{solve}} + \frac{9 T_{\text{solve}}}{20} = 2.45 T_{\text{solve}}.
$$

摊销后 AMG 每时间步仍比 pure-solve 多 1.45×, 但比 DILU 的 $\{N_{\text{DILU}}/N_{\text{AMG}}\}/\{T_{\text{AMG}}/T_{\text{DILU}}\}$ 的总速比仍有量级收益 (见 §3.4 表)。

---

## §4 NVIDIA AMGx 调研 (2026-04 时点)

### 4.1 维护状态确认

**GitHub**: `NVIDIA/AMGX`, tag 最新 v2.4.0. WebSearch 显示近期 (2025-2026) commits 包括:
- Thrust 2.1.0 子模块固定 (避免 CUDA toolkit Thrust 版本漂移)。
- sm_70+ 架构同步 bug 修复 (影响 A100 / H100 / RTX 30/40 系列)。
- 新增 `AMGX_matrix_check_symmetry` API。
- "Superfluous synchronization points" 移除 (performance)。

**关于 RTX 3050 Laptop (sm_86)**: 在 NVIDIA 官方支持列表里 (sm_70+ 涵盖), 但具体 perf 未经 NVIDIA 官方 benchmark 验证 (他们的 perf chart 主打 V100/A100/H100)。我们 Phase 4 benchmark 本身就会产生 3050 Laptop 上的第一手 AMGx 数据。 **较确定** 能跑, **不确定** 跑多快。

**CUDA 兼容性**: 官方支持 CUDA 11.2, 11.8, 12.2。我们当前 stack 是 CUDA 12.4 (Phase 1 toolchain)。 **需验证** 12.4 是否实质兼容 (理论 12.x 系列向后兼容)。**CUDA 13.x 完全未 mentioned** — 若将来升级 13.x, 需重新评估。

**License**: AMGx 历史上是 BSD-3-Clause; 2026-04 GitHub 页面未明示 license tag 改变。 **较确定** 仍 BSD-3, 但 **建议实现前 git clone 后检查 `LICENSE` 文件**。

### 4.2 API 能力

从 `include/amgx_c.h` 与 AMGx Reference PDF 汇总:

**Solver 选项** (通过 config JSON 指定 `solver`):
- `PCG` (preconditioned CG) — 对称 SPD 系统首选 (我们是这个)。
- `PCGF` (flexible PCG) — 对 preconditioner 做 reproj 的版本。
- `BiCGSTAB`, `BiCGStab_F` — 非对称情形。
- `GMRES` — 通用 Krylov。
- `FGMRES` — flexible GMRES。
- `AMG` — 单独的 AMG 作为 "外层求解器", 不加 Krylov。

**Preconditioner 选项** (作为 solver 的 `preconditioner` 字段):
- `AMG` — 多重网格。其子选项 `algorithm` 选 `CLASSICAL` (RS) 或 `AGGREGATION` (SA)。
- `JACOBI_L1` — L1 Jacobi。
- `BLOCK_JACOBI` — 块 Jacobi。
- `MULTICOLOR_ILU`, `MULTICOLOR_GS` — 多色 ILU / GS。**有趣**: AMGx 内部已经实现了我们 Phase 3 的等价物! 可作为 sanity check / 对比。
- `NOSOLVER` — 没有 preconditioner (用于 pure CG 对照)。

**Coarsening 选项** (在 AMG config 里):
- `PMIS` (Parallel Maximal Independent Set, 经典 Cleary 风格) — 默认。
- `HMIS` (Hybrid MIS) — 更激进的粗化。
- `SIZE2`, `SIZE4`, `SIZE8` aggregation — 固定大小聚合。

**Interpolation 选项**:
- `CLASSICAL` (Ruge-Stüben 公式, §2.3 (2.4))。
- `DISTANCE1`, `DISTANCE2` — 缩写版本, 成本-精度权衡。
- `AGGREGATION_INTERPOLATOR` — SA 专用。

**Smoother 选项** (每层 smoother):
- `JACOBI_L1`, `BLOCK_JACOBI`, `MULTICOLOR_GS`, `CHEBYSHEV_POLY`, `NOSOLVER`。

**API 输入**: CSR 格式 (`row_ptr`, `col_idx`, `values`) — **与我们 Phase 1/2/3 完全一致**, 零格式转换。

- `int` 类型: 32-bit (AMGx 默认 `int32` for indices); 对 $N < 2^{31} \approx 2 \times 10^9$ 够用, 我们远低于。
- `double` (float64) values: 默认, 与我们一致。AMGx 也支持 `float` 与 block 版本 (dblock, fblock)。

**关键 API 调用序列**:

```
AMGX_initialize()
AMGX_config_create(config_handle, JSON_string)
AMGX_resources_create_simple(resources_handle, config_handle)

AMGX_matrix_create(matrix_handle, resources, mode)
AMGX_matrix_upload_all(matrix_handle, n, nnz, block_dim_x, block_dim_y,
                       row_ptr, col_idx, values, diag_data)

AMGX_vector_create(b_handle, resources, mode)
AMGX_vector_upload(b_handle, n, block_dim, b_values)

AMGX_vector_create(x_handle, resources, mode)
AMGX_vector_set_zero(x_handle, n, block_dim)

AMGX_solver_create(solver_handle, resources, mode, config)
AMGX_solver_setup(solver_handle, matrix_handle)      # <-- 重成本
AMGX_solver_solve(solver_handle, b_handle, x_handle)  # <-- 热路径

# 取解
AMGX_vector_download(x_handle, x_host_buf)  # OR 保留在 device

# 每时间步更新 (模式不变)
AMGX_matrix_replace_coefficients(matrix_handle, n, nnz, new_values, new_diag)
AMGX_solver_resetup(solver_handle, matrix_handle)  # OR skip, 见下文

# 清理
AMGX_solver_destroy(...); AMGX_matrix_destroy(...); ...
AMGX_config_destroy(config_handle)
AMGX_resources_destroy(resources_handle)
AMGX_finalize()
```

**`replace_coefficients` 行为** (WebSearch 2026-04 confirm): 要求稀疏模式 (`row_ptr`, `col_idx`) 不变; 只更新 `values` (+ 可选 `diag_data`)。之后 `AMGX_solver_resetup` 重建 AMG hierarchy。 **如果不 resetup, 收敛率会退化** (旧的 C/F 适配旧 values)。

**Open question (需验证)**: AMGx 是否有 "partial resetup" 选项, 仅更新 Galerkin 三积的 values 而不重新 coarsening?文档不明示; 需 Phase 4 实现时实测。

### 4.3 VRAM 需求估算

AMGx setup 需要所有层 $A_\ell$, $P_\ell$, 以及内部 workspace。经验系数:

$$
\text{VRAM}_{\text{AMG}} \approx (C_{\text{op}} + \sigma_{\text{op}}) \cdot (\text{sizeof}(A)) + \text{workspace}
$$

其中 $\sigma_{\text{op}} = \sum_\ell \mathrm{nnz}(P_\ell) / \mathrm{nnz}(A)$ 是 "prolongator 复杂度" 类。典型 $\sigma_{\text{op}} \sim 1-2$, $C_{\text{op}} \sim 2$, 所以总系数 $\sim 3-4 \times$ 原矩阵内存。

**我们 128³ 数字**:
- $\mathrm{nnz}(A) = 14.58 \times 10^6$
- $\text{sizeof}(A) = 8 \cdot 14.58\text{M} + 4 \cdot 14.58\text{M} + 4 \cdot 2.1\text{M} = 116 + 58 + 8 \text{ MB} \approx 182 \text{ MB}$
- VRAM AMG $\approx 3.5 \times 182 = 637$ MB
- 加上 PCG workspace ($\sim 8$ vectors of 16.8 MB each = 134 MB)
- **AMG total $\approx 770-850$ MB for 128³**.

3050 Laptop 4 GB VRAM 中可用 $\sim 2-2.5$ GB (扣除 desktop/WSL2 占 $\sim 1.5$ GB), 128³ 应 **刚好可行**, 但 256³ (~5.5 GB 外推) **OOM**。

**Phase 4 验收 §6 条目 6**: 128³ AMG 必须 $\leq 2$ GB VRAM (安全 margin)。

### 4.4 pyamgx 的状态

**`shwina/pyamgx`**: 
- Python 绑定 via Cython。
- 接口: `pyamgx.Matrix`, `pyamgx.Vector`, `pyamgx.Solver`, `pyamgx.Config`。
- 矩阵上传: `matrix.upload_CSR(scipy_csr)` — **从 SciPy CSR (host memory)**。
- 也支持 `matrix.upload(row_ptr, col_idx, values)` 其中 `row_ptr` 等是 NumPy array (host)。
- 文档声称支持 "NumPy arrays and Numba DeviceArrays"; **Numba DeviceArray** 暗示可能有 device-pointer 路径。需 git clone 后验证源码。

**问题 (Phase 4 风险)**:

1. **pyamgx 的 `upload` 签名接受 JAX DeviceArray 吗?** 
   - JAX 的 DeviceArray 实现 `__cuda_array_interface__` v2 协议。Numba DeviceArray 也实现同协议。
   - 理论上: 若 pyamgx 的 Cython 代码走 `__cuda_array_interface__` 读取 device pointer, 那 JAX DeviceArray 可无缝工作。 **需验证** (源码 inspection)。
   - 若 pyamgx 仅接受 SciPy CSR / Numba DeviceArray (有硬编码类型检查), 则需要 **host round-trip** — 破坏 "零拷贝" 目标。

2. **pyamgx 维护度**: WebSearch 显示 2025-05 和 2026-03 都有新 issues, 但无 release 更新。**版本号停留在 0.1**。 **较不确定** 是否能支撑我们生产级的稳定性要求。

3. **JAX-pyamgx 集成**: 未找到既有示例/文档。我们是 "first-of-a-kind" 集成, 风险需自担。

### 4.5 备选路径: hypre BoomerAMG

若 AMGx / pyamgx 路径死, **备选**: hypre `BoomerAMG` via 自定义 C++ FFI。

- hypre 活跃度: **优于 AMGx** (LLNL 主持; exascale computing project 支持)。
- GPU 后端: 成熟, CUDA / HIP / SYCL。
- 纯 C 接口: 易于做 FFI。
- License: LGPL (注意! 与 AMGx 的 BSD-3 不同; 影响分发)。
- 劣势: **API 更复杂** (hypre 用自己的 struct matrix/vector 格式, 而非裸 CSR); FFI 工作量比 AMGx 多 30-50%。

**决策路径** (Phase 4 内部):
- 优先尝试 pyamgx (最小代码代价)。
- 若 pyamgx device-pointer 不通, 自写 `libamgx_wrapper.so` (C-ABI), 路径类似 Phase 2 的 cuSPARSE wrapper。
- 若 AMGx v2.4.0 对 CUDA 12.4 不兼容或性能差, **切换到 hypre BoomerAMG**。这是一个 **项目级 pivot**, 会额外增加 1-2 周工时, 需用户明确签字。

---

## §5 零拷贝集成的数学数据契约

这是工程架构 (jax-cfd-am-expert) 的主战场, 但以下 **数学/算法侧的不变量** 需由本人给出, 作为工程侧的输入规范。

### 5.1 输入契约 (JAX → AMGx)

AMGx 需要的 per-solve 数据:

| 数据 | 类型 | shape | 语义 | 生命周期 |
|---|---|---|---|---|
| `row_ptr` | `int32[n+1]`, device | CSR 行指针; `row_ptr[0]=0, row_ptr[n]=nnz` | 模式的一部分; 不可变 across timesteps (pattern invariance) | setup 开始到 solve 结束 |
| `col_idx` | `int32[nnz]`, device | CSR 列索引; **每行 sorted ascending** (AMGx 要求, 与 cuSPARSE 相同) | 模式; 不可变 | 同上 |
| `values` | `float64[nnz]`, device | CSR 非零值 | 每时间步可变 (via `replace_coefficients`) | solve 期间不可变 |
| `diag_data` | `float64[n]` or NULL, device | 可选: 显式对角. AMGx CSR 语义: 若 non-null, 对角从此数组取; 若 NULL, 从 `col_idx`/`values` 里找 `col == row` 的项 | 随 values 一起变 | 同 values |
| `b` (RHS) | `float64[n]`, device | 右端 $b$ | 每 solve 都变 (每 PCG 新 call 新 b) | solve 开始到结束 |
| `x0` (initial guess) | `float64[n]`, device | 初始猜测 $x_0$ (或全 0) | 每 solve 可独立选 | solve 输入阶段 |

**关键模式不变性**: `row_ptr` 和 `col_idx` 在 AM 模拟中 **静态** (静态网格假设), 因此 AMGx setup 可重用。只有 `values`, `b`, `x0` 每步改变。

### 5.2 输出契约 (AMGx → JAX)

| 数据 | 类型 | shape | 语义 | 何时读 |
|---|---|---|---|---|
| `x` (solution) | `float64[n]`, device | AMGx 写入的解向量 | solve 完成后 | 下一步 PCG 迭代或下一时间步 |
| `iter_count` | `int`, host (scalar) | AMGx 实际 PCG 迭代数 | `solve` 返回后 | 诊断 + benchmark |
| `status` | enum, host | AMGx 内部求解状态: `SUCCESS`, `DIVERGED`, `NOT_CONVERGED`, `ABORTED` | 同上 | 错误处理 |
| `final_residual` | `float64`, host (scalar) | 最终相对残差 | 同上 | 诊断 |

**允许的 host-device 通信**: **仅标量** (`iter_count`, `status`, `final_residual`)。与 Phase 2 的 "8-byte 标量例外" 约定一致。

**严格禁止**: 解向量 `x` 的 host-device 拷贝不进入热路径。它必须留在 device。

### 5.3 模式变化语义

若 `row_ptr` 或 `col_idx` **实际改变** (adaptive refinement, Phase 5 功能, 不在 Phase 4 范围):
- 必须调用 `AMGX_solver_destroy` + `AMGX_matrix_destroy` + 重建。
- AMGx 没有 `updatePattern`-类 API, 也没有合理的途径在不重建对象的情况下换 pattern。
- Phase 4 明确 **scope**: 静态网格, pattern 在解的整个生命周期内不变。

### 5.4 Setup 复杂度 vs 时间步的相互作用

摊销语义 (§3.5):

- **完整 setup** (每 $K$ 步一次): 重做 C/F + 所有层 Galerkin。成本 $T_{\text{setup}}$, 典型 $5-10 \times$ 一次 solve。
- **refresh** (模式+C/F 不变, 仅 values 更新): 重做所有层 Galerkin 的 SpGEMM, 但跳过 C/F 启发式。AMGx 称之为 "resetup" (需要验证这是否是官方术语或 pyamgx wrapper 术语)。成本 $T_{\text{refresh}} \approx 2-4 \times$ 一次 solve。
- **无 resetup** (values 变但 hierarchy 用旧的): 收敛率下降, 可能从 30 iter 增到 50-100 iter。只在紧急 (时间预算紧) 场景用。

**数学契约 for jax-cfd-am-expert**:
- FFI API 必须暴露 `solver_setup` 和 `solver_resetup` (或等价) 两个独立 handle-bound 调用。
- 热路径 (`solver_solve`) 不触发 setup/resetup — 调用者显式控制。
- JAX Python 侧维护 "setup cadence counter" — 每 $K$ 步一次完整 setup, 其余步 resetup 或 nothing。

**推荐 $K$** (凭 AMGx 最佳实践):
- 静态网格 + 密度缓变 (normal AM): $K = 50-100$ 步。
- 密度快速变化 (keyhole 剧烈振荡): $K = 10-20$ 步。
- Phase 4 benchmark: 测 $K \in \{1, 10, 50, \infty\}$, 找最优。

### 5.5 JAX DeviceArray 指针生命周期

**核心风险** (master plan 原话回音):

> "如何将 JAX Device 上的 CSR 格式稀疏矩阵（row_ptrs, col_indices, values 的 Device 物理地址）直接喂给 AMGx, 而不需要先转回 CPU 或者进行昂贵的显存间复制。"

**数学不变量** (架构方必须保证):

1. **跨 FFI call 的指针稳定性**: AMGx 在 setup 和后续 solve 之间假设 matrix values pointer 稳定 (或显式 replace)。JAX DeviceArray 在 `jax.jit` 环境下 **pointer 可能重分配** (allocator reuse + buffer donation 风险)。

2. **所有权边界**: AMGx 不 copy, 只 reference (as much as possible; `AMGX_matrix_upload_all` 实际 copies by default — 有 `_pinned`, `_global`, `_distributed` 变体行为不同)。要验证 `AMGX_matrix_upload_all` 的 copy-vs-reference 语义。

3. **生命周期同步**: AMGx object (matrix, solver) 的销毁必须 happen-before 底层 values buffer 释放。

**工程 burden**: jax-cfd-am-expert 要设计一个 Python-side manager object 持有:
- `AMGX_resources_handle` (单例 per device, process lifetime)。
- `AMGX_config_handle` (单例 per config, process lifetime)。
- `AMGX_matrix_handle` + `AMGX_solver_handle` (per pattern; 由 Python-side cache 保活)。
- JAX 侧 matrix buffers (donation-aware, 避免 XLA 销毁)。

**opaque token 模式** (回响 Phase 2 Option C): 把 matrix/solver handles 编码为 `uint64` 跨 FFI 边界传递。 **这是一个架构决策, 工程方负责**。

### 5.6 我不要求工程方做什么

Phase 4 math doc **不规定**:
- 具体 FFI handler 签名 (how many primitives: 1? 4? 8?)。
- Python manager 类的具体形态。
- JSON config 的构造位置 (Python or C++ side)。
- 如何从 JAX `jax.ffi.pycapsule` 传递 resources/config/solver handles。

这些是工程方的自由度。数学方只要求 **不变量**:
(1) device pointer 不被静默 copy; (2) 模式不变时 setup 可重用; (3) values 更新走 `replace_coefficients` 路径; (4) 标量状态可 D→H 读取。

---

## §6 Phase 4 验收标准 (绑定)

Phase 4 "完成" 的数学定义 = **全部** 以下 8 条 PASS:

### 6.1 C1 — 正确性 (16³ 刚性)

对 Phase 2 T7 (16³ 沿 z 中面 contrast=100) 矩阵, AMG-PCG 求解得到 $x_{\text{AMG}}$, DILU-PCG 求解得到 $x_{\text{DILU}}$, 相对误差:

$$
\frac{\|x_{\text{AMG}} - x_{\text{DILU}}\|_\infty}{\|x_{\text{DILU}}\|_\infty} \leq 10^{-8}.
$$

**理由**: 两者都收敛到同一线性系统 $Ax = b$ 的唯一解 (SPD 系统). $10^{-8}$ 是 PCG 容差 $10^{-10}$ 的合理放大 (考虑浮点误差 + 预处理器不同导致的 Krylov 子空间不同)。

**测试**: 16³ 上 AMG-PCG tol=1e-10, DILU-PCG tol=1e-10, 对比两解。

### 6.2 C2 — 迭代数压缩 (128³ 刚性)

AMG-PCG 在 128³ T7 刚性矩阵上, 收敛到 tol=1e-10 的迭代数:

$$
N_{\text{AMG}} \leq 50.
$$

**理由**: Phase 2 实测 $N_{\text{DILU}} = 186$; §3.3 预测 AMG 20-50 iters。50 是宽松上界, 给 AMG 一次 "未调优" 的宽容。若超过 50 iters, 需 debug AMG config 或质疑 AMGx 对我们问题的适配性。

**压缩比**: $186 / 50 = 3.72 \times$ 最低。

### 6.3 C3 — Wall-time 胜过 (128³ 刚性)

AMG-PCG 总 wall time (包括 setup 或 resetup 成本, 对首次调用即 setup) 在 128³:

$$
T_{\text{AMG, total}} < 0.5 \cdot T_{\text{DILU, total}} = 0.5 \cdot 28.47 \text{s} = 14.24 \text{s}.
$$

**理由**: 这是 "AMG 值得复杂性" 的门槛。若 AMG 总 wall time 反比 DILU 慢 (pathological case), Phase 4 失败 — 硬件 (3050 Laptop) 或 AMGx config 与我们问题不匹配, 需另寻出路。

**注意**: C2 + per-iter cost 合理 (§3.4 给 2-5× DILU per-iter) 自动蕴含 C3 大致, 但 C3 独立测量防止 "iter 对但 per-iter 慢 10×" 的退化情形。

### 6.4 C4 — 物理基准 (64³)

Phase 2.5 的 Test A 和 Test B 在 64³ 网格上 (Phase 3 scaling report 已测过 DILU 版本), AMG 版本必须通过同样的阈值:

- **Test A (variable-density projection)**: $\max|\nabla \cdot \mathbf{u}| \leq 4 \times 10^{-8}$ (Phase 3 实测 DILU 1.25 × 10^{-8}; 允许 AMG 稍宽到 4e-8, 因为 AMG 精度曲线不同)。
- **Test B (static droplet, 10 steps)**: $\|u\|_\infty @ t=10 \leq 3 \times 10^{-6}$; $\max|\nabla \cdot \mathbf{u}| \leq 1 \times 10^{-12}$。

**Test C (32³ 三层密度) 的修改版** 专为 AMG 设计:

- 迭代数固定为 **5-15 iter** (而不是 DILU 用的 50 iter), 测 $\text{corr}(|E_{\text{demean}}|, |\nabla \log \rho|) \leq 0.25$。
- **关键差异**: AMG 应该在 5-15 iter 内接近收敛 (不是 mid-convergence); 如果 5-15 iter 时残差仍有 interface geometry 相关, 说明 AMG 对密度跳跃处理不当 (coarsening 或 interpolation 在跳跃处失效)。

### 6.5 C5 — 零拷贝保持

在 `jax.jit` 下 compile 后的 HLO:
- `custom-call` 数 = 预期 FFI 数 (1 setup + N×(1 solve) + 可选 resetup)。
- `copy-start` / `copy-done` count = 0 for matrix data, b, x0, x。 **允许** 的 host-device 拷贝仅标量 (iter count, status, residual)。

继承 Phase 1 / Phase 2 / Phase 3 的 HLO inspection 方法论, 测试用 `jax.jit(fun).lower().compile().as_text()` 后 grep。

### 6.6 C6 — VRAM 预算

128³ AM-realistic 问题: AMG setup + solve 的 peak JAX-attributable VRAM $\leq 2$ GB (3050 Laptop 4 GB 卡的硬上限 $\sim$ 2.5 GB 可用, 留 500 MB margin)。

**测量**: `nvidia-smi` 在 AMGx setup 完成但未 solve, 以及 solve 进行时各采样一次。取 max。

### 6.7 C7 — 可靠性

AMGx 在以下情景下 **不崩溃, 不 hang, 不静默 diverge**:
- 16³ / 32³ / 64³ / 128³ T7 刚性矩阵 (4 个 size)。
- 64³ 物理测试 A/B (变系数 + CSF)。
- 32³ 三层密度 (Test C 修改版)。

Setup 阶段 (128³) 完成时间 $\leq 5$ s。若 $> 5$ s, 调查是否 config 不当 (e.g., coarsening 过慢), **不是** 直接 fail。

Divergence 可接受 only if AMGx 返回明确的 `AMGX_SOLVE_DIVERGED` 状态, 我们代码路径捕获并 raise Python exception。**静默** divergence (解是 NaN 但返回 SUCCESS) 是严重 bug, 视为 fail。

### 6.8 C8 — Setup 摊销

Setup cost + 50-iter solve 在 128³ 的 wall time $\leq$ Phase 2 DILU 的 186-iter solve wall time (28.47 s)。这确保即使 Phase 4 只做 "一次性 solve" (极端不利于 AMG, 因为 setup 无法摊销), AMG 仍不比 DILU 慢。

若 $T_{\text{setup}} + 50 T_{\text{iter,AMG}} > 28$ s, Phase 4 失败 — 说明 3050 Laptop 上 AMGx 的绝对性能对我们的问题规模不 fit。

---

## §7 Phase 4 失败模式 & Fallback

以下是 honest, 带概率估计。

### 7.1 F1 — pyamgx bindings 死/坏

**表现**: `import pyamgx` 失败, 或 `pyamgx.Matrix(...)` 不接受 JAX DeviceArray, 或运行时段错误。

**概率**: **中等** (pyamgx 版本号 0.1 停滞多年, Cython 代码可能未跟上 CUDA 12.x)。

**检测信号**: Phase 4 实现第 1 日: clone pyamgx, 跑其自带 tests/examples。若 >50% 测试崩溃或结果错, 即触发 F1。

**Fallback**: 自写 C++ FFI wrapper `libamgx_wrapper.so`, 直接调用 AMGx 的 C API (`amgx_c.h`)。架构模式: 借用 Phase 2 cuSPARSE wrapper 的 opaque token 设计, 4-6 个 FFI primitives:
- `amgx_initialize` (一次 process lifetime)
- `amgx_setup(csr_tuple, config_json) -> solver_token`
- `amgx_solve(solver_token, b, x0) -> x`
- `amgx_replace_coefficients(solver_token, new_values) -> new_solver_token` (or in-place)
- `amgx_resetup(solver_token)`
- `amgx_destroy(solver_token)`

工作量: 比 Phase 2 多约 1.5× (AMGx API 复杂度略高; config JSON 管理)。

### 7.2 F2 — AMGx 对我们矩阵模式崩坏

**表现**: 常规 Poisson 可用, 但在 AM 密度跳跃矩阵上 setup 崩 (C/F 分裂失败, 或 Galerkin 三积产生奇异矩阵) 或 solve 不收敛。

**概率**: **未知** (我们矩阵与 AMGx 文献测试集合无重叠)。

**机制**: AMGx 对 M-matrix 假设较强; 变系数 cut-cell 离散化可能产生 "几乎 M-matrix但不完全" 的矩阵 (非对角小正项出现)。 classical RS coarsening 对此敏感, SA 更 robust。

**检测信号**:
- `AMGX_solver_setup` 返回非零状态码或 throw。
- 设置 verbose=2 看 setup log 里 "无 C neighbor 的 F 节点" 警告。
- 收敛历史: 前几 iter 迅速下降然后 stall。

**Mitigation**:
1. 从 simpler setting 起步: 先在 **常系数** 3D Poisson (16³ 均匀) 跑通 AMGx, 验证 API/build 正确。
2. 逐步引入刚性: 常系数 → z-midplane contrast 10 → contrast 100 → 三层 gas/liquid/solid。
3. 若 classical RS 崩, 切换 `algorithm=AGGREGATION` (SA) 重试。
4. 若仍崩, 试 `JACOBI_L1` preconditioner (退化为 single-level, 但应至少不崩), 定位到底是 AMG-specific 还是 AMGx-lib 的 bug。

### 7.3 F3 — AMG setup > 186 DILU iters 在小网格

**表现**: 16³ 或 32³ 上, AMG setup 一次花 2-10 s, 而 DILU 186 iters 只要 0.5 s。C3 条件挡住此情形。

**概率**: **高** (AMG setup 成本结构对小问题非利, 这是 well-known)。

**定性预期**: $n \leq 10^4$ AMG 必败, $n \geq 10^5$ AMG 开始胜。

**Routing rule** (pre-commit 明示, 不是 Phase 4 运行时自适应):
- 小问题 (16³ / 32³): Phase 2 或 Phase 3 DILU。
- 大问题 (64³+): Phase 4 AMG。
- 切换点不需要自动决策; 由 jax 层的问题构造方按 N 选择 preconditioner 实现。

### 7.4 F4 — 3050 Laptop FP64 throttle 吞噬优势

**表现**: AMGx 正确运行但 per-iter 慢 10×+。C3 挡不住的话 (边缘命中), Phase 4 "技术成功但工程无效"。

**概率**: **中等** (3050 Laptop FP64 是 FP32 的 1/32; AMGx V-cycle 内 Chebyshev/Jacobi smoother + Galerkin 三积 都是 FP64-heavy)。

**Mitigation**:
1. 明示: Phase 4 在 3050 上的绝对数字不代表 A100 / H100 表现。C3 (0.5× DILU wall) 是 3050-specific; A100 预计 C3 可达 0.1-0.2×。
2. 不在 3050 上投入重度优化 tuning; 目标是 "可运行 + 测试 API"。
3. 生产部署目标: 用户迁移到 server GPU 后再调参。

### 7.5 F5 — JAX 设备 buffer 被下抽

**表现**: `AMGX_matrix_upload_all` 记住了 values pointer, 但 JAX XLA 在后续 `jit` 调用中 reclaim/reuse 了该 buffer。下次 `AMGX_solver_solve` 调用读到旧/脏数据, 得 NaN 或错解。

**概率**: **中高** (JAX 的 buffer donation + allocator reuse 确实会动 pointer; 这是 Phase 2 cuSPARSE 集成踩过的坑, Phase 4 在 AMGx 上放大)。

**Mitigation** (工程方责任):
1. Python manager 侧显式 hold DeviceArray references, 阻止 GC。
2. 如 values 需更新, **copy into existing buffer** (不要 reallocate); 或调用 `replace_coefficients` 并传新 pointer。
3. 每次 solve 前, 用 `jax.block_until_ready()` 确保 buffer 已 materialize 且不会被异步 reclaim。
4. 单元测试: 故意触发 buffer reuse (创建多个大 arrays 竞争 allocator), 验证 AMGx 仍正确。

---

## §8 用户签字验收清单 (8 条)

下列 8 条是 Phase 4 实现正式启动前, 用户必须书面签字确认的:

1. **Scope 定于集成, 不从零实现 AMG**: Phase 4 是把 NVIDIA AMGx (或 fallback hypre BoomerAMG) 集成到 JAX-FFI stack, **不**是重写 AMG 算法。失败时切换到 hypre 是 Plan B-of-Plan-B, 需用户额外签字。

2. **Gustafsson $N^{1/3}$ 尺度律已实测确认** (§1.1): 16³/64³/128³ 三点拟合精度 3%。这为 "为什么 DILU 对生产不够用" 的论点提供了数据底座, **不是** hand-wave 推测。

3. **Classical RS-AMG 是默认, SA 是 fallback**: Phase 4 默认用 `algorithm=CLASSICAL` 的 AMGx 配置; 若对我们不连续系数矩阵崩坏, 切 `algorithm=AGGREGATION`。进一步崩坏, 切 hypre。

4. **8 条 C1-C8 验收标准 (§6) 全部 PASS** 才算 Phase 4 完成。任意一条挂, Phase 4 未完成。具体阈值:
   - C1: 解相对误差 $\leq 10^{-8}$。
   - C2: 128³ AMG iter $\leq 50$。
   - C3: 128³ 总 wall time $\leq 14$ s。
   - C4: 物理测试 A/B/C(modified) 全过。
   - C5: HLO 中 data copy 数 = 0。
   - C6: VRAM $\leq 2$ GB @ 128³。
   - C7: 无崩溃, setup $\leq 5$s。
   - C8: Setup + 50 iter < DILU 186 iter wall time。

5. **零拷贝数据契约 (§5)** 固定: `row_ptr` / `col_idx` 跨时间步不变; `values` 变则走 `replace_coefficients` + `resetup`; 热路径 D↔H 拷贝仅限标量 (iter count, status, residual)。

6. **5 个已识别失败模式 (§7 F1-F5) 的 detection + fallback 路径全部文档化并在实现中实施**。pyamgx 死亡 (F1) 的 fallback 是自写 C++ wrapper (增 1.5× 工作量); F2 的 fallback 是 SA; F4 是声明 "3050 绝对数字不是生产指标"。

7. **Routing rule 明示** (§7.3): 16³ / 32³ 用 Phase 2/3 DILU, 64³+ 用 Phase 4 AMG。切换点不 adaptive; 由 Python 构造方按 $N$ 显式选。

8. **范围外 (Phase 4 不做)**: 
   - Adaptive mesh refinement 引发的 pattern-change 支持 (Phase 5+)。
   - Multi-GPU 分布式 AMGx (AMGx 支持, 但我们只做单 GPU)。
   - Float32 AMG (Phase 4 纯 float64 保持 AM physics-grade 精度)。
   - hypre BoomerAMG 集成作为主路径 (仅作 F1 fallback)。
   - AMG 算法自身的研究与发明 — 我们是 **调用者**, 不是 **作者**。

---

## §9 工程侧 (jax-cfd-am-expert) 交接规格

本节把 §1-§7 的数学要求翻译成工程代码中必须存在的 artifact 类型。具体实现由工程方自由选择, 但必须覆盖以下功能点。

### 9.1 需要的 FFI primitives (最少 6 个)

按 Phase 2/3 的 opaque token 模式:

| # | Primitive | 输入 | 输出 | 热路径? |
|---|---|---|---|---|
| 1 | `amgx_initialize` | — | resources token (process singleton) | No (one-shot per process) |
| 2 | `amgx_setup` | `(row_ptr, col_idx, values, config_json)` | solver token + matrix token | No (per pattern, 可能每 K 步一次) |
| 3 | `amgx_solve` | `(solver_token, b, x0) -> x, iter_count, status, residual` | device x + host scalars | **Yes** |
| 4 | `amgx_replace_coefficients` | `(solver_token, new_values)` | — | No (per values update, 每步一次) |
| 5 | `amgx_resetup` | `(solver_token)` | — | No (与 replace 配对或独立) |
| 6 | `amgx_destroy` | `(solver_token)` | — | No (teardown) |

**可选第 7**: `amgx_refresh_coefficients_inplace` 合并 (4) + (5), 减少 FFI dispatch。Phase 2/3 经验显示这对 hot path 值得融合。

### 9.2 Handle 生命周期 (Python 侧)

类似 Phase 2/3 的 `SpSVContext` / `MulticolorContext`:

```
class AMGxContext:
    # 持有下列 opaque handles (uint64 tokens):
    resources: int        # 全进程单例
    config: int           # 每 config 一个 (可能多个, 如我们测不同配置)
    matrix: int           # 每 pattern 一个
    solver: int           # 与 matrix 绑定
    # 以及 Python 侧缓存:
    pattern_fingerprint: tuple  # (n, nnz, row_ptr_hash, col_idx_hash)
    setup_count: int            # K-step amortization counter
    step_count: int             # 外部 stepper 自增
```

- **Pattern fingerprint**: 继承 Phase 2 教训 (device pointer 在 `jit` 下不稳), 用 `(n, nnz, row_ptr_sha1, col_idx_sha1)` 代替 pointer 做 cache key。
- **Setup 策略**: 每 $K$ 步一次完整 `amgx_setup`, 其余步 `amgx_replace_coefficients` + `amgx_resetup`。$K$ 作为 `AMGxContext` 构造参数。

### 9.3 测试矩阵集合

Phase 4 benchmark 要覆盖的 4 个规模:

| 规模 | 矩阵 | 期望 AMG iter | 期望 DILU iter (参考) |
|---|---|---:|---:|
| 16³ | T7 刚性 (z-midplane contrast 100) | 5-10 | 24 |
| 32³ | T7 刚性 | 10-20 | 100-128 |
| 64³ | T7 刚性 | 15-30 | 99 |
| 128³ | T7 刚性 | 20-50 | 186 |
| 32³ | 三层 (Test C) | 8-15 | 50 (mid-conv) / 128 (converged) |
| 64³ | variable-density 球 (Test A) | 15-30 | 224 |
| 64³ | CSF droplet (Test B) | 15-30 per step | 166 per step |

### 9.4 Benchmark 结构

Phase 4 benchmark script (对应 Phase 2/3 的 `bench_*.py`) 必须测:
- **Setup time** (首次 + 每次 resetup): wall time mean/p95 across N repetitions。
- **Per-solve time**: wall time per `AMGX_solver_solve` call。
- **Iter count**: AMGx 报告的内部 PCG iter。
- **Total wall time**: setup + 所有 solves 的总和 (对比 DILU 同样结构)。
- **VRAM peak**: via `nvidia-smi` snapshot at critical points。
- **Amortization study**: 固定 pattern, 变 $K \in \{1, 5, 20, 100, \infty\}$ 的 total wall time vs step count 曲线。

### 9.5 Config JSON 最小默认

作为 Phase 4 起点 config (工程方可调整):

```
{
    "config_version": 2,
    "determinism_flag": 1,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "interpolator": "CLASSICAL",
            "selector": "PMIS",
            "max_levels": 20,
            "cycle": "V",
            "presweeps": 2,
            "postsweeps": 2,
            "smoother": {
                "solver": "CHEBYSHEV_POLY",
                "relaxation_factor": 0.9,
                "chebyshev_polynomial_order": 3
            },
            "coarsest_sweeps": 4,
            "coarse_solver": "DENSE_LU_SOLVER"
        },
        "max_iters": 100,
        "convergence": "RELATIVE_INI",
        "tolerance": 1e-10,
        "norm": "L2",
        "print_solve_stats": 1,
        "obtain_timings": 1,
        "monitor_residual": 1
    }
}
```

**说明**:
- `determinism_flag: 1` — 关键, 保证每次运行结果 bit-identical (我们在 Phase 1 就坚持 no `-ffast-math` 原则)。
- `solver: PCG` — SPD 对称系统首选。
- `preconditioner.solver: AMG` — 嵌套 AMG 作为预处理器。
- `algorithm: CLASSICAL` — RS-AMG。
- `selector: PMIS` — 并行 MIS C/F 分裂 (GPU-friendly)。
- `smoother: CHEBYSHEV_POLY` order 3 — 适合 GPU (完全并行, 无同步)。
- `coarsest_sweeps: 4` + `DENSE_LU_SOLVER` — 最粗层足够精度。

Phase 4 调优可从此起点微调; 但不必预先尝试所有组合, 先跑通再优化。

### 9.6 与 Phase 2/3 互不干扰

Phase 4 代码落入 `dilu/amgx/` 子目录 (类比 `dilu/cusparse/`, `dilu/multicolor/`)。不修改 Phase 1/2/3 artifact。

Python 层 routing: `from dilu.amgx import AMGxContext` 与现有 `from dilu.cusparse import SpSVContext` / `from dilu.multicolor import MulticolorContext` 并列; 用户按 $N$ 和问题类型显式选。

---

## §10 References (作者 + 标题, 不伪造 DOI)

- Ruge, J. W. & Stüben, K. "Algebraic multigrid", in *Multigrid Methods* (McCormick, ed.), SIAM Frontiers in Applied Math vol. 3, 1987. — **canonical RS-AMG 定义**。
- Stüben, K. "A review of algebraic multigrid", *J. Comput. Appl. Math.*, 2001. — AMG 综述, 收敛理论与实现工艺。
- Brandt, A. "Algebraic multigrid theory: The symmetric case", *Appl. Math. Comput.*, 1986.
- Hackbusch, W. *Multi-Grid Methods and Applications*, Springer, 1985 — two-grid convergence theorem 原始证明。
- Naumov, M. et al. "AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods", *SIAM J. Sci. Comput.*, 2015 — **AMGx 论文**, 性能数据与 API 总结。
- Vanek, P., Mandel, J., Brezina, M. "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems", *Computing*, 1996 — SA 原始公式。
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003 — §13 (Multigrid Methods) 给 AMG 和 GMG 的完整教学介绍。
- Gustafsson, I. "A class of first order factorization methods", *BIT*, 1978 — DILU 的 $\sqrt{\kappa(A)}$ 尺度律 (我们 §1.1 的反驳基础)。
- Trottenberg, U., Oosterlee, C. W., Schüller, A. *Multigrid*, Academic Press, 2001 — 综合 textbook。
- NVIDIA AMGx 官方文档: `github.com/NVIDIA/AMGX/doc/AMGX_Reference.pdf` (v2.x).
- pyamgx 文档: `pyamgx.readthedocs.io/en/latest/` (v0.1)。
- hypre BoomerAMG 文档: `hypre.readthedocs.io/en/latest/solvers-boomeramg.html`。
- Desjardins, O., McCaslin, J., Owkes, M., Brady, P. "Methods for multiphase flows with high density ratio", Stanford CTR Summer Program, 2010 — AM-adjacent 多相流实际 solver 性能参考。
- Yang, Y., Bao, W., Gibou, F. "A fast pressure-correction method for incompressible two-fluid flows", *J. Comput. Phys.*, 2014 — multiphase pressure Poisson 与 AMG 加速数据。

---

## §11 附 — 与前序 Phase 的对齐

- **Phase 1 §3.3 的 Route C 预览** 在本文 §2 被正式展开。当时 "AMG 压 iter 5-30" 的粗略估计, 本文 §3.3 细化到 "AM discontinuous 下 20-50"。
- **Phase 2 §4.1 的 Gustafsson $\kappa^{1/4}$ 预测** 在本文 §1.1 被 Phase 3 scaling 数据 **验证** — 实测是 $\sqrt{\kappa(A)}$ ($N^{1/3}$) 而不是最优 $\kappa^{1/4}$ ($N^{1/6}$), 确认 "Gustafsson 在不连续系数下崩坏" 的论断。
- **Phase 3 §1.1 区分 (a) reordered DILU vs (b) block-Jacobi-DILU** 的教训延续到本文: AMGx 提供的 `MULTICOLOR_ILU` preconditioner 可作为 Phase 3 multicolor 工作的第三方独立验证点 (同样是 (a) 思路)。
- **Phase 1/2/3 的 "opaque uint64 token" 模式** 在本文 §9 被重用作 AMGx 集成架构的中心 — 因为它已被证明是 JAX 无状态范式下管理 C 库 handle 的正确答案。
- **Phase 2.5 的物理健康测试** 被本文 §6.4 扩展为 AMG-specific 变体: Test C 的 iter 数从 50 降到 5-15 (因为 AMG 5-15 iter 就应该接近收敛), 但 corr 阈值保持 0.25。

---

## §12 数学签字清单 (用户最终确认, 8 条)

以下是用户签字的 8-bullet 数学验收 (非工程), 供在 Phase 4 实现启动前签署:

1. **证据链**: Phase 3 scaling 64³/128³ 的 $N^{1/3}$ 拟合 (§1.1, 误差 3%) 和 Gustafsson 失效论 (§1.1) 已确立 "DILU 打不到 AM 生产分辨率" 的数学结论。
2. **替代方案正确性**: AMG (RS-classical) 对 SPD + 弱对角占优矩阵有 grid-independent 收敛的严格证明 (§2, §3.1), 文献实测支持 20-50 iter 对 $\gamma = 10^3$ 不连续系数问题 (§3.3)。
3. **Per-iter trade-off 明示**: AMG per-iter 比 DILU 慢 2-5× (§3.4); 总 wall time 胜过 DILU 的条件是 $N_{\text{DILU}}/N_{\text{AMG}} > 2-5$ (128³ 已满足: $186/50 = 3.7$; 256³ 外推更友好)。
4. **设置成本摊销**: AMGx setup 比 DILU factor 多 5-10× (§3.5); 每 $K \sim 20-50$ 步一次完整 setup 可把摊销比降到 1.5× DILU factor, 完全可接受。
5. **零拷贝契约可行**: `replace_coefficients` + `resetup` 的 API 支持 values-only 更新 (§4.2, §5.3); JAX DeviceArray 通过 `__cuda_array_interface__` 协议原则上可直喂 (§4.4), 但 F5 指针生命周期风险存在 (§7.5)。
6. **VRAM 约束**: 128³ AMG 预计 $\leq 1$ GB 在 3050 Laptop 4 GB 卡上可行 (§4.3)。256³ 目标 **不保证** 能跑在 3050, 需 A100/H100。
7. **5 种失败模式 + fallback 全文档化** (§7): pyamgx 死 → 自写 wrapper; AMGx 崩 → SA → hypre; 3050 throttle → 声明 "硬件限制不是算法限制"; buffer 生命周期 → Python manager 保活。
8. **Scope 边界**: Phase 4 是 AMGx 集成, 不是 AMG 发明; 不做 adaptive mesh, 不做 multi-GPU, 不做 float32; 不做 "DILU→AMG 自动切换" (由用户按 $N$ 显式选)。

---

*End of Phase 4 Part 1 math foundation. Engineering team (jax-cfd-am-expert) takes over for Part 2 zero-copy FFI architecture against §5 / §9 spec.*
