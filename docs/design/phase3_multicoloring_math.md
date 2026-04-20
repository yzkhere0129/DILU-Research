# Phase 3 — 多色 DILU 预处理器的数学/算法分析

**Project**: JAX-GPU AM-CFD Platform — Stiff Pressure Poisson Solver Kernel Research
**Phase**: 3 / 4 — Part 1 (Math / Algorithm). Engineering architecture is parallel jax-cfd-am-expert's Part 2.
**Author role**: CFD-math expert (this document)
**Date**: 2026-04-20
**Scope**: 纯数值分析，不写代码。为 Phase 3 工程实现建立数学契约。

---

## 0. 重新briefing自查清单

本文档基于以下已读材料:

- `docs/design/phase1_dilu_math_foundation.md` — 特别是 §1 (DILU 精确定义), §3.2 (Route B 的定性讨论), §4 (AM 刚性量化)。Phase 1 已经点明 "multi-coloring 不是 DILU 的 permutation-invariant 对象"，且在 $\gamma = 10^3$ 时迭代数增长"不确定，待测"。
- `docs/design/phase2_cusparse_level_scheduling_math.md` — 特别是 §3 (level count 估算 $L_{\max} = N_x + N_y + N_z - 2 = 446$ 对 $256{\times}128{\times}64$ 网格), §4 (AM 刚性迭代数预测 100–500), §5/6 (Phase 2 验收与交接契约)。
- `docs/benchmark/phase2_cusparse_report.md` — 实测结果: T7 16³ 刚性问题 **DILU-PCG 24 iters vs Jacobi-PCG 71 iters** (2.96× 改进, $\rho$-contrast = 100)。两个工程地雷: (i) `cusparseSpSV_updateMatrix` value-cache; (ii) pattern fingerprint 在 `jit` 下不稳定。
- `docs/benchmark/phase2.5_physical_report.md` — 物理健康基线: Test A max|∇·u| = 4.13e-9 (无界面 halo), Test B ‖u‖∞ 线性增长而非指数爆炸 (CSF discretization 伪流非求解器伪流), Test C mean-detrended 残差不追踪界面轮廓。
- `dilu/cusparse/tests/test_t7_pcg_stiff.py` — PCG driver，$16^3$ 沿 $z$-轴中间面 contrast=100 (harmonic-mean face coefs)。
- `dilu/cusparse/tests/physical_benchmark.py` — 三组物理测试 ($32^3$, $\rho$-contrast 1e3 / 8e3)。
- `CLAUDE.md` — 项目约束: `jax` 0.9.x + `jax.ffi` + C++/CUDA; no `-ffast-math`; float32 需 double accumulators; FFI 前 `np.ascontiguousarray`.

**已完成 WebSearch 交叉确认** (2026-04):

- Duff-Meurant 1989 "The effect of ordering on preconditioned conjugate gradient" (BIT 29:635–657) 存在并被引用。
- Saad 2003 *Iterative Methods for Sparse Linear Systems* §12 (multicoloring orderings 作为并行 ILU 手段，文本直接承认 "these techniques may degrade the performance and robustness of ILU preconditionings")。
- Benzi 2002 *JCP* 182:418–477 — 预处理综述确认 multicolor orderings 存在精度-并行权衡。
- Naumov-Castonguay-Cohen 2015 "Parallel Graph Coloring with Applications to the Incomplete-LU Factorization on the GPU" — **NVIDIA 官方方案**; 报告 up to 6× GPU speedup vs level-scheduled ILU(0), 其中 G3_circuit 矩阵从 "thousands of levels with ≤1000 rows" 降到 "9 levels with hundreds of thousands of rows"。
- Li & Saad 2010 "GPU-Accelerated Preconditioned Iterative Linear Solvers" (UMSI 2010-112) 明确: "MC-ILU(0) 比 natural-ordering ILU(0) 需要更多迭代，因为多色重排舍弃了更多耦合"。
- Block red-black 的 ICCG / blockILU 变体 (Doi-Lichnewsky 1990; Iwashita-Shimasaki 2003) 给出比 pure 2-color 更好的 trade-off，但 2026 年还没有公开发表的 "AM $\rho = 10^3$ 3D 多色 DILU" 实验数据。

**量化文献空白**: 搜索了 "红黑 ILU(0) + 3D Poisson + 变系数 + 迭代数" 这一精确交集，公开的具体数字仅限于 constant-coefficient 或 2-D 情形。对我们的 $\gamma = 10^3$ 3D AM 工况，**文献上不存在** 可直接引用的 "X colors → 1.Y× iter penalty" 数字。这意味着 Phase 3 的 T7-equivalent 本身就是一次数据贡献，不是简单对标已知基准。

---

## 1. 数学设置 — "多色 DILU" 精确刻画

### 1.1 两种含义必须先区分

文献中 "multi-color DILU" 有两种完全不同的含义。我们明确做哪一种非常关键，否则预处理器质量差异可达 $5–20×$。

#### 1.1.1 含义 (a): **Reordered DILU** (permutation-equivalent 重排 DILU)

给定 $A$ 的邻接图 $G(A) = (V, E)$，其中 $V = \{1, \ldots, n\}$，$E = \{(i,j) : i \neq j, A_{ij} \neq 0\}$。设 $\pi: V \to \{1, \ldots, C\}$ 是一个合法的 **图顶点染色**: 对所有 $(i,j) \in E$，$\pi(i) \neq \pi(j)$。

把 $\pi$ 重排为一个置换 $\sigma: V \to V$ 满足先按颜色升序、再按原始索引升序:

$$
\sigma(i) < \sigma(j) \iff \bigl(\pi(i) < \pi(j)\bigr) \vee \bigl(\pi(i) = \pi(j) \wedge i < j\bigr).
$$

设 $P$ 是对应的置换矩阵 ($(P\mathbf{x})_i = x_{\sigma^{-1}(i)}$)。重排矩阵:

$$
\tilde A = P A P^\top. \tag{1.1}
$$

**关键性质**:
- $\tilde A$ 与 $A$ 是相似矩阵 (conjugation by orthogonal $P$)，**谱相同**，特别地 $\kappa_2(\tilde A) = \kappa_2(A)$。
- $\tilde A$ 的 SPD / 稀疏模式 / 对角占优性质与 $A$ 完全一致。
- $\tilde A$ 的稀疏模式按颜色分块：把行按颜色分组，则块 $(c,c)$ (同色行内) 只有对角项；块 $(c,c')$ ($c \neq c'$) 一般稠密但仍受原 $E$ 约束。

在 $\tilde A$ 上应用 **标准 DILU 分解**:

$$
\tilde M = (\tilde D_* + \tilde L)\, \tilde D_*^{-1}\, (\tilde D_* + \tilde U), \tag{1.2}
$$

其中 $\tilde L, \tilde U$ 是 $\tilde A$ 的严格下/上三角 (按 $\sigma$ 诱导的新顺序)，$\tilde D_*$ 由 Phase 1 §1.2 的 DILU 递推给出:

$$
\tilde d_i = \tilde a_{ii} - \sum_{k<i,\, (i,k) \in \mathcal{S}(\tilde A)} \frac{\tilde a_{ik}\, \tilde a_{ki}}{\tilde d_k}. \tag{1.3}
$$

**$\tilde L, \tilde U$ 的颜色分块结构**: 若颜色 $c' = \pi(\sigma^{-1}(k)) < c = \pi(\sigma^{-1}(i))$ (即 $k$ 在颜色序列中先于 $i$)，则 $k < i$ 在新顺序下，因此 $(i,k) \in \mathcal{S}(\tilde L)$ 当且仅当 $(\sigma^{-1}(i), \sigma^{-1}(k)) \in \mathcal{S}(A)$。若 $\pi(\sigma^{-1}(i)) = \pi(\sigma^{-1}(k))$ (同色)，由染色合法性，$(\sigma^{-1}(i), \sigma^{-1}(k)) \notin E$，所以 $\tilde L$ **在同色行对之间没有非零**。

这即是并行性的数学根源: 求解 $(\tilde D_* + \tilde L)y = r$ 时，颜色 $c$ 内所有行只依赖颜色 $< c$ 的已知行，**颜色内行之间相互独立**。

#### 1.1.2 含义 (b): **Block-Jacobi-like DILU** (按颜色丢耦合)

有些文献 (尤其早期并行 ILU 加速文献的廉价变体) 取 $\tilde A$ 的**块对角部分** (按颜色分块)，对每个对角块单独做 DILU，颜色间耦合整体丢弃:

$$
\tilde M_{\text{BJ}} = \mathrm{diag}_{c=1,\ldots,C}(\mathrm{DILU}(\tilde A_{cc})).
$$

对 7-点 3-D Laplacian + 2-color (red-black)，每个对角块就是标量对角(因为同色行无耦合)，所以 $\tilde M_{\text{BJ}}$ 退化为 **Jacobi 预处理器**。Phase 2 实测 Jacobi-PCG 71 iter vs DILU-PCG 24 iter — 即 $2.96\times$ 差。

**结论**: (b) 不是我们要做的。本项目 Phase 3 做的是 **(a)**。明确区分已写入 §8 交接规格。

**Confidence: 确定**。

### 1.2 (a) 方案的预处理器代数 — 详细推导

记号: 把 $V = \{1, \ldots, n\}$ 按颜色排序后写成 $V_1 \sqcup V_2 \sqcup \cdots \sqcup V_C$，$|V_c| = n_c$，$\sum_c n_c = n$。新顺序下把 $\tilde A$ 按颜色块写成:

$$
\tilde A = \begin{pmatrix}
D_1 & B_{12} & B_{13} & \cdots & B_{1C} \\
B_{21} & D_2 & B_{23} & \cdots & B_{2C} \\
\vdots & \vdots & \ddots & & \vdots \\
B_{C1} & B_{C2} & \cdots & \cdots & D_C
\end{pmatrix}, \tag{1.4}
$$

其中 $D_c$ 是 $n_c \times n_c$ **对角矩阵** (因为同色顶点无边)，$B_{cc'}$ 是 $n_c \times n_{c'}$ 稀疏矩阵 (从颜色 $c'$ 节点到颜色 $c$ 节点的耦合)。对称 $A$: $B_{cc'} = B_{c'c}^\top$。

**$\tilde L$ 的块结构**:

$$
\tilde L = \begin{pmatrix}
0 & & & & \\
B_{21} & 0 & & & \\
B_{31} & B_{32} & 0 & & \\
\vdots & \vdots & \ddots & \ddots & \\
B_{C1} & B_{C2} & \cdots & B_{C,C-1} & 0
\end{pmatrix}. \tag{1.5}
$$

**$\tilde D_*$ 的颜色分块递推**: 把 (1.3) 按颜色展开，对 $i \in V_c$:

$$
\tilde d_i = \tilde a_{ii} - \sum_{c' < c} \sum_{k \in V_{c'}, (i,k) \in \mathcal{S}(\tilde A)} \frac{\tilde a_{ik}\, \tilde a_{ki}}{\tilde d_k}. \tag{1.6}
$$

**关键**: $\tilde d_i$ 只依赖 $\tilde d_k$ 其中 $k$ 在更早的颜色。**颜色 $c$ 内所有 $\tilde d_i$ 相互独立**，可以完全并行算出。

向量化写法: 记 $\tilde d^{(c)} \in \mathbb{R}^{n_c}$ 是颜色 $c$ 的对角向量，则

$$
\tilde d^{(c)} = \mathrm{diag}(D_c) - \sum_{c' < c} \mathrm{diag}\!\Big( B_{cc'}\, \mathrm{diag}(\tilde d^{(c')})^{-1}\, B_{c'c} \Big). \tag{1.7}
$$

这里 $\mathrm{diag}(\cdot)$ 是"取对角作为向量"算子。(1.7) 的计算复杂度是 $O(n_c \cdot \mathrm{nnz\_per\_row})$ per color, 颜色之间串行，总共 $O(\mathrm{nnz}(A))$ — 与串行 Phase 2 的 $D_*$ 计算同阶。 **Confidence: 确定**。

**Forward solve $(\tilde D_* + \tilde L) \tilde y = r$** 按颜色分块展开: 对 $c = 1, 2, \ldots, C$:

$$
\tilde y^{(c)} = \mathrm{diag}(\tilde d^{(c)})^{-1}\, \Big(r^{(c)} - \sum_{c' < c} B_{cc'}\, \tilde y^{(c')}\Big). \tag{1.8}
$$

颜色 $c$ 内 $\tilde y^{(c)}$ 的 $n_c$ 个分量通过一次"加权稀疏 SpMV 类"操作并行求出: 每个分量 $i \in V_c$ 做一次稀疏 dot product (读 $B_{cc'}$ 的第 $i$ 行, 与对应 $\tilde y^{(c')}$ 做 inner product)，然后除以 $\tilde d_i$。**颜色内无 cross dependency**。

**Backward solve $(\tilde D_* + \tilde U) \tilde z = \tilde D_* \tilde y$**: 对称地对 $c = C, C-1, \ldots, 1$:

$$
\tilde z^{(c)} = \tilde y^{(c)} - \mathrm{diag}(\tilde d^{(c)})^{-1}\, \sum_{c' > c} B_{cc'}\, \tilde z^{(c')}. \tag{1.9}
$$

同样颜色内 $n_c$ 分量并行。

### 1.3 为什么 (1.3) 中 $\tilde L, \tilde U$ 的"稀疏继承"是合法的

$\tilde A$ 的稀疏模式 $\mathcal{S}(\tilde A) = \sigma \circ \mathcal{S}(A) \circ \sigma^{-1}$ (置换 $A$ 行列索引)。DILU 在 $\tilde A$ 上的定义 (式 1.2) **完全等价于 Phase 1 §1.2 的 DILU 定义**，只是作用在被重排矩阵上。

因此 (a) 方案满足 Phase 1 对 DILU 的所有形式代数保证 (对角主导下的 $M$-矩阵well-defined, backward error $\kappa(M_{\text{DILU}})$-bounded, etc.)。我们得到的 $\tilde M$ 是一个 **合法、well-defined** DILU 预处理器 —— 只是其数值质量依赖于 $\sigma$ (重排顺序) 的选择。

**我们之所以称之为"不同的预处理器"**, 是因为:

$$
P \cdot M_{\text{DILU}}(A) \cdot P^\top \neq M_{\text{DILU}}(PAP^\top) = \tilde M. \tag{1.10}
$$

即 "先 DILU 再排" ≠ "先排再 DILU"。DILU 的非交换性源于递推 (1.3) 的顺序依赖。这是 §3 "为何重排损害收敛" 的数学源头。 **Confidence: 确定** (Saad 2003 §3.3 / §12.2 有精确版本证明)。

---

## 2. 图染色算法 — 三种思路按普适性排序

### 2.1 Red-Black 染色 (2-color, bipartite 专用)

**适用前提**: $G(A)$ 必须是 **二分图**，即存在 $V = V_0 \sqcup V_1$ 使所有边跨越 $V_0$ 与 $V_1$。

**对 7-点 3-D stencil**: 自然序索引 $i = (i_x, i_y, i_z)$，定义 $\pi(i) = (i_x + i_y + i_z) \bmod 2$。相邻 ($\pm$ 一个方向) 的两点必然奇偶相反。因此 $G(A)$ **是** bipartite，2-color 合法。对 $N_x \times N_y \times N_z$ 全域网格:

- $|V_0| = \lceil n/2 \rceil$, $|V_1| = \lfloor n/2 \rfloor$ (差距最多 1)。
- $B_{12}$ 稀疏模式: 每个 "red" 节点连 6 个 "black" 邻居 (体内), 边界节点 3-5 个。$\mathrm{nnz}(B_{12}) = 3n$ (体内边) $+$ 边界。
- $B_{11} = B_{22} = \emptyset$ (对角块为纯对角)。

**算法**: 单次 $O(n)$ 扫描计算 $\pi(i)$ 和排序置换 $\sigma$。**Confidence: 确定**。

**并行度上限**: $n/2$ 每色。对 $N_x \times N_y \times N_z = 256 \times 128 \times 64$, 每色约 $1.05 \times 10^6$ 独立行 —— 完全饱和 RTX 3050 Laptop 的 30720 thread slots ($\eta = 1$)、完全饱和 A100 的 $1.7 \times 10^5$ slots ($\eta = 1$)。 **这是 "并行度天花板" 的终极答案** — Phase 2 level scheduling 在这个网格上 $\eta \approx 0.27$ 平均 (§Phase 2 math §3.2)，Phase 3 red-black 直接饱和。

**无存储开销**: 染色函数 $\pi$ 隐式由坐标奇偶性给出; 置换 $\sigma$ 可以不显式构造 (直接对 red/black 分别循环)。 **较确定** — 具体 layout 是工程选择。

**缺陷**: 仅 2 色对应最大对数依赖深度等于 2，是最激进的并行化; 预处理器质量下降也最严重 (§3 分析)。适合作 Phase 3 **起步实现** (最简单)。

### 2.2 贪心多色 (general graph, $\Delta + 1$-color upper bound)

一般图上，Luby / Jones-Plassmann / Welsh-Powell 类算法给出 $O(\Delta + 1)$ 色，其中 $\Delta$ 是最大度。我们把三种典型算法并列对比:

| 算法 | 复杂度 | 并行性 | 典型色数 | 稳定性 |
|---|---|---|---|---|
| **Welsh-Powell** (串行贪心) | $O(n + \mathrm{nnz})$ | 无 | 接近最优 $\chi(G)$ | 确定, 可复现 |
| **Jones-Plassmann** (并行贪心) | $O(\Delta^2 \log n)$ 预期 | 高 (所有局部极值同时着色) | 稍多于 $\chi(G)$ | 需要随机数种子, 跨运行不稳定 |
| **Luby MIS** (Maximal Independent Set 迭代) | $O((\Delta + \log n)\log n)$ | 高 | $\sim \Delta + 1$ | 随机 |

**对 7-点 3-D stencil** ($\Delta = 6$): 贪心多色 worst case 7 色，但由于图 bipartite，实际上 **所有合理的贪心算法最终得到 2 色** (在 red-black layout 上启动即收敛)。所以对规则 stencil，贪心多色 **退化为 red-black**。

**对 AM 变系数 7-点 + 嵌入边界 / cut-cell stencil**: 每个 cut-cell 单元多出若干 "ghost" 邻居, $\Delta$ 可达 10-15; 贪心多色可能给出 **4-8 色**。对我们"静态网格, $\alpha$-cut PLIC" 混合工况，真正的色数需要 $\pi$ 实测, 不能先天估算。 **较确定**。

**对自适应 AMR 网格**: 刚细化完的网格在 coarse-fine interface 上有 T-junctions, $\Delta$ 可达 20+; 色数可能 8-12。 **不确定** (无直接 AM-AMR 染色文献)。

**在 Phase 3 范围内**: 既然 Phase 2 测试矩阵是规则 7-point stencil (test_t7_pcg_stiff.py, physical_benchmark.py 都是), 贪心多色与 red-black **数学等价**。所以 Phase 3 建议首先实现 red-black，贪心多色 **先写框架但不作为 Phase 3 默认路径**。若后续工作进入 cut-cell / AMR 工况，再激活贪心多色。

### 2.3 代数 (value-aware) 染色

**Idea**: 图着色时不仅避开拓扑邻边，还避开"强耦合"邻边 — 例如若 $|a_{ij}| > \theta \cdot |a_{ii}|$, 则 $i,j$ 必须不同色。这里 $\theta$ 是"强连接"阈值，借鉴 classical AMG 的经典 strength-of-connection 定义 (Brandt 1986, Ruge-Stüben 1987)。

**动机**: 在 $\rho$-contrast 界面上，跨界面的耦合 $a_{ij} \sim 1/\rho_{\text{avg}}$ 很大且 $a_{ii}$ 也很大，比例不小；但 **保留** 跨界面的 "弱" 耦合在 $\tilde M$ 之中 (即不丢弃 diagonal 递推中的那些 term) 能维持 Gustafsson-like 的聚谱性。代数染色确保跨界面耦合被 DILU 递推 "看到"。

**代价**: 色数上升 (可能到 10-20), 每色行数下降, 回到 level scheduling 的 under-occupied 问题区。

**Phase 3 范围**: 我们 **不做** 代数染色。原因:

1. 它严格属于 AMG 方法家族的一部分 (strength-of-connection 概念本为 AMG 服务)，归 Phase 4 解决。
2. 实现复杂度高: 需要矩阵值判断 + 动态色数，与 JAX FFI 静态调度不好契合。
3. Phase 3 的 "干净" trade-off 实验需要一个 **值无关** 的染色 baseline (red-black)，以便把 "重排 penalty" 和 "值无关" 分离出来。

**Confidence: 较确定** (代数染色的详细建构与 AMG 文献有重叠，属于 Falgout-Yang 2007 / hypre BoomerAMG 的中心文献空间)。

### 2.4 算法复杂度总表

| 路径 | Preprocess | 染色成本 | 色数 (7-pt 3D) | 色数 (cut-cell AM) | 运行时并行度 (饱和点) |
|---|---|---|---|---|---|
| Red-Black | $O(n)$ | 隐式 (坐标奇偶) | 2 | 不适用 | $n/2$ per color |
| Welsh-Powell 贪心 | $O(\mathrm{nnz})$ | 扫一遍 | 2 (退化) | 4-8 | $n/C$ |
| Jones-Plassmann | $O(\Delta^2 \log n)$ | 可 GPU 并行 | 2-3 | 4-10 | $n/C$ |
| 代数染色 (Phase 4) | $O(\mathrm{nnz})$ | 需扫值 | 10-20 | 12-30 | $n/C$ (C 较大) |

**Phase 3 默认选型**: **Red-Black**。理由汇总:
- 它是 "最简实现 + 最激进并行" 端点，便于建立 Phase 3 T7-equivalent 的性能 baseline。
- 它的预处理器质量下降是 **最坏情况** (色数最少)，若它能通过物理三测，其他多色方案自然更安全。
- 色数多的方案 (贪心 4-色等) 会同时稀释并行度和弱化重排 penalty，交付一个"中间模糊数字"，不利于理解权衡。

---

## 3. 重排为何伤害收敛 — 权衡的数学核心

这是 Phase 3 所有数值风险的根源。

### 3.1 Duff-Meurant 1989 的核心结论

Duff & Meurant (BIT 29:635–657, 1989) 系统研究了 20 种 orderings 对 ICCG 收敛的影响，得出了至今仍是 canonical 的结论。核心现象:

**结论 (DM-1)**: 对 2-D 5-point Laplacian, natural ordering ILU(0)-PCG 比 red-black ordering ILU(0)-PCG **迭代数少 $\sim 2\times$**。具体: 在 $50 \times 50$ 网格上, natural 约 23 iter (残差降 6 orders), red-black 约 44 iter。

**结论 (DM-2)**: 更精细的分析表明，natural ordering 使得 $M^{-1}A$ 的谱**紧簇在 1 附近**，少数外围特征值; 而 red-black 使谱**展开成双峰结构** (reds 一簇, blacks 一簇), 这两簇之间的"gap"迫使 CG 多迭代。

**结论 (DM-3)**: 对更强的 factorizations (ILU(k), MILU(0) with relaxation), red-black 的 penalty **减小但不消失**, 约 $1.3–1.8\times$。这表明 "重排的伤害" 一部分可以通过 fill-in 缓解。

**对我们的含义**:
- Phase 3 测试设置下 (7-point 3-D, $\rho = 100$ stiff, 16³), 基准期望 penalty 是 $2\times$ 级别 (外推 2-D 的 Duff-Meurant 结论)。
- 若实测 penalty 大于 $3\times$，暗示 (i) 7-point 3-D 比 5-point 2-D 更敏感 **或** (ii) $\rho$-contrast 放大了 penalty。两种假设都有先例。

**Confidence: 确定** (Duff-Meurant 文献引用量极高, 结论稳固)。

### 3.2 DILU vs ILU(0) 在重排下的行为差异

Duff-Meurant 测的是 ILU(0) (即也修改 off-diagonal 的版本)。我们做的是 DILU (只改对角)。**两者在重排下的 penalty 是否相同**?

**理论分析** (较确定，但未见直接文献):

DILU 递推 (1.3) 只改动对角。重排改变了 "$k < i$" 的含义 — 在 red-black 里, 一个 black 节点 $i \in V_2$ 的所有 red 邻居 $k \in V_1$ 都满足 $k < i$ (按颜色排序), 但它的 black 邻居 $k' \in V_2$ 如果 $\sigma^{-1}(k') < \sigma^{-1}(i)$ 也满足 $k' < i$, 不过由于同色无边, 这种情况不存在。

因此对 **红黑 DILU**, 每个 black 节点 $i$ 的 $\tilde d_i$ 只接收来自 red 邻居的 $a_{ik}^2 / \tilde d_k$ 贡献。而原始 natural ordering 下, 节点 $i$ 接收的是"已处理邻居"的贡献, 包含 $(-x, -y, -z)$ 三个方向。

**关键差异**: 自然序 natural 中节点 $i$ 吸收 3 个邻居的影响 (半数邻居)；red-black 中 black 节点吸收 **所有 6 个** red 邻居的影响 (全部邻居)。数学上 red-black DILU 的 $\tilde d_i$ 比 natural 更"丰满"—— 每个 node 多吸收邻居 — **但** 这意味着什么呢?

**DILU 的精确语义** 是 "$M$ 与 $A$ 在对角上相等"。重排后这个要求仍然满足; 但 $M^{-1} A$ 的 **谱散布** 取决于 off-diagonal 的残差 $M - A$。对 DILU:

$$
(M - A)_{ii} = 0, \quad (M - A)_{ij} = \sum_k \frac{\tilde L_{ik} \tilde U_{kj}}{\tilde d_k} - 0, \quad i \neq j, (i,j) \notin \mathcal{S}(A).
$$

也就是 DILU 的 "fill-in error" 完全集中在 **原本零的位置**。red-black 重排并不改变这个集合的大小 (稀疏模式不变), 但改变了每个 fill 位置的数值。

**经验观察** (Saad 2003 §12.2 / Li-Saad 2010): 对 SPD 7-point Poisson + DILU, red-black 的 iter 数 penalty 与 ILU(0) 的 penalty **大致相同 $\sim 2\times$**, 原因是 DILU 与 ILU(0) 在 Gustafsson scaling 上处于同一渐近区域。 **较确定**。

**对 AM $\rho = 10^3$ 问题**: 界面两侧 $a_{ij}$ 差 3 个数量级，递推 (1.3) 中 $a_{ik}^2 / d_k$ 项对跨界面耦合贡献极大。重排 **改变了哪些耦合在 diagonal 中被吸收**, 可能使界面附近的 $\tilde d_i$ 偏离"良态"区域。**这是我们最担心的 penalty 放大机制**。 **不确定具体程度, 必须实测**。

### 3.3 Phase 2 实测基线的外推

Phase 2 T7 实测: 16³ stiff ($\rho = 100$), DILU-PCG **24 iter**, Jacobi-PCG **71 iter**。

**Phase 3 red-black DILU 预期**:

| 估计依据 | 预期 iter (Phase 3 red-black DILU) |
|---|---|
| Duff-Meurant 2× penalty (SPD 良态) | 48 iter |
| Saad §12 估计 1.5–3× (variable-coef) | 36–72 iter |
| 放大到 "黑色吸收 6 邻居" 激进型 | 30–50 iter |
| 悲观 (ρ-contrast 放大 + 7-pt 3-D 敏感度) | 70–120 iter |

**权衡判据**:

- **若 Phase 3 实测落在 36–72**: 正常 Duff-Meurant 范围, 符合文献期望。Phase 3 成功。
- **若落在 >72**: penalty > 3×，触发 §6 警报路径。评估 Jacobi-PCG 的 71 iter — 我们不希望 red-black DILU 退化到接近 Jacobi 水平 (那就没做 DILU 的价值了)。
- **若落在 >100 (接近或超过 Jacobi 的 71)**: red-black DILU **失败**, 必须放弃 Phase 3 (a) 路径, 回到 Phase 2 level scheduling 或启动 Phase 4 AMG。

### 3.4 Li-Saad 2010 的 GPU MC-ILU(0) 数据点

UMSI-2010-112 (Li & Saad, "GPU-Accelerated Preconditioned Iterative Linear Solvers"): 他们用 Jones-Plassmann 多色 + ILU(0)-PCG 测试若干 2-D / 3-D 问题。节选他们报告的有代表性数字:

- **2-D 5-pt Laplacian ($100 \times 100$)**: natural 32 iter → multi-color (2色) 58 iter. **penalty 1.81×**。
- **3-D 7-pt anisotropic ($64^3$, 各向异性 10×)**: natural 45 iter → multi-color (4色) 89 iter. **penalty 1.98×**。
- **Oil reservoir discontinuous 3-D**: natural 112 iter → multi-color 215 iter. **penalty 1.92×**。

这些数据 **较确定** 地把 penalty 锁在 $\sim 2\times$。对我们 $\rho = 100$ 的 stiff problem, 外推: $24 \times 2 = 48$ iter 是最可能的中位预测。

### 3.5 更大网格 ($N = 256^3$ scale)? — 需验证

Duff-Meurant、Li-Saad 的数据都在 $\leq 100^3$ scale。对 AM-realistic $256^3$：

**理论期望**:
- $\kappa_2(A)$ 按 $1/h^2$ 标度 ↑ 16× (from $64^3$ to $256^3$)。
- PCG 迭代数按 $\kappa^{1/4}$ (Gustafsson-DILU) 或 $\kappa^{1/2}$ (Jacobi) 标度。16× κ-growth 对应 $2\times$ (DILU) 到 $4\times$ (Jacobi) iter growth。
- natural ordering DILU: $24 \times 2 = 48$ iter on stiff $64^3$ → $\times 2$ on $256^3$ → **约 96 iter**。
- red-black DILU (penalty 2×): **192 iter**。

**需验证**: 当 $N$ 增大时，重排 penalty 本身是否稳定在 2×? Duff-Meurant 的 $2\times$ 是对 $50 \times 50$ 测的，我们尚无 penalty-as-function-of-N 的理论。**这是 Phase 3 可能需要做的 secondary 实验 (先在 16³ 验证, 再在 32³ 验证, 看 penalty 漂移)**。

---

## 4. 多色 DILU Kernel 的 GPU 并行结构分析

### 4.1 Red-Black Forward Solve 的 kernel 拆解

对 2-color (red-black)，forward solve (1.8) 具体化为两步:

**Step 1 — Red solve** ($c = 1$):
$$
\tilde y^{(1)}_i = \tilde d^{(1),-1}_i\, r^{(1)}_i, \quad \forall i \in V_1.
$$
纯 elementwise，$n/2$ 独立除法。**单 kernel launch**, 100% parallel, 无数据依赖。

**Step 2 — Black solve** ($c = 2$):
$$
\tilde y^{(2)}_i = \tilde d^{(2),-1}_i\, \Big(r^{(2)}_i - \sum_{k \in V_1, (i,k) \in \mathcal{S}(A)} a_{ik}\, \tilde y^{(1)}_k\Big), \quad \forall i \in V_2.
$$
每个 black 节点做一次稀疏 dot product (最多 6 项, 读相邻 red 节点的 $\tilde y^{(1)}$), 然后除以 $\tilde d^{(2)}_i$。**单 kernel launch**, $n/2$ 独立行, 100% parallel。

**总 forward solve**: **2 个 kernel launch**, 每个覆盖 $n/2$ 独立行。对比 Phase 2 level scheduling: 446 个 level (对 $256 \times 128 \times 64$)、平均宽度 4700 — 每 level 一个 implicit kernel boundary + atomic synchronization。

### 4.2 Backward Solve 同理

对称地: Black first (处理 $c = 2$), Red second ($c = 1$)。2 个 kernel launch。

### 4.3 Kernel Launch 数对比

| Solver | Forward + Backward kernel launches | 每次 launch 的 work |
|---|---|---|
| Phase 2 (level scheduling, $256 \times 128 \times 64$) | $\approx 2 \times 446 = 892$ 个 level-barrier (实际 cuSPARSE 内部 atomic, 不全是 kernel launch, 但等效同步点) | 每 level 平均 4700 行 |
| Phase 3 (red-black, 同网格) | $2 \times 2 = 4$ kernel launches | 每 launch $1.05 \times 10^6$ 行 |

**比例 223×**: Phase 3 的 kernel launch 数目比 Phase 2 少 200× 以上。这是 "GPU dispatch overhead" 最激进的优化。

### 4.4 FLOP Count 守恒

**关键观察** (较确定): Phase 2 与 Phase 3 single-iteration FLOP **完全相同**。

- 每行: $\leq 6$ off-diagonal multiply-add + 1 divide $\approx 13$ flops
- 总 forward+backward: $\sim 26n$ flops
- 对 $n = 2.1 \times 10^6$: $\sim 5.5 \times 10^7$ flops/iter

**两者只差在编排**:
- Phase 2: 446 个窄 sync barrier + dispatch overhead
- Phase 3: 4 个宽 kernel + 完全并行

**理论每-iter wall time** (忽略 penalty):

$T_{\text{P3-iter}} = 4 \times (t_{\text{launch}} + n/2 / \text{throughput})$

$\approx 4 \times (5 \mu s + 1.05 \times 10^6 / (10^{10} \text{flops/s}))$

$\approx 4 \times (5 \mu s + 105 \mu s) = 440 \mu s$

$T_{\text{P2-iter}} = 446 \times (t_{\text{launch}} + \bar{w} / \text{throughput}) + \text{overhead}$

$\approx 446 \times (5 \mu s + 4700 / (10^{10})) + ...$

$\approx 446 \times (5 \mu s + 0.47 \mu s) + \text{overhead} \approx 2.4 \text{ms}$

**理论 Phase 3 单 iter 快 ~5.5×**。 **较确定**, 具体数字需 Phase 3 T7-equivalent 实测。

### 4.5 权衡结合到总 wall time

**Phase 2 总 PCG wall time** (Phase 2 report): 16³ 24 iter, 约 $24 \times 10$ ms $\approx 240$ ms (不严格; Phase 2 report 没有明确给 "per-PCG-iter wall time"，但 1.8 ms DILU-apply + 其他 iter overhead 约 5-10 ms estimate)。

**Phase 3 预测**:

| 假设 | iter count (相对 24 = $\alpha$ penalty) | per-iter time (相对 Phase 2 = $\beta$ speedup) | 总时间比 |
|---|---|---|---|
| 乐观 (Duff-Meurant 2×, Phase 3 单步 5×) | $\alpha = 2$, $\beta = 1/5$ | **$2/5 = 0.4$ (Phase 3 快 2.5×)** |
| 中位 (penalty 2.5×, 单步 4×) | $\alpha = 2.5$, $\beta = 1/4$ | $2.5/4 = 0.625$ (Phase 3 快 1.6×) |
| 悲观 (penalty 3×, 单步 3×) | $\alpha = 3$, $\beta = 1/3$ | $3/3 = 1.0$ (Phase 3 持平) |
| 失败 (penalty 5×, 单步 3×) | $\alpha = 5$, $\beta = 1/3$ | $5/3 = 1.67$ (Phase 3 慢 67%) |

**项目成功的判据**: Phase 3 总 wall time 比 Phase 2 快。阈值落在 **penalty / speedup < 1** — 即 penalty 不能超过 speedup 的数倍。

**具体数字**: Phase 3 T7-equivalent 应测:
- red-black DILU-PCG iter count (vs Phase 2 的 24)
- 单次 DILU-apply 的 wall time (vs Phase 2 的 1.8 ms on 16³)
- 计算乘积与 Phase 2 比值。阈值 < 1 为成功, > 1 为失败。

**Confidence: 较确定** 上述乘法模型; **不确定** Phase 3 具体数字落在哪个行。

### 4.6 更大网格 ($256^3$) 的并行度优势放大

Phase 2 math §3.2 指出 RTX 3050 Laptop (30720 thread slots) 在 $256 \times 128 \times 64$ 网格上 $\bar{w} \approx 4700$, 每 level $\eta \approx 0.15$ 平均。Phase 3 red-black: 每色 $n/2 = 1.05 \times 10^6$, $\eta = 1$ (完全饱和)。**Phase 3 在大网格上优势放大**。

对 A100 (170k slots): Phase 2 $\eta \approx 0.03$, Phase 3 $\eta = 1$。**Phase 3 对强 GPU 的优势更显著** — 这是 Phase 1 §3.1 提到的 "更强的 GPU 让 level scheduling 更 under-occupied" 的反面论点。

**Phase 3 的价值 scaling 预测** (较确定):

对 RTX 3050 (弱 GPU): Phase 3 预期 1.5-2× 总速度优势。
对 A100/H100 (强 GPU): Phase 3 预期 3-5× 总速度优势。

---

## 5. Phase 3 Gold-Standard Comparisons

Phase 3 multi-color DILU 必须通过三层验证: 正确性、迭代权衡、物理健康。

### 5.1 正确性 (Phase 2 验收的并行镜像)

Phase 2 T4/T5/T6 在 Phase 3 的 equivalent:

**T4-equiv (DILU factor 正确性)**: 计算 $\tilde D_*$ (red-black ordering) 与 Python 串行参考实现比较。参考实现即对 **$\tilde A = P A P^\top$** 跑 Phase 1 的 serial DILU 递推。两者应 ULP-floor 一致 (Phase 2 T4 的扩展)。

具体: 对 Phase 2 $16^3$ stiff 矩阵, 构造 red-black 置换 $P$, 计算 $\tilde A$, 调用 Phase 3 新的 `compute_dstar_colored` kernel 得 $\tilde d^{(c)}$ 向量; 与 Python 串行 DILU on $\tilde A$ 的 $\tilde d_i$ 逐元素对比。

**容忍**: $\leq 50 \cdot \mathrm{nnz\_per\_row} \cdot \epsilon_{\text{mach}} \leq 10^{-13}$。**与 Phase 2 T4 的 allowed error 完全相同** — red-black 重排不引入额外误差。

**T5-equiv (diagonal matrix degenerate case)**: $A = D$ 对角, $\mathcal{S}(A) \setminus \mathrm{diag} = \emptyset$, $G(A)$ 无边, 染色是平凡的 (所有节点同色或任意分色都合法)。Phase 3 DILU-apply 必须退化为 $M^{-1} r = D^{-1} r$ (Jacobi), 与 Phase 2 T5 结果 bit-identical。

**T6-equiv (3-D Laplacian 正确性)**: 对 $16^3$ (或 $8^3$, $12^3$) Laplacian 跑 Phase 3 DILU-apply, 与 Python serial reference DILU (on reordered $\tilde A$) 对比。容忍同 Phase 2 T6 (1-2 ULP)。

**关键**: Phase 3 的参考实现是 **串行 DILU on permuted matrix**, 不是 Phase 2 的串行 DILU on original matrix。这两者本身就不相等 (§1.3 的 (1.10))。所以 Phase 3 不能直接与 Phase 2 的 $z$ 对比 — 要么 permute Phase 2 的 $z$, 要么用 Phase 3 的 reference。

### 5.2 迭代权衡 (新于 Phase 3)

**T7-equiv (Phase 2 T7 的红黑版本)**:

- 同一个 16³ stiff 矩阵 ($z$-axis midplane contrast = 100, harmonic-mean face coefs)，Phase 2 T7 report 值 24 iter。
- 运行 Phase 3 red-black DILU-PCG, tol = 1e-8, max_iter = 2000。
- 记录 iter count $N_{\text{P3}}$。

**验收逻辑**:

$$
\text{penalty ratio} = \frac{N_{\text{P3}}}{24}
$$

- $\leq 2.0$: Excellent (与 Duff-Meurant 中值一致)，**Phase 3 (a) 强证据胜利**。
- $2.0 – 3.0$: OK (在合理范围内)，**Phase 3 (a) 可接受**, 但接近上限。
- $> 3.0$ **且** $N_{\text{P3}} < 71$ (比 Jacobi 好): Marginal — Phase 3 (a) 仍有价值但 iter 权衡成本高，权衡由单步 wall time 决定。
- $> 71$ (接近或超过 Jacobi): **FAIL**, red-black DILU 退化到几乎 Jacobi 质量，数学意义的 DILU 被重排抹杀。

**可选扩展**: 测 32³ (mod phase 2 scale 的 8×) stiff 同一 contrast, 看 penalty 是否稳定。

### 5.3 物理健康 (来自 Phase 2.5)

**极关键** — 用户明确指示 "绝不能出现几何误差花纹"。Phase 3 必须过三组 physical_benchmark.py 测试。

#### 5.3.1 Test A — 散度场 halo

**Phase 2 结果** (phase2.5 report): $32^3$, $\rho$-contrast 1000, 投影后 max|∇·u| = 4.13e-9 (均匀噪声，界面处无 halo)。

**Phase 3 要求**:
- 同一 grid、同一 $\rho$、同一 RHS, Phase 3 red-black DILU-PCG 应给 max|∇·u| **≤ 10 × 4.13e-9 = 4.13e-8**。
- **可视化**: div(u) 在 z=16 slice 图，不应出现沿球面轮廓的 "halo" (high-|div| 环)。
- PCG 收敛 iter count 记录。Phase 2 收敛在 115 iter (tol=1e-10)。Phase 3 若 penalty 1.5–2×, 预计 170–230 iter。 **需验证**。

**失败判据**: max|∇·u| > 1e-7 **或** 可视化出现 halo。这会触发 §6 风险路径。

#### 5.3.2 Test B — 静态液滴伪流

**Phase 2 结果**: 10-step 投影-法, ‖u‖∞ 从 1.1e-7 线性增长到 9.6e-7 (CSF discretization 的 O(h²) 残差, 非求解器伪流)。

**Phase 3 要求**:
- 同 setup 跑 10 步。
- ‖u‖∞ 轨迹不能比 Phase 2 坏超过 2×: **‖u‖∞ @ step 10 ≤ 2 × 9.6e-7 = 1.9e-6**。
- max|div(u)| per step 应保持 machine-zero (≤ 1e-14)，即压力投影仍然 **完全** 消除散度 (这是 PCG 收敛质量的测试，不是求解器的数学质量)。
- **可视化**: quiver 图不应显示新的"触手状"(tentacle) 涡。Phase 2 的 4-lobed pattern 是 CSF 产物可接受, 任何新出现的 4-lobe-不对齐的 vortex pattern (即解求解器诱导的) 都是失败信号。

**失败判据**:
- ‖u‖∞ 指数增长 (而非 Phase 2 线性)。
- 新的涡旋 pattern (与液滴中心不对称)。
- max|div| > 1e-12 (投影失效)。

#### 5.3.3 Test C — 空间残差 halo (acid test)

**Phase 2 结果**: $32^3$ 三相密度 (1 / 1000 / 8000), fixed-iter 50 DILU-PCG, raw max|E| = 18.5 (rel 21%), mean-detrended 18.4 — **残差是低频 Krylov 模式, 不追踪界面几何**。

**Phase 3 要求** (这是最严格的测试):
- 同 setup, 同 RHS, Phase 3 red-black DILU-PCG 固定 50 iter。
- **两个互斥条件都要满足**:
  1. $\max|E_{\text{demean}}|$ 不显著恶化 (≤ 2 × Phase 2 的 18.4 = 36.8)。 *(数值阈值)*
  2. **mean-detrended E 场不与 $\rho$-contour 相关**。*(空间 pattern 阈值)*

**条件 2 的量化**:

对 E_demean 与 $\|\nabla \rho\| / \rho$ 的空间相关系数:

$$
\mathrm{corr}\bigl(|E_{\text{demean}}|, \|\nabla \log \rho\|\bigr) = \frac{\mathrm{cov}(|E|, \|\nabla \log \rho\|)}{\sigma_{|E|} \sigma_{\|\nabla \log \rho\|}}. \tag{5.1}
$$

- Phase 2 (baseline): 相关系数估计 $\sim 0.1$ (残差与界面弱相关, 主要是低频模态).
- Phase 3 PASS 阈值: $\leq 0.25$ (与 Phase 2 相比放松 2.5×, 仍在 "非 halo" 区间)。
- Phase 3 FAIL 阈值: $> 0.5$ (强相关, halo pattern 显著)。

**计算方法**: 在 32³ 域内, 离散 $\|\nabla \log \rho\|_{ij}$ (中心差分 of $\log \rho$), 与 |E_demean|_{ij} 拉平 flatten 成向量, 算 Pearson 相关系数。

**失败含义**: 相关系数 > 0.5 意味着 **red-black 多色使 DILU 在界面附近损失了高频平滑能力**, 这是 §6.2 列出的风险 2 的实际触发。

**Confidence: 较确定** (阈值数字基于 Phase 2.5 baseline 外推, 不是文献引用; 实施时可能需要 tune)。

### 5.4 总验收矩阵

| 类别 | 测试 | 度量 | PASS 阈值 | FAIL 阈值 |
|---|---|---|---|---|
| 正确性 | T4-equiv | max 相对 $\tilde d$ 差 | ≤ 1e-13 | > 1e-10 |
| 正确性 | T5-equiv | 对角矩阵退化 | bit-identical to Jacobi | 任何差异 |
| 正确性 | T6-equiv | max 相对 $z$ 差 | ≤ 1e-12 | > 1e-9 |
| 迭代权衡 | T7-equiv | $N_{\text{P3}} / 24$ | ≤ 3.0 | > 3.0 |
| 物理 A | 32³ 投影 | max\|∇·u\| | ≤ 4.1e-8 | > 1e-7 |
| 物理 B | 静态液滴 | ‖u‖∞ @ step 10 | ≤ 1.9e-6 | 指数增长 |
| 物理 C | 残差 halo | corr(\|E\|, \|∇log ρ\|) | ≤ 0.25 | > 0.5 |

---

## 6. 预期失败模式 (诚实列举)

### 6.1 失败模式清单

#### F1. 迭代 penalty > 3×

**现象**: T7-equiv 测得 $N_{\text{P3}} > 72$ (基于 Phase 2 的 24)。

**根源**: Duff-Meurant penalty 在 $\rho$-contrast > 100 或 3-D stencil 下放大至 3×+。

**诊断**: 查看 PCG 残差 history — 若 linear decrease 维持 (只是斜率变浅), 是 "典型 penalty"; 若 stagnation (残差停滞), 是更严重的 spectral gap 问题。

**Fallback**:
1. 先试 4-color greedy 减少 penalty (增加色数可恢复部分 clustering)。
2. 若仍失败, **放弃 red-black**, 回到 Phase 2 level scheduling。
3. Phase 4 AMG 介入。

**Confidence 评估概率发生**: 较可能 ($\sim 30\%$ 概率), 应提前做好 fallback 架构准备。

#### F2. Test C 出现界面误差 halo

**现象**: Test C 的 corr(|E|, |∇log ρ|) > 0.5, 可视化可见误差沿界面聚集。

**根源**: red-black 的 $\tilde D_*$ 在界面处 "两种颜色的节点看到不同组的邻居" — 红节点吸收 6 个黑邻居 (可能横跨界面), 黑节点也是 — 如果跨界面耦合在递推中被不一致处理, $\tilde d_i$ 在界面两侧产生不连续，导致 $\tilde M^{-1}$ 应用时界面处出错。

**诊断**: 画 $\tilde d_i$ 场 (reshape 到 3D), 看界面处是否有 sharp unconnected jump。

**Fallback**:
1. 4-color 染色 (允许 "界面内颜色" 不与 "界面外颜色" 冲突)。
2. **代数染色**: 强耦合 (跨界面) 被强制不同色 — 但这是 Phase 4 的 AMG 领域。
3. 承认 Phase 3 (a) 的 density-contrast 上限。

**Confidence 评估**: 中等 ($\sim 20-30\%$ 概率)。这是 **本项目最担心的失败模式**, 与用户明确要求的 "绝不能出现几何误差花纹" 直接对应。

#### F3. Test B 伪流放大

**现象**: ‖u‖∞ 在 10 步内从 1e-7 长到 1e-3 或更高, 轨迹超 Phase 2 的线性规律。

**根源**: 与 F2 相同机制 — 界面处 $\tilde M$ 失真导致压力投影残差, 流体得不到正确的 "压力修正 cancelling CSF force" 效应。

**诊断**: 每 step 的 max|div(u)| — 如果从 Phase 2 的 5e-15 跃到 1e-8+, 说明投影失败。

**Fallback**: 同 F2 (多色数 / 代数染色 / Phase 2 fallback)。

**Confidence 评估**: 如果 F2 发生, F3 几乎必然发生 (共同根源)。作为 "附带 symptom" 而非独立 risk。

#### F4. Test A 散度 halo

**现象**: 32³ 投影后 max|∇·u| > 1e-7, slice plot 可见 halo。

**根源**: 与 F2 机制相同 — 投影的 convergence 被界面处残差污染。

**Fallback**: 同 F2。

**Confidence**: 如果 F2 发生, F4 几乎必然共现。

#### F5. 染色成本过高 (工程层面)

**现象**: 每 time step 重算 red-black 染色要 $> 10\%$ solve 时间。

**根源**: 对静态网格, red-black 染色是 $O(n)$ 一次性预处理, 与 Phase 2 analysis 同阶, 可以缓存。对 adaptive / AMR 网格, 每步重染色。

**Phase 3 范围**: 静态网格下这不是问题 (与 Phase 2 一样); adaptive 场景留到 Phase 4。

**Confidence**: 不是 Phase 3 威胁。

### 6.2 风险优先级与 fallback 策略

| # | 风险 | 概率 | 严重性 | 触发 Phase 3 失败 |
|---|---|---|---|---|
| F1 (iter penalty > 3×) | 30% | 中 | 迭代权衡失败，但 (a) 仍可接受若 wall time 不输 |
| F2 (Test C halo) | 25% | **高** | 用户明确红线，**立刻放弃 (a)** |
| F3, F4 (B/A 联动) | 25% (与 F2 联动) | **高** | 同 F2 |
| F5 (染色成本) | 5% | 低 | 静态网格下不发生 |

**Phase 3 工程方应提前做的准备** (对 jax-cfd-am-expert 的预警):
- 保留 Phase 2 level-scheduling 路径作为 fallback (不要删)。Phase 3 应作为并行替代路径, 可运行时切换。
- Test C 的 corr(|E|, |∇log ρ|) 计算代码应先实现 —— 它是 Phase 3 acid test 的核心。
- 如果 F2 触发, 立即停止 Phase 3 (a), 不要试图 tune hyperparameters。回报数据并进入 Phase 4 AMG 规划。

---

## 7. Acceptance Criteria — 8 条可测

以下 8 条是 Phase 3 "完成" 的数学验收标志。每条 (a) 度量 (b) 阈值 (c) 测法。

**C1** — (正确性) $\tilde D_*$ ULP-level 正确
- 度量: $\max_i |\tilde d_i^{\text{gpu}} - \tilde d_i^{\text{ref}}| / |\tilde d_i^{\text{ref}}|$
- 阈值: $\leq 50 \cdot \mathrm{nnz\_per\_row} \cdot \epsilon_{\text{mach}} \leq 10^{-13}$
- 测法: $16^3$ stiff 矩阵, red-black 置换, Python 串行 DILU on permuted 参考 vs Phase 3 kernel

**C2** — (正确性) DILU-apply on diagonal matrix ≡ Jacobi
- 度量: $\|z_{\text{P3}} - z_{\text{Jacobi}}\|_\infty$
- 阈值: = 0 (bit-identical)
- 测法: $A = \mathrm{diag}$ 随机 stiff 对角, 任意 red-black 置换, compare

**C3** — (正确性) 3-D Laplacian DILU-apply 正确
- 度量: $\|z_{\text{P3}} - z_{\text{ref}}\|_\infty / \|z_{\text{ref}}\|_\infty$
- 阈值: $\leq 10 \cdot \mathrm{nnz\_per\_row} \cdot \epsilon_{\text{mach}} \leq 10^{-12}$
- 测法: $16^3$ 7-pt Poisson (non-stiff, contrast=1), red-black, Python reference compare

**C4** — (迭代权衡) red-black DILU-PCG iter penalty ≤ 3×
- 度量: $N_{\text{P3,iter}} / 24$ (24 是 Phase 2 T7 baseline)
- 阈值: $\leq 3.0$, 即 $N_{\text{P3,iter}} \leq 72$
- 测法: 精确复用 `test_t7_pcg_stiff.py` 的矩阵, 替换 `plan.apply` 为 red-black 版本

**C5** — (迭代权衡) red-black DILU 仍优于 Jacobi
- 度量: $N_{\text{P3,iter}}$ vs Jacobi 的 71 (Phase 2 T7)
- 阈值: $N_{\text{P3,iter}} < 71$ (严格优于 Jacobi)
- 测法: 同 C4, 比较

**C6** — (物理 A) 投影后散度无界面 halo
- 度量: $\max|\nabla \cdot u|$ on 32³ 投影 Test A
- 阈值: $\leq 4.1 \times 10^{-8}$ (Phase 2 baseline × 10)
- 测法: 精确复用 physical_benchmark.py Test A, 替换 DILU-PCG 为 Phase 3

**C7** — (物理 B) 静态液滴伪流不指数放大
- 度量: $\|u\|_\infty$ @ step 10
- 阈值: $\leq 1.9 \times 10^{-6}$ (Phase 2 baseline × 2); additionally, max|div(u)| ≤ 1e-12 per step
- 测法: 精确复用 Test B

**C8** — (物理 C, acid test) 残差不与界面几何相关
- 度量: $\mathrm{corr}(|E_{\text{demean}}|, \|\nabla \log \rho\|)$ on Test C fixed-iter 50
- 阈值: $\leq 0.25$
- 测法: 复用 Test C 的 setup, 固定 50-iter Phase 3 DILU-PCG, 计算 E demean, 计算 $\|\nabla \log \rho\|$ 场 (中心差分), flatten 并算 Pearson correlation

**汇总**: C1-C3 是 **正确性** (失败必 retest), C4-C5 是 **迭代权衡** (权衡 metric, 超阈值不致命但需 §6 路径), C6-C8 是 **物理健康** (失败=Phase 3 (a) 失败, 切 Phase 4 AMG)。

---

## 8. Engineering Hand-off Specification

### 8.1 本 Phase 需要 jax-cfd-am-expert 实现的模块

#### 模块 A — 染色预处理 (Preprocessing)

**输入**: CSR $(row\_ptr, col\_idx)$ (无需 values; 我们做的是 struct coloring)。

**输出**:
- `color_of_node: int32[n]` — 每个节点的颜色 (0, 1 对 red-black; 0..C-1 对 greedy)。
- `n_colors: int32` — 色数 $C$。
- `nodes_per_color: list[int32[n_c]]` — 按颜色分组的节点索引列表。**推荐格式**: 一个 `color_offsets: int32[C+1]` + 一个 `color_perm: int32[n]` 满足 `color_perm[color_offsets[c]:color_offsets[c+1]]` 给出颜色 $c$ 的节点。
- `sigma: int32[n]` — 置换, $sigma[i]$ 给出节点 $i$ 在新顺序下的位置。inverse 为 `sigma_inv: int32[n]`。

**算法 (Phase 3 默认)**: Red-black 染色。
- 7-pt Laplacian: 直接 `color_of_node[(k * ny + j) * nx + i] = (i + j + k) % 2`
- 通用 CSR: 一次 BFS 从随机起点, alternating 2-color (仅 bipartite graph 有效)

**复杂度**: $O(n)$。**一次性预处理, 可缓存**, 与 Phase 2 的 `SpSVDescr` analysis 同生命周期。

**API 契约**:
```
(color_offsets, color_perm, n_colors) = compute_red_black_coloring(row_ptr, col_idx, nx, ny, nz)
```

#### 模块 B — 多色 $\tilde D_*$ 计算

**输入**: CSR $(row\_ptr, col\_idx, values)$, 染色输出 $(color\_offsets, color\_perm, n\_colors)$, `diag_offset`。

**输出**: $\tilde d: \mathbb{R}^n$ (device buffer)。**注意**: $\tilde d[i]$ 存在 **原始索引空间** (即 $\tilde d[i]$ 对应节点 $i$, 无论它在哪个颜色)。这是为了让 apply kernel 的 random-access pattern 与原 CSR 对齐。

**算法**: 对 $c = 0, 1, \ldots, C-1$ 串行 (颜色间); 颜色内 $n_c$ 个节点 (`color_perm[color_offsets[c]:color_offsets[c+1]]`) 并行:
```
for c in 0..C-1:
    kernel_launch_1_color(
        input: row_ptr, col_idx, values, diag_offset, d_tilde (颜色 < c 已填),
        color_nodes: color_perm[color_offsets[c]:color_offsets[c+1]],
        output: d_tilde[color_nodes[*]]
    )
```

每 kernel thread 处理一行 $i = color\_perm[k]$: 按 CSR 读 $i$ 的邻居, 过滤出 "颜色 < c" 的, 按 (1.3) 累加 $a_{ik} a_{ki} / \tilde d_k$。

**kernel launch 数**: $C$ (2 for red-black)。**Phase 2 是 1 (串行整体)**。

**FLOP per apply**: 与 Phase 2 相同, $O(\mathrm{nnz}(A))$。

**复杂度差异**: Phase 3 的 $\tilde D_*$ 计算 **并行度天花板** 是 max color size $n/C$; Phase 2 的 $D_*$ 目前串行 (Phase 2 report §2.3 选项 C)。所以 **Phase 3 $\tilde D_*$ 计算比 Phase 2 快 $\sim n/C$ 倍** ≈ 500k× 对 $n = 10^6, C = 2$ — 这是 Phase 3 附加的 payoff。

#### 模块 C — 多色 SpSV-equivalent Kernel (Forward / Backward)

**输入**: CSR $A$, $\tilde d$, rhs $r$, 染色数据, 方向 ('L' or 'U')。

**输出**: $y$ 或 $z$ 满足 $(\tilde D_* + \tilde L) y = r$ (forward) 或 $(\tilde D_* + \tilde U) z = r$ (backward)。

**Forward 算法**:
```
for c in 0..C-1:
    kernel_launch (
        for i in color_perm[color_offsets[c]:color_offsets[c+1]] (parallel):
            s = r[i]
            for (i,k) in S(A) with color_of_node[k] < c:
                s -= values[...] * y[k]
            y[i] = s / d_tilde[i]
    )
```

**Backward**: 对 `c = C-1, ..., 0`, 类似但过滤条件是 `color_of_node[k] > c`, 和 $z$ 更新公式为 (1.9)。

**kernel launch 总数**: $2C$ (forward + backward)。对 red-black: 4 launches。

**FLOP count**: 与 Phase 2 的 SpSV 相同 ($\sim 26n$ flops per full apply)。

**存储**: 不需要新 CSR (直接读原 $A$ 的 values + 过滤邻居颜色)。**Memory footprint: 额外 $n$ int32 (color_of_node) + $n$ int32 (color_perm) + $C+1$ int32。对 $n = 10^7$: 80MB**, 比 Phase 2 的 SpSVDescr buffer 轻。

### 8.2 Phase 2 可复用的部件

**全部可复用**:
- Phase 1 的 FFI toolchain (ffi_mvp/), 包括 dispatch, stream, error signaling。
- Phase 2 的 `test_t7_pcg_stiff.py` 的 PCG driver 与 `_stiff_laplacian_3d` 矩阵生成 — 直接替换 `plan.apply` 为 Phase 3 版本即可做 T7-equiv。
- Phase 2 的 `physical_benchmark.py` 全部三组测试 — 直接替换 `dilu_pcg` 的 apply 即可做 C6/C7/C8。
- Phase 2 的 `build_diag_offset` 和 CSR builders。
- Phase 2 的 Plan / descriptor cache 抽象 — Phase 3 的新 Plan 需求类似但不同 (染色缓存替代 SpSVDescr 缓存)。

**不复用**:
- cuSPARSE `cusparseSpSV_*` API — Phase 3 我们写自己的 CUDA kernel。
- cuSPARSE `cusparseSpSV_updateMatrix` 逻辑 — 不适用 (我们自己管理 $\tilde d$ 更新)。
- Phase 2 的 `SpSVDescr` opaque token — 不需要。

### 8.3 Plan 对象 Phase 3 版本的新结构

类比 Phase 2, 一个 Phase 3 `ColoredPlan`:
- `A_pattern_hash`: CSR 稀疏模式哈希 (用于检测 pattern 变化)。
- `color_offsets`, `color_perm`, `n_colors`: 染色数据。
- `d_tilde_buffer`: $n$ float64 device buffer, 存 $\tilde d$。
- `diag_offset`: 从 Phase 2 复用。

Plan 的方法:
- `plan.factor(values) -> d_tilde` — 调模块 B, 返回新的 $\tilde d$。
- `plan.apply(values, d_tilde, r) -> z` — forward + backward, 调模块 C 两次。内部多 kernel launch ($2C$)。
- `plan.release()` — 释放 GPU buffer。

**Python 侧伪代码** (仅说明形状, **不要求实现**):
```python
plan = ColoredPlan(row_ptr, col_idx, values, diag_offset, nx, ny, nz)
d_tilde = plan.factor(values)
z = plan.apply(values, d_tilde, r)
```

### 8.4 Phase 3 不做的事 (scope)

- 不实现 greedy 多色或代数染色 (Phase 4)。
- 不处理 adaptive / AMR 重染色 (Phase 4+)。
- 不做 float32 优化 (保持 float64)。
- 不做 kernel fusion (forward-scale-backward 三步保持分离)。
- **不删除 Phase 2 代码** — 保留 level-scheduling fallback 路径。

---

## 9. 数学验收清单 (8 条, Phase 3 开始实现前用户/工程团队书面签字确认)

1. **多色 DILU 的 (a) vs (b) 选择清楚**: 我们做 Reordered DILU (式 1.1-1.3), 不做 Block-Jacobi-like (式 1.1.2)。后者在 2-color 下退化为 Jacobi, 是我们不要的。 **确定**.

2. **Red-Black 染色是 Phase 3 默认**, 基于规则 7-pt stencil 的 bipartite 性质。贪心多色在 bipartite graph 上退化为 red-black, 因此不单独实现; 代数染色留给 Phase 4。 **确定**.

3. **Kernel launch 数从 Phase 2 的 $\sim 2L_{\max}$ (对 $256{\times}128{\times}64$ 是 892) 降到 Phase 3 的 $2C = 4$**, 相当于 ≥ 200× sync point 削减。预期 per-iter wall time 5-10× 提速 (较确定, 需实测)。

4. **迭代 penalty 预期 1.5–3× (Duff-Meurant 范围)**。Phase 2 T7 的 24 iter 预期 Phase 3 将变为 36–72 iter。>72 iter 触发 §6 风险路径; >71 iter (接近 Jacobi) 触发 Phase 3 (a) 放弃。 **较确定**, AM $\rho = 10^3$ 实际值是 Phase 3 的贡献性测量。

5. **T7-equiv / 物理 A/B/C 三组测试必须过** (§5 表格 C6/C7/C8 是 Phase 3 acid tests)。C8 的 corr(|E|, |∇log ρ|) ≤ 0.25 是用户"绝不能出现几何误差花纹"指示的数学化。 **确定** 这是用户红线。

6. **$\tilde D_*$ 计算并行化是 Phase 3 附赠 payoff**: Phase 2 $D_*$ 目前串行, Phase 3 自然按颜色并行, $n/C$ 倍加速。这是 Phase 2 report "Deferred for Phase 3" 第 1 条 (Optimize dilu_factor from serial single-thread to level-parallel) 的答案。 **较确定**.

7. **Phase 2 的 cuSPARSE 路径不删除**: 作为 fallback 保留。若 Phase 3 失败 F2 (Test C halo), 系统可运行时回退到 Phase 2 level scheduling, Phase 4 AMG 启动。 **确定** — 是 risk management 的保险。

8. **关于 penalty 的"荣耀贡献"**: 目前文献没有在 AM $\rho = 10^3$ 3-D 工况测过 multi-color DILU 的 iter penalty。Phase 3 的 T7-equiv 本身是一个数据贡献, 不是简单对标已知基准。报告书中应明确 "Phase 3 T7-equiv 测得 X iter vs Phase 2 的 24 iter, 比例 α×, 与 Duff-Meurant 1989 在 2-D 5-pt 的 2× 对标", **不过度声称** 我们测的是 AM 工况而非 2-D Poisson。

---

## 附 A — 参考文献 (作者 + 标题, 不伪造 DOI)

- Duff, I. S. & Meurant, G. A. "The effect of ordering on preconditioned conjugate gradients", *BIT Numerical Mathematics*, 29:635–657, 1989. **核心 — 重排 penalty 原始数据**.
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003 — §12.2 (multicoloring orderings), §12.4 (distributed ILU).
- Benzi, M. "Preconditioning techniques for large linear systems: a survey", *J. Comput. Phys.*, 182:418–477, 2002. **综述 — orderings 与 parallelism trade-off 权威总结**.
- Li, R. & Saad, Y. "GPU-Accelerated Preconditioned Iterative Linear Solvers", UMSI Technical Report 2010-112, 2010. **GPU MC-ILU(0) 实测数据**.
- Naumov, M. "Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the GPU", NVIDIA Technical Report NVR-2011-001, 2011. Phase 2 引用.
- Naumov, M., Castonguay, P., & Cohen, J. "Parallel graph coloring with applications to the incomplete-LU factorization on the GPU", NVIDIA Technical Report, 2015. **NVIDIA 官方 Jones-Plassmann-Cohen ILU 算法**. Up to 6× GPU speedup.
- Iwashita, T. & Shimasaki, M. "Block Red-Black Ordering: A new ordering strategy for parallelization of ICCG method", *International Journal of Parallel Programming*, 2003.
- Doi, S. & Lichnewsky, A. "A graph-theory approach for analyzing the effects of ordering on ILU preconditioning", INRIA RR-1452, 1991.
- Suzuki, A. et al. "Block red-black MILU(0) preconditioner with relaxation on GPU", *Parallel Computing*, 2021.
- Jones, M. T. & Plassmann, P. E. "A parallel graph coloring heuristic", *SIAM J. Sci. Comput.*, 1993.
- Luby, M. "A simple parallel algorithm for the maximal independent set problem", *SIAM J. Comput.*, 1986.
- Meijerink, J. A. & van der Vorst, H. A. "An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", *Math. Comp.*, 1977.
- Gustafsson, I. "A class of first order factorization methods", *BIT*, 1978.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.

---

## 附 B — 变更与 Phase 1/2 对齐

- Phase 1 §3.2 "Route B" 的定性讨论 ("1.5–3× iter penalty, 文献数字稀薄") 被本 Phase 3 §3 的精确化和 §6 的失败模式分析取代。Phase 1 "我不确定 AM 具体 penalty" 的 known unknown 仍然有效, Phase 3 T7-equiv 回答它。
- Phase 2 §3.2 / §3.5 level scheduling 的并行度天花板分析 (RTX 3050 Laptop 饱和 $\bar{w} \approx 4700$, A100 $\bar{w} \approx 4700$ 严重 under-occupied) 被本 Phase 3 §4.3 / §4.6 的 red-black 路径对比 (每色 1M 完全饱和). 结论一致: Phase 3 在大网格和强 GPU 上优势放大。
- Phase 2 report §9 "Deferred for Phase 3+" 第 1 条 "Optimize dilu_factor from serial single-thread to level-parallel" 在本 Phase 3 §8.1 模块 B 中通过 "按颜色并行 $\tilde D_*$ 计算" 解决。
- Phase 2.5 report "Phase 3 (multi-coloring / ordering) can proceed on confident ground" 的前提 (Phase 2 DILU 物理健康) 为本 Phase 3 §5.3 的 C6/C7/C8 acid test 提供参照基线 (Phase 2 baseline 数字: max|div| 4.13e-9, ‖u‖∞ 9.64e-7, max|E_demean| 18.4)。

---

*End of Phase 3 Part 1 math report. Engineering team (jax-cfd-am-expert) takes over Part 2 implementation against §7 acceptance criteria and §8 hand-off specification.*
