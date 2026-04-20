# Phase 2 — cuSPARSE 级调度 (Level Scheduling) 的数学/算法分析与 DILU 集成规范

**Project**: JAX-GPU AM-CFD Platform — Stiff Pressure Poisson Solver Kernel Research
**Phase**: 2 / 4 — Part 1 (Math/Algorithm). Engineering integration is the parallel jax-cfd-am-expert's Part 2.
**Author role**: CFD-math expert (this document)
**Date**: 2026-04-20
**Scope**: Pure analysis. No code. Sets the numerical contract for Phase 2 implementation.

---

## 0. 重新briefing自查清单

本文档基于以下已读材料：
- `docs/design/phase1_dilu_math_foundation.md` §1 (DILU 定义), §2 (naive JAX 为何失败), §3 (三条出路), §4 (AM 刚性来源).
- `docs/benchmark/phase1_mvp_report.md` — dispatch baseline: RTX 3050 Laptop 约 267 µs 中位 (bare), 414 µs 中位 (under profiler); HLO 确认 zero host-device copy.
- `docs/benchmark/phase1_repro_HR54WV2_gtx1080.md` — GTX 1080 dispatch 约 455 µs 中位 (500 次采样).
- `CLAUDE.md` — 项目约束: `jax` + `jax.ffi` (not `jax.extend.ffi`), no `-ffast-math`, float32 需 double accumulators, FFI 前 `np.ascontiguousarray`.
- Phase 1 architecture 文档 §§4–6 关于 CSR 布局 (`row_ptr int32`, `col_idx int32`, `values float64`) 与 `CUDAToolkit::cusparse` 链接路径。

**未找到**: 任务书提到 "三台机器 (3050 Laptop, GTX 1080, RTX 5060)" 的第三份 repro 报告。磁盘上只有 3050 laptop 与 GTX 1080 两份。本文 baseline 推演仅引用已验证的两条数据，不为 5060 结果作声明。如 5060 repro 后续补齐，§4 的 back-of-envelope 需用实测数替换。 **不确定** 该文件是否仅是未合并，但就本文而言我按 "2 台已验证" 的事实陈述。

**版本警觉 (Master Plan "版本警觉" 条)**: 本文基于 WebSearch 2026-04 所见的 cuSPARSE 13.2 (3/2026 发布) 及 CUDA 12.x 文档。API 声明以实测 `nvcc --version` / `cusparse.h` 为准；本文结论 API 层面 **需验证** 的部分都明确标注。

---

## 1. cuSPARSE API 侦察 (numerical-methods 视角)

### 1.1 两代 API 的方法论区分

cuSPARSE 同时维护两套 triangular-solve / ILU 接口，本文称之 **Legacy** 和 **Generic**。两者解决的数学问题相同 (稀疏三角系统 $Tz = r$ 其中 $T$ 下或上三角)，但 API 设计哲学、内部算法选择空间、以及与 XLA 无状态范式的契合度都有实质差别。

#### 1.1.1 Legacy: `cusparseXcsrilu02` + `cusparseXcsrsv2`

- `cusparseDcsrilu02` (双精度变体) 执行 **数值 ILU(0) 分解**: 给定 CSR 形式的 $A$，原地把对角和非对角项替换为 $\tilde L, \tilde U$ 的非零项。其本身 **不做** 三角求解；只是完成分解。
- `cusparseDcsrsv2_analysis` 扫描稀疏模式，构建 level set 分层，生成一个不透明的 `csrsv2Info_t` 句柄。
- `cusparseDcsrsv2_solve` 使用该句柄做三角求解；内部以 level-set BFS 为基础的并行波前 (wavefront) 调度。

**API 接口状态**: Legacy API 在 CUDA 12.x 系列中正式标记为 DEPRECATED。按照 NVIDIA 文档惯例 "APIs marked [DEPRECATED] on release X.Y will be removed in release X+1.0"：Legacy 被弃用的时刻对应 cuSPARSE 12.x，Generic 为迁移目标。到 cuSPARSE 13.2 (2026-03) legacy 基本转为维护状态。 **较确定** (WebSearch 交叉确认，但我未逐行核对 12.9 头文件)。

**对我们的含义**: 因为 DILU 不是 ILU(0) (见 §2)，`csrilu02` 对我们其实从一开始就 **不相关**；我们只会借用 `csrsv2` 做三角求解。而 `csrsv2` 被 `SpSV` 替代，所以我们应该直接跳过 Legacy。

#### 1.1.2 Generic: `cusparseSpSV`

Generic API 是 cuSPARSE 的现代面貌, 特征:
- `cusparseSpMatDescr_t` / `cusparseDnVecDescr_t` — 通用描述符, 支持 CSR / COO / Blocked-ELL 等多种存储格式。
- 三阶段接口:
  1. `cusparseSpSV_bufferSize(handle, op, α, matA, vecR, vecZ, computeType, SPSV_ALG_DEFAULT, spsvDescr, &bufferSize)` — 返回 analysis 阶段所需工作区字节数，与 $\mathrm{nnz}(A)$ 成正比。
  2. `cusparseSpSV_analysis(..., externalBuffer)` — 做结构分析；对 CSR 输入这就是 level-set 提取 (见下文)。结果存于不透明的 `cusparseSpSVDescr_t`。
  3. `cusparseSpSV_solve(handle, op, α, matA, vecR, vecZ, computeType, SPSV_ALG_DEFAULT, spsvDescr)` — 实际求解。
- CUDA 12.x 之后新增 `cusparseSpSV_updateMatrix(..., newValues, updatePart)` — 允许在 **保持稀疏模式不变** 的前提下更新矩阵值 (含对角)，跳过重做 analysis。对 AM 场景 **关键**: 网格不变时，$D_*$ 随时间步变化但稀疏模式不变，我们就能重用 analysis。 **较确定** (API 存在且用途如描述; 具体在 CUDA 12.几引入 **需验证**)。

**算法 (内部)**: Generic `SpSV_ALG_DEFAULT` 与 Legacy `csrsv2` 的内部数学模型本质相同 — level scheduling。NVIDIA 文档里 (2026-04 确认) 明确说：
> "cuSPARSE partitions rows into several levels based on dependencies. It determines an execution schedule that ensures levels execute in sequence, while all rows in the same level execute in parallel."

以及用户可通过 `CUSPARSE_SOLVE_POLICY_NO_LEVEL` (Legacy) / 等价的 `SPSV_ALG_*` 选项 (Generic) 关闭 level 信息，此时 fallback 为 sync-free approach (可能基于 Naumov 2011 的 atomic-based 方案或更新的自适应算法)。 **较确定**.

**是否仍是 level scheduling**: WebSearch 结果显示 — **是的，Generic API 的 DEFAULT 算法仍基于 level scheduling**, 但在 CUDA 12.6+ 的 release notes 中有 "cusparseSpSV() performance was improved for both analysis and solving phases" 的记录，暗示底层可能已经混合了分级 + 稀疏调度优化 (而非严格 Naumov-2011 算法)。对我们的意义：
- 收敛行为 (iteration count) **不变** — 因为解的是同一个精确三角系统。
- 延迟特性 (kernel launch 模型、level 内并行度) 可能比 2011 年原论文略优。我们的 benchmark 应以实测为准，不要照搬 Naumov 原数据。

#### 1.1.3 没有 `cusparseSpILU` — 我们自己算 $D_*$

**关键事实**: Generic API 中 **不存在** 一个 "generic ILU factorization" 替代 `csrilu02`。NVIDIA 的迁移指引是 "对于 ILU(0) 分解，请继续使用 Legacy `csrilu02`"。

这对 DILU 是一个 **意料之外的便利**: 因为 DILU 本就不需要 ILU(0) 分解 (§2 解释)。我们用自己的 kernel 算 $D_*$，然后把 $A$ 的 off-diagonals + 我们算出的 $D_*$ 合成两个三角矩阵的 CSR 描述符，交给 `SpSV`。全程不碰 `csrilu02`。

### 1.2 Analysis / Solve 阶段分工

| 阶段 | 输入 | 输出 | 复杂度 | 依赖 |
|---|---|---|---|---|
| bufferSize | $(A, \mathrm{op}, \mathrm{computeType})$ | `bufferSize: size_t` | $O(1)$ 常数 | 纯 host 查询 |
| analysis | $(A, r, z)$ + externalBuffer | `cusparseSpSVDescr_t` (opaque) | $O(\mathrm{nnz}(A))$ + BFS on DAG | **只依赖稀疏模式** $\mathcal{S}(A)$，不依赖 values |
| solve | $(A, r, z, \mathrm{spsvDescr})$ | $z$ | $O(\mathrm{nnz}(A))$ | 依赖 values + analysis |
| updateMatrix | $(\mathrm{spsvDescr}, \mathrm{newValues}, \mathrm{updatePart})$ | 就地更新 | $O(\mathrm{nnz}(A))$ | 保持模式不变 |

**NVIDIA 文档明确承诺的数值保证**:
- `SpSV_solve` 给出 $z = T^{-1}r$ 的 **精确解** (在 compute type 的浮点精度内，无近似或迭代截断)。它不是一个迭代近似求解器，而是一个精确的回代。
- 这意味着 cuSPARSE 路径对 DILU 前处理是 **bit-equivalent** 于串行 DILU 的 (up to FP 加法结合律)。
- 没有任何 "容忍因子" / "dropping threshold" / ILU-T 的近似 — 纯粹的稀疏模式保真求解。

**我的解读 (较确定)**: cuSPARSE level scheduling 改变的是 **执行并行性**，不改变 **数值语义**。因此 Phase 2 目标是 "把 Phase 1 测试中的 $y = D^{-1}(b - Ax)$ 升级为 $y = M_{\text{DILU}}^{-1}(b - Ax)$ 并保持 ULP-level 一致性"，没有任何 "算法近似导致的精度退化" 需要担忧。

### 1.3 DAG 抽取 = level scheduling 的数学内容

Analysis 阶段做的事，形式上是：给定 $T$ (下三角 with 稀疏模式 $\mathcal{S}(T)$)，构造分级映射 $\ell: \{1, \ldots, n\} \to \mathbb{Z}_{\geq 0}$:

$$
\ell(i) = 1 + \max\!\big(\{\ell(k) : k < i, (i,k) \in \mathcal{S}(T)\} \cup \{-1\}\big).
$$

等价的递归 BFS 算法: $\ell(i) = 0$ 对所有无 predecessor 的 $i$; 其余 $\ell(i) = 1 + \max_{k \in \mathrm{pred}(i)} \ell(k)$.

这是一个 **拓扑排序** 的特殊形式 (最早可行时间)。BFS 实现 $O(|V| + |E|) = O(n + \mathrm{nnz}(A))$。cuSPARSE analysis phase 内部这就是它要做的事 (可能加上负载均衡/绑核启发式)。 **确定**.

---

## 2. DILU ≠ ILU(0) — cuSPARSE 集成点的精确刻画

### 2.1 Phase 1 数学回顾

重抄 Phase 1 §1.2 的 DILU 递推 (本文需要它的每一项)：

$$
M_{\text{DILU}} = (D_* + L)\, D_*^{-1}\, (D_* + U), \tag{2.1}
$$

$L, U$ 是 $A$ 的 **原始** 严格下/上三角 (未修改)，而

$$
d_i = a_{ii} - \sum_{k<i,\,(i,k)\in\mathcal{S}(A)} \frac{a_{ik}\, a_{ki}}{d_k}. \tag{2.2}
$$

对称 SPD 情形 ($a_{ik} = a_{ki}$) 退化为

$$
d_i = a_{ii} - \sum_{k<i,\,(i,k)\in\mathcal{S}(A)} \frac{a_{ik}^2}{d_k}. \tag{2.3}
$$

**与 ILU(0) 的结构差**:
- ILU(0) 产生 $\tilde L_{\mathrm{off}} \neq L_{\mathrm{off}}$ (非对角被修改); 需要分别存储 $\tilde L$ 和 $\tilde U$。
- DILU 只产生 $D_*$; 非对角复用 $A$ 的。存储节约 $2 \times$ off-diagonal 项。

这是 OpenFOAM `DILU`/`DIC` 在 CPU 上明显领先 `ILU(0)` 的原因之一: 更小的 working set, 更好的 cache 行为 (Phase 1 §1.2 提及)。

### 2.2 与 cuSPARSE SpSV 契合的精确配方

关键观察: `SpSV` 求解的三角系统由一个 **CSR 描述符 + 对角指示** 完全决定。它不关心这个三角矩阵是来自 ILU 还是 DILU; 它只看 CSR 数据。

那么 DILU 的 forward+backward sweep 可如下编排:

**Step 1 — 我们自己算 $D_*$** (§2.3)。结果是一个长度 $n$ 的 device 向量。
**Step 2 — 合成两个 CSR 描述符** $L^* = D_* + L_{\text{strict}}$ 和 $U^* = D_* + U_{\text{strict}}$.
- 对 **对称** $A$ 且采用 CSR 形式时的便利: $L^*$ 和 $U^*$ 共享同一个稀疏模式 (仅对角被替换, 上/下三角视角不同)，只是 diagFirst 标志位相反。
- 可实现为: 用 `A` 的原 CSR, 替换对角项数组中的 $a_{ii}$ 为 $d_i$。注意 CSR 中对角在 `row_ptr[i] .. row_ptr[i+1]` 的位置不固定 — 需要预先记录对角偏移 (一个 `diag_offset[n]` 数组)。

**Step 3 — Forward solve**: 求解 $L^* y_1 = r$。
由 (2.1) 展开: $(D_* + L) y_1 = r$，即

$$
y_{1,i} = \frac{1}{d_i}\!\left(r_i - \sum_{k<i,\,(i,k)\in\mathcal{S}(A)} a_{ik}\, y_{1,k}\right). \tag{2.4}
$$

这 **正是** `SpSV` 求解 $L^* y_1 = r$ 的标量展开，因为 $L^*$ 的对角是 $d_i$ (不是 $a_{ii}$)，而 `SpSV` 把对角项当 divisor。 **这里最容易踩的坑**: 如果把 $L^*$ 构造为 "复用 $A$ 的对角" (误操作)，SpSV 会用 $a_{ii}$ 除，结果不是 DILU 而是 ILU(-1)-style 近似，收敛会恶化。

**Step 4 — Backward solve**: 求解 $U^* z = D_* y_1$。
由 (2.1)' $(D_* + U)z = D_* y_1$:

$$
z_i = y_{1,i} - \frac{1}{d_i}\sum_{k>i,\,(i,k)\in\mathcal{S}(A)} a_{ik}\, z_k. \tag{2.5}
$$

**这里有第二个易错点**: right-hand side 不是 $y_1$ 而是 $D_* y_1$. 可以在 Step 3 末尾就地计算 $y_1 \leftarrow y_1$ (已含 $1/d_i$), 然后在 Step 4 调用 `SpSV(U^*, D_* y_1, z)`; 或者把 $D_*^{-1}$ 吸收到 $U^*$ 的对角 (即让 $U^*$ 对角等于 $1$ 而 off-diagonals 乘以 $1/d_i$). 后一种形式更接近标准 "unit upper triangular" 的 ILU 约定，但会破坏 "CSR values 与 $A$ 基本共享" 的内存省空间优势。 **推荐做法**: 保持 $U^* = D_* + U_{\text{strict}}$，在 Step 4 调用前做一次 cheap kernel `y1_scaled[i] = d_i * y1[i]`，然后 `SpSV` 求解。

### 2.3 $D_*$ 的计算 — 独立的串行 DAG

递推 (2.2) 本身具有 $G_L$ 的依赖图: $d_i$ 需要 $\{d_k : k \in \mathrm{pred}(i)\}$ 已知。这意味着 $D_*$ 的计算也受 level scheduling 制约 — 结构上是一个额外的 forward-sweep。

**几个工程选项**:
- **A. 我们自己写 CUDA kernel 按 level 并行算 $d_i$**. 需要复用 `SpSV_analysis` 的 level set 输出。问题: Generic API 的 `cusparseSpSVDescr_t` 是 **opaque**, 不暴露 level array。我们若想复用 levels, 必须 **自己再跑一次 BFS** (同样 $O(\mathrm{nnz}(A))$), 然后自己按 level 分派。
- **B. 把 $D_*$ 计算也封装成一次 `SpSV` 式操作**: 把递推 (2.2) 改写为一个三角线性系统的对角提取 — 但这不完全等价，需要引入辅助变量，绕路且不清晰。
- **C. 放弃并行 $D_*$，串行在 GPU 上算一次 per time step**. 成本: $O(n)$ 串行步 + 每步 $O(\mathrm{nnz\_per\_row})$ 算术。对 $n = 10^6$ 约 $10^6$ 步 × 每步 sub-µs = ms 量级。相较一次 PCG 可能上百次迭代的总时间 (§4), 这个一次性 $D_*$ 成本可以承受但不理想。

**我的建议 (较确定)**: Phase 2 先走 C (串行 $D_*$, 验证正确性)，Phase 3 升级 A (自行 BFS + 并行 $D_*$)。Phase 2 的目标是端到端跑通，不是每个 kernel 都最优。

### 2.4 数学正确性保证 — 哪里容易出错

整理一下 §2.2 和 §2.3 的坑:

| 错误 | 表现 | 检测方法 |
|---|---|---|
| $L^*$ 对角错用 $a_{ii}$ | 收敛仍近似，iter 数比 reference 多 $\sim 1.5\times$ | 与串行参考实现逐元素对比 residual |
| Step 4 忘乘 $D_*$ | 结果与 DILU reference 不一致 | 1 步 PCG 的预处理残差 $z = M^{-1}r$ 与 reference 对比, ULP floor |
| $D_*$ 计算顺序错乱 (level 分派 bug) | $d_i$ 值对 但部分行用到了未更新 $d_k$, 数值恶化但可能不显式出错 | 对比对 $D_*$ 向量 逐元素 |
| CSR 中对角位置偏移错 | `SpSV` 把某个 off-diagonal 当 divisor, NaN 或巨大 residual | 单步 residual check, diag_offset 单独 unit test |
| `update_matrix` 未调用, analysis 被旧 values 污染 | 正常收敛但 $M$ 不对应当前时间步 | 在 time stepping 中显式要求 per-step update |

**此表应作为 §5 acceptance criteria 的基础**。

---

## 3. Level scheduling 应用到 AM Poisson 矩阵

### 3.1 标准 7-点 3D Laplacian on $N_x \times N_y \times N_z$ (natural lex order)

Phase 1 §1.3 已记: 

$$
L_{\max} = N_x + N_y + N_z - 2. \tag{3.1}
$$

**每级平均宽度**:

$$
\bar w = \frac{n}{L_{\max}} = \frac{N_x N_y N_z}{N_x + N_y + N_z - 2}. \tag{3.2}
$$

**宽度分布 (精确)**: 第 $\ell$ 级包含所有满足 $i_x + i_y + i_z = \ell$ 的单元 ($0 \leq i_x < N_x$ 等)。这是一个三维超平面与立方体的交; 宽度函数先线性增长到峰值 $\sim 3 \bar w / 2$ 附近然后对称下降。第一级和最后一级宽度均为 $1$ (单个角落单元)。

**对 AM realistic grid $256 \times 128 \times 64$**:
- $n = 2.10 \times 10^6$
- $L_{\max} = 256 + 128 + 64 - 2 = 446$
- $\bar w \approx 4.70 \times 10^3$
- 峰值宽度 (在 $\ell \approx L_{\max}/2 = 223$) 约 $8.2 \times 10^3$ (近似, 三角-菱形几何的组合数)

**对 $256^3$ 立方**:
- $n = 1.68 \times 10^7$
- $L_{\max} = 766$
- $\bar w \approx 2.19 \times 10^4$
- 峰值 $\sim 3.3 \times 10^4$

### 3.2 并行度天花板与 SM 饱和点

**RTX 3050 Laptop** (Phase 1 硬件, 我们的最低规格):
- 20 SMs × 1536 threads/SM = **30 720 threads** concurrent
- 每个 thread 处理一行 DILU forward 扫描中的一个 level-内行 (one-thread-per-row 模型)

**饱和公式**: 级 $\ell$ 的并行度利用率
$$
\eta(\ell) = \min\!\left(1, \frac{w(\ell)}{N_{\text{SM-threads}}}\right). \tag{3.3}
$$

对 $256 \times 128 \times 64$ 网格 (AM realistic):
- $\bar w = 4.70 \times 10^3$ 远小于 $3.07 \times 10^4$ 的 thread 容量
- **全部 446 级都处于 under-occupied 状态**: $\eta(\ell) < 0.27$ 平均。
- 这是一个 **结构性 occupancy 问题**: RTX 3050 Laptop 在 $256 \times 128 \times 64$ 上永远填不满。只有 $N_x N_y \geq 3 \times 10^4 / N_z^{\text{slope}}$ 级才饱和。

对 $256^3$:
- $\bar w = 2.19 \times 10^4$, 峰值 $3.3 \times 10^4$
- 中间级 $\eta \approx 0.71$ 平均; 头尾各 $\sim 150$ 级严重 under-occupied
- **有效并行度** $\approx 0.5 \times L_{\max} \times N_{\text{SM-threads}} + 0.5 \times L_{\max} \times \bar w$ ... 粗算为 $\bar\eta \approx 0.45-0.6$.

**RTX 5060** (未验证 baseline, 推断规格): Ada/Blackwell 架构，约 30 SMs, 每 SM 1536 = 46 080 threads concurrent。对我们的矩阵尺寸，5060 **更加** under-occupied — 这是 level scheduling 在 consumer GPU 上的一个反直觉结论: **更强的 GPU 不一定让 DILU SpSV 更快**，因为我们已经触及 DAG 并行度天花板。

**A100 / H100** (未来 Phase 3 目标): 108 / 132 SMs, 约 $1.7 \times 10^5$ / $2.0 \times 10^5$ threads concurrent。
- 对 $256^3$ 中间级 $\eta \approx 0.10-0.20$
- **绝对时间可能更快** (因为 clock + memory BW 好), 但 **相对 utilization 更差** — 这是我们 Phase 3 要和 multi-coloring / AMG 对比的真正关键点.

### 3.3 自适应网格 / 嵌入边界: level count 恶化

AM 场景常见变体:
- **局部加密**: 熔池+HAZ 区域 4×~8× 细化, 其余粗。稀疏模式脱离规则 7-点。
- **嵌入边界 (IB) / cut-cell**: 每个靠近边界的单元多出若干 "ghost" 项。
- **非矩形几何**: 粉末床的 packed-sphere 表征会产生非结构连接。

**后果**:
1. $L_{\max}$ 不再能用 $N_x + N_y + N_z - 2$ 估算。经验上 $L_{\max} \sim O(n^{1/3})$ 的立方根增长依然成立 (critical path 就是从网格 "最早" 节点到 "最晚" 节点的 graph distance), 但常数会膨胀 2-3×。
2. 级宽度分布方差变大 — 负载均衡更差。
3. 最麻烦的: **每次重网格化, cuSPARSE analysis 必须重跑**。`SpSV_updateMatrix` 只能在 **稀疏模式不变** 时重用 analysis。这对我们是关键成本:
   - 静态网格: analysis 跑一次, 摊到后续 $10^4-10^6$ PCG iterations 上, 可忽略。
   - 每 $k$ 时间步 adaptive refine: analysis 每 $k$ 步重跑一次。设 analysis 成本为 $C_{\text{ana}}$, solve 成本 $C_{\text{solve}}$, 每时间步 PCG 迭代数 $N_{\text{pcg}}$, 则单位时间步的等效 solve 成本是 $N_{\text{pcg}} \cdot C_{\text{solve}} + (1/k) \cdot C_{\text{ana}}$. 若 $C_{\text{ana}} \sim 10 C_{\text{solve}}$ (NVIDIA 给的经验) 且 $k = 20$, 则 analysis 摊销后相当于 $0.5 C_{\text{solve}}$ — 可接受。若 $k = 1$ (每步重网格), 成本 $10\times$ 纯 solve — 不可接受。

**Phase 2 决策点 (需与 jax-cfd-am-expert 确认)**: 我们的 AM 场景是否打算支持 adaptive? Phase 1 的 MVP 用的是静态 CSR, 隐式假设 "每次 PCG call 前 analysis 已就绪"。若 Phase 3 要 adaptive, 需要提前设计 `SpSV_updateMatrix` 与 "重新 analysis" 的切换协议。 **不确定** Phase 2 范围内是否需要处理这一点; 默认按 "静态网格, analysis 一次" 来定义性能 baseline。

### 3.4 Level scheduling 的 "critical path" 本质

抽象起来: level scheduling 的 wall-time lower bound 是

$$
T_{\text{solve}} \geq L_{\max} \cdot t_{\text{level}}, \tag{3.4}
$$

其中 $t_{\text{level}}$ 是一级的 fixed overhead (kernel launch + sync)。即使级内无限宽, 我们仍受 $L_{\max}$ × per-level launch 约束。

估算 $t_{\text{level}}$:
- cuSPARSE 在单次 `SpSV_solve` 内**不为每级单独发 kernel** — 通常是单个 kernel 内部用 atomic + level counter 实现 level 之间的同步 (Naumov 2011 / CUDA 12.x 的做法)。
- 因此 $t_{\text{level}}$ 实际是 "同步一个 SM-wide atomic + 访问下一级行列表" 的开销, 大约 $\sim 100$ ns - $1\,\mu s$ 级别。
- 对 $L_{\max} = 766$ ($256^3$), 下界 $T_{\text{solve}} \geq 766 \times 500\,\mathrm{ns} \approx 0.4\,\mathrm{ms}$.
- 实测 (Phase 2 需测) 应该在 $1-5\,\mathrm{ms}$ 级别 (算术 + memory BW 主导)。 **需验证**.

### 3.5 与 Phase 1 dispatch overhead 的比较

Phase 1 baseline: 单次 FFI dispatch 267-455 µs (跨 3050/1080)。

**重要观察**: 一次 PCG iteration 通常包含:
- 1× SpMV ($y = Ax$) — 1 FFI call
- 1× 预处理器应用 ($z = M^{-1}r$) = DILU forward + backward — 若封装为 1 个 FFI handler 则 1 call, 若 forward/backward 分开则 2 calls
- 2-3× 向量轴 (axpy, dot product) — 各 1 FFI call, 或可融合

总计 **4-6 个 FFI dispatch per PCG iteration**. 开销 $\sim 4 \times 300\,\mu s = 1.2\,\mathrm{ms}$ per iteration (3050 baseline).

**关键**: 对 $256 \times 128 \times 64$ ($n = 2 \times 10^6$) kernel 的 arithmetic 部分我预估 SpMV $\sim 1-2$ ms, SpSV $\sim 2-5$ ms (§3.4 下界估算)。

**Dispatch-dominated vs kernel-dominated 的交叉点** 大约在 $n \approx 10^5$ (小网格) vs $n \geq 10^6$ (大网格)。Phase 1 baseline 用 $n = 10^3$ 纯粹测 dispatch, 是 "dispatch-dominated" 的极端; 真实 AM 工况 Phase 2 应落在 "kernel-dominated" 一侧, dispatch 只占总时间 10-20%。

**含义** (对 Phase 2 决策): 不要把 Phase 1 的 270 µs 绝对数字当作 Phase 2 性能 baseline — 它只说明 "toolchain works"。Phase 2 应该用 "PCG 收敛总 wall time" 当指标, 从两端 (iteration count + per-iter time) 各自控制。

---

## 4. 刚性、前处理器质量与预期迭代数

### 4.1 Gustafsson 尺度律在不连续系数下的崩坏 — Phase 1 §4.4 的重访

Phase 1 结论重述:
- SPD, well-conditioned 2D Poisson: Gustafsson 1978 给 $\kappa(M_{\text{DILU}}^{-1}A) \sim \sqrt{\kappa(A)}$, 即 PCG iteration count $\propto \kappa^{1/4}$.
- 对 discontinuous-coefficient 问题 (我们的 $\gamma = 10^3$ AM 场景), 这个 scaling 在 "局部在界面附近" 的特征模式上崩塌, 退化到 Jacobi 级别 ($\sqrt{\kappa(A)}$).

**Phase 2 语境下的量化** (较确定的文献派生 + 需实测的常数):

假设 $A$ 的条件数 $\kappa_2(A) \approx \gamma / h^2 = 10^3 \times 6.5 \times 10^4 \approx 6.5 \times 10^7$ 对 $h = 1/256$ 的 $256^3$ (Phase 1 §4.1):

- **Jacobi-PCG**: iter count $\sim O(\sqrt{\kappa}) \approx 8 \times 10^3$. 不可用 (Phase 1 已结论).
- **DILU-PCG, "好问题"** (Gustafsson): $\sim O(\kappa^{1/4}) \approx 90$ iterations. 可用.
- **DILU-PCG, "坏问题"** (interface-dominated, Gustafsson 失败): $\sim O(\sqrt{\kappa}) \cdot 0.3 \approx 2.5 \times 10^3$, 其中 $0.3$ 是经验的 "DILU vs Jacobi on interface modes" 系数 — **非常不确定**, 文献中没有我能信的硬数字。

### 4.2 OpenFOAM 实践文献中的 AM Poisson 迭代数

我做了 WebSearch, 坦率说 **文献非常稀薄** — 专门发表 "AM Poisson DILU-PCG iteration count vs density ratio" 的学术工作几乎没有。最相关的是:
- OpenFOAM 用户论坛中的各种报告 (非同行评审)，典型 `interFoam` 海空界面问题 ($\gamma = 10^3$) 单时间步 pressure PCG 20-200 iter。
- LaserbeamFoam (本 repo `lasermeltfoam-audit` skill 引用的 Newcastle 工作) 在 LPBF 设定下: pressure PCG 报 50-500 iter 不等, 和时间步长、网格、密度比都强相关。
- AMGx 论文 (Naumov et al. 2015) 报告在 multiphase petroleum flow on $10^6$-cell grids, AMG 带 PCG 5-15 iterations vs ILU(0)-PCG 数百 iterations. **直接引用为 "AMG 赢 ILU 一个数量级" 的依据**。

**我愿意下注的范围** (较确定):
- $256^3$ 网格, $\gamma = 10^3$, 无 AMG, 纯 DILU-PCG: **单时间步 100-500 iterations** 常态, 起步阶段 (pressure 不稳) 可能 $> 10^3$。
- 放宽到 PISO/SIMPLE 多 pressure correction step: 每 correction step 可能只要 20-80 iter, 但每 time step 要 2-4 次 correction。
- 若 AMG 可用 (Phase 4): 可能压到 5-30 iter。

**不确定部分**: 具体数字需 Phase 2 / Phase 3 实测。建议 Phase 2 的基准测试里专门设一个 "AM-like 密度跳跃" 测试 matrix (Phase 1 T3 的扩展, 但带真实 3D 结构), 报告收敛迭代数。

### 4.3 Back-of-envelope: 总 PCG time = N × (per-iter time)

每次 PCG iteration 的组成 (在 Phase 2 的 cuSPARSE DILU 路径下):

| kernel | FFI 次数 | 内核时间估算 ($n=2\times 10^6$, 3050 Laptop) | dispatch 开销 |
|---|---:|---:|---:|
| SpMV ($Ax$) | 1 | ~2 ms | ~300 µs |
| SpSV fwd ($L^* y_1 = r$) | 1 | ~3 ms | ~300 µs |
| SpSV bwd ($U^* z = D_* y_1$) | 1 | ~3 ms | ~300 µs |
| axpy ×2 (residual update, direction update) | 2 | ~0.2 ms | ~600 µs |
| dot ×2 | 2 | ~0.2 ms | ~600 µs |
| **per-iter total** | **7** | **~8.4 ms** | **~2.1 ms** |

per-iter ≈ 10.5 ms (3050, $n = 2 \times 10^6$)。dispatch 占比 ≈ 20%。

**"dispatch 主导" vs "kernel 主导"**:
- 若 $N_{\text{pcg}} = 100$ iter: total ≈ 1.05 s per time step. PCG 内核主导 (80%).
- 若 $N_{\text{pcg}} = 500$ iter (interface-hard case): total ≈ 5.2 s per time step. 同样内核主导。
- 若 $N_{\text{pcg}} = 20$ iter (AMG-accelerated, 未来): 0.21 s per step. Dispatch 仍占 20% — 仍然可接受。

**什么情况下 dispatch 会成为瓶颈?** 当 $n \leq 10^5$ (小网格) 时, SpSV kernel 时间降到 ~0.3 ms, 和 dispatch 持平。此时每次 FFI dispatch 的 300 µs 变成 50% 成本, 需要 kernel fusion (把 SpMV + axpy 合一, 减少 call 次数)。**对 AM 实际工况 ($n \sim 10^6 - 10^7$), 不是问题**。

**Phase 2 有意义的工况下限** (较确定):

$$
n_{\text{threshold}}^{\text{Phase2 useful}} \approx 10^5 \text{ 单元}. \tag{4.1}
$$

低于这个, Phase 2 的 cuSPARSE 改造相对 Phase 1 "Python + fori_loop" 不会有数量级收益 — 因为我们不过是把一个 dispatch-bound 的东西替换成另一个 dispatch-bound 的东西。高于这个, Phase 2 的价值随 $n$ 线性增长。

### 4.4 与 OpenFOAM CPU DILU-PCG 的对比基线

OpenFOAM 单线程 CPU DILU-PCG 在 $10^6$-cell pressure 问题上, 代表性 wall-time ≈ 50-200 ms per PCG iteration (现代 Xeon). 我们 Phase 2 目标应该 **单次 PCG iteration** 上 GPU 不比 CPU 慢 (理想 $2-5\times$ 快); iteration count 必须与 CPU 参考在 **10% 之内**一致 (代表 cuSPARSE level scheduling 没有引入数值退化). **较确定** 这是可达的, 但需要 $n$ 足够大 (§4.3 的阈值)。

---

## 5. Phase 2 验收标准 (可测)

以下 8 条是 Phase 2 "完成" 的数学验收标志。每条都应在 Phase 2 交付物中有对应的自动化 test。

| # | 标准 | 度量 | 验收阈值 |
|---|---|---|---|
| **C1** | cuSPARSE DILU 预处理残差与 Phase 1 reference 一致 | $\|z_{\text{cusparse}} - z_{\text{ref}}\|_\infty / \|z_{\text{ref}}\|_\infty$ | $\leq 100 \cdot \kappa(M_{\text{DILU}}) \cdot \epsilon_{\text{mach}}$. 对 Phase 1 T2 ($\kappa \sim 10^2$): $\leq 2.2 \times 10^{-12}$; 对 T3 ($\kappa \sim 10^3$): $\leq 2.2 \times 10^{-11}$. |
| **C2** | $D_*$ 计算数值正确 | $\max_i \|d_i^{\text{gpu}} - d_i^{\text{ref}}\| / \|d_i^{\text{ref}}\|$ | $\leq 50 \cdot \mathrm{nnz\_per\_row} \cdot \epsilon_{\text{mach}} \leq 1 \times 10^{-13}$ (7-point) |
| **C3** | PCG 迭代数与 OpenFOAM CPU DILU-PCG 一致 | $N_{\text{pcg,gpu}} / N_{\text{pcg,cpu}}$ on 同一矩阵, 同 tolerance | ≤ 1.10 (10% 容差). 在 $64^3$ 和 $128^3$ 7-point Poisson 上测, AM-like 密度跳跃矩阵也测. |
| **C4** | SpSV analysis 摊销 break-even | 测 $C_{\text{ana}}$ 和 $C_{\text{solve}}$, 计算 break-even iteration count $N^* = C_{\text{ana}} / C_{\text{solve}}$ | 报告 $N^*$ 值; 若 $N^* < 10$, analysis 可忽略不计; 若 $N^* > 100$, 需引入 `updateMatrix` 路径 |
| **C5** | `SpSV_updateMatrix` 路径正确 (若启用) | 静态 mode 重复 solve 5 次 + 变动 values 后再 solve 5 次, 对比 "每次 re-analysis" 参考 | 所有 10 次结果在 C1 的容差内 |
| **C6** | 单次 PCG iteration wall time 优于 Phase 1 naive scan baseline | 在 $128^3$ 7-point Poisson 上, Phase 2 per-iter time < Phase 1 ref × 0.5 | 即至少 **2× 加速**, 否则 Phase 2 不算成功. 若 Phase 1 ref 是 "不可运行" (如真的 fori_loop 几秒), 则 Phase 2 必须给出 "可运行且 < 50 ms/iter" 的绝对阈值 |
| **C7** | Zero host-device copy 保持 | HLO 检查: 继承 Phase 1 T4 的做法, 在 `jax.jit` 下 compile 后 HLO 中 `copy-start`/`copy-done` count = 0, `custom-call` count 等于预期 FFI 数 | 严格 == 0 对 copy; custom-call 按协议定 |
| **C8** | 对失败模式的 deterministic error signaling | 故意构造: (a) $d_k = 0$ 触发 (2.2) 除零; (b) CSR `row_ptr[n] ≠ nnz`; (c) 非对称 pattern 传入对称优化的 kernel | FFI handler 返回 `ffi::Error`, 不 `cudaAbort`, 不静默产生 NaN。Python 侧捕获 exception。 |

**C1 中的因子 $\kappa(M_{\text{DILU}})$ 解释**: 回代求解的后向误差分析 (Higham 2002, *Accuracy and Stability of Numerical Algorithms*) 给出 $\|\hat z - z\|/\|z\| \leq \mathrm{cond}(M) \cdot O(n) \cdot \epsilon$. 对 DILU 的良态预处理器, $\kappa(M_{\text{DILU}}) \ll \kappa(A)$, 典型 $10^2 - 10^4$. 所以 C1 的 allowed error 在 $10^{-12}$ 量级; 若看到 $10^{-8}$ 级误差, 表示 kernel 有 bug。

**C3 的 "同一矩阵" 要求**: CPU OpenFOAM 和 GPU cuSPARSE 必须算的是 **bit-identical** 的 CSR; 矩阵生成 pipeline 需要统一。推荐: 用 Python+NumPy 生成参考 CSR, 写入 `.npz`, 两侧都从此文件读。

---

## 6. 工程团队 (jax-cfd-am-expert) 交接规格

### 6.1 本 Phase 需要实现的 kernel 清单

按调用顺序:

1. **`dilu_compute_D` kernel**: 输入 $(A_{\text{csr}}, \mathrm{diag\_offset})$, 输出 $D_* \in \mathbb{R}^n$ (device buffer). 依照 (2.2)/(2.3)。Phase 2 可以先串行 (per-row 单线程 sequential), Phase 3 升级为 level-parallel。
   - 检查: $d_i \neq 0$; 若 $|d_i| < \delta \cdot |a_{ii}|$ (e.g. $\delta = 10^{-12}$) 触发 `ffi::Error`. 这是 DILU 在非 M-matrix 上 breakdown 的唯一自我保护。

2. **`dilu_assemble_triangular` kernel**: 输入 $(A_{\text{csr}}, D_*, \text{"L"or"U"})$, 输出 CSR 形式的 $L^* = D_* + L_{\text{strict}}$ 或 $U^* = D_* + U_{\text{strict}}$. 实际上可以不产生新 values buffer — 用 "view into A's values with diagonal override" 的方式 (原地替换 diag); 但要注意 A 本身不能被污染，所以更干净的做法是复制 values 到新 buffer。
   - 注: 对对称 $A$, $L^*$ 和 $U^*$ 共享 pattern 只是 triangle 不同; cuSPARSE 描述符的 `CUSPARSE_FILL_MODE_LOWER` / `_UPPER` 标志区分。

3. **`cusparseSpSV` wrapper (bufferSize + analysis)**: 纯 C++ FFI handler, 调用 cuSPARSE 做 analysis。返回 `cusparseSpSVDescr_t` 保持为 opaque handle, 跨 JAX 调用边界用 `uintptr_t` 传递。注意: JAX 的无状态范式使 "保持 descriptor" 在 Python 侧需要一个 "manager object" 缓存; 这是 jax-cfd-am-expert 的架构负责部分, 但 math 侧的约束是: analysis 描述符必须与一组特定的 CSR pattern 绑定, 若 pattern 变, 必须重新 analyze (否则 silent wrong result)。

4. **`cusparseSpSV` wrapper (solve)**: FFI handler 调用 `cusparseSpSV_solve`. 必须 stream-aware (使用 XLA 传入的 CUDA stream, 见 Phase 1 architecture §3)。

5. **`dilu_apply` high-level wrapper**: 组合上述到一个 JAX 函数 `z = dilu_apply(A, D_star, r, spsv_L_descr, spsv_U_descr) -> z`. 对 Python 层暴露为一个 `jax.custom_call`-like op; 内部做 forward solve → element-wise $D_* \cdot y_1$ → backward solve。若要避免 3 次 FFI dispatch (forward / scale / backward), 可把三步融合到一个 C++ handler 里。 **建议**: Phase 2 先分三次, Phase 3 再 fuse。

### 6.2 跨 FFI 边界的数据结构

继承 Phase 1 约定 (architecture 文档 §4 / CLAUDE.md 约定), 扩展:

| 变量 | 类型 | 约束 |
|---|---|---|
| `row_ptr` | `int32[n+1]`, device | `row_ptr[0] == 0`, `row_ptr[n] == nnz`, 单调非降 |
| `col_idx` | `int32[nnz]`, device | 每行内列索引 **升序** (cuSPARSE 要求) |
| `values` | `float64[nnz]`, device | C 连续 |
| `diag_offset` | `int32[n]`, device | `col_idx[diag_offset[i]] == i` 对所有 $i$ |
| `d_star` | `float64[n]`, device | 由 `dilu_compute_D` 填充 |
| `spsv_L_desc`, `spsv_U_desc` | `uintptr_t` (opaque) | 由 Python 侧 "descriptor cache" 管理, 每个绑定到一个特定的 CSR pattern |
| `buffer_L`, `buffer_U` | `uint8[bufferSize]`, device | cuSPARSE analysis 阶段返回的 size 决定; 与 descriptor 寿命等同 |
| CUDA `stream` | 从 XLA FFI context 取 | 不能用 default stream |

**Python 侧 descriptor 管理** (此部分是 jax-cfd-am-expert 的主要架构挑战, 我在这里只给数学约束):
- 一个 `SpSVContext` 对象持有 `(A_pattern_hash, spsv_L_descr, spsv_U_descr, buffer_L, buffer_U)`.
- 每次 `dilu_apply` 调用, 先 hash CSR 的 `(row_ptr, col_idx)` (values 不参与 hash), 若命中 cache 则复用 descriptor, 若 miss 则调用 analysis FFI 建立。
- values 变化时 (time stepping) 调用 `cusparseSpSV_updateMatrix` FFI, 不触发 re-analysis。
- 若 pattern 变化 (adaptive refine, Phase 3+), descriptor invalidated, 重 analyze。

### 6.3 验收测试的具体形态 (对应 §5 的 C1-C8)

- **C1 单元测试**: 固定的 random Poisson CSR + random $r$, Python 侧用 NumPy scipy 的 `spsolve` (或手写回代) 算 $z_{\text{ref}}$, GPU 算 $z_{\text{cusparse}}$, 检查 infinity norm 比值。覆盖 Phase 1 T1/T2/T3 三矩阵。
- **C2 单元测试**: 同上但只比 $D_*$ 向量。
- **C3 集成测试**: 跑完整 PCG (在 Python 里手写 PCG 主循环, 调用 GPU 的 SpMV + DILU), 比较总 iter count 与 CPU 参考实现 (SciPy + 手写 DILU 或 PyAMG 的 DILU if available).
- **C4 性能测试**: 测 `cusparseSpSV_analysis` 单次 wall time 与 `cusparseSpSV_solve` 单次 wall time, 取比值, 报 break-even.
- **C5 回归测试**: 显式的 "pattern 不变 values 变" 场景。
- **C6 性能测试**: 总 PCG wall time vs Phase 1 reference. (Phase 1 的 "naive scan" 可能根本跑不动 — 这种情况 C6 退化为 "能在 $< 50\,\mathrm{ms}$ 内跑 1 iter on $128^3$".)
- **C7 HLO 检查**: 照抄 Phase 1 T4 的方法。
- **C8 异常处理测试**: 手工构造三类错误矩阵, 确认 Python 侧收到 exception 而非静默 NaN。

### 6.4 Phase 2 不做的事 (scope clarification)

**不做**:
- Multi-coloring (Route B) — Phase 3.
- AMGx 集成 (Route C) — Phase 4.
- Float32 优化 — Phase 3 (Phase 2 专注 float64 正确性).
- Adaptive refine 支持 — Phase 3+.
- 跨-GPU (multi-GPU) 扩展 — 目前不计划.

**做**:
- 静态 CSR pattern, float64, DILU-preconditioned residual 的 GPU 路径, 封装到 JAX FFI op, 端到端跑通 PCG。

---

## 7. 数学验收清单 (用户最终确认)

以下 6 条是 Phase 2 开始实现前, 用户与工程团队需要书面签字确认的:

1. **DILU vs ILU(0) 的数学区别清楚了**: Phase 2 **不调用** `cusparseXcsrilu02`; 我们自算 $D_*$, 把 $L^* = D_* + L_{\text{strict}}$, $U^* = D_* + U_{\text{strict}}$ 提供给 `cusparseSpSV`. **确定**.

2. **cuSPARSE Generic API (`cusparseSpSV`) 是目标**, 不是 Legacy `cusparseXcsrsv2`. 内部仍采用 level scheduling, 但 CUDA 12.x 做过性能改进 — 预计与 Naumov 2011 原文数据有偏差, 我们的 benchmark 以实测为准. **较确定** (基于 2026-04 WebSearch).

3. **Analysis/solve 分离的无状态挑战** 由 jax-cfd-am-expert 在 Python 侧架构一个 descriptor cache 解决; 数学侧的不变量是: **descriptor 绑定到 CSR pattern**, pattern 不变时可复用, `cusparseSpSV_updateMatrix` 支持 values-only 变化.

4. **验收标准 8 条 (§5 C1-C8)** 在实现开始前转化为 pytest-类的可执行测试; 不满足任意一条即 Phase 2 未完成.

5. **$D_*$ 的并行化延后到 Phase 3** — Phase 2 内 $D_*$ 可以串行在 GPU 上算, 或甚至 copy to CPU 算再 copy back (若串行 GPU 版本太慢)。Phase 2 交付的是 **正确性**, 不是 **每个 kernel 的最优**.

6. **Break-even 分析 (C4)** 给出 analysis 摊销阈值 $N^*$; 若目标 AM 工况 ($N_{\text{pcg}} = 100-500$ per time step) 远 $\gg N^*$, analysis 成本可忽略, 架构可以相对简单 (不必提前支持 adaptive re-analysis). 若 $N^* > N_{\text{pcg}}$, 必须强化 `updateMatrix` 路径.

7. **"dispatch 主导" 阈值** (§4.3): 对 $n < 10^5$ 单元, Phase 2 相对 Phase 1 naive 的性能优势不足以自证, 应在验收报告中注明 "Phase 2 meaningful domain: $n \geq 10^5$"; AM 实际工况 ($n \geq 10^6$) 落在这个域内, 安全.

8. **Gustafsson scaling 崩坏** 对 stiff AM Poisson 是预期行为, 不是 Phase 2 的 bug. Phase 2 只负责 **把 DILU 搬上 GPU 且收敛迭代数与 CPU 参考一致**; 迭代数本身的大小 (100-500 iter) 是 Phase 4 AMG 才能根治的.

---

## 附 A — 参考文献 (只列作者 + 标题, 不伪造 DOI)

- Naumov, M. "Parallel solution of sparse triangular linear systems in the preconditioned iterative methods on the GPU", NVIDIA Technical Report NVR-2011-001, 2011.
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003 — §10 / §12.
- Gustafsson, I. "A class of first order factorization methods", *BIT*, 1978.
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002 — §8 (triangular systems backward error).
- Meijerink, J. A. & van der Vorst, H. A. "An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", *Math. Comp.*, 1977.
- Naumov, M. et al. "AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods", *SIAM J. Sci. Comput.*, 2015.
- NVIDIA, *cuSPARSE Library Documentation*, release 13.2 (PDF dated 2026-03-05) — sections on Generic API / `cusparseSpSV` / level-based solve policy.
- NVIDIA, *CUDA Toolkit 12.x Release Notes* — `cusparseSpSV_updateMatrix` 引入与 perf 改进条目.
- OpenFOAM source: `src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner.C` / `DICPreconditioner.C` — CPU 参考实现, Phase 2 C3 对照基线.
- PETSc source: `src/mat/impls/aij/seq/seqcusparse/aijcusparse.cu` — 成熟的 `cusparseSpSV` 集成范例 (生产代码), 工程团队可参考其 buffer/analysis 管理模式.

---

## 附 B — 变更与 Phase 1 对齐

- Phase 1 §3.1 (Route A 的性能预期) 给出 "$\sim 5$ ms per application" 在 $256^3$ 网格. 本 Phase 2 §3.4 / §4.3 把这个估算细化为 "$2-5$ ms for SpSV_solve, ~8 ms for full PCG iteration including SpMV + axpy + dot". 量级一致, 没有矛盾.
- Phase 1 §5 的 FFI prototype target 是 $y = D^{-1}(b - Ax)$ (Jacobi residual); Phase 2 将这个升级为 $z = M_{\text{DILU}}^{-1}(b - Ax)$. **数学上** 前者是后者 $D_* = D, L = U = 0$ 的退化; **工程上** 后者多了 $D_*$ 计算 + 两次 SpSV. Phase 1 建立的 FFI toolchain 在 Phase 2 的扩展是新增 cuSPARSE 依赖 + descriptor 缓存, 不破坏 Phase 1 的 dtype / layout / stream / error 约定.
- Phase 1 acceptance 第 7 条的 "known unknowns" 里, "multi-coloring penalty" / "DILU→AMG 交叉点" / "α-thin-cell 条件数" 三项仍然是 unknown; Phase 2 不解决它们 (按 §6.4 scope), 但 Phase 2 的测试基础设施应该为 Phase 3-4 实验这些问题做好 ready。

---

*End of Phase 2 Part 1 math report. Engineering team (jax-cfd-am-expert) takes over Part 2 implementation against §5 / §6 specification.*
