# OpenFOAM v2506 DILU + PBiCG CPU 求解链路源码审计

**目的**: 把"复现 OpenFOAM 单进程 CPU DILU + PBiCG"作为一个 reverse-engineering target 锁定下来。所有"OpenFOAM 这么实现"的断言都给出 file:line。本文档**不写代码**，只把仿写需要保留的不变量列清楚。

**OpenFOAM 版本**: openfoam2506 (`/usr/lib/openfoam/openfoam2506`)
**对应 LaserbeamFoam case**: `LPBF_crosscheck` (T 用 `PBiCG + DILU`, pd 用 `PCG + DIC`)

---

## §1 文件清单 + 调用关系

### 1.1 关键文件（绝对路径）

| 文件 | 作用 |
|------|------|
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.H` | DILU 类声明：`rD_`、`precondition()`、`preconditionT()`、`calcReciprocalD()` |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C` | **本文档审计核心 — 全文 187 行**。注册成 `addasymMatrixConstructorToTable<DILUPreconditioner>` |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DICPreconditioner/DICPreconditioner.C` | 对称版（DIC, pd 用），用来对照 DILU 与 DIC 的差别 |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C` | T 方程的 outer solver（256 行）。注册为 `addasymMatrixConstructorToTable<PBiCG>` |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.H` | PBiCG 类声明，`mutable autoPtr<lduMatrix::preconditioner> preconPtr_` 缓存 |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCGStab/PBiCGStab.C` | 备用 outer solver（PBiCG fail 时官方推荐 fallback）— **本次审计**仅作对比，不复现 |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C` | `Amul`、`Tmul`、`sumA`、`residual`、`H1` — SpMV / SpMV^T 的核心 |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrix.H` | `lduMatrix` 类声明（diag/lower/upper + lduMesh ref + lowerCSR） |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C` | `solver` 基类：`normFactor()` (line 235) + `read()` 控制字典 |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixUpdateMatrixInterfaces.C` | 并行边界 halo exchange — **复现单进程版本时跳过** |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduAddressing/lduAddressing.H` | LDU 寻址语义说明（**§2 重点引用**） |
| `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduAddressing/lduAddressing.C` | `losortAddr()`、`ownerStartAddr()`、`losortStartAddr()`、`lowerCSRAddr()` 的 lazy 计算 |

### 1.2 调用关系（PBiCG + DILU on T）

```
laserMeltFoam TEqn.H
  └── TEqn.solve()
        └── fvMatrix::solve(controlDict)
              └── lduMatrix::solver::New("PBiCG", ...)
                    └── PBiCG::solve(psi, source, cmpt)        [PBiCG.C:69]
                          ├── matrix_.Amul(wA, psi, ...)         [PBiCG.C:97]   ← lduMatrixATmul.C:37
                          │     └── (可选) initMatrixInterfaces / updateMatrixInterfaces
                          ├── normFactor(...)                    [PBiCG.C:112]  ← lduMatrixSolver.C:235
                          │     └── matrix_.sumA(...)            [lduMatrixATmul.C:219]
                          ├── matrix_.Tmul(wT, psi, ...)         [PBiCG.C:139]  ← lduMatrixATmul.C:155
                          ├── lduMatrix::preconditioner::New("DILU", *this, controlDict_)
                          │     └── DILUPreconditioner ctor      [DILUPreconditioner.C:46]
                          │           └── calcReciprocalD()      [DILUPreconditioner.C:64]
                          │                 (face-loop: rD[u] -= U[f]*L[f] / rD[l]   then rD[c] = 1/rD[c])
                          │
                          └── do  while (! checkConvergence)
                                ├── precondition(wA, rA)         [DILUPreconditioner.C:95]
                                │     (forward: wA[u] -= rD[u]*L[f]*wA[l];  backward: wA[l] -= rD[l]*U[f]*wA[u])
                                ├── preconditionT(wT, rT)        [DILUPreconditioner.C:139]
                                │     (transpose sweep — uses losortAddr)
                                ├── wArT = sum(wA*rT)            (gSumProd)
                                ├── update p, pT  (β = wArT/wArTold, init: copy wA→p)
                                ├── matrix_.Amul(wA, pA, ...)
                                ├── matrix_.Tmul(wT, pT, ...)
                                ├── α = wArT / sum(wA*pT)
                                └── psi += α p;  rA -= α wA;  rT -= α wT
```

调用链跟 PCG-DIC（pd 路径）基本同形，只少了 transpose 那条腿。

---

## §2 LDU 数据结构

### 2.1 三数组存储

`lduMatrix` 把矩阵拆成三段（声明 `lduMatrix.H:94-100`）：

```
diagPtr_   : scalarField   长度 nCells   存储 a_ii
lowerPtr_  : scalarField   长度 nFaces   存储严格下三角的非零（对应 owner > neighbour 那一侧）
upperPtr_  : scalarField   长度 nFaces   存储严格上三角的非零
```

非零位置由两个 face → cell 索引数组**间接给出**（`lduAddressing.H:189-194`）：

```
lowerAddr() == owner    : labelUList   长度 nFaces, owner[f] = lower-cell index of face f
upperAddr() == neighbour: labelUList   长度 nFaces, neighbour[f] = upper-cell index of face f
```

**对每个 face f**：
- `lower[f]` 是矩阵元素 `A[neighbour[f], owner[f])]`（行 = neighbour, 列 = owner，严格下三角）
- `upper[f]` 是矩阵元素 `A[owner[f], neighbour[f])]`（行 = owner, 列 = neighbour，严格上三角）

注意 OpenFOAM 命名反直觉：**`lowerAddr()` 返回的是 `owner` 列表，`upperAddr()` 返回 `neighbour` 列表**。Amul 里 `lPtr = lowerAddr()`、`uPtr = upperAddr()`（见 `lduMatrixATmul.C:55-56`）—— 写代码时务必用这个约定，**不要按字面意思理解 lower/upper**。

### 2.2 OpenFOAM 把 owner/neighbour 怎么排序

`lduAddressing.H:39-66` 文档原话：

> The ordering of owner addresses is such that the labels are in increasing order, with groups of identical labels for edges "owned" by the same point. The neighbour labels are also ordered in ascending order but only for groups of edges belonging to each point.

具体 example（同文件 `:44-66`）：

```
owner    neighbour
0        1
0        20
1        2
1        21
...
```

**一句话**：face 全局按 `(owner, neighbour)` 字典序排列。**这个顺序是 OpenFOAM 整个求解器栈的隐藏不变量** —— 后面 DILU 的 forward sweep 严格依赖 face index 升序 = owner 升序。

### 2.3 衍生寻址（lazy build, lduAddressing.C）

| 名字 | 计算位置 | 含义 |
|------|----------|------|
| `losortAddr()` | `lduAddressing.C:34-91` | face index 列表，按 `neighbour[f]` 升序排（即"反向 sort"）。让"找出某 cell 作为 neighbour 的所有 face" 变成连续段 |
| `ownerStartAddr()` | `lduAddressing.C:94-127` | 长度 `nCells+1`：`ownerStart[c]` = 第一个 owner==c 的 face index。把 owner-loop 转成 cell-loop |
| `losortStartAddr()` | `lduAddressing.C:130-168` | 长度 `nCells+1`：losort 数组里第一个 neighbour==c 的位置 |
| `lowerCSRAddr()` | `lduAddressing.C:171-182` | 把 lower 系数按 CSR 行序重排后的列索引 |

**这三组寻址只是把同一份 (owner, neighbour) 数组按不同顺序重新索引**。**复现时只要保留 owner / neighbour 两个原数组，losort 系列可以从它们 derive**（OpenFOAM 自己也是 lazy build）。

### 2.4 为什么用 LDU 而不是 CSR

OpenFOAM 历史上 LDU 是**面向 finite-volume 矩阵装配**的最自然格式：FV 装配天然按面循环（每个 face 写两个对称位置：owner-行 加 -aF，neighbour-行 加 -aF），diag 单独累积。

但**对求解器（DILU / SpMV）这是个约束**：

- 一行的所有非零**不连续**存储（既出现在 upper 段也出现在 lower 段，且按 face index 散落）
- 不能像 CSR 那样按行做 `for j in row(i)` 内积
- DILU 因此被迫写成 face-loop + atomic-style scatter，而不是行内规约

**v2506 已经加了 `lowerCSR` lazy 推导（`lduMatrix.H:103`）**，Amul 里 `if (hasLowerCSR())` 分支（`lduMatrixATmul.C:76-121`）走 cell-based looping。但 **DILU.C 里还是 face-loop**（没用 lowerCSR），见 §3.2。

### 2.5 对称矩阵（pd / DIC）vs 非对称（T / DILU）的 LDU 区别

- **对称**：`lower[f] == upper[f]` 是数学事实，所以 OpenFOAM **只存 upper**，`lower()` 在 symmetric matrix 上返回 upper 的别名。`DICPreconditioner.C:80` 里的 `upperPtr[face]*upperPtr[face]` 就是利用 `lower=upper` 把 schur 补化成 `U²/D`。
- **非对称**：`lower != upper`，必须分别存。`DILUPreconditioner.C:81` 的 `upperPtr[face]*lowerPtr[face]` 就是 schur 补的一般形式 `U·L/D`。

DIC 也只需要 `precondition()`（对称），DILU 必须额外提供 `preconditionT()` 给 BiCG 的 transpose 腿（对比 DILUPreconditioner.H:101-106 vs DICPreconditioner 没有 preconditionT）。

---

## §3 DILU 算法的精确数值流程

### 3.1 数学定义

OpenFOAM DILU 是 D-ILU(0)：保留**所有原矩阵的非零位置**作为 fill-in 集合（即只在已有 LDU 三段位置上构造 ILU），仅修改 D。

**reciprocal diag 公式（v2506 实测，见 DILUPreconditioner.C:79-82）**：

```
对每个 cell c (initial):  rD[c] = a_cc

for face f = 0 .. nFaces-1   (按 owner/neighbour 升序):
    rD[ neighbour[f] ]  -=  upper[f] * lower[f] / rD[ owner[f] ]

最后:  rD[c] = 1 / rD[c]
```

数学上等价于：
$$
\tilde{D}_i = a_{ii} - \sum_{j < i, \, (i,j) \in \text{nz}} \frac{a_{ij} \cdot a_{ji}}{\tilde{D}_j}
$$
其中 $\tilde{D}_j$ 是已经更新过的对角（**Gauss-Seidel-style 顺序依赖**）。

> **黄牌警告 #1 — 顺序依赖**：上面 face-loop 严格按 face index 升序执行，且 `rD[neighbour[f]]` 的更新会被后续 face 读到（如果某个后续 face 的 owner 等于当前 neighbour）。**不可以 OpenMP 并行这个 loop**，**不可以 reorder face**。仿写时这是头号雷。

### 3.2 计算实现（DILUPreconditioner.C:64-92）

```
calcReciprocalD(rD, matrix):
    rD = matrix.diag()                           # copy

    uPtr = upperAddr().begin()                   # = neighbour
    lPtr = lowerAddr().begin()                   # = owner
    upperPtr = upper().begin()
    lowerPtr = lower().begin()
    nFaces = upper().size()

    for face = 0 .. nFaces-1:
        rD[uPtr[face]] -= upperPtr[face] * lowerPtr[face] / rD[lPtr[face]]
        # 注意是 rD[uPtr[face]] (= rD[neighbour]) 被更新
        # 用的是 rD[lPtr[face]] (= rD[owner])，且 owner < neighbour，所以 rD[owner] 已经定型

    nCells = rD.size()
    for cell = 0 .. nCells-1:
        rD[cell] = 1.0 / rD[cell]
```

**关键事实**：因为 owner < neighbour 且 face 按 owner 升序排，**当我们要写 `rD[neighbour[f]]` 时，所有 owner ≤ neighbour[f] - 1 的 face 都已经处理完，所以 `rD[owner[f]]` 已经是它的最终值**。这正是 ILU 顺序依赖能在 LDU 格式上一遍跑完的原因。

### 3.3 precondition（forward + backward sweep, DILUPreconditioner.C:95-136）

求 `wA = M⁻¹ rA`，其中 `M = (D̃ + L) D̃⁻¹ (D̃ + U)`，分两步：

**Step A — initialize**:
```
for cell: wA[cell] = rD[cell] * rA[cell]      # = D̃⁻¹ rA
```

**Step B — forward sweep（解 (D̃ + L) y = rA, y stored in-place in wA）**:
```
for face = 0 .. nFaces-1:
    wA[uPtr[face]] -= rD[uPtr[face]] * lowerPtr[face] * wA[lPtr[face]]
    # = wA[neighbour] -= rD[neighbour] * L[f] * wA[owner]
```

**Step C — backward sweep（解 (D̃ + U) wA = D̃ y）**:
```
for face = nFaces-1 .. 0:
    wA[lPtr[face]] -= rD[lPtr[face]] * upperPtr[face] * wA[uPtr[face]]
    # = wA[owner] -= rD[owner] * U[f] * wA[neighbour]
```

> **黄牌警告 #2 — backward 用的是 face 倒序，不是 cell 倒序**。这跟教科书 ILU 后向回代"按 cell 从大到小"不完全一样：face 倒序意味着对同一个 cell c 而言，该 cell 在 backward sweep 中作为"被写入方"出现的次数取决于它作为 owner 出现在多少个 face。**对每个 face 只访问一次，靠 face 排序自动满足"先大 cell 后小 cell"的 substitution 顺序**。仿写时不要"优化"成 cell-loop。

### 3.4 preconditionT（transpose precondition, DILUPreconditioner.C:139-184）

求 `wT = M⁻ᵀ rT`，需要 transpose 的 LU sweep。OpenFOAM 用 `losortAddr()` 来反向遍历：

**Step A**: `wT = rD * rT`（同 precondition）

**Step B — forward (transpose, 用 upper 而不是 lower)**:
```
for face = 0 .. nFaces-1:
    wT[uPtr[face]] -= rD[uPtr[face]] * upperPtr[face] * wT[lPtr[face]]
    # 注意：用的是 upperPtr[face]，不是 lowerPtr[face] —— 这就是 transpose
```

**Step C — backward (transpose, 用 losort 重排 + lower)**:
```
for face = nFaces-1 .. 0:
    sface = losortPtr[face]
    wT[lPtr[sface]] -= rD[lPtr[sface]] * lowerPtr[sface] * wT[uPtr[sface]]
    # losort 让 face 按 neighbour 升序遍历 — transpose 后的"行升序" backward
```

> **黄牌警告 #3 — preconditionT 唯一处用 losortAddr**。precondition (非 T) 不需要 losort，calcReciprocalD 也不需要。仿写时如果只跑 PCG-DIC（对称）可完全不构建 losort；要跑 PBiCG-DILU 必须构建。

### 3.5 跟 textbook ILU(0) 的差异

| 方面 | textbook ILU(0) (Saad) | OpenFOAM DILU |
|------|------------------------|---------------|
| Fill-in 集合 | 与 A 相同稀疏模式 | 与 A 相同（一致） |
| 修改对象 | L, U, D 全部 | **仅 D**，L、U 沿用 A 的下/上三角原值 |
| factor 公式 | row-by-row Gaussian elim. | 单遍 face-loop，face 按 owner 升序保证依赖关系 |
| 回代 | row-loop + 内列循环 | face-loop（forward 自然序，backward 倒序），间接通过 face 排序保证依赖 |
| 并行性 | 极差（行依赖） | 极差（face 依赖），但**单遍 face-loop 缓存友好** |

OpenFOAM 这种"D-ILU = 只修 D + 重用 LU 系数"是 DILU 名字的来源（**D**iagonal-based **ILU**）。比完整 ILU(0) 便宜（不存 L̃, Ũ 副本，省一倍内存），代价是收敛慢于完整 ILU(0)，但对 FV 矩阵效果接近且实现极简。

### 3.6 边界 / interface 怎么并入系数

`lduMatrix` 把内部 face 系数和并行/coupled boundary patch 系数**分开存**：
- `diag, lower, upper` 只放内部 face
- `interfaceBouCoeffs` (FieldField) 放每个 patch 的耦合系数

`Amul` (`lduMatrixATmul.C:64`) 调用 `initMatrixInterfaces` + `updateMatrixInterfaces` 把 patch 贡献加到 `Apsi` 上。**单进程 + 全 zeroGradient/fixedValue 边界**时，interface 列表为空，这段是 no-op。

> **复现策略**：**单进程版本可以完全忽略 interface 这一坨**（dump 出来的 `data.npz` 已经把 interface 系数预先 fold 进 diag/lower/upper 或忽略 —— 见 `docs/design/openfoam_crosscheck_plan.md` 的 dumper 设计）。MPI 分布式版本才需要 halo exchange，这是后期工作。

---

## §4 PBiCG outer solver

### 4.1 LaserbeamFoam 实际配置（验证）

`/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/system/fvSolution:94-101`:
```
T
{
    solver          PBiCG;
    preconditioner  DILU;
    minIter         1;
    tolerance       1e-12;
    reTol           0.0;        # 注意原文件拼写为 reTol，OpenFOAM 默认会 fall back to relTol=0
}
```

**确认**: T 用 `PBiCG + DILU`，**不**是 PBiCGStab。PBiCGStab 是 OpenFOAM 在 PBiCG 失败时的官方 fallback（见 `PBiCG.C:228-238` 的 FatalError）。

pd 用 `PCG + DIC`（同文件 :19-37），**不需要复现 DILU 这条 path 来跑 pd**。

### 4.2 PBiCG 主循环伪码（按 PBiCG.C:69-253）

```
PBiCG::solve(psi, source, cmpt):
    # === 初始化 ===
    pA = wA = pT = wT = zeros(nCells)

    matrix.Amul(wA, psi)                        # wA = A psi
    rA = source - wA                            # initial residual
    normFactor = solver::normFactor(psi, source, wA, pA)
    initialResidual = sum(|rA|) / normFactor
    finalResidual = initialResidual

    if (initialResidual already converged AND minIter == 0) return

    matrix.Tmul(wT, psi)                        # wT = Aᵀ psi
    rT = source - wT                            # initial transpose residual
    wArT = 0                                    # placeholder

    DILU = preconditioner::New(...)             # one-time setup, cached in mutable autoPtr

    iter = 0
    do:
        wArTold = wArT

        DILU.precondition(wA, rA, cmpt)         # wA = M⁻¹ rA
        DILU.preconditionT(wT, rT, cmpt)        # wT = M⁻ᵀ rT

        wArT = sum(wA * rT)                     # Σ wA[i] rT[i]   (gSumProd)

        if (iter == 0):
            pA = wA                             # copy
            pT = wT                             # copy
        else:
            beta = wArT / wArTold
            pA = wA + beta * pA
            pT = wT + beta * pT

        matrix.Amul(wA, pA)                     # wA = A pA
        matrix.Tmul(wT, pT)                     # wT = Aᵀ pT

        wApT = sum(wA * pT)
        if (|wApT| / normFactor < SMALL): break (singular)

        alpha = wArT / wApT

        psi += alpha * pA
        rA  -= alpha * wA
        rT  -= alpha * wT

        finalResidual = sum(|rA|) / normFactor
        iter++
    while (iter < maxIter AND not converged) OR iter < minIter
```

### 4.3 跟 textbook BiCG (Saad Alg 7.3) 的差异

| 方面 | textbook | OpenFOAM PBiCG |
|------|----------|----------------|
| Inner products | `(r, r̃)` | `(M⁻¹r, r̃) = (wA, rT)` — preconditioned form |
| α 分母 | `(Ap, p̃)` | `(A pA, pT) = (wA, pT)` — 一致 |
| 初始 r̃ | 任选（常用 r̃ = r） | `r̃ = source - Aᵀ psi`（用 `Tmul` 算，跟 r 完全独立） |
| Stop | `‖r‖ < tol` | **L1 范数 / normFactor** —— 见 §4.4 |
| 收敛失败 | warn + return | `FatalError`（建议改 PBiCGStab） |

### 4.4 normFactor 与 residual 范数

**normFactor**（`lduMatrixSolver.C:235-274`）：
```
normFactor = Σ |A psi - A·avg(psi)| + Σ |source - A·avg(psi)| + SMALL
```
其中 `A·avg(psi)` 是 `sumA() * mean(psi)` —— 把 ψ 替换成它的平均值后矩阵作用的结果。这是个**问题特定的量级估计**，让 residual 在不同物理 case 间可比。

**residual = `gSumMag(rA, comm) / normFactor`** —— L1 范数，不是 L2，不是 ∞ 范数。

> **黄牌警告 #4**：仿写时 `tolerance=1e-12` 是相对于这个 normFactor 的。如果用 L2 范数判收敛会跟 OpenFOAM iter 数对不上。**复现时必须先复现 normFactor 与 L1 residual**，否则收敛准则不一致，iter 数无法 cross-check。

---

## §5 SpMV / Amul / Tmul 实现

### 5.1 Amul（`lduMatrixATmul.C:37-152`）—— face-loop 版（默认走这个）

```
Amul(Apsi, psi, ...):
    initMatrixInterfaces(...)   # 启动并行 send

    for cell = 0 .. nCells-1:
        Apsi[cell] = diag[cell] * psi[cell]    # diagonal contribution

    nFaces = upper().size()
    for face = 0 .. nFaces-1:
        Apsi[uPtr[face]] += lowerPtr[face] * psi[lPtr[face]]
        # = Apsi[neighbour] += A[neighbour, owner] * psi[owner]
        Apsi[lPtr[face]] += upperPtr[face] * psi[uPtr[face]]
        # = Apsi[owner]    += A[owner, neighbour] * psi[neighbour]

    updateMatrixInterfaces(...) # complete halo exchange
```

**两个非对角更新在同一 face-loop 内做完**。这是 LDU 格式的核心 SpMV 模式：face 是边，每条边对两端 cell 都贡献。**单进程下不需要 atomic**（owner 列表本身是分块连续递增的，不存在写冲突）。

### 5.2 Amul cell-based 分支（`lduMatrixATmul.C:76-121`，仅当 `hasLowerCSR()`）

```
for cell = 0 .. nCells-1:
    val = diag[cell] * psi[cell]
    # 累加 cell 作为 neighbour 的 face（lower）
    for i in [losortStart[cell], losortStart[cell+1]):
        val += lowerCSR[i] * psi[lcsrPtr[i]]
    # 累加 cell 作为 owner 的 face（upper）
    for i in [ownerStart[cell], ownerStart[cell+1]):
        val += upper[i] * psi[uPtr[i]]
    Apsi[cell] = val
```

这是 CSR-flavor SpMV，每行内积写一次 `Apsi[cell]`，更友好 OpenMP。**但默认不用，要 lazy 触发 `lowerCSR()`**。复现时建议两个版本都做：face-loop 版用来对 bit 行为，cell-loop 版用来 bench 多核。

### 5.3 Tmul（`lduMatrixATmul.C:155-216`）

跟 Amul 几乎一样，**唯一区别在 face-loop 里 lower / upper 角色对调**：
```
for face = 0 .. nFaces-1:
    Tpsi[uPtr[face]] += upperPtr[face] * psi[lPtr[face]]   # 注意 upper
    Tpsi[lPtr[face]] += lowerPtr[face] * psi[uPtr[face]]   # 注意 lower
```

数学上 `Aᵀ` 的 (i, j) 元素就是 `A` 的 (j, i) 元素，对应 LDU 里面 lower ↔ upper 互换。

### 5.4 sumA、residual、H1（同文件下半段）

- `sumA` (`:219`): 算 `sumA[c] = Σ_j A[c, j]` —— normFactor 用
- `residual` (`:268`): 一次性算 `r = source - A psi`，比 Amul + 减法快一倍（少一次写读 wA）
- `H1` (`:358`): 算 `-Σ off-diag`（GAMG / pressure 用）

**复现 PBiCG-DILU 只需要 Amul、Tmul、sumA**。residual/H1 可选。

---

## §6 并行化策略（仅做 single-process / OpenMP，跳过 MPI）

OpenFOAM 多核走 **MPI domain decomposition + halo exchange**：
- `decomposePar` 把 mesh 切成 N 块
- 每个 MPI rank 拥有一份本地 `lduMatrix`，本地 cell 1..N_local
- 跨 rank 的 face 变成 `processor` patch，存在 `interfaceBouCoeffs` 里
- `Amul` 的 `initMatrixInterfaces` 发 non-blocking send，`updateMatrixInterfaces` 在本地 face-loop 完后等 recv 收完，把邻居 patch 贡献加进 Apsi
- DILU 本身**完全 local**：每个 rank 独立做自己的 calcReciprocalD + sweep，**不跨 rank 交换**（这就是为什么 N=32 还能并行 12.86× —— 主要靠 SpMV 的 cell 分摊，DILU 部分被 hidden in 局部）

> **本次复现范围明确**：单进程 / 共享内存。**只需要复现 §3-§5**。MPI / interface 整段跳过。OpenMP 只能加在 §5.2 cell-based Amul，**不能加 §3.2-§3.4 DILU sweep**（顺序依赖）。如果要 OpenMP DILU 必须换 multicolor 算法（已有 `dilu/multicolor/` 在做这件事）。

---

## §7 LaserbeamFoam fvSolution 配置（已在 §4.1 列出，此处补 pd）

`/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/system/fvSolution`:

| 方程 | line | solver | preconditioner | tol | relTol | minIter |
|------|------|--------|---------------|-----|--------|---------|
| pd | 19-24 | PCG | DIC | 1e-8 | 0 | — |
| pdFinal | 38-44 | PCG | DIC | 1e-8 | 0 | — |
| pcorr | 85-91 | PCG | DIC | 1e-10 | 0 | — (本 case 不触发) |
| **T** | **94-101** | **PBiCG** | **DILU** | **1e-12** | **0** | **1** |
| TFinal | 102-105 | (alias of T) | | | | |
| U | 135-141 | PBiCGStab | DILU | 1e-8 | 0 | — |

> **本次复现 target = T 一行**：`PBiCG + DILU + tol=1e-12 + minIter=1`。

---

## §8 复现路径建议（unbreakable 不变量列表）

不写代码，只列**仿写时绝对不能动**的契约：

1. **LDU 格式必须保留** —— 不要换成 CSR。原因：dumper 输出已经是 (owner, neighbour, lower, upper, diag) 五段，CSR 化会丢失 face index 顺序，而**face index 顺序是 DILU 收敛性的隐藏不变量**（§3.1 黄牌警告 #1）。
2. **face 必须按 (owner, neighbour) 字典序存储** —— dump 时已经是这个序，不要 reshuffle。
3. **calcReciprocalD 必须单线程一遍** —— OpenMP 加在这里 = 答案错。要并行必须改算法（multicolor 或 level scheduling）。
4. **forward sweep 用 face 升序，backward 用 face 降序** —— 不是 cell 升/降。
5. **preconditionT 必须用 losortAddr** —— losort 要按 §2.3 的 `calcLosort` 算法构造（按 neighbour 排序的 face index），不能省略。
6. **tolerance 是相对 normFactor 的 L1 范数** —— normFactor 必须按 §4.4 公式实现，否则 iter 数对不上。
7. **DILU 不修改 L 和 U** —— `precondition()` 里用的还是原 lower / upper（`DILUPreconditioner.C:128, 134`），只有 D 被改成 rD。
8. **rD 存的是 1/D̃ 而不是 D̃** —— 所有 sweep 是乘 rD 不是除 D，这是性能优化（CPU 除法 ~20 cycle，乘法 ~5 cycle）。
9. **PBiCG 的 r̃ 必须独立维护** —— 不要跟 r 共用 buffer。`pT, wT, rT` 都是独立长度 nCells 数组。
10. **β 的初次计算被 special-case** —— `iter == 0` 时直接 copy wA → pA，**不算** `wArT/wArTold`（除以 0 会爆）。

复现的最小可验证目标：**对单个 dump 出来的 (A, b, x⁰)，用 Python/NumPy 跑 PBiCG-DILU，**iter 数应等于 OpenFOAM log 中该矩阵的 iter 数**（同 tol、同 normFactor 下），最终 ‖x_ours - x_OF‖∞ / ‖x_OF‖∞ < 1e-10**。

---

## §9 待办（TODO）

本次审计**没看**的部分：
- `lduMatrixUpdateMatrixInterfaces.C` 全部 344 行（处理并行 patch / coupled BC，单进程不需要）
- `PBiCGStab.C` 主循环细节（仅扫了 Amul/precondition 调用点，没核每一步公式）
- `lduMatrixOperations.C`（`+=`、`-=`、`scalarMultiply` 等运算符重载）

### §9.1 18 个 Gap 状态

下表对应 §10 的逐项深度补充。**RESOLVED** = 已读到具体源码可直接复现；**PARTIAL** = 核心已答但留小尾巴；**UNRESOLVED** = 需要用户判定。

| Gap | 主题 | 状态 | 备注 |
|-----|------|------|------|
| 1 | lower/upper ↔ A[i,j] | RESOLVED | §10.1，由 Amul 行号反推 |
| 2 | calcLosort 算法 | RESOLVED | §10.2，counting-sort 实现 |
| 3 | normFactor 完整公式 | RESOLVED | §10.3，4 行源码 + small_=1e-20 |
| 4 | checkConvergence | RESOLVED | §10.4，relTol 公式 + minIter 时机 |
| 5 | relax + dump 顺序 | RESOLVED | §10.5，T 不在 relaxationFactors → relax() no-op |
| 6 | BC fold | RESOLVED | §10.6，dump 用 D() 已 fold；T 全 zeroGradient → 实际 0 fold |
| 7 | initial guess psi 来源 | RESOLVED | §10.7，psi = T.primitiveField() at solve entry |
| 8 | pA pT rA rT 初始化 | RESOLVED | §10.8，pA/wA 未初始化但被 Amul/copy 覆写 |
| 9 | singular 阈值 | RESOLVED | §10.9，VSMALL=1e-300 |
| 10 | dump 是否需仿写补 fold | RESOLVED | §10.10，**不需要**额外 fold |
| 11 | psi.correctBoundaryConditions | RESOLVED | §10.11，由 fvMatrix 在 solve 返回后调用，PBiCG 内部不调 |
| 12 | preconditioner 生命期 | RESOLVED | §10.12，每次 PBiCG.solve() 起点 New 一次 |
| 13 | sumA 实现 | RESOLVED | §10.13，diag + face 累加 lower/upper |
| 14 | DILU ctor 是否动 interface | RESOLVED | §10.14，calcReciprocalD 完全不碰 interface |
| 15 | rD=0 防护 | RESOLVED | §10.15，**OpenFOAM 无防护**，直接 1.0/rD |
| 16 | dump face 顺序保真 | RESOLVED | §10.16，dump 是 CSR 但 (owner,nei) 升序可从 CSR 重建 |
| 17 | gSumProd 归约顺序 | RESOLVED | §10.17，朴素 sum，无 Kahan，单进程 = numpy.dot |
| 18 | minIter 路径 | RESOLVED | §10.18，外层 if 双门控 + do-while 末尾 OR |

如果将来需要在仿写里加 PBiCGStab fallback 或跨进程 MPI，需要补审 `lduMatrixUpdateMatrixInterfaces.C` 与 `PBiCGStab.C`。

---

## §10 深度补充答案（Gap 1-18）

每条结论都给出 file:line 与 5-15 行源码引用。所有结论可直接 verify。

### §10.1 Gap 1 — lower / upper 在数学矩阵 A 的精确位置

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduAddressing/lduAddressing.H:39-66`（owner 升序约定）
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C:55-56,180-200`（Amul / Tmul 行号）

**关键源码 — Amul face-loop（lduMatrixATmul.C:55-56 + 实际累加段）**：
```cpp
const label* const __restrict__ uPtr = addr.upperAddr().begin();   // = neighbour
const label* const __restrict__ lPtr = addr.lowerAddr().begin();   // = owner
...
ApsiPtr[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];   // row=neighbour, col=owner
ApsiPtr[lPtr[face]] += upperPtr[face]*psiPtr[uPtr[face]];   // row=owner,    col=neighbour
```

**关键源码 — Tmul face-loop（lduMatrixATmul.C:198-201）**：
```cpp
TpsiPtr[uPtr[face]] += upperPtr[face]*psiPtr[lPtr[face]];   // (Aᵀ)[nei,own] = upper[f]
TpsiPtr[lPtr[face]] += lowerPtr[face]*psiPtr[uPtr[face]];   // (Aᵀ)[own,nei] = lower[f]
```

**结论 — 对于 face f 连接 owner=o, neighbour=n（保证 o < n，由 lduAddressing.H:39-66 的 owner-升序约定）**：

| 矩阵元素 | LDU 取值 |
|----------|----------|
| `A[o, n]` | `upper[f]` （行=o=owner, 列=n=neighbour，严格上三角） |
| `A[n, o]` | `lower[f]` （行=n=neighbour, 列=o=owner，严格下三角） |
| `A[i, i]` | `diag[i]` |

**对称矩阵**（pd 用 DIC）：`lower == upper` 是数学事实，OpenFOAM 实际只存 upper，`lower()` 在 symmetric 矩阵上返回 upper 的 alias（`lduMatrix.H:97,100,716-723` 中 `symmetric()` 测试 `diagPtr_ && !lowerPtr_ && upperPtr_`）。

**非对称**（T 用 DILU）：`lower != upper`，分别存。

**例子**：face f=5，owner=3, neighbour=7：
- `lower[5] = A[7, 3]`
- `upper[5] = A[3, 7]`
- `Aᵀ[3, 7] = A[7, 3] = lower[5]`，`Aᵀ[7, 3] = A[3, 7] = upper[5]`，所以 Tmul 把 lower / upper 角色对调。

> **要点**：`lowerAddr() ≡ owner`、`upperAddr() ≡ neighbour` —— OpenFOAM 命名极易误读。**owner = lower-cell-index**（行号小），**neighbour = upper-cell-index**（行号大），lower/upper 系数名指 LDU 三角划分。

---

### §10.2 Gap 2 — calcLosort 完整算法

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduAddressing/lduAddressing.C:34-91`（91 行 lazy build）

**完整源码**：
```cpp
void Foam::lduAddressing::calcLosort() const
{
    // Scan the neighbour list to find out how many times the cell
    // appears as a neighbour of the face. Done this way to avoid guessing
    // and resizing list
    labelList nNbrOfFace(size(), Foam::zero{});
    const labelUList& nbr = upperAddr();
    forAll(nbr, nbrI)
        nNbrOfFace[nbr[nbrI]]++;

    // Create temporary neighbour addressing
    labelListList cellNbrFaces(size());
    forAll(cellNbrFaces, celli)
        cellNbrFaces[celli].setSize(nNbrOfFace[celli]);

    nNbrOfFace = 0;

    // Scatter the neighbour faces
    forAll(nbr, nbrI)
    {
        cellNbrFaces[nbr[nbrI]][nNbrOfFace[nbr[nbrI]]] = nbrI;
        nNbrOfFace[nbr[nbrI]]++;
    }

    // Gather the neighbours into the losort array
    losortPtr_ = std::make_unique<labelList>(nbr.size(), -1);
    auto& lst = *losortPtr_;
    label lstI = 0;
    forAll(cellNbrFaces, celli)
    {
        const labelUList& curNbr = cellNbrFaces[celli];
        forAll(curNbr, curNbrI)
            lst[lstI++] = curNbr[curNbrI];
    }
}
```

**算法本质 — counting sort（不是 std::sort，不是 argsort）**：
1. 第一遍扫 `nbr[]` 数组数 `nNbrOfFace[c] = #{f : neighbour[f]==c}`。
2. 给每个 cell c 分配长度为 `nNbrOfFace[c]` 的子列表 `cellNbrFaces[c]`。
3. 第二遍扫 `nbr[]`，把 face-index 散到 `cellNbrFaces[neighbour[f]]` 末尾。
4. 按 cell 升序拼接所有 `cellNbrFaces[c]` 得 losort。

**重复 neighbour 怎么处理**：步骤 3 是 stable 的（按 face-index 升序追加同一 neighbour 的 face）。所以最终 losort 数组里：
- 第一级排序：`neighbour[losort[k]]` 升序
- 第二级排序：同一 neighbour 内部，`losort[k]` 自己（即原 face-index）升序

**4-cell 5-face 例子**（owner / neighbour 来自 OpenFOAM 约定的 (owner, neighbour) 升序）：
```
face   owner   neighbour
 0      0        1
 1      0        2
 2      1        2
 3      1        3
 4      2        3
```
- `nNbrOfFace = [0, 1, 2, 2]` (cell 0 不是任何 face 的 neighbour, cell 1 是 face 0 的 neighbour, ...)
- `cellNbrFaces = [[], [0], [1, 2], [3, 4]]`
- losort = `[0, 1, 2, 3, 4]`（按 neighbour=1,2,2,3,3 升序）—— **本例下 losort 与 face-index 顺序巧合一致**，但只要存在某个 face 的 neighbour 比前面 face 的 neighbour 小（cell 重号场景），losort 就会和原 face-index 不同。

> **复现**：仿写时用 `numpy.argsort(neighbour, kind='stable')` 即等价（counting sort 与 stable mergesort 在重复 key 下行为相同）。

---

### §10.3 Gap 3 — normFactor 完整精确公式

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C:235-274`

**逐行源码（全文，关键 24 行）**：
```cpp
Foam::solveScalarField::cmptType Foam::lduMatrix::solver::normFactor
(
    const solveScalarField& psi,
    const solveScalarField& source,
    const solveScalarField& Apsi,
    solveScalarField& tmpField,
    const lduMatrix::normTypes normType
) const
{
    switch (normType)
    {
        case lduMatrix::normTypes::NO_NORM :
            break;

        case lduMatrix::normTypes::DEFAULT_NORM :
        case lduMatrix::normTypes::L1_SCALED_NORM :
        {
            // --- Calculate A dot reference value of psi
            matrix_.sumA(tmpField, interfaceBouCoeffs_, interfaces_);
            tmpField *= gAverage(psi, matrix_.mesh().comm());

            return
                gSum
                (
                    (mag(Apsi - tmpField) + mag(source - tmpField))(),
                    matrix_.mesh().comm()
                ) + solverPerformance::small_;
        }
    }
    return solveScalarField::cmptType(1);
}
```

**精确公式**：
$$
\text{normFactor} = \sum_i \big| (A\psi)_i - s_i \cdot \bar\psi \big| + \sum_i \big| b_i - s_i \cdot \bar\psi \big| + \text{small\_}
$$
其中 $s_i = (A \cdot \mathbf{1})_i = \sum_j A_{ij}$ 由 `sumA()` 算，$\bar\psi = \text{gAverage}(\psi)$ 是全场平均。**这是单次 reduction 内的两个 mag-累加之和**（第 257-263 行），不是分别 `gSumMag(Apsi - tmpField) + gSumMag(source - tmpField)`。在数学上等价（mag 是元素-wise，gSum 是线性），但实现上是同一个 `gSum` 调用接受 `(mag(Apsi - tmpField) + mag(source - tmpField))()` 这个表达式。

**`small_` 精确值**：
- `lduMatrixSolver.C:264` 写的 `solverPerformance::small_`
- 定义在 `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.H:292`：
  ```cpp
  template<>
  const scalar solverPerformance##Type::small_(1e-20);
  ```
- **`small_ = 1e-20`**（不是 `SMALL=1e-15`，不是 `VSMALL=1e-300`）。

**默认 normType**：`lduMatrix.H:114-119` 定义 enum，`DEFAULT_NORM == L1_SCALED_NORM`，PBiCG.C:112 调 4-arg overload 走 `normType_`，缺省即 L1_SCALED。

**算几次**：每次 PBiCG.solve() 调用算 **1 次**（PBiCG.C:112，紧跟在 Amul 后），后续所有 `solverPerf.finalResidual = gSumMag(rA)/normFactor` 都共用这一个值（PBiCG.C:121-122,215-217）。

> **复现 normFactor 的 numpy 表达式**（单进程，$A,\psi,b$ 都是 numpy）：
> ```python
> sumA = A.sum(axis=1)                 # = matrix.sumA() 单进程下
> psi_avg = psi.mean()
> tmp = sumA * psi_avg
> normFactor = np.sum(np.abs(A @ psi - tmp) + np.abs(b - tmp)) + 1e-20
> ```

---

### §10.4 Gap 4 — checkConvergence 完整公式

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.C:62-88`

**完整源码**：
```cpp
template<class Type>
bool Foam::SolverPerformance<Type>::checkConvergence
(
    const Type& Tolerance,
    const Type& RelTolerance,
    const int logLevel
)
{
    if ((logLevel >= 2) || (debug >= 2))
    {
        Info<< solverName_
            << ":  Iteration " << nIterations_
            << " residual = " << finalResidual_
            << endl;
    }

    converged_ =
    (
        finalResidual_ < Tolerance
     || (
            RelTolerance > small_*pTraits<Type>::one
         && finalResidual_ < cmptMultiply(RelTolerance, initialResidual_)
        )
    );

    return converged_;
}
```

**精确公式**：
$$
\text{converged} \;=\; \big( r_{\text{final}} < \text{tol} \big) \;\vee\; \big( \text{relTol} > 10^{-20} \;\wedge\; r_{\text{final}} < \text{relTol} \cdot r_{\text{initial}} \big)
$$

**relTol = 0 路径**：`relTol > small_` 短路 false，整条 OR 退化为 `r_final < tol`。本 case (`reTol 0.0` in fvSolution:100，**注意原文件是 `reTol`，OpenFOAM 在不识别 key 时取默认 0**，结果与 `relTol 0.0` 一致) 即纯绝对收敛 `r_final < 1e-12`。

**minIter 时机**（PBiCG.C:159-225 的 do-while）：
```cpp
} while
(
    (
      ++solverPerf.nIterations() < maxIter_
   && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
 || solverPerf.nIterations() < minIter_
);
```
- iter 计数：`solverPerf.nIterations()` 从 0 起（默认 ctor `nIterations_(Zero)`），**前缀 `++` 在 while 条件求值时累加**，所以第一次 iter 体执行完时 `nIterations()` = 1。
- minIter=1 + initialResidual 已收敛：进入 `if (minIter_ > 0 || !converged)` 分支（PBiCG.C:126-130）→ 进 do-while → 跑完一次 iter 后 `nIterations() = 1`，然后 `nIterations() < maxIter_ && !converged` 通常 true（因为 finalResidual 又算了一遍）会继续；但如果 `finalResidual < tol`，`!converged = false`，**且** `nIterations() (=1) < minIter_ (=1)` 也 false → 退出。
- minIter 检查在 `++` 后，且短路 OR 末项；`converged` 为 true 但 `nIterations() < minIter_` 也会**强制再跑一轮**。
- **iter 的语义就是"循环体已执行的次数"**。

> **本 case 实测语义**（minIter=1, tol=1e-12）：至少跑 1 个 iter；之后只在 finalResidual < 1e-12 时停。

---

### §10.5 Gap 5 — relax + dump 顺序判决

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:1101-1245`（`relax(alpha)` 实现）
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:1248-1262`（无参 `relax()` 查 dict）
- `/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/system/fvSolution:233-244`（relaxationFactors）
- `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy/TEqn.H.patched:47,50`（调用顺序）

**`relax(alpha)` 公式（fvMatrix.C:1111-1244 关键段）**：
```cpp
Field<Type>& S = source();
scalarField& D = diag();
scalarField D0(D);                       // 备份原 diag

scalarField sumOff(D.size(), Zero);
sumMagOffDiag(sumOff);                   // sumOff[c] = Σ_j |A[c,j]| (j≠c)

// (boundary internal/coupled coeffs 临时 fold 进 D 用于 dominance test)

forAll(D, celli)
    D[celli] = max(mag(D[celli]), sumOff[celli]);    // diag dominance enforcement

D /= alpha;                              // ← 真·relax 步：D' = max(|D|, sumOff)/α

// (再把 boundary coeffs 从 D 中扣回去)

S += (D - D0)*psi_.primitiveField();     // 把 diag 增量 fold 进 source
```
**净效果**：`diag` 被强制 ≥ `sumOff`，再除以 α；`source` 加 `(D_new - D_old) * psi_old`。

**无参 `relax()`**（fvMatrix.C:1248-1262）：
```cpp
scalar relaxCoeff = 0;
if (psi_.mesh().relaxEquation(name, relaxCoeff))
    relax(relaxCoeff);
// else: NO-OP
```
查 `system/fvSolution.relaxationFactors.equations` 找 key，找不到就**啥也不做**。

**T 是否在 relaxationFactors**（fvSolution:233-244）：
```
relaxationFactors
{
    fields    { pd 1.0; }
    equations { U  0.99; }
}
```
**`T` 既不在 `fields` 也不在 `equations` → `relaxEquation("T", ...)` 返回 false → `TEqn.relax()` 是 NO-OP**。

**dump 时机**（TEqn.H.patched:47-50）：
```cpp
TEqn.relax();                                            // line 47 — NO-OP for T
matrixDumper_.dumpPreSolve(TEqn, "T", T);                // line 50 — dump
auto _T_t0 = std::chrono::steady_clock::now();
Tp = TEqn.solve();                                       // line 55
```
顺序是 `relax → dump → solve`。**对 T**：因为 relax no-op，dump 拿到的就是 fvm::ddt+div+laplacian 装配出来的原始矩阵（再算上 dump 内部 `D()` 把 boundary internalCoeffs fold 进 diag、源码内手动 fold boundaryCoeffs 进 source）。

**对 pd（pEqn.H.patched:124-127）**：`pdEqn.relax();` 也 no-op（pd 是 fields 不是 equations，`fields` block 走 `psi_.relax()` 修正 psi 而非 matrix），所以 pd dump 也是原始矩阵。

> **判决**：本 case 的 dump = 装配后矩阵（diag 已含 boundary internalCoeffs，source 已含 boundaryCoeffs，未做 relaxation 修改）。

---

### §10.6 Gap 6 — BC fold 进 diag/source

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:122-147`（`addBoundaryDiag`）
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:174-220`（`addBoundarySource`）
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:147,154,168,238`（solver 入口的 fold 时机）
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C:1280-1285`（公开的 `D()` 接口）
- `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy/matrixDumper.H:439,447-468`（dump 用 D() + 手动 fold source）

**fold 机制 — addBoundaryDiag（fvMatrix.C:138-143）**：
```cpp
addToInternalField
(
    lduAddr().patchAddr(patchi),
    internalCoeffs_[patchi].component(solveCmpt),
    diag
);
```
等价于 `for face in patchAddr: diag[face_cell] += internalCoeffs[face]`。

**fold 机制 — addBoundarySource（fvMatrix.C:193-203, 非 coupled 分支）**：
```cpp
const Field<Type>& pbc = boundaryCoeffs_[patchi];
if (!ptf.coupled())
{
    addToInternalField(lduAddr().patchAddr(patchi), pbc, source);
}
```
即 `source[face_cell] += boundaryCoeffs[face]`。

**solver 入口的实际 fold 顺序（fvMatrixSolve.C:147-168）**：
```cpp
scalarField saveDiag(diag());        // 备份
Field<Type> source(source_);          // 拷贝
addBoundarySource(source);            // line 154: source += boundaryCoeffs (非 coupled)
... per cmpt:
    addBoundaryDiag(diag(), cmpt);    // line 168: diag += internalCoeffs
    ...
    solverPerf = solver::New(...)->solve(psiCmpt, sourceCmpt, cmpt);
    diag() = saveDiag;                // line 238: 还原 diag
```
**solver 拿到的是 fold 后的 diag 和 fold 后的 source**。

**dumper 拿什么（matrixDumper.H:439）**：
```cpp
scalarField diagFull(matrix.D()());   // ← D() = diag + cmptAv(boundary internalCoeffs)
```
而 `D()`（fvMatrix.C:1280-1285）：
```cpp
auto tdiag = tmp<scalarField>::New(diag());
addCmptAvBoundaryDiag(tdiag.ref());   // 对 scalar 等价 addBoundaryDiag(_, 0)
return tdiag;
```
**dumper 的 diagFull = solver 看到的 diag**（对 scalar 矩阵，cmptAv == component(0)）。

**dumper 的 source（matrixDumper.H:447-468）**：
```cpp
scalarField sourceFull(matrix.source());
const auto& bCoeffs = matrix.boundaryCoeffs();
forAll(psiField.boundaryField(), patchi)
{
    if (psi.boundaryField()[patchi].coupled()) continue;  // 单进程 OK
    forAll(pa, facei)
        sourceFull[pa[facei]] += pbc[facei];               // 手动 addBoundarySource(non-coupled)
}
```
**dumper 的 sourceFull = solver 看到的 source**（在没有 coupled patch 的单进程场景下逐字节相同）。

**T case BC（`/home/yzk/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_crosscheck/0/T`）**：所有 patch 都是 `zeroGradient`。zeroGradient 的 `internalCoeffs == 0` 且 `boundaryCoeffs == 0`（fvPatchField 默认行为）。所以**对本 case 的 T**：fold 进 diag = 0，fold 进 source = 0 — dumper 与 solver 看到的就是 raw `matrix.diag()` 与 `matrix.source()`。

> **结论**：dump 出来的 A.mm 已经把 internalCoeffs fold 进 diag 了。仿写读 A.mm 后**不需要**再做任何 BC fold。

---

### §10.7 Gap 7 — 初始 guess psi 来源

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:167,237`
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C:69-77`
- `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy/TEqn.H.patched:50`

**fvMatrixSolve.C:167（每个 cmpt 进入 solver 前）**：
```cpp
scalarField psiCmpt(psi.primitiveField().component(cmpt));   // copy of T's internal field
```
然后 line 227：`->solve(psiCmpt, sourceCmpt, cmpt)`，传给 lduMatrix::solver 的是 `psiCmpt` reference。Line 237 `psi.primitiveFieldRef().replace(cmpt, psiCmpt);` —— 解完写回 T。

**PBiCG.C:69-77**：
```cpp
Foam::solverPerformance Foam::PBiCG::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    solveScalarField& psi = tpsi.ref();
```
psi 是引用，会被 PBiCG 直接修改（line 210 `psiPtr[cell] += alpha*pAPtr[cell]`）。

**LaserbeamFoam corrector 内**（TEqn.H.patched:20-55）：每个 nTCorrector 内 `T.storePrevIter()` → 装配新 TEqn → `TEqn.relax()`（no-op for T）→ `dumpPreSolve(TEqn, "T", T)` → `TEqn.solve()`。
- 第 1 个 corrector：psi = T 的 prev-time-step 收敛值
- 第 2+ 个 corrector：psi = 上一个 corrector 解出的 T

**dump 的 x0**（matrixDumper.H:501）：
```cpp
writeMMArray_(dir / "x0.mm", x0.primitiveField(), "initial guess (psi at solve entry)");
```
x0 形参就是 TEqn.H.patched:50 传入的 `T`。**dump 的 x0 = solver 进入时 PBiCG.psi 的 byte-for-byte 副本**。

> **复现**：仿写 PBiCG 时把 `x0.mm` 当作 PBiCG.solve() 的初始 psi 输入，结果应与 OpenFOAM 一致。

---

### §10.8 Gap 8 — pA, pT, rA, rT, wA, wT 初始化

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C:88-146`

**关键源码**：
```cpp
// Line 90-94: pA 与 wA 未初始化 (Field<scalar>(nCells) ctor 不 zero-fill)
solveScalarField pA(nCells);
solveScalarField wA(nCells);

// Line 97: wA 被 Amul 第一时间覆写
matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

// Line 101: rA = source - wA  (PrecisionAdaptor + Field 算术)
solveScalarField rA(tsource() - wA);

// Line 132: pT 显式 zero-fill
solveScalarField pT(nCells, 0);

// Line 135: wT 不 zero-fill
solveScalarField wT(nCells);

// Line 139: wT 被 Tmul 覆写
matrix_.Tmul(wT, psi, interfaceIntCoeffs_, interfaces_, cmpt);

// Line 142: rT = source - wT
solveScalarField rT(tsource() - wT);

// Line 146: wArT = 0  (local solveScalar)
solveScalar wArT = 0;
```

**第一次 iter 的 special-case path（PBiCG.C:159-188）**：
```cpp
do
{
    const solveScalar wArTold = wArT;       // iter 0: wArTold = 0

    preconPtr_->precondition(wA, rA, cmpt);
    preconPtr_->preconditionT(wT, rT, cmpt);

    wArT = gSumProd(wA, rT, matrix().mesh().comm());

    if (solverPerf.nIterations() == 0)      // ← iter==0 special-case
    {
        for (label cell=0; cell<nCells; cell++)
        {
            pAPtr[cell] = wAPtr[cell];      // pA = wA
            pTPtr[cell] = wTPtr[cell];      // pT = wT
        }
    }
    else
    {
        const solveScalar beta = wArT/wArTold;
        for (label cell=0; cell<nCells; cell++)
        {
            pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
            pTPtr[cell] = wTPtr[cell] + beta*pTPtr[cell];
        }
    }
```

**结论**：
- `pA, wA, wT` 不需要任何初始化（被覆写）。`pT` 显式置 0（虽然第一 iter 也覆写）。
- 第一 iter `wArTold = 0` 不会被除（special-case 直接 copy wA→pA）。
- `wArT` 是局部 solveScalar，不是 mutable member。每次 PBiCG.solve() 重新进入都从 0 开始。
- `nIterations()` 是 `solverPerformance` 成员（PBiCG.C:79-84 默认构造 → 0）。

---

### §10.9 Gap 9 — singular 阈值

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C:195-201` + `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/LduMatrix/LduMatrix/SolverPerformance.C:34-47`

**PBiCG 调用点（PBiCG.C:195-201）**：
```cpp
const solveScalar wApT = gSumProd(wA, pT, matrix().mesh().comm());

// --- Test for singularity
if (solverPerf.checkSingularity(mag(wApT)/normFactor))
{
    break;
}
```

**checkSingularity 实现（SolverPerformance.C:34-47）**：
```cpp
template<class Type>
bool Foam::SolverPerformance<Type>::checkSingularity
(
    const Type& wApA
)
{
    for(direction cmpt=0; cmpt<pTraits<Type>::nComponents; cmpt++)
    {
        singular_[cmpt] =
            component(wApA, cmpt) < vsmall_;
    }
    return singular();
}
```

**`vsmall_` 定义（SolverPerformance.H:295）**：`vsmall_(VSMALL)` → `VSMALL = doubleScalarVSMALL = 1.0e-300`（doubleScalar.H:64）。

**结论**：
- 阈值是 **`VSMALL = 1e-300`**（不是 SMALL=1e-15、不是 small_=1e-20）。
- 公式：`mag(wApT) / normFactor < 1e-300` → singular，break。
- 仿写：`if abs(wApT) / normFactor < 1e-300: break`。

---

### §10.10 Gap 10 — 仿写读 npz 后的 fold 判决

**联合 §10.5、§10.6、§10.7 的事实**：
1. dumper 用 `matrix.D()` 写出 diag → solver 看到的 diag 一致
2. dumper 手动 fold boundaryCoeffs 写 source → solver 看到的 source 一致（单进程，无 coupled）
3. dumper x0 = T.primitiveField() at solve entry = PBiCG psi 入参一致
4. T 不在 relaxationFactors → relax no-op → dump 与 solve 都用同一个未 relax 矩阵

**判决**：
> **仿写代码读完 A.mm/b.mm/x0.mm 后，无需任何额外 fold**。直接拿 (A, b, x0) 喂 PBiCG-DILU 即可。等价对 A、b、x0 三者各做了一次 byte-for-byte 复制。

**但有一个隐含约束**：`A` 在 reader 里被读成 CSR（reader.py:154 `sio.mmread(...).tocsr()`）。仿写 DILU 必须从 CSR 重建 `(diag, lower, upper, owner, neighbour)` 五元组（见 §10.16）。

---

### §10.11 Gap 11 — psi.correctBoundaryConditions 时机

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C` 全文 grep `correctBoundary` → **0 hit**
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:241`

**PBiCG.C 全文不调用 correctBoundaryConditions**。它操作的是 `scalarField& psi_s`（line 71），即 `psiCmpt` —— 一个普通 scalarField（非 GeometricField），**没有 boundary 概念**。

**fvMatrixSolve.C:237-241**（solve 完后）：
```cpp
psi.primitiveFieldRef().replace(cmpt, psiCmpt);
diag() = saveDiag;
}

psi.correctBoundaryConditions();   // ← line 241，所有 cmpt 解完后调
```
`psi` 是 GeometricField (volScalarField)，`correctBoundaryConditions()` 在所有 cmpt 都解完后调一次，让边界 patch 重新算（zeroGradient 复制内 cell 值，fixedValue 不变，等等）。

**复现影响**：
- 仿写只 update internalField 即可 match OpenFOAM 的 PBiCG 输出（即 x_final.mm 的内部 cell 部分）。
- x_final.mm 写的是 `xFinal.primitiveField()`（matrixDumper.H:527），即 internalField，**不含 boundary**。
- 所以仿写 PBiCG 单进程正确性验证就是 numpy 数组对比，**完全不需要实现 correctBoundaryConditions**。

---

### §10.12 Gap 12 — preconditioner 生命期

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.H:60`
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C:148-156`
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C:46-59`
- `/usr/lib/openfoam/openfoam2506/src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:219-227`

**PBiCG.H:60**（cache 声明）：
```cpp
mutable autoPtr<lduMatrix::preconditioner> preconPtr_;
```

**PBiCG.C:148-156**（一次性 setup）：
```cpp
// --- Select and construct the preconditioner
if (!preconPtr_)
{
    preconPtr_ = lduMatrix::preconditioner::New
    (
        *this,
        controlDict_
    );
}
```

**DILUPreconditioner.C ctor（46-59）**：
```cpp
Foam::DILUPreconditioner::DILUPreconditioner
(
    const lduMatrix::solver& sol,
    const dictionary&
)
:
    lduMatrix::preconditioner(sol),
    rD_(sol.matrix().diag().size())
{
    const scalarField& diag = sol.matrix().diag();
    std::copy(diag.begin(), diag.end(), rD_.begin());

    calcReciprocalD(rD_, sol.matrix());
}
```

**fvMatrixSolve.C:219-227**（每次 fvMatrix.solve() 都 New PBiCG）：
```cpp
solverPerf = lduMatrix::solver::New
(
    psi.name() + pTraits<Type>::componentNames[cmpt],
    *this,
    bouCoeffsCmpt,
    intCoeffsCmpt,
    interfaces,
    solverControls
)->solve(psiCmpt, sourceCmpt, cmpt);
```
`solver::New` 返回 `autoPtr<lduMatrix::solver>` —— **临时对象**，在 `->solve()` 调用结束后立即析构。

**生命期总结**：
- 每次 `TEqn.solve()` → 一个新 PBiCG 对象 → preconPtr_ 一开始为空 → if-block 触发 New → DILUPreconditioner ctor 调一次 → calcReciprocalD 跑一次。
- 同一 PBiCG.solve() 内 do-while 多次 iter，**rD 复用**（precondition / preconditionT 都读 `rD_` 不重算）。
- 跨 PBiCG.solve() 调用（不同 corrector / 不同 timestep），**PBiCG 对象销毁，rD 重算**。
- 没有跨时间步缓存机制。

> **复现**：每次仿写 PBiCG 调用顶部跑一次 calcReciprocalD 即可。

---

### §10.13 Gap 13 — sumA 实现

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixATmul.C:219-265`

**完整源码**：
```cpp
void Foam::lduMatrix::sumA
(
    solveScalarField& sumA,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
) const
{
    solveScalar* __restrict__ sumAPtr = sumA.begin();
    const scalar* __restrict__ diagPtr  = diag().begin();
    const label* __restrict__ uPtr = lduAddr().upperAddr().begin();
    const label* __restrict__ lPtr = lduAddr().lowerAddr().begin();
    const scalar* __restrict__ lowerPtr = lower().begin();
    const scalar* __restrict__ upperPtr = upper().begin();

    const label nCells = diag().size();
    const label nFaces = upper().size();

    for (label cell=0; cell<nCells; cell++)
        sumAPtr[cell] = diagPtr[cell];

    for (label face=0; face<nFaces; face++)
    {
        sumAPtr[uPtr[face]] += lowerPtr[face];   // sumA[neighbour] += lower (= A[nei,own])
        sumAPtr[lPtr[face]] += upperPtr[face];   // sumA[owner]     += upper (= A[own,nei])
    }

    // Add the interface internal coefficients to diagonal
    // and the interface boundary coefficients to the sum-off-diagonal
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            const labelUList& pa = lduAddr().patchAddr(patchi);
            const scalarField& pCoeffs = interfaceBouCoeffs[patchi];
            forAll(pa, face)
                sumAPtr[pa[face]] -= pCoeffs[face];
        }
    }
}
```

**单进程 + 无 coupled**：interface 段 0 触发，简化为：
```
sumA[c] = diag[c]
for f: sumA[neighbour[f]] += lower[f]
       sumA[owner[f]]     += upper[f]
```
即 **`sumA[i] = Σ_j A[i,j]`**（行和）。

**对称矩阵 vs 非对称**：
- 对称：lower == upper，face-loop 退化成 `sumA[neighbour] += upper; sumA[owner] += upper`（DICPreconditioner 也走 lower() alias upper）—— 数学上仍是行和。
- 非对称：lower 给 row=neighbour，upper 给 row=owner —— 仍是行和。
- **无差别公式：sumA = A · 1**。

**numpy 等价（单进程）**：`sumA = A.sum(axis=1)` 或更精确 `sumA = np.asarray(A @ np.ones(N))`。

**额外注意**：interfaceBouCoeffs 的符号是 `-=`，因为 OpenFOAM 内部 `interfaceBouCoeffs` 已经带反符号约定（fvMatrixSolve.C:147-153 处的注释解释）。本 case 单进程 + zeroGradient → interfaces 为空 → 不影响。

---

### §10.14 Gap 14 — DILU ctor 是否调 interface

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C` 全文 187 行

**全文 grep**：
```bash
grep -n "Interface\|interface\|coupled" DILUPreconditioner.C
```
**0 hit**。DILUPreconditioner.C 整个文件不引用任何 interface / coupled 相关 API。

**calcReciprocalD（DILUPreconditioner.C:64-92）**：只用 `diag()`、`upper()`、`lower()`、`upperAddr()`、`lowerAddr()` —— 全是本 rank 内部 lduMatrix 信息。

**precondition / preconditionT（95-184）**：同样不碰 interface。

**结论**：
- 单进程仿写 DILU **完全不需要**任何 interface 代码。
- 多进程 OpenFOAM 也不在 DILU 内做 halo 交换 —— DILU 是纯 local preconditioner。

---

### §10.15 Gap 15 — rD = 0 防护

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/DILUPreconditioner.C:78-91`

**关键 14 行**：
```cpp
label nFaces = matrix.upper().size();
for (label face=0; face<nFaces; face++)
{
    rDPtr[uPtr[face]] -= upperPtr[face]*lowerPtr[face]/rDPtr[lPtr[face]];
}

// Calculate the reciprocal of the preconditioned diagonal
const label nCells = rD.size();

for (label cell=0; cell<nCells; cell++)
{
    rDPtr[cell] = 1.0/rDPtr[cell];
}
```

**事实**：
- Line 81: `/rDPtr[lPtr[face]]` —— 直接除，无 if 检查、无 max(rD, eps)、无 abs 保护。
- Line 90: `1.0/rDPtr[cell]` —— 同样直接除。
- **OpenFOAM 没有任何 stabilizeDiag / floor 处理**。如果 rD[c] == 0 会得到 IEEE inf/nan，污染整个 sweep。

**实际触发概率**：
- LPBF FV 矩阵的 diag 来自 `fvm::ddt(rhoCp, T)` 的 `rhoCp/dt` 项 + `fvm::laplacian(kEff, T)` 的正系数累积 + 边界 internalCoeffs ≥ 0。**diag 恒正且远离 0**。
- DILU 的 schur 补 `rD[u] -= U*L/rD[l]` 在病态矩阵下能让 rD 变号或趋零，但 LaserbeamFoam T 矩阵实测稳定。

**仿写决策**：
- 严格按 OpenFOAM 复现：直接 `rD = 1.0 / rD`，不加保护。如果出 inf/nan 说明矩阵真坏，OpenFOAM 也会坏。
- 想加诊断可选 `assert (rD != 0).all()`，但**不要改算法**（加 max 会偏离 OpenFOAM 的迭代行为，iter 数对不上）。

---

### §10.16 Gap 16 — dump 的 face 顺序保真

**源文件**：
- `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/deploy/matrixDumper.H:471-496`（COO emit）
- `/home/yzk/DILU-Research/dilu/benchmark/openfoam_crosscheck/reader.py:152-154`（CSR load）

**dumper emit 顺序（matrixDumper.H:480-490）**：
```cpp
auto cooWriter = [&](std::ofstream& os) {
    for (label c = 0; c < N; ++c)
        os << (c+1) << ' ' << (c+1) << ' ' << diagFull[c] << '\n';
    for (label f = 0; f < nFaces; ++f)
    {
        const label own = lAddr[f] + 1;
        const label nei = uAddr[f] + 1;
        os << own << ' ' << nei << ' ' << upper[f] << '\n';   // (own, nei)
        os << nei << ' ' << own << ' ' << lower[f] << '\n';   // (nei, own)
    }
};
```
**dump 时**：face 按原 OpenFOAM index 顺序遍历（f=0..nFaces-1），每个 face 写两行 COO。但 COO 是无序集合 —— **face index 顺序在 .mm 文件里没显式保留**（只是行的物理顺序）。

**reader.py:154**：`A = sio.mmread(directory / "A.mm").tocsr()` —— scipy 的 `mmread` 返回 COO，`.tocsr()` 重排成 row-sorted。**face index 信息彻底丢失**。

**问题严重性评估**：
- DILU 的 face-loop 依赖 face 按 (owner, neighbour) 升序排列（lduAddressing.H:39-66 + §3.2 黄牌警告 #1）。
- 但**这个顺序是 OpenFOAM 的不变量约定**：所有 face 都满足 `owner[f] < neighbour[f]`，且 face 全局按 (owner, neighbour) 字典序。
- 从 CSR 矩阵 A 重建 face 顺序：
  ```python
  A_coo = A.tocoo()
  mask = A_coo.row < A_coo.col      # 上三角 = upper 段
  owner_arr     = A_coo.row[mask]
  neighbour_arr = A_coo.col[mask]
  upper_arr     = A_coo.data[mask]
  # 按 (owner, neighbour) 字典序 sort
  order = np.lexsort((neighbour_arr, owner_arr))
  owner = owner_arr[order]
  neighbour = neighbour_arr[order]
  upper = upper_arr[order]
  # lower 同理：
  mask_l = A_coo.row > A_coo.col
  # 注意 lower 的 (row, col) = (neighbour, owner)，对应同一个 face
  # 用 (col, row) 字典序匹配上面的 (owner, neighbour)
  ```
- 因为 OpenFOAM 的 face 排序就是 (owner, neighbour) 字典序，**重建结果 = OpenFOAM 原 face 顺序**。

**结论 — RESOLVED but 需要 reader 加重建步骤**：
- dumped CSR 不直接保留 face index，**但 OpenFOAM 的 (owner, neighbour) 字典序约定让重建 100% 确定**。
- 仿写读 npz 后第一步：从 CSR 重建 (owner, neighbour, lower, upper, diag) 五元组（用上面的 lexsort）。
- 重建后跑 DILU sweep 与 OpenFOAM 完全等价（face-loop 顺序一致）。

> **建议**：在 `reader.py` 旁加一个 helper `csr_to_ldu(A) -> (diag, lower, upper, owner, neighbour)`，封装上述 lexsort 逻辑。仿写 PBiCG-DILU 直接消费 LDU 五元组。

---

### §10.17 Gap 17 — gSumProd / gSumMag 归约顺序

**源文件**：
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/fields/Fields/Field/FieldFunctions.C:478-491`（`sumProd`）
- `/usr/lib/openfoam/openfoam2506/src/OpenFOAM/fields/Fields/Field/FieldFunctions.C:613,618-631`（`gSumMag`、`gSumProd`）

**sumProd 实现（FieldFunctions.C:479-491）**：
```cpp
template<class Type>
typename scalarProduct<Type, Type>::type
sumProd(const UList<Type>& f1, const UList<Type>& f2)
{
    typedef typename scalarProduct<Type, Type>::type resultType;

    resultType result = Zero;
    if (f1.size() && (f1.size() == f2.size()))
    {
        TFOR_ALL_S_OP_F_OP_F(resultType, result, +=, Type, f1, &&, Type, f2)
    }
    return result;
}
```
`TFOR_ALL_S_OP_F_OP_F` 展开为朴素 for-loop：`for i: result += f1[i] * f2[i]`。**无 Kahan，无 pairwise，纯左折叠**。

**gSumProd（FieldFunctions.C:618-631）**：
```cpp
template<class Type>
typename scalarProduct<Type, Type>::type gSumProd
(
    const UList<Type>& f1,
    const UList<Type>& f2,
    const label comm
)
{
    typedef typename scalarProduct<Type, Type>::type resultType;

    resultType result = sumProd(f1, f2);
    Foam::reduce(result, sumOp<resultType>(), UPstream::msgType(), comm);
    return result;
}
```
单进程下 `Foam::reduce` 是 no-op（MPI rank=0 时直接返回）。

**gSumMag**（FieldFunctions.C:613）：宏 `G_UNARY_FUNCTION(typename typeOfMag<Type>::type, gSumMag, sumMag, sum)`，展开后 `sumMag = Σ |f[i]|`，朴素左折叠。

**结论**：
- 单进程 `gSumProd(wA, rT)` ≡ `for i: result += wA[i] * rT[i]` 起始 0，左折叠。
- numpy 的 `np.dot(wA, rT)` 用 BLAS（`dgemv`/`ddot`），**不一定**用左折叠 —— 实际 BLAS 实现常用 4-way / 8-way SIMD pairwise 累加。
- **bit-wise 不一定 match**，但相对误差 O(N · ε_machine) ≈ 1e-13 for N=1e5（远小于 tol=1e-12 的 normFactor 数量级）。
- iter 数 cross-check **应当 match**，但最后一个 iter 的 finalResidual 可能差 1-2 ULP。

**仿写建议**：
- 严格 bit-match：用 `for i: result += a[i]*b[i]`（Python loop）—— 极慢但忠实。
- 实用 bit-match：用 `np.einsum('i,i->', a, b)`（更接近左折叠）。
- 性能优先 + 接受 O(N·eps) 误差：`np.dot(a, b)` —— iter 数对，最末位略差。

---

### §10.18 Gap 18 — minIter 完整路径

**源文件**：`/usr/lib/openfoam/openfoam2506/src/OpenFOAM/matrices/lduMatrix/solvers/PBiCG/PBiCG.C:125-225`

**外层 if-block（PBiCG.C:126-130）— 决定是否进入 do-while**：
```cpp
if
(
    minIter_ > 0
 || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
)
{
    // ... allocate pT, wT, rT, build preconditioner, do-while loop ...
}
```
- minIter > 0 强制进入循环（即使 initialResidual < tol）。
- LaserbeamFoam T case (minIter=1)：第一个判定即 true，**始终**会构建 pT/wT/rT 与 DILU preconditioner。

**do-while 末尾（PBiCG.C:218-225）**：
```cpp
} while
(
    (
      ++solverPerf.nIterations() < maxIter_
   && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
 || solverPerf.nIterations() < minIter_
);
```
**逐项分析**（按短路求值顺序）：
1. `++solverPerf.nIterations()` —— 先 +1（iter 1 → nIterations=1, ...）
2. `nIterations < maxIter` —— 还没到 maxIter
3. `!checkConvergence(...)` —— 当前 finalResidual 未收敛
4. AND 之上为 false（任一不满足，例如收敛了）
5. 此时若 `nIterations < minIter`，OR 为 true，**强制再跑一轮**

**初始 initialResidual 已 < tol，minIter=1 的执行路径**：
- 外层 if（126-130）：`minIter > 0` 短路 true → 进入。
- 进入后，构造 preconditioner（rD 计算），跑第 1 轮 do-while body。
- 末尾 ++nIterations → 1；checkConvergence 算新 finalResidual（基于 rA -= alpha*wA 后的）。
- **如果**新 finalResidual < tol：第一个 AND-clause = false；OR 末项 `1 < 1` = false → 退出。
- 总共跑 **1 轮 iter**。

**rD 计算时机**：进入 if-block 后，line 148-156 的 `if (!preconPtr_) preconPtr_ = ...New(...)` 调一次，DILU ctor → calcReciprocalD 跑一次。**即使 initialResidual 已收敛但 minIter=1，rD 仍然算**。

**maxIter 默认值**：lduMatrix.H:125 `defaultMaxIter = 1000`。fvSolution 没 explicit `maxIter` → `read()` 用默认。

> **结论**：本 case 路径下，DILU calcReciprocalD 至少执行 1 次/solve，PBiCG body 至少执行 1 次/solve。仿写时不要"快路径短路 initialResidual < tol 直接 return"。

