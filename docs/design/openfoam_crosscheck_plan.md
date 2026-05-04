# OpenFOAM ↔ Our Solvers 矩阵级对标计划

**文档状态**: Design — awaiting first implementation step
**作者**: Claude + yzk
**日期**: 2026-04-23
**对标目标案例**: `LaserbeamFoam/tutorials/laserMeltFoam/LPBF_tutorial`
**适用求解器**: OpenFOAM v2506 + LaserbeamFoam (fork 含 `laserMeltFoam`)

---

## 0. 一句话目标

从 LaserbeamFoam 的 LPBF tutorial 中提取真实线性系统 `Ax = b`（由 OpenFOAM 的 PIMPLE 求解器在 solve 入口冻结的系数矩阵），用我们自研的 **cuSPARSE DILU-PCG** 与 **AMGx**（classical / aggressive）求解同一组矩阵，在同一组 (A, b, x⁰) 上比较 **数值等价性** 与 **性能**。

---

## 1. 对标层级（Validation Tiers）

| 层级 | 定义 | 是否追求 | 原因 |
|------|------|----------|------|
| **L1** bit-identical | 逐位浮点相同 | ❌ | 不同求解器归约顺序、precond 实现、SpMV 调度均不同，FP 非结合性使 L1 物理不可能 |
| **L2** 数值等价 | `‖x_ours − x_ref‖∞ / ‖x_ref‖∞ < 1e-8` | ✅ **主目标** | 这才是"结果一样"在浮点世界里的真实含义 |
| **L3** 行为等价 | 都收敛；iter 数 / setup / solve 可比 | ✅ **次目标** | 这是性能对比本身 |

其中 "ref" 定义为：
- **小规模（nUnknowns < 1e5）**: `scipy.sparse.linalg.spsolve` 直解结果
- **大规模（nUnknowns ≥ 1e5）**: AMGx classical + tol=1e-14 的收敛解作为 pseudo-reference（即 AMGx 自己在超紧 tol 下的收敛解当参考，其余解法与之比较）

---

## 2. Scope

### 2.1 方程范围

LaserMeltFoam PIMPLE loop 实际存在 3 个线性系统入口（每 Δt 内）：

| 方程 | 求解器 | 预条件 | 对称性 | 本对标纳入？ |
|------|--------|--------|--------|--------------|
| **`pd`** (pressure dynamic) | PCG | DIC | **SPD** | ✅ 首要对标对象（能直接喂 cuSPARSE DILU-PCG + AMGx） |
| **`T`** (temperature) | PBiCG | DILU | 非对称 (含 `fvm::div(rhoCpPhi,T)`) | ✅ 仅对 AMGx 对标；cuSPARSE DILU-PCG 不适用（PCG 要求 SPD） |
| `U` (momentum) | — | — | — | ❌ `momentumPredictor: no`，U 不走独立线性求解（由 pEqn 重构） |
| `alpha.metal` (VOF) | isoAdvector | — | — | ❌ 几何推进，非线性代数问题 |
| `pcorr` | PCG | DIC | SPD | ⚠️ 本 case 无 dynamic mesh，不会触发；若出现则一并 dump |

**首轮对标范围**：`pd`（主）+ `T`（副）。
**延伸（用户 Q2(c) 授权后）**：暂时无新增，因 `U` / `alpha` 物理上不产生独立线性系统。

### 2.2 时间步范围

```
Case controlDict modifications:
    endTime  1e-3  →  1e-6       (仅需进入熔池成形阶段)
    maxDeltaT unchanged (2e-8)    (maxCo=0.5 自适应步长)
    writeInterval 1e-5 → 1e-7     (每 ~5 步写一次 field，便于核对)
```

预期时间步：~50 steps 内（首步 Δt=1e-12 → 迅速升至 2e-8 → maxCo 稳定）。

**dump 时间步选择**：
- **冷启动组**：step 1, 2, 3（dt 极小，矩阵刚性最高）
- **准稳态组**：step 40, 45, 50（熔池已成形，系数矩阵稳定）

> 时间步通过 `dumpTimeSteps (1 2 3 40 45 50);` 在新增的 `system/matrixDumperDict` 声明。

### 2.3 每步 dump 策略

单时间步内各方程被调用次数：
- `pd`：`nCorrectors=3` × `nOuterCorrectors=1` = **3 次 solve**（全部 dump）
- `T`：最多 `nTCorrectors=250` 次，实际按 residual 提前 break，经验 ~10-30 次
  - **全部 dump**（`metadata.json` 里记录 iter index 便于后续选取）
- `pcorr`：本 case 不触发

单时间步预计产生 ~3 + 20 ≈ **23 个矩阵**。
6 个 dump 时间步总计 **~140 个矩阵** / 每矩阵 ~200 MB（2M 未知量 ASCII MM）→ 总量 ~28 GB。

> ⚠️ **磁盘预算警告**：若 ASCII MM 太大，切换成 MatrixMarket 二进制（COO）格式，或只 dump 每方程首次 + 末次 solve，把单时间步规模压到 4 个矩阵 / 总量 ~5 GB。**决定：默认 full dump；若磁盘紧张再裁剪（metadata.json 兼容两种）。**

---

## 3. 架构

### 3.1 提取端（OpenFOAM 侧）

```
[LaserbeamFoam 源码 fork: /home/yzk/LaserbeamFoam/LaserbeamFoam/]
 │
 └── applications/solvers/laserMeltFoam/
     ├── matrixDumper.H          (新增 - 所有 dump 逻辑)
     ├── TEqn.H                  (改 - 插一行 dumpFvMatrix)
     ├── pEqn.H                  (改 - 插一行 dumpFvMatrix)
     ├── laserMeltFoam.C         (改 - createMatrixDumper at start)
     └── Make/options            (改 - 可能要 link 一些 IO 库)
```

#### 3.1.1 `matrixDumper.H` 接口

```cpp
// Signature
namespace Foam {
  class matrixDumper {
    public:
      matrixDumper(const Time& runTime);    // reads system/matrixDumperDict

      // Call site in *Eqn.H
      template<class Type>
      void dump(
          const fvMatrix<Type>& eqn,
          const word& eqName,         // "pd", "T"
          const label iterIndex,      // corrector index within time step
          const GeometricField<Type, fvPatchField, volMesh>& x0
      );

      bool shouldDumpNow() const;     // fast-path guard (avoid cost on non-dump steps)

    private:
      // State
      const Time& runTime_;
      List<label> dumpSteps_;         // time step indices from dict
      fileName outputDir_;            // "postProcessing/matrices"
      bool binaryMM_;                 // ASCII vs binary MM
      bool enabled_;                  // master switch
      ...
  };
}
```

#### 3.1.2 调用点

```cpp
// TEqn.H (insert BEFORE TEqn.solve())
if (matrixDumper_.shouldDumpNow()) {
    matrixDumper_.dump(TEqn, "T", i, T);
}
Tp = TEqn.solve();

// pEqn.H (insert BEFORE pdEqn.solve())
if (matrixDumper_.shouldDumpNow()) {
    matrixDumper_.dump(pdEqn, "pd", corrIdx, pd);
}
pdEqn.solve(...);
```

#### 3.1.3 `matrixDumper::dump` 内部

```
1. Determine if current runTime.timeIndex() ∈ dumpSteps_   (hash check)
2. If yes:
   a. Extract A from fvMatrix:
       - diag()   → N diagonal entries
       - upper()  → nFaces upper entries (one per face)
       - lower()  → nFaces lower entries (one per face)
       - lduAddr().upperAddr()  → face→owner→neighbour (cell IDs)
   b. Extract b = fvMatrix.source()   (pre-relax; matters for comparison)
   c. Convert LDU → COO:
       for each cell c in 0..N-1:
           emit (c, c, diag[c])
       for each face f in 0..nFaces-1:
           owner = upperAddr[f]; neigh = lowerAddr[f]
           emit (owner, neigh, upper[f])
           emit (neigh, owner, lower[f])
   d. Write to:
       postProcessing/matrices/<time>/<eqName>_corr<iterIdx>/
           ├── A.mm            (MatrixMarket coordinate, general)
           ├── b.mm            (MatrixMarket array, N×1)
           ├── x0.mm           (MatrixMarket array, N×1)  initial guess
           └── metadata.json   (see §3.2)
```

> ⚠️ **cell ordering 保序**：OpenFOAM 的 cell numbering 不一定是 lexicographic；我们必须在 metadata.json 里标记 "cell order = OpenFOAM native"，读取侧不能假设任何空间结构。

#### 3.1.4 matrixDumperDict 示例

```cpp
// system/matrixDumperDict
FoamFile { version 2.0; format ascii; class dictionary; object matrixDumperDict; }

enabled        true;
binaryMM       false;     // ASCII for first pass (easier debug); switch true later
outputDir      "postProcessing/matrices";

// Whitelist of time step indices (1-based, matches runTime.timeIndex())
dumpTimeSteps  (1 2 3 40 45 50);

// Which equations to dump (omit to dump all registered)
equations      (pd T);
```

### 3.2 数据格式规范

#### 3.2.1 MatrixMarket 扩展约定

- **A.mm**：`%%MatrixMarket matrix coordinate real general`
  (general 因为 T 非对称；pd 虽 SPD 但 DIC 不存对称性到 LDU，按 general 存更安全)
- **b.mm** / **x0.mm**：`%%MatrixMarket matrix array real general`

#### 3.2.2 metadata.json schema

```json
{
  "schema_version": "1.0",
  "case": "LaserbeamFoam/tutorials/laserMeltFoam/LPBF_tutorial",
  "openfoam_version": "v2506",
  "laserbeamfoam_commit": "<git rev-parse HEAD>",

  "time": {
    "step_index": 2,
    "time_value": 1.5e-12,
    "delta_t": 5e-13
  },
  "equation": {
    "name": "pd",
    "corrector_index": 0,
    "total_correctors_in_step": 3,
    "is_final_corrector": false
  },
  "solver_openfoam": {
    "type": "PCG",
    "preconditioner": "DIC",
    "tolerance": 1e-8,
    "relTol": 0.0,
    "initial_residual": 0.02341,
    "final_residual": 8.1e-9,
    "iterations": 42,
    "converged": true,
    "solve_time_seconds": 0.183
  },
  "matrix": {
    "n_rows": 2048000,
    "n_cols": 2048000,
    "nnz": 14284800,
    "symmetric_structure": true,
    "numerically_symmetric": false,
    "diagonal_dominant": null
  },
  "cell_ordering_note": "OpenFOAM native; NO assumed spatial layout"
}
```

### 3.3 桥端（Python 侧，`dilu/benchmark/openfoam_crosscheck/`）

```
dilu/benchmark/openfoam_crosscheck/
├── reader.py                  # MM + metadata → scipy.sparse.csr_matrix + np.ndarray
├── driver_cusparse.py         # Calls dilu.cusparse on pd matrices
├── driver_amgx.py             # Calls dilu.amgx on pd+T matrices
├── driver_scipy.py            # scipy reference (small only)
├── compare.py                 # L2/L3 metrics computation
├── run_all.py                 # End-to-end orchestration
└── data/                      # Symlink → case/postProcessing/matrices/
```

**核心 API**：
```python
# reader.py
def load_ofmm(matrix_dir: Path) -> OFMatrixBundle:
    """Returns (A: csr_matrix, b: ndarray, x0: ndarray, meta: dict)"""

# compare.py
def l2_metric(x_ours: ndarray, x_ref: ndarray) -> dict:
    """{'rel_inf_norm', 'rel_2_norm', 'pass_tol_1e-8'}"""

def residual_metric(A: csr_matrix, x: ndarray, b: ndarray) -> dict:
    """{'rel_residual', 'pass_tol_of'}"""
```

### 3.4 求解器驱动

每个 driver 的职责：
1. 读 (A, b, x0, meta)
2. 喂自己的求解器
3. 记录 iter / time / residual / 解向量
4. 落盘 `<matrix_dir>/results/<solver_name>.json` + `x.mm`

**配对矩阵**（哪些求解器跑哪些方程）：

| 方程 | scipy spsolve | our cuSPARSE DILU-PCG | our AMGx classical | our AMGx aggressive | OpenFOAM PCG-DIC (log) |
|------|---------------|----------------------|---------------------|---------------------|------------------------|
| pd   | ✅(小规模)    | ✅                   | ✅                  | ✅                  | ✅ (原生) |
| T    | ✅(小规模)    | ❌ (非 SPD)          | ✅                  | ✅                  | ✅ PBiCG-DILU (原生) |

---

## 4. 目录布局（全景）

```
/home/yzk/
├── LaserbeamFoam/LaserbeamFoam/applications/solvers/laserMeltFoam/
│   ├── matrixDumper.H                   (新增)
│   ├── TEqn.H                            (改 ~2 lines)
│   ├── pEqn.H                            (改 ~2 lines)
│   ├── laserMeltFoam.C                   (改 ~2 lines)
│   └── Make/options                      (改)
│
├── LaserbeamFoam/LaserbeamFoam/tutorials/laserMeltFoam/LPBF_tutorial/
│   ├── system/matrixDumperDict           (新增)
│   ├── system/controlDict                (改 endTime)
│   └── postProcessing/matrices/          (自动生成)
│       ├── <t0>/pd_corr0/{A.mm, b.mm, x0.mm, metadata.json}
│       ├── <t0>/pd_corr1/...
│       ├── <t0>/pd_corr2/...
│       ├── <t0>/T_iter0/...
│       ├── <t0>/T_iter1/...
│       └── ...
│
└── DILU-Research/
    ├── dilu/benchmark/openfoam_crosscheck/
    │   ├── reader.py                     (新增)
    │   ├── driver_*.py                   (新增)
    │   ├── compare.py                    (新增)
    │   ├── run_all.py                    (新增)
    │   └── data → (symlink)
    │
    └── docs/
        ├── design/openfoam_crosscheck_plan.md    (本文件)
        └── benchmark/OPENFOAM_CROSSCHECK_20260423.md  (最终报告)
```

---

## 5. 验证

### 5.1 提取端自洽测试（每个矩阵 dump 完立刻做）

每次 dump 时额外写一个 `sanity.txt`：
```
OpenFOAM internal: |A @ x_of_solution - b|_2 / |b|_2 = <final_residual>
Dump reload check: |A_dumped @ x_of_solution_dumped - b_dumped|_2 / |b_dumped|_2 = <match?>
```

两者必须一致到 ~1e-15，否则 dump 逻辑有 bug（LDU→COO 错了）。

### 5.2 Python 桥回合测试

```python
A, b, x0, meta = load_ofmm(matrix_dir)
x_openfoam_final = load_mm(matrix_dir / "x_openfoam_final.mm")  # 额外 dump 一份
rel_res = np.linalg.norm(A @ x_openfoam_final - b) / np.linalg.norm(b)
assert rel_res < 1.01 * meta["solver_openfoam"]["final_residual"]  # 允许 1% 冗余
```

若不通过 → 90% 是 cell ordering / LDU→COO 的 bug。

### 5.3 L2 跨求解器对比

对每个 (A, b)：
```python
x_scipy    = scipy.sparse.linalg.spsolve(A, b)     # ref (if small)
x_cusparse = our_cusparse_dilu_pcg(A, b, tol=1e-10)
x_amgx_cla = our_amgx(A, b, cfg="classical_pmis", tol=1e-10)
x_amgx_agg = our_amgx(A, b, cfg="aggressive_d1", tol=1e-10)

for x in [x_cusparse, x_amgx_cla, x_amgx_agg]:
    assert rel_inf_norm(x, x_scipy) < 1e-8
```

### 5.4 L3 性能对比

报告表：iter / setup(s) / solve(s) / rel_residual / 是否 L2 PASS.

---

## 6. Deliverables（实施产物）

| # | 产物 | 位置 | 产出阶段 |
|---|------|------|----------|
| D1 | `matrixDumper.H` + patches + dict | `LaserbeamFoam/.../laserMeltFoam/` | Step A |
| D2 | recompiled `laserMeltFoam` 二进制 | `~/OpenFOAM/yzk-v2506/platforms/.../bin/` | Step A |
| D3 | shortened case + 完整 dump 输出 | `LPBF_tutorial/postProcessing/matrices/` | Step B |
| D4 | Python reader + drivers + compare | `dilu/benchmark/openfoam_crosscheck/` | Step C |
| D5 | 对标报告 | `docs/benchmark/OPENFOAM_CROSSCHECK_20260423.md` | Step D |

---

## 7. 风险与缓解

| # | 风险 | 缓解 |
|---|------|------|
| R1 | LDU→COO 索引搞错 → A @ x ≠ b | §5.1 提取端自洽测试 + §5.2 Python 侧 A @ x_of ≈ b |
| R2 | 磁盘占用过大（ASCII MM 每 2M 矩阵 ~200 MB）| 预案切换 binaryMM=true；或只 dump head + tail corrector |
| R3 | `pdEqn.relax()` 修改 A（relaxation factor=1.0 时是 no-op，但 fvSolution 里 pd 为 1.0 ✓） | 仔细检查：若 relaxFactor < 1，dump 应发生在 relax 之后，匹配实际 solve 的矩阵 |
| R4 | T 是非对称 → cuSPARSE DILU-PCG 不适用 | 文档明确标注；T 仅对 AMGx 对标 |
| R5 | 128³+ 规模下 scipy spsolve OOM | 用 AMGx tol=1e-14 作 pseudo-ref |
| R6 | OpenFOAM 的 `boundaryCoeffs` / `internalCoeffs`（边界对 A 的贡献）是否被 dump？| **必须**：`fvMatrix.diag()` 已包含 internalCoeffs；但 boundary 对 RHS 的贡献在 `fvMatrix.source()` 里。两者都 dump 即可；写单元测试验证 |
| R7 | OpenFOAM 非 SPD 方程存的 upper != lower，我们要同时 dump | LDU→COO 时 upper 和 lower 分别写，不假设对称 |

---

## 8. 执行顺序 & 暂停点

```
Step A — OpenFOAM 侧实现
  A.1 写 matrixDumper.H                   (~2h)
  A.2 patch laserMeltFoam.C / TEqn.H / pEqn.H  (~0.5h)
  A.3 Make/options 调整、wmake             (~0.5h)
  A.4 单元测试: 最小矩阵 dump + reload 验证 sanity  (~1h)
  ✅ 暂停点 1: 给用户展示 dump 的第一份 metadata.json + A.mm 片段

Step B — 跑 case
  B.1 修 controlDict + matrixDumperDict  (~15min)
  B.2 Allclean + blockMesh + laserMeltFoam  (~20-60min 依步长)
  B.3 核对 postProcessing/matrices/ 完整性  (~15min)
  ✅ 暂停点 2: 给用户展示 6 个时间步的矩阵目录树 + 总体磁盘占用

Step C — Python 桥 + 跨求解器跑
  C.1 reader.py + sanity 测试          (~2h)
  C.2 driver_scipy.py (小矩阵 ref)     (~1h)
  C.3 driver_cusparse.py + driver_amgx.py  (~2h)
  C.4 compare.py + run_all.py          (~1h)
  ✅ 暂停点 3: 第一组 (A, b) 上的 L2 PASS 报告

Step D — 报告
  D.1 OPENFOAM_CROSSCHECK_20260423.md  (~2h)
  ✅ 暂停点 4: 交付用户审查
```

---

## 9. 成功判据（全部满足方可声称对标成功）

- [ ] 所有 dump 的矩阵通过 §5.1 自洽测试（dump↔OpenFOAM 内部残差一致到 1e-15）
- [ ] 所有 dump 的矩阵通过 §5.2 回合测试（A @ x_of ≈ b 到 OpenFOAM tol × 1.01）
- [ ] cuSPARSE DILU-PCG 在全部 pd 矩阵上达到 L2 PASS（相对 inf 范数误差 < 1e-8）
- [ ] AMGx (至少一种 config) 在全部 pd + T 矩阵上达到 L2 PASS
- [ ] L3 性能数据完整记录，支持直接比较 iter / setup / solve
- [ ] 报告中对**每个**不通过 PASS 的矩阵给出"为什么"的解释（例如 T 矩阵对 cuSPARSE DILU-PCG 的不适用）

---

## 10. 非目标（明确排除）

- ❌ 复现 OpenFOAM 的 multi-field 耦合（我们只对标单次线性求解）
- ❌ 实现 OpenFOAM 的 GAMG（我们用自己的 AMGx 对标之）
- ❌ 并行通信层（OpenFOAM MPI + 我们单卡）——case 用 `decomposeParDict` 但**不**跑 MPI；`runParallel` 关掉
- ❌ bit-identical L1 层级（见 §1）
